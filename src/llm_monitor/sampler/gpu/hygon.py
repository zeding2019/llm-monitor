"""海光 DCU 采集器。

优先级(按用户偏好):
  1. hy-smi   —— 海光官方工具,输出格式最贴合 DCU
  2. rocm-smi —— ROCm 生态兜底
  3. pyrsmi   —— C 绑定,速度快但对 DCU 不一定支持

⚠️ hy-smi 的 --showhw 会破坏 --json 输出(把表格文本插进 JSON),所以
   PCIe Gen/Width 从 /sys/class/drm/card*/device/ 直接读。

采集字段:
  util_pct        GPU 使用率
  vram_used/total 显存 MB
  temperature_c   温度 °C
  power_w         功耗 W
  extra.pcie_used_mbps       PCIe 总带宽(hy-smi 只给合并值)
  extra.pcie_link_gen/width  链路当前 Gen 数 + lane 数(sysfs)
  extra.pcie_max_mbps        理论峰值 MB/s
  extra.pcie_util_pct        当前 / 峰值 × 100
  extra.sclk_mhz  核心频率
  extra.mclk_mhz  显存频率
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
import subprocess

from ...util.lenient_json import lenient_loads
from ..gpu_base import GpuSample, register

log = logging.getLogger("llm_monitor.gpu.hygon")

# GT/s → PCIe Gen 映射
_GTS_TO_GEN = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
# PCIe 每 lane 的有效带宽(MB/s,单向)
_PCIE_PER_LANE_MBPS = {1: 250, 2: 500, 3: 985, 4: 1969, 5: 3938, 6: 7877}


def _read_sysfs_pcie_info() -> dict[int, dict]:
    """从 /sys/class/drm/card{N}/device/ 读 PCIe 链路。
    路径示例:
      /sys/class/drm/card0/device/current_link_speed  -> "16.0 GT/s PCIe"
      /sys/class/drm/card0/device/current_link_width  -> "16"
      /sys/class/drm/card0/device/max_link_speed
      /sys/class/drm/card0/device/max_link_width
    """
    info: dict[int, dict] = {}
    for path in sorted(glob.glob("/sys/class/drm/card*/device/")):
        m = re.search(r"/card(\d+)/", path)
        if not m:
            continue
        idx = int(m.group(1))
        entry: dict = {}
        for fname in ("current_link_speed", "current_link_width",
                      "max_link_speed", "max_link_width"):
            try:
                with open(os.path.join(path, fname)) as f:
                    entry[fname] = f.read().strip()
            except Exception:  # noqa: BLE001
                continue
        # 解析速率
        for key_in, key_gen in (("current_link_speed", "cur_gen"),
                                ("max_link_speed", "max_gen")):
            s = entry.get(key_in, "")
            m2 = re.match(r"([\d.]+)\s*GT/s", s)
            if m2:
                entry[key_gen] = _GTS_TO_GEN.get(float(m2.group(1)), 0)
        # 解析 lane 宽度
        for key_in in ("current_link_width", "max_link_width"):
            try:
                entry[key_in.replace("_width", "_lane")] = int(entry.get(key_in, "0"))
            except (TypeError, ValueError):
                pass
        info[idx] = entry
    if info:
        log.info("sysfs PCIe info detected for %d cards", len(info))
    return info


# ---- 1. hy-smi / rocm-smi CLI 主力后端 --------------------------------

def _find_smi_cli() -> str | None:
    for name in ("hy-smi", "rocm-smi"):
        path = shutil.which(name)
        if path:
            return path
    return None


# 精简参数集:老版本 hy-smi 若不认 --showmeminfo/--showclocks,用这一组重试
_SMI_FALLBACK_FLAGS = [
    "--showuse", "--showmemuse", "--showtemp", "--showpower",
    "--showbw", "--showproductname", "--json",
]


def _load_smi_output(stdout: str) -> dict:
    """先严格 json.loads(合法输出零开销);失败则用宽松解析兜底。

    真实 hy-smi --json 输出经常不是合法 JSON——漏逗号、数值带单位后缀、
    文本混入等,详见 util/lenient_json 模块。两种解析都失败时,错误信息
    会带上 stdout 头,方便拿到真实样本后继续收敛解析规则。
    """
    s = (stdout or "").strip()
    if not s:
        raise RuntimeError("empty stdout")
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        data = None
    if data is None:
        try:
            data = lenient_loads(s)
        except ValueError as e:
            raise RuntimeError(
                f"hy-smi stdout is neither valid JSON nor lenient-parseable ({e}); "
                f"stdout[:400]={s[:400]!r}"
            ) from None
    if not isinstance(data, dict):
        raise RuntimeError(
            f"hy-smi output parsed as {type(data).__name__}, expected dict; "
            f"stdout[:200]={s[:200]!r}"
        )
    return data


class _SmiCliBackend:
    def __init__(self, cli: str) -> None:
        self.cli = cli
        self.ok = False
        self.count = 0
        self._flags = self._pick_flags()
        self._pcie_link_info = _read_sysfs_pcie_info()  # 启动时读一次,静态信息
        try:
            data = self._query()
            self.count = self._count_cards(data)
            self.ok = self.count > 0
            if self.ok:
                print(f"[llm-monitor] hygon: {self.cli} detected {self.count} card(s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[llm-monitor] hygon: {self.cli} probe failed: {e}", flush=True)
            log.debug("smi cli probe failed: %s", e)

    def _pick_flags(self) -> list[str]:
        """⚠️ 不加 --showhw / --showtopo:这些会把非 JSON 文本插进输出。"""
        return [
            "--showuse",
            "--showmemuse",
            "--showmeminfo", "vram",
            "--showtemp",
            "--showpower",
            "--showclocks",
            "--showbw",
            "--showproductname",
            "--json",
        ]

    def _query(self) -> dict:
        # hy-smi 在某些环境下比较慢(尤其首次调用/多卡时),默认给 15 秒,
        # 可通过 LLM_MONITOR_HYGON_TIMEOUT 环境变量覆盖
        import os
        timeout = int(os.environ.get("LLM_MONITOR_HYGON_TIMEOUT", "15"))
        # 完整参数集在老版本 hy-smi 上可能 rc!=0,此时退回精简参数集重试
        last_err = ""
        for flags in (self._flags, _SMI_FALLBACK_FLAGS):
            result = subprocess.run(
                [self.cli, *flags],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                last_err = f"rc={result.returncode}: {result.stderr[:200]}"
                continue
            return _load_smi_output(result.stdout)
        raise RuntimeError(f"{self.cli} {last_err}")

    def _count_cards(self, data: dict) -> int:
        return sum(1 for k in data if re.match(r"^card\d+$", k))

    def device_count(self) -> int:
        return self.count

    def sample(self) -> list[GpuSample]:
        try:
            data = self._query()
        except Exception as e:  # noqa: BLE001
            log.debug("smi cli sample failed: %s", e)
            return []

        out: list[GpuSample] = []
        for key, fields in data.items():
            m = re.match(r"^card(\d+)$", key)
            if not m:
                continue
            idx = int(m.group(1))

            util = _pick_float(fields, [
                "GPU use (%)", "HCU use (%)", "GPU Utilization (%)",
                "GPU Usage (%)",
            ])
            vram_used_pct = _pick_float(fields, [
                "GPU Memory Allocated (VRAM%)", "GPU memory use (%)",
                "Memory Usage (%)", "HCU memory use (%)",
            ])
            vram_used_bytes = _pick_float(fields, [
                "VRAM Total Used Memory (B)",
                "Used Memory (VRAM) (B)",
                "VRAM Used Memory (B)",
            ])
            vram_total_bytes = _pick_float(fields, [
                "VRAM Total Memory (B)",
                "Total Memory (VRAM) (B)",
            ])
            # hy-smi 直接以 MiB 计(字段前缀小写 vram),rocm-smi 风格为 (B)
            vram_used_mib = _pick_float(fields, [
                "vram Total Used Memory (MiB)",
                "VRAM Total Used Memory (MiB)",
            ])
            vram_total_mib = _pick_float(fields, [
                "vram Total Memory (MiB)",
                "VRAM Total Memory (MiB)",
            ])
            temp = _pick_float(fields, [
                "Temperature (Sensor edge) (C)",
                "Temperature (Sensor junction) (C)",
                "Temperature (Sensor memory) (C)",
                "Temperature (C)",
            ])
            power = _pick_float(fields, [
                "Average Graphics Package Power (W)",
                "Current Socket Graphics Package Power (W)",
                "Power (W)",
            ])
            sclk = _pick_float(fields, [
                "sclk clock speed:", "sclk clock speed", "sclk (MHz)",
                "GPU Clock Speed (MHz)",
            ])
            mclk = _pick_float(fields, [
                "mclk clock speed:", "mclk clock speed", "mclk (MHz)",
                "Memory Clock Speed (MHz)",
            ])
            pcie_bw_mbps = _pick_float(fields, [
                "Estimated maximum PCIe bandwidth over the last second (MB/s)",
                "PCIe Bandwidth (MB/s)",
                "GPU Bandwidth (MB/s)",
                "PCIe Bandwidth Used (MB/s)",
            ])
            pcie_tx_mbps = _pick_float(fields, [
                "PCIe Bandwidth Sent (MB/s)",
                "TX PCIe Bandwidth (MB/s)",
            ])
            pcie_rx_mbps = _pick_float(fields, [
                "PCIe Bandwidth Recv (MB/s)",
                "RX PCIe Bandwidth (MB/s)",
            ])
            mem_bw_util = _pick_float(fields, [
                "GPU Memory Bandwidth (%)",
                "Memory Bandwidth Utilization (%)",
            ])
            name = (fields.get("Card Series") or fields.get("Card series")
                    or fields.get("Card model") or fields.get("Card SKU")
                    or "")

            vram_used_mb = vram_used_bytes / (1024 * 1024) if vram_used_bytes > 0 else float(vram_used_mib)
            vram_total_mb = vram_total_bytes / (1024 * 1024) if vram_total_bytes > 0 else float(vram_total_mib)
            if vram_used_mb == 0 and vram_used_pct > 0 and vram_total_mb > 0:
                vram_used_mb = vram_total_mb * vram_used_pct / 100

            extra: dict[str, float] = {
                "mem_used_pct": vram_used_pct,
                "sclk_mhz": sclk,
                "mclk_mhz": mclk,
            }
            if mem_bw_util > 0:
                extra["mem_bw_util_pct"] = mem_bw_util
            if pcie_bw_mbps > 0:
                extra["pcie_used_mbps"] = pcie_bw_mbps
            if pcie_tx_mbps > 0:
                extra["pcie_tx_mbps"] = pcie_tx_mbps
            if pcie_rx_mbps > 0:
                extra["pcie_rx_mbps"] = pcie_rx_mbps

            # PCIe 链路信息:来自 sysfs (启动时读的静态值)
            link = self._pcie_link_info.get(idx, {})
            cur_gen = link.get("cur_gen", 0)
            cur_lane = link.get("current_link_lane", 0)
            max_gen = link.get("max_gen", 0)
            max_lane = link.get("max_link_lane", 0)
            if cur_gen and cur_lane:
                extra["pcie_link_gen"] = float(cur_gen)
                extra["pcie_link_width"] = float(cur_lane)
                per_lane = _PCIE_PER_LANE_MBPS.get(cur_gen, 0)
                max_mbps = per_lane * cur_lane
                if max_mbps > 0:
                    extra["pcie_max_mbps"] = float(max_mbps)
                    cur = pcie_tx_mbps + pcie_rx_mbps
                    if cur == 0:
                        cur = pcie_bw_mbps
                    if cur > 0:
                        extra["pcie_util_pct"] = round(min(cur / max_mbps * 100, 100), 2)
            if max_gen and max_lane and (max_gen != cur_gen or max_lane != cur_lane):
                # 链路降级警示
                extra["pcie_max_link_gen"] = float(max_gen)
                extra["pcie_max_link_width"] = float(max_lane)

            out.append(GpuSample(
                vendor="hygon",
                index=idx,
                name=str(name).strip(),
                util_pct=util,
                vram_used_mb=vram_used_mb,
                vram_total_mb=vram_total_mb,
                temperature_c=temp,
                power_w=power,
                extra=extra,
            ))
        return out

    def close(self) -> None:
        pass


def _pick_float(d: dict, keys: list[str]) -> float:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        try:
            # 常见格式:"23", "23 %", "45.5 C", "1200Mhz", "12345678 B"
            s = str(v).strip()
            for suffix in ("%", "C", "W", "B", "Mhz", "MHz"):
                if s.endswith(suffix):
                    s = s[: -len(suffix)].strip()
            return float(s)
        except (TypeError, ValueError):
            continue
    return 0.0


# ---- 2. pyrsmi 兜底 ---------------------------------------------------

class _PyrsmiBackend:
    def __init__(self) -> None:
        self.ok = False
        self._smi = None
        self.count = 0
        try:
            from pyrsmi import rocml
            rocml.smi_initialize()
            self._smi = rocml
            self.count = rocml.smi_get_device_count()
            self.ok = self.count > 0
            if self.ok:
                print(f"[llm-monitor] hygon: pyrsmi detected {self.count} card(s)", flush=True)
        except Exception as e:  # noqa: BLE001
            log.debug("pyrsmi init failed: %s", e)

    def device_count(self) -> int:
        return self.count if self.ok else 0

    def sample(self) -> list[GpuSample]:
        if not self.ok:
            return []
        smi = self._smi
        out: list[GpuSample] = []
        for i in range(self.count):
            try:
                name = _safe(lambda: smi.smi_get_device_name(i) or "", "")
                util = _safe(lambda: smi.smi_get_device_utilization(i), 0.0)
                vram_used = _safe(lambda: smi.smi_get_device_memory_used(i) / (1024 * 1024), 0.0)
                vram_total = _safe(lambda: smi.smi_get_device_memory_total(i) / (1024 * 1024), 0.0)
                temp = _safe(lambda: smi.smi_get_device_temperature(i, 0), 0.0)
                power_raw = _safe(lambda: smi.smi_get_device_average_power(i), 0.0)
                power_w = power_raw / 1e6 if power_raw > 1000 else power_raw
                out.append(GpuSample(
                    vendor="hygon", index=i, name=str(name),
                    util_pct=float(util),
                    vram_used_mb=float(vram_used),
                    vram_total_mb=float(vram_total),
                    temperature_c=float(temp),
                    power_w=float(power_w),
                ))
            except Exception as e:  # noqa: BLE001
                log.debug("pyrsmi sample gpu %d failed: %s", i, e)
        return out

    def close(self) -> None:
        if self._smi is not None:
            try:
                self._smi.smi_shutdown()
            except Exception:  # noqa: BLE001
                pass


def _safe(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


# ---- 3. 对外统一入口(hy-smi 优先) -----------------------------------

@register
class HygonCollector:
    vendor = "hygon"

    def __init__(self) -> None:
        self._backend = None
        # 1) hy-smi / rocm-smi(用户偏好)
        cli = _find_smi_cli()
        if cli:
            b = _SmiCliBackend(cli)
            if b.ok:
                self._backend = b
        # 2) pyrsmi 兜底
        if self._backend is None:
            b = _PyrsmiBackend()
            if b.ok:
                self._backend = b
        if self._backend is None:
            print("[llm-monitor] hygon: no collector available (hy-smi/rocm-smi/pyrsmi 均不可用)", flush=True)

    def available(self) -> bool:
        return self._backend is not None

    def device_count(self) -> int:
        return self._backend.device_count() if self._backend else 0

    def sample(self) -> list[GpuSample]:
        return self._backend.sample() if self._backend else []

    def close(self) -> None:
        if self._backend:
            self._backend.close()
