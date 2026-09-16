"""sampler/gpu/hygon:hy-smi 输出解析与探测集成测试(subprocess 全 mock)。"""

import json

import pytest

from llm_monitor.sampler.gpu import hygon

# 模拟真实 hy-smi --json 的畸形输出:字段之间 / card 之间漏逗号、数值带单位后缀。
# 严格 json.loads 会在第一处漏逗号报 "Expecting ',' delimiter"(即线上最初的报错)。
RAW_MALFORMED = (
    '{"card0": {"GPU use (%)": "0%", "GPU Memory Allocated (VRAM%)": "0%" '
    '"GPU memory use (%)": "0%" "VRAM Total Memory (B)": 34359738368 '
    '"VRAM Total Used Memory (B)": 6870269952 '
    '"Temperature (Sensor edge) (C)": 41.0C '
    '"Average Graphics Package Power (W)": 80W} '
    '"card1": {"GPU use (%)": "7%"}}'
)


# 线上真实 hy-smi 输出(card0/card1,原样保留漏逗号畸形):
#   严格 json.loads 恰好报 "Expecting ',' delimiter line 1 col 122 char 121"
RAW_HY_SMI_REAL = """{"card0": {"Card Series": "BW", "Card Vendor": "C-3000 IC Design Co., Ltd.", "Average Graphics Package Power (W)": "91.0""Average GFX Core Power (W)": "20.0""Average Memory Power (W)": "38.0", "Temperature (Sensor edge) (C)": "54.0", "Temperature (Sensor junction) (C)": "57.0", "Temperature (Sensor mem) (C)": "51.0", "Temperature (Sensor core) (C)": "51.0", "HCU use (%)": "0.0", "HCU memory use (%)": "98", "Estimated maximum PCIe bandwidth over the last second (MB/s)": "0.061", "fclk clock level": "0", "fclk clock speed": "1390Mhz", "mclk clock level": "0", "mclk clock speed": "1800Mhz", "sclk clock level": "7", "sclk clock speed": "1350Mhz", "socclk clock level": "0", "socclk clock speed": "1079Mhz", "": "", "": "", "pcie clock level": "2", "pcie clock speed": "32.0GT/s, x16 1100Mhz", "vram Total Memory (MiB)": "65520", "vram Total Used Memory (MiB)": "64169"}, "card1": {"Card Series": "BW", "Card Vendor": "C-3000 IC Design Co., Ltd.", "Average Graphics Package Power (W)": "84.0""Average GFX Core Power (W)": "16.0""Average Memory Power (W)": "34.0", "Temperature (Sensor edge) (C)": "52.0", "Temperature (Sensor junction) (C)": "54.0", "Temperature (Sensor mem) (C)": "52.0", "Temperature (Sensor core) (C)": "50.0", "HCU use (%)": "0.0", "HCU memory use (%)": "98", "Estimated maximum PCIe bandwidth over the last second (MB/s)": "0.711", "fclk clock level": "0", "fclk clock speed": "1390Mhz", "mclk clock level": "0", "mclk clock speed": "1800Mhz", "sclk clock level": "7", "sclk clock speed": "1350Mhz", "socclk clock level": "0", "socclk clock speed": "1079Mhz", "": "", "": "", "pcie clock level": "2", "pcie clock speed": "32.0GT/s, x16 1100Mhz", "vram Total Memory (MiB)": "65520", "vram Total Used Memory (MiB)": "64104"}}"""


class _Result:
    def __init__(self, stdout="", rc=0, stderr=""):
        self.stdout = stdout
        self.returncode = rc
        self.stderr = stderr


def _install_fake_run(monkeypatch, result):
    monkeypatch.setattr(hygon.subprocess, "run", lambda *a, **k: result)


def test_raw_malformed_really_breaks_json_loads():
    # 证明这确实是线上 "Expecting ',' delimiter" 那一类输出
    with pytest.raises(json.JSONDecodeError, match=r"Expecting ',' delimiter"):
        json.loads(RAW_MALFORMED)


def test_lenient_backend_detects_and_samples_cards(monkeypatch):
    _install_fake_run(monkeypatch, _Result(stdout=RAW_MALFORMED))
    backend = hygon._SmiCliBackend("hy-smi")
    assert backend.ok
    assert backend.count == 2

    samples = backend.sample()
    assert len(samples) == 2
    card0 = samples[0]
    assert card0.vendor == "hygon"
    assert card0.index == 0
    assert card0.util_pct == 0.0
    assert card0.temperature_c == 41.0
    assert card0.power_w == 80.0
    assert card0.vram_total_mb == pytest.approx(34359738368 / (1024 * 1024))
    assert card0.vram_used_mb == pytest.approx(6870269952 / (1024 * 1024))
    # 第二张卡使用率 7%
    assert samples[1].util_pct == 7.0


def test_falls_back_to_reduced_flag_set_on_rc_nonzero(monkeypatch):
    def fake_run(cmd, **kw):
        if "--showmeminfo" in cmd:
            return _Result(stderr="unsupported option --showmeminfo", rc=1)
        return _Result(stdout='{"card0": {"GPU use (%)": "9%"}}', rc=0)

    monkeypatch.setattr(hygon.subprocess, "run", fake_run)
    backend = hygon._SmiCliBackend("hy-smi")
    assert backend.ok
    assert backend.count == 1


def test_load_smi_output_valid_fast_path():
    payload = {"card0": {"GPU use (%)": "0%"}}
    assert hygon._load_smi_output(json.dumps(payload)) == payload


def test_load_smi_output_empty_raises():
    with pytest.raises(RuntimeError, match="empty stdout"):
        hygon._load_smi_output("   ")


def test_load_smi_output_non_dict_raises():
    with pytest.raises(RuntimeError, match="expected dict"):
        hygon._load_smi_output("[1, 2]")


def test_load_smi_output_totally_invalid_includes_snippet():
    # 宽松解析也失败时,错误信息要带上 stdout 头,方便拿到真实样本继续收敛
    with pytest.raises(RuntimeError, match=r"stdout\[:400\]"):
        hygon._load_smi_output("not json at all { totally broken")


def test_real_hy_smi_repro_exact_original_error():
    # 逐字符复刻最初的报错位置(char 121),证明修的就是这个样本
    with pytest.raises(json.JSONDecodeError) as ei:
        json.loads(RAW_HY_SMI_REAL)
    assert ei.value.pos == 121
    assert "Expecting ',' delimiter" in ei.value.msg


def test_real_hy_smi_lenient_detects_and_maps_fields(monkeypatch):
    _install_fake_run(monkeypatch, _Result(stdout=RAW_HY_SMI_REAL))
    backend = hygon._SmiCliBackend("hy-smi")
    assert backend.ok
    assert backend.count == 2

    samples = backend.sample()
    assert len(samples) == 2
    card0 = samples[0]
    # hy-smi 词表:HCU 使用率 / vram ... (MiB) / Card Series / sclk clock speed
    assert card0.vendor == "hygon"
    assert card0.index == 0
    assert card0.name == "BW"
    assert card0.util_pct == 0.0
    assert card0.temperature_c == 54.0
    assert card0.power_w == 91.0
    assert card0.vram_total_mb == 65520
    assert card0.vram_used_mb == 64169
    assert card0.extra["mem_used_pct"] == 98.0
    assert card0.extra["sclk_mhz"] == 1350.0
    assert card0.extra["mclk_mhz"] == 1800.0
    # 带宽值被带上
    assert "pcie_used_mbps" in card0.extra

    card1 = samples[1]
    assert card1.vram_total_mb == 65520
    assert card1.vram_used_mb == 64104
    assert card1.power_w == 84.0
