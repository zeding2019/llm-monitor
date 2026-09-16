"""统一 install / uninstall 入口。幂等,可安全多次调用。"""
from __future__ import annotations

import atexit
import contextlib
import logging
import os
import tempfile
import threading

from .config import Config

_lock = threading.Lock()
_state: dict = {
    "installed": False,
    "config": None,
    "sampler": None,
    "gpu_sampler": None,
    "web": None,
    "store": None,
    "writer": None,
    "singleton_fd": None,
}
log = logging.getLogger("llm_monitor.bootstrap")


def _acquire_singleton(port: int) -> bool:
    """获取全机器唯一的运行权。

    vLLM 会用 spawn / fork / ray 起多个 worker 子进程,每个都会重新加载 .pth
    并跑到这里。用 fcntl.flock 排他锁保证:同一台机器上只有一个进程能启动
    web/sampler/writer,其他进程静默跳过。

    进程死掉时内核会自动释放锁,不需要手动清理。
    """
    try:
        import fcntl
    except ImportError:
        # 非 POSIX(极少见,Windows)。降级为端口 bind 探测。
        return _try_port_bind(port)

    lock_path = os.path.join(tempfile.gettempdir(), f"llm-monitor.{port}.lock")
    try:
        f = open(lock_path, "w")
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        f.write(f"{os.getpid()}\n")
        f.flush()
        _state["singleton_fd"] = f  # 保持文件句柄存活,锁才有效
        return True
    except (OSError, BlockingIOError):
        return False


def _try_port_bind(port: int) -> bool:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        s.close()
        return True
    except OSError:
        return False


def _marker_path() -> str:
    """进程间安装标记。固定 /tmp 而非 tempfile.gettempdir():清空 env 启动的
    SGLang 子进程里 TMPDIR 可能不同,.pth(无法 import 本包)用同一字面量判定。"""
    return "/tmp/llm-monitor.cfg"


def write_marker(cfg) -> None:
    """单例进程(带 LLM_MONITOR_ENABLE=1)写入;清空 env 的子进程据此安装。"""
    try:
        db = os.path.abspath(cfg.db_path) if cfg.db_path else ""
        with open(_marker_path(), "w") as f:
            f.write(f"enable=1\nport={cfg.port}\ndb={db}\n")
    except Exception as e:  # noqa: BLE001
        log.debug("marker write failed: %s", e)


def read_marker() -> dict:
    out: dict[str, str] = {}
    try:
        with open(_marker_path()) as f:
            for line in f:
                k, _, v = line.partition("=")
                if k.strip():
                    out[k.strip()] = v.strip()
    except OSError:
        return {}
    return out


def install(config: Config | None = None) -> None:
    from .patch import install_all as install_patches
    from .sampler.runner import GpuSamplerThread, SamplerThread
    from .store.sqlite import DbWriter, SqliteStore
    from .web.server import WebServer

    with _lock:
        if _state["installed"]:
            return
        cfg = config or Config.from_env()
        _state["config"] = cfg

        is_singleton = _acquire_singleton(cfg.port)

        # 统一异步写入器:每进程一个 DbWriter,独占写连接。
        # - 请求追踪(request_trace)按 rid upsert,所有进程都要落库(batch 数据
        #   在 Scheduler 子进程产生,补丁的 import hook 已在每个进程装好)。
        # - 聚合(minute_agg / sampled tx / heartbeat)只在单例进程跑(aggregate=
        #   is_singleton),否则多进程对 minute_agg 的 INSERT OR REPLACE 会互相覆盖。
        store = None
        if cfg.db_path:
            try:
                store = SqliteStore(cfg.db_path)
                _state["store"] = store
                writer = DbWriter(store, aggregate=is_singleton, retention_days=cfg.retention_days)
                writer.start()
                _state["writer"] = writer
            except Exception as e:  # noqa: BLE001
                log.warning("SQLite store disabled: %s", e)
                store = None

        # 单例控制:只有第一个进程启动 web/sampler,其他子进程到此为止。
        # (补丁与 DbWriter 上面已在所有进程装好。)
        if not is_singleton:
            if store is not None:
                # 便于在子进程日志里确认:清空 env 启动的子进程靠标记文件装上并写库
                print(f"[llm-monitor] worker pid={os.getpid()}: trace-writer active "
                      f"(db={cfg.db_path})", flush=True)
            _state["installed"] = True
            return

        # 主进程:写安装标记,供清空 env 启动的 SGLang 子进程读(见 config.auto_install)
        write_marker(cfg)

        install_patches()

        sampler = SamplerThread(interval_ms=cfg.sample_interval_ms)
        sampler.start()
        _state["sampler"] = sampler

        # GPU 采样独立线程:hy-smi 之类 subprocess 慢时不阻塞 host 采样
        gpu_sampler = GpuSamplerThread(
            interval_ms=cfg.gpu_sample_interval_ms,
            gpu_backends=cfg.gpu_backends,
        )
        gpu_sampler.start()
        _state["gpu_sampler"] = gpu_sampler

        web = WebServer(host=cfg.host, port=cfg.port, sqlite_store=store)
        web.start()
        _state["web"] = web

        _state["installed"] = True
        atexit.register(uninstall)
        log.info("llm-monitor installed on http://%s:%d", cfg.host, cfg.port)
        print(f"[llm-monitor] started at http://{cfg.host}:{cfg.port}", flush=True)


def uninstall() -> None:
    with _lock:
        if not _state["installed"]:
            return
        if _state["sampler"]:
            _state["sampler"].stop()
        if _state["gpu_sampler"]:
            _state["gpu_sampler"].stop()
        if _state["writer"]:
            _state["writer"].stop()
        if _state["web"]:
            _state["web"].stop()
        if _state["store"]:
            _state["store"].close()
        if _state["singleton_fd"]:
            try:
                _state["singleton_fd"].close()
            except Exception:  # noqa: BLE001
                pass
            _state["singleton_fd"] = None
        # 主进程退出即清安装标记,避免事后无关 python 误触发安装
        with contextlib.suppress(Exception):
            os.unlink(_marker_path())
        _state["installed"] = False
        log.info("llm-monitor uninstalled")


def is_installed() -> bool:
    return _state["installed"]


def get_config() -> Config | None:
    return _state["config"]
