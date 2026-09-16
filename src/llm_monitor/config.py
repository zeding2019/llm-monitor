"""集中化配置,全部从环境变量读取(带默认)。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None else default


def _env_list(key: str, default: list[str]) -> list[str]:
    v = os.environ.get(key)
    if not v:
        return default
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass
class Config:
    enable: bool = False
    host: str = "0.0.0.0"
    port: int = 9109
    gpu_backends: list[str] = field(default_factory=lambda: ["auto"])
    sample_interval_ms: int = 1000
    gpu_sample_interval_ms: int = 5000
    db_path: str = "./llm_monitor.db"  # 空字符串 => 纯内存
    retention_days: int = 7
    transaction_buf: int = 10000
    heartbeat_buf: int = 3600  # 每个 source 独立保留的条数(分桶,非全局总量)
    event_buf: int = 10000

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            enable=_env_bool("LLM_MONITOR_ENABLE", False),
            host=_env_str("LLM_MONITOR_HOST", "0.0.0.0"),
            port=_env_int("LLM_MONITOR_PORT", 9109),
            gpu_backends=_env_list("LLM_MONITOR_GPU", ["auto"]),
            sample_interval_ms=_env_int("LLM_MONITOR_SAMPLE_INTERVAL_MS", 1000),
            gpu_sample_interval_ms=_env_int("LLM_MONITOR_GPU_SAMPLE_INTERVAL_MS", 5000),
            db_path=_env_str("LLM_MONITOR_DB_PATH", "./llm_monitor.db"),
            retention_days=_env_int("LLM_MONITOR_RETENTION_DAYS", 7),
            transaction_buf=_env_int("LLM_MONITOR_TRANSACTION_BUF", 10000),
            heartbeat_buf=_env_int("LLM_MONITOR_HEARTBEAT_BUF", 3600),
            event_buf=_env_int("LLM_MONITOR_EVENT_BUF", 10000),
        )

    def auto_install(self) -> None:
        # 环境变量直连(主进程)或标记文件(子进程)。
        # 为何有标记文件:SGLang 的 scheduler/detokenizer 子进程是清空环境启动的
        # (exec/spawn 不带父进程 env),LLM_MONITOR_ENABLE 到不了子进程 →
        # .pth 加载时看不到 enable → 补丁从不生效。主进程安装时在 /tmp 写一个
        # 标记(enable + 绝对 DB 路径),子进程启动时读它决定是否安装、写哪个库,
        # 从而多进程共享同一个 request_trace DB。
        cfg = self
        if not cfg.enable:
            from .bootstrap import read_marker
            m = read_marker()
            if m.get("enable") != "1":
                return
            cfg.enable = True
            if m.get("db"):
                cfg.db_path = m["db"]  # 绝对路径,保证所有进程写同一文件
            import contextlib
            with contextlib.suppress(TypeError, ValueError):
                cfg.port = int(m.get("port") or cfg.port)
        # 主进程 / 子进程的判定完全交给 bootstrap.install() 里的文件锁,
        # 这里不再看 LOCAL_RANK(v0/v1/Ray/mp 不同后端行为不一致)
        from .bootstrap import install
        install(cfg)
