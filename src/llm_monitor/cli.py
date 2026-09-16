"""CLI 入口:`llm-monitor` 命令。

子命令:
- enable / disable / status:在当前解释器的 site-packages 装/卸 .pth
  自动加载文件。装完后 vLLM 的启动命令完全不改。
- run:独立启动一个监控进程(不 patch,只做 host 采样 + Web),便于本机预览。
- version
"""
from __future__ import annotations

import argparse
import sys
import time

from . import __version__
from .autoload import disable, enable, status


def _cmd_enable(_args) -> int:
    path = enable()
    print(f"[ok] installed autoload: {path}")
    print("\n下一步:")
    print("  1) 设置环境变量: export LLM_MONITOR_ENABLE=1")
    print("  2) 用原来 vLLM 的启动命令(一个字不用改):")
    print("     python -m vllm.entrypoints.openai.api_server --model xxx")
    print("  3) 打开 http://127.0.0.1:9109")
    return 0


def _cmd_disable(_args) -> int:
    removed = disable()
    if not removed:
        print("[ok] no autoload file to remove")
    else:
        for p in removed:
            print(f"[ok] removed {p}")
    return 0


def _cmd_status(_args) -> int:
    s = status()
    print(f"python:       {s['python']}")
    print("site-packages:")
    for d in s["site_dirs"]:
        print(f"  - {d}")
    print("autoload installed at:")
    if not s["installed_at"]:
        print("  (none)  --  run `llm-monitor enable` to install")
    else:
        for p in s["installed_at"]:
            print(f"  - {p}")
    import os
    on = os.environ.get("LLM_MONITOR_ENABLE", "0") == "1"
    print(f"LLM_MONITOR_ENABLE: {'1 (ON)' if on else '0 (off)'}")
    return 0


def _cmd_run(args) -> int:
    from .bootstrap import install
    from .config import Config

    cfg = Config.from_env()
    cfg.enable = True
    cfg.host = args.host
    cfg.port = args.port
    cfg.sample_interval_ms = args.interval_ms

    install(cfg)
    print(f"[ok] llm-monitor running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="llm-monitor")
    p.add_argument("--version", action="version", version=f"llm-monitor {__version__}")

    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("enable", help="在 site-packages 装 .pth 自动加载(一次性)")
    sub.add_parser("disable", help="卸载 .pth")
    sub.add_parser("status", help="查看当前状态")

    run = sub.add_parser("run", help="独立启动监控进程(仅 host + web)")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=9109)
    run.add_argument("--interval-ms", type=int, default=1000)

    args = p.parse_args(argv)
    dispatch = {
        "enable": _cmd_enable,
        "disable": _cmd_disable,
        "status": _cmd_status,
        "run": _cmd_run,
    }
    if args.cmd is None:
        p.print_help()
        return 0
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
