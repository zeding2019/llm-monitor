#!/usr/bin/env python3
"""部署探针:确认服务器上装的 llm_monitor 是不是最新代码。
用法(在跑着 SGLang 的服务器上):
    python3 diag_deploy.py
把全部输出发回。"""
import os
import sqlite3
import sys
from pathlib import Path

print("=" * 60)
print("1. 安装路径")
print("=" * 60)
import llm_monitor
root = Path(llm_monitor.__file__).parent
print(f"   {root}")

print()
print("=" * 60)
print("2. 关键新功能是否在安装副本里(旧代码没有这些)")
print("=" * 60)
checks = [
    ("patch/sglang.py", "_track_requests"),        # 按 rid 生命周期追踪
    ("patch/sglang.py", "RequestTrace"),           # 引用模型
    ("store/sqlite.py", "class DbWriter"),          # 统一异步写入器
    ("store/sqlite.py", "def upsert_request_traces"),
    ("store/sqlite.py", "def query_request_traces"),
    ("store/schema.sql", "request_trace"),          # 新表
    ("bootstrap.py", "DbWriter"),                   # 每进程起写入器
]
for rel, sym in checks:
    ok = sym in (root / rel).read_text(encoding="utf-8")
    print(f"   {'✓' if ok else '✗ 缺!'}  {rel} 含 {sym!r}")

print()
print("=" * 60)
print("3. 环境变量(决定 DB 在哪、是否启用)")
print("=" * 60)
for k in ("LLM_MONITOR_ENABLE", "LLM_MONITOR_DB_PATH", "LLM_MONITOR_PORT", "LLM_MONITOR_GPU_SAMPLE_INTERVAL_MS"):
    v = os.environ.get(k)
    print(f"   {k} = {v if v is not None else '<未设置>'}")
if os.environ.get("LLM_MONITOR_ENABLE") != "1":
    print("   ⚠ LLM_MONITOR_ENABLE != 1 —— SGLang 启动时不会自动 install!")

print()
print("=" * 60)
print("4. 当前是否有 SGLang/监控进程在跑(及其启动时刻)")
print("=" * 60)
import subprocess
for pat in ("sglang", "llm_monitor"):
    try:
        out = subprocess.run(["ps", "-eo", "pid,lstart,args"], capture_output=True, text=True, timeout=10).stdout
        hits = [l for l in out.splitlines() if pat in l and "grep" not in l]
        print(f"   [{pat}] 命中 {len(hits)} 条")
        for h in hits[:6]:
            print("     " + h.strip()[:150])
    except Exception as e:
        print(f"   ps 失败: {e}")

print()
print("=" * 60)
print("5. 现有 SQLite 里有没有 request_trace 表(看 db 是否建了新 schema)")
print("=" * 60)
candidates = []
if os.environ.get("LLM_MONITOR_DB_PATH"):
    candidates.append(os.environ["LLM_MONITOR_DB_PATH"])
candidates += ["llm_monitor.db", "/tmp/llm_monitor.db",
               str(Path.cwd() / "llm_monitor.db"),
               str(root / "llm_monitor.db")]
seen = set()
found = False
for p in candidates:
    p = os.path.abspath(p)
    if p in seen or not os.path.exists(p):
        continue
    seen.add(p)
    print(f"   找到库: {p}")
    try:
        con = sqlite3.connect(p)
        tabs = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        print(f"     表: {tabs}")
        has_rt = "request_trace" in tabs
        print(f"     request_trace 表: {'✓ 存在' if has_rt else '✗ 没有(旧 schema,说明新代码没跑过)'}")
        if has_rt:
            n = con.execute("SELECT COUNT(*) FROM request_trace").fetchone()[0]
            hb = con.execute("SELECT COUNT(*) FROM heartbeat").fetchone()[0]
            print(f"     request_trace 行数: {n}   heartbeat 行数: {hb}")
        con.close()
        found = True
    except Exception as e:
        print(f"     读取失败: {e}")
if not found:
    print("   没找到任何 .db —— 若 enable 时没设 DB_PATH,默认在 SGLang 启动的 cwd 建 llm_monitor.db")
print("=" * 60)
print("诊断完成,把以上全部输出发回")
