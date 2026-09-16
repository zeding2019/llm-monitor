#!/usr/bin/env python3
"""SGLang patch 诊断:确认补丁是否挂上、数据是否在采集。在跑着 SGLang 的进程环境里执行。"""
import os
import sys

print("=" * 60)
print("1. llm_monitor 安装路径")
print("=" * 60)
import llm_monitor
print(f"   {llm_monitor.__file__}")

print()
print("=" * 60)
print("2. 环境变量")
print("=" * 60)
for k in ["LLM_MONITOR_ENABLE", "LLM_MONITOR_DB_PATH", "LLM_MONITOR_PORT", "PYTHONPATH"]:
    print(f"   {k} = {os.environ.get(k, '<未设置>')}")

print()
print("=" * 60)
print("3. SGLang 模块是否已加载 & 是否被 patch")
print("=" * 60)
targets = [
    ("sglang.srt.managers.scheduler", "Scheduler", "run_batch"),
    ("sglang.srt.managers.scheduler", "Scheduler", "process_batch"),
    ("sglang.srt.entrypoints.http_server", None, None),
    ("sglang.srt.model_executor.model_runner", "ModelRunner", "forward"),
]
for mod_name, cls_name, method in targets:
    loaded = mod_name in sys.modules
    mark = "已加载" if loaded else "未加载"
    detail = ""
    if loaded and cls_name:
        mod = sys.modules[mod_name]
        cls = getattr(mod, cls_name, None)
        if cls and hasattr(cls, method):
            fn = getattr(cls, method)
            patched = hasattr(fn, "__llm_monitor_orig__")
            detail = f" -> {cls_name}.{method} {'✓已打补丁' if patched else '✗未打补丁'}"
        elif cls:
            detail = f" -> {cls_name} 无 {method} 方法"
        else:
            detail = f" -> 模块里无 {cls_name} 类"
    print(f"   [{mark}] {mod_name}{detail}")

print()
print("=" * 60)
print("4. 采集到的数据(注意:REPL 独立进程看不到 SGLang 进程的数据)")
print("=" * 60)
from llm_monitor.core.registry import get_stores
s = get_stores()
print(f"   transactions: {len(s.transactions.snapshot())}")
print(f"   heartbeats:   {len(s.heartbeats.snapshot())}")
print(f"   events:       {len(s.events.snapshot())}")

print()
print("=" * 60)
print("5. SGLang 版本 & 真实的 Scheduler 方法名")
print("=" * 60)
try:
    import sglang
    print(f"   sglang 版本: {getattr(sglang, '__version__', '未知')}")
except ImportError:
    print("   ✗ 当前进程没装/没导入 sglang")

try:
    from sglang.srt.managers import scheduler as sch
    methods = [m for m in dir(sch.Scheduler) if not m.startswith("_") and callable(getattr(sch.Scheduler, m, None))]
    batch_like = [m for m in methods if "batch" in m.lower() or "run" in m.lower() or "step" in m.lower() or "event" in m.lower()]
    print(f"   Scheduler 里 batch/run/step 相关方法: {batch_like}")
except Exception as e:
    print(f"   读取 Scheduler 方法失败: {e}")

print()
print("=" * 60)
print("诊断完成,把以上全部输出发回")
print("=" * 60)
