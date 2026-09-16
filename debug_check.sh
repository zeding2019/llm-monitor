#!/bin/bash
set -e
cd /Users/apple/Documents/vllm-monitor

echo "========== 1. 安装状态 =========="
.venv/bin/python -c "import llm_monitor; print(f'llm_monitor 版本/路径: {llm_monitor.__file__}')"

echo
echo "========== 2. 补丁注册检查 =========="
.venv/bin/python - <<'PY'
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

# 模拟 bootstrap.install()
from llm_monitor.patch import install_all
install_all()
print("✓ install_all() 完成")

# 触发 vLLM import (如果环境里有)
try:
    import vllm.entrypoints.openai.api_server
    print("✓ vllm.entrypoints.openai.api_server 已导入")
except ImportError:
    print("✗ vLLM 未安装(本机没有,正常)")
PY

echo
echo "========== 3. 环境变量 =========="
echo "LLM_MONITOR_ENABLE=${LLM_MONITOR_ENABLE:-未设置}"
echo "PYTHONPATH=${PYTHONPATH:-未设置}"

echo
echo "========== 4. autoload.pth 检查 =========="
.venv/bin/python -c "import site; print('\n'.join(site.getsitepackages()))" | while read sp; do
  if [ -f "$sp/llm_monitor_autoload.pth" ]; then
    echo "✓ 找到: $sp/llm_monitor_autoload.pth"
    cat "$sp/llm_monitor_autoload.pth"
  fi
done
if ! .venv/bin/python -c "import site; any('llm_monitor_autoload.pth' in open(f'{sp}/llm_monitor_autoload.pth').read() for sp in site.getsitepackages() if __import__('os').path.exists(f'{sp}/llm_monitor_autoload.pth'))" 2>/dev/null; then
  echo "✗ 未找到 .pth 文件(需要 pip install -e . 安装)"
fi

echo
echo "========== 5. 采集到的数据快照 =========="
.venv/bin/python - <<'PY'
from llm_monitor.core.registry import get_stores
stores = get_stores()
print(f"transactions: {len(stores.transactions.snapshot())} 条")
print(f"heartbeats: {len(stores.heartbeats.snapshot())} 条")
print(f"events: {len(stores.events.snapshot())} 条")
if stores.heartbeats.snapshot():
    sources = {h.source for h in stores.heartbeats.snapshot()}
    print(f"heartbeat sources: {sorted(sources)}")
PY

echo
echo "========== 诊断完成 =========="
echo "把上面全部输出发给我"
