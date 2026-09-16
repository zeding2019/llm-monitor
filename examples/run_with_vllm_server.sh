#!/usr/bin/env bash
# 演示以环境变量方式挂载 llm-monitor 到 vLLM OpenAI server
set -euo pipefail

export LLM_MONITOR_ENABLE=1
export LLM_MONITOR_HOST=127.0.0.1
export LLM_MONITOR_PORT=9109

# 关键:通过 -c 让解释器一进来就 import llm_monitor,再启动 vllm server
python -c "import llm_monitor; import runpy; runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')" \
  --model "${MODEL:-Qwen/Qwen2-0.5B-Instruct}" \
  --host 0.0.0.0 --port 8000
