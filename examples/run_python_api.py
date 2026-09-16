"""Python API 演示。运行后打开 http://127.0.0.1:9109"""
import os
import time

os.environ["LLM_MONITOR_ENABLE"] = "1"

import llm_monitor
from llm_monitor import event, metric, transaction


def fake_generate(prompt: str) -> str:
    with transaction("vllm.generate", "req-demo") as tx:
        tx.data["prompt_tokens"] = len(prompt)
        with transaction("vllm.prefill", "prefill"):
            time.sleep(0.05)
        gen_tokens = 0
        for _ in range(20):
            with transaction("vllm.decode", "decode"):
                time.sleep(0.01)
                gen_tokens += 1
                metric("vllm.tokens", 1)
        tx.data["gen_tokens"] = gen_tokens
        return "hello"


if __name__ == "__main__":
    print("llm-monitor installed:", llm_monitor.__version__)
    for i in range(1000):
        fake_generate("hello world " * 10)
        if i % 50 == 0:
            event("demo", "tick", ok=True, i=i)
        time.sleep(0.2)
