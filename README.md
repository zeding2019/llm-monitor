# llm-monitor

> **Production-grade observability for vLLM and SGLang — zero code changes required.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-v0%20%7C%20v1-orange.svg)](https://github.com/vllm-project/vllm)
[![SGLang](https://img.shields.io/badge/SGLang-0.4--0.5-purple.svg)](https://github.com/sgl-project/sglang)

Monitor every request, every startup phase, and every GPU across your LLM inference fleet — with a single environment variable and no changes to your inference engine.

```bash
LLM_MONITOR_ENABLE=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B
# Dashboard live at http://127.0.0.1:9109
```


---

## Why llm-monitor?

Most monitoring solutions require you to instrument your code, modify your serving stack, or run a heavyweight sidecar. llm-monitor takes a different approach:

- **Installs via a `.pth` file** — Python loads it before your process does anything
- **Patches engine classes at import time** — no source changes, no forks to maintain
- **Runs in the same process** — no IPC overhead, microsecond-level timing accuracy
- **Handles multi-GPU automatically** — each worker process self-installs and reports per-rank timing

---

## What You Get

### Request-level visibility
Every `generate()` call becomes a traced Transaction with a full message tree:
- Time to first token (TTFT)
- End-to-end latency
- Prefill vs. decode duration
- Per-request token counts

### Startup phase profiling
Know exactly where your cold start time goes:

| Phase | What's measured |
|---|---|
| `env_init` | NCCL / device initialization, per GPU rank |
| `weights` | Model weight loading, per GPU worker |
| `tokenizer_config` | Tokenizer initialization |
| `dist_init` | KV cache allocation |
| `cuda_graph` | CUDA Graph capture / warm-up |
| `total` | Engine `__init__` wall time |

### Host & GPU health
- CPU, memory, disk, network bandwidth — sampled every second
- GPU memory & utilization — via NVML (no CUDA context required)
- Multi-vendor: **NVIDIA**, **Hygon**, **Cambricon**, **Kunlunxin**

### Storage & export
- In-memory ring buffer for real-time queries
- SQLite for historical analysis with configurable retention
- Prometheus exporter for existing dashboards

---

## Quick Start

### Install

```bash
pip install -e ".[dev,nvidia]"
```

### Activate — no code changes needed

```bash
# Recommended: environment variable
export LLM_MONITOR_ENABLE=1
python -m vllm.entrypoints.openai.api_server --model <model>
```

```bash
# Ad-hoc: marker file (survives subprocess env resets, removed on clean exit)
touch /tmp/llm-monitor.cfg
python -m vllm.entrypoints.openai.api_server --model <model>
```

```python
# In scripts or notebooks: explicit import
import llm_monitor  # auto-installs on import
from vllm import LLM

llm = LLM(model="<model>")
llm.generate(["hello"])
```

Open **http://127.0.0.1:9109** for the live dashboard.

### Attach to a custom FastAPI gateway

```python
import llm_monitor
from llm_monitor.patch import LlmMonitorMiddleware
from fastapi import FastAPI

app = FastAPI()
app.add_middleware(LlmMonitorMiddleware)  # wraps every HTTP request as a Transaction
```

---

## How It Works

```
Python process starts  (vllm serve, sglang.launch, your script — anything)
  │
  └─ site-packages/llm_monitor_autoload.pth   ← executed before user code
        └─ LLM_MONITOR_ENABLE=1  or  /tmp/llm-monitor.cfg exists?
              └─ import llm_monitor
                    └─ builtins.__import__ replaced with a hooked version
                          │
                          └─ vLLM / SGLang modules imported later
                                └─ hook fires → engine class methods patched in-place
                                      └─ LLMEngine.__init__, Worker.load_model, ...
                                         now emit Transactions & record phase timings
```

**Multi-process support:** worker subprocesses spawned by vLLM's `MultiprocessingExecutor` or SGLang's scheduler start fresh (no inherited env), but detect `/tmp/llm-monitor.cfg`, self-install patches, and write per-rank metrics to the same shared SQLite database.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_MONITOR_ENABLE` | `0` | Master on/off switch |
| `LLM_MONITOR_PORT` | `9109` | Dashboard port |
| `LLM_MONITOR_HOST` | `127.0.0.1` | Dashboard bind address |
| `LLM_MONITOR_DB_PATH` | `./llm_monitor.db` | SQLite path (empty = in-memory only) |
| `LLM_MONITOR_RETENTION_DAYS` | `7` | History retention window |
| `LLM_MONITOR_GPU` | `auto` | GPU backends: `auto` / `nvidia,hygon,...` |
| `LLM_MONITOR_SAMPLE_INTERVAL_MS` | `1000` | Host metrics sampling interval |
| `LLM_MONITOR_GPU_SAMPLE_INTERVAL_MS` | `5000` | GPU metrics sampling interval |

---

## Project Structure

```
src/llm_monitor/
├── bootstrap.py        install() / uninstall(), per-machine singleton lock
├── config.py           env var parsing and auto-install logic
├── autoload.py         .pth installer / uninstaller (llm-monitor enable/disable)
├── core/               metric models, ring buffer, aggregator, public API
├── sampler/            host + GPU background samplers
├── patch/
│   ├── registry.py     post-import hook engine
│   ├── vllm_startup.py startup phase patches  ← edit _SPECS to add a vLLM version
│   ├── vllm_engine.py  request-level patches (generate, step, scheduler)
│   ├── vllm_kvcache.py KV cache hit rate
│   ├── sglang.py       SGLang patches         ← edit _STARTUP_SPECS for new versions
│   └── util.py         wrap_method / unwrap_method
├── store/              SQLite writer + in-memory ring buffer
└── web/                FastAPI + dashboard
```

### Adding support for a new engine version

All patch targets live in a single declarative table. To adapt to a new vLLM release, add a row — nothing else changes:

```python
# patch/vllm_startup.py
_SPECS: list[tuple[str, str, str, str]] = [
    ...
    # vLLM 0.9: renamed compile_or_warm_up_model → warm_up_model
    ("vllm.v1.worker.gpu_worker", "Worker", "warm_up_model", "cuda_graph"),
]
```

`_try_wrap` silently skips methods that don't exist in the installed version, so old and new rows coexist safely across versions.

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format .
```

---

## Roadmap

- [x] Startup phase profiling with per-GPU-rank breakdown
- [x] vLLM v0 + v1 (sync and async engines, EngineCoreProc subprocess)
- [x] SGLang request tracking, KV cache hit rate, queue depth
- [x] Multi-vendor GPU support (NVIDIA, Hygon, Cambricon, Kunlunxin)
- [ ] Transaction tree UI in dashboard
- [ ] Prometheus exporter
- [ ] Slow-request sampling with full context retention
- [ ] PyPI release

---

## License

MIT
