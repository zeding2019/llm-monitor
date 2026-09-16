# llm-monitor

A lightweight, zero-intrusion request-level monitor for LLM inference engines (vLLM and SGLang). Deployed in the same process as the inference engine via monkey-patching, with no changes required to engine source code.

**Features:**

- **Per-request Transactions** — full message tree with TTFT, e2e latency, prefill/decode breakdown
- **Host health** — CPU, memory, disk, network bandwidth, GPU memory & utilization
- **Multi-vendor GPU support** — NVIDIA (pynvml), Hygon, Cambricon, Kunlunxin (plugin-based, opt-in)
- **Four metric types** — Transaction / Event / Heartbeat / Metric
- **Real-time Web UI** + **Prometheus exporter**
- **In-memory + SQLite** dual-layer storage with historical query support
- **Startup phase profiling** — per-stage timing (weights load, NCCL init, KV cache, CUDA graph compile) across all GPU workers

## Design Principles

1. **Zero intrusion** — vLLM/SGLang source is untouched; set `LLM_MONITOR_ENABLE=1` or `import llm_monitor` to activate
2. **Low overhead** — hot path does only `perf_counter` + ring-buffer append; aggregation and GPU sampling run in background threads
3. **No resource contention** — GPU metrics via NVML (no CUDA context); Web server runs single-threaded uvicorn on localhost

## Quick Start

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,nvidia]"
```

### Option 1: vLLM OpenAI server

```bash
export LLM_MONITOR_ENABLE=1
export LLM_MONITOR_PORT=9109
python -m vllm.entrypoints.openai.api_server --model <model>
# Open http://127.0.0.1:9109
```

### Option 2: Python API

```python
import llm_monitor  # auto-installs on import
from vllm import LLM

llm = LLM(model="<model>")
llm.generate(["hello"])
```

### Option 3: Custom FastAPI gateway

If you wrap the engine behind your own FastAPI app and want HTTP requests as top-level Transactions:

```python
import llm_monitor
from llm_monitor.patch import LlmMonitorMiddleware
from fastapi import FastAPI

app = FastAPI()
app.add_middleware(LlmMonitorMiddleware)
```

## Metric Types

| Type | Purpose | Examples |
|---|---|---|
| Transaction | Timed span, nestable into a tree | generate request, prefill, decode |
| Event | One-shot occurrence | OOM, request cancellation, exception |
| Heartbeat | Periodic sampled value | GPU memory, CPU usage, bandwidth |
| Metric | Business counter / sum / avg | tokens/s, prompt tokens, generated tokens |

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `LLM_MONITOR_ENABLE` | `0` | Master switch |
| `LLM_MONITOR_HOST` | `127.0.0.1` | Web server bind address |
| `LLM_MONITOR_PORT` | `9109` | Web server port |
| `LLM_MONITOR_GPU` | `auto` | `auto` or comma-separated backends: `nvidia,hygon,...` |
| `LLM_MONITOR_SAMPLE_INTERVAL_MS` | `1000` | Host sampling interval (CPU / memory / network) |
| `LLM_MONITOR_GPU_SAMPLE_INTERVAL_MS` | `5000` | GPU sampling interval (separate thread, non-blocking) |
| `LLM_MONITOR_HYGON_TIMEOUT` | `15` | Timeout in seconds for hy-smi / rocm-smi commands |
| `LLM_MONITOR_DB_PATH` | `./llm_monitor.db` | SQLite path; leave empty for in-memory only |
| `LLM_MONITOR_RETENTION_DAYS` | `7` | SQLite retention window in days |
| `LLM_MONITOR_TRANSACTION_BUF` | `10000` | In-memory ring buffer capacity |

## How It Works

llm-monitor uses a **post-import hook** pattern to patch engine classes without touching their source:

1. A `.pth` file in site-packages runs one line on every Python process startup — guarded by `LLM_MONITOR_ENABLE=1` or the presence of `/tmp/llm-monitor.cfg`.
2. On activation, `builtins.__import__` is replaced with a hooked version that fires registered callbacks whenever a target module first appears in `sys.modules`.
3. Callbacks call `wrap_method` to replace class methods with instrumented versions that record timing and emit Transactions / Heartbeats.
4. Worker subprocesses (spawned by vLLM's `MultiprocessingExecutor` or SGLang's scheduler) pick up the marker file at `/tmp/llm-monitor.cfg` and install patches independently, each writing their own per-rank startup timings to the shared SQLite database.

## Project Structure

```
src/llm_monitor/
├── bootstrap.py     install() / uninstall(), singleton lock
├── config.py        env vars and defaults
├── core/            engine-agnostic core (models, context, ring buffer, api, aggregator)
├── sampler/         background samplers (host + gpu/*)
├── patch/           monkey-patches for vLLM and SGLang
│   ├── registry.py      post-import hook infrastructure
│   ├── vllm_startup.py  startup phase timing (declarative _SPECS table)
│   ├── sglang.py        SGLang patches (startup + runtime)
│   └── util.py          wrap_method, unwrap_method
├── store/           in-memory ring buffer + SQLite writer
└── web/             FastAPI + static dashboard
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format .
```

## Roadmap

- [x] Phase 1: scaffold + host sampling + minimal Web UI
- [ ] Phase 2: vLLM patches (generate / step / scheduler) + NVIDIA GPU + Transaction tree UI
- [ ] Phase 3: Prometheus exporter, Hygon / Cambricon / Kunlunxin collectors, slow-transaction sampling

## License

MIT
