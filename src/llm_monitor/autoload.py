"""安装/卸载 site-packages 里的 .pth 自动加载文件。

原理:Python 启动时会扫 site-packages 里所有 .pth 文件,以 `import` 开头
的行会被执行。我们放一行受环境变量控制的懒 import:

    import os; os.environ.get('LLM_MONITOR_ENABLE')=='1' and __import__('llm_monitor')

这样:
- vLLM 的启动命令完全不用改
- 未设 LLM_MONITOR_ENABLE=1 时,零副作用(只跑一次 os.environ.get)
- 卸载:`llm-monitor disable` 或 `pip uninstall`(uninstall 只清 pkg,.pth 也一并清)
"""
from __future__ import annotations

import site
import sys
from pathlib import Path

PTH_NAME = "llm_monitor_autoload.pth"
PTH_CONTENT = (
    "import os; os.environ.get('LLM_MONITOR_ENABLE')=='1'"
    " and __import__('llm_monitor')\n"
)


def _candidate_site_dirs() -> list[Path]:
    """当前解释器的 site-packages 候选目录,优先 venv,再全局。"""
    dirs: list[Path] = []
    # venv / project site
    for d in site.getsitepackages():
        p = Path(d)
        if p.exists():
            dirs.append(p)
    # user site
    user = site.getusersitepackages()
    if user and Path(user).exists():
        dirs.append(Path(user))
    # 去重,保序
    seen = set()
    uniq = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def enable() -> Path:
    """装 .pth 到当前解释器的第一个可写 site-packages。返回文件路径。"""
    for d in _candidate_site_dirs():
        target = d / PTH_NAME
        try:
            target.write_text(PTH_CONTENT, encoding="utf-8")
            return target
        except OSError:
            continue
    raise RuntimeError(
        "no writable site-packages found. "
        f"Tried: {[str(d) for d in _candidate_site_dirs()]}"
    )


def disable() -> list[Path]:
    """从所有已知 site-packages 里清掉 .pth,返回清掉的路径列表。"""
    removed: list[Path] = []
    for d in _candidate_site_dirs():
        p = d / PTH_NAME
        if p.exists():
            try:
                p.unlink()
                removed.append(p)
            except OSError:
                pass
    return removed


def status() -> dict:
    """返回当前解释器可见的所有 .pth 位置。"""
    return {
        "python": sys.executable,
        "installed_at": [str(d / PTH_NAME) for d in _candidate_site_dirs() if (d / PTH_NAME).exists()],
        "site_dirs": [str(d) for d in _candidate_site_dirs()],
    }
