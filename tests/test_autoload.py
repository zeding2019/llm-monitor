import subprocess
import sys
from pathlib import Path

from llm_monitor import autoload


def test_enable_disable_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(autoload, "_candidate_site_dirs", lambda: [tmp_path])
    path = autoload.enable()
    assert path == tmp_path / autoload.PTH_NAME
    assert path.exists()
    content = path.read_text()
    assert "LLM_MONITOR_ENABLE" in content
    assert content.startswith("import ")  # .pth 语法要求

    st = autoload.status()
    assert str(path) in st["installed_at"]

    removed = autoload.disable()
    assert removed == [path]
    assert not path.exists()

    # 幂等
    assert autoload.disable() == []


def test_enable_falls_back_when_first_dir_readonly(tmp_path, monkeypatch):
    ro = tmp_path / "ro"
    rw = tmp_path / "rw"
    ro.mkdir()
    rw.mkdir()
    ro.chmod(0o500)  # 只读
    monkeypatch.setattr(autoload, "_candidate_site_dirs", lambda: [ro, rw])
    try:
        path = autoload.enable()
        assert path.parent == rw
    finally:
        ro.chmod(0o700)


def test_pth_line_is_valid_python():
    """.pth 行必须能被 Python 当代码执行,不能抛异常。"""
    line = autoload.PTH_CONTENT.strip()
    # 模拟未启用:不应抛
    r = subprocess.run(
        [sys.executable, "-c", line],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin"},  # 清 env,LLM_MONITOR_ENABLE 缺失
    )
    assert r.returncode == 0, r.stderr


def test_pth_file_loaded_by_fresh_interpreter(tmp_path):
    """真实端到端:把 .pth 放进一个临时 site 目录,让新解释器加载它。"""
    # 用 PYTHONPATH 挂 llm_monitor 源码
    src = Path(__file__).resolve().parents[1] / "src"

    # 假 site 目录,放我们的 .pth
    fake_site = tmp_path / "fakesite"
    fake_site.mkdir()
    (fake_site / autoload.PTH_NAME).write_text(autoload.PTH_CONTENT)

    # sitecustomize 把 fake_site 加入 site.addsitedir,触发 .pth 处理
    (tmp_path / "sitecustomize.py").write_text(
        f"import site; site.addsitedir({str(fake_site)!r})\n"
    )

    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": f"{tmp_path}:{src}",
        "LLM_MONITOR_ENABLE": "1",
        "LLM_MONITOR_PORT": "0",     # 不实际起端口
    }

    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; print('has_monitor=', 'llm_monitor' in sys.modules)"],
        capture_output=True, text=True, env=env, timeout=10,
    )
    assert r.returncode == 0, r.stderr
    assert "has_monitor= True" in r.stdout
