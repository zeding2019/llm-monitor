"""进程间安装标记:清空 env 启动的 SGLang 子进程靠它装上 llm_monitor 并写同一 DB。"""
from llm_monitor.bootstrap import _marker_path, read_marker, write_marker
from llm_monitor.config import Config


def _cleanup():
    import contextlib
    import os
    with contextlib.suppress(OSError):
        os.unlink(_marker_path())


def test_marker_roundtrip():
    _cleanup()
    try:
        cfg = Config(enable=True, port=9199, db_path="./sub/dir/llm_monitor.db")
        write_marker(cfg)
        m = read_marker()
        assert m["enable"] == "1"
        assert m["port"] == "9199"
        assert m["db"].endswith("sub/dir/llm_monitor.db") and m["db"].startswith("/")
    finally:
        _cleanup()


def test_read_marker_missing_returns_empty():
    _cleanup()
    assert read_marker() == {}


def test_auto_install_marker_driven_uses_marker_db(monkeypatch):
    """env 无 enable 时,标记存在 → 仍安装,且 DB 用主进程写下的绝对路径。"""
    _cleanup()
    captured = {}

    def fake_install(cfg):
        captured["cfg"] = cfg

    import llm_monitor.bootstrap as B
    monkeypatch.setattr(B, "install", fake_install)
    try:
        write_marker(Config(enable=True, port=9123, db_path="/tmp/llm_monitor_shared.db"))
        cfg = Config(enable=False)  # 模拟清空 env 的子进程
        cfg.auto_install()
        assert captured["cfg"].enable is True
        assert captured["cfg"].db_path == "/tmp/llm_monitor_shared.db"
        assert captured["cfg"].port == 9123
    finally:
        monkeypatch.undo()
        _cleanup()


def test_auto_install_no_marker_no_env_is_noop(monkeypatch):
    _cleanup()
    called = []

    def fake_install(cfg):
        called.append(cfg)

    import llm_monitor.bootstrap as B
    monkeypatch.setattr(B, "install", fake_install)
    try:
        Config(enable=False).auto_install()
        assert called == []
    finally:
        monkeypatch.undo()
