from llm_monitor.config import Config


def test_default_disabled():
    c = Config()
    assert c.enable is False
    assert c.port == 9109


def test_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MONITOR_ENABLE", "1")
    monkeypatch.setenv("LLM_MONITOR_PORT", "9200")
    monkeypatch.setenv("LLM_MONITOR_GPU", "nvidia,hygon")
    c = Config.from_env()
    assert c.enable is True
    assert c.port == 9200
    assert c.gpu_backends == ["nvidia", "hygon"]
