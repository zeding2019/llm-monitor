from llm_monitor.core.registry import get_stores, reset_stores_for_test
from llm_monitor.sampler import host


def setup_function(_):
    reset_stores_for_test()


def test_host_sample_emits_heartbeat():
    host.sample()  # 首次
    host.sample()  # 第二次才有 net_bps
    items = get_stores().heartbeats.snapshot()
    assert len(items) == 2
    hb = items[-1]
    assert hb.source == "host"
    assert "cpu_pct" in hb.values
    assert "mem_pct" in hb.values
    assert "net_tx_bps" in hb.values
