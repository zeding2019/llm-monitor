from llm_monitor.core.api import event, metric, transaction
from llm_monitor.core.registry import get_stores, reset_stores_for_test


def setup_function(_):
    reset_stores_for_test()


def test_transaction_tree_nested():
    with transaction("root", "r") as root:
        assert root.tx_id
        with transaction("child", "c1") as c1:
            with transaction("grand", "g"):
                pass
            assert len(c1.children) == 1
        with transaction("child", "c2"):
            pass
        assert len(root.children) == 2

    stored = get_stores().transactions.snapshot()
    assert len(stored) == 1
    tree = stored[0]
    assert tree.type == "root"
    assert [c.name for c in tree.children] == ["c1", "c2"]
    assert tree.children[0].children[0].type == "grand"
    assert tree.children[0].children[0].parent_id == tree.children[0].tx_id


def test_transaction_records_exception_status():
    try:
        with transaction("op", "fail"):
            raise ValueError("boom")
    except ValueError:
        pass
    stored = get_stores().transactions.snapshot()
    assert stored[-1].status == "ValueError"


def test_event_and_metric():
    event("app", "startup", version="1")
    metric("qps", 3.0, count=1)
    assert len(get_stores().events) == 1
    assert len(get_stores().metrics) == 1
