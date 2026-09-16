from llm_monitor.core.ringbuf import RingBuffer


def test_append_and_snapshot():
    rb: RingBuffer[int] = RingBuffer(3)
    for i in range(5):
        rb.append(i)
    assert rb.snapshot() == [2, 3, 4]


def test_tail():
    rb: RingBuffer[int] = RingBuffer(10)
    for i in range(5):
        rb.append(i)
    assert rb.tail(3) == [2, 3, 4]
    assert rb.tail(100) == [0, 1, 2, 3, 4]


def test_len_and_clear():
    rb: RingBuffer[str] = RingBuffer(4)
    rb.append("a")
    rb.append("b")
    assert len(rb) == 2
    rb.clear()
    assert len(rb) == 0
