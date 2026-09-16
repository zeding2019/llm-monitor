"""util/lenient_json:hy-smi 宽松 JSON 解析器单测。"""

import json

import pytest

from llm_monitor.util.lenient_json import lenient_loads


def test_valid_json_parses_identically():
    samples = [
        '{"card0": {"a": 1, "b": "x", "c": [1, 2, 3], "d": null}}',
        '[{"a": 1}, {"b": "中"}, 3.5, true, false]',
        '{"s": "a\\nb\\u4e2d", "f": 1.5e3}',
    ]
    for s in samples:
        assert lenient_loads(s) == json.loads(s)


def test_missing_commas_between_object_members():
    assert lenient_loads('{"a": 1 "b": 2 "c": 3}') == {"a": 1, "b": 2, "c": 3}


def test_missing_comma_between_nested_and_top_level_members():
    # 复刻线上报错形态:成员之间漏逗号,严格 json.loads 报
    # "Expecting ',' delimiter"(与本仓库 hygon 采集器最初的报错一致)
    s = (
        '{"card0": {"GPU use (%)": "0%", "GPU Memory Allocated (VRAM%)": "0%" '
        '"GPU memory use (%)": "0%"} "card1": {"GPU use (%)": "7%"}}'
    )
    with pytest.raises(json.JSONDecodeError, match=r"Expecting ',' delimiter"):
        json.loads(s)
    data = lenient_loads(s)
    assert set(data) == {"card0", "card1"}
    assert data["card1"]["GPU use (%)"] == "7%"


def test_missing_commas_between_array_elements():
    assert lenient_loads("[1 2 3]") == [1, 2, 3]
    assert lenient_loads('[{"a": 1} {"b": 2}]') == [{"a": 1}, {"b": 2}]


def test_unquoted_values_with_unit_suffix():
    # "23%" "41.0C" "80W" "1300MHz" 这类裸值带单位,应解析成数字
    s = (
        '{"GPU use (%)": 23%, "Temperature (Sensor edge) (C)": 41.0C, '
        '"Average Graphics Package Power (W)": 80W, "sclk (MHz)": 1300MHz}'
    )
    assert lenient_loads(s) == {
        "GPU use (%)": 23,
        "Temperature (Sensor edge) (C)": 41.0,
        "Average Graphics Package Power (W)": 80,
        "sclk (MHz)": 1300,
    }


def test_leading_and_trailing_junk_text():
    s = 'Hygon System Management Interface\n{"card0": {"u": "0%"}}\ndone'
    assert lenient_loads(s) == {"card0": {"u": "0%"}}


def test_multiple_top_level_objects_merged():
    # 每卡单独打一个对象的情况,应合并成一个 dict
    s = '{"card0": {"u": "0%"}}\n{"card1": {"u": "7%"}}\n'
    assert lenient_loads(s) == {"card0": {"u": "0%"}, "card1": {"u": "7%"}}
    assert lenient_loads("[1] [2]") == [1, 2]


def test_placeholder_and_null_atoms():
    assert lenient_loads('{"a": N/A, "b": null, "c": unknown}') == {
        "a": None,
        "b": None,
        "c": None,
    }


def test_hexish_value_kept_as_string():
    # "0x67e1" 应保留字符串,不能被解析成数字 0
    assert lenient_loads('{"Card model": 0x67e1}') == {"Card model": "0x67e1"}


def test_missing_quotes_around_key_raises():
    # 键无引号的畸形太不可靠,选择报错而不是猜
    with pytest.raises(ValueError):
        lenient_loads("{GPU use (%): 0%}")


def test_unparseable_input_raises():
    for bad in ("", "hello world", "no json here at all", "[unbalanced"):
        with pytest.raises(ValueError):
            lenient_loads(bad)


def test_realistic_rocm_smi_style_is_round_tripped():
    # rocm-smi 风格的合法 JSON 走严格语义也应原样读回
    s = json.dumps(
        {
            "card0": {
                "GPU use (%)": "0%",
                "VRAM Total Memory (B)": 34359738368,
                "Temperature (Sensor edge) (C)": "41.0C",
            }
        }
    )
    assert lenient_loads(s) == json.loads(s)
