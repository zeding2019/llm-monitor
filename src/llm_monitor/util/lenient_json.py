"""宽松 JSON 解析(用于 hy-smi / rocm-smi 等 GPU 厂商 CLI 的 --json 输出)。

这些工具声称支持 --json,但真实输出经常不是合法 JSON,常见问题:

  1. object / array 成员之间漏逗号,例如:  {"card0": {...} "card1": {...}}
  2. 数值后面直接跟单位且未加引号,例如:   "x": 23%   "y": 41.0C   "z": 1300MHz
  3. 多个顶层 JSON 文档拼接 / 文档前后夹带说明文本 / 文本混进文档内部
  4. "N/A"、"unknown" 之类的占位符

解析策略(不引入第三方依赖):
  - 顶层按 `{ }` / `[ ]` 配对切出若干段(能正确跳过字符串里的括号),
    丢弃无法配对的花括号文本,多个顶层对象最终合并成一个 dict;
  - 每段用递归下降解析,允许漏逗号、值后跟单位后缀、N/A 占位;
  - 键必须是引号字符串(与严格 JSON 相同)——如果连键都没有引号,
    说明格式和我们见过的 hy-smi 输出差异太大,这里选择报错而不是猜。

调用方应先试严格 json.loads(合法输出零开销),失败后再用 lenient_loads。
"""

from __future__ import annotations

import re

_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
# 原子值(裸数字/裸词)在这些字符处结束
_ATOM_STOP = frozenset(' \t\r\n,]}":{[')
# 数值后面允许直接跟上的单位后缀(排除 e/E,避免吃掉科学计数法指数)
_UNIT_SUFFIX_RE = re.compile(r"[%A-Za-z]+")
_NONE_TOKENS = frozenset({"n/a", "na", "nan", "none", "null", "unknown", "-"})


def lenient_loads(text: str):
    """宽容解析 hy-smi 这类输出,返回 Python 值。

    顶层出现多个文档时:多个 dict 合并为一个;多个 list 合并为一个 list。
    解析不了任何内容时抛 ValueError。
    """
    s = (text or "").lstrip("\ufeff")
    docs: list = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c not in "{[":
            i += 1
            continue
        end = _balanced_end(s, i)
        if end < 0:
            # 无法配对:当成说明文本跳过(例如日志里的花括号)
            i += 1
            continue
        seg = s[i:end]
        try:
            val, _ = _parse_value(seg, 0)
        except ValueError:
            # 配平但内容解析不了:说明格式超出预期,宁可直接报错暴露原始输出,
            # 也不要静默返回不完整/错误的数据。
            raise
        docs.append(val)
        i = end
    if not docs:
        raise ValueError("no JSON container found in output")
    return _merge_docs(docs)


# ---- 顶层切分 ----------------------------------------------------------


def _balanced_end(text: str, start: int) -> int:
    """返回从 start 处开始的容器配平后的结束下标(不含);无法配平返回 -1。"""
    open_ch = text[start]
    n = len(text)
    i = start + 1
    stack = [open_ch]
    while i < n:
        c = text[i]
        if c == '"':
            i = _string_end(text, i)
            continue
        if c in "{[":
            stack.append(c)
        elif c in "}]":
            if not stack or stack[-1] != ("{" if c == "}" else "["):
                return -1
            stack.pop()
            if not stack:
                return i + 1
        i += 1
    return -1


def _string_end(text: str, i: int) -> int:
    """text[i] == '"',返回其后第一个未转义引号的下标(不含);未闭合返回 n。"""
    n = len(text)
    j = i + 1
    while j < n:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == '"':
            return j + 1
        j += 1
    return n


# ---- 递归下降(单段,下标从 0 开始) -------------------------------------


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in " \t\r\n":
        i += 1
    return i


def _parse_value(text: str, i: int):
    i = _skip_ws(text, i)
    n = len(text)
    if i >= n:
        raise ValueError(f"unexpected end of input at char {i}")
    c = text[i]
    if c == "{":
        return _parse_object(text, i)
    if c == "[":
        return _parse_array(text, i)
    if c == '"':
        return _read_string(text, i)
    return _read_atom(text, i)


def _parse_object(text: str, i: int):
    obj: dict = {}
    n = len(text)
    i = _skip_ws(text, i + 1)
    while i < n:
        c = text[i]
        if c == "}":
            return obj, i + 1
        if c == ",":
            i += 1  # 容错:多余/前置逗号
            continue
        if c != '"':
            # 成员之间夹着非引号文本(hy-smi 常见),跳到下一个 key / 收尾
            nxt = text.find('"', i + 1)
            if nxt == -1:
                raise ValueError(f"unterminated object: expected quoted key at char {i}")
            i = nxt
            continue
        key, i = _read_string(text, i)
        i = _skip_ws(text, i)
        if i < n and text[i] == ":":
            i = _skip_ws(text, i + 1)
        else:
            # 极宽松:缺冒号就当缺值,继续找下一成员
            if i >= n or text[i] in "}":
                obj[key] = None
                continue
            k = text.find(":", i)
            q = text.find('"', i)
            if k == -1 or (q != -1 and q < k):
                obj[key] = None
                continue
            i = _skip_ws(text, k + 1)
        if i >= n:
            raise ValueError(f"unterminated object after key {key!r}")
        if text[i] in "}]":
            obj[key] = None  # 缺值
            continue
        val, i = _parse_value(text, i)
        obj[key] = val
        i = _skip_ws(text, i)
        if i < n and text[i] == ",":
            i += 1  # 正常分隔;漏逗号则原地进入下一轮循环
    raise ValueError(f"unterminated object (EOF), near {str(obj)[:120]}")


def _parse_array(text: str, i: int):
    arr: list = []
    n = len(text)
    i = _skip_ws(text, i + 1)
    while i < n:
        c = text[i]
        if c == "]":
            return arr, i + 1
        if c == ",":
            i += 1
            continue
        val, i = _parse_value(text, i)
        arr.append(val)
        i = _skip_ws(text, i)
        if i < n and text[i] == ",":
            i += 1
        # 漏逗号:下一轮直接解析下一个元素
    raise ValueError("unterminated array (EOF)")


def _read_atom(text: str, i: int):
    """读取裸值(数字/裸词/占位符),结束于 _ATOM_STOP 中的字符或 EOF。"""
    n = len(text)
    j = i
    while j < n and text[j] not in _ATOM_STOP:
        j += 1
    if j == i:
        # 空原子(例如 "a": ,"b": 1 里的逗号) —— 消费一个字符保证前进
        return _atom_value(""), min(i + 1, n)
    return _atom_value(text[i:j]), j


def _atom_value(tok: str):
    t = tok.strip()
    if not t:
        return None
    low = t.lower()
    if low in _NONE_TOKENS:
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    m = _NUM_RE.match(t)
    if m:
        num, rest = m.group(), t[m.end() :]
        if not rest or _UNIT_SUFFIX_RE.fullmatch(rest):
            # 整数形式(且不含小数点/指数)返回 int,避免 34359738368 这类值失真
            if any(ch in num for ch in ".eE"):
                try:
                    return float(num)
                except ValueError:
                    pass
            else:
                try:
                    return int(num)
                except ValueError:
                    pass
    return t  # 0x67e1 这类不是可解析数值的,保持字符串原样


def _read_string(text: str, i: int):
    """text[i] == '"',读取字符串字面量(可含转义、可含原始换行,未闭合尽量返回)。"""
    n = len(text)
    j = i + 1
    buf: list[str] = []
    simple = {
        "n": "\n",
        "t": "\t",
        "r": "\r",
        "b": "\b",
        "f": "\f",
        '"': '"',
        "\\": "\\",
        "/": "/",
    }
    while j < n:
        c = text[j]
        if c == '"':
            return "".join(buf), j + 1
        if c == "\\":
            if j + 1 >= n:
                buf.append("\\")
                j += 1
                continue
            e = text[j + 1]
            if e == "u":
                try:
                    ch = chr(int(text[j + 2 : j + 6], 16))
                except ValueError:
                    buf.append("u")
                    j += 2
                    continue
                if 0xD800 <= ord(ch) <= 0xDBFF:
                    tail = text[j + 6 : j + 12]
                    if tail.startswith("\\u"):
                        try:
                            lo = chr(int(tail[2:6], 16))
                            valid_lo = 0xDC00 <= ord(lo) <= 0xDFFF
                        except ValueError:
                            lo, valid_lo = "", False
                        if valid_lo:
                            buf.append(ch + lo)
                            j += 12
                            continue
                buf.append(ch)
                j += 6
                continue
            buf.append(simple.get(e, e))
            j += 2
            continue
        buf.append(c)
        j += 1
    return "".join(buf), n


def _merge_docs(docs: list):
    dicts = [d for d in docs if isinstance(d, dict)]
    lists = [d for d in docs if isinstance(d, list)]
    if dicts:
        out: dict = {}
        for d in dicts:
            out.update(d)
        return out
    if lists:
        merged: list = []
        for lst in lists:
            merged.extend(lst)
        return merged
    return docs[-1]  # 顶层标量,极罕见
