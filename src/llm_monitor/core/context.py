"""请求级 Message Tree 上下文。

用 contextvars 承载,天然支持协程与线程。
"""
from __future__ import annotations

import contextvars
import uuid

from .models import Transaction

_current: contextvars.ContextVar[TreeContext | None] = contextvars.ContextVar(
    "llm_monitor_tree_ctx", default=None
)


class TreeContext:
    """一个请求内的消息树。"""

    def __init__(self, root: Transaction) -> None:
        self.root = root
        self._stack: list[Transaction] = [root]

    def current(self) -> Transaction:
        return self._stack[-1]

    def push(self, tx: Transaction) -> None:
        parent = self._stack[-1]
        tx.parent_id = parent.tx_id
        parent.children.append(tx)
        self._stack.append(tx)

    def pop(self) -> Transaction:
        return self._stack.pop()

    def is_root_only(self) -> bool:
        return len(self._stack) == 1


def new_tx_id() -> str:
    return uuid.uuid4().hex[:16]


def get_context() -> TreeContext | None:
    return _current.get()


def set_context(ctx: TreeContext | None) -> contextvars.Token:
    return _current.set(ctx)


def reset_context(token: contextvars.Token) -> None:
    _current.reset(token)
