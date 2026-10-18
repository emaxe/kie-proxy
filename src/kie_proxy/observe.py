"""Request log, statistics and the event bus the TUI renders."""

from __future__ import annotations

import contextlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from .errors import UpstreamError


@dataclass
class RequestEvent:
    ts: float
    path: str
    model: str
    provider: str = ""
    inbound: str = ""
    native: str = ""
    status: int = 0
    ms: int = 0
    stream: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    credits: float | None = None
    error: str = ""
    key: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 400 and not self.error


@dataclass
class Stats:
    requests: int = 0
    errors: int = 0
    stream_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    credits: float = 0.0
    started_at: float = field(default_factory=time.time)


class Observer:
    def __init__(self, limit: int = 500):
        self.events: deque[RequestEvent] = deque(maxlen=limit)
        self.stats = Stats()
        self.listeners: list[Callable[[RequestEvent], None]] = []

    def add_listener(self, fn: Callable[[RequestEvent], None]) -> None:
        self.listeners.append(fn)

    def record(self, event: RequestEvent) -> None:
        self.events.appendleft(event)
        st = self.stats
        st.requests += 1
        if not event.ok:
            st.errors += 1
        if event.stream:
            st.stream_requests += 1
        st.prompt_tokens += event.prompt_tokens
        st.completion_tokens += event.completion_tokens
        if event.credits:
            st.credits += event.credits
        for fn in self.listeners:
            with contextlib.suppress(Exception):
                fn(event)

    @staticmethod
    def from_error(exc: UpstreamError) -> str:
        return exc.message


def usage_from_chat(payload: object) -> tuple[int, int, float | None]:
    """Extract (prompt, completion, credits) from an OpenAI-style response body."""
    if not isinstance(payload, dict):
        return 0, 0, None
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    credits = payload.get("credits_consumed")
    return prompt, completion, float(credits) if isinstance(credits, (int, float)) else None


def usage_from_anthropic(payload: object) -> tuple[int, int, float | None]:
    if not isinstance(payload, dict):
        return 0, 0, None
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    credits = payload.get("credits_consumed")
    return prompt, completion, float(credits) if isinstance(credits, (int, float)) else None
