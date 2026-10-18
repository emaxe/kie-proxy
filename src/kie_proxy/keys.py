"""Round-robin pool of KIE API keys with failure handling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

from .config import KeyEntry


class KeyStatus(StrEnum):
    OK = "ok"
    COOLDOWN = "cooldown"
    EXHAUSTED = "exhausted"
    INVALID = "invalid"


COOLDOWN_SECONDS = 60.0
EXHAUSTED_COOLDOWN_SECONDS = 300.0


@dataclass
class KeyState:
    key: str
    label: str = ""
    status: KeyStatus = KeyStatus.OK
    until: float = 0.0
    credits: float | None = None
    errors: int = 0
    last_error: str = ""
    last_used: float = 0.0

    @property
    def masked(self) -> str:
        if len(self.key) <= 8:
            return "***"
        return f"{self.key[:4]}…{self.key[-4:]}"

    @property
    def label_or_masked(self) -> str:
        return self.label or self.masked

    def ready(self, now: float) -> bool:
        if self.status is KeyStatus.INVALID:
            return False
        if self.status is KeyStatus.OK:
            return True
        return now >= self.until

    def refresh(self, now: float) -> None:
        if self.status in (KeyStatus.COOLDOWN, KeyStatus.EXHAUSTED) and now >= self.until:
            self.status = KeyStatus.OK
            self.until = 0.0


class NoKeyAvailable(Exception):
    pass


class KeyPool:
    def __init__(self, entries: list[KeyEntry]):
        if not entries:
            raise ValueError("no API keys configured")
        self._keys = [
            KeyState(key=e.key, label=e.label or f"key-{i + 1}") for i, e in enumerate(entries)
        ]
        self._cursor = 0
        self._snapshot: list[KeyState] = []

    @property
    def states(self) -> list[KeyState]:
        return list(self._keys)

    def acquire(self) -> KeyState:
        now = time.time()
        n = len(self._keys)
        for _ in range(n):
            state = self._keys[self._cursor % n]
            self._cursor += 1
            state.refresh(now)
            if state.ready(now):
                state.last_used = now
                return state
        raise NoKeyAvailable("all keys are rate-limited, exhausted or invalid")

    def get(self, key: str) -> KeyState | None:
        return next((k for k in self._keys if k.key == key), None)

    def mark_rate_limited(self, key: str, message: str = "") -> None:
        if state := self.get(key):
            state.status = KeyStatus.COOLDOWN
            state.until = time.time() + COOLDOWN_SECONDS
            state.errors += 1
            state.last_error = message or "rate limited"

    def mark_exhausted(self, key: str, message: str = "") -> None:
        if state := self.get(key):
            state.status = KeyStatus.EXHAUSTED
            state.until = time.time() + EXHAUSTED_COOLDOWN_SECONDS
            state.errors += 1
            state.last_error = message or "insufficient credits"

    def mark_invalid(self, key: str, message: str = "") -> None:
        if state := self.get(key):
            state.status = KeyStatus.INVALID
            state.errors += 1
            state.last_error = message or "invalid key"

    def report_success(self, key: str) -> None:
        if state := self.get(key):
            state.errors = 0
            state.last_error = ""

    def set_credits(self, key: str, credits: float) -> None:
        if state := self.get(key):
            state.credits = credits

    def snapshot(self) -> list[KeyState]:
        now = time.time()
        out = []
        for state in self._keys:
            state.refresh(now)
            out.append(
                KeyState(
                    key=state.key,
                    label=state.label,
                    status=state.status,
                    until=state.until,
                    credits=state.credits,
                    errors=state.errors,
                    last_error=state.last_error,
                    last_used=state.last_used,
                )
            )
        self._snapshot = out
        return out

    def __len__(self) -> int:
        return len(self._keys)
