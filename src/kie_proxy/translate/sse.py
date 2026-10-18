"""Server-sent events parsing/emitting helpers shared by all translators."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class SSEEvent:
    event: str | None
    data: str

    @property
    def json(self) -> object | None:
        if self.data.strip() == "[DONE]":
            return None
        try:
            return json.loads(self.data)
        except ValueError:
            return None

    @property
    def done(self) -> bool:
        return self.data.strip() == "[DONE]"


def encode(event: str | None, data: object) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    out = b""
    if event:
        out += f"event: {event}\n".encode()
    out += f"data: {payload}\n\n".encode()
    return out


def encode_done() -> bytes:
    return b"data: [DONE]\n\n"


async def iter_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[SSEEvent]:
    """Split a byte stream into SSE messages (``event:`` / ``data:`` blocks)."""
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while True:
            sep = _find_separator(buffer)
            if sep is None:
                break
            block, buffer = buffer[: sep[0]], buffer[sep[1] :]
            event = _parse_block(block)
            if event is not None:
                yield event
    tail = buffer.strip()
    if tail:
        event = _parse_block(tail)
        if event is not None:
            yield event


def _find_separator(buffer: bytes) -> tuple[int, int] | None:
    for sep in (b"\n\n", b"\r\n\r\n"):
        idx = buffer.find(sep)
        if idx >= 0:
            return idx, idx + len(sep)
    return None


def _parse_block(block: bytes) -> SSEEvent | None:
    event: str | None = None
    data_lines: list[str] = []
    text = block.decode("utf-8", "replace")
    for line in text.splitlines():
        if line.startswith(":"):
            continue  # comment / keep-alive
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines and event is None:
        return None
    return SSEEvent(event=event, data="\n".join(data_lines))


class UsageSniffer:
    """Watches passthrough SSE traffic for usage/credit fields without buffering it."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.credits: float | None = None

    def feed(self, data: str) -> None:
        try:
            payload = json.loads(data)
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        if isinstance(payload.get("credits_consumed"), (int, float)):
            self.credits = float(payload["credits_consumed"])
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return
        if (prompt := usage.get("prompt_tokens")) or (prompt := usage.get("input_tokens")):
            self.prompt_tokens = int(prompt)
        if (completion := usage.get("completion_tokens")) or (completion := usage.get("output_tokens")):
            self.completion_tokens = int(completion)
