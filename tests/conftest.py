from __future__ import annotations

import json
from pathlib import Path

import pytest

from kie_proxy.translate.sse import SSEEvent, iter_sse

FIXTURES = Path(__file__).parent / "fixtures"


def load_body(name: str) -> str:
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return payload["body"]


def load_json(name: str) -> dict:
    return json.loads(load_body(name))


async def sse_from(text: str) -> list[SSEEvent]:
    async def chunks():
        # split arbitrarily to exercise the buffer logic
        for i in range(0, len(text), 37):
            yield text[i : i + 37].encode()

    return [event async for event in iter_sse(chunks())]


async def collect_bytes(async_iter) -> list[bytes]:
    return [chunk async for chunk in async_iter]


def sse_json_payloads(text: str) -> list[dict]:
    out: list[dict] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


def sse_events_payloads(text: str) -> list[dict]:
    """JSON payloads of events, including `event:`-labelled anthropic frames."""
    out: list[dict] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


@pytest.fixture
def chat_body() -> dict:
    return {
        "model": "claude-haiku-4-5",
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Say only: hi"},
        ],
        "max_tokens": 100,
    }
