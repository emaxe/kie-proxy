"""Protocol translation between the canonical wire format (OpenAI Chat Completions)
and each KIE-native upstream protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from ..registry import (
    PROTOCOL_CHAT,
    PROTOCOL_MESSAGES,
    PROTOCOL_RESPONSES,
)
from ..routing import Inbound
from . import anthropic as anthropic_t
from . import responses as responses_t
from .sse import SSEEvent, iter_sse

Transcoder = Callable[[AsyncIterator[SSEEvent], str], AsyncIterator[bytes]]

__all__ = [
    "anthropic_t",
    "iter_sse",
    "prepare_upstream_body",
    "render_inbound_body",
    "responses_t",
    "stream_transcoder",
]


def prepare_upstream_body(
    inbound: Inbound,
    protocol: str,
    canonical_chat_body: dict,
    upstream_model_id: str,
) -> dict:
    """Canonical chat body -> upstream-native request body (model already swapped)."""
    body = {**canonical_chat_body, "model": upstream_model_id}
    if protocol == PROTOCOL_CHAT:
        return body
    if protocol == PROTOCOL_RESPONSES:
        return responses_t.request_from_chat(body)
    if protocol == PROTOCOL_MESSAGES:
        return anthropic_t.request_from_chat(body)
    raise ValueError(f"unsupported upstream protocol: {protocol}")


def render_inbound_body(
    inbound: Inbound,
    protocol: str,
    native_body: dict,
    requested_model: str,
) -> dict:
    """Upstream-native response -> inbound-format response body."""
    if inbound == Inbound.OPENAI:
        if protocol == PROTOCOL_CHAT:
            return native_body
        if protocol == PROTOCOL_RESPONSES:
            return responses_t.response_to_chat(native_body, requested_model)
        if protocol == PROTOCOL_MESSAGES:
            return anthropic_t.response_to_chat(native_body, requested_model)
    else:
        if protocol == PROTOCOL_MESSAGES:
            return native_body
        if protocol == PROTOCOL_CHAT:
            chat = native_body
            return anthropic_t.response_from_chat(chat, requested_model)
        if protocol == PROTOCOL_RESPONSES:
            chat = responses_t.response_to_chat(native_body, requested_model)
            return anthropic_t.response_from_chat(chat, requested_model)
    raise ValueError(f"unsupported route: inbound={inbound} protocol={protocol}")


def stream_transcoder(inbound: Inbound, protocol: str) -> Transcoder | None:
    """Return a streaming transcoder, or None for raw passthrough."""
    if inbound == Inbound.OPENAI:
        if protocol == PROTOCOL_MESSAGES:
            return anthropic_t.stream_to_chat
        if protocol == PROTOCOL_RESPONSES:
            return responses_t.stream_to_chat
        return None
    if protocol == PROTOCOL_CHAT:
        return anthropic_t.stream_from_chat
    if protocol == PROTOCOL_RESPONSES:
        return _responses_to_anthropic
    return None


async def _responses_to_anthropic(events: AsyncIterator[SSEEvent], model: str) -> AsyncIterator[bytes]:
    """Responses SSE -> chat SSE -> Anthropic Messages SSE."""
    chat_chunks = responses_t.stream_to_chat(events, model)
    chat_events = iter_sse(chat_chunks)
    async for chunk in anthropic_t.stream_from_chat(chat_events, model):
        yield chunk


def canonicalize_inbound(inbound: Inbound, body: dict) -> dict:
    """Inbound request body -> canonical chat body (identity for OpenAI inbound)."""
    if inbound == Inbound.ANTHROPIC:
        return anthropic_t.request_to_chat(body)
    return body
