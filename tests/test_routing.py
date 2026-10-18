from __future__ import annotations

from kie_proxy.registry import (
    PROTOCOL_CHAT,
    PROTOCOL_GEMINI,
    PROTOCOL_MESSAGES,
    PROTOCOL_RESPONSES,
    ModelInfo,
)
from kie_proxy.routing import Inbound, plan


def _model(protocol: str) -> ModelInfo:
    return ModelInfo(id="m", protocol=protocol, upstream_id="m-up")


def test_openai_inbound():
    assert plan(_model(PROTOCOL_CHAT), Inbound.OPENAI).passthrough
    assert not plan(_model(PROTOCOL_MESSAGES), Inbound.OPENAI).passthrough
    assert not plan(_model(PROTOCOL_RESPONSES), Inbound.OPENAI).passthrough
    assert not plan(_model(PROTOCOL_GEMINI), Inbound.OPENAI).passthrough


def test_anthropic_inbound():
    assert plan(_model(PROTOCOL_MESSAGES), Inbound.ANTHROPIC).passthrough
    assert not plan(_model(PROTOCOL_CHAT), Inbound.ANTHROPIC).passthrough
    assert not plan(_model(PROTOCOL_RESPONSES), Inbound.ANTHROPIC).passthrough


def test_upstream_body_model_prefers_upstream_id():
    route = plan(_model(PROTOCOL_CHAT), Inbound.OPENAI)
    assert route.upstream_body_model == "m-up"
