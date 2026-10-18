"""Routing: which upstream path a request takes and whether it needs translation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .registry import (
    PROTOCOL_CHAT,
    PROTOCOL_GEMINI,
    PROTOCOL_MESSAGES,
    PROTOCOL_RESPONSES,
    ModelInfo,
)


class Inbound(StrEnum):
    OPENAI = "openai"          # POST /v1/chat/completions
    ANTHROPIC = "anthropic"    # POST /v1/messages


# Which inbound formats can travel to an upstream unmodified.
_PASSTHROUGH_PROTOCOLS = {
    Inbound.OPENAI: PROTOCOL_CHAT,
    Inbound.ANTHROPIC: PROTOCOL_MESSAGES,
}


@dataclass
class RoutePlan:
    model: ModelInfo
    inbound: Inbound
    passthrough: bool

    @property
    def upstream_path(self) -> str:
        return self.model.path

    @property
    def upstream_body_model(self) -> str:
        return self.model.upstream_id or self.model.id


def plan(model: ModelInfo, inbound: Inbound) -> RoutePlan:
    native = _PASSTHROUGH_PROTOCOLS[inbound]
    return RoutePlan(model=model, inbound=inbound, passthrough=model.protocol == native)


def supported_protocols() -> set[str]:
    return {PROTOCOL_CHAT, PROTOCOL_MESSAGES, PROTOCOL_RESPONSES, PROTOCOL_GEMINI}
