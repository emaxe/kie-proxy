"""Merged model registry: every text model KIE serves, with its native protocol.

Sources (all polled at refresh time):

* ``GET /api/v1/models?taskType=Chat``  - the wide catalog (ids, providers, pricing)
* ``GET /openai/v1/models``             - OpenAI Responses family (rich metadata)
* ``GET /anthropic/v1/models``          - Claude family (Anthropic Messages)
* ``GET /xai/v1/models``                - Grok family (Responses)

The catalog does not say which wire protocol a model is served over, so the
protocol is derived from listings first and from the per-model documentation
paths (``DOC_PATHS``, taken from docs.kie.ai model pages) second.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .errors import UpstreamError
from .upstream import KieClient

CATALOG_PATH = "/api/v1/models"
OPENAI_LIST_PATH = "/openai/v1/models"
ANTHROPIC_LIST_PATH = "/anthropic/v1/models"
XAI_LIST_PATH = "/xai/v1/models"

PROTOCOL_CHAT = "openai_chat"
PROTOCOL_MESSAGES = "anthropic_messages"
PROTOCOL_RESPONSES = "openai_responses"
PROTOCOL_GEMINI = "gemini"

# Canonical per-model paths from the OpenAPI specs on docs.kie.ai model pages.
# Only exceptions to the listing/provider heuristics live here.
DOC_PATHS: dict[str, tuple[str, str]] = {
    "gpt-5-2": (PROTOCOL_CHAT, "/gpt-5-2/v1/chat/completions"),
    "gpt-codex": (PROTOCOL_RESPONSES, "/codex/v1/responses"),
    "gemini-2.5-pro": (PROTOCOL_CHAT, "/gemini-2.5-pro/v1/chat/completions"),
    "gemini-3-pro": (PROTOCOL_CHAT, "/gemini-3-pro/v1/chat/completions"),
    "gemini-3.1-pro": (PROTOCOL_CHAT, "/gemini-3.1-pro/v1/chat/completions"),
    "kimi-k3": (PROTOCOL_RESPONSES, "/openai/v1/responses"),
    "deepseek-v4-1-flash": (PROTOCOL_RESPONSES, "/openai/v1/responses"),
}


@dataclass
class ModelInfo:
    id: str
    provider: str = ""
    title: str = ""
    description: str = ""
    pricing: str = ""
    protocol: str = PROTOCOL_CHAT
    path: str = "/v1/chat/completions"
    upstream_id: str = ""
    context_window: int | None = None
    max_tokens: int | None = None
    max_input_tokens: int | None = None
    aliases: list[str] = field(default_factory=list)

    @property
    def native_family(self) -> str:
        return {
            PROTOCOL_CHAT: "openai",
            PROTOCOL_MESSAGES: "anthropic",
            PROTOCOL_RESPONSES: "responses",
            PROTOCOL_GEMINI: "gemini",
        }.get(self.protocol, self.protocol)

    @property
    def owned_by(self) -> str:
        return (self.provider or "kie").lower().replace(" ", "-")


def derive_protocol(
    slug: str,
    provider: str,
    in_anthropic: bool,
    in_xai: bool,
    in_openai: bool,
) -> tuple[str, str]:
    """Pick the wire protocol and upstream path for a catalog slug."""
    if in_anthropic:
        return PROTOCOL_MESSAGES, "/anthropic/v1/messages"
    if slug in DOC_PATHS:
        return DOC_PATHS[slug]
    if in_xai:
        return PROTOCOL_RESPONSES, "/grok/v1/responses"
    if in_openai:
        return PROTOCOL_RESPONSES, "/openai/v1/responses"
    if slug.endswith("-openai"):
        return PROTOCOL_CHAT, f"/{slug}/v1/chat/completions"
    if slug.startswith("claude-"):
        return PROTOCOL_MESSAGES, "/anthropic/v1/messages"
    if slug.startswith("grok-"):
        return PROTOCOL_RESPONSES, "/grok/v1/responses"
    if provider == "Google":
        return PROTOCOL_GEMINI, f"/gemini/v1/models/{slug}:generateContent"
    if provider in {"OpenAI", "Other"}:
        return PROTOCOL_RESPONSES, "/openai/v1/responses"
    return PROTOCOL_CHAT, "/v1/chat/completions"


class Registry:
    def __init__(self, refresh_seconds: int = 600):
        self.refresh_seconds = refresh_seconds
        self.models: dict[str, ModelInfo] = {}
        self.updated_at: float = 0.0
        self.last_error: str = ""

    # ------------------------------------------------------------------ refresh

    async def refresh(self, client: KieClient) -> None:
        errors: list[str] = []
        catalog = await self._try(client, f"{CATALOG_PATH}?taskType=Chat", errors)
        openai_models = await self._try(client, OPENAI_LIST_PATH, errors)
        anthropic_models = await self._try(client, ANTHROPIC_LIST_PATH, errors)
        xai_models = await self._try(client, XAI_LIST_PATH, errors)

        openai_slugs = _openai_slugs(openai_models)
        anthropic_ids = {m["id"] for m in _field_list(anthropic_models, "data") if "id" in m}
        xai_ids = {m["id"] for m in _field_list(xai_models, "data") if "id" in m}

        merged: dict[str, ModelInfo] = {}

        entries = _catalog_entries(catalog)
        for entry in entries:
            slug = str(entry.get("slug") or entry.get("model") or "")
            if not slug:
                continue
            provider = str(entry.get("provider") or "")
            protocol, path = derive_protocol(
                slug, provider, slug in anthropic_ids, slug in xai_ids, slug in openai_slugs
            )
            merged[slug] = ModelInfo(
                id=slug,
                provider=provider,
                title=str(entry.get("title") or slug),
                description=str(entry.get("description") or ""),
                pricing=str(entry.get("pricingDesc") or ""),
                protocol=protocol,
                path=path,
                upstream_id=_upstream_id(slug, openai_slugs, protocol),
            )

        # Models present in protocol listings but missing from the catalog.
        for slug in sorted(openai_slugs | xai_ids | anthropic_ids):
            if slug in merged:
                merged[slug].aliases.append(slug)
                continue
            provider = "Anthropic" if slug in anthropic_ids else ("Grok" if slug in xai_ids else "OpenAI")
            protocol, path = derive_protocol(
                slug, provider, slug in anthropic_ids, slug in xai_ids, slug in openai_slugs
            )
            merged[slug] = ModelInfo(
                id=slug,
                provider=provider,
                title=slug,
                protocol=protocol,
                path=path,
                upstream_id=slug,
            )

        self._enrich_openai(merged, openai_models)
        self._enrich_anthropic(merged, anthropic_models)
        self._apply_gemini_twins(merged)
        self.models = merged
        self.updated_at = time.time()
        self.last_error = "; ".join(errors)

    async def _try(self, client: KieClient, path: str, errors: list[str]) -> Any:
        try:
            return await client.get_json(path)
        except UpstreamError as exc:
            errors.append(f"{path}: {exc.message}")
            return None

    def _enrich_openai(self, merged: dict[str, ModelInfo], payload: Any) -> None:
        rich = {m.get("slug"): m for m in _field_list(payload, "models") if isinstance(m, dict)}
        plain = {m.get("id"): m for m in _field_list(payload, "data") if isinstance(m, dict)}
        for slug, meta in rich.items():
            info = _find(merged, slug)
            if info is None:
                continue
            info.context_window = _int(meta.get("context_window")) or info.context_window
            if info.upstream_id == info.id:
                info.upstream_id = slug
        for slug in plain:
            if (info := _find(merged, slug)) is not None and not info.title:
                info.title = slug

    def _enrich_anthropic(self, merged: dict[str, ModelInfo], payload: Any) -> None:
        for meta in _field_list(payload, "data"):
            if not isinstance(meta, dict) or "id" not in meta:
                continue
            info = _find(merged, meta["id"]) or merged.get(str(meta["id"]))
            if info is None:
                continue
            info.max_input_tokens = _int(meta.get("max_input_tokens")) or info.max_input_tokens
            info.max_tokens = _int(meta.get("max_tokens")) or info.max_tokens
            if display := meta.get("display_name"):
                info.title = str(display)

    def _apply_gemini_twins(self, merged: dict[str, ModelInfo]) -> None:
        """Gemini native models are reachable only through the Gemini protocol;
        every one of them ships an ``-openai`` twin served over chat completions.
        Point the native id at the twin so chat/messages clients work."""
        for info in list(merged.values()):
            if info.protocol != PROTOCOL_GEMINI:
                continue
            twin = merged.get(f"{info.id}-openai")
            if twin is None:
                continue
            info.aliases.append(info.id)
            info.aliases.extend(twin.aliases)
            info.protocol = twin.protocol
            info.path = twin.path
            info.upstream_id = twin.upstream_id
            info.description = (info.description + " " if info.description else "") + (
                f"routed via {twin.id}"
            )

    # ------------------------------------------------------------------ lookup

    def get(self, model_id: str) -> ModelInfo | None:
        if info := self.models.get(model_id):
            return info
        for info in self.models.values():
            if model_id in info.aliases:
                return info
            if info.upstream_id and model_id == info.upstream_id:
                return info
        return None

    def list(self) -> list[ModelInfo]:
        return sorted(self.models.values(), key=lambda m: (m.provider, m.id))

    @property
    def stale(self) -> bool:
        return not self.models or (time.time() - self.updated_at) > self.refresh_seconds

    def openai_payload(self) -> dict:
        data = [
            {
                "id": m.id,
                "object": "model",
                "created": 0,
                "owned_by": m.owned_by,
                "provider": m.provider,
                "title": m.title,
                "native": m.native_family,
                "context_window": m.context_window,
            }
            for m in self.list()
        ]
        return {"object": "list", "data": data}

    def anthropic_payload(self) -> dict:
        data = [
            {
                "type": "model",
                "id": m.id,
                "display_name": m.title or m.id,
                "created_at": "1970-01-01T00:00:00Z",
                "max_input_tokens": m.max_input_tokens,
                "max_tokens": m.max_tokens,
                "provider": m.provider,
                "native": m.native_family,
            }
            for m in self.list()
        ]
        return {"data": data, "has_more": False, "first_id": None, "last_id": None}


# --------------------------------------------------------------------- helpers


def _field_list(payload: Any, key: str) -> list[dict]:
    if isinstance(payload, dict):
        value = payload.get(key)
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
    return []


def _catalog_entries(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            return _field_list(data, "models")
    return _field_list(payload, "models")


def _openai_slugs(payload: Any) -> set[str]:
    slugs = {str(m["slug"]) for m in _field_list(payload, "models") if m.get("slug")}
    slugs |= {str(m["id"]) for m in _field_list(payload, "data") if m.get("id")}
    return slugs


def _dotted(slug: str) -> str:
    """``gpt-5-6-sol`` -> ``gpt-5.6-sol`` (KIE listings dot the version group)."""
    i = slug.find("-")
    j = slug.find("-", i + 1) if i >= 0 else -1
    if i < 0 or j < 0:
        return slug
    return f"{slug[:j]}.{slug[j + 1:]}"


def _find(models: dict[str, ModelInfo], slug: str) -> ModelInfo | None:
    if slug in models:
        return models[slug]
    dotted = _dotted(slug)
    if dotted != slug and dotted in models:
        return models[dotted]
    for key, info in models.items():
        if _dotted(key) == dotted:
            return info
    return None


def _upstream_id(slug: str, openai_slugs: set[str], protocol: str) -> str:
    if protocol == PROTOCOL_RESPONSES and openai_slugs:
        if slug in openai_slugs:
            return slug
        dotted = _dotted(slug)
        if dotted in openai_slugs:
            return dotted
    return slug


def _int(value: Any) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
