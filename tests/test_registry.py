from __future__ import annotations

from kie_proxy.registry import (
    DOC_PATHS,
    PROTOCOL_CHAT,
    PROTOCOL_GEMINI,
    PROTOCOL_MESSAGES,
    PROTOCOL_RESPONSES,
    ModelInfo,
    Registry,
    _dotted,
    derive_protocol,
)


def test_dotted():
    assert _dotted("gpt-5-6-sol") == "gpt-5.6-sol"
    assert _dotted("gpt-5-2") == "gpt-5.2"
    assert _dotted("grok-4-7") == "grok-4.7"


def test_derive_protocol_from_listings():
    assert derive_protocol("claude-haiku-4-5", "Anthropic", True, False, False) == (
        PROTOCOL_MESSAGES,
        "/anthropic/v1/messages",
    )
    assert derive_protocol("grok-4-7", "Grok", False, True, False) == (
        PROTOCOL_RESPONSES,
        "/grok/v1/responses",
    )
    assert derive_protocol("gpt-6-luna", "OpenAI", False, False, True) == (
        PROTOCOL_RESPONSES,
        "/openai/v1/responses",
    )


def test_derive_protocol_doc_paths_win():
    assert derive_protocol("gpt-5-2", "OpenAI", False, False, False) == DOC_PATHS["gpt-5-2"]
    assert derive_protocol("kimi-k3", "Other", False, False, False) == DOC_PATHS["kimi-k3"]


def test_derive_protocol_heuristics():
    assert derive_protocol("gemini-3-5-flash-openai", "Google", False, False, False)[0] == PROTOCOL_CHAT
    assert derive_protocol("some-model", "Google", False, False, False)[0] == PROTOCOL_GEMINI
    assert derive_protocol("mystery", "Other", False, False, False)[0] == PROTOCOL_RESPONSES
    assert derive_protocol("mystery2", "Mistral", False, False, False)[0] == PROTOCOL_CHAT


def test_gemini_twin_rerouting():
    registry = Registry()
    registry.models = {
        "gemini-3-5-flash": ModelInfo(id="gemini-3-5-flash", protocol=PROTOCOL_GEMINI, path="/gemini/…"),
        "gemini-3-5-flash-openai": ModelInfo(
            id="gemini-3-5-flash-openai",
            protocol=PROTOCOL_CHAT,
            path="/gemini-3-5-flash-openai/v1/chat/completions",
            upstream_id="gemini-3-5-flash-openai",
        ),
    }
    registry._apply_gemini_twins(registry.models)
    native = registry.models["gemini-3-5-flash"]
    assert native.protocol == PROTOCOL_CHAT
    assert native.path == "/gemini-3-5-flash-openai/v1/chat/completions"
    assert native.upstream_id == "gemini-3-5-flash-openai"
    assert registry.get("gemini-3-5-flash") is native


def test_get_finds_aliases_and_upstream_ids():
    registry = Registry()
    info = ModelInfo(id="gpt-5-6-luna", upstream_id="gpt-5.6-luna", aliases=["gpt-5-6-luna-old"])
    registry.models = {"gpt-5-6-luna": info}
    assert registry.get("gpt-5-6-luna") is info
    assert registry.get("gpt-5.6-luna") is info
    assert registry.get("gpt-5-6-luna-old") is info
    assert registry.get("nope") is None


def test_payloads_shape():
    registry = Registry()
    registry.models = {
        "claude-haiku-4-5": ModelInfo(
            id="claude-haiku-4-5", provider="Anthropic", title="Claude Haiku", protocol=PROTOCOL_MESSAGES
        ),
        "gpt-5-2": ModelInfo(id="gpt-5-2", provider="OpenAI", protocol=PROTOCOL_CHAT, context_window=400000),
    }
    openai = registry.openai_payload()
    assert openai["object"] == "list"
    ids = {m["id"] for m in openai["data"]}
    assert ids == {"claude-haiku-4-5", "gpt-5-2"}
    anthropic = registry.anthropic_payload()
    kinds = {m["type"] for m in anthropic["data"]}
    assert kinds == {"model"}
