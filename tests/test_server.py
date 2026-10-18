from __future__ import annotations

import json

import httpx
import pytest

from kie_proxy.config import KeyEntry, Settings
from kie_proxy.registry import (
    PROTOCOL_CHAT,
    PROTOCOL_MESSAGES,
    PROTOCOL_RESPONSES,
    ModelInfo,
)
from kie_proxy.server import create_app
from kie_proxy.upstream import StreamResult
from tests.conftest import load_body, load_json, sse_json_payloads


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes):
        self._data = data

    async def __aiter__(self):
        for i in range(0, len(self._data), 64):
            yield self._data[i : i + 64]


def make_settings(**over) -> Settings:
    settings = Settings(keys=[KeyEntry(key="test-key", label="test")])
    for k, v in over.items():
        setattr(settings, k, v)
    return settings


def make_app(models: dict[str, ModelInfo], **settings_over):
    app = create_app(make_settings(**settings_over))
    registry = app.state.kie.registry
    registry.models = models
    registry.updated_at = 10**12  # not stale
    return app


def catalog() -> dict[str, ModelInfo]:
    return {
        "gpt-5-2": ModelInfo(
            id="gpt-5-2", provider="OpenAI", protocol=PROTOCOL_CHAT, path="/gpt-5-2/v1/chat/completions"
        ),
        "claude-haiku-4-5": ModelInfo(
            id="claude-haiku-4-5",
            provider="Anthropic",
            protocol=PROTOCOL_MESSAGES,
            path="/anthropic/v1/messages",
        ),
        "deepseek-v4-1-flash": ModelInfo(
            id="deepseek-v4-1-flash",
            provider="DeepSeek",
            protocol=PROTOCOL_RESPONSES,
            path="/openai/v1/responses",
            upstream_id="deepseek-v4-1-flash-260910",
        ),
    }


async def request(app, method, url, **kw) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kw)


def install_post_json(app, handler):
    calls = []

    async def fake(path, body):
        calls.append((path, body))
        return handler(path, body)

    app.state.kie.client.post_json = fake
    return calls


# --------------------------------------------------------------------- basics


async def test_healthz():
    app = make_app({})
    resp = await request(app, "GET", "/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_models_openai_and_anthropic_shapes():
    app = make_app(catalog())
    resp = await request(app, "GET", "/v1/models")
    assert resp.json()["object"] == "list"
    resp = await request(app, "GET", "/v1/models", headers={"anthropic-version": "2023-06-01"})
    payload = resp.json()
    assert payload["data"][0]["type"] == "model"


async def test_auth_required_when_token_set():
    app = make_app(catalog(), auth_token="s3cret")
    body = {"model": "gpt-5-2", "messages": [{"role": "user", "content": "hi"}]}
    resp = await request(app, "POST", "/v1/chat/completions", json=body)
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"

    resp = await request(
        app, "POST", "/v1/chat/completions", json=body, headers={"Authorization": "Bearer s3cret"}
    )
    assert resp.status_code != 401

    resp = await request(
        app, "POST", "/v1/chat/completions", json=body, headers={"x-api-key": "Bearer s3cret"}
    )
    assert resp.status_code != 401

    resp = await request(app, "GET", "/healthz")
    assert resp.status_code == 200  # healthz is always open


async def test_unknown_model_openai_shape():
    app = make_app(catalog())
    resp = await request(
        app,
        "POST",
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "invalid_request_error"


async def test_unknown_model_anthropic_shape():
    app = make_app(catalog())
    resp = await request(
        app,
        "POST",
        "/v1/messages",
        json={"model": "nope", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
    assert resp.json()["type"] == "error"


async def test_missing_model_field():
    app = make_app(catalog())
    resp = await request(app, "POST", "/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 400


# ------------------------------------------------------------------ passthru


async def test_chat_passthrough_to_chat_native():
    app = make_app(catalog())
    calls = install_post_json(app, lambda path, body: (load_json("chat_nostream"), 200))
    resp = await request(
        app,
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-5-2", "messages": [{"role": "user", "content": "Say only: hi"}]},
    )
    assert resp.status_code == 200
    path, upstream_body = calls[0]
    assert path == "/gpt-5-2/v1/chat/completions"
    assert upstream_body["model"] == "gpt-5-2"
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "hi"
    assert data["usage"]["prompt_tokens"] == 86


async def test_anthropic_passthrough_to_claude():
    app = make_app(catalog())
    calls = install_post_json(app, lambda path, body: (load_json("msg_nostream"), 200))
    resp = await request(
        app,
        "POST",
        "/v1/messages",
        json={
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say only: hi"}],
        },
    )
    assert resp.status_code == 200
    path, upstream_body = calls[0]
    assert path == "/anthropic/v1/messages"
    assert upstream_body["model"] == "claude-haiku-4-5"
    assert upstream_body["max_tokens"] == 100
    data = resp.json()
    assert data["type"] == "message"
    assert data["content"][0]["text"] == "hi"
    assert data["usage"]["output_tokens"] == 39


# --------------------------------------------------------------- translation


async def test_chat_client_to_claude_model():
    app = make_app(catalog())
    calls = install_post_json(app, lambda path, body: (load_json("msg_tools"), 200))
    resp = await request(
        app,
        "POST",
        "/v1/chat/completions",
        json={
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                }
            ],
            "max_tokens": 50,
        },
    )
    assert resp.status_code == 200
    _, upstream_body = calls[0]
    assert upstream_body["tools"][0]["name"] == "get_weather"
    assert "input_schema" in upstream_body["tools"][0]
    data = resp.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    call = data["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"city": "Boston"}


async def test_anthropic_client_to_chat_model():
    app = make_app(catalog())
    calls = install_post_json(app, lambda path, body: (load_json("chat_nostream"), 200))
    resp = await request(
        app,
        "POST",
        "/v1/messages",
        json={
            "model": "gpt-5-2",
            "max_tokens": 100,
            "system": "Be terse.",
            "messages": [{"role": "user", "content": "Say only: hi"}],
        },
    )
    assert resp.status_code == 200
    _, upstream_body = calls[0]
    assert upstream_body["model"] == "gpt-5-2"
    assert upstream_body["messages"][0] == {"role": "system", "content": "Be terse."}
    data = resp.json()
    assert data["type"] == "message"
    assert data["content"][0]["text"] == "hi"
    assert data["usage"]["input_tokens"] == 86


async def test_chat_client_to_responses_model():
    app = make_app(catalog())
    calls = install_post_json(app, lambda path, body: (load_json("resp_tools"), 200))
    resp = await request(
        app,
        "POST",
        "/v1/chat/completions",
        json={
            "model": "deepseek-v4-1-flash",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                }
            ],
        },
    )
    assert resp.status_code == 200
    path, upstream_body = calls[0]
    assert path == "/openai/v1/responses"
    assert upstream_body["model"] == "deepseek-v4-1-flash-260910"
    assert upstream_body["tools"][0]["type"] == "function"
    data = resp.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert json.loads(data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "city": "Boston"
    }


# ------------------------------------------------------------------- streaming


def install_stream(app, sse_body: str):
    calls = []

    async def fake(path, body):
        calls.append((path, body))
        resp = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=BytesStream(sse_body.encode()),
        )
        key = app.state.kie.pool.acquire()
        return StreamResult(key=key, response=resp)

    app.state.kie.client.post_stream = fake
    return calls


async def test_stream_chat_passthrough():
    app = make_app(catalog())
    calls = install_stream(app, load_body("chat_stream"))
    resp = await request(
        app,
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-5-2", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert resp.text.endswith("data: [DONE]\n\n")
    payloads = sse_json_payloads(resp.text)
    text = "".join((p.get("choices") or [{}])[0].get("delta", {}).get("content", "") for p in payloads)
    assert text == "hi"
    assert calls[0][1]["stream"] is True


async def test_stream_claude_to_chat_client():
    app = make_app(catalog())
    calls = install_stream(app, load_body("msg_stream"))
    resp = await request(
        app,
        "POST",
        "/v1/chat/completions",
        json={
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert calls[0][0] == "/anthropic/v1/messages"
    payloads = sse_json_payloads(resp.text)
    text = "".join((p.get("choices") or [{}])[0].get("delta", {}).get("content", "") for p in payloads)
    assert text == "hi"
    final = [p for p in payloads if p["choices"][0].get("finish_reason")][-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["prompt_tokens"] == 10
    assert resp.text.endswith("data: [DONE]\n\n")


async def test_stream_chat_to_anthropic_client():
    app = make_app(catalog())
    install_stream(app, load_body("chat_stream"))
    resp = await request(
        app,
        "POST",
        "/v1/messages",
        json={
            "model": "gpt-5-2",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    text = resp.text
    assert "event: message_start" in text
    payloads = sse_json_payloads(resp.text)
    deltas = [p for p in payloads if p.get("type") == "content_block_delta"]
    assert deltas[0]["delta"] == {"type": "text_delta", "text": "hi"}
    assert any(p.get("type") == "message_stop" for p in payloads)
    stop = next(p for p in payloads if p.get("type") == "message_delta")
    assert stop["delta"]["stop_reason"] == "end_turn"


async def test_stream_responses_to_chat_client():
    app = make_app(catalog())
    install_stream(app, load_body("resp_stream"))
    resp = await request(
        app,
        "POST",
        "/v1/chat/completions",
        json={
            "model": "deepseek-v4-1-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    payloads = sse_json_payloads(resp.text)
    text = "".join((p.get("choices") or [{}])[0].get("delta", {}).get("content", "") for p in payloads)
    assert text == "hi"
    final = [p for p in payloads if p["choices"][0].get("finish_reason")][-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert resp.text.endswith("data: [DONE]\n\n")


async def test_stream_usage_recorded_in_stats():
    app = make_app(catalog())
    install_stream(app, load_body("chat_stream"))
    await request(
        app,
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-5-2",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    stats = app.state.kie.observer.stats
    assert stats.requests == 1
    assert stats.stream_requests == 1
    assert stats.prompt_tokens == 86
    assert stats.completion_tokens == 1
    assert stats.credits == pytest.approx(0.01)


# ---------------------------------------------------------------------- errors


async def test_upstream_error_rendered_openai_style():
    app = make_app(catalog())

    async def failing(path, body):
        from kie_proxy.errors import UpstreamError

        raise UpstreamError.from_body(200, {"code": 402, "msg": "Insufficient account credits", "data": None})

    app.state.kie.client.post_json = failing
    resp = await request(
        app,
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-5-2", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 402
    assert resp.json()["error"]["type"] == "insufficient_quota"


async def test_upstream_error_rendered_anthropic_style():
    app = make_app(catalog())

    async def failing(path, body):
        from kie_proxy.errors import UpstreamError

        raise UpstreamError.from_body(200, {"code": 402, "msg": "Insufficient account credits", "data": None})

    app.state.kie.client.post_json = failing
    resp = await request(
        app,
        "POST",
        "/v1/messages",
        json={"model": "claude-haiku-4-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 402
    assert resp.json()["type"] == "error"
    assert resp.json()["error"]["type"] == "insufficient_quota"

