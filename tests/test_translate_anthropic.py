from __future__ import annotations

import json

from kie_proxy.translate import anthropic as at
from tests.conftest import load_json, sse_events_payloads, sse_from


def test_request_from_chat_maps_system_and_defaults(chat_body):
    body = at.request_from_chat(chat_body)
    assert body["model"] == "claude-haiku-4-5"
    assert body["system"] == "Be terse."
    assert body["max_tokens"] == 100
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "Say only: hi"}]


def test_request_from_chat_default_max_tokens():
    body = at.request_from_chat({"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]})
    assert body["max_tokens"] == 4096


def test_request_from_chat_tools_and_tool_calls():
    body = at.request_from_chat(
        {
            "model": "claude-x",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Boston"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "17C"},
            ],
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
        }
    )
    assistant = body["messages"][1]
    assert assistant["content"][0] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {"city": "Boston"},
    }
    tool_msg = body["messages"][2]
    assert tool_msg["content"][0]["type"] == "tool_result"
    assert tool_msg["content"][0]["tool_use_id"] == "call_1"
    assert tool_msg["content"][0]["content"] == "17C"
    assert body["tools"][0] == {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }


def test_request_from_chat_image_url_data_uri():
    body = at.request_from_chat(
        {
            "model": "claude-x",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
                    ],
                }
            ],
            "max_tokens": 10,
        }
    )
    blocks = body["messages"][0]["content"]
    assert blocks[1]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}


def test_request_to_chat_from_anthropic():
    body = at.request_to_chat(
        {
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "system": "Be terse.",
            "messages": [
                {"role": "user", "content": "Say only: hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "checking"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Boston"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "17C"}
                    ],
                },
            ],
        }
    )
    assert body["messages"][0] == {"role": "system", "content": "Be terse."}
    assert body["max_tokens"] == 100
    assistant = body["messages"][2]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"city": "Boston"}
    assert body["messages"][3]["role"] == "tool"
    assert body["messages"][3]["tool_call_id"] == "toolu_1"


def test_response_to_chat_from_kie_fixture():
    native = load_json("msg_nostream")
    chat = at.response_to_chat(native, "claude-haiku-4-5")
    assert chat["object"] == "chat.completion"
    assert chat["model"] == "claude-haiku-4-5"
    choice = chat["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "hi"}
    assert choice["finish_reason"] == "stop"
    assert chat["usage"] == {"prompt_tokens": 10, "completion_tokens": 39, "total_tokens": 49}
    assert chat["credits_consumed"] == 0.05


def test_response_to_chat_tools_fixture():
    native = load_json("msg_tools")
    chat = at.response_to_chat(native, "claude-haiku-4-5")
    message = chat["choices"][0]["message"]
    assert message["content"] is None
    assert chat["choices"][0]["finish_reason"] == "tool_calls"
    call = message["tool_calls"][0]
    assert call["id"] == "toolu_01LwZJHUB5DjF2B24dkg1489"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Boston"}


def test_response_from_chat_uses_kie_chat_fixture():
    native = load_json("chat_nostream")
    message = at.response_from_chat(native, "gpt-5-2")
    assert message["type"] == "message"
    assert message["role"] == "assistant"
    assert message["content"] == [{"type": "text", "text": "hi"}]
    assert message["stop_reason"] == "end_turn"
    assert message["usage"] == {"input_tokens": 86, "output_tokens": 1}
    assert message["credits_consumed"] == 0.01


async def test_stream_to_chat_text_fixture():
    events = await sse_from(_body("msg_stream"))
    chunks = []
    async for b in at.stream_to_chat(_aiter(events), "claude-haiku-4-5"):
        chunks.append(b)
    payloads = _payloads(chunks)
    text = "".join(
        (c.get("choices") or [{}])[0].get("delta", {}).get("content", "") for c in payloads
    )
    assert text == "hi"
    final = payloads[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["prompt_tokens"] == 10
    assert final["usage"]["completion_tokens"] == 38
    assert final["credits_consumed"] == 0.01
    assert chunks[-1] == b"data: [DONE]\n\n"


async def test_stream_to_chat_tools_fixture():
    events = await sse_from(_body("msg_tools_stream"))
    payloads = _payloads([b async for b in at.stream_to_chat(_aiter(events), "claude-haiku-4-5")])
    tool_deltas = [
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in payloads
        if c.get("choices") and c["choices"][0].get("delta", {}).get("tool_calls")
    ]
    assert tool_deltas[0]["function"]["name"] == "get_weather"
    args = "".join(td["function"].get("arguments", "") for td in tool_deltas)
    assert json.loads(args) == {"city": "Boston"}
    final = [c for c in payloads if c.get("choices") and c["choices"][0].get("finish_reason")][-1]
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    assert final["usage"]["completion_tokens"] == 119


async def test_stream_from_chat_text_fixture():
    events = await sse_from(_body("chat_stream"))
    raw = b"".join([b async for b in at.stream_from_chat(_aiter(events), "gpt-5-2")])
    text = raw.decode()
    assert "event: message_start" in text
    payloads = sse_events_payloads(text)
    kinds = [p["type"] for p in payloads]
    assert "content_block_start" in kinds
    assert "content_block_stop" in kinds
    deltas = [p["delta"] for p in payloads if p["type"] == "content_block_delta"]
    assert deltas == [{"type": "text_delta", "text": "hi"}]
    assert any(p["type"] == "message_stop" for p in payloads)
    stop = next(p for p in payloads if p["type"] == "message_delta")
    assert stop["delta"]["stop_reason"] == "end_turn"
    assert stop["usage"]["output_tokens"] == 1
    assert stop["credits_consumed"] == 0.01
    # input token count is only known from the final upstream chunk, so
    # message_start carries zeros for translated streams
    start = next(p for p in payloads if p["type"] == "message_start")
    assert start["message"]["usage"]["input_tokens"] == 0


def _body(name: str) -> str:
    from tests.conftest import load_body

    return load_body(name)


async def _aiter(events):
    for ev in events:
        yield ev


def _payloads(chunks: list[bytes]) -> list[dict]:
    out = []
    for chunk in chunks:
        for line in chunk.decode().splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[6:]))
    return out

