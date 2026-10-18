from __future__ import annotations

import json

from kie_proxy.translate import responses as rt
from tests.conftest import load_body, load_json, sse_from, sse_json_payloads


def _aiter(events):
    async def gen():
        for ev in events:
            yield ev

    return gen()


def _payloads(chunks: list[bytes]) -> list[dict]:
    out = []
    for chunk in chunks:
        for line in chunk.decode().splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[6:]))
    return out


def test_request_from_chat_basic():
    body = rt.request_from_chat(
        {
            "model": "deepseek-v4-1-flash",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Say only: hi"},
            ],
            "max_tokens": 100,
            "stream": True,
        }
    )
    assert body["model"] == "deepseek-v4-1-flash"
    assert body["instructions"] == "Be terse."
    assert body["stream"] is True
    assert body["max_output_tokens"] == 100
    assert body["input"] == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say only: hi"}]}
    ]


def test_request_from_chat_tools():
    body = rt.request_from_chat(
        {
            "model": "deepseek-v4-1-flash",
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
        }
    )
    assert body["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    kinds = [item["type"] for item in body["input"]]
    assert kinds == ["message", "function_call", "function_call_output"]
    assert body["input"][1]["call_id"] == "call_1"
    assert body["input"][1]["arguments"] == '{"city": "Boston"}'


def test_request_to_chat():
    body = rt.request_to_chat(
        {
            "model": "deepseek-v4-1-flash",
            "instructions": "Be terse.",
            "max_output_tokens": 100,
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "Boston"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "17C"},
            ],
        }
    )
    assert body["messages"][0] == {"role": "system", "content": "Be terse."}
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert body["messages"][2]["tool_calls"][0]["id"] == "call_1"
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "17C"}
    assert body["max_tokens"] == 100


def test_response_to_chat_text():
    chat = rt.response_to_chat(load_json("resp_nostream"), "deepseek-v4-1-flash")
    assert chat["object"] == "chat.completion"
    choice = chat["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "hi"}
    assert choice["finish_reason"] == "stop"
    assert chat["usage"] == {"prompt_tokens": 34, "completion_tokens": 31, "total_tokens": 65}
    assert chat["credits_consumed"] == 0.01


def test_response_to_chat_tools():
    chat = rt.response_to_chat(load_json("resp_tools"), "deepseek-v4-1-flash")
    message = chat["choices"][0]["message"]
    assert message["content"] is None
    assert chat["choices"][0]["finish_reason"] == "tool_calls"
    call = message["tool_calls"][0]
    assert call["id"] == "call_00_r2xlqjnoxh2bec885c9hif4j"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Boston"}


def test_response_from_chat():
    resp = rt.response_from_chat(load_json("chat_nostream"), "gpt-5-2")
    assert resp["object"] == "response"
    assert resp["status"] == "completed"
    assert resp["output"][0]["content"][0]["text"] == "hi"
    assert resp["usage"]["input_tokens"] == 86
    assert resp["usage"]["output_tokens"] == 1
    assert resp["credits_consumed"] == 0.01


async def test_stream_to_chat_text():
    events = await sse_from(load_body("resp_stream"))
    payloads = _payloads([b async for b in rt.stream_to_chat(_aiter(events), "deepseek-v4-1-flash")])
    text = "".join(
        (c.get("choices") or [{}])[0].get("delta", {}).get("content", "") for c in payloads
    )
    assert text == "hi"
    final = [c for c in payloads if c["choices"][0].get("finish_reason")][-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["prompt_tokens"] == 34
    assert final["usage"]["completion_tokens"] == 30


async def test_stream_from_chat_emits_responses_events():
    events = await sse_from(load_body("chat_stream"))
    raw = b"".join([b async for b in rt.stream_from_chat(_aiter(events), "gpt-5-2")]).decode()
    payloads = sse_json_payloads(raw)
    kinds = [p["type"] for p in payloads]
    assert "response.output_item.added" in kinds
    assert "response.output_text.delta" in kinds
    text = "".join(p["delta"] for p in payloads if p["type"] == "response.output_text.delta")
    assert text == "hi"
    completed = next(p for p in payloads if p["type"] == "response.completed")["response"]
    assert completed["status"] == "completed"
    assert completed["usage"]["input_tokens"] == 86
    assert completed["credits_consumed"] == 0.01


async def test_responses_stream_to_anthropic_client():
    from kie_proxy.routing import Inbound
    from kie_proxy.translate import stream_transcoder

    transcoder = stream_transcoder(Inbound.ANTHROPIC, "openai_responses")
    assert transcoder is not None
    events = await sse_from(load_body("resp_stream"))
    raw = b"".join([b async for b in transcoder(_aiter(events), "deepseek-v4-1-flash")]).decode()
    payloads = sse_json_payloads(raw)
    kinds = [p["type"] for p in payloads]
    assert "message_start" in kinds
    assert "content_block_start" in kinds
    deltas = [p["delta"] for p in payloads if p["type"] == "content_block_delta"]
    assert deltas == [{"type": "text_delta", "text": "hi"}]
    assert "message_stop" in kinds
    stop = next(p for p in payloads if p["type"] == "message_delta")
    assert stop["delta"]["stop_reason"] == "end_turn"
    assert stop["usage"]["output_tokens"] == 30
