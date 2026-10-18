"""OpenAI Chat Completions <-> OpenAI Responses translation (canonical <-> upstream).

Used for models whose KIE-native protocol is the Responses API (gpt-5.5+/gpt-6,
grok-4.x, kimi-k3, deepseek-v4) when the inbound format is Anthropic Messages, and
for Anthropic clients talking to inbound chat-native upstreams in reverse.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .sse import SSEEvent, encode, encode_done

DEFAULT_MAX_TOKENS = 4096


# --------------------------------------------------------------------- request


def request_from_chat(body: dict) -> dict:
    """OpenAI Chat Completions request -> OpenAI Responses request."""
    input_items: list[dict] = []
    instructions: list[str] = []
    for message in body.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "developer"}:
            text = _content_text(content)
            if text:
                instructions.append(text)
            continue
        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or "",
                    "output": _content_text(content),
                }
            )
            continue
        if role == "assistant":
            text = _content_text(content)
            if text:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or "",
                        "name": fn.get("name") or "",
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue
        input_items.append(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": _content_text(content)}],
            }
        )

    out: dict[str, Any] = {
        "model": body.get("model"),
        "input": input_items,
        "stream": bool(body.get("stream")),
    }
    if instructions:
        out["instructions"] = "\n\n".join(instructions)
    if body.get("max_tokens") is not None or body.get("max_completion_tokens") is not None:
        raw_max = body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS
        out["max_output_tokens"] = int(raw_max)
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if stop := body.get("stop"):
        out["stop"] = [stop] if isinstance(stop, str) else list(stop)
    if tools := body.get("tools"):
        out["tools"] = [_tool_from_chat(t) for t in tools if _tool_from_chat(t)]
    if choice := body.get("tool_choice"):
        out["tool_choice"] = _tool_choice_from_chat(choice)
    if body.get("parallel_tool_calls") is not None:
        out["parallel_tool_calls"] = body["parallel_tool_calls"]
    if body.get("reasoning_effort"):
        out["reasoning_effort"] = body["reasoning_effort"]
    if fmt := body.get("response_format"):
        out["text"] = _text_format_from_chat(fmt)
    return out


def request_to_chat(body: dict) -> dict:
    """OpenAI Responses request -> canonical OpenAI Chat Completions request."""
    messages: list[dict] = []
    if instructions := body.get("instructions"):
        messages.append({"role": "system", "content": str(instructions)})

    for item in body.get("input") or []:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        itype = item.get("type") or "message"
        if itype == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or "",
                    "content": str(item.get("output") or ""),
                }
            )
            continue
        if itype == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or "",
                            "type": "function",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": item.get("arguments") or "{}",
                            },
                        }
                    ],
                }
            )
            continue
        if itype != "message":
            continue
        role = item.get("role") or "user"
        text = _responses_content_text(item.get("content"))
        if role in {"system", "developer"}:
            messages.append({"role": "system", "content": text})
        else:
            messages.append({"role": role, "content": text})

    out: dict[str, Any] = {"model": body.get("model"), "messages": messages}
    if body.get("max_output_tokens") is not None:
        out["max_tokens"] = body["max_output_tokens"]
    if body.get("stream"):
        out["stream"] = True
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if stop := body.get("stop"):
        out["stop"] = stop[0] if isinstance(stop, list) and len(stop) == 1 else stop
    if tools := body.get("tools"):
        out["tools"] = [_tool_to_chat(t) for t in tools if _tool_to_chat(t)]
    if choice := body.get("tool_choice"):
        out["tool_choice"] = _tool_choice_to_chat(choice)
    if body.get("reasoning_effort"):
        out["reasoning_effort"] = body["reasoning_effort"]
    return out


# ------------------------------------------------------------------- response


def response_to_chat(response: dict, requested_model: str) -> dict:
    """OpenAI Responses response -> OpenAI Chat Completions response."""
    content, tool_calls = _output_to_chat(response.get("output") or [])
    finish = "tool_calls" if tool_calls else "stop"
    if response.get("status") == "incomplete":
        finish = "length"
    usage = response.get("usage") or {}
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    out: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(response.get("created_at") or time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }
    if tool_calls:
        out["choices"][0]["message"]["tool_calls"] = tool_calls
    if isinstance(response.get("credits_consumed"), (int, float)):
        out["credits_consumed"] = response["credits_consumed"]
    return out


def response_from_chat(completion: dict, requested_model: str) -> dict:
    """OpenAI Chat Completions response -> OpenAI Responses response."""
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict] = []
    if text := message.get("content"):
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": call.get("id") or "",
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "{}",
            }
        )
    usage = completion.get("usage") or {}
    finish = choice.get("finish_reason")
    out: dict[str, Any] = {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(completion.get("created") or time.time()),
        "status": "completed" if finish != "length" else "incomplete",
        "model": requested_model,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "input_tokens_details": {"cached_tokens": 0},
        },
    }
    if isinstance(completion.get("credits_consumed"), (int, float)):
        out["credits_consumed"] = completion["credits_consumed"]
    return out


# ------------------------------------------------------------------ streaming


async def stream_to_chat(events: AsyncIterator[SSEEvent], requested_model: str) -> AsyncIterator[bytes]:
    """OpenAI Responses SSE -> OpenAI Chat Completions SSE."""
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    started = False
    finish_reason: str | None = None
    saw_tool = False
    prompt = completion = 0
    credits: float | None = None

    def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> bytes:
        payload: dict[str, Any] = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage:
            payload["usage"] = usage
        if credits is not None:
            payload["credits_consumed"] = credits
        return encode(None, payload)

    async for event in events:
        if event.done:
            break
        data = event.json
        if not isinstance(data, dict):
            continue
        etype = data.get("type") or event.event or ""
        if etype == "response.output_text.delta":
            if not started:
                started = True
                yield chunk({"role": "assistant", "content": ""})
            yield chunk({"content": data.get("delta") or ""})
        elif etype == "response.output_item.added":
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                saw_tool = True
                if not started:
                    started = True
                    yield chunk({"role": "assistant", "content": ""})
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": data.get("output_index", 0),
                                "id": item.get("call_id") or "",
                                "type": "function",
                                "function": {"name": item.get("name") or "", "arguments": ""},
                            }
                        ]
                    }
                )
        elif etype == "response.function_call_arguments.delta":
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": data.get("output_index", 0),
                            "function": {"arguments": data.get("delta") or ""},
                        }
                    ]
                }
            )
        elif etype == "response.completed":
            response = data.get("response") or {}
            usage = response.get("usage") or {}
            prompt = int(usage.get("input_tokens") or 0)
            completion = int(usage.get("output_tokens") or 0)
            if isinstance(response.get("credits_consumed"), (int, float)):
                credits = float(response["credits_consumed"])
            if response.get("status") == "incomplete":
                finish_reason = "length"
            elif saw_tool:
                finish_reason = "tool_calls"
            else:
                finish_reason = "stop"
        elif etype in {"response.failed", "error"}:
            err = (data.get("response") or {}).get("error") or data.get("error") or data
            message = err.get("message") if isinstance(err, dict) else str(err)
            yield encode(
                None, {"error": {"message": message or "upstream stream error", "type": "api_error"}}
            )
            yield encode_done()
            return

    if not started:
        started = True
        yield chunk({"role": "assistant", "content": ""})
    yield chunk(
        {},
        finish=finish_reason or ("tool_calls" if saw_tool else "stop"),
        usage={
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    )
    yield encode_done()


async def stream_from_chat(
    events: AsyncIterator[SSEEvent], requested_model: str
) -> AsyncIterator[bytes]:
    """OpenAI Chat Completions SSE -> OpenAI Responses SSE."""
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    text_opened = False
    tool_opened: set[int] = set()
    prompt = completion = 0
    credits: float | None = None
    finish_reason: str | None = None
    response_snapshot: dict[str, Any] = {}

    def base(payload: dict) -> bytes:
        return encode(payload.get("type"), payload)

    async for event in events:
        if event.done:
            break
        data = event.json
        if not isinstance(data, dict):
            continue
        if data.get("error"):
            err = data["error"] if isinstance(data["error"], dict) else {"message": str(data["error"])}
            yield base(
                {"type": "error", "code": "upstream_error", "message": err.get("message") or "stream error"}
            )
            return
        if data.get("object") != "chat.completion.chunk":
            continue
        usage = data.get("usage")
        if isinstance(usage, dict):
            if usage.get("prompt_tokens"):
                prompt = int(usage["prompt_tokens"])
            if usage.get("completion_tokens"):
                completion = int(usage["completion_tokens"])
        if isinstance(data.get("credits_consumed"), (int, float)):
            credits = float(data["credits_consumed"])
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            if text := delta.get("content"):
                if not text_opened:
                    text_opened = True
                    yield base(
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "type": "message",
                                "id": f"msg_{uuid.uuid4().hex[:24]}",
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                        }
                    )
                    yield base(
                        {
                            "type": "response.content_part.added",
                            "item_id": "",
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": ""},
                        }
                    )
                yield base(
                    {
                        "type": "response.output_text.delta",
                        "item_id": "",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text,
                    }
                )
            for call in delta.get("tool_calls") or []:
                cidx = int(call.get("index") or 0)
                fn = call.get("function") or {}
                if cidx not in tool_opened:
                    tool_opened.add(cidx)
                    yield base(
                        {
                            "type": "response.output_item.added",
                            "output_index": cidx + 1,
                            "item": {
                                "type": "function_call",
                                "id": f"fc_{uuid.uuid4().hex[:24]}",
                                "call_id": call.get("id") or "",
                                "name": fn.get("name") or "",
                                "arguments": "",
                                "status": "in_progress",
                            },
                        }
                    )
                if args := fn.get("arguments"):
                    yield base(
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": "",
                            "output_index": cidx + 1,
                            "delta": args,
                        }
                    )
            if finish := choice.get("finish_reason"):
                finish_reason = finish

    output: list[dict] = []
    if text_opened:
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
        )
    for i in sorted(tool_opened):
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": "",
                "name": "",
                "arguments": "",
                "status": "completed",
                "output_index": i + 1,
            }
        )
    response_snapshot = {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": "completed" if finish_reason != "length" else "incomplete",
        "model": requested_model,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
            "output_tokens_details": {"reasoning_tokens": 0},
            "input_tokens_details": {"cached_tokens": 0},
        },
    }
    if credits is not None:
        response_snapshot["credits_consumed"] = credits
    if text_opened:
        yield base(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": response_snapshot["output"][0],
            }
        )
    yield base({"type": "response.completed", "response": response_snapshot})


# --------------------------------------------------------------------- helpers


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(str(p) for p in parts if p is not None)
    return str(content)


def _responses_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text")
            for p in content
            if isinstance(p, dict)
            and p.get("type") in {"input_text", "output_text", "text"}
            and p.get("text")
        ]
        return "\n".join(str(p) for p in parts if p is not None)
    return str(content)


def _tool_from_chat(tool: dict) -> dict | None:
    if tool.get("type") not in (None, "function"):
        return None
    fn = tool.get("function") or tool
    name = fn.get("name")
    if not name:
        return None
    out: dict[str, Any] = {"type": "function", "name": name, "description": fn.get("description") or ""}
    out["parameters"] = fn.get("parameters") or {"type": "object", "properties": {}}
    return out


def _tool_to_chat(tool: dict) -> dict | None:
    if tool.get("type") not in (None, "function"):
        return None
    name = tool.get("name") or (tool.get("function") or {}).get("name")
    if not name:
        return None
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description") or "",
            "parameters": tool.get("parameters") or {"type": "object"},
        },
    }


def _tool_choice_from_chat(choice: Any) -> Any:
    if choice is None:
        return None
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name")
        if name:
            return {"type": "function", "name": name}
    return "auto"


def _tool_choice_to_chat(choice: Any) -> Any:
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        name = choice.get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return "auto"


def _text_format_from_chat(fmt: dict) -> dict | None:
    ftype = fmt.get("type")
    if ftype == "json_object":
        return {"format": {"type": "json_object"}}
    if ftype == "json_schema":
        schema = fmt.get("json_schema") or {}
        return {
            "format": {
                "type": "json_schema",
                "name": schema.get("name") or "response",
                "schema": schema.get("schema") or {},
                **({"strict": schema["strict"]} if "strict" in schema else {}),
            }
        }
    return None


def _output_to_chat(output: list[Any]) -> tuple[str | None, list[dict] | None]:
    texts: list[str] = []
    tool_calls: list[dict] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            texts.append(_responses_content_text(item.get("content")))
        elif item.get("type") == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": item.get("name") or "",
                        "arguments": item.get("arguments") or "{}",
                    },
                }
            )
    content = "\n".join(t for t in texts if t) or None
    return content, tool_calls or None
