"""OpenAI Chat Completions <-> Anthropic Messages translation (canonical <-> inbound).

The canonical wire format inside the proxy is OpenAI Chat Completions. This module
converts Anthropic Messages requests into it and renders results back into the
Anthropic shape, for both buffered and streaming responses.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ..errors import UpstreamError
from .sse import SSEEvent, encode, encode_done

DEFAULT_MAX_TOKENS = 4096


# --------------------------------------------------------------------- request


def request_from_chat(body: dict) -> dict:
    """OpenAI Chat Completions request -> Anthropic Messages request."""
    system_parts: list[str] = []
    messages: list[dict] = []
    for message in body.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "developer"}:
            system_parts.append(_content_text(content))
            continue
        if role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id") or "",
                            "content": _content_text(content),
                        }
                    ],
                }
            )
            continue
        if role == "assistant":
            blocks: list[dict] = []
            text = _content_text(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for part in _content_parts(content):
                if part.get("type") == "image_url":
                    blocks.append(_image_block(part))
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": fn.get("name") or "",
                        "input": _loads(fn.get("arguments")),
                    }
                )
            messages.append({"role": "assistant", "content": blocks or text})
            continue
        parts = _content_parts(content)
        if parts:
            blocks = []
            for part in parts:
                if part.get("type") == "text":
                    blocks.append({"type": "text", "text": part.get("text") or ""})
                elif part.get("type") == "image_url":
                    blocks.append(_image_block(part))
            messages.append({"role": "user", "content": blocks})
        else:
            messages.append({"role": "user", "content": _content_text(content)})

    out: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "max_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS),
        "stream": bool(body.get("stream")),
    }
    if system_parts:
        out["system"] = "\n\n".join(system_parts)
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if stop := body.get("stop"):
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if tools := body.get("tools"):
        out["tools"] = [_tool_from_chat(t) for t in tools if _tool_from_chat(t)]
    if choice := body.get("tool_choice"):
        out["tool_choice"] = _tool_choice_from_chat(choice)
    if "thinking" in body:
        out["thinkingFlag"] = bool(body["thinking"])
    if "thinkingFlag" in body:
        out["thinkingFlag"] = body["thinkingFlag"]
    if body.get("stream_options"):
        out["stream_options"] = body["stream_options"]
    return out


def request_to_chat(body: dict) -> dict:
    """Anthropic Messages request -> canonical OpenAI Chat Completions request."""
    messages: list[dict] = []
    if system := body.get("system"):
        text = system if isinstance(system, str) else _anthropic_text(system)
        if text:
            messages.append({"role": "system", "content": text})
    for message in body.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        blocks = content if isinstance(content, list) else []
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for block in blocks:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text") or ""))
            elif btype == "image":
                text_parts.append(_image_url_from_anthropic(block))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") or "",
                            "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                        },
                    }
                )
            elif btype == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") or "",
                        "content": _anthropic_text(block.get("content")),
                    }
                )
        if tool_results:
            messages.extend(tool_results)
            continue
        entry: dict[str, Any] = {"role": role, "content": "\n".join(text_parts)}
        if tool_calls:
            entry["tool_calls"] = tool_calls
            entry["content"] = entry["content"] or None
        messages.append(entry)

    out: dict[str, Any] = {"model": body.get("model"), "messages": messages}
    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    if body.get("stream"):
        out["stream"] = True
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if stop := body.get("stop_sequences"):
        out["stop"] = stop[0] if len(stop) == 1 else list(stop)
    if tools := body.get("tools"):
        out["tools"] = [_tool_to_chat(t) for t in tools]
    if choice := body.get("tool_choice"):
        out["tool_choice"] = _tool_choice_to_chat(choice)
    if usage := body.get("usage"):  # anthropic clients sometimes ask for usage accounting
        out["usage"] = usage
    return out


# ------------------------------------------------------------------- response


def response_to_chat(message: dict, requested_model: str) -> dict:
    """Anthropic Messages response -> OpenAI Chat Completions response."""
    content, tool_calls = _blocks_to_chat_content(message.get("content") or [])
    finish = _stop_to_finish(message.get("stop_reason"))
    usage = message.get("usage") or {}
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
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
        response["choices"][0]["message"]["tool_calls"] = tool_calls
    if isinstance(message.get("credits_consumed"), (int, float)):
        response["credits_consumed"] = message["credits_consumed"]
    return response


def response_from_chat(completion: dict, requested_model: str) -> dict:
    """OpenAI Chat Completions response -> Anthropic Messages response."""
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    blocks: list[dict] = []
    if text := message.get("content"):
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name") or "",
                "input": _loads(fn.get("arguments")),
            }
        )
    usage = completion.get("usage") or {}
    out: dict[str, Any] = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": blocks,
        "stop_reason": _finish_to_stop(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }
    if isinstance(completion.get("credits_consumed"), (int, float)):
        out["credits_consumed"] = completion["credits_consumed"]
    return out


# ------------------------------------------------------------------ streaming


async def stream_to_chat(events: AsyncIterator[SSEEvent], requested_model: str) -> AsyncIterator[bytes]:
    """Anthropic Messages SSE -> OpenAI Chat Completions SSE."""
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    started = False
    finish_reason: str | None = None
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
        if etype == "message_start":
            usage = (data.get("message") or {}).get("usage") or {}
            prompt = int(usage.get("input_tokens") or 0)
            if not started:
                started = True
                yield chunk({"role": "assistant", "content": ""})
        elif etype == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": data.get("index", 0),
                                "id": block.get("id") or "",
                                "type": "function",
                                "function": {"name": block.get("name") or "", "arguments": ""},
                            }
                        ]
                    }
                )
        elif etype == "content_block_delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                if not started:
                    started = True
                    yield chunk({"role": "assistant"})
                yield chunk({"content": delta.get("text") or ""})
            elif delta.get("type") == "input_json_delta":
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": data.get("index", 0),
                                "function": {"arguments": delta.get("partial_json") or ""},
                            }
                        ]
                    }
                )
        elif etype == "message_delta":
            usage = data.get("usage") or {}
            if usage.get("output_tokens"):
                completion = int(usage["output_tokens"])
            if stop := (data.get("delta") or {}).get("stop_reason"):
                finish_reason = _stop_to_finish(stop)
            if isinstance(data.get("credits_consumed"), (int, float)):
                credits = float(data["credits_consumed"])
        elif etype == "message_stop":
            if not started:
                started = True
                yield chunk({"role": "assistant", "content": ""})
            yield chunk({}, finish=finish_reason or "stop", usage={
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            })
            yield encode_done()
            return
        elif etype == "error":
            err = data.get("error") or data
            payload = {
                "error": {
                    "message": err.get("message") or "upstream stream error",
                    "type": "api_error",
                    "code": None,
                }
            }
            yield encode(None, payload)
            yield encode_done()
            return

    if started:
        yield chunk({}, finish=finish_reason or "stop", usage={
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        })
    yield encode_done()


async def stream_from_chat(
    events: AsyncIterator[SSEEvent], requested_model: str
) -> AsyncIterator[bytes]:
    """OpenAI Chat Completions SSE -> Anthropic Messages SSE."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    started = False
    next_index = 0
    text_index: int | None = None
    tool_index: dict[int, int] = {}
    open_index: int | None = None
    prompt = completion = 0
    credits: float | None = None
    stop_reason: str | None = None

    def start_message(usage: dict | None = None) -> bytes:
        payload: dict[str, Any] = {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": requested_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage or {"input_tokens": prompt, "output_tokens": 0},
            },
        }
        if credits is not None:
            payload["message"]["credits_consumed"] = credits
        return encode("message_start", payload)

    async for event in events:
        if event.done:
            break
        data = event.json
        if not isinstance(data, dict):
            continue
        if data.get("error"):
            err = data["error"] if isinstance(data["error"], dict) else {"message": str(data["error"])}
            yield encode(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": err.get("message") or "stream error"},
                },
            )
            yield encode("message_stop", {"type": "message_stop"})
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
            if not started:
                started = True
                yield start_message()
            if text := delta.get("content"):
                if text_index is None:
                    if open_index is not None:
                        yield encode(
                            "content_block_stop", {"type": "content_block_stop", "index": open_index}
                        )
                    text_index = next_index
                    next_index += 1
                    open_index = text_index
                    yield encode(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": text_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield encode(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": text_index,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            for call in delta.get("tool_calls") or []:
                cidx = int(call.get("index") or 0)
                fn = call.get("function") or {}
                if cidx not in tool_index:
                    if open_index is not None:
                        yield encode(
                            "content_block_stop", {"type": "content_block_stop", "index": open_index}
                        )
                    tool_index[cidx] = next_index
                    next_index += 1
                    open_index = tool_index[cidx]
                    yield encode(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": open_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                                "name": fn.get("name") or "",
                                "input": {},
                            },
                        },
                    )
                if args := fn.get("arguments"):
                    yield encode(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": tool_index[cidx],
                            "delta": {"type": "input_json_delta", "partial_json": args},
                        },
                    )
            if finish := choice.get("finish_reason"):
                stop_reason = _finish_to_stop(finish)

    if not started:
        started = True
        yield start_message()
    if open_index is not None:
        yield encode("content_block_stop", {"type": "content_block_stop", "index": open_index})
    yield encode(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason or "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": completion},
            **({"credits_consumed": credits} if credits is not None else {}),
        },
    )
    yield encode("message_stop", {"type": "message_stop"})


# --------------------------------------------------------------------- helpers


def _content_parts(content: Any) -> list[dict]:
    if isinstance(content, list):
        return [p for p in content if isinstance(p, dict)]
    return []


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(str(p) for p in parts if p is not None)
    return str(content)


def _anthropic_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text")
            for b in content
            if isinstance(b, dict) and b.get("type") in {"text", "tool_result"} and b.get("text")
        ]
        if not parts:
            parts = [
                b.get("content")
                for b in content
                if isinstance(b, dict)
                and b.get("type") == "tool_result"
                and isinstance(b.get("content"), str)
            ]
        return "\n".join(str(p) for p in parts if p is not None)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content)


def _image_block(part: dict) -> dict:
    url = ((part.get("image_url") or {}).get("url")) or ""
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media = header[5:].split(";")[0] or "image/png"
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _image_url_from_anthropic(block: dict) -> str:
    source = block.get("source") or {}
    if source.get("type") == "base64":
        return f"data:{source.get('media_type') or 'image/png'};base64,{source.get('data') or ''}"
    return str(source.get("url") or "")


def _tool_from_chat(tool: dict) -> dict | None:
    if tool.get("type") not in (None, "function"):
        return None
    fn = tool.get("function") or tool
    name = fn.get("name")
    if not name:
        return None
    out = {"name": name, "description": fn.get("description") or ""}
    out["input_schema"] = fn.get("parameters") or {"type": "object", "properties": {}}
    return out


def _tool_to_chat(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.get("name") or "",
            "description": tool.get("description") or "",
            "parameters": tool.get("input_schema") or tool.get("parameters") or {"type": "object"},
        },
    }


def _tool_choice_from_chat(choice: Any) -> Any:
    if choice is None:
        return None
    if isinstance(choice, str):
        return {"auto": {"type": "auto"}, "none": {"type": "none"}, "required": {"type": "any"}}.get(
            choice, {"type": "auto"}
        )
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name")
        if name:
            return {"type": "tool", "name": name}
    return {"type": "auto"}


def _tool_choice_to_chat(choice: Any) -> Any:
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        ctype = choice.get("type")
        if ctype == "any":
            return "required"
        if ctype == "none":
            return "none"
        if ctype == "tool":
            return {"type": "function", "function": {"name": choice.get("name") or ""}}
        return "auto"
    return choice


def _blocks_to_chat_content(blocks: Any) -> tuple[str | None, list[dict] | None]:
    texts: list[str] = []
    tool_calls: list[dict] = []
    for block in blocks if isinstance(blocks, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
    content = "\n".join(t for t in texts if t) or None
    if not content and tool_calls:
        content = None
    return content, tool_calls or None


def _stop_to_finish(stop: Any) -> str:
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "refusal": "content_filter",
    }.get(str(stop), "stop")


def _finish_to_stop(finish: Any) -> str:
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
    }.get(str(finish), "end_turn")


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {}
    return value or {}


def error_event(exc: UpstreamError) -> bytes:
    return encode(
        "error",
        {"type": "error", "error": {"type": "api_error", "message": exc.message}},
    )
