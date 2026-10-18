from __future__ import annotations

import json

from kie_proxy.translate.sse import UsageSniffer, encode, encode_done
from tests.conftest import sse_from


async def test_iter_sse_splits_and_skips_comments():
    text = (
        ": keep-alive\n\n"
        'event: message_start\ndata: {"a":1}\n\n'
        "data: [DONE]\n\n"
    )
    events = await sse_from(text)
    assert events[0].event == "message_start"
    assert events[0].json == {"a": 1}
    assert events[1].done


async def test_iter_sse_handles_split_chunks_and_crlf():
    text = "event: e\r\ndata: {\"x\":1}\r\n\r\ndata: [DONE]\r\n\r\n"
    events = await sse_from(text)
    assert events[0].json == {"x": 1}
    assert events[1].done


def test_encode_roundtrip():
    assert encode("ping", {"a": 1}) == b'event: ping\ndata: {"a": 1}\n\n'
    assert encode_done() == b"data: [DONE]\n\n"
    assert encode(None, "plain") == b"data: plain\n\n"


def test_usage_sniffer_chat_shape():
    sniffer = UsageSniffer()
    sniffer.feed(
        json.dumps({"usage": {"prompt_tokens": 12, "completion_tokens": 3}, "credits_consumed": 0.02})
    )
    assert (sniffer.prompt_tokens, sniffer.completion_tokens, sniffer.credits) == (12, 3, 0.02)


def test_usage_sniffer_anthropic_shape():
    sniffer = UsageSniffer()
    sniffer.feed(json.dumps({"usage": {"input_tokens": 7, "output_tokens": 5}}))
    assert (sniffer.prompt_tokens, sniffer.completion_tokens) == (7, 5)


def test_usage_sniffer_ignores_junk():
    sniffer = UsageSniffer()
    sniffer.feed("not json")
    sniffer.feed(json.dumps({"usage": "nope"}))
    assert sniffer.prompt_tokens == 0
