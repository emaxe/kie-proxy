from __future__ import annotations

from kie_proxy.errors import UpstreamError


def test_kie_style_200_body_with_code():
    err = UpstreamError.from_body(200, {"code": 402, "msg": "Insufficient account credits", "data": None})
    assert err.status == 402
    assert err.kind == "credits"
    assert err.retryable_with_another_key


def test_openai_style_error_body():
    err = UpstreamError.from_body(
        404,
        {"error": {"type": "server_error", "message": 'Model "gpt-5.5" targets GPT-5.5, which retired'}},
    )
    assert err.status == 404
    assert "retired" in err.message


def test_anthropic_style_error_body():
    err = UpstreamError.from_body(
        401, {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}
    )
    assert err.kind == "auth"


def test_as_openai_shape():
    err = UpstreamError(status=429, code=429, message="slow down", kind="rate")
    payload = err.as_openai("gpt-5-2")
    assert payload["error"]["type"] == "rate_limit_error"
    assert payload["error"]["code"] == 429


def test_as_anthropic_shape():
    err = UpstreamError(status=400, message="bad", kind="invalid_request")
    payload = err.as_anthropic()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"


def test_non_4xx_status_becomes_502():
    err = UpstreamError.from_body(200, {"unexpected": True})
    assert err.status == 502


def test_from_exception_is_connect():
    err = UpstreamError.from_exception(TimeoutError("boom"))
    assert err.kind == "connect"
    assert err.status == 502
