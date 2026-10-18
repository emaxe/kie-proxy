"""Upstream (KIE) error model and its rendering in client formats.

KIE answers some errors with HTTP 200 and ``{"code": N, "msg": "..."}`` in the body,
others with real HTTP status codes and either an OpenAI-ish or Anthropic-ish error
object. Everything is normalised into :class:`UpstreamError` and then re-rendered
into the format the client speaks.
"""

from __future__ import annotations

from dataclasses import dataclass

CODE_TO_STATUS: dict[int, int] = {
    400: 400,
    401: 401,
    402: 402,
    403: 403,
    404: 404,
    408: 408,
    422: 422,
    429: 429,
    455: 503,
    500: 500,
    501: 502,
    505: 503,
}


@dataclass
class UpstreamError(Exception):
    status: int
    message: str
    code: int | None = None
    kind: str = "server"
    upstream: str = "kie"

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"[{self.status}/{self.code}] {self.message}"

    @property
    def retryable_with_another_key(self) -> bool:
        return self.kind in {"auth", "credits", "rate"}

    @classmethod
    def from_body(cls, status: int, body: object) -> UpstreamError:
        """Build an error from a decoded JSON body, whatever shape it has."""
        code: int | None = None
        message = "upstream error"
        if isinstance(body, dict):
            if isinstance(body.get("code"), int) and "msg" in body:
                code = body["code"]
                message = str(body.get("msg") or message)
                status = CODE_TO_STATUS.get(code, status if status >= 400 else 200)
            elif isinstance(body.get("error"), dict):
                err = body["error"]
                message = str(err.get("message") or err.get("msg") or message)
                etype = str(err.get("type") or "")
                if "authentication" in etype:
                    code = 401
                elif "rate" in etype:
                    code = 429
            elif "message" in body and body.get("status"):
                message = str(body["message"])
        if status < 400:
            status = 502
        return cls(status=status, code=code, message=message, kind=_kind(code, status))

    @classmethod
    def from_exception(cls, exc: Exception) -> UpstreamError:
        return cls(status=502, message=f"upstream request failed: {exc}", kind="connect")

    def as_openai(self, model: str | None = None) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": _openai_type(self.kind),
                "param": None,
                "code": self.code if self.code is not None else self.status,
            }
        }

    def as_anthropic(self) -> dict:
        return {
            "type": "error",
            "error": {
                "type": _anthropic_type(self.kind),
                "message": self.message,
            },
        }


def _kind(code: int | None, status: int) -> str:
    if code in (401, 403) or status in (401, 403):
        return "auth"
    if code == 402 or status == 402:
        return "credits"
    if code == 429 or status == 429:
        return "rate"
    if code in (404, 422) or status in (404, 422):
        return "invalid_request"
    if status in (408, 502, 503, 504):
        return "connect"
    return "server"


def _openai_type(kind: str) -> str:
    return {
        "auth": "authentication_error",
        "credits": "insufficient_quota",
        "rate": "rate_limit_error",
        "invalid_request": "invalid_request_error",
        "connect": "api_connection_error",
    }.get(kind, "api_error")


def _anthropic_type(kind: str) -> str:
    return {
        "auth": "authentication_error",
        "credits": "insufficient_quota",
        "rate": "rate_limit_error",
        "invalid_request": "invalid_request_error",
        "connect": "api_connection_error",
    }.get(kind, "api_error")
