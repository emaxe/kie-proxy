"""HTTP client for the KIE upstream with key failover and error normalisation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from .errors import UpstreamError
from .keys import KeyPool, KeyState

TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0)
MAX_ATTEMPTS = 4


@dataclass
class StreamResult:
    key: KeyState
    response: httpx.Response

    async def bytes_iter(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self.response.aiter_raw():
                if chunk:
                    yield chunk
        finally:
            await self.response.aclose()


class KieClient:
    def __init__(
        self,
        pool: KeyPool,
        base_url: str,
        client: httpx.AsyncClient | None = None,
    ):
        self.pool = pool
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=TIMEOUT, http2=False)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # ------------------------------------------------------------------ json api

    async def get_json(self, path: str, params: dict | None = None) -> object:
        """GET with key failover; returns the decoded body."""
        last: UpstreamError | None = None
        for _ in range(MAX_ATTEMPTS):
            key = self.pool.acquire()
            try:
                resp = await self._client.get(
                    self.base_url + path, params=params, headers=self._headers(key.key)
                )
            except httpx.HTTPError as exc:
                last = UpstreamError.from_exception(exc)
                continue
            if resp.status_code >= 400:
                try:
                    body: object = resp.json()
                except ValueError:
                    body = {"code": resp.status_code, "msg": resp.text[:300]}
                err = UpstreamError.from_body(resp.status_code, body)
                self._report(key, err)
                if err.retryable_with_another_key:
                    last = err
                    continue
                raise err
            try:
                data: object = resp.json()
            except ValueError as exc:
                raise UpstreamError(status=502, message=f"invalid JSON from upstream: {exc}") from exc
            if isinstance(data, dict) and isinstance(data.get("code"), int) and data["code"] != 200:
                err = UpstreamError.from_body(resp.status_code, data)
                self._report(key, err)
                if err.retryable_with_another_key:
                    last = err
                    continue
                raise err
            self.pool.report_success(key.key)
            return data
        raise last or UpstreamError(status=503, message="no usable API key")

    async def post_json(self, path: str, body: dict) -> tuple[object, int]:
        """POST a JSON body and fully read the response. Returns (data, http_status)."""
        last: UpstreamError | None = None
        for _ in range(MAX_ATTEMPTS):
            key = self.pool.acquire()
            try:
                resp = await self._client.post(
                    self.base_url + path, json=body, headers=self._headers(key.key)
                )
            except httpx.HTTPError as exc:
                last = UpstreamError.from_exception(exc)
                continue
            try:
                data: object = resp.json()
            except ValueError:
                data = {"code": resp.status_code, "msg": resp.text[:500]}
            if resp.status_code >= 400 or _is_error_payload(data):
                err = UpstreamError.from_body(resp.status_code, data)
                self._report(key, err)
                if err.retryable_with_another_key:
                    last = err
                    continue
                raise err
            self.pool.report_success(key.key)
            return data, resp.status_code
        raise last or UpstreamError(status=503, message="no usable API key")

    # ------------------------------------------------------------------ streaming

    async def post_stream(self, path: str, body: dict) -> StreamResult:
        """POST and keep the response stream open.

        Retries on key-scoped failures (401/402/429) while the upstream has not
        started streaming yet. A JSON body arriving where SSE was requested is an
        upstream error; a JSON body when ``stream: false`` was sent is the result.
        """
        want_stream = bool(body.get("stream"))
        last: UpstreamError | None = None
        for _ in range(MAX_ATTEMPTS):
            key = self.pool.acquire()
            request = self._client.build_request(
                "POST", self.base_url + path, json=body, headers=self._headers(key.key)
            )
            try:
                resp = await self._client.send(request, stream=True)
            except httpx.HTTPError as exc:
                last = UpstreamError.from_exception(exc)
                continue

            ctype = resp.headers.get("content-type", "")
            if "text/event-stream" in ctype:
                self.pool.report_success(key.key)
                return StreamResult(key=key, response=resp)

            # JSON body: either an error or the full non-streamed result.
            raw = await resp.aread()
            try:
                data: object = json.loads(raw)
            except ValueError:
                data = {"code": resp.status_code, "msg": raw[:500].decode("utf-8", "replace")}
            status = resp.status_code
            await resp.aclose()
            if status >= 400 or _is_error_payload(data) or want_stream:
                err = UpstreamError.from_body(status, data)
                self._report(key, err)
                if err.retryable_with_another_key:
                    last = err
                    continue
                raise err
            self.pool.report_success(key.key)
            return StreamResult(key=key, response=_wrap_json(data, status))
        raise last or UpstreamError(status=503, message="no usable API key")

    def _report(self, key: KeyState, err: UpstreamError) -> None:
        if err.kind == "auth":
            self.pool.mark_invalid(key.key, err.message)
        elif err.kind == "credits":
            self.pool.mark_exhausted(key.key, err.message)
        elif err.kind == "rate":
            self.pool.mark_rate_limited(key.key, err.message)


def _is_error_payload(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    if isinstance(data.get("code"), int) and data["code"] != 200:
        return True
    if data.get("error"):
        return True
    return isinstance(data.get("type"), str) and data["type"] == "error" 


def _wrap_json(data: object, status: int) -> httpx.Response:
    """Wrap an already-read JSON body into a streaming-looking httpx.Response."""
    return httpx.Response(
        status_code=status,
        headers={"content-type": "application/json"},
        content=json.dumps(data).encode(),
    )
