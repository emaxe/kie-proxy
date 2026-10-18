"""Starlette application: the OpenAI/Anthropic-compatible front door of the proxy."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .config import Settings
from .errors import UpstreamError
from .keys import KeyPool, NoKeyAvailable
from .observe import Observer, RequestEvent, usage_from_anthropic, usage_from_chat
from .registry import Registry
from .routing import Inbound, plan
from .translate import (
    canonicalize_inbound,
    iter_sse,
    prepare_upstream_body,
    render_inbound_body,
    stream_transcoder,
)
from .translate.sse import UsageSniffer
from .upstream import KieClient

log = logging.getLogger("kie-proxy")

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class ProxyState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool = KeyPool(settings.keys)
        self.client = KieClient(self.pool, settings.upstream)
        self.registry = Registry(settings.refresh_seconds)
        self.observer = Observer(settings.log_limit)
        self._refresh_task: asyncio.Task | None = None

    async def startup(self) -> None:
        await self._refresh_soon()
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def shutdown(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task
        await self.client.aclose()

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.refresh_seconds)
            try:
                await self.registry.refresh(self.client)
            except Exception as exc:
                log.warning("registry refresh failed: %s", exc)

    async def _refresh_soon(self) -> None:
        try:
            await self.registry.refresh(self.client)
        except Exception as exc:
            log.warning("initial registry refresh failed: %s", exc)


def create_app(settings: Settings) -> Starlette:
    state = ProxyState(settings)

    @asynccontextmanager
    async def lifespan(_: Starlette):
        await state.startup()
        yield
        await state.shutdown()

    app = Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/stats", stats, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/models/{model_id:path}", model_detail, methods=["GET"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/messages", messages, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.state.kie = state
    return app


# ---------------------------------------------------------------------- routes


async def healthz(request: Request) -> JSONResponse:
    state: ProxyState = request.app.state.kie
    return JSONResponse(
        {
            "status": "ok",
            "models": len(state.registry.models),
            "registry_updated_at": state.registry.updated_at,
            "registry_error": state.registry.last_error,
            "keys": [
                {"label": k.label_or_masked, "status": k.status.value, "credits": k.credits}
                for k in state.pool.snapshot()
            ],
        }
    )


async def stats(request: Request) -> JSONResponse:
    state: ProxyState = request.app.state.kie
    st = state.observer.stats
    return JSONResponse(
        {
            "requests": st.requests,
            "errors": st.errors,
            "stream_requests": st.stream_requests,
            "prompt_tokens": st.prompt_tokens,
            "completion_tokens": st.completion_tokens,
            "credits": st.credits,
            "uptime_seconds": int(time.time() - st.started_at),
            "recent": [
                {
                    "ts": e.ts,
                    "path": e.path,
                    "model": e.model,
                    "inbound": e.inbound,
                    "native": e.native,
                    "status": e.status,
                    "ms": e.ms,
                    "stream": e.stream,
                    "prompt_tokens": e.prompt_tokens,
                    "completion_tokens": e.completion_tokens,
                    "credits": e.credits,
                    "error": e.error,
                    "key": e.key,
                }
                for e in list(state.observer.events)[:50]
            ],
        }
    )


async def models(request: Request) -> JSONResponse:
    state: ProxyState = request.app.state.kie
    _ensure_fresh(state)
    if "anthropic-version" in request.headers:
        return JSONResponse(state.registry.anthropic_payload())
    return JSONResponse(state.registry.openai_payload())


async def model_detail(request: Request) -> JSONResponse:
    state: ProxyState = request.app.state.kie
    _ensure_fresh(state)
    info = state.registry.get(request.path_params["model_id"])
    if info is None:
        missing = request.path_params["model_id"]
    return JSONResponse(
        {"error": {"message": f"model not found: {missing}", "type": "invalid_request_error"}},
        status_code=404,
    )
    return JSONResponse(
        {
            "id": info.id,
            "object": "model",
            "owned_by": info.owned_by,
            "provider": info.provider,
            "title": info.title,
            "description": info.description,
            "native": info.native_family,
            "protocol": info.protocol,
            "context_window": info.context_window,
            "max_tokens": info.max_tokens,
            "max_input_tokens": info.max_input_tokens,
            "aliases": info.aliases,
        }
    )


async def chat_completions(request: Request) -> Response:
    return await _handle(request, Inbound.OPENAI, "/v1/chat/completions")


async def messages(request: Request) -> Response:
    return await _handle(request, Inbound.ANTHROPIC, "/v1/messages")


# --------------------------------------------------------------------- handler


async def _handle(request: Request, inbound: Inbound, path: str) -> Response:
    state: ProxyState = request.app.state.kie
    started = time.monotonic()
    model_id = ""
    provider = ""
    native = ""

    if not _authorized(request, state.settings.auth_token):
        return _error_response(
            inbound,
            UpstreamError(status=401, message="invalid proxy credentials", kind="auth"),
        )

    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error_response(
            inbound,
            UpstreamError(status=400, message="request body is not valid JSON", kind="invalid_request"),
        )
    if not isinstance(body, dict):
        return _error_response(
            inbound,
            UpstreamError(status=400, message="request body must be a JSON object", kind="invalid_request"),
        )

    model_id = str(body.get("model") or "")
    if not model_id:
        return _error_response(
            inbound,
            UpstreamError(status=400, message="field 'model' is required", kind="invalid_request"),
        )
    info = state.registry.get(model_id)
    if info is None:
        return _error_response(
            inbound,
            UpstreamError(status=404, message=f"model not found: {model_id}", kind="invalid_request"),
        )
    provider, native = info.provider, info.native_family
    route = plan(info, inbound)

    try:
        if route.passthrough:
            upstream_body = {**body, "model": route.upstream_body_model}
        else:
            canonical = canonicalize_inbound(inbound, body)
            upstream_body = prepare_upstream_body(
                route.inbound, info.protocol, canonical, route.upstream_body_model
            )
    except ValueError as exc:
        return _error_response(
            inbound,
            UpstreamError(status=501, message=str(exc), kind="server"),
        )

    want_stream = bool(body.get("stream"))
    try:
        if want_stream:
            return await _stream_response(
                state,
                request,
                inbound,
                info,
                upstream_body,
                model_id,
                started,
                provider=provider,
                native=native,
                path=path,
            )
        data, status = await state.client.post_json(info.path, upstream_body)
    except NoKeyAvailable as exc:
        return _error_response(
            inbound, UpstreamError(status=503, message=str(exc), kind="server"), model_id
        )
    except UpstreamError as exc:
        state.observer.record(
            RequestEvent(
                ts=time.time(),
                path=path,
                model=model_id,
                provider=provider,
                inbound=str(inbound),
                native=native,
                status=exc.status,
                ms=int((time.monotonic() - started) * 1000),
                stream=want_stream,
                error=exc.message,
            )
        )
        return _error_response(inbound, exc, model_id)

    rendered = render_inbound_body(inbound, info.protocol, data, model_id)
    if inbound == Inbound.OPENAI:
        prompt, completion, credits = usage_from_chat(rendered)
    else:
        prompt, completion, credits = usage_from_anthropic(rendered)
    state.observer.record(
        RequestEvent(
            ts=time.time(),
            path=path,
            model=model_id,
            provider=provider,
            inbound=str(inbound),
            native=native,
            status=status,
            ms=int((time.monotonic() - started) * 1000),
            stream=False,
            prompt_tokens=prompt,
            completion_tokens=completion,
            credits=credits,
        )
    )
    return JSONResponse(rendered, status_code=status if status >= 400 else 200)


async def _stream_response(
    state: ProxyState,
    request: Request,
    inbound: Inbound,
    info: Any,
    upstream_body: dict,
    model_id: str,
    started: float,
    *,
    provider: str,
    native: str,
    path: str,
) -> Response:
    result = await state.client.post_stream(info.path, upstream_body)
    key_masked = result.key.masked
    sniffer = UsageSniffer()
    events = _sniff(iter_sse(result.bytes_iter()), sniffer)
    transcoder = stream_transcoder(inbound, info.protocol)

    if transcoder is None:
        raw = _passthrough(events)
        out: Any = raw
    else:
        out = transcoder(events, model_id)

    async def generate():
        try:
            async for chunk in out:
                yield chunk
        finally:
            state.observer.record(
                RequestEvent(
                    ts=time.time(),
                    path=path,
                    model=model_id,
                    provider=provider,
                    inbound=str(inbound),
                    native=native,
                    status=200,
                    ms=int((time.monotonic() - started) * 1000),
                    stream=True,
                    prompt_tokens=sniffer.prompt_tokens,
                    completion_tokens=sniffer.completion_tokens,
                    credits=sniffer.credits,
                    key=key_masked,
                )
            )

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)


async def _sniff(events: Any, sniffer: UsageSniffer) -> Any:
    async for event in events:
        sniffer.feed(event.data)
        yield event


async def _passthrough(events: Any) -> Any:
    from .translate.sse import encode, encode_done

    async for event in events:
        if event.done:
            yield encode_done()
            continue
        yield encode(event.event, event.data)


# -------------------------------------------------------------------- helpers


def _ensure_fresh(state: ProxyState) -> None:
    if state.registry.stale and state.registry.models:
        asyncio.get_running_loop().create_task(_safe_refresh(state))


async def _safe_refresh(state: ProxyState) -> None:
    try:
        await state.registry.refresh(state.client)
    except Exception as exc:
        log.warning("background registry refresh failed: %s", exc)


def _authorized(request: Request, token: str | None) -> bool:
    if not token:
        return True
    auth = request.headers.get("authorization") or ""
    if auth.startswith("Bearer ") and auth[7:].strip() == token:
        return True
    xapi = (request.headers.get("x-api-key") or "").strip()
    if xapi == token:
        return True
    return xapi.lower().removeprefix("bearer ").strip() == token.lower()


def _error_response(inbound: Inbound, exc: UpstreamError, model: str | None = None) -> JSONResponse:
    if inbound == Inbound.OPENAI:
        return JSONResponse(exc.as_openai(model), status_code=exc.status)
    return JSONResponse(exc.as_anthropic(), status_code=exc.status)
