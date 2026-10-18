"""CLI entry point: serve, TUI, one-shot chat and model listing."""

from __future__ import annotations

import asyncio
import json
import sys

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from .config import Settings, load_settings, save_settings

app = typer.Typer(
    name="kie-proxy",
    help="OpenAI/Anthropic-compatible proxy in front of KIE.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _settings(config: str | None, host: str | None = None, port: int | None = None,
              token: str | None = None) -> Settings:
    from pathlib import Path

    settings = load_settings(Path(config) if config else None)
    if host:
        settings.host = host
    if port:
        settings.port = port
    if token:
        settings.auth_token = token
    return settings


@app.command()
def run(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config.toml"),
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    token: str | None = typer.Option(None, "--token", help="Proxy auth token (Bearer / x-api-key)"),
    log_level: str = typer.Option("info", "--log-level"),
) -> None:
    """Start the HTTP proxy server."""
    settings = _settings(config, host, port, token)
    if not settings.keys:
        err_console.print("[red]No API keys configured. Set KIE_API_KEY or add [[keys]] to config.toml[/red]")
        raise typer.Exit(2)
    application = _app(settings)
    console.print(f"kie-proxy listening on [bold]{settings.base_url}[/bold]  ({len(settings.keys)} key(s))")
    uvicorn.run(application, host=settings.host, port=settings.port, log_level=log_level)


@app.command()
def tui(
    config: str | None = typer.Option(None, "--config", "-c"),
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    token: str | None = typer.Option(None, "--token"),
) -> None:
    """Start the server and open the live monitoring TUI."""
    from .tui import run_tui

    settings = _settings(config, host, port, token)
    if not settings.keys:
        err_console.print("[red]No API keys configured. Set KIE_API_KEY or add [[keys]] to config.toml[/red]")
        raise typer.Exit(2)
    run_tui(settings)


@app.command()
def models(
    config: str | None = typer.Option(None, "--config", "-c"),
    provider: str | None = typer.Option(None, "--provider", help="Filter by provider"),
    protocol: str | None = typer.Option(None, "--protocol", help="Filter by native protocol"),
) -> None:
    """List every model the proxy knows about."""
    settings = _settings(config)
    data = asyncio.run(_fetch_models(settings, provider, protocol))
    if data is None:
        raise typer.Exit(1)


@app.command()
def chat(
    model: str = typer.Argument(..., help="Model id as served by the proxy"),
    prompt: str = typer.Argument(..., help="User prompt"),
    config: str | None = typer.Option(None, "--config", "-c"),
    system: str | None = typer.Option(None, "--system"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
    raw: bool = typer.Option(False, "--raw", help="Print the raw JSON response"),
) -> None:
    """One-shot completion through the proxy routing (no local server needed)."""
    settings = _settings(config)
    ok = asyncio.run(_chat_once(settings, model, prompt, system, stream, raw))
    if not ok:
        raise typer.Exit(1)


@app.command("init")
def init_cmd(
    config: str | None = typer.Option(None, "--config", "-c"),
    key: str | None = typer.Option(None, "--key", help="KIE API key to store"),
    token: str | None = typer.Option(None, "--token", help="Proxy auth token to require from clients"),
) -> None:
    """Write a default config file (chmod 600)."""
    settings = _settings(config)
    if key:
        from .config import KeyEntry

        settings.keys = [KeyEntry(key=key, label="kie")]
    if token:
        settings.auth_token = token
    path = save_settings(settings)
    console.print(f"Wrote [bold]{path}[/bold]")


# --------------------------------------------------------------------- helpers


def _app(settings: Settings):
    from .server import create_app

    return create_app(settings)


async def _fetch_models(settings: Settings, provider: str | None, protocol: str | None) -> bool:
    from .keys import KeyPool
    from .registry import Registry
    from .upstream import KieClient

    if not settings.keys:
        err_console.print("[red]No API keys configured.[/red]")
        return False
    pool = KeyPool(settings.keys)
    client = KieClient(pool, settings.upstream)
    registry = Registry(settings.refresh_seconds)
    try:
        await registry.refresh(client)
    finally:
        await client.aclose()
    if not registry.models:
        err_console.print(f"[red]Registry is empty[/red] {registry.last_error}")
        return False
    table = Table(title="kie-proxy models", show_lines=False)
    table.add_column("id", style="cyan")
    table.add_column("provider")
    table.add_column("native")
    table.add_column("title", overflow="fold")
    table.add_column("context", justify="right")
    for m in registry.list():
        if provider and provider.lower() not in m.provider.lower():
            continue
        if protocol and m.protocol != protocol:
            continue
        table.add_row(
            m.id,
            m.provider,
            m.native_family,
            m.title or m.id,
            str(m.context_window or ""),
        )
    console.print(table)
    return True


async def _chat_once(
    settings: Settings, model: str, prompt: str, system: str | None, stream: bool, raw: bool
) -> bool:
    from .errors import UpstreamError
    from .keys import KeyPool, NoKeyAvailable
    from .registry import Registry
    from .routing import Inbound, plan
    from .translate import (
        canonicalize_inbound,
        iter_sse,
        prepare_upstream_body,
        render_inbound_body,
        stream_transcoder,
    )
    from .upstream import KieClient

    if not settings.keys:
        err_console.print("[red]No API keys configured.[/red]")
        return False
    pool = KeyPool(settings.keys)
    client = KieClient(pool, settings.upstream)
    registry = Registry(settings.refresh_seconds)
    try:
        await registry.refresh(client)
        info = registry.get(model)
        if info is None:
            err_console.print(f"[red]Unknown model:[/red] {model}")
            return False
        route = plan(info, Inbound.OPENAI)
        body: dict = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": stream}
        if system:
            body["messages"].insert(0, {"role": "system", "content": system})
        canonical = canonicalize_inbound(Inbound.OPENAI, body)
        upstream_body = prepare_upstream_body(
            route.inbound, info.protocol, canonical, route.upstream_body_model
        )
        if not stream:
            data, _ = await client.post_json(info.path, upstream_body)
            rendered = render_inbound_body(Inbound.OPENAI, info.protocol, data, model)
            if raw:
                console.print_json(json.dumps(rendered))
            else:
                choice = (rendered.get("choices") or [{}])[0]
                console.print((choice.get("message") or {}).get("content") or "")
                usage = rendered.get("usage") or {}
                err_console.print(
                    f"[dim]tokens: {usage.get('prompt_tokens')}+{usage.get('completion_tokens')}"
                    f"  credits: {rendered.get('credits_consumed', '-')}[/dim]"
                )
            return True
        result = await client.post_stream(info.path, upstream_body)
        transcoder = stream_transcoder(route.inbound, info.protocol)
        events = iter_sse(result.bytes_iter())
        out = transcoder(events, model) if transcoder else _passthrough(events)
        async for chunk in out:
            for line in chunk.decode("utf-8", "replace").splitlines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    data = json.loads(payload)
                except ValueError:
                    continue
                _print_stream_delta(data)
        err_console.print()
        return True
    except (UpstreamError, NoKeyAvailable) as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        return False
    finally:
        await client.aclose()


def _print_stream_delta(data: dict) -> None:
    if data.get("object") == "chat.completion.chunk":
        for choice in data.get("choices") or []:
            if text := (choice.get("delta") or {}).get("content"):
                sys.stdout.write(text)
                sys.stdout.flush()
        return
    if data.get("type") == "content_block_delta":
        delta = data.get("delta") or {}
        if delta.get("type") == "text_delta":
            sys.stdout.write(delta.get("text") or "")
            sys.stdout.flush()


async def _passthrough(events):
    from .translate.sse import encode, encode_done

    async for event in events:
        if event.done:
            yield encode_done()
        else:
            yield encode(event.event, event.data)


if __name__ == "__main__":  # pragma: no cover
    app()
