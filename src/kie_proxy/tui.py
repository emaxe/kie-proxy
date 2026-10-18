"""Live monitoring TUI (Textual): request log, stats, keys.

The HTTP server runs in a dedicated thread with its own event loop so uvicorn and
Textual never fight over signal handlers. The TUI polls the shared observer state
once a second.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Any, ClassVar

import uvicorn
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, Static

from .config import Settings
from .server import create_app


class StatsBar(Static):
    def update_stats(self, text: str) -> None:
        self.update(text)


class KieTui(App):
    CSS = """
    StatsBar {
        dock: top;
        height: 1;
        background: $boost;
        color: $text;
        padding: 0 1;
    }
    #events { height: 1fr; }
    #keys { width: 42; }
    #statusline { dock: bottom; height: 1; padding: 0 1; color: $text-muted; }
    """

    BINDINGS: ClassVar = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh models"),
    ]

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.application = create_app(settings)
        self.state = self.application.state.kie
        self._server_thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._server_loop: asyncio.AbstractEventLoop | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield StatsBar("starting…")
        with Horizontal():
            yield DataTable(id="events", cursor_type="row")
            yield DataTable(id="keys", cursor_type="row")
        yield Static("", id="statusline")
        yield Footer()

    # ------------------------------------------------------------------ lifecycle

    async def on_mount(self) -> None:
        self.title = "kie-proxy"
        self.sub_title = self.settings.base_url
        events = self.query_one("#events", DataTable)
        events.add_columns("time", "in", "model", "native", "st", "ms", "tok in/out", "cr", "err")
        keys = self.query_one("#keys", DataTable)
        keys.add_columns("key", "status", "credits")
        self._start_server_thread()
        self.set_interval(1.0, self.refresh_view)

    async def on_unmount(self) -> None:
        await asyncio.to_thread(self._stop_server)

    # ----------------------------------------------------------------- server side

    def _start_server_thread(self) -> None:
        ready = threading.Event()

        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._server_loop = loop
            loop.run_until_complete(self.state.startup())
            config = uvicorn.Config(
                self.application,
                host=self.settings.host,
                port=self.settings.port,
                log_level="warning",
            )
            self._server = uvicorn.Server(config)
            ready.set()
            loop.run_until_complete(self._server.serve())

        self._server_thread = threading.Thread(target=run, name="kie-proxy-server", daemon=True)
        self._server_thread.start()
        ready.wait(timeout=30)

    def _stop_server(self) -> None:
        if self._server and self._server_loop:
            self._server.should_exit = True
            self._server_loop.call_soon_threadsafe(lambda: None)
        if self._server_thread:
            self._server_thread.join(timeout=5)

    async def action_refresh(self) -> None:
        if self._server_loop and self.state.registry.models:
            fut = asyncio.run_coroutine_threadsafe(
                self.state.registry.refresh(self.state.client), self._server_loop
            )
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.wrap_future(fut), timeout=30)
        self.refresh_view()

    # -------------------------------------------------------------------- display

    def refresh_view(self) -> None:
        st = self.state.observer.stats
        models = len(self.state.registry.models)
        uptime = int(time.time() - st.started_at)
        self.query_one(StatsBar).update_stats(
            f"requests {st.requests}  errors {st.errors}  streams {st.stream_requests}  "
            f"tokens {st.prompt_tokens}/{st.completion_tokens}  credits {st.credits:.2f}  "
            f"models {models}  uptime {uptime}s"
        )
        self._refresh_events()
        self._refresh_keys()
        self.query_one("#statusline", Static).update(
            f"{self.settings.base_url}  ·  registry "
            + (
                "ok"
                if not self.state.registry.last_error
                else f"error: {self.state.registry.last_error[:80]}"
            )
        )

    def _refresh_events(self) -> None:
        table = self.query_one("#events", DataTable)
        try:
            snapshot = list(self.state.observer.events)
        except RuntimeError:  # concurrent mutation
            return
        rows = [self._event_row(e) for e in snapshot[:100]]
        table.clear()
        for row in rows:
            table.add_row(*row)

    @staticmethod
    def _event_row(e: Any) -> tuple[str, str, str, str, str, str, str, str, str]:
        ts = time.strftime("%H:%M:%S", time.localtime(e.ts))
        status = "ERR" if not e.ok else str(e.status)
        err = (e.error or "")[:60]
        return (
            ts,
            e.inbound,
            e.model[:28],
            e.native,
            status,
            str(e.ms),
            f"{e.prompt_tokens}/{e.completion_tokens}",
            "-" if e.credits is None else f"{e.credits:g}",
            err,
        )

    def _refresh_keys(self) -> None:
        table = self.query_one("#keys", DataTable)
        states = self.state.pool.snapshot()
        table.clear()
        for s in states:
            table.add_row(
                s.label_or_masked, s.status.value, "-" if s.credits is None else f"{s.credits:g}"
            )


def run_tui(settings: Settings) -> None:
    KieTui(settings).run()
