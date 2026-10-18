from __future__ import annotations

import time

from kie_proxy.config import KeyEntry, Settings
from kie_proxy.observe import RequestEvent
from kie_proxy.tui import KieTui


async def _noop_refresh(client):
    return None


async def test_tui_renders_and_lists_events():
    app = KieTui(Settings(host="127.0.0.1", port=0, keys=[KeyEntry(key="test-key")]))
    app.state.registry.refresh = _noop_refresh  # type: ignore[method-assign]
    async with app.run_test() as pilot:
        await pilot.pause(1.2)
        assert app.title == "kie-proxy"
        stats = app.query_one("StatsBar")
        assert "requests" in str(stats.render())
        app.state.observer.record(
            RequestEvent(
                ts=time.time(),
                path="/v1/chat/completions",
                model="gpt-5-2",
                provider="OpenAI",
                inbound="openai",
                native="openai",
                status=200,
                ms=12,
                stream=True,
                prompt_tokens=10,
                completion_tokens=2,
                credits=0.01,
            )
        )
        await pilot.pause(1.2)
        table = app.query_one("#events")
        assert table.row_count == 1
        keys = app.query_one("#keys")
        assert keys.row_count == 1
