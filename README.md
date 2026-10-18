# kie-proxy

Local console proxy in front of [KIE](https://api.kie.ai): exposes **every KIE text
chat model** (Claude, GPT, Gemini, Grok, Kimi, DeepSeek, …) through **one
OpenAI-compatible and one Anthropic-compatible endpoint**, with protocol
translation on the fly.

```
OpenAI SDK / Claude Code / any client
        │  (chat.completions or messages)
        ▼
   kie-proxy  ──► api.kie.ai   (per-model native protocol: openai chat,
        ▲                        openai responses, anthropic messages)
        └── keys: round-robin, cooldown, failover
```

## Highlights

- **Two front doors**: `POST /v1/chat/completions` (OpenAI) and `POST /v1/messages` (Anthropic).
- **Full protocol translation** in both directions, streaming included:
  - OpenAI Chat ↔ Anthropic Messages (Claude models)
  - OpenAI Chat ↔ OpenAI Responses (gpt-5.5+/gpt-6, grok-4.x, kimi-k3, deepseek-v4)
  - Anthropic clients can talk to GPT/Grok/Kimi/DeepSeek models and vice versa
- **Native passthrough** when the client already speaks the model's protocol
  (Claude Code ↔ Claude models keeps thinking blocks intact).
- **Key pool**: round-robin over several KIE keys, automatic cooldown on 429,
  failover on 401/402 (bad key / out of credits).
- **Live TUI**: request log, token/credit stats, key statuses.
- **Model registry**: merged from KIE catalog + protocol listings, refreshed in
  the background (`GET /v1/models` in both OpenAI and Anthropic shapes).
- Errors from KIE (sometimes `HTTP 200 + {"code":N,"msg":…}`) are normalised and
  re-rendered in the format the client speaks.

## Install & run

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd kie-proxy
./kie.sh          # interactive menu: sync, test, lint, run, tui, models, chat
```

Or directly:

```bash
uv sync
export KIE_API_KEY=<your KIE key>     # or [[keys]] in config.toml
uv run kie-proxy run --port 8888       # HTTP server
uv run kie-proxy tui                   # server + live monitoring TUI
```

## Use

**Base URL**: `http://127.0.0.1:8888/v1` — **API key**: any non-empty string unless
you start the proxy with `--token` (clients never need your KIE key).

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8888/v1", api_key="kie")
resp = client.chat.completions.create(
    model="deepseek-v4-1-flash",          # any id from GET /v1/models
    messages=[{"role": "user", "content": "hi"}],
)
print(resp.choices[0].message.content)
```

Claude Code / Anthropic SDK:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8888
export ANTHROPIC_AUTH_TOKEN=kie
```

curl:

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"claude-haiku-4-5","messages":[{"role":"user","content":"hi"}]}'
```

## Endpoints

| Method | Path                 | Notes                                        |
| ------ | -------------------- | -------------------------------------------- |
| POST   | `/v1/chat/completions` | OpenAI Chat Completions, streaming supported |
| POST   | `/v1/messages`         | Anthropic Messages, streaming supported      |
| GET    | `/v1/models`           | OpenAI list; Anthropic shape when `anthropic-version` header is present |
| GET    | `/v1/models/{id}`      | Single model (protocol, context window, aliases) |
| GET    | `/healthz`             | Status, model count, key health              |
| GET    | `/stats`               | Request/token/credit counters + recent log   |

## Configuration

Config file: `~/.config/kie-proxy/config.toml` (created by `kie-proxy init`, chmod 600).

```toml
host = "127.0.0.1"
port = 8888
upstream = "https://api.kie.ai"
refresh_seconds = 600
auth_token = "optional-proxy-secret"   # then clients must send it

[[keys]]
key = "<kie api key>"
label = "main"
```

Environment overrides: `KIE_PROXY_CONFIG`, `KIE_PROXY_HOST`, `KIE_PROXY_PORT`,
`KIE_PROXY_TOKEN`, `KIE_PROXY_UPSTREAM`, `KIE_PROXY_KEYS` (comma-separated),
`KIE_API_KEY`.

If `auth_token` is set, clients authenticate with `Authorization: Bearer <token>`
or `x-api-key: <token>` (a `Bearer ` prefix inside `x-api-key` is accepted too).

## CLI

```
kie-proxy run    [--host] [--port] [--token] [--config]   # HTTP server
kie-proxy tui    [--host] [--port] [--token]              # server + TUI
kie-proxy models [--provider] [--protocol]                # model catalog
kie-proxy chat <model> <prompt> [--system] [--no-stream]  # one-shot, no server
kie-proxy init   [--key] [--token]                        # write config (600)
```

## Development

```bash
uv run pytest -q    # 68 tests, fixture-based on real KIE responses
uv run ruff check src tests
```

Protocol mapping is derived from KIE listings and per-model docs
(`src/kie_proxy/registry.py`); translators live in `src/kie_proxy/translate/`.

## Notes & limitations

- Gemini-native ids are routed through their `-openai` twins (the upstream
  native Gemini endpoint is unreliable); thinking/reasoning summaries are kept
  only on native passthrough routes, cross-protocol translation drops them.
- KIE sometimes answers errors with `HTTP 200` + `{"code":…,"msg":…}` — the
  proxy converts them to proper 4xx/5xx in the client's error format.
- Rate limit on the KIE side: 20 generation requests / 10 s per key.

## License

MIT
