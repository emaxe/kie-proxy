# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-18

Initial release.

### Added

- OpenAI-compatible `POST /v1/chat/completions` and Anthropic-compatible
  `POST /v1/messages` front doors, streaming supported.
- Protocol translators (both directions, streaming):
  - OpenAI Chat ↔ Anthropic Messages (Claude family)
  - OpenAI Chat ↔ OpenAI Responses (gpt-5.5+/gpt-6, grok-4.x, kimi-k3,
    deepseek-v4 and other Responses-native models)
  - composed Responses → Chat → Messages path for Anthropic clients on
    Responses-native models
- Native passthrough when the inbound format matches the model's protocol
  (thinking blocks preserved for Claude Code ↔ Claude).
- Hybrid model registry: KIE catalog + openai/anthropic/xai protocol listings,
  per-model doc paths, background refresh, `GET /v1/models` in both OpenAI and
  Anthropic shapes, `GET /v1/models/{id}`.
- Key pool: round-robin, cooldown on 429, failover on 401/402, invalid-key
  quarantine, `/healthz` status.
- Error normalisation: KIE `HTTP 200 + {"code","msg"}` and OpenAI/Anthropic
  error objects re-rendered into the client's format.
- Observability: `/stats`, in-memory request log with token/credit usage.
- Textual TUI (`kie-proxy tui`): live request table, stats bar, key statuses.
- CLI: `run`, `tui`, `models`, `chat`, `init`.
- Interactive `kie.sh` menu (sync, build, test, lint, init, run, tui, models,
  chat); output via `printf`, POSIX sh.
- Test suite (68 tests) based on recorded real KIE fixtures: translators,
  SSE parsing, registry routing, key pool, HTTP surface, TUI render.
