"""Configuration for kie-proxy (TOML file + environment overrides)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_UPSTREAM = "https://api.kie.ai"


@dataclass
class KeyEntry:
    key: str
    label: str = ""


def default_config_path() -> Path:
    if env := os.environ.get("KIE_PROXY_CONFIG"):
        return Path(env).expanduser()
    if xdg := os.environ.get("XDG_CONFIG_HOME"):
        return Path(xdg).expanduser() / "kie-proxy" / "config.toml"
    return Path.home() / ".config" / "kie-proxy" / "config.toml"


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8080
    auth_token: str | None = None
    upstream: str = DEFAULT_UPSTREAM
    refresh_seconds: int = 600
    log_limit: int = 500
    keys: list[KeyEntry] = field(default_factory=list)
    path: Path = field(default_factory=default_config_path)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def key_values(self) -> list[str]:
        return [k.key for k in self.keys]


def _env_settings(settings: Settings) -> Settings:
    if v := os.environ.get("KIE_PROXY_HOST"):
        settings.host = v
    if v := os.environ.get("KIE_PROXY_PORT"):
        settings.port = int(v)
    if v := os.environ.get("KIE_PROXY_TOKEN"):
        settings.auth_token = v
    if v := os.environ.get("KIE_PROXY_UPSTREAM"):
        settings.upstream = v.rstrip("/")
    if v := os.environ.get("KIE_PROXY_KEYS"):
        entries = [KeyEntry(key=k.strip()) for k in v.split(",") if k.strip()]
        if entries:
            settings.keys = entries
    if not settings.keys and (k := os.environ.get("KIE_API_KEY")):
        settings.keys = [KeyEntry(key=k, label="env")]
    return settings


def load_settings(path: Path | None = None) -> Settings:
    path = (path or default_config_path()).expanduser()
    settings = Settings(path=path)
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for name in ("host", "port", "auth_token", "upstream", "refresh_seconds", "log_limit"):
            if name in data and data[name] is not None:
                setattr(settings, name, data[name])
        keys = data.get("keys") or []
        settings.keys = [
            KeyEntry(key=str(k["key"]), label=str(k.get("label") or "")) for k in keys if k.get("key")
        ]
    return _env_settings(settings)


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def save_settings(settings: Settings) -> Path:
    path = settings.path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# kie-proxy configuration",
        f'host = "{_toml_escape(settings.host)}"',
        f"port = {settings.port}",
        f"upstream = \"{_toml_escape(settings.upstream)}\"",
        f"refresh_seconds = {settings.refresh_seconds}",
        f"log_limit = {settings.log_limit}",
    ]
    if settings.auth_token:
        lines.append(f'auth_token = "{_toml_escape(settings.auth_token)}"')
    for entry in settings.keys:
        lines.append("")
        lines.append("[[keys]]")
        lines.append(f'key = "{_toml_escape(entry.key)}"')
        if entry.label:
            lines.append(f'label = "{_toml_escape(entry.label)}"')
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o600)
    return path
