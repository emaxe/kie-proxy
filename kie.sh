#!/bin/sh
# kie-proxy — интерактивное меню сборки и запуска.
# Все режимы: установка зависимостей, сборка, тесты, линт, сервер, TUI,
# список моделей, одноразовый чат, инициализация конфига.

set -u

cd "$(dirname "$0")" || exit 1

if ! command -v uv >/dev/null 2>&1; then
    printf 'error: uv not found (https://docs.astral.sh/uv/)\n' >&2
    exit 1
fi

die() {
    printf '%s\n' "$1" >&2
    exit 1
}

pause() {
    printf '\nPress Enter to continue...'
    # shellcheck disable=SC2034
    read -r _ || true
}

banner() {
    printf '\n=== %s ===\n\n' "$1"
}

need_keys() {
    if [ -n "${KIE_API_KEY:-}" ] || [ -n "${KIE_PROXY_KEYS:-}" ]; then
        return 0
    fi
    cfg="${KIE_PROXY_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/kie-proxy/config.toml}"
    if [ -f "$cfg" ]; then
        return 0
    fi
    printf 'No API key: set KIE_API_KEY or run "init" first.\n'
    printf 'Paste KIE API key for this session (empty to cancel): '
    read -r key || return 1
    [ -n "$key" ] || return 1
    KIE_API_KEY="$key"
    export KIE_API_KEY
}

ask_port() {
    printf 'Port [%s]: ' "${KIE_PROXY_PORT:-8080}"
    read -r port || port=""
    if [ -n "$port" ]; then
        KIE_PROXY_PORT="$port"
        export KIE_PROXY_PORT
    fi
}

mode_sync() {
    banner "install dependencies (uv sync)"
    uv sync || die "uv sync failed"
}

mode_build() {
    banner "build wheel (uv build)"
    uv build || die "uv build failed"
}

mode_test() {
    banner "tests (pytest)"
    uv run pytest -q || die "tests failed"
}

mode_lint() {
    banner "lint (ruff)"
    uv run ruff check src tests || die "ruff check failed"
    printf 'ruff: ok\n'
}

mode_run() {
    banner "server"
    need_keys || return 0
    ask_port
    printf 'Starting kie-proxy on http://%s:%s ...\n' \
        "${KIE_PROXY_HOST:-127.0.0.1}" "${KIE_PROXY_PORT:-8080}"
    uv run kie-proxy run
}

mode_tui() {
    banner "TUI (server + live monitor)"
    need_keys || return 0
    ask_port
    uv run kie-proxy tui
}

mode_models() {
    banner "models"
    need_keys || return 0
    uv run kie-proxy models
}

mode_chat() {
    banner "one-shot chat"
    need_keys || return 0
    printf 'Model id: '
    read -r model || return 0
    [ -n "$model" ] || return 0
    printf 'Prompt: '
    read -r prompt || return 0
    [ -n "$prompt" ] || return 0
    uv run kie-proxy chat "$model" "$prompt"
}

mode_init() {
    banner "init config"
    cfg="${KIE_PROXY_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/kie-proxy/config.toml}"
    if [ -f "$cfg" ]; then
        printf 'Config already exists: %s\n' "$cfg"
        printf 'Overwrite? [y/N]: '
        read -r ans || return 0
        case "$ans" in
            y | Y | yes | YES) ;;
            *) return 0 ;;
        esac
    fi
    if [ -z "${KIE_API_KEY:-}" ]; then
        printf 'KIE API key (stored in config, chmod 600): '
        read -r key || return 0
        [ -n "$key" ] || return 0
        uv run kie-proxy init --key "$key" || die "init failed"
    else
        uv run kie-proxy init || die "init failed"
    fi
    printf 'Done: %s\n' "$cfg"
}

menu() {
    printf '\n'
    printf 'kie-proxy  (uv · starlette · textual)\n'
    printf '\n'
    printf '  1) install dependencies (uv sync)\n'
    printf '  2) build wheel            (uv build)\n'
    printf '  3) run tests              (pytest)\n'
    printf '  4) lint                   (ruff)\n'
    printf '  5) init / overwrite config\n'
    printf '  6) run server\n'
    printf '  7) run server + TUI monitor\n'
    printf '  8) list models\n'
    printf '  9) one-shot chat\n'
    printf '  0) exit\n'
    printf '\n'
    printf 'choice: '
}

main() {
    while true; do
        menu
        read -r choice || exit 0
        case "$choice" in
            1) mode_sync; pause ;;
            2) mode_build; pause ;;
            3) mode_test; pause ;;
            4) mode_lint; pause ;;
            5) mode_init; pause ;;
            6) mode_run ;;
            7) mode_tui ;;
            8) mode_models; pause ;;
            9) mode_chat; pause ;;
            0) printf 'bye\n'; exit 0 ;;
            *) printf 'unknown choice: %s\n' "$choice" ;;
        esac
    done
}

main "$@"
