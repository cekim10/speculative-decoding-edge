#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: tools/netem.sh <apply|clear|show> [options]

Options:
  -i <iface>      Network interface (default: IFACE env or auto-detect default route)
  -d <delay_ms>   One-way delay in ms for apply (required for apply)
  -h              Show help

Notes:
- This script uses Linux tc/netem.
- Delay configured is one-way. Approximate RTT impact is ~2x delay.
- Interface selection order: explicit -i > IFACE env > detected default route interface.
USAGE
}

require_linux_tc() {
    if [ "$(uname -s)" != "Linux" ]; then
        echo "Error: tc/netem is supported by this script on Linux only. Current OS: $(uname -s)" >&2
        exit 1
    fi

    if ! command -v tc >/dev/null 2>&1; then
        echo "Error: 'tc' command not found. Install iproute2 first." >&2
        exit 1
    fi
}

detect_iface() {
    if [ -n "${iface:-}" ]; then
        echo "$iface"
        return 0
    fi

    if [ -n "${IFACE:-}" ]; then
        echo "$IFACE"
        return 0
    fi

    if command -v ip >/dev/null 2>&1; then
        local default_line
        default_line="$(ip route show default 2>/dev/null | head -n1 || true)"
        if [ -n "$default_line" ]; then
            echo "$default_line" | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}'
            return 0
        fi
    fi

    if command -v route >/dev/null 2>&1; then
        local route_line
        route_line="$(route -n get default 2>/dev/null | awk '/interface:/{print $2; exit}' || true)"
        if [ -n "$route_line" ]; then
            echo "$route_line"
            return 0
        fi
    fi

    return 1
}

show_qdisc() {
    local _iface="$1"
    echo "[netem] qdisc for ${_iface}:"
    tc qdisc show dev "$_iface" || true
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

action="$1"
shift

iface=""
delay_ms=""

while getopts "i:d:h" opt; do
    case "$opt" in
        i) iface="$OPTARG" ;;
        d) delay_ms="$OPTARG" ;;
        h)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

require_linux_tc

iface="$(detect_iface)" || {
    echo "Error: failed to detect interface. Pass -i <iface> or set IFACE env." >&2
    exit 1
}

echo "[netem] action=${action} iface=${iface}"
show_qdisc "$iface"

case "$action" in
    apply)
        if [ -z "$delay_ms" ]; then
            echo "Error: -d <delay_ms> is required for apply" >&2
            exit 1
        fi
        echo "[netem] applying one-way delay=${delay_ms}ms (approx RTT impact ~${delay_ms}*2 ms)"
        sudo tc qdisc replace dev "$iface" root netem delay "${delay_ms}ms"
        ;;
    clear)
        echo "[netem] clearing root qdisc on ${iface}"
        sudo tc qdisc del dev "$iface" root || true
        ;;
    show)
        ;;
    *)
        usage
        exit 1
        ;;
esac

show_qdisc "$iface"
