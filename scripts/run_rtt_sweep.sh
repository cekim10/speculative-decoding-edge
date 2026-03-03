#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/run_rtt_sweep.sh [options]

Options:
  -f <config>      Base config (default: config/dasd.workstation.yaml)
  -r <requests>    max_request_num override (default: 4)
  -n <max_tokens>  max_new_tokens override (default: 128)
  -o <output_dir>  output folder for merged logs (default: result/rtt_sweep)
  -i <iface>       network interface for netem (optional; else IFACE or auto-detect)
  -h               show help

RTT sweep:
  - RTT14: no netem shaping
  - RTT50: applies one-way netem delay 25ms
  - RTT100: applies one-way netem delay 50ms
USAGE
}

base_config="config/dasd.workstation.yaml"
max_requests=4
max_new_tokens=128
output_dir="result/rtt_sweep"
iface="${IFACE:-}"

while getopts "f:r:n:o:i:h" opt; do
    case "$opt" in
        f) base_config="$OPTARG" ;;
        r) max_requests="$OPTARG" ;;
        n) max_new_tokens="$OPTARG" ;;
        o) output_dir="$OPTARG" ;;
        i) iface="$OPTARG" ;;
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

cd "$(dirname "$0")/.." || { echo "Failed to change directory to project root"; exit 1; }

if [ ! -d .venv ]; then
    echo "You need to create a virtual environment first."
    exit 1
fi

source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

mkdir -p "$output_dir"
manifest="$output_dir/manifest.tsv"
echo -e "rtt_ms\tvariant\tclient_jsonl\tserver_jsonl\traw_result_dir" > "$manifest"

server_pid=""

cleanup() {
    if [ -n "${server_pid:-}" ]; then
        kill -INT "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [ -x "tools/netem.sh" ] && command -v tc >/dev/null 2>&1 && [ "$(uname -s)" = "Linux" ]; then
        IFACE="${iface:-}" tools/netem.sh clear ${iface:+-i "$iface"} >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

apply_rtt_profile() {
    local rtt_label="$1"

    if [ "$rtt_label" = "14" ]; then
        echo "[sweep] RTT14 profile: no shaping"
        if [ -x "tools/netem.sh" ] && command -v tc >/dev/null 2>&1 && [ "$(uname -s)" = "Linux" ]; then
            IFACE="${iface:-}" tools/netem.sh clear ${iface:+-i "$iface"} || true
        fi
        return 0
    fi

    local one_way_delay=0
    case "$rtt_label" in
        50) one_way_delay=25 ;;
        100) one_way_delay=50 ;;
        *)
            echo "Unsupported RTT label: $rtt_label" >&2
            exit 1
            ;;
    esac

    if [ ! -x tools/netem.sh ]; then
        echo "Error: tools/netem.sh not found" >&2
        exit 1
    fi

    IFACE="${iface:-}" tools/netem.sh apply ${iface:+-i "$iface"} -d "$one_way_delay"
}

run_variant() {
    local rtt_label="$1"
    local variant="$2"

    local tmp_config
    tmp_config="$(mktemp /tmp/specedge_rtt_sweep.XXXXXX.yaml)"

    local cfg_meta
    cfg_meta="$(python - "$base_config" "$tmp_config" "$variant" "$rtt_label" "$max_requests" "$max_new_tokens" <<'PY'
import sys
from datetime import datetime

import yaml

src, dst, variant, rtt_label, reqs, max_new = sys.argv[1:7]
reqs = int(reqs)
max_new = int(max_new)

with open(src, "r") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("base", {})
cfg.setdefault("client", {})
cfg.setdefault("dasd", {})

cfg["client"]["max_request_num"] = reqs
cfg["client"]["max_new_tokens"] = max_new

if variant == "baseline":
    cfg["mode"] = "specedge"
    cfg["dasd"]["enable_async"] = False
elif variant == "dasd_fixed":
    cfg["mode"] = "dasd"
    cfg["dasd"]["enable_async"] = True
    cfg["dasd"]["start_window"] = 8
    cfg["dasd"]["W_min"] = 8
    cfg["dasd"]["W_max"] = 8
    cfg["dasd"]["max_inflight_bundles"] = max(2, int(cfg["dasd"].get("max_inflight_bundles", 2)))
elif variant == "dasd_adaptive":
    cfg["mode"] = "dasd"
    cfg["dasd"]["enable_async"] = True
    cfg["dasd"]["start_window"] = 8
    cfg["dasd"]["W_min"] = 2
    cfg["dasd"]["W_max"] = 32
    cfg["dasd"]["max_inflight_bundles"] = max(2, int(cfg["dasd"].get("max_inflight_bundles", 2)))
else:
    raise ValueError(f"Unknown variant: {variant}")

base_exp_name = cfg["base"].get("exp_name", "specedge")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
cfg["base"]["exp_name"] = f"{base_exp_name}_{variant}_rtt{rtt_label}_{ts}"

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(cfg["base"].get("result_path", "result/demo"))
print(cfg["base"]["exp_name"])
PY
)"

    local result_path exp_name raw_dir
    result_path="$(echo "$cfg_meta" | sed -n '1p')"
    exp_name="$(echo "$cfg_meta" | sed -n '2p')"
    raw_dir="$result_path/$exp_name"

    echo "[sweep] running variant=$variant rtt=$rtt_label config=$tmp_config"

    bash ./script/batch_server.sh -f "$tmp_config" &
    server_pid="$!"
    sleep 5

    bash ./script/client_host.sh -f "$tmp_config"

    kill -INT "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""

    local rtt_dir
    rtt_dir="$output_dir/rtt${rtt_label}"
    mkdir -p "$rtt_dir"

    local merged_client_jsonl
    merged_client_jsonl="$rtt_dir/${variant}.jsonl"
    : > "$merged_client_jsonl"

    local found_clients=0
    for cfile in "$raw_dir"/client_*.jsonl; do
        if [ -f "$cfile" ]; then
            cat "$cfile" >> "$merged_client_jsonl"
            found_clients=1
        fi
    done

    if [ "$found_clients" -eq 0 ]; then
        echo "[sweep] warning: no client_*.jsonl found in $raw_dir" >&2
    fi

    local server_jsonl
    server_jsonl="$rtt_dir/${variant}_server.jsonl"
    if [ -f "$raw_dir/server.jsonl" ]; then
        cp "$raw_dir/server.jsonl" "$server_jsonl"
    else
        : > "$server_jsonl"
        echo "[sweep] warning: no server.jsonl found in $raw_dir" >&2
    fi

    echo -e "${rtt_label}\t${variant}\t${merged_client_jsonl}\t${server_jsonl}\t${raw_dir}" >> "$manifest"

    rm -f "$tmp_config"

    echo "[sweep] done variant=$variant rtt=$rtt_label"
    echo "        client: $merged_client_jsonl"
    echo "        server: $server_jsonl"
}

for rtt in 14 50 100; do
    echo "[sweep] ==== RTT ${rtt} ===="
    apply_rtt_profile "$rtt"

    run_variant "$rtt" "baseline"
    run_variant "$rtt" "dasd_fixed"
    run_variant "$rtt" "dasd_adaptive"

    if [ -x "tools/netem.sh" ] && command -v tc >/dev/null 2>&1 && [ "$(uname -s)" = "Linux" ]; then
        IFACE="${iface:-}" tools/netem.sh clear ${iface:+-i "$iface"} || true
    fi
done

echo "[sweep] complete. Manifest: $manifest"
