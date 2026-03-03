#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") [-f <config>] [-r <max_requests>] [-n <max_new_tokens>]" >&2
    echo "  -f <config>          Base config path (default: config/dasd.workstation.yaml)" >&2
    echo "  -r <max_requests>    Client max_request_num override (default: 2)" >&2
    echo "  -n <max_new_tokens>  Client max_new_tokens override (default: 64)" >&2
    echo "  -h                   Show help" >&2
}

config_file="config/dasd.workstation.yaml"
max_requests=2
max_new_tokens=64

while getopts "f:r:n:h" opt; do
    case "$opt" in
        f) config_file="$OPTARG" ;;
        r) max_requests="$OPTARG" ;;
        n) max_new_tokens="$OPTARG" ;;
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

tmp_config="$(mktemp /tmp/dasd_smoke.XXXXXX.yaml)"
server_pid=""

cleanup() {
    if [ -n "${server_pid:-}" ]; then
        kill -INT "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    rm -f "$tmp_config"
}
trap cleanup EXIT

config_meta="$(
python - "$config_file" "$tmp_config" "$max_requests" "$max_new_tokens" <<'PY'
import sys
from datetime import datetime

import yaml

src, dst, reqs, max_new = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

with open(src, "r") as f:
    cfg = yaml.safe_load(f)

cfg["mode"] = "dasd"
cfg.setdefault("dasd", {})
cfg["dasd"]["enable_async"] = True
cfg["dasd"]["start_window"] = 8
cfg["dasd"]["W_min"] = 8
cfg["dasd"]["W_max"] = 8
cfg["dasd"]["max_inflight_bundles"] = max(2, int(cfg["dasd"].get("max_inflight_bundles", 2)))
cfg["dasd"]["max_spec_buffer_tokens"] = int(cfg["dasd"].get("max_spec_buffer_tokens", 256))

cfg.setdefault("client", {})
cfg["client"]["max_request_num"] = reqs
cfg["client"]["max_new_tokens"] = max_new

cfg.setdefault("base", {})
base_exp_name = cfg["base"].get("exp_name", "specedge")
cfg["base"]["exp_name"] = f"{base_exp_name}_dasd_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(cfg["base"]["result_path"])
print(cfg["base"]["exp_name"])
PY
)"

result_path="$(echo "$config_meta" | sed -n '1p')"
exp_name="$(echo "$config_meta" | sed -n '2p')"

echo "Using temp smoke config: $tmp_config"
echo "Results will be saved under: $result_path/$exp_name"
echo "Starting DASD smoke run..."

bash ./script/batch_server.sh -f "$tmp_config" &
server_pid="$!"
sleep 5

bash ./script/client_host.sh -f "$tmp_config"

kill -INT "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
server_pid=""

echo "DASD smoke run complete."
echo "Client logs: $result_path/$exp_name/client_*.jsonl"
echo "Server logs: $result_path/$exp_name/server.jsonl"
