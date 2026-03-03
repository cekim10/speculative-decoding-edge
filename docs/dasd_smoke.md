# DASD Smoke Test

## Run baseline vs DASD

Baseline (existing SpecEdge path):

```bash
bash ./script/batch_server.sh -f config/specedge.example.yaml
bash ./script/client_host.sh -f config/specedge.example.yaml
```

DASD smoke run (fixed `W=8`, inflight >= 2, small request count):

```bash
bash ./script/run_dasd_smoke.sh -f config/dasd.workstation.yaml -r 2 -n 64
```

The smoke script writes a temporary config and prints the output folder:

- `result_path/exp_name/client_*.jsonl`
- `result_path/exp_name/server.jsonl`

## Confirm multi-inflight from logs

In client JSONL, inspect `type=="dasd_bundle"` events.

Healthy multi-inflight signals:

1. `inflight_at_send >= 2` appears in a meaningful fraction of bundles.
2. Multiple bundle `send_ts` values occur before the first corresponding `recv_ts`.

Quick check with the summary script:

```bash
python tools/summarize_dasd_logs.py result/demo/<exp_name>/client_0.jsonl
```

Look at `% inflight>=2` and RTT percentiles.

## Acceptance sanity

For `r_obs` in `dasd_bundle`:

- Healthy: varies in `(0, 1]` over time.
- Suspicious:
  - Always `0.0` (likely base/index mismatch or severe protocol issue).
  - Always `1.0` (possible overfitting/trivial windowing in a specific test).

Also inspect `dasd_request_summary.goodput` and `rollbacks_count`.

## RTT sweep and netem

For automated baseline-vs-DASD sweeps across RTT profiles, see:

- [experiments.md](/Users/cekim/Desktop/specedge/docs/experiments.md)
