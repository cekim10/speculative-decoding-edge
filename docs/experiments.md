# RTT Sweep Experiments

## Netem interface selection approach

`tools/netem.sh` selects interface in this order:

1. `-i <iface>` argument
2. `IFACE` environment variable
3. default route interface auto-detection (`ip route` on Linux, fallback `route`)

In this current environment (`Darwin`), `tc` is unavailable.  
`tools/netem.sh` is Linux-only and will fail fast with a clear error on non-Linux hosts.

## Netem usage

Show current qdisc:

```bash
tools/netem.sh show -i eth0
```

Apply one-way delay:

```bash
tools/netem.sh apply -i eth0 -d 25
```

Clear:

```bash
tools/netem.sh clear -i eth0
```

Note: `-d` is one-way delay. Approximate RTT impact is about `2 * delay`.

## Run RTT sweep

```bash
bash scripts/run_rtt_sweep.sh -f config/dasd.workstation.yaml -r 4 -n 128 -o result/rtt_sweep
```

This runs:

- baseline SpecEdge (`mode=specedge`)
- DASD fixed window (`W_min=W_max=start_window=8`, inflight >= 2)
- DASD adaptive (`W_min=2`, `W_max=32`, `start_window=8`, inflight >= 2)

across RTT profiles:

- RTT14: no shaping
- RTT50: netem one-way delay 25ms
- RTT100: netem one-way delay 50ms

Outputs:

- merged client logs: `result/rtt_sweep/rtt{14|50|100}/{baseline|dasd_fixed|dasd_adaptive}.jsonl`
- server logs: `result/rtt_sweep/rtt{...}/{variant}_server.jsonl`
- manifest: `result/rtt_sweep/manifest.tsv`

## Quick comparison

```bash
python tools/summarize_dasd_logs.py \
  result/rtt_sweep/rtt14/baseline.jsonl \
  result/rtt_sweep/rtt14/dasd_fixed.jsonl \
  result/rtt_sweep/rtt14/dasd_adaptive.jsonl
```

The table includes throughput, goodput (when available), latency p50/p95, and `%inflight>=2` (DASD only).
