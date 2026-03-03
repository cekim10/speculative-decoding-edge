#!/usr/bin/env python3

import argparse
import json
import math
from datetime import datetime
from pathlib import Path


def percentile(values: list[float], p: float):
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def avg(values: list[float]):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def fmt_float(x: float):
    if math.isnan(x):
        return "n/a"
    return f"{x:.3f}"


def parse_timestamp(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def load_rows(jsonl_path: Path):
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
    return rows


def summarize_file(jsonl_path: Path):
    rows = load_rows(jsonl_path)
    if not rows:
        return {
            "file": str(jsonl_path),
            "mode": "empty",
            "throughput_tok_s": float("nan"),
            "goodput": float("nan"),
            "lat_p50_ms": float("nan"),
            "lat_p95_ms": float("nan"),
            "pct_inflight_ge2": float("nan"),
            "rtt_p50_ms": float("nan"),
            "rtt_p95_ms": float("nan"),
        }

    dasd_bundles = [r for r in rows if r.get("type") == "dasd_bundle"]
    dasd_summaries = [r for r in rows if r.get("type") == "dasd_request_summary"]

    if dasd_bundles or dasd_summaries:
        mode = "dasd"
        time_points = []
        for r in dasd_bundles + dasd_summaries:
            if isinstance(r.get("timestamp"), str):
                dt = parse_timestamp(r["timestamp"])
                if dt is not None:
                    time_points.append(dt)

        wall_s = float("nan")
        if len(time_points) >= 2:
            wall_s = (max(time_points) - min(time_points)).total_seconds()

        accepted_tokens = 0.0
        verified_tokens = 0.0
        if dasd_summaries:
            accepted_tokens = sum(
                float(r.get("total_accepted_tokens", 0.0)) for r in dasd_summaries
            )
            # verified in summary is per-request total; sum if present
            verified_tokens = sum(
                float(r.get("total_verified_tokens", 0.0)) for r in dasd_summaries
            )
        else:
            accepted_tokens = sum(float(r.get("accepted_len", 0.0)) for r in dasd_bundles)
            verified_tokens = sum(float(r.get("W", 0.0)) for r in dasd_bundles)

        throughput_tok_s = (
            accepted_tokens / wall_s if wall_s and wall_s > 0 else float("nan")
        )

        if dasd_summaries:
            goodput = avg(
                [
                    float(r.get("goodput"))
                    for r in dasd_summaries
                    if isinstance(r.get("goodput"), (int, float))
                ]
            )
            lat_values = [
                float(r.get("total_latency_ms"))
                for r in dasd_summaries
                if isinstance(r.get("total_latency_ms"), (int, float))
            ]
        else:
            goodput = (
                accepted_tokens / verified_tokens
                if verified_tokens > 0
                else float("nan")
            )
            lat_values = [
                float(r.get("rtt_ms"))
                for r in dasd_bundles
                if isinstance(r.get("rtt_ms"), (int, float))
            ]

        inflight_values = [
            float(r.get("inflight_at_send"))
            for r in dasd_bundles
            if isinstance(r.get("inflight_at_send"), (int, float))
        ]
        pct_inflight_ge2 = (
            100.0 * sum(1 for x in inflight_values if x >= 2.0) / len(inflight_values)
            if inflight_values
            else float("nan")
        )

        rtt_values = [
            float(r.get("rtt_ms"))
            for r in dasd_bundles
            if isinstance(r.get("rtt_ms"), (int, float))
        ]

        return {
            "file": str(jsonl_path),
            "mode": mode,
            "throughput_tok_s": throughput_tok_s,
            "goodput": goodput,
            "lat_p50_ms": percentile(lat_values, 50),
            "lat_p95_ms": percentile(lat_values, 95),
            "pct_inflight_ge2": pct_inflight_ge2,
            "rtt_p50_ms": percentile(rtt_values, 50),
            "rtt_p95_ms": percentile(rtt_values, 95),
        }

    # baseline SpecEdge client logs
    mode = "specedge"
    step_rows = [r for r in rows if isinstance(r.get("num_accepted_tokens"), (int, float))]

    time_points = []
    for r in step_rows:
        if isinstance(r.get("timestamp"), str):
            dt = parse_timestamp(r["timestamp"])
            if dt is not None:
                time_points.append(dt)

    wall_s = float("nan")
    if len(time_points) >= 2:
        wall_s = (max(time_points) - min(time_points)).total_seconds()

    accepted_tokens = sum(float(r.get("num_accepted_tokens", 0.0)) for r in step_rows)
    throughput_tok_s = accepted_tokens / wall_s if wall_s and wall_s > 0 else float("nan")

    # Request-level latency approximation from first to last per req_idx timestamp.
    by_req: dict[int, list[datetime]] = {}
    for r in step_rows:
        req_idx = r.get("req_idx")
        ts = r.get("timestamp")
        if not isinstance(req_idx, int) or not isinstance(ts, str):
            continue
        dt = parse_timestamp(ts)
        if dt is None:
            continue
        by_req.setdefault(req_idx, []).append(dt)

    req_lat_ms = []
    for dts in by_req.values():
        if len(dts) >= 2:
            req_lat_ms.append((max(dts) - min(dts)).total_seconds() * 1000.0)

    if not req_lat_ms:
        # fallback to per-step end-to-end latency if request boundaries are incomplete
        req_lat_ms = [
            float(r.get("draft", {}).get("end_to_end", 0.0))
            + float(r.get("target", {}).get("end_to_end", 0.0))
            for r in step_rows
            if isinstance(r.get("draft", {}).get("end_to_end"), (int, float))
            and isinstance(r.get("target", {}).get("end_to_end"), (int, float))
        ]

    return {
        "file": str(jsonl_path),
        "mode": mode,
        "throughput_tok_s": throughput_tok_s,
        "goodput": float("nan"),
        "lat_p50_ms": percentile(req_lat_ms, 50),
        "lat_p95_ms": percentile(req_lat_ms, 95),
        "pct_inflight_ge2": float("nan"),
        "rtt_p50_ms": float("nan"),
        "rtt_p95_ms": float("nan"),
    }


def print_table(summaries: list[dict]):
    headers = [
        "file",
        "mode",
        "throughput_tok/s",
        "goodput",
        "lat_p50_ms",
        "lat_p95_ms",
        "%inflight>=2",
        "rtt_p50_ms",
        "rtt_p95_ms",
    ]

    rows = []
    for s in summaries:
        rows.append(
            [
                s["file"],
                s["mode"],
                fmt_float(s["throughput_tok_s"]),
                fmt_float(s["goodput"]),
                fmt_float(s["lat_p50_ms"]),
                fmt_float(s["lat_p95_ms"]),
                fmt_float(s["pct_inflight_ge2"]),
                fmt_float(s["rtt_p50_ms"]),
                fmt_float(s["rtt_p95_ms"]),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, col in enumerate(row):
            widths[i] = max(widths[i], len(col))

    def fmt_row(cols):
        return " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def main():
    parser = argparse.ArgumentParser(
        description="Summarize SpecEdge/DASD client JSONL logs for quick comparison."
    )
    parser.add_argument("jsonl_paths", nargs="+", type=Path, help="One or more JSONL files")
    args = parser.parse_args()

    summaries = []
    for path in args.jsonl_paths:
        if not path.exists():
            print(f"warning: missing file skipped: {path}")
            continue
        summaries.append(summarize_file(path))

    if not summaries:
        raise FileNotFoundError("No valid input JSONL files were found.")

    print_table(summaries)


if __name__ == "__main__":
    main()
