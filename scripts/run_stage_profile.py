#!/usr/bin/env python
"""Stage-separated profiling: collect prefill (EXTEND) and decode traces independently.

Uses SGLang's native `profile_by_stage` API to automatically split a single
inference request into two traces:
  - <ts>-TP-<rank>-EXTEND.trace.json.gz   (prefill phase)
  - <ts>-TP-<rank>-DECODE.trace.json.gz   (decode phase)

Workflow
--------
1. Launch or connect to a running SGLang server.
2. Send warmup requests so CUDA graphs are captured *before* profiling.
3. Call ``/start_profile`` with ``profile_by_stage=True, num_steps=1``.
4. Send a single inference request — the profiler automatically stops
   after 1 prefill batch + 1 decode batch.
5. Optionally parse the resulting traces with ``run_parse.py``.

Sweep mode
----------
With ``--sweep``, the script iterates over a grid of (batch_size, context_len)
and collects one (EXTEND + DECODE) trace pair per configuration.  Results are
organised into ``<output_dir>/<bs>_<ctx>/`` sub-directories.

Notes
-----
* The ``cross_device_reduce`` kernel contains a spin-wait barrier.  The rank
  that arrives last records only the true transfer time; earlier ranks include
  wait time.  When computing the real communication cost, use the **minimum**
  all_reduce time across ranks.
* For Mixture-of-Experts models the first few requests may trigger additional
  JIT compilation.  Increase ``--warmup-n`` if the first trace looks anomalous.

Example — single point
  python scripts/run_stage_profile.py \\
      --host 0.0.0.0 --port 30001 \\
      --bs 1 --ctx 2048 --decode-tokens 32 \\
      --output-dir /flowsim/stage_traces

Example — sweep
  python scripts/run_stage_profile.py \\
      --host 0.0.0.0 --port 30001 \\
      --sweep \\
      --output-dir /flowsim/stage_traces_sweep

Example — launch server + profile (all-in-one)
  python scripts/run_stage_profile.py \\
      --launch-server \\
      --server-opts "--model-path Qwen/Qwen3-235B-A22B-FP8 --tp 4 --host 0.0.0.0 --port 30001" \\
      --sweep \\
      --output-dir /flowsim/stage_traces_sweep
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BS_GRID = [1, 4, 16, 64, 256]
DEFAULT_CTX_GRID = [512, 2048, 8192, 32768]
DEFAULT_WARMUP_N = 5
DEFAULT_DECODE_TOKENS = 32
DEFAULT_NUM_STEPS = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wait_for_port(host: str, port: int, timeout: int = 600) -> bool:
    """Block until *host:port* accepts a TCP connection."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except Exception:
            time.sleep(2)
    return False


def _post(url: str, payload: dict, timeout: int = 300) -> dict | str:
    """Send a POST request and return JSON (dict) or plain text (str)."""
    import urllib.request

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return body.strip()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def warmup(host: str, port: int, n: int, bs: int, ctx: int) -> None:
    """Send *n* short requests to trigger CUDA graph capture before profiling."""
    url = f"http://{host}:{port}/generate"
    prompt = "Hello " * max(1, ctx // 2)
    print(f"[warmup] Sending {n} warmup requests (bs={bs}, ctx≈{ctx}) …")
    for i in range(n):
        payload = {
            "text": prompt,
            "sampling_params": {"max_new_tokens": 4, "temperature": 0},
        }
        try:
            _post(url, payload, timeout=120)
            print(f"  warmup {i + 1}/{n} ✓")
        except Exception as exc:
            print(f"  warmup {i + 1}/{n} FAILED: {exc}")
    print()


def start_stage_profile(
    host: str,
    port: int,
    output_dir: str,
    num_steps: int = DEFAULT_NUM_STEPS,
) -> bool:
    """Call ``/start_profile`` with ``profile_by_stage=True``."""
    url = f"http://{host}:{port}/start_profile"
    payload = {
        "profile_by_stage": True,
        "num_steps": num_steps,
        "output_dir": output_dir,
    }
    try:
        resp = _post(url, payload, timeout=30)
        msg = resp if isinstance(resp, str) else resp.get("message", str(resp))
        print(f"[profile] start → {msg}")
        # The endpoint returns plain text "Start profiling." on success
        if isinstance(resp, str):
            return "profil" in resp.lower() or "start" in resp.lower()
        return resp.get("success", True)
    except Exception as exc:
        print(f"[profile] start FAILED: {exc}")
        return False


def send_requests(
    host: str, port: int, bs: int, ctx: int, decode_tokens: int
) -> list[threading.Thread]:
    """Send *bs* inference requests **concurrently** with ~*ctx* input tokens.

    All requests are fired in parallel so the scheduler batches them together
    in a single EXTEND / DECODE step.  Returns a list of daemon threads that
    are still running (waiting for the server to finish generating).
    """
    url = f"http://{host}:{port}/generate"
    prompt = "Hello " * max(1, ctx // 2)
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": decode_tokens,
            "temperature": 0,
        },
    }

    if bs == 1:
        # Single request — keep it simple & synchronous
        print(f"[request] bs=1  ctx≈{ctx}  decode={decode_tokens}")
        try:
            resp = _post(url, payload, timeout=600)
            out_text = resp.get("text", "") if isinstance(resp, dict) else str(resp)
            print(f"  req 0: {len(out_text.split())} output tokens")
        except Exception as exc:
            print(f"  req 0: FAILED {exc}")
        return []

    # bs > 1 — fire all requests concurrently via threads
    print(f"[request] bs={bs}  ctx≈{ctx}  decode={decode_tokens}  (concurrent)")
    done_count = [0]  # mutable counter shared across threads
    lock = threading.Lock()

    def _send_one(_i: int) -> None:
        try:
            _post(url, payload, timeout=600)
        except Exception:
            pass
        with lock:
            done_count[0] += 1

    threads: list[threading.Thread] = []
    for i in range(bs):
        t = threading.Thread(target=_send_one, args=(i,), daemon=True)
        t.start()
        threads.append(t)
    print(f"  fired {bs} concurrent requests")
    return threads


def wait_for_traces(
    output_dir: str,
    timeout: int = 60,
    expect_extend: bool = True,
    expect_decode: bool = True,
) -> list[str]:
    """Wait for EXTEND/DECODE trace files to appear in *output_dir*."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        files = glob.glob(os.path.join(output_dir, "*.trace.json.gz"))
        has_ext = any("EXTEND" in f for f in files)
        has_dec = any("DECODE" in f for f in files)
        if (not expect_extend or has_ext) and (not expect_decode or has_dec):
            return sorted(files)
        time.sleep(2)
    return sorted(glob.glob(os.path.join(output_dir, "*.trace.json.gz")))


def collect_one(
    host: str,
    port: int,
    bs: int,
    ctx: int,
    decode_tokens: int,
    output_dir: str,
    warmup_n: int,
    num_steps: int,
) -> list[str]:
    """Collect one (EXTEND + DECODE) trace pair for a single (bs, ctx) point."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. warmup
    warmup(host, port, n=warmup_n, bs=bs, ctx=ctx)

    # 2. start profiler
    if not start_stage_profile(host, port, output_dir, num_steps):
        print("[ERROR] Could not start profiler — skipping this config")
        return []

    # 3. send inference request(s) — concurrently for bs > 1
    req_threads = send_requests(host, port, bs, ctx, decode_tokens)

    # 4. wait for traces (profiler auto-stops after num_steps batches)
    print("[wait] Waiting for profiler to auto-stop …")
    traces = wait_for_traces(output_dir, timeout=120)
    # Stabilisation wait: let slower DP workers export their traces
    prev_count = len(traces)
    for _ in range(6):  # up to 12s extra
        time.sleep(2)
        new_traces = sorted(glob.glob(os.path.join(output_dir, "*.trace.json.gz")))
        if len(new_traces) == prev_count:
            break
        traces = new_traces
        prev_count = len(traces)
    print(f"[done] {len(traces)} trace files in {output_dir}")
    for t in traces:
        sz = os.path.getsize(t) / 1024
        print(f"       {os.path.basename(t)}  ({sz:.1f} KB)")
    print()

    # 5. wait for in-flight requests to complete (avoid dangling connections)
    if req_threads:
        print(f"[cleanup] waiting for {len(req_threads)} in-flight requests …")
        for t in req_threads:
            t.join(timeout=300)
        print("[cleanup] done")

    return traces


# ---------------------------------------------------------------------------
# Server launch (optional)
# ---------------------------------------------------------------------------
def launch_server(server_opts: str, log_dir: str) -> subprocess.Popen:
    """Start an SGLang server process with profiling env vars."""
    os.makedirs(log_dir, exist_ok=True)
    ts = int(time.time())
    stdout_f = open(os.path.join(log_dir, f"server_{ts}.stdout.log"), "w")
    stderr_f = open(os.path.join(log_dir, f"server_{ts}.stderr.log"), "w")

    env = os.environ.copy()
    env["SGLANG_PROFILE_KERNELS"] = "1"

    args = shlex.split(server_opts)
    cmd = [sys.executable, "-m", "sglang.launch_server"] + args
    print(f"[server] Launching: {' '.join(cmd)}")
    preexec = getattr(os, "setsid", None)
    proc = subprocess.Popen(
        cmd,
        stdout=stdout_f,
        stderr=stderr_f,
        preexec_fn=preexec,
        env=env,
    )
    return proc


def kill_server(proc: subprocess.Popen) -> None:
    """Terminate a previously launched server process."""
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


# ---------------------------------------------------------------------------
# Kernel classification
# ---------------------------------------------------------------------------
_COMM_KEYWORDS = ("cross_device_reduce", "all_reduce", "all_gather", "ncclKernel", "ncclDev")


def _classify_kernel(name: str) -> str:
    """Map a CUDA kernel name to a human-readable category."""
    nl = name.lower()
    if any(k in nl for k in _COMM_KEYWORDS):
        if "all_gather" in nl or "AllGather" in name:
            return "all_gather"
        return "all_reduce"
    if "fused_moe" in nl:
        return "moe"
    if "deep_gemm" in nl or "fp8_gemm" in nl:
        return "gemm_fp8"
    if "flash" in nl or "sm90" in nl or "fmha" in nl:
        return "attention"
    if "nvjet" in nl or "splitk" in nl:
        return "nvjet_gemm"
    if "rmsnorm" in nl or "rms_norm" in nl or "fused_add_rmsnorm" in nl:
        return "rmsnorm"
    if "per_token" in nl or "quant" in nl:
        return "quantize"
    if "topk" in nl or "gating" in nl:
        return "topk_gating"
    if "moe_sum" in nl or "moe_align" in nl:
        return "moe_misc"
    return "other"


def _is_comm(cat: str) -> bool:
    return cat in ("all_reduce", "all_gather")


# ---------------------------------------------------------------------------
# Analyze: multi-rank aggregation with min-comm
# ---------------------------------------------------------------------------
def analyze_traces(
    trace_dir: str,
    parse_output_dir: str,
    stage: str = "DECODE",
) -> dict:
    """Aggregate kernel stats across TP ranks for one (bs, ctx, stage).

    For **communication** kernels (``all_reduce``, ``all_gather``), the
    ``cross_device_reduce`` kernel includes spin-wait synchronisation time
    on all ranks except the last to arrive.  Therefore we report the
    **minimum** total communication time across ranks as the true cost.

    For **compute** kernels the values are nearly identical across ranks,
    so we simply average them.

    Returns a dict::

        {
            "stage": "DECODE",
            "num_ranks": 4,
            "total_kernel_us": 10300.0,   # corrected total
            "categories": {
                "moe":        {"us": 2580, "pct": 25.0},
                "all_reduce": {"us":  880, "pct":  8.5, "method": "min-across-ranks"},
                ...
            },
            "per_rank_comm_us": [147320, 173700, 174020, 880],
        }
    """
    csvs = sorted(glob.glob(os.path.join(parse_output_dir, f"*{stage}*.csv")))
    if not csvs:
        # Try parsing first
        parse_traces(trace_dir, parse_output_dir)
        csvs = sorted(glob.glob(os.path.join(parse_output_dir, f"*{stage}*.csv")))
    if not csvs:
        print(f"[analyze] No {stage} CSVs found in {parse_output_dir}")
        return {}

    # Per-rank stats
    rank_stats: list[dict[str, float]] = []  # [{cat: total_us}, ...]
    for csv_path in csvs:
        cats: dict[str, float] = defaultdict(float)
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Name", "")
                dur = float(row.get("Duration (us)", 0))
                cat = _classify_kernel(name)
                cats[cat] += dur
        rank_stats.append(dict(cats))

    num_ranks = len(rank_stats)
    all_cats = sorted({c for s in rank_stats for c in s})

    # Collect per-rank communication totals
    per_rank_comm = []
    for s in rank_stats:
        per_rank_comm.append(
            sum(v for k, v in s.items() if _is_comm(k))
        )

    # Build result: min for comm, mean for compute
    result_cats: dict[str, dict] = {}
    for cat in all_cats:
        vals = [s.get(cat, 0) for s in rank_stats]
        if _is_comm(cat):
            chosen = min(vals)
            result_cats[cat] = {
                "us": round(chosen, 1),
                "method": "min-across-ranks",
                "all_ranks_us": [round(v, 1) for v in vals],
            }
        else:
            chosen = sum(vals) / num_ranks
            result_cats[cat] = {
                "us": round(chosen, 1),
                "method": "mean-across-ranks",
            }

    corrected_total = sum(c["us"] for c in result_cats.values())
    for cat, info in result_cats.items():
        info["pct"] = round(info["us"] / corrected_total * 100, 1) if corrected_total > 0 else 0

    return {
        "stage": stage,
        "num_ranks": num_ranks,
        "total_kernel_us": round(corrected_total, 1),
        "categories": dict(sorted(result_cats.items(), key=lambda x: -x[1]["us"])),
        "per_rank_comm_us": [round(v, 1) for v in per_rank_comm],
    }


def print_analysis(result: dict) -> None:
    """Pretty-print an analysis result dict."""
    if not result:
        return
    stage = result["stage"]
    total = result["total_kernel_us"]
    nr = result["num_ranks"]
    print(f"\n{'=' * 60}")
    print(f"  {stage}  (corrected, {nr} ranks, min-comm)")
    print(f"  Total kernel time: {total / 1000:.2f} ms")
    print(f"{'=' * 60}")
    print(f"  {'Category':>20}  {'Time(ms)':>9}  {'Pct':>6}  Method")
    print(f"  {'-' * 55}")
    for cat, info in result["categories"].items():
        ms = info["us"] / 1000
        pct = info["pct"]
        method = info.get("method", "")
        extra = ""
        if "all_ranks_us" in info:
            extra = f"  (ranks: {[round(v/1000, 2) for v in info['all_ranks_us']]})"
        print(f"  {cat:>20}  {ms:>9.2f}  {pct:>5.1f}%  {method}{extra}")
    print()


# ---------------------------------------------------------------------------
# Parse helpers (thin wrapper around run_parse.py)
# ---------------------------------------------------------------------------
def parse_traces(trace_dir: str, parse_output_dir: str) -> None:
    """Call ``scripts/run_parse.py`` for every trace in *trace_dir*."""
    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_parse.py"
    )
    if not os.path.exists(script):
        print(f"[parse] run_parse.py not found at {script} — skipping parse")
        return

    os.makedirs(parse_output_dir, exist_ok=True)
    traces = sorted(glob.glob(os.path.join(trace_dir, "*.trace.json.gz")))
    for t in traces:
        print(f"[parse] {os.path.basename(t)} …")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        subprocess.run(
            [
                sys.executable,
                script,
                "--trace-file",
                t,
                "--output-dir",
                parse_output_dir,
            ],
            env=env,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-separated profiling (prefill vs decode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    conn = p.add_argument_group("connection")
    conn.add_argument("--host", default="0.0.0.0")
    conn.add_argument("--port", type=int, default=30001)

    wl = p.add_argument_group("workload")
    wl.add_argument("--bs", type=int, default=1, help="Batch size")
    wl.add_argument("--ctx", type=int, default=2048, help="Approx input length")
    wl.add_argument(
        "--decode-tokens",
        type=int,
        default=DEFAULT_DECODE_TOKENS,
        help="Max new tokens per request",
    )
    wl.add_argument(
        "--warmup-n",
        type=int,
        default=DEFAULT_WARMUP_N,
        help="Number of warmup requests before profiling",
    )
    wl.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="Number of prefill + decode batches to capture (1 = one of each)",
    )

    sweep = p.add_argument_group("sweep")
    sweep.add_argument(
        "--sweep",
        action="store_true",
        help="Iterate over a grid of (bs, ctx) configurations",
    )
    sweep.add_argument(
        "--bs-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BS_GRID),
        help="Comma-separated batch sizes for sweep",
    )
    sweep.add_argument(
        "--ctx-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_CTX_GRID),
        help="Comma-separated context lengths for sweep",
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "--output-dir",
        default="/flowsim/stage_traces",
        help="Root directory for trace output",
    )
    out.add_argument(
        "--parse",
        action="store_true",
        help="Run run_parse.py on collected traces",
    )
    out.add_argument(
        "--analyze",
        action="store_true",
        help="Parse traces and show kernel breakdown with min-comm correction",
    )

    srv = p.add_argument_group("server launch (optional)")
    srv.add_argument(
        "--launch-server",
        action="store_true",
        help="Launch an SGLang server before profiling",
    )
    srv.add_argument(
        "--server-opts",
        type=str,
        default="",
        help="Server options (e.g. '--model-path Qwen/… --tp 4 --host 0.0.0.0 --port 30001')",
    )
    srv.add_argument(
        "--log-dir",
        default="/flowsim/tests/test-artifacts",
        help="Directory for server logs",
    )

    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    server_proc = None

    try:
        # Optionally launch server
        if args.launch_server:
            if not args.server_opts:
                print("[ERROR] --launch-server requires --server-opts")
                return 1
            server_proc = launch_server(args.server_opts, args.log_dir)
            print(f"[server] Waiting for {args.host}:{args.port} …")
            if not wait_for_port(args.host, args.port, timeout=600):
                print("[ERROR] Server did not start within timeout")
                return 1
            print("[server] Ready.\n")

        # Build workload grid
        if args.sweep:
            bs_list = [int(x) for x in args.bs_grid.split(",")]
            ctx_list = [int(x) for x in args.ctx_grid.split(",")]
        else:
            bs_list = [args.bs]
            ctx_list = [args.ctx]

        total = len(bs_list) * len(ctx_list)
        idx = 0
        summary: list[dict] = []

        for bs in bs_list:
            for ctx in ctx_list:
                idx += 1
                tag = f"bs{bs}_ctx{ctx}"
                sub_dir = os.path.join(args.output_dir, tag)
                print(
                    f"{'=' * 60}\n"
                    f"[{idx}/{total}] bs={bs}  ctx={ctx}\n"
                    f"{'=' * 60}"
                )
                traces = collect_one(
                    host=args.host,
                    port=args.port,
                    bs=bs,
                    ctx=ctx,
                    decode_tokens=args.decode_tokens,
                    output_dir=sub_dir,
                    warmup_n=args.warmup_n,
                    num_steps=args.num_steps,
                )
                summary.append(
                    {"bs": bs, "ctx": ctx, "traces": len(traces), "dir": sub_dir}
                )

                # Optionally parse and/or analyze
                if (args.parse or args.analyze) and traces:
                    parse_dir = os.path.join(sub_dir, "parsed")
                    parse_traces(sub_dir, parse_dir)

                if args.analyze and traces:
                    for stage in ("EXTEND", "DECODE"):
                        result = analyze_traces(sub_dir, parse_dir, stage=stage)
                        if result:
                            print_analysis(result)
                            # Save per-config analysis
                            analysis_path = os.path.join(
                                sub_dir, f"analysis_{stage.lower()}.json"
                            )
                            with open(analysis_path, "w") as af:
                                json.dump(result, af, indent=2)
                            summary[-1][f"{stage.lower()}_total_ms"] = round(
                                result["total_kernel_us"] / 1000, 2
                            )

        # Write summary
        summary_path = os.path.join(args.output_dir, "sweep_summary.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[summary] {summary_path}")
        for s in summary:
            status = "✓" if s["traces"] > 0 else "✗"
            print(f"  {status} bs={s['bs']:>4}  ctx={s['ctx']:>6}  traces={s['traces']}")

        return 0

    finally:
        if server_proc is not None:
            print("\n[server] Shutting down …")
            kill_server(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
