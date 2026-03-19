#!/usr/bin/env python
"""Stage-separated profiling: collect prefill (EXTEND) and decode traces from each request.

Each inference request produces **two** separate traces via SGLang's
``profile_by_stage`` API:
  - ``<ts>-TP-<rank>-EXTEND.trace.json.gz``  (prefill phase)
  - ``<ts>-TP-<rank>-DECODE.trace.json.gz``  (decode phase)

Request parameters
------------------
``--input-len``
    Number of **new** prefill tokens per request.  These are the tokens
    actually processed during the EXTEND phase.
``--existing-ctx``
    Number of tokens already present in KV cache (radix cache hit).
    A seed request populates the cache before profiling.  Set to 0
    for cold prefill (no cache hit).  Default: 0.
``--bs``
    Batch size — number of prompts included in one batched profiling request.
``--decode-tokens``
    Number of decode tokens to generate (and decode batches to profile).
    The profiler captures 1 EXTEND batch and ``decode_tokens`` DECODE
    batches.  Default: 32.

    .. note:: SGLang's profiler uses a ``count > target`` stop condition
       (fencepost), so we pass ``num_steps = decode_tokens - 1`` to
       capture exactly ``decode_tokens`` batches.

Workflow
--------
1. Launch or connect to a running SGLang server.
2. Send warmup requests so CUDA graphs are captured *before* profiling.
3. (if existing_ctx > 0) Flush cache, send a seed request to populate KV cache.
4. Call ``/start_profile`` with ``profile_by_stage=True``.
5. Send one batched inference request with batch size *bs*.
6. The profiler automatically stops after 1 EXTEND + ``decode_tokens`` DECODE
   batches and writes both traces.  (Internally ``num_steps = decode_tokens - 1``
   because the profiler stop condition is ``count > target``.)
7. Parse the resulting traces with ``run_parse.py``.

Modes
-----
Use ``--collect {perf,shapes,all}`` to choose what to collect:

- ``perf``    — collect traces for a single (bs, input_len, existing_ctx) point,
              parse, and run cross-rank analysis.
- ``shapes``  — re-collect without CUDA graph to capture kernel input shapes,
              then merge shapes into the timing CSVs (both EXTEND and DECODE).
- ``all``     — run perf, auto-restart the server, then run shapes.

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
      --collect perf \\
      --host 0.0.0.0 --port 30001 \\
      --bs 1 --input-len 2048 --decode-tokens 32 \\
      --output-dir /flowsim/stage_traces

Example — with existing KV cache context
  python scripts/run_stage_profile.py \\
      --collect perf \\
      --host 0.0.0.0 --port 30001 \\
      --bs 4 --input-len 512 --existing-ctx 4096 --decode-tokens 32 \\
      --output-dir /flowsim/stage_traces

Example — launch server + full pipeline (perf → shapes)
  python scripts/run_stage_profile.py \\
      --collect all \\
      --launch-server \\
      --server-opts "--model-path Qwen/Qwen3-235B-A22B-FP8 --tp 4 --host 0.0.0.0 --port 30001" \\
      --bs 1 --input-len 2048 \\
      --output-dir /flowsim/stage_traces
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Optional

# Add project root to path so utils package is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.cross_rank_agg import (
    aggregate as analyze_traces_from_csvs,
    print_result as print_analysis,
)
from utils.net import wait_for_port
from utils.shape_merge import merge_shapes_dir
from scripts import load_sweep_file, parse_sweep_point

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_WARMUP_N = 5
DEFAULT_DECODE_TOKENS = 32
DEFAULT_MAX_PREFILL_TOKENS = 131072


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sample_token_ids(
    n: int, vocab_size: int = 32000, seed: int = 42
) -> list[int]:
    """Sample *n* random token IDs from ``[0, vocab_size)``.

    Uses a fixed seed so prompts are deterministic across runs,
    ensuring radix-cache prefix matching works correctly.
    """
    rng = random.Random(seed)
    return [rng.randint(0, vocab_size - 1) for _ in range(n)]


# ---------------------------------------------------------------------------
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
def flush_cache(host: str, port: int) -> bool:
    """Flush the server's radix cache and KV cache via ``/flush_cache``.

    Must be called when no requests are in flight.  Returns True on success.
    """
    url = f"http://{host}:{port}/flush_cache"
    try:
        import urllib.request

        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            ok = resp.status == 200
            if ok:
                print("[flush] Cache flushed ✓")
            else:
                print(f"[flush] Flush returned {resp.status}: {body}")
            return ok
    except Exception as exc:
        print(f"[flush] FAILED: {exc}")
        return False


def warmup(host: str, port: int, n: int, bs: int, ctx: int) -> None:
    """Send *n* short requests to trigger CUDA graph capture before profiling.

    Uses explicit ``input_ids`` with a deterministic pattern so warmup
    shapes are well defined and independent of text tokenization.
    """
    url = f"http://{host}:{port}/generate"
    warmup_len = max(1, min(ctx, DEFAULT_MAX_PREFILL_TOKENS))
    token_ids = _sample_token_ids(warmup_len)
    print(
        f"[warmup] Sending {n} warmup requests (bs={bs}, tokens={warmup_len}) …"
    )
    for i in range(n):
        payload = {
            "input_ids": token_ids,
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
    num_steps: int = 1,
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


def collect_one_prefill(
    host: str,
    port: int,
    bs: int,
    input_len: int,
    existing_ctx: int,
    decode_tokens: int,
    output_dir: str,
    warmup_n: int,
    num_steps: int,
) -> tuple[list[str], bool]:
    """Collect traces for a single (bs, input_len, existing_ctx) point.

    Controls the exact prefill workload:

    - ``input_len`` new tokens are prefilled (EXTEND).
    - ``existing_ctx`` tokens are already in KV cache (radix cache hit).

    Returns ``(traces, ok)`` where *ok* is False if OOM or fatal error
    occurred (caller should skip larger batch sizes).

    Protocol:

    1. ``/flush_cache`` — clear radix + KV cache.
    2. (if existing_ctx > 0) Send a seed request to populate the radix
       cache with ``existing_ctx`` tokens.
    3. ``/start_profile`` — arm the profiler.
    4. Send *bs* concurrent profiling requests, each with
       ``existing_ctx + input_len`` tokens.  The prefix hits cache;
       EXTEND processes only ``input_len`` new tokens per request.
    5. Wait for traces.

    For ``existing_ctx == 0`` (cold prefill), step 2 is skipped.
    """
    os.makedirs(output_dir, exist_ok=True)
    total_prompt = existing_ctx + input_len

    # Token-exact prompts via random token IDs from vocabulary.
    # Using fixed seeds ensures deterministic prompts so radix-cache
    # prefix matching works across seed and profile requests.
    seed_ids: list[int] = (
        _sample_token_ids(existing_ctx, seed=42) if existing_ctx > 0 else []
    )
    profile_ids: list[int] = _sample_token_ids(
        existing_ctx, seed=42
    ) + _sample_token_ids(input_len, seed=123)

    url = f"http://{host}:{port}/generate"

    # ── Step 0: warmup ──
    warmup(host, port, n=warmup_n, bs=1, ctx=max(total_prompt, 2048))

    # ── Step 1: flush cache ──
    flush_cache(host, port)
    time.sleep(1)

    # ── Step 2: seed prefix (if existing_ctx > 0) ──
    if existing_ctx > 0:
        print(f"[prefill] Seeding existing_ctx={existing_ctx} tokens …")
        payload_seed = {
            "input_ids": seed_ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        }
        try:
            _post(url, payload_seed, timeout=300)
            print("[prefill] Seed request done ✓")
        except Exception as exc:
            print(f"[prefill] Seed FAILED: {exc}")
            return [], False
        time.sleep(0.5)

    # ── Step 3: start profiler ──
    if not start_stage_profile(host, port, output_dir, num_steps):
        print("[ERROR] Could not start profiler — skipping")
        return [], True  # profiler issue, not OOM

    # ── Step 4: send bs profiling requests as a single batch ──
    print(
        f"[prefill] bs={bs}  input_len={input_len}  "
        f"existing_ctx={existing_ctx}  total_tokens={bs * total_prompt}"
    )

    if bs == 1:
        batch_ids = profile_ids
    else:
        batch_ids = [profile_ids] * bs

    payload_profile = {
        "input_ids": batch_ids,
        "sampling_params": {
            "max_new_tokens": decode_tokens,
            "temperature": 0.8,
            "ignore_eos": True,
        },
    }

    oom_detected = False
    try:
        _post(url, payload_profile, timeout=600)
        print(f"[prefill] Batch of {bs} done ✓")
    except Exception as exc:
        exc_str = str(exc).lower()
        if "out of memory" in exc_str or "oom" in exc_str:
            print(f"[prefill] OOM at bs={bs} — stopping")
            oom_detected = True
        else:
            print(f"[prefill] Profile request FAILED: {exc}")
            return [], False

    if oom_detected:
        time.sleep(3)
        flush_cache(host, port)
        return [], False

    # ── Step 5: wait for traces ──
    print("[wait] Waiting for profiler to auto-stop …")
    traces = wait_for_traces(output_dir, timeout=120)
    prev_count = len(traces)
    for _ in range(6):
        time.sleep(2)
        new_traces = sorted(
            glob.glob(os.path.join(output_dir, "*.trace.json.gz"))
        )
        if len(new_traces) == prev_count:
            break
        traces = new_traces
        prev_count = len(traces)
    print(f"[done] {len(traces)} trace files in {output_dir}")
    for t in traces:
        sz = os.path.getsize(t) / 1024
        print(f"       {os.path.basename(t)}  ({sz:.1f} KB)")
    print()

    return traces, True


# ---------------------------------------------------------------------------
# Server launch (optional)
# ---------------------------------------------------------------------------
def launch_server(
    server_opts: str,
    log_dir: str,
    *,
    disable_cuda_graph: bool = False,
) -> subprocess.Popen:
    """Start an SGLang server process with profiling env vars.

    Parameters
    ----------
    disable_cuda_graph : bool
        If True, append ``--disable-cuda-graph --disable-cuda-graph-padding``
        to the server command.  Used for shape collection where the PyTorch
        profiler needs full kernel-level ``Input Dims`` info.
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = int(time.time())
    prefix = "shape_server" if disable_cuda_graph else "server"
    stdout_path = os.path.join(log_dir, f"{prefix}_{ts}.stdout.log")
    stderr_path = os.path.join(log_dir, f"{prefix}_{ts}.stderr.log")

    env = os.environ.copy()
    env["SGLANG_PROFILE_KERNELS"] = "1"

    args = shlex.split(server_opts)
    if disable_cuda_graph:
        if "--disable-cuda-graph" not in args:
            args.append("--disable-cuda-graph")
        if "--disable-cuda-graph-padding" not in args:
            args.append("--disable-cuda-graph-padding")

    cmd = [sys.executable, "-m", "sglang.launch_server"] + args
    label = "(no-CUDA-graph)" if disable_cuda_graph else ""
    print(f"[server] Launching {label}: {' '.join(cmd)}")
    preexec = getattr(os, "setsid", None)
    stdout_f = open(stdout_path, "w")
    stderr_f = open(stderr_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=stdout_f,
        stderr=stderr_f,
        preexec_fn=preexec,
        env=env,
    )
    # Attach file handles so kill_server can close them.
    proc._log_files = (stdout_f, stderr_f)  # type: ignore[attr-defined]
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
    for fh in getattr(proc, "_log_files", ()):
        try:
            fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Analyze: thin wrappers around cross_rank_agg module
# ---------------------------------------------------------------------------
def analyze_traces(
    trace_dir: str,
    parse_output_dir: str,
    stage: str = "DECODE",
) -> dict:
    """Aggregate kernel stats across TP ranks for one (bs, ctx, stage).

    Delegates to ``cross_rank_agg.aggregate()`` with auto-parse fallback.
    """
    csvs = sorted(glob.glob(os.path.join(parse_output_dir, f"*{stage}*.csv")))
    if not csvs:
        parse_traces(trace_dir, parse_output_dir)
        csvs = sorted(
            glob.glob(os.path.join(parse_output_dir, f"*{stage}*.csv"))
        )
    if not csvs:
        print(f"[analyze] No {stage} CSVs found in {parse_output_dir}")
        return {}
    return analyze_traces_from_csvs(csv_files=csvs, stage=stage)


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
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--trace-file",
                t,
                "--output-dir",
                parse_output_dir,
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"[parse] FAILED for {os.path.basename(t)} "
                f"(exit {result.returncode}):\n{result.stderr[-2000:]}"
            )


# ---------------------------------------------------------------------------
# Shape collection (no-CUDA-graph pass)
# ---------------------------------------------------------------------------
_SUBDIR_RE = re.compile(r"^bs(\d+)_input(\d+)_ctx(\d+)$")


def discover_subdirs(sweep_dir: str) -> list[tuple[str, int, int, int]]:
    """Discover profiling subdirectories created by ``_run_perf``.

    Returns a sorted list of ``(dirname, bs, input_len, existing_ctx)``
    tuples for directories matching ``bs{N}_input{M}_ctx{K}``.
    """
    results = []
    for entry in sorted(os.listdir(sweep_dir)):
        m = _SUBDIR_RE.match(entry)
        if m and os.path.isdir(os.path.join(sweep_dir, entry)):
            results.append(
                (entry, int(m.group(1)), int(m.group(2)), int(m.group(3)))
            )
    return results


def collect_shapes(
    host: str,
    port: int,
    sweep_dir: str,
    *,
    decode_tokens: int = DEFAULT_DECODE_TOKENS,
    warmup_n: int = 3,
    num_steps: int = 1,
) -> list[str]:
    """Run a shape-only profiling pass for all points in the sweep.

    Collects traces into ``<sweep>/<subdir>/shape_traces/`` and parses them
    into ``<sweep>/<subdir>/shape_parsed/``.  Shape data is needed to map
    kernel names to tensor dimensions (unavailable when CUDA graph is active).

    Uses ``collect_one_prefill`` (exact token counts via ``input_ids``) so
    that kernel shapes match the timing pass exactly.
    """
    subdirs = discover_subdirs(sweep_dir)
    if not subdirs:
        print(f"[shapes] No bs*_input*_ctx* dirs found in {sweep_dir}")
        return []

    print(f"[shapes] Collecting shapes for {len(subdirs)} points")
    print(f"[shapes] Dirs: {[s[0] for s in subdirs]}\n")

    results = []
    for i, (tag, bs, input_len, existing_ctx) in enumerate(subdirs):
        trace_dir = os.path.join(sweep_dir, tag, "shape_traces")
        parse_dir = os.path.join(sweep_dir, tag, "shape_parsed")

        # Skip if shapes already collected for both stages
        has_decode = glob.glob(os.path.join(parse_dir, "*DECODE*.csv"))
        has_extend = glob.glob(os.path.join(parse_dir, "*EXTEND*.csv"))
        if has_decode and has_extend:
            print(f"[{i+1}/{len(subdirs)}] {tag}: shape CSVs exist, skipping")
            results.append(parse_dir)
            continue

        print(f"[{i+1}/{len(subdirs)}] {tag}: collecting shape traces …")

        traces, ok = collect_one_prefill(
            host=host,
            port=port,
            bs=bs,
            input_len=input_len,
            existing_ctx=existing_ctx,
            decode_tokens=decode_tokens,
            output_dir=trace_dir,
            warmup_n=warmup_n,
            num_steps=num_steps,
        )

        if not ok:
            print(f"  [WARN] OOM or error for {tag}")
            continue

        if traces:
            parse_traces(trace_dir, parse_dir)
            results.append(parse_dir)

    return results


def merge_shapes(sweep_dir: str, stage: str = "DECODE") -> list[str]:
    """Merge shape CSVs into timing CSVs for every point in the sweep."""
    subdirs = discover_subdirs(sweep_dir)
    all_merged: list[str] = []
    for tag, _bs, _il, _ec in subdirs:
        timing_dir = os.path.join(sweep_dir, tag, "parsed")
        shape_dir = os.path.join(sweep_dir, tag, "shape_parsed")
        merged_dir = os.path.join(sweep_dir, tag, "merged")
        if not os.path.isdir(timing_dir):
            print(f"[merge] {tag}: no timing parsed dir — skipping")
            continue
        if not os.path.isdir(shape_dir):
            print(f"[merge] {tag}: no shape parsed dir — skipping")
            continue
        print(f"[merge] {tag} …")
        merged = merge_shapes_dir(
            timing_dir,
            shape_dir,
            merged_dir,
            stage=stage,
            verbose=True,
        )
        all_merged.extend(merged)
    print(f"\n[merge] Total: {len(all_merged)} merged CSVs")
    return all_merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-separated profiling (prefill vs decode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = p.add_argument_group("collection mode")
    mode.add_argument(
        "--collect",
        choices=["perf", "shapes", "all"],
        required=True,
        help=(
            "Collection mode.\n"
            "  perf    — trace sweep (bs, ctx) + parse + analyze\n"
            "  shapes  — shape-only pass (no CUDA graph) + merge into timing CSVs\n"
            "  all     — perf, then auto-restart server, then shapes + merge\n"
        ),
    )

    conn = p.add_argument_group("connection")
    conn.add_argument("--host", default="0.0.0.0")
    conn.add_argument("--port", type=int, default=30001)

    wl = p.add_argument_group("workload")
    wl.add_argument("--bs", type=int, default=1, help="Batch size")
    wl.add_argument(
        "--input-len",
        type=int,
        default=2048,
        help="Number of new prefill tokens per request (EXTEND)",
    )
    wl.add_argument(
        "--existing-ctx",
        type=int,
        default=0,
        help="Number of tokens already in KV cache (0 = cold prefill)",
    )
    wl.add_argument(
        "--decode-tokens",
        type=int,
        default=DEFAULT_DECODE_TOKENS,
        help=(
            "Number of decode tokens to generate per request (>= 2). "
            "Also controls how many decode batches the profiler captures."
        ),
    )
    wl.add_argument(
        "--warmup-n",
        type=int,
        default=DEFAULT_WARMUP_N,
        help="Number of warmup requests before profiling",
    )
    wl.add_argument(
        "--disable-chunked-prefill",
        action="store_true",
        help="Add --chunked-prefill-size -1 to server opts to disable chunking",
    )
    wl.add_argument(
        "--max-prefill-tokens",
        type=int,
        default=DEFAULT_MAX_PREFILL_TOKENS,
        help="Max tokens per prefill batch (used by server config)",
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "--output-dir",
        default="/flowsim/stage_traces",
        help="Root directory for trace output",
    )

    sweep = p.add_argument_group("sweep (multi-point profiling)")
    sweep.add_argument(
        "--sweep",
        type=str,
        nargs="+",
        default=[],
        metavar="BS:INPUT_LEN:CTX",
        help=(
            "Profile multiple (bs, input_len, existing_ctx) points in one job. "
            "Each value is a colon-separated tuple, e.g. --sweep 1:2048:0 4:8192:0 16:2048:4096. "
            "Overrides --bs, --input-len, --existing-ctx."
        ),
    )
    sweep.add_argument(
        "--sweep-file",
        type=str,
        default="",
        metavar="FILE",
        help=(
            "Read sweep points from a file (one BS:INPUT_LEN:CTX per line, "
            "# comments allowed). Overrides --bs, --input-len, --existing-ctx."
        ),
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
        default="",
        help="Directory for server logs (default: {output-dir}/logs/)",
    )

    return p.parse_args(argv)


def _load_sweep_points(args) -> list[tuple[int, int, int]]:
    """Resolve sweep points from --sweep, --sweep-file, or single-point args."""
    if args.sweep and args.sweep_file:
        print("[ERROR] --sweep and --sweep-file are mutually exclusive")
        raise SystemExit(1)

    if args.sweep:
        return [parse_sweep_point(s) for s in args.sweep]
    if args.sweep_file:
        return load_sweep_file(args.sweep_file)
    # Single-point from --bs / --input-len / --existing-ctx
    return [(args.bs, args.input_len, args.existing_ctx)]


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------
def _start_server(
    args, *, disable_cuda_graph: bool = False
) -> subprocess.Popen:
    """Launch server, wait for readiness, return Popen handle."""
    if not args.server_opts:
        print("[ERROR] --launch-server requires --server-opts")
        raise SystemExit(1)
    server_opts = args.server_opts
    # Disable chunked prefill for saturation testing
    if getattr(args, "disable_chunked_prefill", False):
        max_pt = getattr(args, "max_prefill_tokens", DEFAULT_MAX_PREFILL_TOKENS)
        if "--chunked-prefill-size" not in server_opts:
            server_opts += (
                f" --chunked-prefill-size -1"
                f" --max-prefill-tokens {max_pt}"
                f" --mem-fraction-static 0.80"
            )
            print(
                f"[server] Chunked prefill disabled"
                f" (size=-1, max_prefill={max_pt}, mem_frac=0.80)"
            )
    proc = launch_server(
        server_opts,
        args.log_dir,
        disable_cuda_graph=disable_cuda_graph,
    )
    print(f"[server] Waiting for {args.host}:{args.port} …")
    if not wait_for_port(args.host, args.port, timeout=600):
        print("[ERROR] Server did not start within timeout")
        kill_server(proc)
        raise SystemExit(1)
    print("[server] Ready.\n")
    return proc


def _run_perf(
    args,
    summary: list[dict],
    *,
    bs: Optional[int] = None,
    input_len: Optional[int] = None,
    existing_ctx: Optional[int] = None,
) -> int:
    """Collect traces for a single (bs, input_len, existing_ctx, decode_tokens) point."""
    bs = bs if bs is not None else args.bs
    input_len = input_len if input_len is not None else args.input_len
    existing_ctx = (
        existing_ctx if existing_ctx is not None else args.existing_ctx
    )

    tag = f"bs{bs}_input{input_len}_ctx{existing_ctx}"
    sub_dir = os.path.join(args.output_dir, tag)
    print(
        f"{'=' * 60}\n"
        f"bs={bs}  input_len={input_len}  existing_ctx={existing_ctx}  "
        f"decode_tokens={args.decode_tokens}\n"
        f"{'=' * 60}"
    )

    # collect_one_prefill uses input_ids for exact token count control.
    # SGLang's profiler stops when batch_count > num_steps (not >=),
    # so num_steps=N actually requires N+1 batches.  To capture exactly
    # decode_tokens decode batches we pass num_steps = decode_tokens - 1.
    traces, ok = collect_one_prefill(
        host=args.host,
        port=args.port,
        bs=bs,
        input_len=input_len,
        existing_ctx=existing_ctx,
        decode_tokens=args.decode_tokens,
        output_dir=sub_dir,
        warmup_n=args.warmup_n,
        num_steps=max(1, args.decode_tokens - 1),
    )
    if not ok:
        print("[ERROR] OOM during profiling")
        return 1

    summary.append(
        {
            "bs": bs,
            "input_len": input_len,
            "existing_ctx": existing_ctx,
            "traces": len(traces),
            "dir": sub_dir,
        }
    )

    if traces:
        parse_dir = os.path.join(sub_dir, "parsed")
        parse_traces(sub_dir, parse_dir)

        for stage in ("EXTEND", "DECODE"):
            result = analyze_traces(sub_dir, parse_dir, stage=stage)
            if result:
                print_analysis(result)
                analysis_path = os.path.join(
                    sub_dir, f"analysis_{stage.lower()}.json"
                )
                with open(analysis_path, "w") as af:
                    json.dump(result, af, indent=2)
                summary[-1][f"{stage.lower()}_total_ms"] = round(
                    result["total_kernel_us"] / 1000, 2
                )
    return 0


def _run_shapes(args) -> int:
    """Collect shapes (no-CUDA-graph pass) and merge into timing CSVs."""
    sweep_dir = args.output_dir
    print(f"\n{'=' * 60}")
    print(f"  SHAPE COLLECTION  (sweep_dir={sweep_dir})")
    print(f"{'=' * 60}\n")

    collect_shapes(
        args.host,
        args.port,
        sweep_dir,
        decode_tokens=args.decode_tokens,
        warmup_n=max(
            2, args.warmup_n // 2
        ),  # less warmup needed without CUDA graph
        num_steps=max(1, args.decode_tokens - 1),
    )
    merge_shapes(sweep_dir, stage="DECODE")
    merge_shapes(sweep_dir, stage="EXTEND")
    return 0


def _write_summary(args, summary: list[dict]) -> None:
    """Write sweep summary JSON and print a table."""
    if not summary:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[summary] {summary_path}")
    for s in summary:
        status = "✓" if s["traces"] > 0 else ("⊘" if s.get("skipped") else "✗")
        if "input_len" in s:
            print(
                f"  {status} bs={s.get('bs', 1):>4}  "
                f"input={s['input_len']:>5}  "
                f"ctx={s['existing_ctx']:>6}  "
                f"traces={s['traces']}"
                + (f"  ({s['skipped']})" if s.get("skipped") else "")
            )
        else:
            print(
                f"  {status} bs={s['bs']:>4}  ctx={s['ctx']:>6}  "
                f"traces={s['traces']}"
            )


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    # Default log_dir to {output_dir}/logs/ if not specified
    if not args.log_dir:
        args.log_dir = os.path.join(args.output_dir, "logs")

    if args.decode_tokens < 2:
        print(
            "[ERROR] --decode-tokens must be >= 2. "
            "SGLang's profiler uses a count > target stop condition, "
            "so decode_tokens=1 would capture 2 decode batches."
        )
        return 1

    server_proc = None
    summary: list[dict] = []
    sweep_points = _load_sweep_points(args)
    is_sweep = len(sweep_points) > 1

    if is_sweep:
        print(f"\n[sweep] {len(sweep_points)} points to profile:")
        for i, (bs, il, ctx) in enumerate(sweep_points):
            print(f"  [{i+1}] bs={bs}  input_len={il}  existing_ctx={ctx}")
        print()

    try:
        # ==================================================================
        # --collect all: perf → restart server → shapes → merge
        # ==================================================================
        if args.collect == "all":
            if not args.launch_server:
                print(
                    "[ERROR] --collect all requires --launch-server "
                    "(server must be restarted without CUDA graph for shape pass).\n"
                    "Run separately:\n"
                    "  --collect perf   (with normal server)\n"
                    "  --collect shapes (with --disable-cuda-graph server)"
                )
                return 1

            # Phase 1: perf
            print("\n" + "=" * 60)
            print("  PHASE 1 / 2 : PERF COLLECTION")
            print("=" * 60 + "\n")
            server_proc = _start_server(args, disable_cuda_graph=False)
            for idx, (bs, il, ctx) in enumerate(sweep_points):
                if is_sweep:
                    print(f"\n[sweep] Point {idx+1}/{len(sweep_points)}")
                _run_perf(args, summary, bs=bs, input_len=il, existing_ctx=ctx)
            _write_summary(args, summary)
            print("\n[server] Shutting down for shape pass …")
            kill_server(server_proc)
            server_proc = None
            time.sleep(5)

            # Phase 2: shapes
            print("\n" + "=" * 60)
            print("  PHASE 2 / 2 : SHAPE COLLECTION")
            print("=" * 60 + "\n")
            server_proc = _start_server(args, disable_cuda_graph=True)
            _run_shapes(args)
            return 0

        # ==================================================================
        # --collect perf
        # ==================================================================
        if args.collect == "perf":
            if args.launch_server:
                server_proc = _start_server(args, disable_cuda_graph=False)
            for idx, (bs, il, ctx) in enumerate(sweep_points):
                if is_sweep:
                    print(f"\n[sweep] Point {idx+1}/{len(sweep_points)}")
                _run_perf(args, summary, bs=bs, input_len=il, existing_ctx=ctx)
            _write_summary(args, summary)
            return 0

        # ==================================================================
        # --collect shapes
        # ==================================================================
        if args.collect == "shapes":
            if args.launch_server:
                server_proc = _start_server(args, disable_cuda_graph=True)
            _run_shapes(args)
            return 0

        return 0  # unreachable (argparse enforces --collect)

    finally:
        if server_proc is not None:
            print("\n[server] Shutting down …")
            kill_server(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
