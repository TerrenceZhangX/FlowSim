#!/usr/bin/env python
"""Cross-rank aggregation for multi-GPU kernel profiling.

When profiling a model with tensor/data/expert parallelism, each rank produces
its own trace CSV.  This module aggregates kernel statistics across ranks with
the correct methodology:

- **Symmetric collectives** (``all_reduce``, ``all_gather``, ``reduce_scatter``):
  use **per-invocation minimum** across ranks.  These collectives transfer
  identical data volumes on every rank.  The ``cross_device_reduce`` kernel
  contains a spin-wait barrier — the rank that arrives last records only the
  true transfer time; earlier ranks include wait time.  Because the "fast
  rank" can rotate between invocations (profiling noise), we take the min
  duration for *each* invocation separately, then sum.

- **Asymmetric collectives** (``all_to_all``):
  use the **maximum** total time across ranks.  In EP with token dispatch,
  each rank sends/receives a different data volume depending on the MoE
  routing.  The collective blocks until the heaviest communicator finishes,
  so max is the true wall-clock cost.  All ranks arrive at the all-to-all
  at roughly the same time (gating is cheap), so barrier inflation is small.

- **Compute** kernels: use the **mean** across ranks (values are nearly
  identical since every rank performs the same computation).

Usage — Python API
------------------
    from utils.cross_rank_agg import aggregate, classify_kernel, print_result

    result = aggregate("path/to/parsed_csvs/", stage="DECODE")
    print_result(result)

    # Or pass explicit CSV files
    result = aggregate(csv_files=["rank0.csv", "rank1.csv", "rank2.csv", "rank3.csv"])

Usage — CLI
-----------
    python -m utils.cross_rank_agg --csv-dir parsed/ --stage DECODE
    python -m utils.cross_rank_agg --csv-dir parsed/ --stage DECODE --output-json analysis.json

    # Exclude communication kernels (NCCL / custom allreduce) for compute-only timing
    python -m utils.cross_rank_agg --csv-dir parsed/ --stage EXTEND --compute-only
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Optional

# ---------------------------------------------------------------------------
# Kernel classification
# ---------------------------------------------------------------------------
# NOTE: all keywords are **lowercase** — compared against name.lower()
_COMM_KEYWORDS = (
    "cross_device_reduce",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "reducescatter",
    "ncclkernel",
    "nccldev",
    "alltoall",
    "all_to_all",
)


def classify_kernel(
    name: str,
    op: str = "",
    source: str = "",
    callstack: str = "",
) -> str:
    """Map a CUDA kernel name to a human-readable category.

    Primary classification uses the kernel *name*.  When that yields
    ``"other"``, optional CSV columns (*op*, *source*, *callstack*) are
    consulted for a second-pass classification.

    Categories
    ----------
    Communication: ``all_reduce``, ``all_gather``, ``reduce_scatter``, ``all_to_all``
    Compute: ``moe``, ``gemm_fp8``, ``attention``, ``nvjet_gemm``,
    ``rmsnorm``, ``quantize``, ``topk_gating``, ``moe_misc``, ``sampler``,
    ``copy``, ``dp_gather``, ``embedding``, ``other``
    """
    nl = name.lower()

    # ---- Communication (NCCL + custom collectives) ----
    if any(k in nl for k in _COMM_KEYWORDS):
        if "alltoall" in nl or "all_to_all" in nl:
            return "all_to_all"
        if "allgather" in nl or "all_gather" in nl:
            return "all_gather"
        if "reducescatter" in nl or "reduce_scatter" in nl:
            return "reduce_scatter"
        return "all_reduce"
    if "all_reduce" in nl:
        return "all_reduce"
    if "all_gather" in nl:
        return "all_gather"
    if "alltoall" in nl or "all_to_all" in nl:
        return "all_to_all"

    # ---- Core compute kernels ----
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
    if "moe_sum" in nl or "moe_align" in nl or "expert_tokens" in nl:
        return "moe_misc"

    # ---- Name-based secondary patterns ----
    if "fused_mul_sum" in nl:
        return "moe_misc"
    if "argmax" in nl:
        return "sampler"
    if (
        "copy_kernel" in nl
        or "catarraybatchedcopy" in nl
        or "memcpy" in nl
        or "fillfunctor" in nl
    ):
        return "copy"

    # ---- Second-pass: use op / source / callstack ----
    sl = source.lower()
    cl = callstack.lower()
    ol = op.lower()

    if "fused_moe" in sl or "moe_sum_reduce" in sl or "moe_align" in sl:
        return "moe_misc"
    if "dp_attention" in sl or "dp_attention" in cl:
        return "dp_gather"
    if "communicator" in cl and ("gather" in cl or "scatter" in cl):
        return "dp_gather"
    if "sampler" in cl:
        return "sampler"
    if "embedding" in cl or "embedding" in ol:
        return "embedding"

    return "other"


def is_comm(cat: str) -> bool:
    """Return whether a category represents a communication kernel."""
    return cat in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")


def _comm_agg_method(cat: str) -> str:
    """Return the cross-rank aggregation function name for a comm category.

    - Symmetric collectives (allreduce / allgather / reduce_scatter):
      **min** — the last-arriving rank records the true transfer time;
      earlier ranks include spin-wait barrier inflation.
    - Asymmetric collectives (all_to_all):
      **max** — each rank transfers a different volume (MoE token routing);
      the collective blocks until the heaviest communicator finishes.
    """
    if cat == "all_to_all":
        return "max"
    return "min"


# ---------------------------------------------------------------------------
# Per-rank CSV reading
# ---------------------------------------------------------------------------
def _read_rank_stats(
    csv_path: str,
    compute_only: bool = False,
) -> dict[str, float]:
    """Read a single rank CSV and return ``{category: total_us}``.

    Parameters
    ----------
    compute_only : bool
        If True, skip communication kernels entirely (NCCL, custom allreduce).
    """
    cats: dict[str, float] = defaultdict(float)
    skipped = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "")
            raw_dur = row.get("Duration (us)")
            if raw_dur is None or raw_dur == "":
                skipped += 1
                continue
            dur = float(raw_dur)
            op = row.get("op", "")
            source = row.get("Source Code", "")
            callstack = row.get("Call Stack", "")
            cat = classify_kernel(name, op, source, callstack)
            if compute_only and is_comm(cat):
                continue
            cats[cat] += dur
    if skipped:
        print(
            f"  [warn] {os.path.basename(csv_path)}: skipped {skipped} rows with missing Duration"
        )
    return dict(cats)


def _read_rank_rows(csv_path: str) -> list[dict]:
    """Read all rows from a rank CSV with classification added."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = classify_kernel(
                row.get("Name", ""),
                row.get("op", ""),
                row.get("Source Code", ""),
                row.get("Call Stack", ""),
            )
            row["_category"] = cat
            rows.append(row)
    return rows


def _read_rank_comm_seq(
    csv_path: str,
) -> dict[str, list[float]]:
    """Read per-invocation durations for each comm category from one rank.

    Returns ``{comm_category: [dur_call_0, dur_call_1, ...]}``.
    The list preserves call order so that invocations align across ranks.
    """
    seq: dict[str, list[float]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "")
            cat = classify_kernel(
                name,
                row.get("op", ""),
                row.get("Source Code", ""),
                row.get("Call Stack", ""),
            )
            if not is_comm(cat):
                continue
            raw_dur = row.get("Duration (us)")
            if raw_dur is None or raw_dur == "":
                continue
            seq[cat].append(float(raw_dur))
    return dict(seq)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(
    csv_dir: Optional[str] = None,
    *,
    csv_files: Optional[list[str]] = None,
    stage: str = "DECODE",
    compute_only: bool = False,
) -> dict:
    """Aggregate kernel stats across TP ranks for one (bs, ctx, stage).

    Parameters
    ----------
    csv_dir : str, optional
        Directory containing per-rank parsed CSVs.  Files matching
        ``*{stage}*.csv`` are discovered automatically.
    csv_files : list[str], optional
        Explicit list of CSV files to aggregate (overrides *csv_dir*).
    stage : str
        Stage to filter on when discovering CSVs (``"DECODE"`` or ``"EXTEND"``).
    compute_only : bool
        If True, exclude all communication kernels from the result.  Useful
        for getting pure compute time when the trace includes NCCL or
        custom allreduce kernels (``cross_device_reduce``).

    Returns
    -------
    dict
        Aggregated result with structure::

            {
                "stage": "DECODE",
                "num_ranks": 4,
                "total_kernel_us": 10300.0,
                "categories": {
                    "moe":        {"us": 2580, "pct": 25.0, "method": "mean-across-ranks"},
                    "all_reduce": {"us":  880, "pct":  8.5, "method": "min-across-ranks",
                                   "all_ranks_us": [147320, 173700, 174020, 880]},
                    ...
                },
                "per_rank_comm_us": [147320, 173700, 174020, 880],
            }
    """
    if csv_files is None:
        if csv_dir is None:
            raise ValueError("Provide either csv_dir or csv_files")
        csv_files = sorted(glob.glob(os.path.join(csv_dir, f"*{stage}*.csv")))

    if not csv_files:
        print(f"[cross_rank_agg] No {stage} CSVs found")
        return {}

    # Per-rank stats (category totals — compute only; comm handled separately)
    rank_stats: list[dict[str, float]] = []
    for csv_path in csv_files:
        rank_stats.append(_read_rank_stats(csv_path, compute_only=compute_only))

    # Per-rank comm invocation sequences for per-invocation min
    rank_comm_seqs: list[dict[str, list[float]]] = []
    if not compute_only:
        for csv_path in csv_files:
            rank_comm_seqs.append(_read_rank_comm_seq(csv_path))

    num_ranks = len(rank_stats)
    all_cats = sorted({c for s in rank_stats for c in s})

    # Collect per-rank communication totals (raw, for diagnostics)
    per_rank_comm = []
    for s in rank_stats:
        per_rank_comm.append(sum(v for k, v in s.items() if is_comm(k)))

    # Build result: comm → per-invocation agg; compute → mean
    result_cats: dict[str, dict] = {}
    for cat in all_cats:
        vals = [s.get(cat, 0) for s in rank_stats]
        if is_comm(cat):
            method = _comm_agg_method(cat)
            if method == "max":
                # Asymmetric (all_to_all): max-across-ranks total
                chosen = max(vals)
                result_cats[cat] = {
                    "us": round(chosen, 1),
                    "method": "max-across-ranks",
                    "all_ranks_us": [round(v, 1) for v in vals],
                }
            else:
                # Symmetric: per-invocation min across ranks
                per_rank_seqs = [rcs.get(cat, []) for rcs in rank_comm_seqs]
                n_calls = max((len(s) for s in per_rank_seqs), default=0)
                if n_calls > 0 and all(
                    len(s) == n_calls for s in per_rank_seqs
                ):
                    # All ranks have the same number of invocations — ideal case
                    chosen = sum(
                        min(per_rank_seqs[r][i] for r in range(num_ranks))
                        for i in range(n_calls)
                    )
                else:
                    # Mismatched call counts — fall back to min-rank-total
                    chosen = min(vals) if vals else 0
                result_cats[cat] = {
                    "us": round(chosen, 1),
                    "method": (
                        "per-invocation-min"
                        if (
                            n_calls > 0
                            and all(len(s) == n_calls for s in per_rank_seqs)
                        )
                        else "min-across-ranks"
                    ),
                    "n_invocations": n_calls,
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
        info["pct"] = (
            round(info["us"] / corrected_total * 100, 1)
            if corrected_total > 0
            else 0
        )

    return {
        "stage": stage,
        "num_ranks": num_ranks,
        "total_kernel_us": round(corrected_total, 1),
        "categories": dict(
            sorted(result_cats.items(), key=lambda x: -x[1]["us"])
        ),
        "per_rank_comm_us": [round(v, 1) for v in per_rank_comm],
    }


def print_result(result: dict) -> None:
    """Pretty-print an aggregation result dict."""
    if not result:
        return
    stage = result["stage"]
    total = result["total_kernel_us"]
    nr = result["num_ranks"]
    print(f"\n{'=' * 60}")
    print(
        f"  {stage}  (corrected, {nr} ranks, sym-per-inv-min / asym-max / compute-mean)"
    )
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
            extra = (
                f"  (ranks: {[round(v/1000, 2) for v in info['all_ranks_us']]})"
            )
        print(f"  {cat:>20}  {ms:>9.2f}  {pct:>5.1f}%  {method}{extra}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate kernel stats across TP/DP ranks (sym-comm: per-invocation min, asym-comm: max, compute: mean).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing per-rank parsed CSVs",
    )
    p.add_argument(
        "--stage",
        default="DECODE",
        choices=["DECODE", "EXTEND"],
        help="Stage to aggregate (default: DECODE)",
    )
    p.add_argument(
        "--output-json",
        "-o",
        help="Write result to JSON file",
    )
    p.add_argument(
        "--compute-only",
        action="store_true",
        help="Exclude communication kernels (NCCL / custom allreduce)",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only write JSON, no console output",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)

    result = aggregate(
        args.csv_dir, stage=args.stage, compute_only=args.compute_only
    )
    if not result:
        return 1

    if not args.quiet:
        print_result(result)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        if not args.quiet:
            print(f"[cross_rank_agg] Saved → {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
