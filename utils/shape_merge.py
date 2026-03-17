#!/usr/bin/env python
"""Merge kernel shape information from a no-CUDA-graph pass into a CUDA-graph pass.

When CUDA graphs are enabled, PyTorch profiler cannot associate "External id"
events with "Input Dims" metadata, resulting in ``N/A`` for most kernel shapes
in the parsed CSV.  Running a second profiling pass with ``--disable-cuda-graph``
yields accurate shape info (Dims, Data Type, Input/Output) but less
representative timing (no graph-launch overhead, different scheduling).

This module merges shape columns from the *shape pass* (no CUDA graph) into the
*timing pass* (with CUDA graph), producing a final CSV with both accurate
timings **and** populated shape information.

Matching strategy
-----------------
Kernels are matched by **(kernel name, occurrence index)**: the *n*-th occurrence
of a given kernel name in the timing CSV is matched to the *n*-th occurrence in
the shape CSV.  This works because the same model + same workload produces the
same deterministic kernel dispatch sequence regardless of CUDA-graph mode.

For any kernel name present in *both* CSVs, the occurrence counts must match
exactly (a ``ValueError`` is raised otherwise).  Kernels that appear only in
the timing CSV (e.g. CUDA-graph launcher stubs) are kept as-is with ``N/A``
dims — this is expected and not treated as an error.

Usage — Python API
------------------
    from utils.shape_merge import merge_shapes, merge_shapes_dir

    # Single pair
    merge_shapes("timing.csv", "shape.csv", "merged.csv")

    # Directory (auto-matches by rank + stage)
    merge_shapes_dir("timing_parsed/", "shape_parsed/", "merged_parsed/")

Usage — CLI
-----------
    # Single pair
    python -m utils.shape_merge --timing-csv timing.csv --shape-csv shape.csv -o merged.csv

    # Directory
    python -m utils.shape_merge --timing-dir timing_parsed/ --shape-dir shape_parsed/ \\
                          --output-dir merged_parsed/
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from typing import Optional

# CSV columns produced by BaseKernelInfoParser.save_individual_csv
_SHAPE_COLS = ("Dims", "Data Type", "Input/Output")
_CSV_HEADER = [
    "Name",
    "Dims",
    "Data Type",
    "Input/Output",
    "Descriptions",
    "Duration (us)",
    "op",
    "operation",
    "Source Code",
    "Call Stack",
]


# ---------------------------------------------------------------------------
# Rank / stage extraction from filenames
# ---------------------------------------------------------------------------
_RANK_STAGE_RE = re.compile(
    r"(TP-\d+(?:-DP-\d+)?(?:-EP-\d+)?)"  # rank identifier
    r"-(EXTEND|DECODE)",  # stage
    re.IGNORECASE,
)


def _rank_stage_key(filename: str) -> tuple[str, str] | None:
    """Extract ``(rank_id, stage)`` from a CSV filename.

    Examples
    --------
    >>> _rank_stage_key("1772525862-TP-0-DECODE.trace.csv")
    ('TP-0', 'DECODE')
    >>> _rank_stage_key("1772529412-TP-1-DP-1-EXTEND.trace.csv")
    ('TP-1-DP-1', 'EXTEND')
    """
    m = _RANK_STAGE_RE.search(os.path.basename(filename))
    if m:
        return m.group(1), m.group(2).upper()
    return None


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------
def _build_shape_lookup(
    shape_rows: list[dict],
) -> dict[str, list[dict]]:
    """Build ``{kernel_name: [row0, row1, ...]}`` from shape CSV rows."""
    lookup: dict[str, list[dict]] = defaultdict(list)
    for row in shape_rows:
        lookup[row["Name"]].append(row)
    return lookup


def merge_shapes(
    timing_csv: str,
    shape_csv: str,
    output_csv: Optional[str] = None,
    *,
    verbose: bool = False,
) -> str:
    """Merge shape columns from *shape_csv* into *timing_csv*.

    Parameters
    ----------
    timing_csv : str
        CSV from the CUDA-graph profiling pass (accurate durations, N/A dims).
    shape_csv : str
        CSV from the no-CUDA-graph profiling pass (accurate dims, less
        representative durations).
    output_csv : str, optional
        Path for the merged CSV.  Defaults to ``<timing_csv_stem>_merged.csv``
        in the same directory as *timing_csv*.
    verbose : bool
        Print matching statistics.

    Returns
    -------
    str
        Path to the written merged CSV.
    """
    if output_csv is None:
        base, ext = os.path.splitext(timing_csv)
        output_csv = f"{base}_merged{ext}"

    # Read shape CSV
    with open(shape_csv, newline="") as f:
        shape_rows = list(csv.DictReader(f))
    shape_lookup = _build_shape_lookup(shape_rows)

    # Track how many times we've seen each kernel name in the timing CSV
    name_counter: dict[str, int] = defaultdict(int)

    # Read timing CSV and merge
    with open(timing_csv, newline="") as f:
        timing_rows = list(csv.DictReader(f))

    merged_rows: list[dict] = []
    stats = {"total": 0, "merged": 0, "already_ok": 0, "no_match": 0}

    for row in timing_rows:
        kname = row["Name"]
        idx = name_counter[kname]
        name_counter[kname] += 1
        stats["total"] += 1

        # Check if timing row already has valid dims
        dims_val = row.get("Dims", "N/A")
        has_dims = dims_val and dims_val != "N/A"

        if has_dims:
            # Already populated — keep as-is
            merged_rows.append(row)
            stats["already_ok"] += 1
            continue

        # Look up the nth occurrence in shape CSV.  For kernels present
        # in both CSVs, occurrence counts must match 1:1.  Kernels only
        # in the timing CSV (e.g. graph-launcher stubs) keep N/A dims.
        shape_entries = shape_lookup.get(kname, [])
        if shape_entries:
            if idx >= len(shape_entries):
                raise ValueError(
                    f"Kernel {kname!r} has {len(shape_entries)} entries in "
                    f"shape CSV but timing CSV needs occurrence #{idx}. "
                    f"Timing and shape passes captured different batch counts."
                )
            shape_row = shape_entries[idx]
            # Copy shape columns if non-N/A
            for col in _SHAPE_COLS:
                shape_val = shape_row.get(col, "N/A")
                if shape_val and shape_val != "N/A":
                    row[col] = shape_val
            # Also copy Descriptions if timing row is empty
            if not row.get("Descriptions") and shape_row.get("Descriptions"):
                row["Descriptions"] = shape_row["Descriptions"]
            # Copy 'op' and 'operation' when timing row lacks them
            # (CUDA-graph mode often produces empty or 'TBD' ops)
            for op_col in ("op", "operation"):
                timing_op = (row.get(op_col) or "").strip()
                if timing_op in ("", "TBD"):
                    shape_op = (shape_row.get(op_col) or "").strip()
                    if shape_op and shape_op != "TBD":
                        row[op_col] = shape_op
            merged_rows.append(row)
            stats["merged"] += 1
        else:
            # No matching shape entry — keep timing row as-is
            merged_rows.append(row)
            stats["no_match"] += 1

    # Write output
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    if not merged_rows:
        # Empty timing CSV (header-only) — write header and return.
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
            writer.writeheader()
        return output_csv

    fieldnames = (
        _CSV_HEADER
        if set(_CSV_HEADER).issubset(merged_rows[0].keys())
        else list(merged_rows[0].keys())
    )
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    if verbose:
        print(
            f"[shape_merge] {os.path.basename(timing_csv)}: "
            f"{stats['total']} kernels, "
            f"{stats['merged']} shapes merged, "
            f"{stats['already_ok']} already had dims, "
            f"{stats['no_match']} unmatched"
        )

    return output_csv


def merge_shapes_dir(
    timing_dir: str,
    shape_dir: str,
    output_dir: Optional[str] = None,
    *,
    stage: Optional[str] = None,
    verbose: bool = True,
) -> list[str]:
    """Merge shapes for all matching CSV pairs across two directories.

    CSVs are matched by **(rank, stage)** extracted from filenames.  For example,
    ``...-TP-0-DECODE.trace.csv`` in *timing_dir* is matched with the
    ``...-TP-0-DECODE.trace.csv`` in *shape_dir* (timestamps may differ).

    Parameters
    ----------
    timing_dir : str
        Directory with parsed CSVs from the CUDA-graph pass.
    shape_dir : str
        Directory with parsed CSVs from the no-CUDA-graph pass.
    output_dir : str, optional
        Directory for merged CSVs.  Defaults to ``<timing_dir>/merged/``.
    stage : str, optional
        If given, only process CSVs matching this stage (``"DECODE"`` or
        ``"EXTEND"``).  By default, process all stages.
    verbose : bool
        Print per-file statistics.

    Returns
    -------
    list[str]
        Paths to the written merged CSVs.
    """
    if output_dir is None:
        output_dir = os.path.join(timing_dir, "merged")
    os.makedirs(output_dir, exist_ok=True)

    # Index shape CSVs by (rank, stage)
    shape_csvs = sorted(glob.glob(os.path.join(shape_dir, "*.csv")))
    shape_index: dict[tuple[str, str], str] = {}
    for sc in shape_csvs:
        key = _rank_stage_key(sc)
        if key is None:
            continue
        if stage and key[1] != stage.upper():
            continue
        existing = shape_index.get(key)
        if existing is not None and verbose:
            print(
                f"[shape_merge] Multiple shape CSVs for {key}: "
                f"{os.path.basename(existing)}, {os.path.basename(sc)} "
                f"→ using {os.path.basename(sc)}"
            )
        shape_index[key] = sc

    # Process timing CSVs
    timing_csvs = sorted(glob.glob(os.path.join(timing_dir, "*.csv")))
    results: list[str] = []

    for tc in timing_csvs:
        key = _rank_stage_key(tc)
        if key is None:
            continue
        if stage and key[1] != stage.upper():
            continue
        sc = shape_index.get(key)
        if sc is None:
            if verbose:
                print(
                    f"[shape_merge] No shape CSV for {key} — skipping {os.path.basename(tc)}"
                )
            continue

        out_name = os.path.basename(tc).replace(
            ".trace.csv", "_merged.trace.csv"
        )
        out_path = os.path.join(output_dir, out_name)
        merge_shapes(tc, sc, out_path, verbose=verbose)
        results.append(out_path)

    if verbose:
        print(f"[shape_merge] Merged {len(results)} CSV pairs → {output_dir}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge shape info from no-CUDA-graph CSV into CUDA-graph CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Single-file mode
    p.add_argument("--timing-csv", help="Timing CSV (CUDA-graph pass)")
    p.add_argument("--shape-csv", help="Shape CSV (no-CUDA-graph pass)")
    p.add_argument("-o", "--output-csv", help="Output merged CSV")

    # Directory mode
    p.add_argument("--timing-dir", help="Directory of timing CSVs")
    p.add_argument("--shape-dir", help="Directory of shape CSVs")
    p.add_argument("--output-dir", help="Directory for merged CSVs")

    # Options
    p.add_argument(
        "--stage",
        choices=["DECODE", "EXTEND"],
        help="Only process CSVs for this stage",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    verbose = not args.quiet

    if args.timing_csv and args.shape_csv:
        out = merge_shapes(
            args.timing_csv, args.shape_csv, args.output_csv, verbose=verbose
        )
        print(f"Merged CSV: {out}")
        return 0

    if args.timing_dir and args.shape_dir:
        results = merge_shapes_dir(
            args.timing_dir,
            args.shape_dir,
            args.output_dir,
            stage=args.stage,
            verbose=verbose,
        )
        for r in results:
            print(f"  {r}")
        return 0

    print(
        "Error: provide either (--timing-csv + --shape-csv) or (--timing-dir + --shape-dir)"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
