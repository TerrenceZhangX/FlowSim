"""Integration tests for stage profiling (perf / shapes / all modes).

Exercises the three ``--collect`` modes of ``run_stage_profile.py``:

1. **perf**   — collect traces, parse, analyze.
2. **shapes** — collect kernel shapes (no CUDA graph), merge into timing CSVs.
3. **all**    — perf → auto-restart server → shapes → merge (full pipeline).

Each request produces both EXTEND (prefill) and DECODE traces.
Request parameters: ``--input-len`` (new prefill tokens), ``--existing-ctx``
(cached KV context, default 0), ``--bs`` (batch size), ``--decode-tokens``
(decode length).

Requirements
------------
* Running inside the ``flowsim`` Docker container with GPUs.
* Model config accessible at ``MODEL`` path.

Environment Variables
---------------------
``MODEL``
    Model path (default: ``/flowsim/workload/models/configs/Qwen3-235B-A22B``).
``LOAD_FORMAT``
    Load format (default: ``dummy``).
``RUN_CONFIGS``
    Comma-separated config tags to run (default: all).
"""

import ast
import csv
import glob
import os
import subprocess
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")

MODEL = os.environ.get(
    "MODEL", "/flowsim/workload/models/configs/Qwen3-235B-A22B"
)
LOAD_FORMAT = os.environ.get("LOAD_FORMAT", "dummy")
HOST = "0.0.0.0"
PORT = 30001

# (tag, dir_suffix, server_opts)
_CONFIGS = [
    ("P1", "sweep_P1_tp2", "--tp 2"),
]

# Allow filtering at runtime via RUN_CONFIGS env var
_ALLOWED = os.environ.get("RUN_CONFIGS", "")
if _ALLOWED:
    _allowed_set = {t.strip() for t in _ALLOWED.split(",")}
    _CONFIGS = [c for c in _CONFIGS if c[0] in _allowed_set]


def _config_ids():
    return [c[0] for c in _CONFIGS]


def _make_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = _PROJECT_ROOT + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_stage_profile(cmd, tag, mode, artifact_dir):
    """Run a stage profile command, write logs, return subprocess result."""
    os.makedirs(artifact_dir, exist_ok=True)
    stderr_path = os.path.join(
        artifact_dir, f"stage_profile_{tag}_{mode}.stderr.log"
    )
    with open(stderr_path, "w") as ferr:
        result = subprocess.run(
            cmd,
            stdout=ferr,
            stderr=ferr,
            env=_make_env(),
            timeout=1800,
        )
    if result.returncode != 0:
        with open(stderr_path) as f:
            tail = f.read()[-3000:]
        pytest.fail(
            f"stage_profile {mode} failed for {tag}.\n"
            f"Log: {stderr_path}\nstderr tail:\n{tail}"
        )
    return result


def _server_opts(extra):
    return (
        f"--model-path {MODEL} --load-format {LOAD_FORMAT} "
        f"--host {HOST} --port {PORT} {extra}"
    )


# -----------------------------------------------------------------------
# Shape validation helpers
# -----------------------------------------------------------------------
def _read_csv(path):
    """Read a parsed CSV and return list of row dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# Kernel name patterns that indicate a GEMM operation
_GEMM_NAME_PATTERNS = ("nvjet", "cublasLt", "cublas_", "cutlass_gemm")


def _first_matmul_dim0(rows):
    """Return the first dimension of the first matmul kernel's first input.

    For a GEMM kernel ``[M, K] x [K, N]``, returns ``M``.
    Matches by ``op == "matmul"`` first, then falls back to kernel name
    patterns (``nvjet``, ``cublasLt``, etc.) for CUDA-graph traces where
    the ``op`` field may be empty.
    """
    # Pass 1: exact op match
    for row in rows:
        if row.get("op", "") == "matmul":
            dims = ast.literal_eval(row["Dims"])
            return dims[0][0]
    # Pass 2: kernel name pattern (CUDA-graph traces lose op labels)
    for row in rows:
        name = row["Name"]
        dims_str = row.get("Dims", "N/A")
        if dims_str == "N/A" or not dims_str:
            continue
        if any(pat in name for pat in _GEMM_NAME_PATTERNS):
            dims = ast.literal_eval(dims_str)
            # GEMM has exactly 2 inputs, both 2-D
            if len(dims) >= 2 and len(dims[0]) == 2 and len(dims[1]) == 2:
                return dims[0][0]
    return None


def _attention_seqlen_pair(rows, bs, seq_len):
    """Check that ``[bs, seq_len]`` (or ``[bs, seq_len+1]``) appears in FlashAttn dims.

    Flash Attention's varlen kernel receives a ``[num_seqs, max_seqlen]``
    shaped parameter.  This function searches all dim lists of the first
    non-Combine, non-prepare FlashAttn kernel for that exact pair.

    Returns the matching ``[num_seqs, max_seqlen]`` list, or None.
    """
    for row in rows:
        name = row["Name"]
        if "FlashAttn" not in name:
            continue
        if "Combine" in name or "prepare" in name:
            continue
        dims = ast.literal_eval(row["Dims"])
        for d in dims:
            if (
                isinstance(d, list)
                and len(d) == 2
                and d[0] == bs
                and d[1] in (seq_len, seq_len + 1)
            ):
                return d
        return None
    return None


def _validate_shapes(output_dir, bs, input_len, existing_ctx):
    """Validate kernel shapes in merged/shape_parsed CSVs reflect the workload.

    Checks (any TP-0 CSV in the first ``bs*_input*_ctx*`` subdir):

    1. **EXTEND first GEMM** ``dim0 == bs * input_len``
       The QKV projection processes all new prefill tokens.
    2. **EXTEND attention** ``[bs, seq_len] ∈ FlashAttn dims``
       where ``seq_len = input_len + existing_ctx``.  Flash Attention's
       varlen kernel receives ``[num_seqs, max_seqlen]``; we check
       the exact pair (``+1`` tolerance for BOS).
    3. **DECODE first GEMM** ``dim0 == bs``
       Each decode step processes one token per sequence.
    """
    tag = f"bs{bs}_input{input_len}_ctx{existing_ctx}"
    # Try merged first, fall back to shape_parsed
    for csv_subdir in ("merged", "shape_parsed"):
        extend_csvs = sorted(
            glob.glob(
                os.path.join(output_dir, tag, csv_subdir, "*TP-0*EXTEND*.csv")
            )
        )
        decode_csvs = sorted(
            glob.glob(
                os.path.join(output_dir, tag, csv_subdir, "*TP-0*DECODE*.csv")
            )
        )
        if extend_csvs and decode_csvs:
            break
    else:
        pytest.fail(
            f"No EXTEND+DECODE CSVs for TP-0 in {output_dir}/{tag}/{{merged,shape_parsed}}/"
        )

    extend_rows = _read_csv(extend_csvs[0])
    decode_rows = _read_csv(decode_csvs[0])

    # Rule 1: EXTEND first GEMM dim0 == bs * input_len
    ext_gemm_dim0 = _first_matmul_dim0(extend_rows)
    assert ext_gemm_dim0 is not None, "No matmul kernel found in EXTEND CSV"
    expected_ext = bs * input_len
    assert (
        ext_gemm_dim0 == expected_ext
    ), f"EXTEND first GEMM dim0={ext_gemm_dim0}, expected bs*input_len={expected_ext}"

    # Rule 2: EXTEND FlashAttn dims contain [bs, seq_len] (varlen parameter)
    seq_len = input_len + existing_ctx
    attn_pair = _attention_seqlen_pair(extend_rows, bs, seq_len)
    assert attn_pair is not None, (
        f"No FlashAttention dim matching [bs={bs}, seqlen={seq_len}(+1)] "
        f"in EXTEND CSV"
    )
    assert (
        attn_pair[0] == bs
    ), f"FlashAttn num_seqs={attn_pair[0]}, expected bs={bs}"
    assert attn_pair[1] in (seq_len, seq_len + 1), (
        f"FlashAttn max_seqlen={attn_pair[1]}, "
        f"expected {seq_len} or {seq_len + 1}"
    )

    # Rule 3: DECODE first GEMM dim0 == bs
    dec_gemm_dim0 = _first_matmul_dim0(decode_rows)
    assert dec_gemm_dim0 is not None, "No matmul kernel found in DECODE CSV"
    assert (
        dec_gemm_dim0 == bs
    ), f"DECODE first GEMM dim0={dec_gemm_dim0}, expected bs={bs}"


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------
@pytest.fixture
def artifact_dir():
    d = os.environ.get("PYTEST_ARTIFACT_DIR", "/flowsim/tests/test-artifacts")
    os.makedirs(d, exist_ok=True)
    return d


# -----------------------------------------------------------------------
# test_stage_profile_perf
# -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "tag,dir_suffix,server_opts", _CONFIGS, ids=_config_ids()
)
def test_stage_profile_perf(tag, dir_suffix, server_opts, artifact_dir):
    """``--collect perf``: single point, produce traces + parsed CSVs."""
    output_dir = os.path.join(artifact_dir, f"{tag}_perf_output")
    log_dir = os.path.join(artifact_dir, f"{tag}_perf_server_logs")

    cmd = [
        sys.executable,
        "-u",
        os.path.join(_SCRIPTS_DIR, "run_stage_profile.py"),
        "--collect",
        "perf",
        "--launch-server",
        "--server-opts",
        _server_opts(server_opts),
        "--bs",
        "1",
        "--input-len",
        "2048",
        "--decode-tokens",
        "32",
        "--output-dir",
        output_dir,
        "--log-dir",
        log_dir,
    ]

    _run_stage_profile(cmd, tag, "perf", artifact_dir)

    # ── Verify outputs ──
    assert os.path.isdir(output_dir)

    traces = glob.glob(
        os.path.join(output_dir, "**/*.trace.json.gz"), recursive=True
    )
    assert len(traces) > 0, f"No trace files found under {output_dir}"
    extend_traces = [t for t in traces if "EXTEND" in os.path.basename(t)]
    decode_traces = [t for t in traces if "DECODE" in os.path.basename(t)]
    assert len(extend_traces) > 0, f"No EXTEND traces found under {output_dir}"
    assert len(decode_traces) > 0, f"No DECODE traces found under {output_dir}"
    parsed = glob.glob(
        os.path.join(output_dir, "**/parsed/*.csv"), recursive=True
    )
    assert len(parsed) > 0, f"No parsed CSVs found under {output_dir}"
    extend_csvs = [p for p in parsed if "EXTEND" in os.path.basename(p)]
    decode_csvs = [p for p in parsed if "DECODE" in os.path.basename(p)]
    assert len(extend_csvs) > 0, f"No EXTEND parsed CSVs under {output_dir}"
    assert len(decode_csvs) > 0, f"No DECODE parsed CSVs under {output_dir}"


# -----------------------------------------------------------------------
# test_stage_profile_all_with_ctx
# -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "tag,dir_suffix,server_opts", _CONFIGS, ids=_config_ids()
)
def test_stage_profile_all_with_ctx(tag, dir_suffix, server_opts, artifact_dir):
    """``--collect all`` with existing KV cache context (--existing-ctx > 0)."""
    output_dir = os.path.join(artifact_dir, f"{tag}_all_ctx_output")
    log_dir = os.path.join(artifact_dir, f"{tag}_all_ctx_server_logs")

    cmd = [
        sys.executable,
        "-u",
        os.path.join(_SCRIPTS_DIR, "run_stage_profile.py"),
        "--collect",
        "all",
        "--launch-server",
        "--server-opts",
        _server_opts(server_opts),
        "--bs",
        "1",
        "--input-len",
        "512",
        "--existing-ctx",
        "2048",
        "--decode-tokens",
        "32",
        "--output-dir",
        output_dir,
        "--log-dir",
        log_dir,
    ]

    _run_stage_profile(cmd, tag, "all_ctx", artifact_dir)

    assert os.path.isdir(output_dir)
    traces = glob.glob(
        os.path.join(output_dir, "**/*.trace.json.gz"), recursive=True
    )
    assert len(traces) > 0, f"No trace files under {output_dir}"
    extend_traces = [t for t in traces if "EXTEND" in os.path.basename(t)]
    decode_traces = [t for t in traces if "DECODE" in os.path.basename(t)]
    assert len(extend_traces) > 0, f"No EXTEND traces under {output_dir}"
    assert len(decode_traces) > 0, f"No DECODE traces under {output_dir}"
    parsed = glob.glob(
        os.path.join(output_dir, "**/parsed/*.csv"), recursive=True
    )
    assert len(parsed) > 0, f"No parsed CSVs under {output_dir}"
    extend_csvs = [p for p in parsed if "EXTEND" in os.path.basename(p)]
    decode_csvs = [p for p in parsed if "DECODE" in os.path.basename(p)]
    assert len(extend_csvs) > 0, f"No EXTEND parsed CSVs under {output_dir}"
    assert len(decode_csvs) > 0, f"No DECODE parsed CSVs under {output_dir}"
    shape_traces = glob.glob(
        os.path.join(output_dir, "**/shape_traces/*.trace.json.gz"),
        recursive=True,
    )
    assert len(shape_traces) > 0, f"No shape traces under {output_dir}"
    merged = glob.glob(
        os.path.join(output_dir, "**/merged/*.csv"), recursive=True
    )
    assert len(merged) > 0, f"No merged CSVs under {output_dir}"

    # ── Validate kernel shapes reflect the workload ──
    _validate_shapes(output_dir, bs=1, input_len=512, existing_ctx=2048)


# -----------------------------------------------------------------------
# test_stage_profile_shapes
# -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "tag,dir_suffix,server_opts", _CONFIGS, ids=_config_ids()
)
def test_stage_profile_shapes(tag, dir_suffix, server_opts, artifact_dir):
    """``--collect shapes``: run perf first, then shapes separately."""
    output_dir = os.path.join(artifact_dir, f"{tag}_shapes_output")
    log_dir = os.path.join(artifact_dir, f"{tag}_shapes_server_logs")
    sopts = _server_opts(server_opts)

    # Step 1: generate perf data (shapes needs existing subdirs)
    perf_cmd = [
        sys.executable,
        "-u",
        os.path.join(_SCRIPTS_DIR, "run_stage_profile.py"),
        "--collect",
        "perf",
        "--launch-server",
        "--server-opts",
        sopts,
        "--bs",
        "1",
        "--input-len",
        "2048",
        "--decode-tokens",
        "32",
        "--output-dir",
        output_dir,
        "--log-dir",
        log_dir,
    ]
    _run_stage_profile(perf_cmd, tag, "shapes_prep", artifact_dir)

    # Step 2: collect shapes (server launched with --disable-cuda-graph)
    shapes_cmd = [
        sys.executable,
        "-u",
        os.path.join(_SCRIPTS_DIR, "run_stage_profile.py"),
        "--collect",
        "shapes",
        "--launch-server",
        "--server-opts",
        sopts,
        "--bs",
        "1",
        "--input-len",
        "2048",
        "--decode-tokens",
        "32",
        "--output-dir",
        output_dir,
        "--log-dir",
        log_dir,
    ]
    _run_stage_profile(shapes_cmd, tag, "shapes", artifact_dir)

    # ── Verify shape outputs ──
    shape_traces = glob.glob(
        os.path.join(output_dir, "**/shape_traces/*.trace.json.gz"),
        recursive=True,
    )
    assert len(shape_traces) > 0, f"No shape traces under {output_dir}"
    shape_parsed = glob.glob(
        os.path.join(output_dir, "**/shape_parsed/*.csv"), recursive=True
    )
    assert len(shape_parsed) > 0, f"No shape CSVs under {output_dir}"
    merged = glob.glob(
        os.path.join(output_dir, "**/merged/*.csv"), recursive=True
    )
    assert len(merged) > 0, f"No merged CSVs under {output_dir}"

    # ── Validate kernel shapes reflect the workload ──
    _validate_shapes(output_dir, bs=1, input_len=2048, existing_ctx=0)


# -----------------------------------------------------------------------
# test_stage_profile_all
# -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "tag,dir_suffix,server_opts", _CONFIGS, ids=_config_ids()
)
def test_stage_profile_all(tag, dir_suffix, server_opts, artifact_dir):
    """``--collect all``: full pipeline — perf → restart → shapes → merge."""
    output_dir = os.path.join(artifact_dir, f"{tag}_all_output")
    log_dir = os.path.join(artifact_dir, f"{tag}_all_server_logs")

    cmd = [
        sys.executable,
        "-u",
        os.path.join(_SCRIPTS_DIR, "run_stage_profile.py"),
        "--collect",
        "all",
        "--launch-server",
        "--server-opts",
        _server_opts(server_opts),
        "--bs",
        "1",
        "--input-len",
        "2048",
        "--decode-tokens",
        "32",
        "--output-dir",
        output_dir,
        "--log-dir",
        log_dir,
    ]

    _run_stage_profile(cmd, tag, "all", artifact_dir)

    # ── Verify full pipeline outputs ──
    assert os.path.isdir(output_dir)

    traces = glob.glob(
        os.path.join(output_dir, "**/*.trace.json.gz"), recursive=True
    )
    assert len(traces) > 0, f"No trace files under {output_dir}"
    extend_traces = [t for t in traces if "EXTEND" in os.path.basename(t)]
    decode_traces = [t for t in traces if "DECODE" in os.path.basename(t)]
    assert len(extend_traces) > 0, f"No EXTEND traces under {output_dir}"
    assert len(decode_traces) > 0, f"No DECODE traces under {output_dir}"
    parsed = glob.glob(
        os.path.join(output_dir, "**/parsed/*.csv"), recursive=True
    )
    assert len(parsed) > 0, f"No parsed CSVs under {output_dir}"
    extend_csvs = [p for p in parsed if "EXTEND" in os.path.basename(p)]
    decode_csvs = [p for p in parsed if "DECODE" in os.path.basename(p)]
    assert len(extend_csvs) > 0, f"No EXTEND parsed CSVs under {output_dir}"
    assert len(decode_csvs) > 0, f"No DECODE parsed CSVs under {output_dir}"
    shape_traces = glob.glob(
        os.path.join(output_dir, "**/shape_traces/*.trace.json.gz"),
        recursive=True,
    )
    assert len(shape_traces) > 0, f"No shape traces under {output_dir}"
    merged = glob.glob(
        os.path.join(output_dir, "**/merged/*.csv"), recursive=True
    )
    assert len(merged) > 0, f"No merged CSVs under {output_dir}"

    summary_path = os.path.join(output_dir, "sweep_summary.json")
    assert os.path.exists(summary_path), "sweep_summary.json not created"

    # ── Validate kernel shapes reflect the workload ──
    _validate_shapes(output_dir, bs=1, input_len=2048, existing_ctx=0)
