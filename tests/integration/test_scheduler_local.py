"""Integration tests for the FlowSim scheduler CLI.

Tests all three scheduler backends (local, k8s, slurm) end-to-end.

* **local** — submits jobs via ``flowsim submit --scheduler local`` which
  launches Docker containers on the host.  Validates job lifecycle (submit,
  list, status) and trace CSV correctness (GEMM dim0, FlashAttn seqlen).
* **k8s**   — submits a real Job to a Kind cluster, retrieves traces via
  ``docker cp``, and validates trace CSVs.  Auto-sets up the Kind cluster
  via ``dev-setup.sh`` if not already running.
* **slurm** — submits a real job to a local docker-compose Slurm cluster,
  retrieves traces via ``docker cp``, and validates trace CSVs.  Auto-sets
  up the Slurm cluster via ``dev-setup.sh slurm`` if not already running.

Requirements
------------
* Docker with ``flowsim-image:latest`` built (for local tests).
* A GPU-equipped host machine (local tests run on the physical host,
  NOT inside a Docker container).
* ``dockerfiles/dev-setup.sh`` available (Kind and Slurm clusters are
  automatically created if missing).
* ``schedulers/`` available on PYTHONPATH.

Environment Variables
---------------------
``MODEL``
    Model path relative to project root
    (default: ``workload/models/configs/Qwen3-235B-A22B``).
``LOAD_FORMAT``
    Load format (default: ``dummy``).

Usage
-----
    # On host (local scheduler tests — needs Docker + GPU):
    cd FlowSim && python -m pytest \
        tests/integration/test_scheduler_local.py -v -x -k "local"

    # On host (k8s tests — needs kubeconfig):
    python -m pytest tests/integration/test_scheduler_local.py \
        -v -x -k "k8s"
"""

import ast
import csv
import glob
import json
import os
import subprocess
import sys
import time

import pytest

from schedulers.base import JobResult, ProfileJobSpec
from schedulers.local import LocalScheduler

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_DEV_SETUP = os.path.join(_PROJECT_ROOT, "dockerfiles", "dev-setup.sh")
_DEV_TEARDOWN = os.path.join(_PROJECT_ROOT, "dockerfiles", "dev-teardown.sh")

MODEL = os.environ.get(
    "MODEL", "workload/models/configs/Qwen3-235B-A22B"
)
LOAD_FORMAT = os.environ.get("LOAD_FORMAT", "dummy")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flowsim_cli(*args: str, timeout: int = 1200) -> subprocess.CompletedProcess:
    """Run a ``flowsim`` subcommand via Python entry point."""
    cmd = [
        sys.executable, "-u", "-c",
        "from scripts.cli import main; main()",
        *args,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = _PROJECT_ROOT + (
        ":" + env.get("PYTHONPATH", "")
    )
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
        env=env,
        timeout=timeout,
    )


def _assert_traces(output_dir: str) -> None:
    """Assert EXTEND + DECODE traces and parsed CSVs exist."""
    traces = glob.glob(
        os.path.join(output_dir, "**/*.trace.json.gz"), recursive=True
    )
    assert len(traces) > 0, f"No trace files under {output_dir}"
    extend = [t for t in traces if "EXTEND" in os.path.basename(t)]
    decode = [t for t in traces if "DECODE" in os.path.basename(t)]
    assert len(extend) > 0, "No EXTEND traces"
    assert len(decode) > 0, "No DECODE traces"

    csvs = glob.glob(
        os.path.join(output_dir, "**/parsed/*.csv"), recursive=True
    )
    assert len(csvs) > 0, f"No parsed CSVs under {output_dir}"
    # At least EXTEND should be parsed; DECODE CSV may be absent for short sequences
    extend_csvs = [c for c in csvs if "EXTEND" in os.path.basename(c)]
    assert len(extend_csvs) > 0, "No EXTEND parsed CSVs"


def _assert_logs(output_dir: str) -> None:
    """Assert server log files exist under {output_dir}/logs/."""
    log_dir = os.path.join(output_dir, "logs")
    assert os.path.isdir(log_dir), f"Log directory not found: {log_dir}"
    log_files = os.listdir(log_dir)
    assert len(log_files) > 0, f"No log files in {log_dir}"
    stdout_logs = [f for f in log_files if f.endswith(".stdout.log")]
    stderr_logs = [f for f in log_files if f.endswith(".stderr.log")]
    assert len(stdout_logs) > 0, f"No stdout logs in {log_dir}"
    assert len(stderr_logs) > 0, f"No stderr logs in {log_dir}"
    # At least one log should be non-empty
    sizes = [
        os.path.getsize(os.path.join(log_dir, f))
        for f in stdout_logs
    ]
    assert max(sizes) > 0, "All stdout logs are empty"


# ---------------------------------------------------------------------------
# Shape validation helpers (same logic as test_stage_profile_configs.py)
# ---------------------------------------------------------------------------
def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


_GEMM_NAME_PATTERNS = ("nvjet", "cublasLt", "cublas_", "cutlass_gemm")


def _first_matmul_dim0(rows):
    """Return dim0 of the first GEMM kernel (the M dimension)."""
    for row in rows:
        if row.get("op", "") == "matmul":
            dims = ast.literal_eval(row["Dims"])
            return dims[0][0]
    for row in rows:
        name = row["Name"]
        dims_str = row.get("Dims", "N/A")
        if dims_str == "N/A" or not dims_str:
            continue
        if any(pat in name for pat in _GEMM_NAME_PATTERNS):
            dims = ast.literal_eval(dims_str)
            if len(dims) >= 2 and len(dims[0]) == 2 and len(dims[1]) == 2:
                return dims[0][0]
    return None


def _attention_seqlen_pair(rows, bs, seq_len):
    """Check that [bs, seq_len] (or +1) appears in FlashAttn dims."""
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
    """Validate GEMM dim0 and FlashAttn seqlen in merged/shape_parsed CSVs."""
    tag = f"bs{bs}_input{input_len}_ctx{existing_ctx}"
    for csv_subdir in ("merged", "shape_parsed"):
        extend_csvs = sorted(
            glob.glob(os.path.join(output_dir, tag, csv_subdir, "*TP-0*EXTEND*.csv"))
        )
        decode_csvs = sorted(
            glob.glob(os.path.join(output_dir, tag, csv_subdir, "*TP-0*DECODE*.csv"))
        )
        if extend_csvs and decode_csvs:
            break
    else:
        pytest.fail(
            f"No EXTEND+DECODE CSVs for TP-0 in {output_dir}/{tag}/{{merged,shape_parsed}}/"
        )

    extend_rows = _read_csv(extend_csvs[0])
    decode_rows = _read_csv(decode_csvs[0])

    # EXTEND first GEMM dim0 == bs * input_len
    ext_gemm_dim0 = _first_matmul_dim0(extend_rows)
    assert ext_gemm_dim0 is not None, "No matmul kernel found in EXTEND CSV"
    expected_ext = bs * input_len
    assert ext_gemm_dim0 == expected_ext, (
        f"EXTEND first GEMM dim0={ext_gemm_dim0}, expected bs*input_len={expected_ext}"
    )

    # EXTEND FlashAttn dims contain [bs, seq_len]
    seq_len = input_len + existing_ctx
    attn_pair = _attention_seqlen_pair(extend_rows, bs, seq_len)
    assert attn_pair is not None, (
        f"No FlashAttention dim matching [bs={bs}, seqlen={seq_len}(+1)] in EXTEND CSV"
    )

    # DECODE first GEMM dim0 == bs
    dec_gemm_dim0 = _first_matmul_dim0(decode_rows)
    assert dec_gemm_dim0 is not None, "No matmul kernel found in DECODE CSV"
    assert dec_gemm_dim0 == bs, (
        f"DECODE first GEMM dim0={dec_gemm_dim0}, expected bs={bs}"
    )


# =====================================================================
# LOCAL SCHEDULER — real profiling (4-step flow)
# =====================================================================
class TestLocalScheduler:
    """Run real profiling via ``flowsim`` CLI on the local Docker scheduler.

    Flow per test point:
    1. ``flowsim submit`` — submit the job (collect all)
    2. ``flowsim list``   — verify the job appears
    3. ``flowsim status`` — poll until Completed
    4. Validate trace CSVs — GEMM dim0, FlashAttn seqlen for EXTEND & DECODE
    """

    _TP1_POINTS = [
        {"bs": 1, "input_len": 2048, "existing_ctx": 0, "decode_tokens": 2},
        {"bs": 1, "input_len": 2048, "existing_ctx": 2048, "decode_tokens": 2},
    ]

    @pytest.mark.parametrize(
        "point",
        _TP1_POINTS,
        ids=[f"bs{p['bs']}_il{p['input_len']}_ctx{p['existing_ctx']}" for p in _TP1_POINTS],
    )
    def test_local_tp1_all(self, point):
        bs = point["bs"]
        input_len = point["input_len"]
        existing_ctx = point["existing_ctx"]
        decode_tokens = point["decode_tokens"]

        # ── Step 1: submit ──
        r = _flowsim_cli(
            "submit",
            "--scheduler", "local",
            "--collect", "all",
            "--model-path", MODEL,
            "--tp", "1",
            "--bs", str(bs),
            "--input-len", str(input_len),
            "--existing-ctx", str(existing_ctx),
            "--decode-tokens", str(decode_tokens),
            "--warmup-n", "2",
            "--gpus", "1",
            "--local-gpus", "0",
            "--extra-server-opts", f"--load-format {LOAD_FORMAT}",
        )
        if r.returncode != 0:
            print("STDOUT:", r.stdout[-3000:])
            print("STDERR:", r.stderr[-3000:])
        assert r.returncode == 0, f"flowsim submit failed (exit {r.returncode})"

        # Extract job_id from output (line like "flowsim-all-... completed successfully")
        combined = r.stdout + r.stderr
        job_id = None
        for line in combined.splitlines():
            if "flowsim-all-" in line:
                for word in line.split():
                    if word.startswith("flowsim-all-"):
                        job_id = word.rstrip(".,;:")
                        break
                if job_id:
                    break
        assert job_id, f"Could not find job_id in submit output:\n{combined[-1000:]}"

        # ── Step 2: list — verify job appears ──
        r_list = _flowsim_cli("list", "--scheduler", "local")
        assert r_list.returncode == 0, "flowsim list failed"
        assert job_id in r_list.stdout, (
            f"Job {job_id} not found in list output:\n{r_list.stdout}"
        )

        # ── Step 3: status — should be Completed (submit is synchronous) ──
        r_status = _flowsim_cli("status", "--scheduler", "local", "--job", job_id)
        assert r_status.returncode == 0, "flowsim status failed"
        status_out = r_status.stdout.lower()
        assert "completed" in status_out, (
            f"Job {job_id} not completed:\n{r_status.stdout}"
        )

        # ── Step 4: validate trace CSVs ──
        # Extract output_dir from status output (Traces dir: ...)
        output_dir = None
        for line in r_status.stdout.splitlines():
            if "Traces dir:" in line:
                output_dir = line.split("Traces dir:", 1)[1].strip()
                break
        assert output_dir and os.path.isdir(output_dir), (
            f"Could not find traces dir in status output:\n{r_status.stdout}"
        )
        _assert_traces(output_dir)
        _assert_logs(output_dir)
        _validate_shapes(output_dir, bs=bs, input_len=input_len, existing_ctx=existing_ctx)


# =====================================================================
# Cluster setup helpers & fixtures
# =====================================================================

def _run_dev_setup(target: str) -> None:
    """Run ``dockerfiles/dev-setup.sh <target>`` and assert success."""
    r = subprocess.run(
        ["bash", _DEV_SETUP, target],
        capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"dev-setup.sh {target} failed (exit {r.returncode}):\n"
            f"stdout: {r.stdout[-2000:]}\nstderr: {r.stderr[-2000:]}"
        )


def _run_dev_teardown(target: str) -> None:
    """Run ``dockerfiles/dev-teardown.sh <target>``."""
    subprocess.run(
        ["bash", _DEV_TEARDOWN, target],
        capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=120,
    )


def _kind_cluster_running() -> bool:
    """Check if the Kind cluster named 'flowsim' is reachable."""
    try:
        r = subprocess.run(
            ["kubectl", "--context", "kind-flowsim", "get", "nodes"],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0 and "Ready" in r.stdout
    except Exception:
        return False


@pytest.fixture(scope="session")
def kind_cluster():
    """Ensure Kind cluster is running; auto-setup if needed.

    The cluster is kept alive after the test session to avoid
    re-loading the 34 GB image every time.  Use ``dev-teardown.sh kind``
    to clean up manually.
    """
    if not _kind_cluster_running():
        _run_dev_setup("kind")
    assert _kind_cluster_running(), "Kind cluster not reachable after setup"
    yield


@pytest.fixture(scope="session")
def slurm_cluster():
    """Ensure Slurm cluster is running; auto-setup if needed.

    Cluster is kept alive after tests.  Use ``dev-teardown.sh slurm``
    to clean up manually.
    """
    if not _slurm_cluster_running():
        _run_dev_setup("slurm")
    assert _slurm_cluster_running(), "Slurm cluster not reachable after setup"
    yield


# =====================================================================
# K8S SCHEDULER
# =====================================================================
class TestK8sScheduler:
    """K8s scheduler: real submit to Kind cluster.

    Automatically sets up the Kind cluster via ``dev-setup.sh`` if not
    already running.
    """

    def test_k8s_real_submit_to_kind(self, kind_cluster):
        """Submit a real Job to Kind cluster: submit → list → status → retrieve → validate."""
        import shutil
        import tempfile

        job_name = f"test-integ-{int(time.time()) % 100000}"
        local_traces = tempfile.mkdtemp(prefix="flowsim-k8s-traces-")

        try:
            # ── Step 0: clean stale test traces on host ──
            host_traces = os.path.join(_PROJECT_ROOT, "stage_traces")
            os.makedirs(host_traces, exist_ok=True)

            # ── Step 1: submit (host mount for trace retrieval) ──
            r = _flowsim_cli(
                "submit",
                "--scheduler", "k8s",
                "--collect", "all",
                "--model-path", MODEL,
                "--tp", "1",
                "--bs", "1",
                "--input-len", "2048",
                "--existing-ctx", "0",
                "--decode-tokens", "2",
                "--warmup-n", "2",
                "--gpus", "1",
                "--k8s-namespace", "default",
                "--k8s-host-output-dir", "/host-stage-traces",
                "--job-name", job_name,
                "--extra-server-opts", f"--load-format {LOAD_FORMAT}",
            )
            combined = r.stdout + r.stderr
            if r.returncode != 0:
                print("Submit output:", combined[-3000:])
            assert r.returncode == 0, f"K8s submit failed: {combined[-1000:]}"

            # ── Step 2: list — verify job appears ──
            r_list = _flowsim_cli("list", "--scheduler", "k8s")
            assert r_list.returncode == 0
            assert job_name in r_list.stdout, (
                f"Job {job_name} not in list:\n{r_list.stdout}"
            )

            # ── Step 3: status — poll until Completed/Succeeded (max 20 min) ──
            deadline = time.time() + 1200
            state = ""
            while time.time() < deadline:
                r_status = _flowsim_cli("status", "--scheduler", "k8s", "--job", job_name)
                assert r_status.returncode == 0
                state = r_status.stdout.lower()
                if "completed" in state or "succeeded" in state:
                    break
                if "failed" in state:
                    pytest.fail(f"K8s job failed:\n{r_status.stdout}")
                time.sleep(15)
            assert "completed" in state or "succeeded" in state, (
                f"K8s job did not complete in time:\n{r_status.stdout}"
            )

            # ── Step 4: traces are on host via Kind mount ──
            # output_dir inside container: /flowsim/stage_traces/k8s/{ts}
            # host_output_dir on worker: /host-stage-traces
            # → host: {project}/stage_traces/k8s/{ts}/
            k8s_traces = os.path.join(host_traces, "k8s")
            assert os.path.isdir(k8s_traces), (
                f"No k8s traces dir at {k8s_traces}"
            )
            # Find the latest timestamped subdir
            ts_dirs = sorted(os.listdir(k8s_traces))
            assert ts_dirs, f"No timestamp dirs in {k8s_traces}"
            local_traces = os.path.join(k8s_traces, ts_dirs[-1])

            # ── Step 5: validate trace CSVs ──
            _assert_traces(local_traces)
            _assert_logs(local_traces)
            _validate_shapes(local_traces, bs=1, input_len=2048, existing_ctx=0)

        finally:
            # Cleanup: cancel job (traces stay on host for inspection)
            _flowsim_cli("cancel", "--scheduler", "k8s", "--job", job_name)


# =====================================================================
# SLURM SCHEDULER
# =====================================================================

def _slurm_cluster_running() -> bool:
    """Check if local Slurm test cluster (docker compose) is running."""
    try:
        r = subprocess.run(
            ["docker", "exec", "slurmctld", "sinfo", "-h"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0 and r.stdout.strip() != ""
    except Exception:
        return False


# CLI prefix for running Slurm commands inside the slurmctld container.
# Uses -i so sbatch can read scripts from stdin.
_SLURM_CLI_PREFIX = "docker exec -i slurmctld"


class TestSlurmScheduler:
    """Slurm scheduler: real submit to local docker-compose cluster.

    Uses CLI mode (sbatch/squeue/scancel) — no slurmrestd needed.
    Automatically sets up the Slurm cluster via ``dev-setup.sh slurm``
    if not already running.
    """

    def test_slurm_real_submit(self, slurm_cluster):
        """Submit to local Slurm cluster: submit → list → status → retrieve → validate."""

        # Compute node has /flowsim/stage_traces mounted writable to host.
        # output_dir inside the container maps directly to the host.
        host_traces = os.path.join(_PROJECT_ROOT, "stage_traces")
        os.makedirs(host_traces, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"/flowsim/stage_traces/slurm/{ts}"

        job_id = None
        try:
            # ── Step 1: submit (CLI mode, container_runtime=none) ──
            r = _flowsim_cli(
                "submit",
                "--scheduler", "slurm",
                "--collect", "all",
                "--model-path", MODEL,
                "--tp", "1",
                "--bs", "1",
                "--input-len", "2048",
                "--existing-ctx", "0",
                "--decode-tokens", "2",
                "--warmup-n", "2",
                "--gpus", "1",
                "--slurm-partition", "normal",
                "--slurm-submit-via", "cli",
                "--slurm-cli-prefix", _SLURM_CLI_PREFIX,
                "--slurm-container-runtime", "none",
                "--output-dir", output_dir,
                "--extra-server-opts", f"--load-format {LOAD_FORMAT}",
            )
            combined = r.stdout + r.stderr
            if r.returncode != 0:
                print("Submit output:", combined[-3000:])
            assert r.returncode == 0, f"Slurm submit failed: {combined[-1000:]}"

            # Extract job_id from output (line like "Submitted batch job 123")
            for line in combined.splitlines():
                if "submitted" in line.lower():
                    for word in line.split():
                        if word.isdigit():
                            job_id = word
                            break
                if job_id:
                    break
            assert job_id, f"Could not find job_id in submit output:\n{combined[-1000:]}"

            # ── Step 2: status — poll until Completed (max 20 min) ──
            deadline = time.time() + 1200
            state = ""
            while time.time() < deadline:
                r_status = _flowsim_cli(
                    "status", "--scheduler", "slurm",
                    "--job", job_id,
                    "--slurm-submit-via", "cli",
                    "--slurm-cli-prefix", _SLURM_CLI_PREFIX,
                )
                assert r_status.returncode == 0
                state = r_status.stdout.lower()
                if "completed" in state or "succeeded" in state:
                    break
                if "failed" in state:
                    pytest.fail(f"Slurm job failed:\n{r_status.stdout}")
                time.sleep(15)
            assert "completed" in state or "succeeded" in state, (
                f"Slurm job did not complete in time:\n{r_status.stdout}"
            )

            # ── Step 3: traces are on host via mount ──
            slurm_traces = os.path.join(host_traces, "slurm")
            assert os.path.isdir(slurm_traces), (
                f"No slurm traces dir at {slurm_traces}"
            )
            ts_dirs = sorted(os.listdir(slurm_traces))
            assert ts_dirs, f"No test dirs in {slurm_traces}"
            local_traces = os.path.join(slurm_traces, ts_dirs[-1])

            # ── Step 4: validate trace CSVs ──
            _assert_traces(local_traces)
            _assert_logs(local_traces)
            _validate_shapes(local_traces, bs=1, input_len=2048, existing_ctx=0)

        finally:
            # Cleanup: cancel job (traces stay on host for inspection)
            if job_id:
                _flowsim_cli(
                    "cancel", "--scheduler", "slurm",
                    "--job", job_id,
                    "--slurm-submit-via", "cli",
                    "--slurm-cli-prefix", _SLURM_CLI_PREFIX,
                )
