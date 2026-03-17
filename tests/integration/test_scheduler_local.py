"""Integration tests for ``flowsim submit``, ``flowsim status``, ``flowsim logs``.

Tests all three scheduler backends (local, k8s, slurm) end-to-end.

* **local** — runs real TP=1 profiling and verifies traces, parsed CSVs,
  and log files are all produced in the correct locations.
* **k8s**   — submits a real Job to a Kind cluster, verifies it was created,
  then checks ``flowsim status`` / ``flowsim logs`` output.  Also validates
  that dry-run YAML has the correct volume mounts and log paths.
* **slurm** — dry-run only; verifies the sbatch script has the correct
  ``output_dir``, ``--log-dir``, and ``#SBATCH --output`` directives.

Requirements
------------
* The ``flowsim-test`` container with GPUs (for local tests).
* A Kind cluster named ``flowsim`` (for K8s tests).
* ``schedulers/`` available on PYTHONPATH.

Environment Variables
---------------------
``MODEL``
    Model path (default: ``/flowsim/workload/models/configs/Qwen3-235B-A22B``).
``LOAD_FORMAT``
    Load format (default: ``dummy``).

Usage
-----
    # Inside container (local tests):
    docker exec flowsim-test python -m pytest \
        tests/integration/test_scheduler_local.py -v -x

    # On host (k8s tests — needs kubeconfig):
    python -m pytest tests/integration/test_scheduler_local.py \
        -v -x -k "k8s"
"""

import glob
import os
import subprocess
import sys
import time

import pytest

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL = os.environ.get(
    "MODEL", "/flowsim/workload/models/configs/Qwen3-235B-A22B"
)
LOAD_FORMAT = os.environ.get("LOAD_FORMAT", "dummy")
ARTIFACT_DIR = os.environ.get(
    "PYTEST_ARTIFACT_DIR", "/flowsim/tests/test-artifacts"
)

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


# =====================================================================
# LOCAL SCHEDULER — real profiling
# =====================================================================
class TestLocalScheduler:
    """Run real profiling via ``flowsim submit --scheduler local``."""

    def test_local_perf_tp1(self):
        """TP=1 perf profiling: traces + parsed CSVs + log files."""
        output_dir = os.path.join(ARTIFACT_DIR, "sched_local_tp1")

        r = _flowsim_cli(
            "submit",
            "--scheduler", "local",
            "--collect", "perf",
            "--model-path", MODEL,
            "--tp", "1",
            "--bs", "1",
            "--input-len", "512",
            "--decode-tokens", "8",
            "--warmup-n", "2",
            "--gpus", "1",
            "--local-gpus", "0",
            "--output-dir", output_dir,
            "--extra-server-opts", f"--load-format {LOAD_FORMAT}",
        )

        if r.returncode != 0:
            print("STDOUT:", r.stdout[-3000:])
            print("STDERR:", r.stderr[-3000:])
        assert r.returncode == 0, f"flowsim submit failed (exit {r.returncode})"

        # Verify traces and parsed CSVs
        _assert_traces(output_dir)

        # Verify log files under output_dir/logs/
        _assert_logs(output_dir)

        # Verify submit output mentions log/trace locations
        combined = r.stdout + r.stderr
        assert "Traces:" in combined, "Submit output should show trace location"
        assert "Logs:" in combined, "Submit output should show log location"

    def test_local_status(self):
        """flowsim status --scheduler local should find logs from the previous run."""
        r = _flowsim_cli(
            "status",
            "--scheduler", "local",
            "--job", "flowsim-perf",
        )
        # Should either find logs or say not found — should not crash
        assert r.returncode == 0

    def test_local_logs(self):
        """flowsim logs --scheduler local should list log files and give paths."""
        r = _flowsim_cli(
            "logs",
            "--scheduler", "local",
            "--job", "flowsim-perf",
        )
        assert r.returncode == 0
        output = r.stdout
        # Should contain file listing or "No logs" — not crash
        assert "Log directory:" in output or "No logs" in output


# =====================================================================
# K8S SCHEDULER
# =====================================================================
class TestK8sScheduler:
    """K8s scheduler: dry-run validates YAML structure, real submit to Kind."""

    def test_k8s_dry_run_has_volume_and_log_path(self):
        """Dry-run YAML should mount output volume and pass --log-dir."""
        r = _flowsim_cli(
            "submit",
            "--scheduler", "k8s",
            "--collect", "perf",
            "--model-path", MODEL,
            "--k8s-namespace", "default",
            "--k8s-pvc", "test-traces",
            "--output-dir", "/data/traces",
            "--dry-run",
        )
        assert r.returncode == 0
        yaml_output = r.stdout

        # Job structure
        assert "apiVersion: batch/v1" in yaml_output
        assert "kind: Job" in yaml_output

        # PVC volume mount
        assert "test-traces" in yaml_output
        assert "persistentVolumeClaim" in yaml_output

        # output_dir and derived log_dir appear in the command
        assert "--output-dir" in yaml_output
        assert "/data/traces" in yaml_output
        assert "--log-dir" in yaml_output
        assert "/data/traces/logs" in yaml_output

    def test_k8s_dry_run_hostpath(self):
        """Dry-run with hostPath should have hostPath volume."""
        r = _flowsim_cli(
            "submit",
            "--scheduler", "k8s",
            "--collect", "perf",
            "--model-path", MODEL,
            "--k8s-namespace", "default",
            "--k8s-host-output-dir", "/mnt/traces",
            "--dry-run",
        )
        assert r.returncode == 0
        assert "hostPath" in r.stdout
        assert "/mnt/traces" in r.stdout

    def test_k8s_refuses_without_storage(self):
        """Submit (not dry-run) without PVC or hostPath should fail."""
        r = _flowsim_cli(
            "submit",
            "--scheduler", "k8s",
            "--collect", "perf",
            "--model-path", MODEL,
            "--k8s-namespace", "default",
            # Explicitly clear any config defaults
            "--k8s-pvc", "",
            "--k8s-host-output-dir", "",
        )
        assert r.returncode != 0
        combined = r.stdout + r.stderr
        assert "persistent storage" in combined or "pvc" in combined.lower()

    @pytest.mark.skipif(
        not os.path.exists(os.path.expanduser("~/.kube/config")),
        reason="No kubeconfig — skip K8s real submit (run on host with Kind cluster)",
    )
    def test_k8s_real_submit_to_kind(self):
        """Submit a real Job to Kind cluster, verify status + logs commands work."""
        job_name = f"test-integ-{int(time.time()) % 100000}"
        r = _flowsim_cli(
            "submit",
            "--scheduler", "k8s",
            "--collect", "perf",
            "--model-path", MODEL,
            "--k8s-namespace", "default",
            "--k8s-host-output-dir", "/tmp/flowsim-test-traces",
            "--job-name", job_name,
        )
        combined = r.stdout + r.stderr

        if r.returncode != 0:
            print("Submit output:", combined[-3000:])
        assert r.returncode == 0, f"K8s submit failed: {combined[-1000:]}"
        assert "created" in combined.lower()

        # Verify submit output has location hints
        assert "Traces:" in combined
        assert "Logs:" in combined
        assert "flowsim status" in combined
        assert "flowsim logs" in combined

        # Check status
        r2 = _flowsim_cli("status", "--scheduler", "k8s", "--job", job_name)
        assert r2.returncode == 0
        assert job_name in r2.stdout

        # Check logs (may say "pending" or show pod info)
        r3 = _flowsim_cli("logs", "--scheduler", "k8s", "--job", job_name)
        assert r3.returncode == 0
        # Should mention kubectl or pod name or "No pods"
        assert "kubectl" in r3.stdout or "No pods" in r3.stdout or "Pod:" in r3.stdout

        # Cleanup: delete the K8s job
        subprocess.run(
            ["kubectl", "--context", "kind-flowsim", "delete", "job", job_name,
             "-n", "default", "--ignore-not-found"],
            capture_output=True, timeout=30,
        )


# =====================================================================
# SLURM SCHEDULER — dry-run only (no real cluster)
# =====================================================================
class TestSlurmScheduler:
    """Slurm scheduler: verify sbatch script has correct paths."""

    def test_slurm_dry_run_output_and_log_paths(self):
        """Dry-run sbatch script should reference output_dir and log_dir."""
        r = _flowsim_cli(
            "submit",
            "--scheduler", "slurm",
            "--collect", "perf",
            "--model-path", MODEL,
            "--slurm-partition", "gpu",
            "--slurm-rest-url", "http://fake:6820",
            "--slurm-jwt-token", "fake-token",
            "--output-dir", "/shared/flowsim_traces",
            "--dry-run",
        )
        assert r.returncode == 0
        script = r.stdout

        # sbatch directives
        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=" in script
        assert "#SBATCH --partition=gpu" in script

        # output_dir in the profiling command
        assert "--output-dir" in script
        assert "/shared/flowsim_traces" in script

        # log_dir = output_dir + /logs/
        assert "--log-dir" in script
        assert "/shared/flowsim_traces/logs" in script

    def test_slurm_dry_run_default_output_dir(self):
        """Default output_dir for Slurm should be ~/flowsim_traces."""
        r = _flowsim_cli(
            "submit",
            "--scheduler", "slurm",
            "--collect", "perf",
            "--model-path", MODEL,
            "--slurm-partition", "gpu",
            "--slurm-rest-url", "http://fake:6820",
            "--slurm-jwt-token", "fake-token",
            "--dry-run",
        )
        assert r.returncode == 0
        assert "flowsim_traces" in r.stdout

    def test_slurm_dry_run_pd_pair(self):
        """PD disaggregation dry-run should produce both scripts with correct paths."""
        r = _flowsim_cli(
            "submit",
            "--scheduler", "slurm",
            "--collect", "perf",
            "--model-path", MODEL,
            "--slurm-partition", "gpu",
            "--slurm-rest-url", "http://fake:6820",
            "--slurm-jwt-token", "fake-token",
            "--output-dir", "/shared/traces",
            "--pd",
            "--dry-run",
        )
        assert r.returncode == 0
        output = r.stdout
        assert "PREFILL INSTANCE" in output
        assert "DECODE INSTANCE" in output
        assert "--disaggregation-mode prefill" in output
        assert "--disaggregation-mode decode" in output
        # Both scripts should reference the same output_dir
        assert output.count("--output-dir") >= 2
        assert output.count("/shared/traces/logs") >= 2
