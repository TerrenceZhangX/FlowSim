"""Integration tests for `flowsim submit --scheduler local`.

Runs actual profiling jobs inside the FlowSim Docker container and verifies
that traces and parsed CSVs are produced.

Requirements
------------
* Running inside the ``flowsim`` Docker container with GPUs.
* ``pip install -e .`` done (or schedulers/ available on PYTHONPATH).

Environment Variables
---------------------
``MODEL``
    Model path (default: ``/flowsim/workload/models/configs/Qwen3-235B-A22B``).
``LOAD_FORMAT``
    Load format (default: ``dummy``).

Usage
-----
    docker exec flowsim-test python -m pytest tests/integration/test_scheduler_local.py -v -x
"""

import glob
import os
import subprocess
import sys

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


def _flowsim_submit(*args: str, timeout: int = 1200) -> subprocess.CompletedProcess:
    """Run ``flowsim submit`` via Python entry point."""
    cmd = [
        sys.executable, "-u", "-c",
        "from scripts.cli import main; main()",
        "submit", *args,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = _PROJECT_ROOT + (
        ":" + env.get("PYTHONPATH", "")
    )
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
        env=env,
        timeout=timeout,
    )
    return result


class TestLocalSubmitPerf:
    """flowsim submit --scheduler local --collect perf — runs real profiling."""

    def test_local_perf_tp1(self):
        """Single-GPU perf profiling via flowsim submit."""
        output_dir = os.path.join(ARTIFACT_DIR, "local_perf_tp1")
        log_dir = os.path.join(ARTIFACT_DIR, "local_perf_tp1_logs")

        r = _flowsim_submit(
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
            "--log-dir", log_dir,
            "--extra-server-opts", f"--load-format {LOAD_FORMAT}",
        )

        # Print output for debugging
        if r.returncode != 0:
            print("STDOUT:", r.stdout[-3000:])
            print("STDERR:", r.stderr[-3000:])
        assert r.returncode == 0, f"flowsim submit failed (exit {r.returncode})"

        # Verify trace files exist
        traces = glob.glob(
            os.path.join(output_dir, "**/*.trace.json.gz"), recursive=True
        )
        assert len(traces) > 0, f"No trace files under {output_dir}"

        extend = [t for t in traces if "EXTEND" in os.path.basename(t)]
        decode = [t for t in traces if "DECODE" in os.path.basename(t)]
        assert len(extend) > 0, "No EXTEND traces"
        assert len(decode) > 0, "No DECODE traces"

        # Verify parsed CSVs
        csvs = glob.glob(
            os.path.join(output_dir, "**/parsed/*.csv"), recursive=True
        )
        assert len(csvs) > 0, f"No parsed CSVs under {output_dir}"

    def test_local_perf_tp2(self):
        """Multi-GPU perf profiling (TP=2) via flowsim submit."""
        output_dir = os.path.join(ARTIFACT_DIR, "local_perf_tp2")
        log_dir = os.path.join(ARTIFACT_DIR, "local_perf_tp2_logs")

        r = _flowsim_submit(
            "--scheduler", "local",
            "--collect", "perf",
            "--model-path", MODEL,
            "--tp", "2",
            "--bs", "1",
            "--input-len", "1024",
            "--decode-tokens", "8",
            "--warmup-n", "2",
            "--gpus", "2",
            "--local-gpus", "0,1",
            "--output-dir", output_dir,
            "--log-dir", log_dir,
            "--extra-server-opts", f"--load-format {LOAD_FORMAT}",
        )

        if r.returncode != 0:
            print("STDOUT:", r.stdout[-3000:])
            print("STDERR:", r.stderr[-3000:])
        assert r.returncode == 0, f"flowsim submit failed (exit {r.returncode})"

        traces = glob.glob(
            os.path.join(output_dir, "**/*.trace.json.gz"), recursive=True
        )
        assert len(traces) > 0, f"No trace files under {output_dir}"

        # TP=2 should produce traces for both ranks
        tp0 = [t for t in traces if "TP-0" in os.path.basename(t)]
        tp1 = [t for t in traces if "TP-1" in os.path.basename(t)]
        assert len(tp0) > 0, "No TP-0 traces"
        assert len(tp1) > 0, "No TP-1 traces"


class TestLocalSubmitDryRun:
    """flowsim submit --scheduler local --dry-run — verify command generation."""

    def test_dry_run_output(self):
        r = _flowsim_submit(
            "--scheduler", "local",
            "--collect", "perf",
            "--model-path", MODEL,
            "--tp", "2",
            "--local-gpus", "0,1",
            "--dry-run",
        )
        assert r.returncode == 0
        assert "CUDA_VISIBLE_DEVICES=0,1" in r.stdout
        assert "scripts/run_stage_profile.py" in r.stdout
        assert "--tp 2" in r.stdout

    def test_dry_run_pd(self):
        r = _flowsim_submit(
            "--scheduler", "local",
            "--collect", "perf",
            "--model-path", MODEL,
            "--pd",
            "--dry-run",
        )
        assert r.returncode == 0
        assert "PREFILL INSTANCE" in r.stdout
        assert "DECODE INSTANCE" in r.stdout
        assert "--disaggregation-mode prefill" in r.stdout
        assert "--disaggregation-mode decode" in r.stdout


class TestK8sSubmitDryRun:
    """flowsim submit --scheduler k8s --dry-run — verify YAML generation."""

    def test_k8s_dry_run(self):
        r = _flowsim_submit(
            "--scheduler", "k8s",
            "--collect", "perf",
            "--model-path", MODEL,
            "--k8s-namespace", "default",
            "--dry-run",
        )
        assert r.returncode == 0
        assert "apiVersion: batch/v1" in r.stdout
        assert "kind: Job" in r.stdout
        assert MODEL in r.stdout

    def test_k8s_pd_dry_run(self):
        r = _flowsim_submit(
            "--scheduler", "k8s",
            "--collect", "perf",
            "--model-path", MODEL,
            "--k8s-namespace", "default",
            "--pd",
            "--dry-run",
        )
        assert r.returncode == 0
        assert "PREFILL INSTANCE" in r.stdout
        assert "DECODE INSTANCE" in r.stdout


class TestSlurmSubmitDryRun:
    """flowsim submit --scheduler slurm --dry-run — verify sbatch script."""

    def test_slurm_dry_run(self):
        r = _flowsim_submit(
            "--scheduler", "slurm",
            "--collect", "perf",
            "--model-path", MODEL,
            "--slurm-partition", "gpu",
            "--slurm-rest-url", "http://fake:6820",
            "--slurm-jwt-token", "fake",
            "--dry-run",
        )
        assert r.returncode == 0
        assert "#!/bin/bash" in r.stdout
        assert "#SBATCH --partition=gpu" in r.stdout
        assert MODEL in r.stdout
