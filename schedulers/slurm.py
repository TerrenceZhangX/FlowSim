"""Slurm sbatch scheduler for FlowSim profiling.

``render()`` / ``dry_run()`` produce a standalone bash script (zero deps).
``submit()`` pipes the script to ``sbatch`` via subprocess (CLI mode).

Requires ``sbatch``/``squeue``/``scancel`` on PATH (or reachable
via ``cli_prefix``, e.g. ``"docker exec slurmctld"``).
"""

from __future__ import annotations

import shlex
import subprocess

from schedulers.base import BaseScheduler, JobResult, ProfileJobSpec


class SlurmScheduler(BaseScheduler):
    """Generate and optionally submit an sbatch script for profiling.

    Parameters
    ----------
    partition : str
        Slurm partition to submit to.
    time_limit : str
        Wall-clock time limit (e.g., ``"01:00:00"``).
    account : str, optional
        ``--account`` for which allocation to charge.
    constraint : str, optional
        ``--constraint`` node feature (e.g., ``"gpu80g"``).
    container_runtime : str
        How to run the container inside the allocation.
        ``"docker"``  -> ``docker run``
        ``"enroot"``  -> ``srun --container-image``
        ``"none"``    -> run bare-metal (no container)
    container_mounts : str
        Bind-mount string passed to the container runtime
        (e.g., ``"/data:/data"``).
    modules : list[str]
        ``module load`` commands to run before the job
        (relevant for ``"none"`` runtime).
    extra_sbatch : list[str]
        Additional ``#SBATCH`` lines, each *without* the ``#SBATCH`` prefix.
    cli_prefix : str
        Shell prefix for CLI commands (e.g. ``"docker exec -i slurmctld"``).
    """

    def __init__(
        self,
        *,
        partition: str = "gpu",
        time_limit: str = "02:00:00",
        account: str = "",
        constraint: str = "",
        container_runtime: str = "none",
        container_mounts: str = "",
        modules: list[str] | None = None,
        extra_sbatch: list[str] | None = None,
        cli_prefix: str = "",
    ) -> None:
        self.partition = partition
        self.time_limit = time_limit
        self.account = account
        self.constraint = constraint
        self.container_runtime = container_runtime
        self.container_mounts = container_mounts
        self.modules = modules or []
        self.extra_sbatch = extra_sbatch or []
        self.cli_prefix = cli_prefix

    def render(self, spec: ProfileJobSpec) -> str:
        job_name = spec.default_job_name()
        cmd = spec.build_shell_command()

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --gpus-per-node={spec.gpus}",
            f"#SBATCH --ntasks=1",
            f"#SBATCH --time={self.time_limit}",
            f"#SBATCH --output={spec.output_dir}/{job_name}_%j.log",
        ]

        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.constraint:
            lines.append(f"#SBATCH --constraint={self.constraint}")
        for extra in self.extra_sbatch:
            lines.append(f"#SBATCH {extra}")

        lines.append("")
        lines.append("set -euo pipefail")
        lines.append("")

        # Ensure output dir exists (needed for #SBATCH --output)
        lines.append(f"mkdir -p {spec.output_dir}")
        lines.append("")

        if self.modules:
            for mod in self.modules:
                lines.append(f"module load {mod}")
            lines.append("")

        lines.append("export SGLANG_PROFILE_KERNELS=1")
        lines.append("")

        if self.container_runtime == "docker":
            mounts = ""
            if self.container_mounts:
                mounts = f" -v {self.container_mounts}"
            lines.append(
                f"docker run --gpus all --ipc=host --shm-size=16g"
                f"{mounts} -w /flowsim {spec.image} \\"
            )
            lines.append(f"  {cmd}")
        elif self.container_runtime == "enroot":
            mounts = ""
            if self.container_mounts:
                mounts = f" --container-mounts={self.container_mounts}"
            lines.append(
                f"srun --container-image={spec.image}"
                f" --container-workdir=/flowsim"
                f"{mounts} \\"
            )
            lines.append(f"  {cmd}")
        elif self.container_runtime == "none":
            lines.append(f"cd /flowsim")
            lines.append(cmd)
        else:
            raise ValueError(
                f"Unknown container_runtime: {self.container_runtime!r}. "
                "Choose from: docker, enroot, none"
            )

        lines.append("")
        return "\n".join(lines)

    def submit(self, spec: ProfileJobSpec) -> JobResult:
        """Submit the job via ``sbatch``."""
        return self._submit_cli(spec)

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    def _cli_cmd(self, *args: str) -> list[str]:
        """Build a command list, prepending ``cli_prefix`` if set."""
        prefix = shlex.split(self.cli_prefix) if self.cli_prefix else []
        return prefix + list(args)

    def _cli_run(
        self,
        *args: str,
        input_data: str | None = None,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess:
        """Run a Slurm CLI command and return the CompletedProcess."""
        cmd = self._cli_cmd(*args)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=input_data,
            timeout=timeout,
        )

    def _submit_cli(self, spec: ProfileJobSpec) -> JobResult:
        """Submit via ``sbatch`` (piping the script on stdin)."""
        script = self.render(spec)
        job_name = spec.default_job_name()

        r = self._cli_run("sbatch", "--parsable", input_data=script, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (exit {r.returncode}):\n{r.stderr}"
            )

        job_id = r.stdout.strip().split(";")[
            0
        ]  # parsable: "jobid" or "jobid;cluster"
        return JobResult(
            job_id=job_id,
            scheduler="slurm",
            state="Submitted",
            output_dir=spec.output_dir,
            message=f"Submitted batch job {job_id}",
        )

    def cancel(self, job_id: str) -> str:
        """Cancel a Slurm job."""
        return self._cancel_cli(job_id)

    def status(self, job_id: str) -> dict:
        """Query Slurm job status."""
        return self._status_cli(job_id)

    def logs(
        self, job_id: str, *, tail: int = 100, follow: bool = False
    ) -> str:
        """Show Slurm job log information."""
        return self._logs_cli(job_id, tail=tail, follow=follow)

    def list_jobs(self, *, status_filter: str = "") -> list[dict]:
        """List Slurm jobs."""
        return self._list_jobs_cli(status_filter=status_filter)

    # ------------------------------------------------------------------
    # CLI implementations
    # ------------------------------------------------------------------

    def _cancel_cli(self, job_id: str) -> str:
        r = self._cli_run("scancel", job_id)
        if r.returncode != 0:
            raise RuntimeError(f"scancel failed: {r.stderr}")
        return f"Cancelled Slurm job {job_id}"

    def _status_cli(self, job_id: str) -> dict:
        # Use scontrol show job — works for both running and completed jobs
        # (completed jobs stay in memory for MinJobAge seconds, default 300s)
        r = self._cli_run("scontrol", "show", "job", job_id)
        if r.returncode != 0 or not r.stdout.strip():
            return {
                "state": "Unknown",
                "message": f"No job found with ID {job_id}",
                "output_hint": "",
            }

        # Parse key=value output
        fields: dict[str, str] = {}
        for token in r.stdout.replace("\n", " ").split():
            if "=" in token:
                k, _, v = token.partition("=")
                fields[k] = v

        state = fields.get("JobState", "UNKNOWN")
        name = fields.get("JobName", "")
        nodes = fields.get("NodeList", "")
        output_file = fields.get("StdOut", "")

        # Normalize Slurm uppercase states to capitalized format
        _STATE_MAP = {
            "PENDING": "Pending",
            "RUNNING": "Running",
            "SUSPENDED": "Suspended",
            "COMPLETED": "Completed",
            "CANCELLED": "Cancelled",
            "FAILED": "Failed",
            "TIMEOUT": "Timeout",
            "NODE_FAIL": "Failed",
            "PREEMPTED": "Preempted",
            "OUT_OF_MEMORY": "Failed",
        }
        state = _STATE_MAP.get(state, state)

        msg_parts = [
            f"Job ID: {job_id}  Name: {name}  State: {state}",
            f"Nodes: {nodes}" if nodes else "Nodes: (not yet assigned)",
        ]
        if output_file:
            msg_parts.append(f"Output log: {output_file}")

        return {
            "state": state,
            "message": "\n".join(msg_parts),
            "output_hint": output_file,
        }

    def _logs_cli(
        self, job_id: str, *, tail: int = 100, follow: bool = False
    ) -> str:
        info = self._status_cli(job_id)
        output_file = info.get("output_hint", "")

        if not output_file:
            return info["message"] + "\n(no log file path found)"

        # Try to read the log file via CLI prefix (handles remote Slurm)
        if follow:
            return (
                f"{info['message']}\n\n"
                f"Follow logs:\n"
                f"  tail -f {output_file}"
            )

        r = self._cli_run("tail", f"-{tail}", output_file, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout

        # Fallback: file may not exist yet or be on a remote node
        return (
            f"{info['message']}\n\n"
            f"Log file: {output_file}\n"
            f"View on login node:\n"
            f"  tail -{tail} {output_file}\n"
            f"Follow:\n"
            f"  tail -f {output_file}"
        )

    def _list_jobs_cli(self, *, status_filter: str = "") -> list[dict]:
        r = self._cli_run(
            "squeue",
            "-o",
            "%i|%j|%T|%P|%N",
            "-h",
        )
        if r.returncode != 0:
            raise RuntimeError(f"squeue failed: {r.stderr}")
        result: list[dict] = []
        for line in r.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split("|", 4)
            name = parts[1] if len(parts) > 1 else ""
            if not name.startswith("flowsim-"):
                continue
            state = parts[2] if len(parts) > 2 else "UNKNOWN"
            if status_filter and state.upper() != status_filter.upper():
                continue
            result.append(
                {
                    "job_id": parts[0] if parts else "",
                    "name": name,
                    "state": state,
                    "partition": parts[3] if len(parts) > 3 else "",
                    "nodes": parts[4] if len(parts) > 4 else "",
                }
            )
        return result
