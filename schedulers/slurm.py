"""Slurm sbatch scheduler for FlowSim profiling.

``render()`` / ``dry_run()`` produce a standalone bash script (zero deps).
``submit()`` posts the script to a slurmrestd endpoint via stdlib
``urllib.request`` — no extra packages needed.

Two submission modes are supported:

* **cli** (default) — pipe the script to ``sbatch`` via subprocess.
  Requires ``sbatch``/``squeue``/``scancel`` on PATH (or reachable
  via ``cli_prefix``, e.g. ``"docker exec slurmctld"``).
* **rest** (deprecated) — POST the script to a slurmrestd endpoint.
  Requires ``rest_url`` and ``jwt_token``.
"""

from __future__ import annotations

import json
import shlex
import ssl
import subprocess
import urllib.error
import urllib.request

from schedulers.base import BaseScheduler, JobResult, ProfileJobSpec


_DEFAULT_API_VERSION = "v0.0.40"


class SlurmScheduler(BaseScheduler):
    """Generate and optionally submit an sbatch script for profiling.

    Parameters
    ----------
    partition : str
        Slurm partition to submit to.
    time_limit : str
        Wall-clock time limit (e.g., ``"01:00:00"``).
    rest_url : str
        Base URL of the slurmrestd daemon
        (e.g., ``"https://slurm.example.com:6820"``).
        Required only for ``submit()``.
    jwt_token : str
        JWT/auth token for slurmrestd.  Required only for ``submit()``.
    api_version : str
        slurmrestd OpenAPI version (default: ``"v0.0.40"``).
        Adjust to match your cluster (``v0.0.39``, ``v0.0.41``, …).
    verify_ssl : bool
        Whether to verify the slurmrestd TLS certificate (default True).
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
    submit_via : str
        ``"cli"``  (default) — use ``sbatch`` / ``squeue`` / ``scancel`` subprocess.
        ``"rest"`` (deprecated) — use slurmrestd REST API.
    cli_prefix : str
        Shell prefix for CLI commands (e.g. ``"docker exec -i slurmctld"``).
        Only used when ``submit_via="cli"``.
    """

    def __init__(
        self,
        *,
        partition: str = "gpu",
        time_limit: str = "02:00:00",
        rest_url: str = "",
        jwt_token: str = "",
        api_version: str = _DEFAULT_API_VERSION,
        verify_ssl: bool = True,
        account: str = "",
        constraint: str = "",
        container_runtime: str = "none",
        container_mounts: str = "",
        modules: list[str] | None = None,
        extra_sbatch: list[str] | None = None,
        submit_via: str = "cli",
        cli_prefix: str = "",
    ) -> None:
        self.partition = partition
        self.time_limit = time_limit
        self.rest_url = rest_url.rstrip("/")
        self.jwt_token = jwt_token
        self.api_version = api_version
        self.verify_ssl = verify_ssl
        self.account = account
        self.constraint = constraint
        self.container_runtime = container_runtime
        self.container_mounts = container_mounts
        self.modules = modules or []
        self.extra_sbatch = extra_sbatch or []
        self.submit_via = submit_via
        self.cli_prefix = cli_prefix

        if self.submit_via != "cli":
            import warnings
            warnings.warn(
                "Slurm REST mode (slurmrestd) is deprecated and will be "
                "removed in a future release. Use submit_via='cli' "
                "(sbatch) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

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
        """Submit the job via REST API or CLI, depending on ``submit_via``."""
        if self.submit_via == "cli":
            return self._submit_cli(spec)
        return self._submit_rest(spec)

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

        job_id = r.stdout.strip().split(";")[0]  # parsable: "jobid" or "jobid;cluster"
        return JobResult(
            job_id=job_id,
            scheduler="slurm",
            state="Submitted",
            output_dir=spec.output_dir,
            message=f"Submitted batch job {job_id}",
        )

    # ------------------------------------------------------------------
    # REST submit
    # ------------------------------------------------------------------

    def _submit_rest(self, spec: ProfileJobSpec) -> JobResult:
        """Submit the job via slurmrestd REST API.

        Requires ``rest_url`` and ``jwt_token`` to be set.
        Uses only ``urllib.request`` from the standard library.
        """
        if not self.rest_url:
            raise RuntimeError(
                "--slurm-rest-url is required for --submit. "
                "Point it at your slurmrestd endpoint "
                "(e.g. https://slurm.example.com:6820)."
            )
        if not self.jwt_token:
            raise RuntimeError(
                "--slurm-jwt-token is required for --submit. "
                "Generate one via: scontrol token lifespan=3600"
            )

        script = self.render(spec)
        job_name = spec.default_job_name()

        url = (
            f"{self.rest_url}/slurm/{self.api_version}/job/submit"
        )

        # slurmrestd job submission payload
        payload = {
            "script": script,
            "job": {
                "name": job_name,
                "partition": self.partition,
                "time_limit": {"number": self._parse_time_minutes(), "set": True},
                "tasks": 1,
                "current_working_directory": "/flowsim",
                "environment": ["PATH=/usr/local/bin:/usr/bin:/bin"],
            },
        }
        if self.account:
            payload["job"]["account"] = self.account

        data = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "X-SLURM-USER-TOKEN": self.jwt_token,
        }
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        ctx: ssl.SSLContext | None = None
        if not self.verify_ssl:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        try:
            with urllib.request.urlopen(req, context=ctx) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"slurmrestd returned HTTP {exc.code}:\n{detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach slurmrestd at {self.rest_url}: {exc.reason}"
            ) from exc

        # Response contains job_id on success, errors array on failure
        errors = body.get("errors") or []
        if errors:
            msgs = "; ".join(e.get("error", str(e)) for e in errors)
            raise RuntimeError(f"slurmrestd job submit failed: {msgs}")

        job_id = str(body.get("job_id", "unknown"))
        return JobResult(
            job_id=job_id,
            scheduler="slurm",
            state="Submitted",
            output_dir=spec.output_dir,
            message=f"Submitted batch job {job_id}",
        )

    def _rest_request(self, path: str, *, method: str = "GET") -> dict:
        """Send a request to slurmrestd and return parsed JSON."""
        if not self.rest_url:
            raise RuntimeError("--slurm-rest-url is required")
        if not self.jwt_token:
            raise RuntimeError("--slurm-jwt-token is required")

        url = f"{self.rest_url}{path}"
        headers = {
            "X-SLURM-USER-TOKEN": self.jwt_token,
        }
        req = urllib.request.Request(url, headers=headers, method=method)

        ctx: ssl.SSLContext | None = None
        if not self.verify_ssl:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        try:
            with urllib.request.urlopen(req, context=ctx) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"slurmrestd returned HTTP {exc.code}:\n{detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach slurmrestd at {self.rest_url}: {exc.reason}") from exc

    def _rest_get(self, path: str) -> dict:
        """GET a slurmrestd endpoint and return parsed JSON."""
        return self._rest_request(path, method="GET")

    def cancel(self, job_id: str) -> str:
        """Cancel a Slurm job."""
        if self.submit_via == "cli":
            return self._cancel_cli(job_id)
        return self._cancel_rest(job_id)

    def status(self, job_id: str) -> dict:
        """Query Slurm job status."""
        if self.submit_via == "cli":
            return self._status_cli(job_id)
        return self._status_rest(job_id)

    def logs(self, job_id: str, *, tail: int = 100, follow: bool = False) -> str:
        """Show Slurm job log information."""
        if self.submit_via == "cli":
            return self._logs_cli(job_id, tail=tail, follow=follow)
        return self._logs_rest(job_id, tail=tail, follow=follow)

    def list_jobs(self, *, status_filter: str = "") -> list[dict]:
        """List Slurm jobs."""
        if self.submit_via == "cli":
            return self._list_jobs_cli(status_filter=status_filter)
        return self._list_jobs_rest(status_filter=status_filter)

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
            return {"state": "Unknown", "message": f"No job found with ID {job_id}", "output_hint": ""}

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

        # Normalize to match test expectations
        if state == "COMPLETED":
            state = "Completed"
        elif state == "FAILED":
            state = "Failed"

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

    def _logs_cli(self, job_id: str, *, tail: int = 100, follow: bool = False) -> str:
        # TODO: read actual Slurm log file (StdOut from scontrol)
        # and support tail/follow properly.
        info = self._status_cli(job_id)
        return info["message"]

    def _list_jobs_cli(self, *, status_filter: str = "") -> list[dict]:
        r = self._cli_run(
            "squeue", "-o", "%i|%j|%T|%P|%N", "-h",
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
            result.append({
                "job_id": parts[0] if parts else "",
                "name": name,
                "state": state,
                "partition": parts[3] if len(parts) > 3 else "",
                "nodes": parts[4] if len(parts) > 4 else "",
            })
        return result

    # ------------------------------------------------------------------
    # REST implementations
    # ------------------------------------------------------------------

    def _cancel_rest(self, job_id: str) -> str:
        """Cancel a Slurm job via slurmrestd DELETE."""
        body = self._rest_request(
            f"/slurm/{self.api_version}/job/{job_id}",
            method="DELETE",
        )
        errors = body.get("errors") or []
        if errors:
            msgs = "; ".join(e.get("error", str(e)) for e in errors)
            raise RuntimeError(f"slurmrestd cancel failed: {msgs}")
        return f"Cancelled Slurm job {job_id}"

    def _status_rest(self, job_id: str) -> dict:
        """Query Slurm job status via slurmrestd."""
        body = self._rest_get(f"/slurm/{self.api_version}/job/{job_id}")

        errors = body.get("errors") or []
        if errors:
            msgs = "; ".join(e.get("error", str(e)) for e in errors)
            raise RuntimeError(f"slurmrestd error: {msgs}")

        jobs = body.get("jobs", [])
        if not jobs:
            return {"state": "Unknown", "message": f"No job found with ID {job_id}", "output_hint": ""}

        job = jobs[0]
        state = job.get("job_state", ["UNKNOWN"])
        if isinstance(state, list):
            state = state[0] if state else "UNKNOWN"
        name = job.get("name", "")
        node_list = job.get("nodes", "")
        output_file = job.get("standard_output", "")
        work_dir = job.get("current_working_directory", "")

        msg_parts = [
            f"Job ID: {job_id}  Name: {name}  State: {state}",
            f"Nodes: {node_list}" if node_list else "Nodes: (not yet assigned)",
        ]
        if output_file:
            msg_parts.append(f"Output log: {output_file}")
        if work_dir:
            msg_parts.append(f"Working dir: {work_dir}")

        return {
            "state": state,
            "message": "\n".join(msg_parts),
            "output_hint": output_file,
        }

    def _logs_rest(self, job_id: str, *, tail: int = 100, follow: bool = False) -> str:
        """Show where Slurm job logs are and how to access them."""
        info = self._status_rest(job_id)
        output_file = info.get("output_hint", "")
        state = info.get("state", "UNKNOWN")

        parts = [info["message"], ""]

        if output_file:
            parts.append(f"Log file (on cluster shared filesystem):")
            parts.append(f"  {output_file}")
            parts.append("")
            if follow:
                parts.append("Follow logs:")
                parts.append(f"  tail -f {output_file}")
            else:
                parts.append("View on login node:")
                parts.append(f"  less {output_file}")
                parts.append(f"  tail -{tail} {output_file}")
                parts.append("")
                parts.append("Follow logs:")
                parts.append(f"  tail -f {output_file}")
            parts.append("")
            parts.append("Copy to local machine:")
            parts.append(f"  scp <login-node>:{output_file} .")
        else:
            parts.append("No output file path found in job metadata.")

        # Trace files location
        parts.append("")
        parts.append("Trace files (on cluster shared filesystem):")
        parts.append("  ~/flowsim_traces/")
        parts.append("  ls ~/flowsim_traces/")

        return "\n".join(parts)

    def _list_jobs_rest(self, *, status_filter: str = "") -> list[dict]:
        """List Slurm jobs via slurmrestd /jobs endpoint."""
        body = self._rest_get(f"/slurm/{self.api_version}/jobs")
        errors = body.get("errors") or []
        if errors:
            msgs = "; ".join(e.get("error", str(e)) for e in errors)
            raise RuntimeError(f"slurmrestd error: {msgs}")

        result: list[dict] = []
        for job in body.get("jobs", []):
            name = job.get("name", "")
            # Only show flowsim jobs (name starts with "flowsim-")
            if not name.startswith("flowsim-"):
                continue

            state = job.get("job_state", ["UNKNOWN"])
            if isinstance(state, list):
                state = state[0] if state else "UNKNOWN"

            if status_filter and state.upper() != status_filter.upper():
                continue

            result.append({
                "job_id": str(job.get("job_id", "")),
                "name": name,
                "state": state,
                "partition": job.get("partition", ""),
                "nodes": job.get("nodes", ""),
            })
        return result

    def _parse_time_minutes(self) -> int:
        """Convert HH:MM:SS time_limit to total minutes."""
        parts = self.time_limit.split(":")
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 60 + m + (1 if s > 0 else 0)
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(parts[0])
