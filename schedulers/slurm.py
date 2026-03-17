"""Slurm sbatch scheduler for FlowSim profiling.

``render()`` / ``dry_run()`` produce a standalone bash script (zero deps).
``submit()`` posts the script to a slurmrestd endpoint via stdlib
``urllib.request`` — no extra packages needed.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request

from schedulers.base import BaseScheduler, ProfileJobSpec


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

    def submit(self, spec: ProfileJobSpec) -> str:
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

        job_id = body.get("job_id", "unknown")
        return f"Submitted batch job {job_id}"

    def _parse_time_minutes(self) -> int:
        """Convert HH:MM:SS time_limit to total minutes."""
        parts = self.time_limit.split(":")
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 60 + m + (1 if s > 0 else 0)
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(parts[0])
