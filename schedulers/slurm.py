"""Slurm sbatch scheduler for FlowSim profiling."""

from __future__ import annotations

import subprocess
import tempfile
import textwrap

from schedulers.base import BaseScheduler, ProfileJobSpec


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
    ) -> None:
        self.partition = partition
        self.time_limit = time_limit
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
        script = self.render(spec)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False
        ) as f:
            f.write(script)
            f.flush()
            result = subprocess.run(
                ["sbatch", f.name],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"sbatch failed:\n{result.stderr.strip()}"
                )
            return result.stdout.strip()
