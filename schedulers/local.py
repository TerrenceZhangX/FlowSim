"""Local scheduler — run profiling directly on this machine.

``render()`` returns the shell command string.
``submit()`` executes it as a subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys

from schedulers.base import BaseScheduler, ProfileJobSpec


class LocalScheduler(BaseScheduler):
    """Run profiling jobs locally via subprocess.

    Parameters
    ----------
    gpus : str
        ``CUDA_VISIBLE_DEVICES`` value (e.g., ``"0"`` or ``"0,1"``).
        Empty string means use all visible GPUs.
    workdir : str
        Working directory for the subprocess.
        Defaults to the FlowSim project root.
    """

    def __init__(
        self,
        *,
        gpus: str = "",
        workdir: str = "",
    ) -> None:
        self.gpus = gpus
        self.workdir = workdir or self._find_project_root()

    @staticmethod
    def _find_project_root() -> str:
        """Walk up from this file to find the FlowSim project root."""
        d = os.path.dirname(os.path.abspath(__file__))
        # schedulers/ is one level below project root
        return os.path.dirname(d)

    def render(self, spec: ProfileJobSpec) -> str:
        lines = []
        if self.gpus:
            lines.append(f"export CUDA_VISIBLE_DEVICES={self.gpus}")
        lines.append("export SGLANG_PROFILE_KERNELS=1")
        lines.append(f"cd {self.workdir}")
        lines.append(spec.build_shell_command())
        return "\n".join(lines)

    def submit(self, spec: ProfileJobSpec) -> str:
        """Run the profiling command locally as a subprocess."""
        cmd = spec.build_shell_command()

        env = os.environ.copy()
        env["SGLANG_PROFILE_KERNELS"] = "1"
        if self.gpus:
            env["CUDA_VISIBLE_DEVICES"] = self.gpus

        job_name = spec.default_job_name()
        print(f"[local] Running {job_name}...")
        print(f"[local] cmd: {cmd}")
        print(f"[local] workdir: {self.workdir}")
        if self.gpus:
            print(f"[local] CUDA_VISIBLE_DEVICES={self.gpus}")
        print()

        result = subprocess.run(
            cmd,
            shell=True,
            cwd=self.workdir,
            env=env,
        )

        if result.returncode != 0:
            return f"[local] {job_name} FAILED (exit code {result.returncode})"
        return f"[local] {job_name} completed successfully"
