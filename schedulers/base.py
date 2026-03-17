"""Abstract base class for FlowSim job schedulers."""

from __future__ import annotations

import abc
import shlex
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JobResult:
    """Structured return value from ``submit()``."""

    job_id: str
    scheduler: str          # "local", "k8s", "slurm"
    state: str              # "Submitted", "Completed", "Failed"
    output_dir: str = ""
    message: str = ""


@dataclass
class ProfileJobSpec:
    """All parameters needed to run a stage-profiling job.

    The scheduler backends render this into a K8s Job YAML or Slurm
    sbatch script.
    """

    # -- Profiling workload --
    collect: str  # "perf", "shapes", or "all"
    model_path: str
    tp: int = 1
    dp: int = 1
    bs: int = 1
    input_len: int = 2048
    existing_ctx: int = 0
    decode_tokens: int = 32
    warmup_n: int = 5
    disable_chunked_prefill: bool = False
    max_prefill_tokens: int = 131072

    # -- Infrastructure --
    image: str = "flowsim-image:latest"
    gpus: int = 1  # total GPU count (must be >= tp * dp)
    host: str = "0.0.0.0"
    port: int = 30001
    output_dir: str = "/flowsim/stage_traces"
    job_name: str = ""

    # -- PD disaggregation --
    disagg_mode: str = ""  # "prefill", "decode", or "" (unified)
    disagg_transfer_backend: str = "mooncake"  # "mooncake" or "nixl"
    disagg_bootstrap_port: int = 8998
    disagg_prefill_pp: int = 1
    disagg_ib_device: str = ""

    # -- Extra server opts (appended verbatim) --
    extra_server_opts: str = ""

    def build_server_opts(self) -> str:
        """Build the ``--server-opts`` string for run_stage_profile.py."""
        parts = [
            f"--model-path {self.model_path}",
            f"--tp {self.tp}",
            f"--host {self.host}",
            f"--port {self.port}",
        ]
        if self.dp > 1:
            parts.append(f"--dp {self.dp}")
        if self.disagg_mode:
            parts.append(f"--disaggregation-mode {self.disagg_mode}")
            parts.append(f"--disaggregation-transfer-backend {self.disagg_transfer_backend}")
            parts.append(f"--disaggregation-bootstrap-port {self.disagg_bootstrap_port}")
            if self.disagg_prefill_pp > 1:
                parts.append(f"--disaggregation-prefill-pp {self.disagg_prefill_pp}")
            if self.disagg_ib_device:
                parts.append(f"--disaggregation-ib-device {self.disagg_ib_device}")
        if self.extra_server_opts:
            parts.append(self.extra_server_opts)
        return " ".join(parts)

    @property
    def log_dir(self) -> str:
        """Server logs go under ``{output_dir}/logs/``."""
        return self.output_dir + "/logs"

    def build_profile_command(self) -> list[str]:
        """Build the full ``python scripts/run_stage_profile.py ...`` command."""
        cmd = [
            "python3",
            "scripts/run_stage_profile.py",
            "--collect",
            self.collect,
            "--launch-server",
            "--server-opts",
            self.build_server_opts(),
            "--bs",
            str(self.bs),
            "--input-len",
            str(self.input_len),
            "--existing-ctx",
            str(self.existing_ctx),
            "--decode-tokens",
            str(self.decode_tokens),
            "--warmup-n",
            str(self.warmup_n),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--output-dir",
            self.output_dir,
            "--log-dir",
            self.log_dir,
        ]
        if self.disable_chunked_prefill:
            cmd.append("--disable-chunked-prefill")
            cmd.extend(["--max-prefill-tokens", str(self.max_prefill_tokens)])
        return cmd

    def build_shell_command(self) -> str:
        """Build a single shell command string (properly quoted)."""
        cmd = self.build_profile_command()
        # Quote the --server-opts value since it contains spaces
        quoted = []
        i = 0
        while i < len(cmd):
            if cmd[i] == "--server-opts" and i + 1 < len(cmd):
                quoted.append(cmd[i])
                quoted.append(shlex.quote(cmd[i + 1]))
                i += 2
            else:
                quoted.append(cmd[i])
                i += 1
        return " ".join(quoted)

    def default_job_name(self) -> str:
        """Generate a default job name from workload params."""
        if self.job_name:
            return self.job_name
        model_short = self.model_path.split("/")[-1].lower().replace(".", "-")
        name = f"flowsim-{self.collect}-{model_short}-bs{self.bs}-il{self.input_len}"
        if self.disagg_mode:
            name += f"-{self.disagg_mode}"
        return name

    def as_prefill(self) -> "ProfileJobSpec":
        """Return a copy configured as the prefill instance."""
        from dataclasses import replace
        return replace(self, disagg_mode="prefill")

    def as_decode(self) -> "ProfileJobSpec":
        """Return a copy configured as the decode instance."""
        from dataclasses import replace
        return replace(self, disagg_mode="decode")


class BaseScheduler(abc.ABC):
    """Abstract scheduler backend."""

    @abc.abstractmethod
    def render(self, spec: ProfileJobSpec) -> str:
        """Render the job manifest / script as a string."""

    @abc.abstractmethod
    def submit(self, spec: ProfileJobSpec) -> JobResult:
        """Submit the job and return a structured :class:`JobResult`."""

    def cancel(self, job_id: str) -> str:
        """Cancel a running or pending job. Returns a status message."""
        raise NotImplementedError(f"{type(self).__name__} does not support cancel")

    def status(self, job_id: str) -> dict:
        """Query job status. Returns dict with at least 'state' key.

        Subclasses should return::

            {
                "state": "Pending" | "Running" | "Succeeded" | "Failed" | ...,
                "message": "human-readable detail",
                "output_hint": "where to find trace files",
            }
        """
        raise NotImplementedError(f"{type(self).__name__} does not support status queries")

    def logs(self, job_id: str, *, tail: int = 100, follow: bool = False) -> str:
        """Retrieve recent log output for a job.

        Parameters
        ----------
        job_id : str
            Job name (K8s) or job ID (Slurm) or log prefix (local).
        tail : int
            Number of lines from the end to return.
        follow : bool
            If True, stream logs in real time (blocking).
        """
        raise NotImplementedError(f"{type(self).__name__} does not support log retrieval")

    def list_jobs(self, *, status_filter: str = "") -> list[dict]:
        """List jobs managed by this scheduler.

        Parameters
        ----------
        status_filter : str
            If non-empty, only return jobs matching this state
            (e.g., ``"Running"``, ``"Succeeded"``, ``"PENDING"``).

        Returns
        -------
        list[dict]
            Each dict has at least ``{"job_id": ..., "state": ..., "name": ...}``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support list")

    def dry_run(self, spec: ProfileJobSpec) -> str:
        """Render and return the manifest without submitting."""
        return self.render(spec)

    def render_pd_pair(self, spec: ProfileJobSpec) -> str:
        """Render both prefill and decode manifests for PD disaggregation."""
        prefill = self.render(spec.as_prefill())
        decode = self.render(spec.as_decode())
        return f"# === PREFILL INSTANCE ===\n{prefill}\n# === DECODE INSTANCE ===\n{decode}"

    def submit_pd_pair(self, spec: ProfileJobSpec) -> list[JobResult]:
        """Submit both prefill and decode jobs."""
        r1 = self.submit(spec.as_prefill())
        r2 = self.submit(spec.as_decode())
        return [r1, r2]
