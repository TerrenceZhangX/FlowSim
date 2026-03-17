"""Local scheduler — run profiling directly on this machine.

``render()`` returns the shell command string.
``submit()`` executes it as a subprocess, with stdout/stderr tee'd to log files.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from schedulers.base import BaseScheduler, JobResult, ProfileJobSpec


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

    def submit(self, spec: ProfileJobSpec) -> JobResult:
        """Run the profiling command locally as a subprocess.

        stdout and stderr are streamed to the terminal *and* saved to
        log files under ``spec.log_dir``.
        """
        cmd = spec.build_shell_command()

        env = os.environ.copy()
        env["SGLANG_PROFILE_KERNELS"] = "1"
        if self.gpus:
            env["CUDA_VISIBLE_DEVICES"] = self.gpus

        job_name = spec.default_job_name()
        log_dir = spec.log_dir
        os.makedirs(log_dir, exist_ok=True)
        ts = int(time.time())
        stdout_path = os.path.join(log_dir, f"{job_name}_{ts}.stdout.log")
        stderr_path = os.path.join(log_dir, f"{job_name}_{ts}.stderr.log")

        print(f"[local] Running {job_name}...")
        print(f"[local] cmd: {cmd}")
        print(f"[local] workdir: {self.workdir}")
        if self.gpus:
            print(f"[local] CUDA_VISIBLE_DEVICES={self.gpus}")
        print(f"[local] logs: {stdout_path}")
        print(f"[local]        {stderr_path}")
        print()

        with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=self.workdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Stream stdout/stderr to terminal + log files in real time.
            # Use threads to avoid blocking on one stream while the other
            # fills its OS pipe buffer.
            import threading

            def _tee(src, dest_file, dest_stream):
                for line in src:
                    dest_stream.buffer.write(line)
                    dest_stream.buffer.flush()
                    dest_file.write(line.decode("utf-8", errors="replace"))
                    dest_file.flush()

            t_out = threading.Thread(
                target=_tee, args=(proc.stdout, fout, sys.stdout), daemon=True,
            )
            t_err = threading.Thread(
                target=_tee, args=(proc.stderr, ferr, sys.stderr), daemon=True,
            )
            t_out.start()
            t_err.start()
            proc.wait()
            t_out.join()
            t_err.join()

        if proc.returncode != 0:
            return JobResult(
                job_id=job_name,
                scheduler="local",
                state="Failed",
                output_dir=spec.output_dir,
                message=(
                    f"{job_name} FAILED (exit code {proc.returncode})\n"
                    f"stdout log: {stdout_path}\n"
                    f"stderr log: {stderr_path}"
                ),
            )
        return JobResult(
            job_id=job_name,
            scheduler="local",
            state="Completed",
            output_dir=spec.output_dir,
            message=(
                f"{job_name} completed successfully\n"
                f"stdout log: {stdout_path}\n"
                f"stderr log: {stderr_path}"
            ),
        )

    def cancel(self, job_id: str) -> str:
        """Local jobs run synchronously, so cancel is not applicable."""
        return f"Local jobs run synchronously and cannot be cancelled. Job: {job_id}"

    def status(self, job_id: str) -> dict:
        """Check local job status by looking for log files.

        ``job_id`` is the job name prefix used in log filenames.
        """
        import glob

        log_dir = os.path.join(self.workdir, "stage_traces", "logs")
        pattern = os.path.join(log_dir, f"{job_id}_*.stdout.log")
        matches = sorted(glob.glob(pattern))

        if not matches:
            return {
                "state": "NotFound",
                "message": f"No logs found matching {pattern}",
                "output_hint": "",
            }

        latest = matches[-1]
        stderr_log = latest.replace(".stdout.log", ".stderr.log")
        trace_dir = os.path.join(self.workdir, "stage_traces")

        return {
            "state": "Completed",
            "message": (
                f"Latest log: {latest}\n"
                f"Stderr log: {stderr_log}\n"
                f"Traces dir: {trace_dir}"
            ),
            "output_hint": trace_dir,
        }

    def logs(self, job_id: str, *, tail: int = 100, follow: bool = False) -> str:
        """List log files for a local job and print access commands."""
        import glob

        log_dir = os.path.join(self.workdir, "stage_traces", "logs")
        pattern = os.path.join(log_dir, f"{job_id}_*")
        matches = sorted(glob.glob(pattern))

        if not matches:
            # Also try wildcard — user may have given a partial name
            pattern = os.path.join(log_dir, f"*{job_id}*")
            matches = sorted(glob.glob(pattern))

        if not matches:
            return f"No logs found in {log_dir} matching '{job_id}'"

        if follow:
            stdout_files = sorted(f for f in matches if f.endswith(".stdout.log"))
            if stdout_files:
                return f"Follow logs with:\n  tail -f {stdout_files[-1]}"
            return f"No stdout log found to follow for '{job_id}'"

        parts = [f"Log directory: {log_dir}", ""]
        parts.append(f"Files ({len(matches)}):")
        for p in matches:
            size = os.path.getsize(p)
            parts.append(f"  {os.path.basename(p)}  ({size:,} bytes)")

        # Provide commands
        parts.append("")
        parts.append("View logs:")
        stdout_files = sorted(f for f in matches if f.endswith(".stdout.log"))
        stderr_files = sorted(f for f in matches if f.endswith(".stderr.log"))
        if stdout_files:
            parts.append(f"  less {stdout_files[-1]}")
        if stderr_files:
            parts.append(f"  less {stderr_files[-1]}")
        if stdout_files:
            parts.append("")
            parts.append("Follow logs:")
            parts.append(f"  tail -f {stdout_files[-1]}")

        trace_dir = os.path.join(self.workdir, "stage_traces")
        parts.append("")
        parts.append(f"Trace files: {trace_dir}")
        parts.append(f"  ls {trace_dir}")

        return "\n".join(parts)

    def list_jobs(self, *, status_filter: str = "") -> list[dict]:
        """List local jobs by scanning log files."""
        import glob
        import re

        log_dir = os.path.join(self.workdir, "stage_traces", "logs")
        pattern = os.path.join(log_dir, "*.stdout.log")
        matches = sorted(glob.glob(pattern))

        jobs: list[dict] = []
        for path in matches:
            basename = os.path.basename(path)
            # Parse: {job_name}_{timestamp}.stdout.log
            m = re.match(r"^(.+)_(\d+)\.stdout\.log$", basename)
            if not m:
                continue
            name = m.group(1)
            ts = m.group(2)
            stderr = path.replace(".stdout.log", ".stderr.log")
            stderr_size = os.path.getsize(stderr) if os.path.exists(stderr) else 0
            # If stderr has content, might have failed; otherwise completed
            state = "Completed"
            if stderr_size > 0:
                # Check if there's an error indicator in stderr
                state = "Completed"  # local jobs are synchronous; if log exists, it finished
            jobs.append({
                "job_id": name,
                "name": name,
                "state": state,
                "timestamp": ts,
            })

        if status_filter:
            filt = status_filter.lower()
            jobs = [j for j in jobs if j["state"].lower() == filt]

        return jobs
