"""Local scheduler — run profiling via Docker on the local machine.

``render()`` returns the ``docker run`` command string.
``submit()`` executes it as a subprocess, with stdout/stderr tee'd to log files.
The profiling runs inside the FlowSim Docker image with GPU access.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from schedulers.base import BaseScheduler, JobResult, ProfileJobSpec


def _shell_quote(s: str) -> str:
    """Quote a string for safe embedding in a bash -c '...' invocation."""
    import shlex
    return shlex.quote(s)


class LocalScheduler(BaseScheduler):
    """Run profiling jobs locally inside a Docker container.

    Parameters
    ----------
    gpus : str
        GPU device IDs for Docker ``--gpus`` (e.g., ``"0"`` or ``"0,1"``).
        Empty string means all GPUs.
    workdir : str
        Host directory to use as the FlowSim project root for log scanning.
        Defaults to the FlowSim project root on the host.
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

    def _docker_gpu_flag(self) -> str:
        """Build the ``--gpus`` flag for ``docker run``."""
        if not self.gpus:
            return "--gpus all"
        return f"--gpus '\"device={self.gpus}\"'"

    def _build_docker_cmd(self, spec: ProfileJobSpec) -> str:
        """Build the full ``docker run`` command."""
        job_name = spec.default_job_name()[:63]
        # Container always works with /flowsim/stage_traces internally.
        container_output = "/flowsim/stage_traces"
        container_log_dir = container_output + "/logs"
        host_output = os.path.abspath(spec.output_dir)
        host_log_dir = host_output + "/logs"

        # Build the inner command, then replace host paths with container paths.
        inner_cmd = spec.build_shell_command()
        inner_cmd = inner_cmd.replace(host_log_dir, container_log_dir)
        inner_cmd = inner_cmd.replace(host_output, container_output)

        parts = [
            "docker run --rm",
            f"--name {job_name}",
            self._docker_gpu_flag(),
            "--ipc=host --shm-size=16g",
            "--network=host",
            f"-e SGLANG_PROFILE_KERNELS=1",
            f"-v {host_output}:{container_output}",
            f"-w /flowsim",
            spec.image,
            f"bash -c {_shell_quote(inner_cmd)}",
        ]
        return " \\\n  ".join(parts)

    def render(self, spec: ProfileJobSpec) -> str:
        return self._build_docker_cmd(spec)

    def submit(self, spec: ProfileJobSpec) -> JobResult:
        """Launch a Docker container for profiling.

        stdout and stderr are streamed to the terminal *and* saved to
        log files under ``spec.output_dir/logs/`` on the host.
        """
        # Ensure host output dir exists before mounting
        host_output = os.path.abspath(spec.output_dir)
        log_dir = os.path.join(host_output, "logs")
        os.makedirs(log_dir, exist_ok=True)

        docker_cmd = self._build_docker_cmd(spec)
        job_name = spec.default_job_name()
        ts = time.strftime("%Y%m%d_%H%M%S")

        # Remove stale container with the same name (e.g. from a killed run)
        subprocess.run(
            ["docker", "rm", "-f", job_name[:63]],
            capture_output=True, timeout=10,
        )
        stdout_path = os.path.join(log_dir, f"{job_name}_{ts}.stdout.log")
        stderr_path = os.path.join(log_dir, f"{job_name}_{ts}.stderr.log")

        print(f"[local] Running {job_name} in Docker...")
        print(f"[local] image: {spec.image}")
        print(f"[local] gpus: {self.gpus or 'all'}")
        print(f"[local] host output: {host_output}")
        print(f"[local] logs: {stdout_path}")
        print(f"[local]        {stderr_path}")
        print(f"[local] cmd:\n  {docker_cmd}")
        print()

        with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
            proc = subprocess.Popen(
                docker_cmd,
                shell=True,
                cwd=self.workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
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
                output_dir=host_output,
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
            output_dir=host_output,
            message=(
                f"{job_name} completed successfully\n"
                f"stdout log: {stdout_path}\n"
                f"stderr log: {stderr_path}"
            ),
        )

    def cancel(self, job_id: str) -> str:
        """Stop the Docker container for a local job."""
        proc = subprocess.run(
            ["docker", "stop", job_id],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            return f"Stopped container {job_id}"
        return f"Could not stop container {job_id}: {proc.stderr.strip()}"

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
            # Parse: {job_name}_{YYYYMMDD_HHMMSS}.stdout.log
            # Also support old epoch format {job_name}_{digits}.stdout.log
            m = re.match(r"^(.+)_(\d{8}_\d{6}|\d{10,})\.stdout\.log$", basename)
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
