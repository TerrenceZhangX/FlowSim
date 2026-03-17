#!/usr/bin/env python
"""Submit parsed kernels from a trace to LLMCompass backend.

This script parses a trace file and submits each kernel to the LLMCompass backend
for simulation.

Example:
  python scripts/run_simulate.py \\
    --trace-file /flowsim/server_profile/your-trace.trace.json.gz \\
    --api-url http://127.0.0.1:8000 \\
    --artifact-dir /flowsim/artifacts
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

from backend.interface import submit_task, get_result, wait_for_health
from simulator.base_parser import BaseKernelInfoParser
from simulator.utils import parse_kernel_entry


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit parsed kernels to LLMCompass backend"
    )
    p.add_argument(
        "--trace-file",
        required=True,
        help="Path to profiler trace (.trace.json.gz) file",
    )
    p.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Base URL of LLMCompass backend (default: http://127.0.0.1:8000)",
    )
    p.add_argument(
        "--artifact-dir",
        default="/flowsim/artifacts",
        help="Directory to write request/response artifacts",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of kernel entries to submit (0 = no limit)",
    )
    p.add_argument(
        "--system-key",
        default="A100_4_fp16",
        help="System key for backend simulation (default: A100_4_fp16)",
    )
    return p.parse_args(argv)


def write_summary(artifact_dir: Path, submitted: dict, results: dict) -> None:
    """Write summary files (JSON and CSV) with all task results."""

    # Collect summary data
    summary_data = {
        "total_tasks": len(submitted),
        "successful_tasks": 0,
        "running_tasks": 0,
        "failed_tasks": 0,
        "tasks": [],
    }

    for task_id, task_info in submitted.items():
        payload = task_info["payload"]
        result = results.get(task_id, {})

        # Get top-level status (queued/running/done)
        status = result.get("status", "unknown")

        task_entry = {
            "task_id": task_id,
            "kernel_name": payload.get("kernel_name", ""),
            "op": payload.get("op", ""),
            "input_dim": str(payload.get("input_dim", "")),
            "dtype": str(payload.get("dtype", "")),
            "system_key": payload.get("system_key", ""),
            "status": status,
        }

        # Check result body for simulation status and data
        result_body = result.get("result", {})
        if isinstance(result_body, dict):
            result_status = result_body.get("status")  # "success" or "failed"

            # Extract simulated_time (in seconds) if successful
            simulated_time = result_body.get("simulated_time")
            if simulated_time is not None:
                task_entry["latency_s"] = float(simulated_time)

            # Extract failure reason if failed
            failure_reason = result_body.get("failure_reason", {})
            if isinstance(failure_reason, dict):
                error_msg = failure_reason.get("error", "")
                error_code = failure_reason.get("error_code", "")
                if error_msg or error_code:
                    task_entry["error"] = (
                        f"[{error_code}] {error_msg}"
                        if error_code
                        else error_msg
                    )

        # Categorize task status
        if status == "done":
            # Check if simulation actually succeeded
            if result_body.get("status") == "success":
                summary_data["successful_tasks"] += 1
            else:
                # Task completed but simulation failed
                summary_data["failed_tasks"] += 1
        elif status in ("timeout", "pending", "running", "queued", "unknown"):
            # Task hasn't completed yet
            summary_data["running_tasks"] += 1
        else:
            # Real failure (error in submission)
            summary_data["failed_tasks"] += 1
            if "error" not in task_entry:
                error_msg = result.get("error", "Unknown error")
                task_entry["error"] = error_msg

        summary_data["tasks"].append(task_entry)

    # Write JSON summary
    summary_json = artifact_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Write CSV summary
    if summary_data["tasks"]:
        summary_csv = artifact_dir / "summary.csv"
        fieldnames = [
            "task_id",
            "kernel_name",
            "op",
            "input_dim",
            "dtype",
            "system_key",
            "status",
            "latency_s",
            "error",
        ]

        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(summary_data["tasks"])

    # Print summary to console
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total tasks:      {summary_data['total_tasks']}")
    print(f"Successful:       {summary_data['successful_tasks']}")
    print(f"Running:          {summary_data['running_tasks']}")
    print(f"Failed:           {summary_data['failed_tasks']}")
    print("=" * 60 + "\n")


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"[ERROR] Trace file not found: {trace_path}")
        return 1

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for individual task files
    tasks_dir = artifact_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    api_url = args.api_url.rstrip("/")

    print(f"[INFO] Using trace: {trace_path}")
    print(f"[INFO] Backend URL: {api_url}")
    print(f"[INFO] Artifacts directory: {artifact_dir}")
    print(f"[INFO] Tasks directory: {tasks_dir}")

    print("[INFO] Parsing trace...")
    parser = BaseKernelInfoParser(
        str(trace_path), enable_comm_calibration=False
    )
    entries = getattr(parser, "individual_info", None) or []

    if not isinstance(entries, list) or not entries:
        print("[ERROR] No kernel entries found in trace")
        return 1

    print(f"[INFO] Found {len(entries)} kernel entries")

    session = requests.Session()

    try:
        print("[INFO] Waiting for backend /health...")
        healthy = wait_for_health(api_url, timeout=30.0)
        if not healthy:
            print("[ERROR] Backend did not become healthy within timeout")
            return 1

        submitted = {}
        max_entries = (
            args.limit if args.limit and args.limit > 0 else len(entries)
        )

        for idx, entry in enumerate(entries, start=1):
            if idx > max_entries:
                break

            kernel_name, input_dim, dtype, op = parse_kernel_entry(entry)
            payload = {
                "kernel_name": kernel_name,
                "op": op,
                "input_dim": input_dim,
                "dtype": dtype,
                "system_key": args.system_key,
            }

            out_file = tasks_dir / f"task_{idx}_{kernel_name[:10]}.json"

            resp = submit_task(api_url, payload, timeout=10, session=session)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({"request": payload, "response": resp}, f, indent=2)

            if "error" in resp:
                print(
                    f"[WARN] submit_task error for entry {idx}: {resp['error']}"
                )
                continue

            if resp.get("status_code") != 200:
                print(
                    f"[WARN] Non-200 status for entry {idx}: {resp.get('status_code')}"
                )
                continue

            body = resp.get("body") or {}
            task_id = body.get("task_id")
            if not task_id:
                print(f"[WARN] Missing task_id for entry {idx}")
                continue

            submitted[task_id] = {"out_file": out_file, "payload": payload}

            if idx % 10 == 0:
                print(f"[INFO] Submitted {idx} tasks so far...")

            time.sleep(0.02)

        if not submitted:
            print("[ERROR] No tasks were successfully submitted")
            return 1

        print(
            f"[INFO] Submitted {len(submitted)} tasks. Polling for completion..."
        )

        pending = set(submitted.keys())
        results = {}
        poll_deadline = time.time() + max(120.0, len(pending) * 5)

        while time.time() < poll_deadline and pending:
            for task_id in list(pending):
                res = get_result(api_url, task_id, timeout=10, session=session)

                with open(
                    tasks_dir / f"task_{task_id}_poll.json",
                    "w",
                    encoding="utf-8",
                ) as pf:
                    json.dump(res, pf, indent=2)

                if "error" in res:
                    results[task_id] = res
                    pending.discard(task_id)
                    # Update summary immediately after each task completes/fails
                    write_summary(artifact_dir, submitted, results)
                    continue

                if res.get("status") == "done":
                    result = res.get("result")
                    if not isinstance(result, dict):
                        print(
                            f"[WARN] Task {task_id} done but result not a dict: {res}"
                        )
                    results[task_id] = res
                    pending.discard(task_id)
                    # Update summary immediately after each task completes
                    write_summary(artifact_dir, submitted, results)

            if pending:
                time.sleep(0.5)

        # Mark incomplete tasks
        for task_id in pending:
            results[task_id] = {
                "status": "timeout",
                "error": "Task did not complete within timeout",
            }

        # Final summary write
        write_summary(artifact_dir, submitted, results)

        if pending:
            print(f"[ERROR] Some tasks did not complete: {sorted(pending)}")
            return 1

        print("[INFO] All tasks completed successfully")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
