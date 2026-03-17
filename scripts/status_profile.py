#!/usr/bin/env python3
"""Query FlowSim profiling job status, logs, list, and cancel.

Usage examples
--------------

Check K8s job status::

    flowsim status --scheduler k8s --job flowsim-perf-qwen3-8b-bs1-il2048

Get K8s job logs::

    flowsim logs --scheduler k8s --job flowsim-perf-qwen3-8b-bs1-il2048

Follow K8s job logs::

    flowsim logs --scheduler k8s --job flowsim-perf-qwen3-8b-bs1-il2048 --follow

List all FlowSim jobs::

    flowsim list --scheduler k8s
    flowsim list --scheduler k8s --status Running

Cancel a job::

    flowsim cancel --scheduler k8s --job flowsim-perf-qwen3-8b-bs1-il2048
"""

from __future__ import annotations

import argparse
import os
import sys

from schedulers.config import cfg_get, load_k8s_config, load_slurm_config, resolve_jwt_token
from schedulers.k8s import K8sScheduler
from schedulers.local import LocalScheduler
from schedulers.slurm import SlurmScheduler


def _d(env_var: str, cfg: dict, key: str, fallback: str = "") -> str:
    return os.environ.get(env_var, "") or cfg_get(cfg, key, fallback)


def _add_scheduler_args(p: argparse.ArgumentParser) -> None:
    """Add common scheduler connection args to a parser."""
    k8s_cfg = load_k8s_config()
    slurm_cfg = load_slurm_config()

    p.add_argument(
        "--scheduler",
        choices=["local", "k8s", "slurm"],
        required=True,
    )

    # -- Local options --
    p.add_argument("--local-workdir", default="")

    # -- K8s options --
    p.add_argument(
        "--k8s-namespace",
        default=_d("FLOWSIM_K8S_NAMESPACE", k8s_cfg, "namespace", "default"),
    )
    p.add_argument(
        "--k8s-kubeconfig",
        default=_d("KUBECONFIG", k8s_cfg, "kubeconfig", ""),
    )
    p.add_argument(
        "--k8s-context",
        default=_d("FLOWSIM_K8S_CONTEXT", k8s_cfg, "context", ""),
    )

    # -- Slurm options --
    p.add_argument(
        "--slurm-rest-url",
        default=_d("FLOWSIM_SLURM_REST_URL", slurm_cfg, "rest_url", ""),
    )
    p.add_argument(
        "--slurm-jwt-token",
        default=_d("FLOWSIM_SLURM_JWT_TOKEN", slurm_cfg, "jwt_token", ""),
    )
    p.add_argument(
        "--slurm-api-version",
        default=_d("FLOWSIM_SLURM_API_VERSION", slurm_cfg, "api_version", "v0.0.40"),
    )
    p.add_argument(
        "--slurm-no-verify-ssl",
        action="store_true",
    )


def _resolve_slurm_jwt(args: argparse.Namespace) -> None:
    """Resolve Slurm JWT from config if not provided."""
    if args.scheduler == "slurm" and not args.slurm_jwt_token:
        slurm_cfg = load_slurm_config()
        token = resolve_jwt_token(slurm_cfg)
        if token:
            args.slurm_jwt_token = token


def _build_scheduler(args: argparse.Namespace):
    if args.scheduler == "local":
        return LocalScheduler(workdir=args.local_workdir)
    elif args.scheduler == "k8s":
        return K8sScheduler(
            namespace=args.k8s_namespace,
            kubeconfig=args.k8s_kubeconfig,
            context=args.k8s_context,
        )
    else:
        return SlurmScheduler(
            rest_url=args.slurm_rest_url,
            jwt_token=args.slurm_jwt_token,
            api_version=args.slurm_api_version,
            verify_ssl=not args.slurm_no_verify_ssl,
        )


def main_status(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Query FlowSim job status.")
    _add_scheduler_args(p)
    p.add_argument("--job", required=True, help="Job name or ID")
    args = p.parse_args(argv)

    _resolve_slurm_jwt(args)
    scheduler = _build_scheduler(args)
    try:
        info = scheduler.status(args.job)
        print(info["message"])
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def main_logs(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Retrieve FlowSim job logs.")
    _add_scheduler_args(p)
    p.add_argument("--job", required=True, help="Job name or ID")
    p.add_argument("--tail", type=int, default=100, help="Number of log lines (default: 100)")
    p.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    args = p.parse_args(argv)

    _resolve_slurm_jwt(args)
    scheduler = _build_scheduler(args)
    try:
        text = scheduler.logs(args.job, tail=args.tail, follow=args.follow)
        print(text)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def main_list(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="List FlowSim jobs.")
    _add_scheduler_args(p)
    p.add_argument("--status", default="", help="Filter by job state (e.g. Running, Succeeded, PENDING)")
    args = p.parse_args(argv)

    _resolve_slurm_jwt(args)
    scheduler = _build_scheduler(args)
    try:
        jobs = scheduler.list_jobs(status_filter=args.status)
        if not jobs:
            print("No jobs found.")
            return
        # Print table header
        headers = list(jobs[0].keys())
        widths = {h: max(len(h), max(len(str(j.get(h, ""))) for j in jobs)) for h in headers}
        header_line = "  ".join(h.upper().ljust(widths[h]) for h in headers)
        print(header_line)
        print("-" * len(header_line))
        for job in jobs:
            print("  ".join(str(job.get(h, "")).ljust(widths[h]) for h in headers))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def main_cancel(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Cancel a FlowSim job.")
    _add_scheduler_args(p)
    p.add_argument("--job", required=True, help="Job name or ID to cancel")
    args = p.parse_args(argv)

    _resolve_slurm_jwt(args)
    scheduler = _build_scheduler(args)
    try:
        msg = scheduler.cancel(args.job)
        print(msg)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
