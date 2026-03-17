"""FlowSim CLI — unified entry point.

Usage::

    flowsim init k8s --kubeconfig ~/.kube/config --namespace ml-team ...
    flowsim init slurm --rest-url https://slurm:6820 --partition gpu ...
    flowsim submit --scheduler k8s --collect perf --model-path ...
    flowsim submit ... --dry-run   # debug: preview manifest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_CONFIG_DIR = Path.home() / ".flowsim"


def _init_k8s_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("k8s", help="Configure Kubernetes scheduler")
    p.add_argument("--kubeconfig", required=True,
                   help="Path to kubeconfig file (REQUIRED)")
    p.add_argument("--context", default="",
                   help="Kubeconfig context (empty = current-context)")
    p.add_argument("--namespace", required=True,
                   help="Kubernetes namespace (REQUIRED)")
    p.add_argument("--pvc", default="",
                   help="PVC name for trace output")
    p.add_argument("--host-output-dir", default="",
                   help="hostPath alternative to PVC")
    p.add_argument("--service-account", default="",
                   help="Service account for the job pod")
    p.add_argument("--shm-size", default="16Gi",
                   help="Shared memory size (default: 16Gi)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing config file")


def _init_slurm_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("slurm", help="Configure Slurm scheduler")
    p.add_argument("--rest-url", required=True,
                   help="slurmrestd endpoint URL (REQUIRED)")
    p.add_argument("--partition", required=True,
                   help="Slurm partition (REQUIRED)")
    p.add_argument("--account", required=True,
                   help="Slurm account (REQUIRED)")
    p.add_argument("--jwt-token-cmd", default="",
                   help='Command to get JWT token, e.g. "scontrol token lifespan=3600"')
    p.add_argument("--jwt-token", default="",
                   help="Static JWT token (not recommended)")
    p.add_argument("--api-version", default="v0.0.40",
                   help="slurmrestd API version (default: v0.0.40)")
    p.add_argument("--time", default="02:00:00",
                   help="Job time limit (default: 02:00:00)")
    p.add_argument("--constraint", default="",
                   help="Node constraint")
    p.add_argument("--container-runtime", default="none",
                   choices=["docker", "enroot", "none"],
                   help="Container runtime (default: none)")
    p.add_argument("--container-mounts", default="",
                   help="Container mount spec")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing config file")


def _cmd_init(argv: list[str]) -> int:
    """Save scheduler config to ~/.flowsim/ from CLI args."""
    from schedulers.config import _save_yaml

    parser = argparse.ArgumentParser(
        prog="flowsim init",
        description=(
            "Configure a scheduler and save to ~/.flowsim/.\n\n"
            "Examples:\n"
            "  flowsim init k8s --kubeconfig ~/.kube/config --namespace ml-team\n"
            "  flowsim init slurm --rest-url https://slurm:6820 "
            "--partition gpu --account proj"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="scheduler")
    sub.required = True
    _init_k8s_parser(sub)
    _init_slurm_parser(sub)

    args = parser.parse_args(argv)

    if args.scheduler == "k8s":
        kube_path = Path(args.kubeconfig).expanduser()
        if not kube_path.is_file():
            print(f"Error: kubeconfig not found: {kube_path}", file=sys.stderr)
            return 1
        cfg = {
            "kubeconfig": str(kube_path),
            "context": args.context,
            "namespace": args.namespace,
            "pvc": args.pvc,
            "host_output_dir": args.host_output_dir,
            "service_account": args.service_account,
            "shm_size": args.shm_size,
        }
        dst = _CONFIG_DIR / "k8s.yaml"

    elif args.scheduler == "slurm":
        if not args.jwt_token_cmd and not args.jwt_token:
            print("Error: provide --jwt-token-cmd or --jwt-token", file=sys.stderr)
            return 1
        cfg = {
            "rest_url": args.rest_url,
            "jwt_token_cmd": args.jwt_token_cmd,
            "jwt_token": args.jwt_token,
            "partition": args.partition,
            "account": args.account,
            "api_version": args.api_version,
            "time": args.time,
            "constraint": args.constraint,
            "container_runtime": args.container_runtime,
            "container_mounts": args.container_mounts,
        }
        dst = _CONFIG_DIR / "slurm.yaml"
    else:
        parser.print_help()
        return 1

    if dst.exists() and not args.force:
        print(f"Error: {dst} already exists (use --force to overwrite)",
              file=sys.stderr)
        return 1

    _save_yaml(dst, cfg)
    print(f"Saved {dst}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="flowsim",
        description="FlowSim: workload simulation pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    sub.add_parser(
        "init",
        help="Configure a scheduler (k8s/slurm) and save to ~/.flowsim/",
        add_help=False,
    )
    sub.add_parser(
        "submit",
        help="Submit a profiling job to K8s or Slurm",
        add_help=False,
    )

    args, remaining = parser.parse_known_args(argv)

    if args.command == "init":
        return _cmd_init(remaining)

    if args.command == "submit":
        from scripts.submit_profile import main as submit_main

        submit_main(remaining)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
