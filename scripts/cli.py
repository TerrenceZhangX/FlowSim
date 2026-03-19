"""FlowSim CLI — unified entry point.

Usage::

    flowsim init k8s            # create ~/.flowsim/k8s.yaml template
    flowsim init slurm          # create ~/.flowsim/slurm.yaml template
    flowsim submit --scheduler k8s --collect perf --model-path ...
    flowsim submit ... --dry-run   # debug: preview manifest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_CONFIG_DIR = Path.home() / ".flowsim"

# ---- Annotated config templates (written by `flowsim init`) ----

_K8S_TEMPLATE = """\
# FlowSim Kubernetes scheduler config
# Edit this file, then run: flowsim submit --scheduler k8s ...

# Path to kubeconfig file (required)
kubeconfig: ~/.kube/config

# Kubeconfig context (empty = current-context)
context: ""

# Kubernetes namespace (required)
namespace: default

# Persistent storage for trace output (set one):
#   pvc: my-traces-pvc
#   host_output_dir: /data/flowsim-traces
pvc: ""
host_output_dir: ""

# Service account for the job pod (empty = default)
service_account: ""

# Shared memory size (for /dev/shm in the pod)
shm_size: "16Gi"

# RuntimeClass (e.g. "nvidia" for CDI GPU passthrough)
runtime_class_name: ""
"""

_SLURM_TEMPLATE = """\
# FlowSim Slurm scheduler config
# Edit this file, then run: flowsim submit --scheduler slurm ...

# Slurm partition (required)
partition: gpu

# Billing account (empty = default)
account: ""

# Job time limit
time: "02:00:00"

# Node constraint (e.g. "h100")
constraint: ""

# CLI prefix for remote sbatch/squeue/scancel
# Examples:
#   "docker exec -i slurmctld"   (via Docker container)
#   "ssh login-node"             (via SSH)
cli_prefix: ""

# Container runtime: docker | enroot | none
container_runtime: none

# Container mount spec (for enroot/docker)
container_mounts: ""
"""


def _cmd_init(argv: list[str]) -> int:
    """Copy an annotated config template to ~/.flowsim/."""
    parser = argparse.ArgumentParser(
        prog="flowsim init",
        description=(
            "Generate a scheduler config template under ~/.flowsim/.\n\n"
            "Examples:\n"
            "  flowsim init k8s          # creates ~/.flowsim/k8s.yaml\n"
            "  flowsim init slurm        # creates ~/.flowsim/slurm.yaml\n"
            "  flowsim init slurm --force # overwrite existing"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scheduler", choices=["k8s", "slurm"],
        help="Scheduler type",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing config file",
    )
    args = parser.parse_args(argv)

    templates = {"k8s": _K8S_TEMPLATE, "slurm": _SLURM_TEMPLATE}
    dst = _CONFIG_DIR / f"{args.scheduler}.yaml"

    if dst.exists() and not args.force:
        print(f"Error: {dst} already exists (use --force to overwrite)",
              file=sys.stderr)
        return 1

    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    dst.write_text(templates[args.scheduler])
    print(f"Created {dst}")
    print("Edit the file, then run: flowsim submit --scheduler "
          f"{args.scheduler} ...")
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
    sub.add_parser(
        "status",
        help="Query job status (local/k8s/slurm)",
        add_help=False,
    )
    sub.add_parser(
        "logs",
        help="Retrieve job logs (local/k8s/slurm)",
        add_help=False,
    )
    sub.add_parser(
        "list",
        help="List FlowSim jobs (local/k8s/slurm)",
        add_help=False,
    )
    sub.add_parser(
        "cancel",
        help="Cancel a running job (k8s/slurm)",
        add_help=False,
    )

    args, remaining = parser.parse_known_args(argv)

    if args.command == "init":
        return _cmd_init(remaining)

    if args.command == "submit":
        from scripts.submit_profile import main as submit_main

        submit_main(remaining)
        return 0

    if args.command == "status":
        from scripts.status_profile import main_status

        main_status(remaining)
        return 0

    if args.command == "logs":
        from scripts.status_profile import main_logs

        main_logs(remaining)
        return 0

    if args.command == "list":
        from scripts.status_profile import main_list

        main_list(remaining)
        return 0

    if args.command == "cancel":
        from scripts.status_profile import main_cancel

        main_cancel(remaining)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
