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
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "schedulers" / "templates"


def _cmd_init(argv: list[str]) -> int:
    """Install a scheduler config to ~/.flowsim/.

    Without --config: copies the bundled template from schedulers/templates/.
    With --config: copies the specified file.
    """
    parser = argparse.ArgumentParser(
        prog="flowsim init",
        description=(
            "Install scheduler config under ~/.flowsim/.\n\n"
            "Examples:\n"
            "  flowsim init k8s                    # install bundled template\n"
            "  flowsim init k8s --config my.yaml   # install your own file\n"
            "  flowsim init slurm --force           # overwrite existing"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scheduler", choices=["k8s", "slurm"],
        help="Scheduler type",
    )
    parser.add_argument(
        "--config", "-c", default="",
        help="Path to a config YAML to install (default: bundled template)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing config file",
    )
    args = parser.parse_args(argv)

    dst = _CONFIG_DIR / f"{args.scheduler}.yaml"

    if dst.exists() and not args.force:
        print(f"Error: {dst} already exists (use --force to overwrite)",
              file=sys.stderr)
        return 1

    if args.config:
        src = Path(args.config).expanduser()
    else:
        src = _TEMPLATES_DIR / f"{args.scheduler}.yaml"

    if not src.is_file():
        print(f"Error: config file not found: {src}", file=sys.stderr)
        return 1

    import shutil
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Installed {src} → {dst}")
    print(f"Edit {dst}, then run: flowsim submit --scheduler "
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
