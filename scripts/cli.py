"""FlowSim CLI — unified entry point.

Usage::

    flowsim init                           # set up ~/.flowsim/ config files
    flowsim submit --scheduler k8s ...     # submit a profiling job
    flowsim submit ... --dry-run           # preview manifest without submitting
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "schedulers" / "templates"
_CONFIG_DIR = Path.home() / ".flowsim"


def _cmd_init(argv: list[str]) -> int:
    """Copy config templates to ~/.flowsim/."""
    parser = argparse.ArgumentParser(
        prog="flowsim init",
        description="Initialize ~/.flowsim/ with scheduler config templates.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config files",
    )
    args = parser.parse_args(argv)

    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    templates = list(_TEMPLATE_DIR.glob("*.yaml"))
    if not templates:
        print(f"Error: no templates found in {_TEMPLATE_DIR}", file=sys.stderr)
        return 1

    for src in templates:
        dst = _CONFIG_DIR / src.name
        if dst.exists() and not args.force:
            print(f"  skip  {dst}  (already exists, use --force to overwrite)")
        else:
            shutil.copy2(src, dst)
            print(f"  wrote {dst}")

    print(f"\nEdit the files in {_CONFIG_DIR}/ to configure your cluster,")
    print("then run: flowsim submit --scheduler <k8s|slurm> ...")
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
        help="Initialize ~/.flowsim/ with config templates",
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
