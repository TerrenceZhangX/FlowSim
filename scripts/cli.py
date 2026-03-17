"""FlowSim CLI — unified entry point.

Usage::

    flowsim submit --scheduler k8s --collect perf --model-path ... --dry-run
    flowsim submit --scheduler slurm --collect perf --model-path ... --submit
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="flowsim",
        description="FlowSim: workload simulation pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # ---- submit ----
    sub.add_parser(
        "submit",
        help="Submit a profiling job to K8s or Slurm",
        add_help=False,  # submit_profile has its own --help
    )

    # Parse only the subcommand, pass the rest through
    args, remaining = parser.parse_known_args(argv)

    if args.command == "submit":
        from scripts.submit_profile import main as submit_main

        submit_main(remaining)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
