#!/usr/bin/env python3
"""Submit FlowSim profiling jobs to Kubernetes or Slurm.

Usage examples
--------------

Dry-run (print Kubernetes Job YAML to stdout):

    python scripts/submit_profile.py \\
        --scheduler k8s \\
        --collect perf \\
        --model-path Qwen/Qwen3-235B-A22B-FP8 \\
        --tp 4 --gpus 4 \\
        --bs 1 --input-len 2048 --decode-tokens 32 \\
        --image flowsim-image:latest \\
        --k8s-namespace default \\
        --k8s-pvc flowsim-traces \\
        --dry-run

Dry-run (print Slurm sbatch script to stdout):

    python scripts/submit_profile.py \\
        --scheduler slurm \\
        --collect perf \\
        --model-path Qwen/Qwen3-235B-A22B-FP8 \\
        --tp 4 --gpus 4 \\
        --slurm-partition gpu-a100 \\
        --slurm-time 02:00:00 \\
        --dry-run

Submit directly to cluster:

    python scripts/submit_profile.py \\
        --scheduler k8s \\
        ... \\
        --submit
"""

from __future__ import annotations

import argparse
import sys

from schedulers.base import ProfileJobSpec
from schedulers.k8s import K8sScheduler
from schedulers.slurm import SlurmScheduler


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit FlowSim profiling jobs to K8s or Slurm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # -- Scheduler choice --
    p.add_argument(
        "--scheduler",
        choices=["k8s", "slurm"],
        required=True,
        help="Scheduler backend.",
    )

    # -- Profiling workload (mirrors run_stage_profile.py) --
    wl = p.add_argument_group("workload")
    wl.add_argument(
        "--collect",
        choices=["perf", "shapes", "all"],
        required=True,
    )
    wl.add_argument("--model-path", required=True, help="HF model path")
    wl.add_argument("--tp", type=int, default=1)
    wl.add_argument("--dp", type=int, default=1)
    wl.add_argument("--bs", type=int, default=1, help="Batch size")
    wl.add_argument("--input-len", type=int, default=2048)
    wl.add_argument("--existing-ctx", type=int, default=0)
    wl.add_argument("--decode-tokens", type=int, default=32)
    wl.add_argument("--warmup-n", type=int, default=5)
    wl.add_argument(
        "--disable-chunked-prefill", action="store_true",
    )
    wl.add_argument("--max-prefill-tokens", type=int, default=131072)
    wl.add_argument(
        "--extra-server-opts",
        default="",
        help="Extra server options appended verbatim",
    )

    # -- Infrastructure --
    infra = p.add_argument_group("infrastructure")
    infra.add_argument("--image", default="flowsim-image:latest")
    infra.add_argument(
        "--gpus", type=int, default=1, help="Total GPU count",
    )
    infra.add_argument("--host", default="0.0.0.0")
    infra.add_argument("--port", type=int, default=30001)
    infra.add_argument("--output-dir", default="/flowsim/stage_traces")
    infra.add_argument(
        "--log-dir", default="/flowsim/tests/test-artifacts",
    )
    infra.add_argument("--job-name", default="")

    # -- Kubernetes-specific --
    k8s = p.add_argument_group("kubernetes options")
    k8s.add_argument("--k8s-namespace", default="default")
    k8s.add_argument(
        "--k8s-kubeconfig",
        default="",
        help="Path to kubeconfig file (empty = default lookup)",
    )
    k8s.add_argument(
        "--k8s-context",
        default="",
        help="kubeconfig context to use",
    )
    k8s.add_argument(
        "--k8s-pvc",
        default="",
        help="PVC name for output volume (omit for emptyDir)",
    )
    k8s.add_argument(
        "--k8s-host-output-dir",
        default="",
        help="hostPath for output (used when --k8s-pvc is empty)",
    )
    k8s.add_argument(
        "--k8s-node-selector",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Node selector labels (repeatable)",
    )
    k8s.add_argument("--k8s-service-account", default="")
    k8s.add_argument("--k8s-shm-size", default="16Gi")

    # -- Slurm-specific --
    slurm = p.add_argument_group("slurm options")
    slurm.add_argument("--slurm-partition", default="gpu")
    slurm.add_argument("--slurm-time", default="02:00:00")
    slurm.add_argument(
        "--slurm-rest-url",
        default="",
        help="slurmrestd base URL (e.g. https://slurm.example.com:6820). "
             "Required for --submit.",
    )
    slurm.add_argument(
        "--slurm-jwt-token",
        default="",
        help="JWT token for slurmrestd auth. "
             "Generate via: scontrol token lifespan=3600",
    )
    slurm.add_argument(
        "--slurm-api-version",
        default="v0.0.40",
        help="slurmrestd OpenAPI version (default: v0.0.40)",
    )
    slurm.add_argument(
        "--slurm-no-verify-ssl",
        action="store_true",
        help="Skip TLS certificate verification for slurmrestd",
    )
    slurm.add_argument("--slurm-account", default="")
    slurm.add_argument("--slurm-constraint", default="")
    slurm.add_argument(
        "--slurm-container-runtime",
        choices=["docker", "enroot", "none"],
        default="none",
    )
    slurm.add_argument("--slurm-container-mounts", default="")
    slurm.add_argument(
        "--slurm-module",
        action="append",
        default=[],
        help="Modules to load (repeatable)",
    )
    slurm.add_argument(
        "--slurm-extra-sbatch",
        action="append",
        default=[],
        metavar="DIRECTIVE",
        help="Extra #SBATCH directives (repeatable, without prefix)",
    )

    # -- Action --
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the rendered manifest; do not submit",
    )

    return p.parse_args(argv)


def _build_spec(args: argparse.Namespace) -> ProfileJobSpec:
    return ProfileJobSpec(
        collect=args.collect,
        model_path=args.model_path,
        tp=args.tp,
        dp=args.dp,
        bs=args.bs,
        input_len=args.input_len,
        existing_ctx=args.existing_ctx,
        decode_tokens=args.decode_tokens,
        warmup_n=args.warmup_n,
        disable_chunked_prefill=args.disable_chunked_prefill,
        max_prefill_tokens=args.max_prefill_tokens,
        image=args.image,
        gpus=args.gpus,
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        job_name=args.job_name,
        extra_server_opts=args.extra_server_opts,
    )


def _build_scheduler(args: argparse.Namespace):
    if args.scheduler == "k8s":
        node_sel = {}
        for item in args.k8s_node_selector:
            k, _, v = item.partition("=")
            if not v:
                sys.exit(f"Bad --k8s-node-selector format: {item!r} (use KEY=VALUE)")
            node_sel[k] = v
        return K8sScheduler(
            namespace=args.k8s_namespace,
            kubeconfig=args.k8s_kubeconfig,
            context=args.k8s_context,
            pvc_name=args.k8s_pvc,
            host_output_dir=args.k8s_host_output_dir,
            node_selector=node_sel,
            service_account=args.k8s_service_account,
            shm_size=args.k8s_shm_size,
        )
    else:
        return SlurmScheduler(
            partition=args.slurm_partition,
            time_limit=args.slurm_time,
            rest_url=args.slurm_rest_url,
            jwt_token=args.slurm_jwt_token,
            api_version=args.slurm_api_version,
            verify_ssl=not args.slurm_no_verify_ssl,
            account=args.slurm_account,
            constraint=args.slurm_constraint,
            container_runtime=args.slurm_container_runtime,
            container_mounts=args.slurm_container_mounts,
            modules=args.slurm_module,
            extra_sbatch=args.slurm_extra_sbatch,
        )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    spec = _build_spec(args)
    scheduler = _build_scheduler(args)

    if args.dry_run:
        print(scheduler.dry_run(spec))
    else:
        result = scheduler.submit(spec)
        print(result)


if __name__ == "__main__":
    main()
