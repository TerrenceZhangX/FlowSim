#!/usr/bin/env python3
"""Submit FlowSim profiling jobs locally, to Kubernetes, or to Slurm.

Usage examples
--------------

Run locally (no cluster needed):

    flowsim submit \\
        --scheduler local \\
        --collect perf \\
        --model-path Qwen/Qwen3-8B \\
        --tp 1 --local-gpus 0

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
import os
import sys

from schedulers.base import ProfileJobSpec
from schedulers.config import cfg_get, load_k8s_config, load_slurm_config, resolve_jwt_token
from schedulers.k8s import K8sScheduler
from schedulers.local import LocalScheduler
from schedulers.slurm import SlurmScheduler


def _d(env_var: str, cfg: dict, key: str, fallback: str = "") -> str:
    """Resolve default: env var > config file > fallback."""
    return os.environ.get(env_var, "") or cfg_get(cfg, key, fallback)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Load per-scheduler config files for defaults
    k8s_cfg = load_k8s_config()
    slurm_cfg = load_slurm_config()

    p = argparse.ArgumentParser(
        description="Submit FlowSim profiling jobs to local, K8s, or Slurm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # -- Scheduler choice --
    p.add_argument(
        "--scheduler",
        choices=["local", "k8s", "slurm"],
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
    infra.add_argument("--output-dir", default="")
    infra.add_argument("--job-name", default="")

    # -- Action --
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="[debug] Print rendered manifest without submitting",
    )

    # -- PD disaggregation --
    pd = p.add_argument_group("PD disaggregation")
    pd.add_argument(
        "--pd",
        action="store_true",
        help="Submit a prefill + decode job pair (PD disaggregation)",
    )
    pd.add_argument(
        "--disagg-transfer-backend",
        default="mooncake",
        choices=["mooncake", "nixl"],
        help="KV transfer backend (default: mooncake)",
    )
    pd.add_argument(
        "--disagg-bootstrap-port",
        type=int,
        default=8998,
        help="Bootstrap port for PD coordination (default: 8998)",
    )
    pd.add_argument(
        "--disagg-prefill-pp",
        type=int,
        default=1,
        help="Pipeline parallelism for prefill instance (default: 1)",
    )
    pd.add_argument(
        "--disagg-ib-device",
        default="",
        help="InfiniBand device for RDMA transfer",
    )

    # ---- Two-pass: peek at --scheduler, then add only relevant args ----
    # Use a minimal pre-parser to avoid required-arg errors during peek.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--scheduler", choices=["local", "k8s", "slurm"])
    pre, _ = _pre.parse_known_args(argv)

    if pre.scheduler == "local":
        loc = p.add_argument_group("local options")
        loc.add_argument(
            "--local-gpus",
            default="",
            help="CUDA_VISIBLE_DEVICES for local execution (e.g. '0' or '0,1')",
        )
        loc.add_argument(
            "--local-workdir",
            default="",
            help="Working directory for local execution (default: FlowSim project root)",
        )

    elif pre.scheduler == "k8s":
        k8s = p.add_argument_group("kubernetes options (config: ~/.flowsim/k8s.yaml)")
        k8s.add_argument(
            "--k8s-namespace",
            default=_d("FLOWSIM_K8S_NAMESPACE", k8s_cfg, "namespace", "default"),
            help="K8s namespace (env: FLOWSIM_K8S_NAMESPACE)",
        )
        k8s.add_argument(
            "--k8s-kubeconfig",
            default=_d("KUBECONFIG", k8s_cfg, "kubeconfig", ""),
            help="Path to kubeconfig file (env: KUBECONFIG)",
        )
        k8s.add_argument(
            "--k8s-context",
            default=_d("FLOWSIM_K8S_CONTEXT", k8s_cfg, "context", ""),
            help="kubeconfig context (env: FLOWSIM_K8S_CONTEXT)",
        )
        k8s.add_argument(
            "--k8s-pvc",
            default=cfg_get(k8s_cfg, "pvc", ""),
            help="PVC name for output volume (omit for emptyDir)",
        )
        k8s.add_argument(
            "--k8s-host-output-dir",
            default=cfg_get(k8s_cfg, "host_output_dir", ""),
            help="hostPath for output (used when --k8s-pvc is empty)",
        )
        k8s.add_argument(
            "--k8s-node-selector",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Node selector labels (repeatable)",
        )
        k8s.add_argument(
            "--k8s-service-account",
            default=cfg_get(k8s_cfg, "service_account", ""),
        )
        k8s.add_argument(
            "--k8s-shm-size",
            default=cfg_get(k8s_cfg, "shm_size", "16Gi"),
        )
        k8s.add_argument(
            "--k8s-runtime-class",
            default=cfg_get(k8s_cfg, "runtime_class_name", ""),
            help="RuntimeClass for pod (e.g. 'nvidia' for CDI mode)",
        )

    elif pre.scheduler == "slurm":
        slurm = p.add_argument_group("slurm options (config: ~/.flowsim/slurm.yaml)")
        slurm.add_argument(
            "--slurm-partition",
            default=_d("FLOWSIM_SLURM_PARTITION", slurm_cfg, "partition", ""),
            help="Slurm partition (env: FLOWSIM_SLURM_PARTITION)",
        )
        slurm.add_argument(
            "--slurm-time",
            default=_d("FLOWSIM_SLURM_TIME", slurm_cfg, "time", "02:00:00"),
            help="Wall time limit (env: FLOWSIM_SLURM_TIME)",
        )
        slurm.add_argument(
            "--slurm-rest-url",
            default=_d("FLOWSIM_SLURM_REST_URL", slurm_cfg, "rest_url", ""),
            help="slurmrestd base URL (env: FLOWSIM_SLURM_REST_URL)",
        )
        slurm.add_argument(
            "--slurm-jwt-token",
            default=_d("FLOWSIM_SLURM_JWT_TOKEN", slurm_cfg, "jwt_token", ""),
            help="JWT token for slurmrestd (env: FLOWSIM_SLURM_JWT_TOKEN)",
        )
        slurm.add_argument(
            "--slurm-api-version",
            default=_d("FLOWSIM_SLURM_API_VERSION", slurm_cfg, "api_version", "v0.0.40"),
            help="slurmrestd API version (env: FLOWSIM_SLURM_API_VERSION)",
        )
        slurm.add_argument(
            "--slurm-no-verify-ssl",
            action="store_true",
            help="Skip TLS certificate verification for slurmrestd",
        )
        slurm.add_argument(
            "--slurm-account",
            default=cfg_get(slurm_cfg, "account", ""),
        )
        slurm.add_argument(
            "--slurm-constraint",
            default=cfg_get(slurm_cfg, "constraint", ""),
        )
        slurm.add_argument(
            "--slurm-container-runtime",
            choices=["docker", "enroot", "none"],
            default=cfg_get(slurm_cfg, "container_runtime", "none"),
        )
        slurm.add_argument(
            "--slurm-container-mounts",
            default=cfg_get(slurm_cfg, "container_mounts", ""),
        )
        # Modules from config (list) + CLI (append)
        cfg_modules = slurm_cfg.get("modules") if isinstance(slurm_cfg.get("modules"), list) else []
        slurm.add_argument(
            "--slurm-module",
            action="append",
            default=[str(m) for m in cfg_modules],
            help="Modules to load (repeatable, merged with config)",
        )
        slurm.add_argument(
            "--slurm-extra-sbatch",
            action="append",
            default=[],
            metavar="DIRECTIVE",
            help="Extra #SBATCH directives (repeatable, without prefix)",
        )
        slurm.add_argument(
            "--slurm-submit-via",
            choices=["rest", "cli"],
            default=cfg_get(slurm_cfg, "submit_via", "rest"),
            help="Submission mode: rest (slurmrestd) or cli (sbatch subprocess)",
        )
        slurm.add_argument(
            "--slurm-cli-prefix",
            default=cfg_get(slurm_cfg, "cli_prefix", ""),
            help='Shell prefix for CLI mode (e.g. "docker exec -i slurmctld")',
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
        job_name=args.job_name,
        extra_server_opts=args.extra_server_opts,
        disagg_transfer_backend=args.disagg_transfer_backend,
        disagg_bootstrap_port=args.disagg_bootstrap_port,
        disagg_prefill_pp=args.disagg_prefill_pp,
        disagg_ib_device=args.disagg_ib_device,
    )


def _build_scheduler(args: argparse.Namespace):
    if args.scheduler == "local":
        return LocalScheduler(
            gpus=args.local_gpus,
            workdir=args.local_workdir,
        )
    elif args.scheduler == "k8s":
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
            runtime_class_name=args.k8s_runtime_class,
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
            submit_via=args.slurm_submit_via,
            cli_prefix=args.slurm_cli_prefix,
        )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Smart defaults for output_dir based on scheduler.
    # Layout: stage_traces/{scheduler}/{timestamp}/
    import time as _time
    _ts = _time.strftime("%Y%m%d_%H%M%S")
    if not args.output_dir:
        if args.scheduler == "local":
            args.output_dir = f"/flowsim/stage_traces/local/{_ts}"
        elif args.scheduler == "slurm":
            args.output_dir = f"/flowsim/stage_traces/slurm/{_ts}"
        else:
            args.output_dir = f"/flowsim/stage_traces/k8s/{_ts}"

    # Resolve Slurm JWT token from jwt_token_cmd in config if needed
    if args.scheduler == "slurm" and not args.slurm_jwt_token:
        slurm_cfg = load_slurm_config()
        token = resolve_jwt_token(slurm_cfg)
        if token:
            args.slurm_jwt_token = token

    # Validate required connection params before submit
    if not args.dry_run and args.scheduler not in ("local",):
        _validate_connection(args)

    # For local scheduler, convert absolute host model_path to relative
    # so it resolves correctly inside the container (workdir=/flowsim).
    if args.scheduler == "local" and os.path.isabs(args.model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if args.model_path.startswith(project_root):
            args.model_path = os.path.relpath(args.model_path, project_root)

    spec = _build_spec(args)
    scheduler = _build_scheduler(args)

    is_pd = args.pd

    if args.dry_run:
        if is_pd:
            print(scheduler.render_pd_pair(spec))
        else:
            print(scheduler.dry_run(spec))
    else:
        if is_pd:
            results = scheduler.submit_pd_pair(spec)
            for r in results:
                print(r.message)
            # Use the first result for follow-up hints
            result = results[0]
        else:
            result = scheduler.submit(spec)
            print(result.message)

        # Tell user where to find results
        print()
        print(f"Traces: {result.output_dir}")
        print(f"Logs:   {result.output_dir}/logs/")
        job_id = result.job_id
        sched = args.scheduler

        if sched == "k8s":
            if args.k8s_pvc:
                print(f"  (persisted on PVC '{args.k8s_pvc}')")
            else:
                print(f"  (persisted at hostPath '{args.k8s_host_output_dir}' on the node)")
            print(f"\nTo check status:  flowsim status --scheduler k8s --job {job_id}")
            print(f"To view logs:     flowsim logs   --scheduler k8s --job {job_id}")
            print(f"To follow logs:   flowsim logs   --scheduler k8s --job {job_id} --follow")
            print(f"To cancel:        flowsim cancel --scheduler k8s --job {job_id}")
        elif sched == "slurm":
            print(f"  (on cluster shared filesystem)")
            print(f"\nTo check status:  flowsim status --scheduler slurm --job {job_id}")
            print(f"To view logs:     flowsim logs   --scheduler slurm --job {job_id}")
            print(f"To cancel:        flowsim cancel --scheduler slurm --job {job_id}")
        else:
            print(f"\nTo view logs:     flowsim logs   --scheduler local --job {job_id}")
        print(f"To list all jobs: flowsim list   --scheduler {sched}")


_INIT_HINT = "Run 'flowsim init' to create config files."


def _validate_connection(args: argparse.Namespace) -> None:
    """Fail fast if required cluster connection params are missing."""
    if args.scheduler == "k8s":
        if not args.k8s_namespace:
            sys.exit(
                "Error: K8s namespace not set.\n"
                "Set it in ~/.flowsim/k8s.yaml, FLOWSIM_K8S_NAMESPACE env var,\n"
                f"or --k8s-namespace flag. {_INIT_HINT}"
            )
        # Traces + logs must survive pod termination
        if not args.k8s_pvc and not args.k8s_host_output_dir:
            sys.exit(
                "Error: no persistent storage configured for K8s job output.\n"
                "Traces and logs are written to output_dir inside the pod —\n"
                "without a volume mount they are lost when the pod exits.\n\n"
                "Set one of:\n"
                "  --k8s-pvc <pvc-name>           (PersistentVolumeClaim)\n"
                "  --k8s-host-output-dir <path>   (hostPath on the node)\n\n"
                "Or configure in ~/.flowsim/k8s.yaml:\n"
                "  pvc: my-traces-pvc\n"
                "  # or\n"
                "  host_output_dir: /data/flowsim-traces"
            )
        # kubeconfig is optional (in-cluster auto-discovery), but warn
        if not args.k8s_kubeconfig and not args.k8s_context:
            print(
                "Note: no kubeconfig or context specified. "
                "Will try ~/.kube/config and in-cluster auto-discovery.",
                file=sys.stderr,
            )
    elif args.scheduler == "slurm":
        if args.slurm_submit_via == "cli":
            # CLI mode only needs partition
            if not args.slurm_partition:
                sys.exit(
                    "Error: missing required Slurm config:\n"
                    "  - partition (--slurm-partition)\n\n"
                    f"Set it in ~/.flowsim/slurm.yaml or via CLI flag.\n"
                    + _INIT_HINT
                )
            return
        missing = []
        if not args.slurm_rest_url:
            missing.append("rest_url (--slurm-rest-url)")
        if not args.slurm_jwt_token:
            missing.append("jwt_token/jwt_token_cmd (--slurm-jwt-token)")
        if not args.slurm_partition:
            missing.append("partition (--slurm-partition)")
        if missing:
            sys.exit(
                "Error: missing required Slurm config:\n"
                + "\n".join(f"  - {m}" for m in missing)
                + f"\n\nSet them in ~/.flowsim/slurm.yaml or via CLI flags.\n"
                + _INIT_HINT
            )


if __name__ == "__main__":
    main()
