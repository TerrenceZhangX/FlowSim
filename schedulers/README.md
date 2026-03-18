# FlowSim Schedulers

FlowSim supports three scheduler backends for submitting GPU profiling jobs:

| Backend | Use Case | Runs On | Dependencies |
|---------|----------|---------|--------------|
| **local** | Single-machine dev/testing | Host Docker container | Docker + NVIDIA GPU |
| **k8s** | Kubernetes cluster | K8s Job Pod | `kubernetes` Python package |
| **slurm** | HPC cluster | Slurm compute node | Slurm CLI (`sbatch`/`squeue`/`scancel`) |

## Quick Start

```bash
# Install (from FlowSim project root)
cd FlowSim
pip install -e .  # or ensure PYTHONPATH includes the project root

# Show help
flowsim --help
flowsim submit --help
```

## Common Workflow

All schedulers share the same CLI interface:

```bash
# 1. Submit a job
flowsim submit --scheduler <local|k8s|slurm> --collect <perf|shapes|all> \
    --model-path <model> [options...]

# 2. List jobs
flowsim list --scheduler <local|k8s|slurm>

# 3. Check job status
flowsim status --scheduler <local|k8s|slurm> --job <job_id>

# 4. View logs
flowsim logs --scheduler <local|k8s|slurm> --job <job_id>

# 5. Cancel a job
flowsim cancel --scheduler <local|k8s|slurm> --job <job_id>

# 6. Dry-run (print script/manifest without submitting)
flowsim submit --scheduler <local|k8s|slurm> ... --dry-run
```

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--collect` | Collection mode: `perf` / `shapes` / `all` | required |
| `--model-path` | HuggingFace model path | required |
| `--tp` | Tensor parallelism | `1` |
| `--dp` | Data parallelism | `1` |
| `--bs` | Batch size | `1` |
| `--input-len` | Input sequence length | `2048` |
| `--existing-ctx` | Existing KV cache length | `0` |
| `--decode-tokens` | Decode token count | `32` |
| `--warmup-n` | Warmup iterations | `5` |
| `--image` | Docker image | `flowsim-image:latest` |
| `--gpus` | GPU count | `1` |
| `--output-dir` | Output directory (auto-generated if omitted) | `stage_traces/{scheduler}/{timestamp}/` |
| `--dry-run` | Print script only, do not submit | `false` |

---

## 1. Local Scheduler

Runs profiling directly on the host via `docker run`. The simplest option, suitable for single-machine development and testing.

### Usage

```bash
# Simplest usage — run on GPU 0
flowsim submit --scheduler local \
    --collect all \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 1 --input-len 2048 \
    --gpus 1 --local-gpus 0 \
    --extra-server-opts "--load-format dummy"

# Multi-GPU
flowsim submit --scheduler local \
    --collect perf \
    --model-path Qwen/Qwen3-8B \
    --tp 2 --gpus 2 --local-gpus 0,1
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--local-gpus` | `CUDA_VISIBLE_DEVICES` (e.g. `0` or `0,1`) | empty (all GPUs) |
| `--local-workdir` | Host working directory | FlowSim project root |

### How It Works

1. `render()` generates a `docker run --gpus` command
2. `submit()` runs the container on the host, waits for completion
3. Traces are written to `stage_traces/local/{YYYYMMDD_HHMMSS}/`
4. `status()` / `logs()` / `list_jobs()` scan log files

---

## 2. Kubernetes Scheduler

Submits profiling jobs as Kubernetes Jobs to a cluster. Supports both PVC and hostPath storage.

### First-Time Setup

```bash
flowsim init k8s \
    --kubeconfig ~/.kube/config \
    --namespace default \
    --host-output-dir /host-stage-traces \
    --runtime-class-name nvidia \
    --force
```

Config is saved to `~/.flowsim/k8s.yaml` and automatically loaded on subsequent submissions.

### Usage

```bash
# Submit to K8s cluster
flowsim submit --scheduler k8s \
    --collect all \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 1 --input-len 2048 --gpus 1 \
    --extra-server-opts "--load-format dummy"

# Override config file values
flowsim submit --scheduler k8s \
    --collect perf \
    --model-path Qwen/Qwen3-8B \
    --k8s-namespace ml-team \
    --k8s-pvc my-traces-pvc \
    --gpus 4 --tp 4

# Dry-run to preview the generated YAML
flowsim submit --scheduler k8s ... --dry-run
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--k8s-namespace` | K8s namespace | `default` |
| `--k8s-kubeconfig` | kubeconfig path | `~/.kube/config` |
| `--k8s-context` | kubeconfig context | current context |
| `--k8s-pvc` | PVC name (persistent storage) | empty |
| `--k8s-host-output-dir` | hostPath mount (used when PVC is empty) | empty |
| `--k8s-node-selector` | Node selector labels (repeatable), format `KEY=VALUE` | empty |
| `--k8s-service-account` | ServiceAccount | empty |
| `--k8s-shm-size` | Shared memory size | `16Gi` |
| `--k8s-runtime-class` | RuntimeClass (e.g. `nvidia` for CDI mode) | empty |

### How It Works

1. `render()` generates a Kubernetes Job YAML/JSON manifest
2. `submit()` creates the Job via the `kubernetes` Python client
3. Traces are persisted via PVC or hostPath
4. `status()` / `cancel()` / `list_jobs()` operate via the K8s API

### Kind Local Test Cluster

```bash
# Start a Kind cluster (GPU passthrough + CDI mode)
bash tests/integration/infra/dev-setup.sh kind

# Run K8s integration tests
python -m pytest tests/integration/test_scheduler_local.py::TestK8sScheduler -v -x

# Teardown
bash tests/integration/infra/dev-teardown.sh kind
```

---

## 3. Slurm Scheduler

Generates sbatch scripts and submits them to a Slurm cluster via `sbatch`/`squeue`/`scancel`.

### First-Time Setup

```bash
flowsim init slurm \
    --partition gpu \
    --account my-project \
    --container-runtime none \
    --force
```

### Usage

```bash
# Submit via sbatch
flowsim submit --scheduler slurm \
    --collect all \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 1 --input-len 2048 --gpus 1 \
    --slurm-partition gpu \
    --extra-server-opts "--load-format dummy"

# CLI prefix (e.g. via docker exec or ssh)
flowsim submit --scheduler slurm \
    --slurm-cli-prefix "docker exec -i slurmctld" \
    --slurm-partition normal \
    --collect perf --model-path Qwen/Qwen3-8B --gpus 1

# Dry-run to preview the generated sbatch script
flowsim submit --scheduler slurm ... --dry-run

# Check status
flowsim status --scheduler slurm --job 12345 \
    --slurm-cli-prefix "docker exec -i slurmctld"

# Cancel a job
flowsim cancel --scheduler slurm --job 12345
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--slurm-partition` | Slurm partition | empty |
| `--slurm-time` | Job time limit | `02:00:00` |
| `--slurm-account` | Billing account | empty |
| `--slurm-constraint` | Node constraint | empty |
| `--slurm-cli-prefix` | Shell prefix for CLI commands (e.g. `"docker exec -i slurmctld"`) | empty |
| `--slurm-container-runtime` | Container runtime: `docker` / `enroot` / `none` | `none` |
| `--slurm-container-mounts` | Container mounts | empty |
| `--slurm-module` | `module load` commands (repeatable) | empty |
| `--slurm-extra-sbatch` | Extra `#SBATCH` directives (repeatable) | empty |

### container_runtime Options

| Value | Description |
|-------|-------------|
| `none` | Run directly on compute node (Python/sglang must be installed) |
| `docker` | Run via `docker run` on the allocated node |
| `enroot` | Run via `srun --container-image` (NVIDIA enroot) |

### How It Works

1. `render()` generates a complete sbatch script (`#SBATCH` directives + profiling command)
2. `submit()` pipes the script to `sbatch --parsable`
3. `status()` queries via `scontrol show job`
4. `cancel()` runs `scancel`
5. `list_jobs()` runs `squeue`

If Slurm commands are not on the local PATH, use `--slurm-cli-prefix` to specify a prefix, e.g.:
- `"docker exec -i slurmctld"` — via Docker container
- `"ssh login-node"` — via SSH

### Docker Compose Local Test Cluster

```bash
# Start Slurm cluster (slurmctld + 1 compute node + 1 GPU)
cd tests/integration/infra/
docker compose -f slurm-compose.yaml up -d

# Check cluster status
docker exec slurmctld sinfo

# Run Slurm integration tests
python -m pytest tests/integration/test_scheduler_local.py::TestSlurmScheduler -v -x

# Teardown
docker compose -f slurm-compose.yaml down -v
```

---

## Configuration

Config files are stored in `~/.flowsim/` and generated via `flowsim init`:

```
~/.flowsim/
├── k8s.yaml      # K8s scheduler config
└── slurm.yaml    # Slurm scheduler config
```

Parameter priority (highest to lowest):
1. CLI flag (`--slurm-partition gpu`)
2. Environment variable (see table below)
3. Config file (`~/.flowsim/slurm.yaml`)
4. Built-in default

### Supported Environment Variables

| Variable | Overrides | Example |
|----------|-----------|--------|
| `KUBECONFIG` | `--k8s-kubeconfig` | `/home/user/.kube/config` |
| `FLOWSIM_K8S_NAMESPACE` | `--k8s-namespace` | `ml-team` |
| `FLOWSIM_K8S_CONTEXT` | `--k8s-context` | `kind-flowsim` |
| `FLOWSIM_K8S_CONFIG` | Config file path | `/etc/flowsim/k8s.yaml` |
| `FLOWSIM_SLURM_PARTITION` | `--slurm-partition` | `gpu-h100` |
| `FLOWSIM_SLURM_TIME` | `--slurm-time` | `04:00:00` |
| `FLOWSIM_SLURM_CONFIG` | Config file path | `/etc/flowsim/slurm.yaml` |

### Example k8s.yaml

```yaml
kubeconfig: /home/user/.kube/config
namespace: default
host_output_dir: /host-stage-traces
runtime_class_name: nvidia
shm_size: 16Gi
```

### Example slurm.yaml

```yaml
partition: gpu
account: my-project
time: "02:00:00"
container_runtime: none
cli_prefix: ""
```

---

## Output Directory Structure

All schedulers produce a unified trace output structure:

```
stage_traces/{scheduler}/{YYYYMMDD_HHMMSS}/
├── bs1_input2048_ctx0/
│   ├── *.trace.json.gz           # Raw traces
│   ├── parsed/*.csv              # Parsed CSVs
│   ├── merged/*_merged.trace.csv # Merged trace CSV
│   ├── shape_traces/             # Shape traces (collect=shapes/all)
│   ├── shape_parsed/*.csv        # Shape parsed CSVs
│   ├── analysis_extend.json      # Extend stage analysis
│   └── analysis_decode.json      # Decode stage analysis
├── logs/
│   ├── server_*.stdout.log
│   └── server_*.stderr.log
└── sweep_summary.json
```

