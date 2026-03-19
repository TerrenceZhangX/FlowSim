# FlowSim Schedulers

FlowSim supports three scheduler backends for submitting GPU profiling jobs:

| Backend | Use Case | Runs On | Dependencies |
|---------|----------|---------|--------------|
| **local** | Single-machine dev/testing | Host Docker container | Docker + NVIDIA GPU |
| **k8s** | Kubernetes cluster | K8s Job Pod | `kubernetes` Python package |
| **slurm** | HPC cluster | Slurm compute node | Slurm CLI (`sbatch`/`squeue`/`scancel`) |

## Quick Start

```bash
pip install -e .
flowsim --help
```

## Common Workflow

```bash
# Submit a job (same interface for all backends)
flowsim submit --scheduler <local|k8s|slurm> \
    --collect <perf|shapes|all> \
    --model-path <model> \
    --tp 1 --bs 1 --input-len 2048 --decode-tokens 2 --gpus 1

# Job lifecycle
flowsim list   --scheduler <backend>
flowsim status --scheduler <backend> --job <job_id>
flowsim logs   --scheduler <backend> --job <job_id>
flowsim cancel --scheduler <backend> --job <job_id>

# Preview without submitting
flowsim submit --scheduler <backend> ... --dry-run

# Multi-point sweep
flowsim submit --scheduler <backend> \
    --collect all --model-path workload/models/configs/Qwen3-235B-A22B \
    --sweep 1:2048:0 4:2048:0 8:2048:0 --gpus 1
```

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--collect` | `perf` / `shapes` / `all` | required |
| `--model-path` | HuggingFace model path | required |
| `--tp` | Tensor parallelism | `1` |
| `--dp` | Data parallelism | `1` |
| `--bs` | Batch size | `1` |
| `--input-len` | Input sequence length | `2048` |
| `--existing-ctx` | Existing KV cache length | `0` |
| `--decode-tokens` | Decode batches to profile | `2` |
| `--gpus` | GPU count | `1` |
| `--image` | Docker image | `flowsim-image:latest` |
| `--output-dir` | Output directory | `stage_traces/{scheduler}/{timestamp}/` |
| `--extra-server-opts` | Extra sglang server flags (quoted string) | `""` |
| `--sweep` | Multi-point sweep `BS:INPUT_LEN:CTX` (repeatable) | empty |
| `--job-name` | Custom job name | auto-generated |
| `--dry-run` | Print script only | `false` |

---

## 1. Local Scheduler

Runs profiling via `docker run` on the host machine.

```bash
flowsim submit --scheduler local \
    --collect all \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 1 --input-len 2048 --existing-ctx 0 --decode-tokens 2 --gpus 1 \
    --local-gpus 0 \
    --extra-server-opts "--load-format dummy"
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--local-gpus` | `CUDA_VISIBLE_DEVICES` (e.g. `0` or `0,1`) | all GPUs |
| `--local-workdir` | Host working directory | FlowSim project root |

---

## 2. Kubernetes Scheduler

Submits profiling jobs as Kubernetes Jobs. Supports PVC and hostPath storage.

### Setup

```bash
flowsim init k8s                           # install bundled template
flowsim init k8s --config my-cluster.yaml  # or use your own
# Edit ~/.flowsim/k8s.yaml
```

### Usage

```bash
flowsim submit --scheduler k8s \
    --collect all \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 1 --input-len 2048 --existing-ctx 0 --decode-tokens 2 --gpus 1 \
    --extra-server-opts "--load-format dummy"
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--k8s-namespace` | K8s namespace | `default` |
| `--k8s-kubeconfig` | kubeconfig path | `~/.kube/config` |
| `--k8s-context` | kubeconfig context | current context |
| `--k8s-pvc` | PVC name for traces | empty |
| `--k8s-host-output-dir` | hostPath (when no PVC) | empty |
| `--k8s-node-selector` | Node selector `KEY=VALUE` (repeatable) | empty |
| `--k8s-service-account` | ServiceAccount | empty |
| `--k8s-shm-size` | Shared memory size | `16Gi` |
| `--k8s-runtime-class` | RuntimeClass (e.g. `nvidia`) | empty |

---

## 3. Slurm Scheduler

Generates sbatch scripts and submits via `sbatch`/`squeue`/`scancel`.

### Setup

```bash
flowsim init slurm                         # install bundled template
flowsim init slurm --config my-slurm.yaml  # or use your own
# Edit ~/.flowsim/slurm.yaml
```

### Usage

```bash
flowsim submit --scheduler slurm \
    --collect all \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 1 --input-len 2048 --existing-ctx 0 --decode-tokens 2 --gpus 1 \
    --slurm-partition gpu \
    --extra-server-opts "--load-format dummy"
```

For remote clusters, use `--slurm-cli-prefix`:
```bash
flowsim submit --scheduler slurm ... \
    --slurm-cli-prefix "docker exec -i slurmctld"
# or: --slurm-cli-prefix "ssh login-node"
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--slurm-partition` | Slurm partition | empty |
| `--slurm-time` | Job time limit | `02:00:00` |
| `--slurm-account` | Billing account | empty |
| `--slurm-constraint` | Node constraint | empty |
| `--slurm-cli-prefix` | Shell prefix for remote CLI | empty |
| `--slurm-container-runtime` | `docker` / `enroot` / `none` | `none` |
| `--slurm-container-mounts` | Container mounts | empty |
| `--slurm-module` | `module load` commands (repeatable) | empty |
| `--slurm-extra-sbatch` | Extra `#SBATCH` directives (repeatable) | empty |

---

## Configuration

Config files live in `~/.flowsim/` and are installed via `flowsim init`.
Templates with comments are in `schedulers/templates/`.

```
~/.flowsim/
├── k8s.yaml
└── slurm.yaml
```

**Priority** (highest to lowest):
CLI flag → environment variable → config file → built-in default

### Environment Variables

| Variable | Overrides | Example |
|----------|-----------|--------|
| `KUBECONFIG` | `--k8s-kubeconfig` | `/home/user/.kube/config` |
| `FLOWSIM_K8S_NAMESPACE` | `--k8s-namespace` | `ml-team` |
| `FLOWSIM_K8S_CONTEXT` | `--k8s-context` | `kind-flowsim` |
| `FLOWSIM_K8S_CONFIG` | Config file path | `/etc/flowsim/k8s.yaml` |
| `FLOWSIM_SLURM_PARTITION` | `--slurm-partition` | `gpu-h100` |
| `FLOWSIM_SLURM_TIME` | `--slurm-time` | `04:00:00` |
| `FLOWSIM_SLURM_CONFIG` | Config file path | `/etc/flowsim/slurm.yaml` |

---

## Output Structure

```
stage_traces/{scheduler}/{YYYYMMDD_HHMMSS}/
├── bs1_input2048_ctx0/
│   ├── *.trace.json.gz
│   ├── parsed/*.csv
│   ├── merged/*_merged.trace.csv
│   ├── shape_traces/ + shape_parsed/
│   ├── analysis_extend.json
│   └── analysis_decode.json
├── logs/
│   ├── server_*.stdout.log
│   └── server_*.stderr.log
└── sweep_summary.json
```

---

## Development

### Test Clusters

```bash
# Kind (K8s) — GPU passthrough via CDI
bash tests/integration/infra/dev-setup.sh kind
bash tests/integration/infra/dev-teardown.sh kind

# Slurm — Docker Compose cluster
cd tests/integration/infra/
docker compose -f slurm-compose.yaml up -d
docker compose -f slurm-compose.yaml down -v
```

### Running Tests

```bash
# Unit tests
python -m pytest tests/unit/test_scheduler_cli.py -v

# Integration tests
python -m pytest tests/integration/test_scheduler_local.py::TestK8sScheduler -v -x
python -m pytest tests/integration/test_scheduler_local.py::TestSlurmScheduler -v -x
```

