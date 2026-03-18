# FlowSim Schedulers

FlowSim 支持三种调度器后端，用于提交 GPU profiling 任务：

| 后端 | 适用场景 | 运行位置 | 依赖 |
|------|----------|----------|------|
| **local** | 单机开发/测试 | 宿主机 Docker 容器 | Docker + NVIDIA GPU |
| **k8s** | Kubernetes 集群 | K8s Job Pod | `kubernetes` Python 包 |
| **slurm** | HPC 集群 | Slurm 计算节点 | Slurm CLI 或 slurmrestd |

## 快速上手

```bash
# 安装（从 FlowSim 项目根目录）
cd FlowSim
pip install -e .  # 或确保 PYTHONPATH 包含项目根目录

# 查看帮助
flowsim --help
flowsim submit --help
```

## 通用工作流

所有调度器共享相同的 CLI 接口：

```bash
# 1. 提交任务
flowsim submit --scheduler <local|k8s|slurm> --collect <perf|shapes|all> \
    --model-path <model> [选项...]

# 2. 查看任务列表
flowsim list --scheduler <local|k8s|slurm>

# 3. 查看任务状态
flowsim status --scheduler <local|k8s|slurm> --job <job_id>

# 4. 查看日志
flowsim logs --scheduler <local|k8s|slurm> --job <job_id>

# 5. 取消任务
flowsim cancel --scheduler <local|k8s|slurm> --job <job_id>

# 6. Dry-run（仅打印脚本/manifest，不提交）
flowsim submit --scheduler <local|k8s|slurm> ... --dry-run
```

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--collect` | 收集模式：`perf`(性能) / `shapes`(形状) / `all`(两者) | 必填 |
| `--model-path` | HuggingFace 模型路径 | 必填 |
| `--tp` | Tensor parallelism | `1` |
| `--dp` | Data parallelism | `1` |
| `--bs` | Batch size | `1` |
| `--input-len` | 输入序列长度 | `2048` |
| `--existing-ctx` | 已有 KV cache 长度 | `0` |
| `--decode-tokens` | Decode 生成 token 数 | `32` |
| `--warmup-n` | Warmup 迭代数 | `5` |
| `--image` | Docker 镜像 | `flowsim-image:latest` |
| `--gpus` | GPU 数量 | `1` |
| `--output-dir` | 输出目录（自动生成如不指定） | `stage_traces/{scheduler}/{timestamp}/` |
| `--dry-run` | 仅打印脚本，不提交 | `false` |

---

## 1. Local 调度器

直接在宿主机上通过 `docker run` 启动容器运行 profiling。最简单的方式，适合单机开发和测试。

### 使用

```bash
# 最简单的用法 — 使用 GPU 0 运行
flowsim submit --scheduler local \
    --collect all \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 1 --input-len 2048 \
    --gpus 1 --local-gpus 0 \
    --extra-server-opts "--load-format dummy"

# 多 GPU
flowsim submit --scheduler local \
    --collect perf \
    --model-path Qwen/Qwen3-8B \
    --tp 2 --gpus 2 --local-gpus 0,1
```

### 专有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--local-gpus` | `CUDA_VISIBLE_DEVICES`（如 `0` 或 `0,1`） | 空（使用所有 GPU） |
| `--local-workdir` | 主机工作目录 | FlowSim 项目根目录 |

### 工作原理

1. `render()` 生成一条 `docker run --gpus` 命令
2. `submit()` 在宿主机执行该容器，同步等待完成
3. Traces 写入宿主机 `stage_traces/local/{YYYYMMDD_HHMMSS}/`
4. `status()` / `logs()` / `list_jobs()` 扫描日志文件

---

## 2. Kubernetes 调度器

将 profiling 任务作为 Kubernetes Job 提交到集群。支持 PVC 和 hostPath 两种存储方式。

### 首次配置

```bash
flowsim init k8s \
    --kubeconfig ~/.kube/config \
    --namespace default \
    --host-output-dir /host-stage-traces \
    --runtime-class-name nvidia \
    --force
```

配置保存到 `~/.flowsim/k8s.yaml`，后续提交时自动读取。

### 使用

```bash
# 提交到 K8s 集群
flowsim submit --scheduler k8s \
    --collect all \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 1 --input-len 2048 --gpus 1 \
    --extra-server-opts "--load-format dummy"

# 覆盖配置文件中的值
flowsim submit --scheduler k8s \
    --collect perf \
    --model-path Qwen/Qwen3-8B \
    --k8s-namespace ml-team \
    --k8s-pvc my-traces-pvc \
    --gpus 4 --tp 4

# Dry-run 查看生成的 YAML
flowsim submit --scheduler k8s ... --dry-run
```

### 专有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--k8s-namespace` | K8s 命名空间 | `default` |
| `--k8s-kubeconfig` | kubeconfig 路径 | `~/.kube/config` |
| `--k8s-context` | kubeconfig context | 当前 context |
| `--k8s-pvc` | PVC 名称（持久存储） | 空 |
| `--k8s-host-output-dir` | hostPath 挂载路径（PVC 为空时使用） | 空 |
| `--k8s-node-selector` | 节点选择标签（可重复），格式 `KEY=VALUE` | 空 |
| `--k8s-service-account` | ServiceAccount | 空 |
| `--k8s-shm-size` | 共享内存大小 | `16Gi` |
| `--k8s-runtime-class` | RuntimeClass（如 `nvidia`，用于 CDI 模式） | 空 |

### 工作原理

1. `render()` 生成 Kubernetes Job YAML/JSON manifest
2. `submit()` 通过 `kubernetes` Python 客户端创建 Job
3. Traces 通过 PVC 或 hostPath 持久化
4. `status()` / `cancel()` / `list_jobs()` 通过 K8s API 操作

### Kind 本地测试集群

```bash
# 启动 Kind 集群（GPU passthrough + CDI 模式）
bash dockerfiles/dev-setup.sh kind

# 运行 K8s 集成测试
python -m pytest tests/integration/test_scheduler_local.py::TestK8sScheduler -v -x

# 清理
bash dockerfiles/dev-teardown.sh kind
```

---

## 3. Slurm 调度器

生成 sbatch 脚本并提交到 Slurm 集群。支持两种提交模式：

- **CLI 模式**（推荐，默认）：通过 `sbatch`/`squeue`/`scancel` 命令
- **REST 模式**（已弃用）：通过 slurmrestd REST API + JWT 认证

### 首次配置

```bash
# CLI 模式（推荐，无需 slurmrestd）
flowsim init slurm \
    --partition gpu \
    --account my-project \
    --container-runtime none \
    --force

# REST 模式（已弃用，需要 slurmrestd）
flowsim init slurm \
    --rest-url https://slurm.example.com:6820 \
    --partition gpu \
    --account my-project \
    --jwt-token-cmd "scontrol token lifespan=3600" \
    --force
```

### 使用

```bash
# CLI 模式 — 直接调用 sbatch（最常用）
flowsim submit --scheduler slurm \
    --collect all \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 1 --input-len 2048 --gpus 1 \
    --slurm-partition gpu \
    --slurm-submit-via cli \
    --extra-server-opts "--load-format dummy"

# CLI 模式 + 远程前缀（通过 docker exec 或 ssh）
flowsim submit --scheduler slurm \
    --slurm-submit-via cli \
    --slurm-cli-prefix "docker exec -i slurmctld" \
    --slurm-partition normal \
    --collect perf --model-path Qwen/Qwen3-8B --gpus 1

# REST 模式（已弃用）
flowsim submit --scheduler slurm \
    --slurm-submit-via rest \
    --slurm-rest-url http://localhost:6820 \
    --slurm-jwt-token "$(scontrol token lifespan=3600 | cut -d= -f2)" \
    --collect perf --model-path Qwen/Qwen3-8B --gpus 1

# Dry-run 查看生成的 sbatch 脚本
flowsim submit --scheduler slurm ... --dry-run

# 查看状态（CLI 模式）
flowsim status --scheduler slurm --job 12345 \
    --slurm-submit-via cli \
    --slurm-cli-prefix "docker exec -i slurmctld"

# 取消任务
flowsim cancel --scheduler slurm --job 12345 \
    --slurm-submit-via cli
```

### 专有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--slurm-submit-via` | 提交模式：`cli`（sbatch）或 `rest`（slurmrestd，已弃用） | `cli` |
| `--slurm-cli-prefix` | CLI 命令前缀（如 `"docker exec -i slurmctld"`） | 空 |
| `--slurm-partition` | Slurm 分区 | 空 |
| `--slurm-time` | 任务时间限制 | `02:00:00` |
| `--slurm-account` | 计费账户 | 空 |
| `--slurm-constraint` | 节点约束 | 空 |
| `--slurm-container-runtime` | 容器运行时：`docker` / `enroot` / `none` | `none` |
| `--slurm-container-mounts` | 容器挂载 | 空 |
| `--slurm-module` | `module load` 命令（可重复） | 空 |
| `--slurm-extra-sbatch` | 额外 `#SBATCH` 指令（可重复） | 空 |
| `--slurm-rest-url` | slurmrestd URL（REST 模式需要） | 空 |
| `--slurm-jwt-token` | JWT token（REST 模式需要） | 空 |
| `--slurm-api-version` | slurmrestd API 版本 | `v0.0.40` |
| `--slurm-no-verify-ssl` | 跳过 TLS 验证 | `false` |

### container_runtime 说明

| 值 | 说明 |
|----|------|
| `none` | 直接在计算节点上运行（节点已有 Python/sglang 环境）|
| `docker` | 在分配的节点上 `docker run` |
| `enroot` | 使用 `srun --container-image` (NVIDIA enroot) |

### 工作原理

**CLI 模式：**
1. `render()` 生成完整的 sbatch 脚本（含 `#SBATCH` 指令 + profiling 命令）
2. `submit()` 通过 `sbatch --parsable` 提交（脚本通过 stdin 传入）
3. `status()` 通过 `scontrol show job` 查询（无需 slurmdbd）
4. `cancel()` 通过 `scancel` 取消
5. `list_jobs()` 通过 `squeue` 列出

如果 Slurm 命令不在本地 PATH 中，可通过 `--slurm-cli-prefix` 指定前缀，例如：
- `"docker exec -i slurmctld"` — 通过 Docker 容器
- `"ssh login-node"` — 通过 SSH

**REST 模式（已弃用）：**
1. 同上生成 sbatch 脚本
2. `submit()` 通过 HTTP POST 到 slurmrestd 的 `/slurm/{version}/job/submit`
3. 所有操作通过 slurmrestd REST API + JWT 认证

### Docker Compose 本地测试集群

```bash
# 启动 Slurm 集群（slurmctld + 1 计算节点 + 1 GPU）
cd dockerfiles/
docker compose -f slurm-compose.yaml up -d

# 检查集群状态
docker exec slurmctld sinfo

# 运行 Slurm 集成测试
python -m pytest tests/integration/test_scheduler_local.py::TestSlurmScheduler -v -x

# 清理
docker compose -f slurm-compose.yaml down -v
```

---

## 配置文件

配置保存在 `~/.flowsim/` 目录下，通过 `flowsim init` 生成：

```
~/.flowsim/
├── k8s.yaml      # K8s 调度器配置
└── slurm.yaml    # Slurm 调度器配置
```

参数优先级（从高到低）：
1. CLI flag（`--slurm-partition gpu`）
2. 环境变量（`FLOWSIM_SLURM_PARTITION=gpu`）
3. 配置文件（`~/.flowsim/slurm.yaml`）
4. 内置默认值

### 示例 k8s.yaml

```yaml
kubeconfig: /home/user/.kube/config
namespace: default
host_output_dir: /host-stage-traces
runtime_class_name: nvidia
shm_size: 16Gi
```

### 示例 slurm.yaml

```yaml
partition: gpu
account: my-project
time: "02:00:00"
container_runtime: none
submit_via: cli
cli_prefix: ""
```

---

## 输出目录结构

所有调度器产生统一的 trace 输出结构：

```
stage_traces/{scheduler}/{YYYYMMDD_HHMMSS}/
├── bs1_input2048_ctx0/
│   ├── *.trace.json.gz           # 原始 trace
│   ├── parsed/*.csv              # 解析后的 CSV
│   ├── merged/*_merged.trace.csv # 合并的 trace CSV
│   ├── shape_traces/             # Shape trace（collect=shapes/all）
│   ├── shape_parsed/*.csv        # Shape 解析 CSV
│   ├── analysis_extend.json      # Extend 阶段分析
│   └── analysis_decode.json      # Decode 阶段分析
├── logs/
│   ├── server_*.stdout.log
│   └── server_*.stderr.log
└── sweep_summary.json
```

---

## PD Disaggregation（实验性）

支持 Prefill-Decode 分离部署：

```bash
flowsim submit --scheduler k8s \
    --pd \
    --collect perf \
    --model-path Qwen/Qwen3-235B-A22B-FP8 \
    --tp 4 --gpus 8 \
    --disagg-transfer-backend mooncake
```

这会生成两个 Job：一个 prefill 实例，一个 decode 实例。
