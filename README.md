# FlowSim: Workload Simulation Pipeline

FlowSim is a lightweight and extensible toolchain for simulating and analyzing kernel-level performance based on real-world inference workloads. It bridges high-level framework profiling and low-level hardware simulation through a three-stage workflow:

1. **Profiling**: Extracts real AI workload traces from optimized inference frameworks (e.g., [sglang](https://docs.sglang.ai/)), capturing operator-level execution from production-like workloads.
2. **Trace Translation**: Converts profiled pytorch traces into detailed kernel-level representations enriched with fine-grained tensor information such as shapes and data types.
3. **Simulation**: Feeds the translated traces into hardware-level simulators (e.g., LLMCompass) to enable GPU kernel simulation, simulator calibration, and performance prediction.

Although simulation is driven at the operator/kernel level, performance analysis and observation are performed end-to-end (E2E) to provide a holistic view of workload behavior.

FlowSim is most suitable for:

- Developers who need detailed, fine-grained end-to-end workload profiling.
- Researchers and simulator developers who require accurate kernel-level traces to calibrate and evaluate GPU performance simulators.

The project supports rapid deployment using Docker, includes scripts for environment setup and profiling, and offers flexible configuration options.

---

## Table of Contents

- [Getting Started](#getting-started)
- [For Developers](#for-developers)
- [Risks and limitations](#risks-and-limitations)
- [License](#license)
- [Trademarks](#trademarks)

---


## Getting Started

### Prerequisites

- Linux system with NVIDIA GPU(s) (for profiling)
- [Docker](https://docs.docker.com/get-docker/) with [NVIDIA Container Runtime](https://github.com/NVIDIA/nvidia-container-runtime)
- [Make](https://www.gnu.org/software/make/)
- [NVIDIA NGC account](https://org.ngc.nvidia.com/setup/api-key) for pulling NVIDIA base images
- ~50GB disk space for images and traces

**Note:** Run `git submodule update --init --recursive` before building, as the LLMCompass submodule requires initialization.

### 1. Build the Docker Image

```bash
cd /path/to/flowsim
make build-docker
```

This creates a local image named `flowsim-image` with FlowSim patches already applied to sglang.

### 2. Run Profile → Parse → Simulate

Create workspace directories on your host for storing traces and results:

```bash
mkdir -p /data/flowsim-profile
mkdir -p /data/flowsim-simulate
```

#### Step 1: Profile (Generate Traces)

```bash
sudo docker run --gpus=all \
  -v /data/flowsim-profile:/workspace/profile \
  -v /data/flowsim-simulate:/workspace/simulate \
  -w /flowsim \
  --cap-add=SYS_ADMIN \
  --network=host \
  --shm-size 911G \
  flowsim-image \
  python scripts/run_profile.py \
    --profile-dir /workspace/profile \
    --log-dir /workspace/profile/logs \
    --bench-timeout 3600 \
    --server-opts "--model-path /flowsim/workload/models/configs/deepseek/ --load-format dummy --tp 4 --ep 4 --host 0.0.0.0 --port 30001 --attention-backend flashinfer --disable-cuda-graph" \
    --bench-opts "--backend sglang --host 0.0.0.0 --port 30001 --dataset-name defined-len --prefill-decode-lens 1024:8 --num-prompts 1 --profile"
```

**What this does:**
- Starts an sglang server with profiling enabled
- Runs benchmark requests against it
- Generates `*.trace.json.gz` files in `/data/flowsim-profile` (mounted as `/workspace/profile`)

**Note:** The first run will be slow (~10 minutes) due to DeepGEMM kernel warmup and compilation. For stable performance, avoid using `--rm` flag and reuse the same container using `sudo docker exec -it <container_id> bash`. Subsequent runs with similar configurations will be faster.

**Tip:** 
- Adjust `--server-opts` and `--bench-opts` to match your model, parallelism (TP/DP/EP), and workload requirements. All `sglang.launch_server` and `bench_serving.py` parameters are supported.
- Trace files can be visualized using [Perfetto UI](https://ui.perfetto.dev/) by uploading the `.trace.json.gz` files directly.
- For multi-GPU profiling (TP > 1), merge individual traces into a single file for a global view:
  ```bash
  python /flowsim/utils/merge_trace.py \
    --trace_dir /data/flowsim-profile \
    --output /data/flowsim-profile/merged_trace.json
  ```
  Then visualize the merged trace at [Perfetto UI](https://ui.perfetto.dev/).

#### Step 2: Parse (Convert Trace to CSV)

```bash
sudo docker run --rm \
  -v /data/flowsim-profile:/workspace/profile \
  -v /data/flowsim-simulate:/workspace/simulate \
  -w /flowsim \
  flowsim-image \
  python -m scripts.run_parse \
    --trace-file /workspace/profile/your-trace-name-TP-0.trace.json.gz \
    --output-dir /workspace/simulate
```

Replace `your-trace-name-TP-0.trace.json.gz` with the actual filename from step 1.

**What this does:**
- Parses the trace file
- Extracts kernel-level information (operator, shapes, dtypes)
- Generates a CSV file and JSON summary in `/data/flowsim-simulate` (mounted as `/workspace/simulate`)

**Fallback:** If you don't have a GPU or can't run profiling, use the demo trace shipped with the repo:

```bash
sudo docker run --rm \
  -v /data/flowsim-simulate:/workspace/simulate \
  -w /flowsim \
  flowsim-image \
  python -m scripts.run_parse \
    --trace-file /flowsim/demo/deepseekv3-TP-0.trace.json.gz \
    --output-dir /workspace/simulate
```

#### Step 3: Simulate (Run Hardware Simulation)

This step requires a running LLMCompass backend. First, build the backend image:

```bash
sudo docker build -t llmcompass-backend -f backend/LLMCompass/Dockerfile backend/LLMCompass/
```

Then start the backend:

```bash
# Terminal 1: Start LLMCompass backend
sudo docker run --rm -p 8000:8000 llmcompass-backend
```

Then in another terminal, run the simulation:

```bash
# Terminal 2: Run simulation
sudo docker run --rm \
  --network=host \
  -v /data/flowsim-profile:/workspace/profile \
  -v /data/flowsim-simulate:/workspace/simulate \
  flowsim-image \
  python -m scripts.run_simulate \
    --trace-file /workspace/profile/your-trace-name-TP-0.trace.json.gz \
    --api-url http://127.0.0.1:8000 \
    --artifact-dir /workspace/simulate/llmcompass
```

**What this does:**
- Parses the trace into kernels
- Submits each kernel to the LLMCompass backend `/tasks` API
- Polls until all tasks complete
- Writes request/response artifacts to `/workspace/simulate/llmcompass`

### 3. Inspect Results

All generated files are available on your host at `/data/`:

```bash
ls -lh /data/flowsim-profile/      # Raw trace files
ls -lh /data/flowsim-simulate/     # Parsed CSV, summary, simulation artifacts
```

---

## Stage Profiling (`run_stage_profile.py`)

`scripts/run_stage_profile.py` is the single entry-point for **stage-separated** profiling: it captures prefill (EXTEND) and decode traces independently, parses them, runs cross-rank kernel analysis, and optionally collects kernel input shapes.

### Quick reference

Each profiling request produces **two** stage-separated traces:
- **EXTEND** (prefill) — processes `input_len` new tokens (with optional `existing_ctx` tokens already in KV cache)
- **DECODE** — profiler captures `decode-tokens` decode batch steps

The profiler captures exactly **one** EXTEND batch and **decode-tokens** DECODE batches per run.

| Flag | Description | Default |
|---|---|---|
| `--input-len` | Number of new prefill tokens per request (EXTEND) | 2048 |
| `--existing-ctx` | Tokens already in KV cache from a prior request (0 = cold prefill) | 0 |
| `--bs` | Batch size (concurrent requests) | 1 |
| `--decode-tokens` | Number of decode tokens to generate (= number of decode batches profiled) | 32 |

| Mode | What it does |
|---|---|
| `--collect perf` | Profile a single (bs, input_len, existing_ctx) point → trace (EXTEND + DECODE) → parse → cross-rank analysis |
| `--collect shapes` | Re-run **without CUDA graph** to capture kernel input shapes, then merge into timing CSVs (both EXTEND and DECODE) |
| `--collect all` | Both phases back-to-back (auto-restarts the server in between). Requires `--launch-server`. |

`--collect` is required. Use `perf`, `shapes`, or `all`.

### Examples

**Cold prefill** (server already running):

```bash
python3 scripts/run_stage_profile.py \
    --collect perf \
    --bs 1 --input-len 2048 --decode-tokens 32 \
    --output-dir /workspace/traces \
    --host 0.0.0.0 --port 30001
```

**With existing KV cache context:**

```bash
python3 scripts/run_stage_profile.py \
    --collect perf \
    --bs 4 --input-len 512 --existing-ctx 4096 --decode-tokens 32 \
    --output-dir /workspace/traces \
    --launch-server \
    --server-opts "--model-path Qwen/Qwen3-235B-A22B-FP8 --tp 4 --host 0.0.0.0 --port 30001"
```

**Collect shapes only** (requires a no-CUDA-graph server):

```bash
python3 scripts/run_stage_profile.py \
    --collect shapes \
    --output-dir /workspace/sweep_P1_tp4 \
    --launch-server \
    --server-opts "--model-path Qwen/Qwen3-235B-A22B-FP8 --tp 4 --host 0.0.0.0 --port 30001"
```

When `--collect shapes` is used with `--launch-server`, the server is automatically started with `--disable-cuda-graph --disable-cuda-graph-padding`.

**Full pipeline** (perf → auto-restart → shapes → merge):

```bash
python3 scripts/run_stage_profile.py \
    --collect all \
    --output-dir /workspace/sweep_P1_tp4 \
    --launch-server \
    --server-opts "--model-path Qwen/Qwen3-235B-A22B-FP8 --tp 4 --host 0.0.0.0 --port 30001"
```


### Output structure

```
sweep_P1_tp4/
├── sweep_summary.json
├── bs1_input2048_ctx0/
│   ├── *-TP-*-EXTEND.trace.json.gz
│   ├── *-TP-*-DECODE.trace.json.gz
│   ├── parsed/
│   │   ├── TP-0-EXTEND.csv
│   │   ├── TP-0-DECODE.csv
│   │   └── ...
│   ├── analysis_extend.json
│   └── analysis_decode.json
└── ...
```

After `--collect shapes`, each `parsed/TP-*-DECODE.csv` gains an `Input Dims` column with kernel tensor shapes.

### Helper scripts

| Script | Purpose |
|---|---|
| `tests/integration/test_stage_profile_configs.py` | Integration tests for `--collect {perf,shapes,all}` across parallelism configs. Run with `pytest` inside Docker. Filter with `RUN_CONFIGS=P1`. |

### Utilities (`utils/`)

| File | Purpose |
|---|---|
| `utils/cross_rank_agg.py` | Cross-rank kernel aggregation (symmetric collectives → min, asymmetric → max, compute → mean) |
| `utils/shape_merge.py` | Merge kernel shape data into timing CSVs |
| `utils/net.py` | Shared networking helpers (`wait_for_port`) |
| `utils/merge_trace.py` | Merge multi-rank traces into a single Perfetto-compatible file |

---

## For Developers

### Customizing Profiling Workloads

For programmatic profiling setup, see `tests/integration/test_profile.py`, which shows how to:

- Launch an sglang server with profiling enabled via environment variables (`SGLANG_TORCH_PROFILER_DIR`, `SGLANG_PROFILE_KERNELS`)
- Run custom benchmarks against the server to generate trace files

Adjust `--server-opts` and `--bench-opts` in `scripts/run_profile.py` to match your model and workload. All `sglang.launch_server` and `bench_serving.py` parameters are supported. See the [sglang profiling documentation](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html) for details.

### LLMCompass Backend Integration

FlowSim currently integrates with [LLMCompass](https://github.com/TerrenceZhangX/LLMCompass) as a reference GPU performance simulator. In this setup:

- Each parsed kernel (from a FlowSim trace) is turned into a small JSON payload and submitted to the LLMCompass backend via its `/tasks` API.
- The backend estimates runtime characteristics per kernel (e.g., latency) under a user-specified hardware configuration.
- Results are polled asynchronously until all tasks reach a terminal state, then written as JSON artifacts for further analysis.

LLMCompass itself supports richer workflows (e.g., compiling full operator graphs, system-level roofline analysis, and running graphs on real GPUs). FlowSim focuses on the **kernel-level, trace-driven** usage: taking end-to-end traces from real inference workloads and feeding them into a calibrated backend to study per-kernel performance, compare hardware configurations, or validate simulator behavior.

### Kernel metadata and unknown kernels

After you obtain a profile trace (`*.trace.json.gz`), you will typically run the parser once to inspect kernel-level status:

```bash
python -m scripts.run_parse \
  --trace-file /flowsim/server_profile/your-trace-name.trace.json.gz \
  --output-dir /flowsim/server_simulate
```

During parsing, FlowSim looks up kernel metadata (e.g., tensor shapes and dtypes) in `kernels.json`. Any kernels it cannot match are written to `unknown_kernels.json` at the project root, with incomplete or `unknown` parameter descriptions.

To enrich metadata for new or unsupported kernels:

- Open `unknown_kernels.json`, locate the entries of interest, and fill in the missing information (e.g., `operation`, `params[*].role`, `example_dim`, `example_dtype`, `description`).
- Copy the completed entries into `kernels.json` to make them part of the known-kernel database.
- Re-run `scripts/run_parse.py` on your trace; those kernels should now be treated as known and will no longer appear in `unknown_kernels.json`.

Tensor shapes and dtypes for Triton kernels are surfaced via the FlowSim tracing hooks. When `SGLANG_PROFILE_KERNELS=1`, `sglang.launch_server` calls `register_kernels_for_profiling` from `sglang.srt.tracing.hook_register`, which attaches tensor metadata to PyTorch profiler labels for registered kernels. If you introduce custom Triton kernels that still appear as "unknown" after parsing, you may need to extend this registration logic and/or add corresponding entries to `kernels.json`.


---

## Risks and limitations

FlowSim was not designed or evaluated for all possible downstream purposes. Users should consider its inherent limitations when selecting use cases and must evaluate and mitigate accuracy, safety, and fairness concerns specific to each intended downstream use.

---

## License

This project is released under the MIT License. For the full license text, see the `LICENSE` file in the repository root.

---

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.

Any use of third-party trademarks or logos is subject to those third parties' trademark and brand policies.