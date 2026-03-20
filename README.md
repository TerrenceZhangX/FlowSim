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
- [Stage Profiling](#stage-profiling)
- [Scheduler Backends](#scheduler-backends)
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

### 2. Profile (Generate Traces)

Use `flowsim submit` to capture stage-separated traces (EXTEND + DECODE), parse them, and run cross-rank analysis — all in one step. See [Stage Profiling](#stage-profiling) for how stages and collection modes work.

```bash
pip install -e .
flowsim submit --scheduler local \
    --collect all \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 1 --input-len 2048 --existing-ctx 0 --decode-tokens 2 --gpus 1 \
    --extra-server-opts "--load-format dummy"
```

For K8s / Slurm clusters, see [Scheduler Backends](#scheduler-backends).

**Tip:** Trace files can be visualized at [Perfetto UI](https://ui.perfetto.dev/). For multi-GPU traces, merge them first:
```bash
python utils/merge_trace.py --trace_dir stage_traces/local/*/bs1_input2048_ctx0 --output merged.json
```

### 3. Simulate (Run Hardware Simulation)

Build and start the LLMCompass backend, then submit parsed traces for kernel-level simulation:

```bash
# Build backend image
sudo docker build -t llmcompass-backend -f backend/LLMCompass/Dockerfile backend/LLMCompass/

# Terminal 1: Start backend
sudo docker run --rm -p 8000:8000 llmcompass-backend

# Terminal 2: Run simulation
sudo docker run --rm --network=host \
  -v /data/flowsim:/workspace \
  flowsim-image \
  python -m scripts.run_simulate \
    --trace-file /workspace/traces/bs1_input2048_ctx0/*-TP-0-EXTEND.trace.json.gz \
    --api-url http://127.0.0.1:8000 \
    --artifact-dir /workspace/simulate/llmcompass
```

### 4. Inspect Results

```bash
ls -lh /data/flowsim/traces/       # Stage-separated traces + parsed CSVs
ls -lh /data/flowsim/simulate/     # Simulation artifacts
```

---

## Stage Profiling

FlowSim performs **stage-separated** profiling: it captures prefill (EXTEND) and decode traces independently, parses them, runs cross-rank kernel analysis, and optionally collects kernel input shapes.

### How stages work

Each profiling request produces **two** stage-separated traces:
- **EXTEND** (prefill) — processes `input_len` new tokens (with optional `existing_ctx` tokens already in KV cache)
- **DECODE** — captures `decode-tokens` decode batch steps (default 2)

### Collection modes

| Mode | What it does |
|---|---|
| `--collect perf` | Profile a single (bs, input_len, existing_ctx) point → trace → parse → cross-rank analysis |
| `--collect shapes` | Re-run **without CUDA graph** to capture kernel input shapes, then merge into timing CSVs |
| `--collect all` | Both phases back-to-back (auto-restarts the server in between) |

### Examples

```bash
# Basic profiling
flowsim submit --scheduler local \
    --collect perf \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 1 --input-len 2048 --existing-ctx 0 --decode-tokens 2 --gpus 1 \
    --extra-server-opts "--load-format dummy"

# With existing KV cache context
flowsim submit --scheduler local \
    --collect perf \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 4 --input-len 512 --existing-ctx 4096 --decode-tokens 2 --gpus 1 \
    --extra-server-opts "--load-format dummy"

# Full pipeline (perf + shapes)
flowsim submit --scheduler local \
    --collect all \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --tp 1 --bs 1 --input-len 2048 --existing-ctx 0 --decode-tokens 2 --gpus 1 \
    --extra-server-opts "--load-format dummy"

# Multi-point sweep
flowsim submit --scheduler local \
    --collect all \
    --model-path workload/models/configs/Qwen3-235B-A22B \
    --sweep 1:2048:0 4:2048:0 8:2048:0 --decode-tokens 2 --gpus 1 \
    --extra-server-opts "--load-format dummy"
```

For K8s / Slurm clusters, replace `--scheduler local` with `k8s` or `slurm`. See [schedulers/README.md](schedulers/README.md) for full scheduler documentation.

### Output structure

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
│   ├── server_*.{stdout,stderr}.log
│   ├── shape_server_*.{stdout,stderr}.log
│   └── {job_name}_*.{stdout,stderr}.log
└── sweep_summary.json
```

- `parsed/`: Per-rank timing CSVs extracted from traces.
- `merged/`: Timing + shape columns joined into a single CSV per rank/stage.
- `shape_traces/` / `shape_parsed/`: Raw and parsed shape-profiling traces (generated by `--collect shapes` or `--collect all`).
- `logs/`: Server, shape-server, and job stdout/stderr logs.

### Utilities (`utils/`)

| File | Purpose |
|---|---|
| `utils/cross_rank_agg.py` | Cross-rank kernel aggregation (symmetric collectives → min, asymmetric → max, compute → mean) |
| `utils/shape_merge.py` | Merge kernel shape data into timing CSVs |
| `utils/merge_trace.py` | Merge multi-rank traces into a single Perfetto-compatible file |

---

## Scheduler Backends

For submitting profiling jobs to **local Docker**, **Kubernetes**, or **Slurm** clusters, use the `flowsim` CLI. See [schedulers/README.md](schedulers/README.md) for full documentation including per-scheduler parameters, configuration, and environment variables.

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