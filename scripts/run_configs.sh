#!/usr/bin/env bash
# ┌─────────────────────────────────────────────────────────────────────┐
# │  Run profiling across 4 parallelism configs for Qwen3-235B-A22B-FP8│
# │                                                                     │
# │  Supports all --collect modes plus offline re-analysis:             │
# │    perf      — collect traces + parse + analyze                     │
# │    shapes    — collect kernel shapes (no CUDA graph)                │
# │    all       — perf → restart server → shapes                      │
# │    reanalyze — re-run cross_rank_agg on existing parsed CSVs       │
# │               (offline, no server needed)                           │
# └─────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   bash run_configs.sh perf           # collect perf traces
#   bash run_configs.sh shapes         # collect kernel shapes
#   bash run_configs.sh all            # perf → restart server → shapes
#   bash run_configs.sh reanalyze      # re-run analysis offline
#
# Filter configs:
#   RUN_CONFIGS=P1 bash run_configs.sh perf
#   RUN_CONFIGS=P1,P3 bash run_configs.sh reanalyze
#
# From host:
#   sg docker -c "docker exec flowsim-sglang bash -c \
#     'cd /workspace/scripts && bash run_configs.sh perf'"

set -euo pipefail

# ── Resolve mode ──────────────────────────────────────────
MODE="${1:-perf}"
case "$MODE" in
    perf|shapes|all|reanalyze) ;;
    *)
        echo "Usage: $0 {perf|shapes|all|reanalyze} [RUN_CONFIGS=P1,P2,...]"
        exit 1
        ;;
esac

# ── Shared settings ───────────────────────────────────────
MODEL="Qwen/Qwen3-235B-A22B-FP8"
HOST="0.0.0.0"
PORT=30001
SCRIPTS="/workspace/scripts"
export PYTHONPATH="/workspace/utils${PYTHONPATH:+:$PYTHONPATH}"

BS_GRID="1,4,16,64,128,256"
CTX_GRID="2048,4096,8192,16384,32768"

RUN_CONFIGS="${RUN_CONFIGS:-P1,P2,P3,P4}"

cd "$SCRIPTS"

# ── Config table: tag → (dir_name, server_opts) ──────────
declare -A DIR_NAMES=(
    [P1]="sweep_P1_tp4"
    [P2]="sweep_P2_ep4"
    [P3]="sweep_P3_dpattn"
    [P4]="sweep_P4_dpattn_ep4"
)
declare -A SERVER_OPTS=(
    [P1]="--tp 4"
    [P2]="--tp 4 --ep 4"
    [P3]="--tp 4 --dp 4 --enable-dp-attention"
    [P4]="--tp 4 --dp 4 --ep 4 --enable-dp-attention"
)

UTILS="/workspace/utils"

# ── reanalyze: offline re-run of cross_rank_agg ──────────
reanalyze_config() {
    local tag="$1"
    local sweep_dir="/workspace/${DIR_NAMES[$tag]}"

    if [[ ! ",$RUN_CONFIGS," == *",$tag,"* ]]; then
        echo "[SKIP] $tag (not in RUN_CONFIGS=$RUN_CONFIGS)"
        return
    fi
    if [ ! -d "$sweep_dir" ]; then
        echo "[SKIP] $tag: $sweep_dir not found"
        return
    fi

    echo ""
    echo "========================================================"
    echo "  Re-analyzing $tag  ($sweep_dir)"
    echo "========================================================"

    local stages="EXTEND DECODE"
    for bp in "$sweep_dir"/bs*; do
        [ -d "$bp/parsed" ] || continue
        local bn
        bn=$(basename "$bp")

        for stage in $stages; do
            local csv_count
            csv_count=$(find "$bp/parsed" -name "*${stage}*.csv" 2>/dev/null | wc -l)
            [ "$csv_count" -eq 0 ] && continue
            _total=$((_total + 1))

            if python3 "$UTILS/cross_rank_agg.py" \
                   --csv-dir "$bp/parsed" --stage "$stage" \
                   --output-json "$bp/analysis_${stage,,}.json" -q 2>/dev/null; then
                _ok=$((_ok + 1))
            else
                echo "  [FAIL] $bn $stage"
                _fail=$((_fail + 1))
            fi
        done
    done
    echo "  [$tag] done"
}

# ── perf / shapes / all: server-based profiling ──────────
run_config() {
    local tag="$1"

    if [[ ! ",$RUN_CONFIGS," == *",$tag,"* ]]; then
        echo "[SKIP] $tag (not in RUN_CONFIGS=$RUN_CONFIGS)"
        return 0
    fi

    local dir_name="${DIR_NAMES[$tag]}"
    local opts="${SERVER_OPTS[$tag]}"

    echo ""
    echo "========================================================"
    echo "  $tag [$MODE]: $opts"
    echo "  output → /workspace/$dir_name"
    echo "========================================================"

    python3 run_stage_profile.py \
        --collect "$MODE" \
        --launch-server \
        --server-opts "--model-path $MODEL --host $HOST --port $PORT $opts" \
        --bs-grid "$BS_GRID" --ctx-grid "$CTX_GRID" \
        --output-dir "/workspace/$dir_name" \
        --log-dir "/workspace/sweep_server_logs/${tag}_${MODE}"

    echo ""
    echo "[$tag] $MODE DONE ✓"
    echo ""
}

# ── Execute ───────────────────────────────────────────────
if [[ "$MODE" == "reanalyze" ]]; then
    _total=0; _ok=0; _fail=0
    for tag in P1 P2 P3 P4; do
        reanalyze_config "$tag"
    done
    echo ""
    echo "========================================================"
    echo "  Re-analysis complete: $_ok/$_total OK, $_fail failed"
    echo "========================================================"
else
    for tag in P1 P2 P3 P4; do
        run_config "$tag"
    done
    echo "========================================================"
    echo "  ALL CONFIGS COMPLETE  (mode=$MODE)"
    echo "========================================================"
fi
