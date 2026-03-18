#!/usr/bin/env bash
# dev-teardown.sh — tear down FlowSim test clusters
#
# Usage:
#   ./dockerfiles/dev-teardown.sh          # teardown both
#   ./dockerfiles/dev-teardown.sh kind     # kind only
#   ./dockerfiles/dev-teardown.sh slurm    # slurm only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KIND_CLUSTER_NAME="flowsim"

log()  { printf "\033[1;32m[teardown]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[teardown]\033[0m %s\n" "$*"; }

teardown_kind() {
    # Delete device plugin namespace (contains per-node DaemonSets)
    if command -v kubectl >/dev/null; then
        kubectl delete namespace nvidia-device-plugin --ignore-not-found 2>/dev/null || true
    fi
    if command -v kind >/dev/null && kind get clusters 2>/dev/null | grep -q "^${KIND_CLUSTER_NAME}$"; then
        log "Deleting kind cluster '${KIND_CLUSTER_NAME}'..."
        kind delete cluster --name "${KIND_CLUSTER_NAME}"
    else
        warn "kind cluster '${KIND_CLUSTER_NAME}' not found, skipping"
    fi
}

teardown_slurm() {
    if docker compose -f "${SCRIPT_DIR}/slurm-compose.yaml" ps --quiet 2>/dev/null | head -1 | grep -q .; then
        log "Stopping Slurm containers..."
        docker compose -f "${SCRIPT_DIR}/slurm-compose.yaml" down -v
    else
        warn "Slurm containers not running, skipping"
    fi
}

target="${1:-all}"

case "${target}" in
    kind)   teardown_kind ;;
    slurm)  teardown_slurm ;;
    all)    teardown_kind; teardown_slurm ;;
    *)      echo "Usage: $0 [kind|slurm|all]"; exit 1 ;;
esac

log "Done."
