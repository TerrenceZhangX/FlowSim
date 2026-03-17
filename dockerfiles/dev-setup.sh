#!/usr/bin/env bash
# dev-setup.sh — one-shot setup for FlowSim test clusters (kind + Slurm)
#
# Usage:
#   ./dockerfiles/dev-setup.sh          # setup both kind + slurm
#   ./dockerfiles/dev-setup.sh kind     # kind only
#   ./dockerfiles/dev-setup.sh slurm    # slurm only
#
# Teardown:
#   ./dockerfiles/dev-teardown.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KIND_VERSION="v0.27.0"
KIND_CLUSTER_NAME="flowsim"
KUBECTL_STABLE_URL="https://dl.k8s.io/release/stable.txt"
NVIDIA_DEVICE_PLUGIN="https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml"

log()  { printf "\033[1;32m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[setup]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[setup]\033[0m %s\n" "$*" >&2; exit 1; }

# ----------------------------------------------------------------
# Dependency checks & auto-install
# ----------------------------------------------------------------
ensure_docker() {
    command -v docker >/dev/null || err "Docker is required but not installed."
    docker info >/dev/null 2>&1 || err "Docker daemon not running."
    log "Docker: $(docker --version)"
}

ensure_kind() {
    if command -v kind >/dev/null; then
        log "kind already installed: $(kind version)"
        return
    fi
    log "Installing kind ${KIND_VERSION}..."
    curl -fsSLo /tmp/kind "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-amd64"
    chmod +x /tmp/kind
    sudo mv /tmp/kind /usr/local/bin/kind
    log "kind installed: $(kind version)"
}

ensure_kubectl() {
    if command -v kubectl >/dev/null; then
        log "kubectl already installed"
        return
    fi
    log "Installing kubectl..."
    local ver
    ver="$(curl -fsSL "${KUBECTL_STABLE_URL}")"
    curl -fsSLo /tmp/kubectl "https://dl.k8s.io/release/${ver}/bin/linux/amd64/kubectl"
    chmod +x /tmp/kubectl
    sudo mv /tmp/kubectl /usr/local/bin/kubectl
    log "kubectl installed: $(kubectl version --client --short 2>/dev/null || true)"
}

# ----------------------------------------------------------------
# Kind cluster
# ----------------------------------------------------------------
setup_kind() {
    ensure_docker
    ensure_kind
    ensure_kubectl

    if kind get clusters 2>/dev/null | grep -q "^${KIND_CLUSTER_NAME}$"; then
        warn "kind cluster '${KIND_CLUSTER_NAME}' already exists, skipping creation"
    else
        log "Creating kind cluster '${KIND_CLUSTER_NAME}' (1 control-plane + 2 GPU workers)..."
        kind create cluster --name "${KIND_CLUSTER_NAME}" \
            --config "${SCRIPT_DIR}/kind-multi-node.yaml"
        log "Installing NVIDIA device plugin..."
        kubectl apply -f "${NVIDIA_DEVICE_PLUGIN}"
    fi

    log "Cluster nodes:"
    kubectl get nodes -o wide
    echo

    log "Initializing FlowSim K8s config..."
    local kubeconfig
    kubeconfig="${HOME}/.kube/config"
    flowsim init k8s \
        --kubeconfig "${kubeconfig}" \
        --context "kind-${KIND_CLUSTER_NAME}" \
        --namespace default \
        --force
    echo
    log "Kind cluster ready. Test with:"
    log "  flowsim submit --scheduler k8s --collect perf --model-path <path> --dry-run"
}

# ----------------------------------------------------------------
# Slurm cluster (docker compose)
# ----------------------------------------------------------------
setup_slurm() {
    ensure_docker

    if ! docker compose version >/dev/null 2>&1; then
        err "docker compose v2 is required but not available."
    fi

    log "Building and starting Slurm cluster (slurmctld + 2 slurmd + slurmrestd)..."
    docker compose -f "${SCRIPT_DIR}/slurm-compose.yaml" up -d --build

    log "Waiting for slurmctld to become ready..."
    local retries=30
    while ! docker exec slurmctld sinfo >/dev/null 2>&1; do
        retries=$((retries - 1))
        if [ "${retries}" -le 0 ]; then
            err "slurmctld did not become ready in time"
        fi
        sleep 2
    done

    log "Slurm cluster status:"
    docker exec slurmctld sinfo
    echo

    log "Initializing FlowSim Slurm config..."
    flowsim init slurm \
        --rest-url "http://localhost:6820" \
        --partition normal \
        --account default \
        --jwt-token-cmd "docker exec slurmctld scontrol token lifespan=3600" \
        --force
    echo
    log "Slurm cluster ready. Test with:"
    log "  flowsim submit --scheduler slurm --collect perf --model-path <path> --dry-run"
}

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
target="${1:-all}"

case "${target}" in
    kind)
        setup_kind
        ;;
    slurm)
        setup_slurm
        ;;
    all)
        setup_kind
        echo
        setup_slurm
        ;;
    *)
        echo "Usage: $0 [kind|slurm|all]"
        exit 1
        ;;
esac

echo
log "All done. Teardown with: ./dockerfiles/dev-teardown.sh"
