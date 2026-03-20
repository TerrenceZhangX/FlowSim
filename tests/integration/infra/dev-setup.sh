#!/usr/bin/env bash
# dev-setup.sh — one-shot setup for FlowSim test clusters (kind + Slurm)
#
# Usage:
#   ./tests/integration/infra/dev-setup.sh          # setup both kind + slurm
#   ./tests/integration/infra/dev-setup.sh kind     # kind only
#   ./tests/integration/infra/dev-setup.sh slurm    # slurm only
#
# Teardown:
#   ./tests/integration/infra/dev-teardown.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KIND_VERSION="v0.27.0"
KIND_CLUSTER_NAME="flowsim"
KIND_WORKERS=("${KIND_CLUSTER_NAME}-worker")
KUBECTL_STABLE_URL="https://dl.k8s.io/release/stable.txt"
HELM_INSTALL_URL="https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3"
NVIDIA_CTK_KEYRING="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"

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
# Kind cluster with NVIDIA GPU via CDI
# (Official approach from NVIDIA k8s-device-plugin demo)
# https://github.com/NVIDIA/k8s-device-plugin/tree/main/demo/clusters/kind
# ----------------------------------------------------------------
ensure_nvidia_runtime() {
    # Docker must use nvidia as default runtime so Kind node containers get GPU access
    command -v nvidia-ctk >/dev/null || err "nvidia-container-toolkit is required (nvidia-ctk not found)."
    command -v nvidia-smi >/dev/null || err "NVIDIA driver not found (nvidia-smi missing)."
    log "nvidia-ctk: $(nvidia-ctk --version 2>&1 | head -1)"

    if ! docker info 2>/dev/null | grep -q "Default Runtime: nvidia"; then
        log "Setting nvidia as default Docker runtime..."
        sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
        sudo systemctl restart docker
        log "Docker restarted with nvidia runtime as default"
    else
        log "Docker already using nvidia as default runtime"
    fi

    # Required: accept-nvidia-visible-devices-as-volume-mounts must be true
    # for Kind GPU passthrough via /var/run/nvidia-container-devices/all
    local cfg="/etc/nvidia-container-runtime/config.toml"
    if grep -qE '^\s*accept-nvidia-visible-devices-as-volume-mounts\s*=\s*true' "$cfg" 2>/dev/null; then
        log "accept-nvidia-visible-devices-as-volume-mounts already enabled"
    else
        log "Enabling accept-nvidia-visible-devices-as-volume-mounts in $cfg..."
        if grep -qE '#?\s*accept-nvidia-visible-devices-as-volume-mounts' "$cfg" 2>/dev/null; then
            sudo sed -i 's/#*\s*accept-nvidia-visible-devices-as-volume-mounts.*/accept-nvidia-visible-devices-as-volume-mounts = true/' "$cfg"
        else
            echo 'accept-nvidia-visible-devices-as-volume-mounts = true' | sudo tee -a "$cfg" >/dev/null
        fi
        sudo systemctl restart docker
        log "Host nvidia-container-runtime config updated and Docker restarted"
    fi
}

ensure_helm() {
    if command -v helm >/dev/null; then
        log "helm already installed: $(helm version --short 2>/dev/null)"
        return
    fi
    log "Installing helm..."
    curl -fsSL "${HELM_INSTALL_URL}" | bash
    log "helm installed: $(helm version --short)"
}

setup_kind() {
    ensure_docker
    ensure_nvidia_runtime
    ensure_kind
    ensure_kubectl
    ensure_helm

    if kind get clusters 2>/dev/null | grep -q "^${KIND_CLUSTER_NAME}$"; then
        warn "kind cluster '${KIND_CLUSTER_NAME}' already exists, skipping creation"
    else
        log "Creating kind cluster '${KIND_CLUSTER_NAME}' (1 control-plane + 1 GPU worker)..."
        kind create cluster --name "${KIND_CLUSTER_NAME}" \
            --config "${SCRIPT_DIR}/kind-multi-node.yaml"
    fi

    # ── Post-creation: configure GPU support inside each worker node ──
    for worker in "${KIND_WORKERS[@]}"; do
        log "=== Configuring ${worker} ==="

        # Step 1: Unmount masked /proc/driver/nvidia
        log "Unmounting /proc/driver/nvidia in ${worker}..."
        docker exec "${worker}" umount -R /proc/driver/nvidia 2>/dev/null || true

        # Step 2: Install nvidia-container-toolkit inside the worker node
        log "Installing nvidia-container-toolkit inside ${worker}..."
        docker exec "${worker}" bash -c "apt-get update && apt-get install -y gpg"
        docker exec "${worker}" bash -c "\
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | gpg --dearmor -o ${NVIDIA_CTK_KEYRING} \
            && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sed 's#deb https://#deb [signed-by=${NVIDIA_CTK_KEYRING}] https://#g' \
                | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
            && apt-get update \
            && apt-get install -y nvidia-container-toolkit"

        # Step 3: Configure CDI mode in containerd inside worker
        log "Configuring CDI mode for containerd in ${worker}..."
        docker exec "${worker}" bash -c "\
            nvidia-ctk config --set nvidia-container-runtime.modes.cdi.annotation-prefixes=nvidia.cdi.k8s.io/ \
            && nvidia-ctk runtime configure --runtime=containerd --cdi.enabled --config-source=command \
            && systemctl restart containerd"

        # Step 4: Label worker node for GPU presence
        kubectl --context "kind-${KIND_CLUSTER_NAME}" label node "${worker}" \
            --overwrite nvidia.com/gpu.present=true
    done

    # Step 5: Create nvidia RuntimeClass
    log "Creating nvidia RuntimeClass..."
    kubectl --context "kind-${KIND_CLUSTER_NAME}" apply -f - <<'RTEOF'
apiVersion: node.k8s.io/v1
handler: nvidia
kind: RuntimeClass
metadata:
  name: nvidia
RTEOF

    # Step 6: Deploy per-node NVIDIA device plugin DaemonSets
    # Each worker gets its own DaemonSet with a specific NVIDIA_VISIBLE_DEVICES
    # so the device plugin only discovers/advertises that worker's assigned GPU.
    # (Helm's single DaemonSet can't set different env per node.)
    log "Deploying NVIDIA device plugin (per-node GPU assignment)..."
    local CTX="kind-${KIND_CLUSTER_NAME}"
    local PLUGIN_IMAGE="nvcr.io/nvidia/k8s-device-plugin:v0.17.1"
    local gpu_idx=0
    for worker in "${KIND_WORKERS[@]}"; do
        local ds_name="nvidia-device-plugin-${worker##*-}"   # e.g. nvidia-device-plugin-worker
        kubectl --context "$CTX" apply -f - <<DPEOF
apiVersion: v1
kind: Namespace
metadata:
  name: nvidia-device-plugin
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: ${ds_name}
  namespace: nvidia-device-plugin
  labels:
    app: nvidia-device-plugin
    node: ${worker}
spec:
  selector:
    matchLabels:
      app: nvidia-device-plugin
      node: ${worker}
  template:
    metadata:
      labels:
        app: nvidia-device-plugin
        node: ${worker}
    spec:
      runtimeClassName: nvidia
      nodeSelector:
        kubernetes.io/hostname: ${worker}
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      priorityClassName: system-node-critical
      containers:
        - name: nvidia-device-plugin
          image: ${PLUGIN_IMAGE}
          env:
            - name: NVIDIA_VISIBLE_DEVICES
              value: "${gpu_idx}"
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
          volumeMounts:
            - name: device-plugin
              mountPath: /var/lib/kubelet/device-plugins
      volumes:
        - name: device-plugin
          hostPath:
            path: /var/lib/kubelet/device-plugins
DPEOF
        log "  ${worker} → GPU ${gpu_idx} (DaemonSet: ${ds_name})"
        gpu_idx=$((gpu_idx + 1))
    done

    # Step 7: Load flowsim-image into worker nodes (skip if already present)
    local FLOWSIM_IMAGE="flowsim-image:latest"
    if docker image inspect "${FLOWSIM_IMAGE}" >/dev/null 2>&1; then
        for worker in "${KIND_WORKERS[@]}"; do
            if docker exec "${worker}" crictl images 2>/dev/null | grep -q "flowsim-image.*latest"; then
                log "${FLOWSIM_IMAGE} already loaded in ${worker}, skipping"
            else
                log "Loading ${FLOWSIM_IMAGE} into ${worker} (~34GB, may take several minutes)..."
                if command -v pv >/dev/null; then
                    docker save "${FLOWSIM_IMAGE}" | pv -f -a -b | \
                        docker exec -i "${worker}" ctr -n k8s.io images import -
                else
                    docker save "${FLOWSIM_IMAGE}" | \
                        docker exec -i "${worker}" ctr -n k8s.io images import -
                fi
                log "${FLOWSIM_IMAGE} loaded into ${worker}"
            fi
        done
    else
        warn "${FLOWSIM_IMAGE} not found on host, skipping image load (build it first)"
    fi

    # Step 9: Wait for GPU resources
    log "Waiting for nvidia.com/gpu resources to appear (up to 180s)..."
    local gpu_retries=36
    while true; do
        gpu_count=$(kubectl --context "kind-${KIND_CLUSTER_NAME}" get nodes \
            -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null \
            | grep -cE '^[1-9]' || true)
        if [ "${gpu_count}" -ge 1 ]; then
            log "GPUs registered on ${gpu_count} node(s)"
            break
        fi
        gpu_retries=$((gpu_retries - 1))
        if [ "${gpu_retries}" -le 0 ]; then
            warn "GPUs not registered after 180s — debugging info:"
            kubectl --context "kind-${KIND_CLUSTER_NAME}" get pods -n nvidia-device-plugin -o wide 2>/dev/null || true
            kubectl --context "kind-${KIND_CLUSTER_NAME}" describe nodes 2>/dev/null | grep -A5 "Allocatable" || true
            break
        fi
        sleep 5
    done

    # Step 10: Init FlowSim K8s config
    log "Initializing FlowSim K8s config..."
    flowsim init k8s \
        --kubeconfig "${HOME}/.kube/config" \
        --context "kind-${KIND_CLUSTER_NAME}" \
        --namespace default \
        --host-output-dir /tmp/flowsim-traces \
        --runtime-class-name nvidia \
        --force

    log "Cluster nodes:"
    kubectl --context "kind-${KIND_CLUSTER_NAME}" get nodes -o wide
    echo

    log "GPU resources:"
    kubectl --context "kind-${KIND_CLUSTER_NAME}" get nodes \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null || true
    echo

    log "Kind cluster with GPU (CDI mode) ready."
}

# ----------------------------------------------------------------
# Slurm cluster (docker compose)
# ----------------------------------------------------------------
setup_slurm() {
    ensure_docker

    if ! docker compose version >/dev/null 2>&1; then
        err "docker compose v2 is required but not available."
    fi

    # HOST_WORKSPACE is used by slurm-compose.yaml for the read-only /workspace mount.
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
    export HOST_WORKSPACE="${HOST_WORKSPACE:-$(dirname "${REPO_ROOT}")}"

    log "Building and starting Slurm cluster (slurmctld + 2 slurmd + slurmrestd)..."
    log "  HOST_WORKSPACE=${HOST_WORKSPACE}"
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
log "All done. Teardown with: ./tests/integration/infra/dev-teardown.sh"
