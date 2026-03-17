"""Kubernetes Job scheduler for FlowSim profiling."""

from __future__ import annotations

import subprocess
import tempfile

from schedulers.base import BaseScheduler, ProfileJobSpec


class K8sScheduler(BaseScheduler):
    """Generate and optionally submit a Kubernetes Job for profiling.

    Parameters
    ----------
    namespace : str
        Kubernetes namespace for the Job.
    pvc_name : str, optional
        Name of a PersistentVolumeClaim to mount for trace output.
        If empty, uses ``emptyDir`` (traces are lost when the pod exits).
    host_output_dir : str, optional
        If set (and *pvc_name* is empty), use a ``hostPath`` volume at
        this path instead of a PVC.
    node_selector : dict, optional
        Kubernetes nodeSelector labels (e.g., ``{"gpu": "a100"}``).
    service_account : str, optional
        ServiceAccount name for the pod.
    shm_size : str
        Size of ``/dev/shm`` (shared memory).  Defaults to ``"16Gi"``.
    """

    def __init__(
        self,
        *,
        namespace: str = "default",
        pvc_name: str = "",
        host_output_dir: str = "",
        node_selector: dict[str, str] | None = None,
        service_account: str = "",
        shm_size: str = "16Gi",
    ) -> None:
        self.namespace = namespace
        self.pvc_name = pvc_name
        self.host_output_dir = host_output_dir
        self.node_selector = node_selector or {}
        self.service_account = service_account
        self.shm_size = shm_size

    def render(self, spec: ProfileJobSpec) -> str:
        job_name = spec.default_job_name()[:63]  # K8s name limit
        cmd = spec.build_profile_command()

        lines: list[str] = []
        _a = lines.append

        _a("apiVersion: batch/v1")
        _a("kind: Job")
        _a("metadata:")
        _a(f"  name: {job_name}")
        _a(f"  namespace: {self.namespace}")
        _a("  labels:")
        _a("    app: flowsim")
        _a("    component: profiling")
        _a(f"    collect: {spec.collect}")
        _a("spec:")
        _a("  backoffLimit: 0")
        _a("  ttlSecondsAfterFinished: 86400")
        _a("  template:")
        _a("    metadata:")
        _a("      labels:")
        _a("        app: flowsim")
        _a("        component: profiling")
        _a("    spec:")
        if self.service_account:
            _a(f"      serviceAccountName: {self.service_account}")
        if self.node_selector:
            _a("      nodeSelector:")
            for k, v in self.node_selector.items():
                _a(f"        {k}: {v}")
        _a("      restartPolicy: Never")
        _a("      containers:")
        _a("        - name: profiler")
        _a(f"          image: {spec.image}")
        _a("          imagePullPolicy: IfNotPresent")
        _a("          workingDir: /flowsim")
        _a("          command:")
        for c in cmd:
            _a(f'            - "{c}"')
        _a("          env:")
        _a("            - name: SGLANG_PROFILE_KERNELS")
        _a('              value: "1"')
        _a("          resources:")
        _a("            limits:")
        _a(f'              nvidia.com/gpu: "{spec.gpus}"')
        _a("            requests:")
        _a(f'              nvidia.com/gpu: "{spec.gpus}"')

        # volumeMounts
        _a("          volumeMounts:")
        _a("            - name: dshm")
        _a("              mountPath: /dev/shm")
        if self.pvc_name or self.host_output_dir:
            _a("            - name: output")
            _a(f"              mountPath: {spec.output_dir}")

        # volumes
        _a("      volumes:")
        _a("        - name: dshm")
        _a("          emptyDir:")
        _a("            medium: Memory")
        _a(f"            sizeLimit: {self.shm_size}")
        if self.pvc_name:
            _a("        - name: output")
            _a("          persistentVolumeClaim:")
            _a(f"            claimName: {self.pvc_name}")
        elif self.host_output_dir:
            _a("        - name: output")
            _a("          hostPath:")
            _a(f"            path: {self.host_output_dir}")
            _a("            type: DirectoryOrCreate")

        return "\n".join(lines) + "\n"

    def submit(self, spec: ProfileJobSpec) -> str:
        manifest = self.render(spec)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(manifest)
            f.flush()
            result = subprocess.run(
                ["kubectl", "apply", "-f", f.name],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"kubectl apply failed:\n{result.stderr.strip()}"
                )
            return result.stdout.strip()
