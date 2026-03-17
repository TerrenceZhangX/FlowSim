"""Kubernetes Job scheduler for FlowSim profiling.

Uses the ``kubernetes`` Python client for remote submission.
The ``render()`` / ``dry_run()`` path uses stdlib only (json fallback if
PyYAML is not installed — JSON is valid YAML 1.2 and ``kubectl`` accepts it).
"""

from __future__ import annotations

import json

from schedulers.base import BaseScheduler, ProfileJobSpec

# Optional: nicer YAML output for dry-run.
try:
    import yaml as _yaml  # type: ignore[import-untyped]

    def _dump(obj: dict) -> str:
        return _yaml.safe_dump(obj, default_flow_style=False, sort_keys=False)

except ImportError:
    _yaml = None  # type: ignore[assignment]

    def _dump(obj: dict) -> str:  # type: ignore[misc]
        return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


class K8sScheduler(BaseScheduler):
    """Generate and optionally submit a Kubernetes Job for profiling.

    Parameters
    ----------
    namespace : str
        Kubernetes namespace for the Job.
    kubeconfig : str, optional
        Path to a kubeconfig file.  When empty, the ``kubernetes`` client
        tries in-cluster config, then ``~/.kube/config``.
    context : str, optional
        kubeconfig context to activate.
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
        kubeconfig: str = "",
        context: str = "",
        pvc_name: str = "",
        host_output_dir: str = "",
        node_selector: dict[str, str] | None = None,
        service_account: str = "",
        shm_size: str = "16Gi",
    ) -> None:
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.context = context
        self.pvc_name = pvc_name
        self.host_output_dir = host_output_dir
        self.node_selector = node_selector or {}
        self.service_account = service_account
        self.shm_size = shm_size

    def render(self, spec: ProfileJobSpec) -> str:
        return _dump(self._build_job_dict(spec))

    # -----------------------------------------------------------------
    # Build a plain-dict manifest (used by both render and submit)
    # -----------------------------------------------------------------
    def _build_job_dict(self, spec: ProfileJobSpec) -> dict:
        """Return the Job manifest as a nested Python dict."""
        job_name = spec.default_job_name()[:63]
        cmd = spec.build_profile_command()

        # volumes + mounts
        volume_mounts = [{"name": "dshm", "mountPath": "/dev/shm"}]
        volumes: list[dict] = [
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": self.shm_size}},
        ]
        if self.pvc_name:
            volume_mounts.append({"name": "output", "mountPath": spec.output_dir})
            volumes.append({"name": "output", "persistentVolumeClaim": {"claimName": self.pvc_name}})
        elif self.host_output_dir:
            volume_mounts.append({"name": "output", "mountPath": spec.output_dir})
            volumes.append({"name": "output", "hostPath": {"path": self.host_output_dir, "type": "DirectoryOrCreate"}})

        container = {
            "name": "profiler",
            "image": spec.image,
            "imagePullPolicy": "IfNotPresent",
            "workingDir": "/flowsim",
            "command": cmd,
            "env": [{"name": "SGLANG_PROFILE_KERNELS", "value": "1"}],
            "resources": {
                "limits": {"nvidia.com/gpu": str(spec.gpus)},
                "requests": {"nvidia.com/gpu": str(spec.gpus)},
            },
            "volumeMounts": volume_mounts,
        }

        pod_spec: dict = {
            "restartPolicy": "Never",
            "containers": [container],
            "volumes": volumes,
        }
        if self.service_account:
            pod_spec["serviceAccountName"] = self.service_account
        if self.node_selector:
            pod_spec["nodeSelector"] = dict(self.node_selector)

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.namespace,
                "labels": {"app": "flowsim", "component": "profiling", "collect": spec.collect},
            },
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 86400,
                "template": {
                    "metadata": {"labels": {"app": "flowsim", "component": "profiling"}},
                    "spec": pod_spec,
                },
            },
        }

    def submit(self, spec: ProfileJobSpec) -> str:
        """Submit via the ``kubernetes`` Python client (``pip install kubernetes``)."""
        try:
            from kubernetes import client as k8s_client, config as k8s_config
        except ImportError:
            raise RuntimeError(
                "The 'kubernetes' package is required for --submit. "
                "Install it with: pip install kubernetes"
            )

        # Load kubeconfig / in-cluster config
        config_kwargs: dict = {}
        if self.kubeconfig:
            config_kwargs["config_file"] = self.kubeconfig
        if self.context:
            config_kwargs["context"] = self.context

        try:
            k8s_config.load_kube_config(**config_kwargs)
        except k8s_config.ConfigException:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                hint = ""
                if not self.kubeconfig:
                    hint = " Try --k8s-kubeconfig /path/to/kubeconfig."
                raise RuntimeError(
                    "No valid Kubernetes configuration found. "
                    "Checked kubeconfig file and in-cluster environment."
                    + hint
                )

        body = self._build_job_dict(spec)
        batch_api = k8s_client.BatchV1Api()
        resp = batch_api.create_namespaced_job(
            namespace=self.namespace,
            body=body,
        )
        return f"job.batch/{resp.metadata.name} created (namespace={resp.metadata.namespace})"

    # -----------------------------------------------------------------
    # Helpers shared by status / logs
    # -----------------------------------------------------------------

    def _load_k8s(self):
        """Load kubeconfig and return (BatchV1Api, CoreV1Api)."""
        from kubernetes import client as k8s_client, config as k8s_config

        config_kwargs: dict = {}
        if self.kubeconfig:
            config_kwargs["config_file"] = self.kubeconfig
        if self.context:
            config_kwargs["context"] = self.context
        try:
            k8s_config.load_kube_config(**config_kwargs)
        except k8s_config.ConfigException:
            k8s_config.load_incluster_config()

        return k8s_client.BatchV1Api(), k8s_client.CoreV1Api()

    def status(self, job_id: str) -> dict:
        """Query K8s Job status by job name."""
        try:
            from kubernetes import client as k8s_client
        except ImportError:
            raise RuntimeError("pip install kubernetes")

        batch_api, core_api = self._load_k8s()

        job = batch_api.read_namespaced_job(name=job_id, namespace=self.namespace)
        st = job.status

        # Determine state
        if st.succeeded and st.succeeded > 0:
            state = "Succeeded"
        elif st.failed and st.failed > 0:
            state = "Failed"
        elif st.active and st.active > 0:
            state = "Running"
        else:
            state = "Pending"

        # Pod info
        pods = core_api.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"job-name={job_id}",
        )
        pod_statuses = []
        for pod in pods.items:
            phase = pod.status.phase
            node = pod.spec.node_name or "unscheduled"
            pod_statuses.append(f"{pod.metadata.name} ({phase}, node={node})")

        output_hint = ""
        if self.pvc_name:
            output_hint = f"Traces persisted on PVC '{self.pvc_name}'"
        elif self.host_output_dir:
            output_hint = f"Traces at hostPath {self.host_output_dir} on the scheduled node"
        else:
            output_hint = "WARNING: no PVC or hostPath configured — traces are lost when pod exits"

        msg_parts = [f"Job: {job_id}  Namespace: {self.namespace}  State: {state}"]
        if pod_statuses:
            msg_parts.append("Pods: " + ", ".join(pod_statuses))
        msg_parts.append(output_hint)

        return {
            "state": state,
            "message": "\n".join(msg_parts),
            "output_hint": output_hint,
        }

    def logs(self, job_id: str, *, tail: int = 100) -> str:
        """Retrieve logs from the pod(s) of a K8s Job."""
        try:
            from kubernetes import client as k8s_client
        except ImportError:
            raise RuntimeError("pip install kubernetes")

        _, core_api = self._load_k8s()

        pods = core_api.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"job-name={job_id}",
        )
        if not pods.items:
            return f"No pods found for job {job_id} in namespace {self.namespace}"

        parts = []
        for pod in pods.items:
            name = pod.metadata.name
            try:
                log_text = core_api.read_namespaced_pod_log(
                    name=name,
                    namespace=self.namespace,
                    tail_lines=tail,
                )
            except Exception as exc:
                log_text = f"(error reading logs: {exc})"
            parts.append(f"=== {name} ===\n{log_text}")

        return "\n".join(parts)
