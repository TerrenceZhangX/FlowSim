"""Kubernetes Job scheduler for FlowSim profiling.

Uses the ``kubernetes`` Python client for remote submission.
The ``render()`` / ``dry_run()`` path uses stdlib only (json fallback if
PyYAML is not installed — JSON is valid YAML 1.2 and ``kubectl`` accepts it).
"""

from __future__ import annotations

import json

from schedulers.base import BaseScheduler, JobResult, ProfileJobSpec


def _k8s_job_state(status) -> str:
    """Derive a human-readable state string from a K8s Job status object."""
    if status.succeeded and status.succeeded > 0:
        return "Succeeded"
    if status.failed and status.failed > 0:
        return "Failed"
    if status.active and status.active > 0:
        return "Running"
    return "Pending"


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
    runtime_class_name : str, optional
        Kubernetes RuntimeClass name for the pod (e.g., ``"nvidia"`` for
        CDI-based GPU injection in Kind clusters).
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
        runtime_class_name: str = "",
    ) -> None:
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.context = context
        self.pvc_name = pvc_name
        self.host_output_dir = host_output_dir
        self.node_selector = node_selector or {}
        self.service_account = service_account
        self.shm_size = shm_size
        self.runtime_class_name = runtime_class_name

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
            {
                "name": "dshm",
                "emptyDir": {"medium": "Memory", "sizeLimit": self.shm_size},
            },
        ]
        if self.pvc_name:
            volume_mounts.append(
                {"name": "output", "mountPath": spec.output_dir}
            )
            volumes.append(
                {
                    "name": "output",
                    "persistentVolumeClaim": {"claimName": self.pvc_name},
                }
            )
        elif self.host_output_dir:
            # Mount at base traces dir so the full directory structure
            # (e.g. k8s/{timestamp}/bs1_...) is preserved on the host.
            volume_mounts.append(
                {"name": "output", "mountPath": "/flowsim/stage_traces"}
            )
            volumes.append(
                {
                    "name": "output",
                    "hostPath": {
                        "path": self.host_output_dir,
                        "type": "DirectoryOrCreate",
                    },
                }
            )

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
        if self.runtime_class_name:
            pod_spec["runtimeClassName"] = self.runtime_class_name
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
                "labels": {
                    "app": "flowsim",
                    "component": "profiling",
                    "collect": spec.collect,
                },
            },
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 86400,
                "template": {
                    "metadata": {
                        "labels": {"app": "flowsim", "component": "profiling"}
                    },
                    "spec": pod_spec,
                },
            },
        }

    def submit(self, spec: ProfileJobSpec) -> JobResult:
        """Submit via the ``kubernetes`` Python client (``pip install kubernetes``)."""
        batch_api, _ = self._load_k8s()

        body = self._build_job_dict(spec)
        resp = batch_api.create_namespaced_job(
            namespace=self.namespace,
            body=body,
        )
        return JobResult(
            job_id=resp.metadata.name,
            scheduler="k8s",
            state="Submitted",
            output_dir=spec.output_dir,
            message=f"job.batch/{resp.metadata.name} created (namespace={resp.metadata.namespace})",
        )

    # -----------------------------------------------------------------
    # Helpers shared by status / logs
    # -----------------------------------------------------------------

    def _load_k8s(self):
        """Load kubeconfig and return (BatchV1Api, CoreV1Api).

        Raises RuntimeError with actionable message on failure.
        """
        try:
            from kubernetes import client as k8s_client, config as k8s_config
        except ImportError:
            raise RuntimeError(
                "The 'kubernetes' package is required. "
                "Install it with: pip install kubernetes"
            )

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
                hint = (
                    " Try --k8s-kubeconfig /path/to/kubeconfig."
                    if not self.kubeconfig
                    else ""
                )
                raise RuntimeError(
                    "No valid Kubernetes configuration found. "
                    "Checked kubeconfig file and in-cluster environment." + hint
                )

        return k8s_client.BatchV1Api(), k8s_client.CoreV1Api()

    def cancel(self, job_id: str) -> str:
        """Delete a K8s Job (and its pods) by name."""
        from kubernetes import client as k8s_client

        batch_api, _ = self._load_k8s()
        batch_api.delete_namespaced_job(
            name=job_id,
            namespace=self.namespace,
            body=k8s_client.V1DeleteOptions(propagation_policy="Foreground"),
        )
        return f"job.batch/{job_id} deleted (namespace={self.namespace})"

    def status(self, job_id: str) -> dict:
        """Query K8s Job status by job name."""
        batch_api, core_api = self._load_k8s()

        job = batch_api.read_namespaced_job(
            name=job_id, namespace=self.namespace
        )

        # Determine state
        state = _k8s_job_state(job.status)

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

        msg_parts = [
            f"Job: {job_id}  Namespace: {self.namespace}  State: {state}"
        ]
        if pod_statuses:
            msg_parts.append("Pods: " + ", ".join(pod_statuses))
        msg_parts.append(output_hint)

        return {
            "state": state,
            "message": "\n".join(msg_parts),
            "output_hint": output_hint,
        }

    def logs(
        self, job_id: str, *, tail: int = 100, follow: bool = False
    ) -> str:
        """Show where logs are and how to access them for a K8s Job."""
        _, core_api = self._load_k8s()

        pods = core_api.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"job-name={job_id}",
        )
        if not pods.items:
            return (
                f"No pods found for job {job_id} in namespace {self.namespace}"
            )

        if follow:
            # Stream logs from the first running/succeeded pod
            for pod in pods.items:
                name = pod.metadata.name
                if pod.status.phase in ("Running", "Succeeded"):
                    # Use kubectl follow since the Python client follow is blocking
                    return (
                        f"Follow logs:\n"
                        f"  kubectl logs -f {name} -n {self.namespace}"
                    )
            name = pods.items[0].metadata.name
            return f"Follow logs:\n  kubectl logs -f {name} -n {self.namespace}"

        parts: list[str] = []

        # Pod info
        for pod in pods.items:
            name = pod.metadata.name
            phase = pod.status.phase
            parts.append(f"Pod: {name}  ({phase})")

        parts.append("")

        # Commands to view pod stdout
        parts.append("View profiling script output:")
        for pod in pods.items:
            name = pod.metadata.name
            parts.append(f"  kubectl logs {name} -n {self.namespace}")
            parts.append(
                f"  kubectl logs {name} -n {self.namespace} --tail={tail}"
            )

        parts.append("")

        # Persistent log files
        if self.pvc_name:
            parts.append(
                f"Server logs + traces persisted on PVC '{self.pvc_name}'."
            )
            parts.append("Copy to local machine:")
            for pod in pods.items:
                name = pod.metadata.name
                if pod.status.phase in ("Running", "Succeeded"):
                    parts.append(
                        f"  kubectl cp {self.namespace}/{name}:/flowsim/stage_traces ./stage_traces"
                    )
                    break
            else:
                parts.append(
                    "  (pod not running — mount the PVC in another pod to retrieve files)"
                )
        elif self.host_output_dir:
            parts.append(f"Server logs + traces at hostPath on the node:")
            parts.append(f"  {self.host_output_dir}/")
            parts.append(f"  {self.host_output_dir}/logs/")
            # Identify node
            for pod in pods.items:
                if pod.spec.node_name:
                    parts.append(f"  Node: {pod.spec.node_name}")
                    parts.append(
                        f"  scp {pod.spec.node_name}:{self.host_output_dir}/ ./stage_traces/"
                    )
                    break

        return "\n".join(parts)

    def list_jobs(self, *, status_filter: str = "") -> list[dict]:
        """List FlowSim Jobs in the namespace (label: app=flowsim)."""
        batch_api, _ = self._load_k8s()

        jobs = batch_api.list_namespaced_job(
            namespace=self.namespace,
            label_selector="app=flowsim",
        )
        result: list[dict] = []
        for job in jobs.items:
            state = _k8s_job_state(job.status)

            if status_filter and state.lower() != status_filter.lower():
                continue

            created = ""
            if job.metadata.creation_timestamp:
                created = job.metadata.creation_timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            result.append(
                {
                    "job_id": job.metadata.name,
                    "name": job.metadata.name,
                    "state": state,
                    "namespace": self.namespace,
                    "created": created,
                }
            )
        return result
