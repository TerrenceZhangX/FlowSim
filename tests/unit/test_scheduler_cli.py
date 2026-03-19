"""Unit tests for the scheduler CLI (flowsim init / submit) and backends."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from schedulers.base import ProfileJobSpec
from schedulers.k8s import K8sScheduler
from schedulers.local import LocalScheduler
from schedulers.slurm import SlurmScheduler

# =========================================================================
# ProfileJobSpec
# =========================================================================


class TestProfileJobSpec:
    """Tests for ProfileJobSpec dataclass methods."""

    @pytest.fixture()
    def spec(self) -> ProfileJobSpec:
        return ProfileJobSpec(
            collect="perf",
            model_path="Qwen/Qwen3-8B",
            tp=2,
            bs=4,
            input_len=1024,
        )

    def test_default_job_name(self, spec: ProfileJobSpec):
        name = spec.default_job_name()
        assert name == "flowsim-perf-qwen3-8b-bs4-il1024"

    def test_custom_job_name(self, spec: ProfileJobSpec):
        spec.job_name = "my-job"
        assert spec.default_job_name() == "my-job"

    def test_build_server_opts_basic(self, spec: ProfileJobSpec):
        opts = spec.build_server_opts()
        assert "--model-path Qwen/Qwen3-8B" in opts
        assert "--tp 2" in opts

    def test_build_server_opts_dp(self, spec: ProfileJobSpec):
        spec.dp = 4
        assert "--dp 4" in spec.build_server_opts()

    def test_build_server_opts_extra(self, spec: ProfileJobSpec):
        spec.extra_server_opts = "--some-flag"
        assert "--some-flag" in spec.build_server_opts()

    def test_build_profile_command(self, spec: ProfileJobSpec):
        cmd = spec.build_profile_command()
        assert cmd[0] == "python3"
        assert "scripts/run_stage_profile.py" in cmd[1]
        assert "--collect" in cmd
        assert "perf" in cmd
        assert "--bs" in cmd
        assert "4" in cmd

    def test_build_shell_command_quotes_server_opts(self, spec: ProfileJobSpec):
        shell = spec.build_shell_command()
        # server-opts contains spaces, must be quoted
        assert "--server-opts '" in shell or '--server-opts "' in shell


# =========================================================================
# K8sScheduler.render
# =========================================================================


class TestK8sScheduler:
    """Tests for K8s Job manifest generation."""

    @pytest.fixture()
    def scheduler(self) -> K8sScheduler:
        return K8sScheduler(
            namespace="ml-team",
            kubeconfig="/fake/kubeconfig",
            context="prod",
            shm_size="32Gi",
        )

    @pytest.fixture()
    def spec(self) -> ProfileJobSpec:
        return ProfileJobSpec(
            collect="perf",
            model_path="Qwen/Qwen3-8B",
            gpus=2,
        )

    def test_render_valid_yaml(self, scheduler, spec):
        rendered = scheduler.render(spec)
        doc = yaml.safe_load(rendered)
        assert doc["apiVersion"] == "batch/v1"
        assert doc["kind"] == "Job"

    def test_render_namespace(self, scheduler, spec):
        doc = yaml.safe_load(scheduler.render(spec))
        assert doc["metadata"]["namespace"] == "ml-team"

    def test_render_gpu_resources(self, scheduler, spec):
        doc = yaml.safe_load(scheduler.render(spec))
        container = doc["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"

    def test_render_shm_size(self, scheduler, spec):
        doc = yaml.safe_load(scheduler.render(spec))
        volumes = doc["spec"]["template"]["spec"]["volumes"]
        dshm = [v for v in volumes if v["name"] == "dshm"][0]
        assert dshm["emptyDir"]["sizeLimit"] == "32Gi"

    def test_render_pvc_volume(self, spec):
        sched = K8sScheduler(namespace="default", pvc_name="my-pvc")
        doc = yaml.safe_load(sched.render(spec))
        volumes = doc["spec"]["template"]["spec"]["volumes"]
        pvc_vol = [v for v in volumes if v["name"] == "output"]
        assert len(pvc_vol) == 1
        assert pvc_vol[0]["persistentVolumeClaim"]["claimName"] == "my-pvc"

    def test_render_host_output_dir(self, spec):
        sched = K8sScheduler(namespace="default", host_output_dir="/data/out")
        doc = yaml.safe_load(sched.render(spec))
        volumes = doc["spec"]["template"]["spec"]["volumes"]
        host_vol = [v for v in volumes if v["name"] == "output"]
        assert len(host_vol) == 1
        assert host_vol[0]["hostPath"]["path"] == "/data/out"

    def test_render_node_selector(self, spec):
        sched = K8sScheduler(namespace="default", node_selector={"gpu": "h100"})
        doc = yaml.safe_load(sched.render(spec))
        pod_spec = doc["spec"]["template"]["spec"]
        assert pod_spec["nodeSelector"]["gpu"] == "h100"

    def test_render_service_account(self, spec):
        sched = K8sScheduler(namespace="default", service_account="runner")
        doc = yaml.safe_load(sched.render(spec))
        pod_spec = doc["spec"]["template"]["spec"]
        assert pod_spec["serviceAccountName"] == "runner"

    def test_render_labels(self, scheduler, spec):
        doc = yaml.safe_load(scheduler.render(spec))
        labels = doc["metadata"]["labels"]
        assert labels["app"] == "flowsim"
        assert labels["collect"] == "perf"


# =========================================================================
# SlurmScheduler.render
# =========================================================================


class TestSlurmScheduler:
    """Tests for Slurm sbatch script generation."""

    @pytest.fixture()
    def scheduler(self) -> SlurmScheduler:
        return SlurmScheduler(
            partition="gpu-h100",
            time_limit="01:00:00",
            account="my-proj",
        )

    @pytest.fixture()
    def spec(self) -> ProfileJobSpec:
        return ProfileJobSpec(
            collect="perf",
            model_path="Qwen/Qwen3-8B",
            gpus=4,
        )

    def test_render_shebang(self, scheduler, spec):
        script = scheduler.render(spec)
        assert script.startswith("#!/bin/bash\n")

    def test_render_sbatch_directives(self, scheduler, spec):
        script = scheduler.render(spec)
        assert "#SBATCH --partition=gpu-h100" in script
        assert "#SBATCH --gpus-per-node=4" in script
        assert "#SBATCH --time=01:00:00" in script
        assert "#SBATCH --account=my-proj" in script

    def test_render_env_vars(self, scheduler, spec):
        script = scheduler.render(spec)
        assert "SGLANG_PROFILE_KERNELS=1" in script

    def test_render_command(self, scheduler, spec):
        script = scheduler.render(spec)
        assert "scripts/run_stage_profile.py" in script
        assert "--collect perf" in script

    def test_render_docker_runtime(self, spec):
        sched = SlurmScheduler(
            partition="gpu",
            container_runtime="docker",
            container_mounts="/data:/data",
        )
        script = sched.render(spec)
        assert "docker run" in script
        assert "-v /data:/data" in script

    def test_render_enroot_runtime(self, spec):
        sched = SlurmScheduler(
            partition="gpu",
            container_runtime="enroot",
        )
        script = sched.render(spec)
        assert "srun --container-image" in script

    def test_render_modules(self, spec):
        sched = SlurmScheduler(
            partition="gpu",
            modules=["cuda/12.6", "anaconda3"],
        )
        script = sched.render(spec)
        assert "module load cuda/12.6" in script
        assert "module load anaconda3" in script

    def test_render_extra_sbatch(self, spec):
        sched = SlurmScheduler(
            partition="gpu",
            extra_sbatch=["--mem=64G", "--exclusive"],
        )
        script = sched.render(spec)
        assert "#SBATCH --mem=64G" in script
        assert "#SBATCH --exclusive" in script

    def test_render_constraint(self, spec):
        sched = SlurmScheduler(partition="gpu", constraint="gpu80g")
        script = sched.render(spec)
        assert "#SBATCH --constraint=gpu80g" in script


# =========================================================================
# LocalScheduler.render
# =========================================================================


class TestLocalScheduler:
    """Tests for local execution backend."""

    @pytest.fixture(autouse=True)
    def _skip_image_check(self):
        with mock.patch.object(LocalScheduler, "_check_image_exists"):
            yield

    @pytest.fixture()
    def spec(self) -> ProfileJobSpec:
        return ProfileJobSpec(
            collect="perf",
            model_path="Qwen/Qwen3-8B",
        )

    def test_render_with_gpus(self, spec):
        sched = LocalScheduler(gpus="0,1")
        output = sched.render(spec)
        assert "device=0,1" in output
        assert "docker run" in output

    def test_render_without_gpus(self, spec):
        sched = LocalScheduler(gpus="")
        output = sched.render(spec)
        assert "CUDA_VISIBLE_DEVICES" not in output

    def test_render_has_command(self, spec):
        sched = LocalScheduler()
        output = sched.render(spec)
        assert "scripts/run_stage_profile.py" in output
        assert "SGLANG_PROFILE_KERNELS=1" in output

    def test_render_workdir(self, spec):
        sched = LocalScheduler(workdir="/my/project")
        output = sched.render(spec)
        # Docker mode: workdir is used for log scanning, not in the docker command
        assert "docker run" in output
        assert "scripts/run_stage_profile.py" in output

    def test_dry_run_equals_render(self, spec):
        sched = LocalScheduler(gpus="0")
        assert sched.dry_run(spec) == sched.render(spec)


# =========================================================================
# CLI: flowsim init
# =========================================================================


class TestCLIInit:
    """Tests for `flowsim init` subcommand."""

    def test_init_no_args_shows_help(self, capsys):
        from scripts.cli import _cmd_init

        with pytest.raises(SystemExit) as exc_info:
            _cmd_init([])
        assert exc_info.value.code != 0

    def test_init_k8s_creates_template(self, tmp_path: Path):
        config_dir = tmp_path / "flowsim"
        with mock.patch("scripts.cli._CONFIG_DIR", config_dir):
            from scripts.cli import _cmd_init

            rc = _cmd_init(["k8s"])
        assert rc == 0
        cfg_file = config_dir / "k8s.yaml"
        assert cfg_file.exists()
        content = cfg_file.read_text()
        assert "kubeconfig:" in content
        assert "namespace:" in content
        # Template should have comments
        assert content.startswith("#")
        # Should be valid YAML
        cfg = yaml.safe_load(content)
        assert "kubeconfig" in cfg
        assert "namespace" in cfg

    def test_init_slurm_creates_template(self, tmp_path: Path):
        config_dir = tmp_path / "flowsim"
        with mock.patch("scripts.cli._CONFIG_DIR", config_dir):
            from scripts.cli import _cmd_init

            rc = _cmd_init(["slurm"])
        assert rc == 0
        cfg_file = config_dir / "slurm.yaml"
        assert cfg_file.exists()
        content = cfg_file.read_text()
        assert "partition:" in content
        assert "cli_prefix:" in content
        # Template should have comments
        assert content.startswith("#")
        cfg = yaml.safe_load(content)
        assert "partition" in cfg

    def test_init_refuses_overwrite(self, tmp_path: Path):
        config_dir = tmp_path / "flowsim"
        config_dir.mkdir()
        (config_dir / "slurm.yaml").write_text("existing: true\n")

        with mock.patch("scripts.cli._CONFIG_DIR", config_dir):
            from scripts.cli import _cmd_init

            rc = _cmd_init(["slurm"])
        assert rc != 0  # should refuse

    def test_init_force_overwrite(self, tmp_path: Path):
        config_dir = tmp_path / "flowsim"
        config_dir.mkdir()
        (config_dir / "slurm.yaml").write_text("existing: true\n")

        with mock.patch("scripts.cli._CONFIG_DIR", config_dir):
            from scripts.cli import _cmd_init

            rc = _cmd_init(["slurm", "--force"])
        assert rc == 0
        content = (config_dir / "slurm.yaml").read_text()
        assert "partition:" in content
        assert "existing" not in content

    def test_init_config_copies_file(self, tmp_path: Path):
        # User has an existing config
        user_cfg = tmp_path / "my-k8s.yaml"
        user_cfg.write_text("namespace: prod\nkubeconfig: /etc/kube\n")

        config_dir = tmp_path / "flowsim"
        with mock.patch("scripts.cli._CONFIG_DIR", config_dir):
            from scripts.cli import _cmd_init

            rc = _cmd_init(["k8s", "--config", str(user_cfg)])
        assert rc == 0
        installed = config_dir / "k8s.yaml"
        assert installed.exists()
        cfg = yaml.safe_load(installed.read_text())
        assert cfg["namespace"] == "prod"

    def test_init_config_missing_file(self):
        from scripts.cli import _cmd_init

        rc = _cmd_init(["k8s", "--config", "/nonexistent/path.yaml"])
        assert rc != 0


# =========================================================================
# CLI: flowsim submit (parse/dry-run only, no actual submission)
# =========================================================================


class TestCLISubmit:
    """Tests for `flowsim submit` argument parsing and dry-run."""

    @pytest.fixture(autouse=True)
    def _skip_image_check(self):
        with mock.patch.object(LocalScheduler, "_check_image_exists"):
            yield

    def _run(self, *args: str, expect_ok: bool = True) -> str:
        """Run submit via the Python function, capture stdout."""
        from scripts.submit_profile import main as submit_main
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            submit_main(list(args))
        return buf.getvalue()

    def test_submit_help(self, capsys):
        from scripts.submit_profile import main as submit_main

        with pytest.raises(SystemExit) as exc_info:
            submit_main(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--scheduler" in out
        assert "local" in out

    def test_submit_missing_required(self):
        from scripts.submit_profile import main as submit_main

        with pytest.raises(SystemExit):
            submit_main([])

    def test_submit_local_dry_run(self):
        out = self._run(
            "--scheduler",
            "local",
            "--collect",
            "perf",
            "--model-path",
            "Qwen/Qwen3-8B",
            "--dry-run",
        )
        assert "scripts/run_stage_profile.py" in out
        assert "SGLANG_PROFILE_KERNELS=1" in out

    def test_submit_local_dry_run_with_gpus(self):
        out = self._run(
            "--scheduler",
            "local",
            "--collect",
            "perf",
            "--model-path",
            "Qwen/Qwen3-8B",
            "--local-gpus",
            "0,1",
            "--dry-run",
        )
        assert "device=0,1" in out

    def test_submit_k8s_dry_run(self):
        out = self._run(
            "--scheduler",
            "k8s",
            "--collect",
            "perf",
            "--model-path",
            "Qwen/Qwen3-8B",
            "--k8s-namespace",
            "default",
            "--dry-run",
        )
        assert "apiVersion: batch/v1" in out
        assert "kind: Job" in out

    def test_submit_slurm_dry_run(self):
        out = self._run(
            "--scheduler",
            "slurm",
            "--collect",
            "perf",
            "--model-path",
            "Qwen/Qwen3-8B",
            "--slurm-partition",
            "gpu",
            "--dry-run",
        )
        assert "#!/bin/bash" in out
        assert "#SBATCH --partition=gpu" in out


# =========================================================================
# Config loading
# =========================================================================


class TestConfig:
    """Tests for config file loading and saving."""

    def test_save_and_load_yaml(self, tmp_path: Path):
        from schedulers.config import _save_yaml, _load_yaml

        data = {"partition": "gpu", "account": "proj"}
        path = tmp_path / "test.yaml"
        _save_yaml(path, data)
        loaded = _load_yaml(path)
        assert loaded == data

    def test_cfg_get(self):
        from schedulers.config import cfg_get

        cfg = {"key": "value", "empty": ""}
        assert cfg_get(cfg, "key", "default") == "value"
        assert cfg_get(cfg, "empty", "default") == ""
        assert cfg_get(cfg, "missing", "default") == "default"
