"""Load FlowSim scheduler config from per-scheduler YAML files.

Config file lookup (per scheduler):

K8s:
  1. ``FLOWSIM_K8S_CONFIG`` env var
  2. ``~/.flowsim/k8s.yaml``

Slurm:
  1. ``FLOWSIM_SLURM_CONFIG`` env var
  2. ``~/.flowsim/slurm.yaml``

Priority (highest → lowest):
    CLI flag  >  env var  >  config file  >  built-in default

Template files are in ``schedulers/templates/k8s.yaml`` and
``schedulers/templates/slurm.yaml``.  Copy to ``~/.flowsim/`` and edit.

For Slurm, use ``jwt_token_cmd`` instead of ``jwt_token`` to avoid
storing secrets in plaintext.  The command is executed at submit time
and its stdout is used as the token.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

# Optional: try PyYAML, fall back to JSON
try:
    import yaml as _yaml

    def _load_yaml(path: Path) -> dict:
        with open(path) as f:
            return _yaml.safe_load(f) or {}

except ImportError:
    import json as _json

    def _load_yaml(path: Path) -> dict:  # type: ignore[misc]
        """Fallback: accept JSON (valid YAML 1.2 subset)."""
        with open(path) as f:
            return _json.load(f)


_CONFIG_DIR = Path.home() / ".flowsim"


def _save_yaml(path: Path, data: dict) -> None:
    """Write a dict to a YAML file (uses PyYAML if available, else JSON)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml as _y
        with open(path, "w") as f:
            _y.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        import json as _j
        with open(path, "w") as f:
            _j.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")


def _resolve_path(env_var: str, filename: str) -> Path | None:
    """Return the config file path, or None if it doesn't exist."""
    env = os.environ.get(env_var)
    if env:
        p = Path(env)
        return p if p.is_file() else None
    default = _CONFIG_DIR / filename
    return default if default.is_file() else None


def load_k8s_config() -> dict:
    """Load ``~/.flowsim/k8s.yaml`` (or ``FLOWSIM_K8S_CONFIG``)."""
    path = _resolve_path("FLOWSIM_K8S_CONFIG", "k8s.yaml")
    if path is None:
        return {}
    try:
        return _load_yaml(path)
    except Exception:
        return {}


def load_slurm_config() -> dict:
    """Load ``~/.flowsim/slurm.yaml`` (or ``FLOWSIM_SLURM_CONFIG``)."""
    path = _resolve_path("FLOWSIM_SLURM_CONFIG", "slurm.yaml")
    if path is None:
        return {}
    try:
        return _load_yaml(path)
    except Exception:
        return {}


def resolve_jwt_token(slurm_cfg: dict) -> str:
    """Get the JWT token from config, executing jwt_token_cmd if needed."""
    token = slurm_cfg.get("jwt_token", "")
    if token:
        return str(token)

    cmd = slurm_cfg.get("jwt_token_cmd", "")
    if cmd:
        try:
            result = subprocess.run(
                shlex.split(cmd),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, OSError):
            pass

    return ""


def cfg_get(cfg: dict, key: str, fallback: str = "") -> str:
    """Get a value from a flat config dict, or fallback."""
    val = cfg.get(key)
    if val is not None:
        return str(val)
    return fallback


def resolve_default(env_var: str, cfg: dict, key: str, fallback: str = "") -> str:
    """Resolve a config value: env var > config file > fallback."""
    return os.environ.get(env_var, "") or cfg_get(cfg, key, fallback)
