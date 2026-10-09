"""Repository-relative paths shared by every command.

Code, configuration, and the formal market data live in ``quant_project``.
Tests and rehearsals can redirect the data pair with an explicit data root; no
path depends on the current directory or a machine-specific absolute location.
"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
CONFIG_ROOT = PROJECT_ROOT / "configs"
RUNS_ROOT = PROJECT_ROOT / "runs"
FORMAL_DATA_ROOT = PROJECT_ROOT / "data"

ENV_DATA_ROOT = "QUANT_PROJECT_DATA_ROOT"
LEGACY_ENV_DATA_ROOT = "STOCK_QLIB_DATA_ROOT"
ENV_CONFIG_ROOT = "QUANT_PROJECT_CONFIG_ROOT"
ENV_RUNS_ROOT = "QUANT_PROJECT_RUNS_ROOT"


def _environment_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
    return None


def data_root(explicit=None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return _environment_path(ENV_DATA_ROOT, LEGACY_ENV_DATA_ROOT) or FORMAL_DATA_ROOT.resolve()


def config_root(explicit=None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return _environment_path(ENV_CONFIG_ROOT) or CONFIG_ROOT.resolve()


def runs_root(explicit=None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return _environment_path(ENV_RUNS_ROOT) or RUNS_ROOT.resolve()


def resolve_workspace_path(value) -> Path:
    """Resolve new paths and the two legacy path roles without using cwd."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    relative = candidate.as_posix()
    redirects = {
        "quant_project/data": data_root(),
        # Keep old workflow and report paths resolving to the canonical store.
        "qlib_stack/data": data_root(),
        "qlib_stack/config": config_root(),
        "qlib_stack/reports": runs_root(),
        "qlib_stack/runtime": runs_root(),
        "quant_project/configs": config_root(),
        "quant_project/runs": runs_root(),
    }
    for prefix, target_root in redirects.items():
        if relative == prefix or relative.startswith(prefix + "/"):
            suffix = relative[len(prefix):].lstrip("/")
            return (target_root / suffix).resolve()
    return (WORKSPACE_ROOT / candidate).resolve()
