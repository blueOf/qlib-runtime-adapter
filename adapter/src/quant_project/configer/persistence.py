"""Persist the fully resolved configuration as research evidence."""
from __future__ import annotations

import os
import re
from io import StringIO
from pathlib import Path
from uuid import uuid4

from ruamel.yaml import YAML

from ..paths import RUNS_ROOT
from .models import ResolvedExperiment

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def persist_resolved_config(resolved: ResolvedExperiment, *, run_id: str | None = None, reports_root=None) -> Path:
    run_id = run_id or uuid4().hex
    if not SAFE_RUN_ID.fullmatch(str(run_id)):
        raise ValueError(f"research run_id is invalid: {run_id!r}")
    root = Path(reports_root or RUNS_ROOT / "experiments").resolve()
    destination = root / resolved.experiment.id / run_id / "resolved_config.yaml"
    destination.parent.mkdir(parents=True, exist_ok=False)
    payload = resolved.to_dict() | {"resolved_config_sha256": resolved.sha256}
    stream = StringIO()
    YAML().dump(payload, stream)
    temporary = destination.with_suffix(".yaml.tmp")
    temporary.write_text(stream.getvalue(), encoding="utf-8")
    os.replace(temporary, destination)
    return destination
