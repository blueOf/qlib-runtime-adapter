"""Controlled optional dependencies and reproducible implementation identity."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path

from .contracts import DependencyUnavailableError, sha256_json

LOCK = Path(__file__).resolve().parents[3] / "requirements-model.lock"
PACKAGES = {"numpy": "2.2.6", "scipy": "1.18.1", "scikit-learn": "1.9.1",
            "joblib": "1.6.0", "threadpoolctl": "3.6.0"}


def runtime_identity(*, optional_models=False, strict=True):
    versions = {}
    if optional_models:
        for name, expected in PACKAGES.items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as error:
                raise DependencyUnavailableError(f"model runtime package missing: {name}") from error
            versions[name] = actual
            if strict and actual != expected:
                raise DependencyUnavailableError(f"model runtime version drift: {name} expected {expected}, got {actual}")
        if strict and platform.python_version() != "3.12.14":
            raise DependencyUnavailableError("model runtime requires pinned Python 3.12.14")
    sources = {}
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"python": platform.python_version(), "platform": platform.system(),
            "machine": platform.machine(), "packages": versions,
            "environment_lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
            "implementation_sha256": sha256_json(sources)}


def adapter_runtime(adapter):
    optional = adapter.capabilities().get("library") == "scikit-learn"
    if adapter.adapter_id == "composite-graph.v1":
        # The graph may include learned child nodes. Record the controlled
        # environment whenever it is available; no-fit graphs stay usable in
        # a standard-library-only process.
        try:
            return runtime_identity(optional_models=True)
        except DependencyUnavailableError:
            pass
    return runtime_identity(optional_models=optional)
