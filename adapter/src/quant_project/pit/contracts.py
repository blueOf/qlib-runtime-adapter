"""Shared data-mode and dependency contracts."""
from __future__ import annotations

from collections.abc import Mapping

DATA_MODES = frozenset({"latest_snapshot", "snapshot_compatible", "point_in_time"})
DATASETS = ("market_data", "fundamentals", "industry")


class DataModeError(ValueError):
    pass


def normalize_data_mode(value, *, snapshot_compatible: bool | None = None) -> str:
    if value is None:
        value = "snapshot_compatible" if snapshot_compatible else "point_in_time"
    mode = str(value)
    if mode not in DATA_MODES:
        raise DataModeError(f"unsupported data_mode: {mode!r}")
    return mode


def normalize_dependencies(value=None, *, defaults=None) -> dict[str, bool]:
    result = {dataset: False for dataset in DATASETS}
    if defaults:
        for dataset, enabled in normalize_dependencies(defaults).items():
            result[dataset] = enabled
    if value is None:
        return result
    if not isinstance(value, Mapping):
        raise DataModeError("dependencies must be an object")
    unknown = set(value) - set(DATASETS)
    if unknown:
        raise DataModeError(f"unknown data dependencies: {', '.join(sorted(map(str, unknown)))}")
    for dataset, enabled in value.items():
        if type(enabled) is not bool:
            raise DataModeError(f"dependencies.{dataset} must be a boolean")
        result[dataset] = enabled
    return result


def merge_dependencies(*values) -> dict[str, bool]:
    result = {dataset: False for dataset in DATASETS}
    for value in values:
        normalized = normalize_dependencies(value)
        for dataset, enabled in normalized.items():
            result[dataset] = result[dataset] or enabled
    return result


def validate_data_request(*, data_mode, dependencies, as_of=None, as_of_policy=None) -> dict:
    mode = normalize_data_mode(data_mode)
    deps = normalize_dependencies(dependencies)
    if not any(deps.values()):
        raise DataModeError("at least one data dependency must be declared")
    if mode == "point_in_time":
        if not as_of:
            raise DataModeError("point_in_time research requires an explicit as_of")
        if not as_of_policy:
            raise DataModeError("point_in_time research requires an explicit as_of_policy")
    return {"data_mode": mode, "dependencies": deps,
            "point_in_time": mode == "point_in_time"}
