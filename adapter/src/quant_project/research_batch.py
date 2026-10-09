"""Immutable batch orchestration and explicit report comparison for research runs."""
from __future__ import annotations

import re
from pathlib import Path

from .common import atomic_json, read_json, sha256
from .evaluation.research import run_experiment
from .paths import CONFIG_ROOT, RUNS_ROOT

SCHEMA = "quant-project-research-batch-result-v1"
MANIFEST_SCHEMA = "quant-project-research-batch-v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ResearchBatchError(RuntimeError):
    pass


def _safe(value, kind):
    value = str(value or "")
    if not SAFE_ID.fullmatch(value):
        raise ResearchBatchError(f"invalid {kind}: {value!r}")
    return value


def _input_payload(item, manifest_root):
    has_payload, has_path = isinstance(item.get("payload"), dict), item.get("input") is not None
    if has_payload == has_path:
        raise ResearchBatchError("each batch run must choose exactly one of payload or input")
    if has_payload:
        return item["payload"], {"kind": "inline"}
    path = Path(item["input"]).expanduser()
    if not path.is_absolute():
        path = manifest_root / path
    path = path.resolve()
    if not path.is_file():
        raise ResearchBatchError(f"batch input does not exist: {path}")
    return read_json(path), {"kind": "file", "path": str(path), "sha256": sha256(path)}


def _metric(payload, path):
    value = payload
    for name in str(path).split("."):
        if not isinstance(value, dict) or name not in value:
            raise ResearchBatchError(f"comparison metric does not exist: {path}")
        value = value[name]
    return value


def run_research_batch(manifest, *, config_root=None, reports_root=None) -> dict:
    manifest_path = Path(manifest).expanduser().resolve()
    payload = read_json(manifest_path)
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ResearchBatchError(f"batch manifest schema must be {MANIFEST_SCHEMA}")
    batch_id = _safe(payload.get("batch_id"), "batch_id")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ResearchBatchError("batch manifest requires at least one run")
    config_root = Path(config_root or CONFIG_ROOT).resolve()
    reports_root = Path(reports_root or RUNS_ROOT / "experiments").resolve()
    batch_root = reports_root / "batches" / batch_id
    if batch_root.exists():
        raise ResearchBatchError(f"batch result already exists and will not be overwritten: {batch_root}")

    results, by_name, names = [], {}, set()
    for item in runs:
        if not isinstance(item, dict):
            raise ResearchBatchError("each batch run must be an object")
        name = _safe(item.get("name"), "batch run name")
        if name in names:
            raise ResearchBatchError(f"duplicate batch run name: {name}")
        names.add(name)
        stage = str(item.get("stage", "full"))
        if stage not in {"factor", "model", "signal", "strategy", "full"}:
            raise ResearchBatchError(f"invalid batch run stage: {stage}")
        experiment_id = str(item.get("experiment_id", ""))
        run_id = _safe(item.get("run_id") or f"{batch_id}-{name}", "run_id")
        input_payload, source = _input_payload(item, manifest_path.parent)
        result = run_experiment(
            experiment_id, input_payload, config_root=config_root, stage=stage,
            reports_root=reports_root, run_id=run_id,
            execution_engine=item.get("execution_engine", "deterministic"))
        output = Path(result["output"])
        record = {
            "name": name, "experiment_id": experiment_id, "run_id": run_id,
            "stage": stage, "output": str(output), "input": source,
            "summary": str(output / "summary.json"),
            "summary_sha256": sha256(output / "summary.json"),
            "registry_key": result["registry"]["key"],
        }
        results.append(record)
        by_name[name] = {"record": record, "result": result}

    comparisons = []
    for item in payload.get("comparisons") or []:
        comparison_id = _safe(item.get("id"), "comparison id")
        left_name, right_name = str(item.get("left", "")), str(item.get("right", ""))
        if left_name not in by_name or right_name not in by_name:
            raise ResearchBatchError(f"comparison references an unknown run: {comparison_id}")
        report_key = str(item.get("report", "strategy_report"))
        metrics = item.get("metrics") or ["performance.total_return", "performance.max_drawdown"]
        if not isinstance(metrics, list) or not metrics:
            raise ResearchBatchError(f"comparison metrics must be a non-empty array: {comparison_id}")
        sides, documents = {}, {}
        for side, run_name in (("left", left_name), ("right", right_name)):
            entry = by_name[run_name]
            relative = entry["result"]["summary"]["reports"].get(report_key)
            if not relative:
                raise ResearchBatchError(f"run {run_name} has no report {report_key}")
            path = Path(entry["record"]["output"]) / relative
            documents[side] = read_json(path)
            sides[side] = {"run": run_name, "path": str(path), "sha256": sha256(path)}
        values = []
        for metric in metrics:
            left, right = _metric(documents["left"], metric), _metric(documents["right"], metric)
            delta = right - left if (isinstance(left, (int, float)) and not isinstance(left, bool)
                                     and isinstance(right, (int, float)) and not isinstance(right, bool)) else None
            values.append({"metric": str(metric), "left": left, "right": right,
                           "delta": delta, "equal": left == right})
        comparisons.append({"id": comparison_id, "report": report_key,
                            "left": sides["left"], "right": sides["right"],
                            "metrics": values})

    result = {"schema": SCHEMA, "batch_id": batch_id,
              "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
              "config_root": str(config_root), "reports_root": str(reports_root),
              "runs": results, "comparisons": comparisons, "status": "success"}
    batch_root.mkdir(parents=True, exist_ok=False)
    atomic_json(batch_root / "batch_result.json", result)
    return result | {"output": str(batch_root / "batch_result.json")}
