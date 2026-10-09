"""Optional sklearn-backed adapters.

The import is intentionally lazy.  Importing ``quant_project.modeling`` must
continue to work when the optional model environment is not installed.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import AdapterBase
from ..runtime import runtime_identity, PACKAGES
from ..contracts import (ConfigError, DataContractError, DependencyUnavailableError,
                         FitContext, FitError, ModelInputBatch, ModelSpec, PredictContext,
                         PredictError, finite_score)


def _sklearn_modules():
    try:
        import joblib
        import numpy as np
    except ImportError as error:
        raise DependencyUnavailableError(
            "sklearn adapters require the controlled model environment (numpy/joblib)") from error
    return joblib, np


class SklearnAdapter(AdapterBase):
    estimator_kind = "base"
    adapter_version = "1.0.0"

    def __init__(self, adapter_id: str):
        self.adapter_id = adapter_id

    def capabilities(self) -> Mapping[str, Any]:
        tasks = (("regression",) if self.estimator_kind == "ridge" else
                 ("binary_classification", "multiclass_classification") if self.estimator_kind == "logistic" else
                 ("regression", "binary_classification", "multiclass_classification"))
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "task_types": tasks,
            "fit_policies": ("train_per_fold", "frozen_artifact"),
            "supports_missing": False,
            "supports_multi_output": False,
            "deterministic": True,
            "library": "scikit-learn",
            "library_version": PACKAGES["scikit-learn"],
            "supports_probability": self.estimator_kind != "ridge",
            "supports_sample_weight": False,
            "supports_categorical": False,
            "input_dtype": "float64",
            "serialization": "trusted_joblib",
        }

    def availability(self) -> tuple[bool, str | None]:
        try:
            runtime_identity(optional_models=True)
            _sklearn_modules()
            from sklearn import __version__ as _sklearn_version  # noqa: F401
        except (ImportError, DependencyUnavailableError) as error:
            return False, str(error)
        return True, None

    def validate(self, model_spec: ModelSpec, feature_ids: Sequence[str], target_field: str | None) -> None:
        super().validate(model_spec, feature_ids, target_field)
        runtime_identity(optional_models=True)
        if model_spec.task_type not in self.capabilities()["task_types"]:
            raise ConfigError(f"unsupported sklearn task type: {model_spec.task_type}")
        if model_spec.fit_policy == "train_per_fold" and not (target_field or model_spec.target_field):
            raise ConfigError(f"{self.adapter_id} requires target_field")
        if model_spec.params.get("score_sign", 1) not in (-1, 1):
            raise ConfigError("score_sign must explicitly preserve or reverse score direction")
        if model_spec.task_type == "multiclass_classification" and "score_class" not in model_spec.params:
            raise ConfigError("multiclass LONG scoring requires an explicit score_class")

    def _feature_matrix(self, batch: ModelInputBatch, feature_ids: Sequence[str]):
        _, np = _sklearn_modules()
        values = []
        for row in batch.rows:
            current = []
            for field in feature_ids:
                raw = row.get(field)
                if raw is None:
                    raise DataContractError(f"sklearn adapter cannot consume missing feature: {field}")
                value = float(raw)
                if not math.isfinite(value):
                    raise DataContractError(f"sklearn adapter cannot consume non-finite feature: {field}")
                current.append(value)
            values.append(current)
        return np.asarray(values, dtype="float64")

    def _target(self, batch: ModelInputBatch, target_field: str):
        _, np = _sklearn_modules()
        values = []
        for index, row in enumerate(batch.rows):
            if target_field not in row or row[target_field] is None:
                raise DataContractError(f"training row {index} missing target {target_field}")
            values.append(row[target_field])
        return np.asarray(values)

    def _build_estimator(self, model_spec: Mapping[str, Any], fit_context: FitContext):
        try:
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
            from sklearn.linear_model import LogisticRegression, Ridge
        except ImportError as error:
            raise DependencyUnavailableError("scikit-learn is not installed") from error
        params = dict(model_spec.get("params") or {})
        params.pop("score_class", None)
        params.pop("score_sign", None)
        task_type = str(model_spec.get("task_type", "regression"))
        seed = params.pop("random_state", fit_context.random_seed)
        if self.estimator_kind == "ridge":
            return Ridge(alpha=float(params.pop("alpha", 1.0)), **params)
        if self.estimator_kind == "logistic":
            params.setdefault("max_iter", 500)
            if seed is not None:
                params.setdefault("random_state", int(seed))
            return LogisticRegression(**params)
        if self.estimator_kind == "hist_gradient_boosting":
            cls = HistGradientBoostingClassifier if task_type != "regression" else HistGradientBoostingRegressor
            if seed is not None:
                params.setdefault("random_state", int(seed))
            return cls(**params)
        if self.estimator_kind == "extra_trees":
            cls = ExtraTreesClassifier if task_type != "regression" else ExtraTreesRegressor
            params.setdefault("n_jobs", 1)
            if seed is not None:
                params.setdefault("random_state", int(seed))
            return cls(**params)
        raise ConfigError(f"unknown sklearn estimator kind: {self.estimator_kind}")

    def fit(self, train_batch: ModelInputBatch, fit_context: FitContext) -> Mapping[str, Any]:
        spec = fit_context.metadata.get("model_spec") or {}
        feature_ids = tuple(spec.get("input_factor_ids") or train_batch.feature_ids)
        target_field = fit_context.target_field or spec.get("target_field")
        if not target_field:
            raise ConfigError(f"{self.adapter_id} fit requires target_field")
        estimator = self._build_estimator(spec, fit_context)
        x = self._feature_matrix(train_batch, feature_ids)
        y = self._target(train_batch, str(target_field))
        try:
            estimator.fit(x, y)
        except Exception as error:  # sklearn exposes many estimator-specific exceptions.
            raise FitError(f"{self.adapter_id} fit failed: {error}") from error
        return {"estimator": estimator, "feature_ids": list(feature_ids),
                "target_field": str(target_field), "task_type": str(spec.get("task_type", "regression")),
                "params": dict(spec.get("params") or {})}

    def predict(self, fitted_handle: Mapping[str, Any], inference_batch: ModelInputBatch,
                predict_context: PredictContext) -> Sequence[Mapping[str, Any]]:
        _, np = _sklearn_modules()
        estimator = fitted_handle.get("estimator")
        if estimator is None:
            raise ConfigError(f"{self.adapter_id} artifact has no estimator")
        feature_ids = tuple(fitted_handle["feature_ids"])
        x = self._feature_matrix(inference_batch, feature_ids)
        try:
            if fitted_handle["task_type"] != "regression" and hasattr(estimator, "predict_proba"):
                probabilities = estimator.predict_proba(x)
                classes = list(estimator.classes_)
                score_class = fitted_handle.get("params", {}).get("score_class", 1)
                if score_class not in classes:
                    raise PredictError("classification score_class is absent from fitted classes")
                values = probabilities[:, classes.index(score_class)]
            elif hasattr(estimator, "decision_function") and fitted_handle["task_type"] != "regression":
                values = estimator.decision_function(x)
            else:
                values = estimator.predict(x)
        except Exception as error:
            raise PredictError(f"{self.adapter_id} predict failed: {error}") from error
        output = []
        for row, value in zip(inference_batch.rows, values):
            key = f"{row['timestamp']}|{row['symbol']}"
            diagnostics = {"raw_prediction": float(value), "score_sign": fitted_handle.get("params", {}).get("score_sign", 1.0)}
            if fitted_handle["task_type"] != "regression":
                diagnostics["score_class"] = fitted_handle.get("params", {}).get("score_class", 1)
            output.append({"timestamp": str(row["timestamp"]), "symbol": str(row["symbol"]),
                           "score": finite_score(value * float(fitted_handle.get("params", {}).get("score_sign", 1.0)), key=key),
                           "diagnostics": diagnostics})
        return output

    def dump(self, fitted_handle: Mapping[str, Any], artifact_directory: Path) -> Mapping[str, Any]:
        joblib, _ = _sklearn_modules()
        artifact_directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted_handle["estimator"], artifact_directory / "estimator.joblib")
        self.write_json(artifact_directory, "adapter_state.json", {
            "feature_ids": list(fitted_handle["feature_ids"]),
            "target_field": fitted_handle["target_field"],
            "task_type": fitted_handle["task_type"],
            "params": fitted_handle.get("params", {}),
        })
        return {"kind": "sklearn_joblib", "state_file": "adapter_state.json",
                "payload_file": "estimator.joblib"}

    def load(self, artifact_manifest: Mapping[str, Any], artifact_directory: Path) -> Mapping[str, Any]:
        joblib, _ = _sklearn_modules()
        state = self.read_json(artifact_directory, "adapter_state.json")
        payload = artifact_directory / "estimator.joblib"
        if not payload.is_file():
            raise ConfigError(f"sklearn artifact payload is missing: {payload}")
        try:
            estimator = joblib.load(payload)
        except Exception as error:
            raise ConfigError(f"sklearn artifact cannot be loaded: {error}") from error
        return state | {"estimator": estimator}


class RidgeAdapter(SklearnAdapter):
    def __init__(self):
        super().__init__("sklearn-ridge.v1")
        self.estimator_kind = "ridge"


class LogisticAdapter(SklearnAdapter):
    def __init__(self):
        super().__init__("sklearn-logistic.v1")
        self.estimator_kind = "logistic"
