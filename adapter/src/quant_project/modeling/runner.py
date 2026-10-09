"""The single execution boundary for model adapters.

The runner owns policy and lineage.  Adapters only implement mechanical fit,
predict and serialization operations on already-authorized batches; they do
not choose folds, discover models, or decide how a score is consumed.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import ArtifactStore
from .contracts import (
    ArtifactError,
    CapabilityError,
    ConfigError,
    DataContractError,
    DependencyUnavailableError,
    FitContext,
    FitError,
    FittedModel,
    LeakageError,
    ModelInputBatch,
    ModelScore,
    ModelSpec,
    PredictContext,
    PredictError,
    ResourceBudgetError,
    finite_score,
    sha256_json,
)
from .registry import AdapterRegistry, build_default_registry


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["timestamp"]), str(row["symbol"])


def _window_overlap(left: tuple[str, str] | None, right: tuple[str, str] | None) -> bool:
    if not left or not right:
        return False
    if len(left) != 2 or len(right) != 2:
        raise DataContractError("time windows must be two-element sequences")
    left_start, left_end = str(left[0]), str(left[1])
    right_start, right_end = str(right[0]), str(right[1])
    if left_start > left_end or right_start > right_end:
        raise DataContractError("time window start must not be after its end")
    return max(left_start, right_start) <= min(left_end, right_end)


class ModelRunner:
    """Resolve, validate, execute and standardize registered model adapters."""

    def __init__(self, registry: AdapterRegistry | None = None, artifact_store: ArtifactStore | str | Path | None = None):
        self.registry = registry or build_default_registry()
        if artifact_store is None or isinstance(artifact_store, ArtifactStore):
            self.artifact_store = artifact_store
        else:
            self.artifact_store = ArtifactStore(artifact_store)

    @staticmethod
    def _with_target(context: FitContext, spec: ModelSpec) -> FitContext:
        if context.target_field and spec.target_field and context.target_field != spec.target_field:
            raise ConfigError("FitContext target does not match ModelSpec")
        if context.target_field is not None or spec.target_field is None:
            return context
        return replace(context, target_field=spec.target_field)

    @staticmethod
    def _feature_ids(spec: ModelSpec, batch: ModelInputBatch | None, context: FitContext) -> tuple[str, ...]:
        if batch is not None:
            batch.validate()
            actual = tuple(batch.feature_ids)
        else:
            actual = tuple(context.feature_order)
        expected = tuple(spec.input_factor_ids)
        if expected:
            if actual and actual != expected:
                raise DataContractError(
                    f"feature order does not match ModelSpec: expected {expected}, got {actual}")
            return expected
        if actual:
            return actual
        raise DataContractError("model input feature order is required")

    @staticmethod
    def _check_resources(spec: ModelSpec, train_batch: ModelInputBatch | None, feature_ids: Sequence[str]) -> None:
        resources = dict(spec.resources)
        for name in ("max_train_rows", "max_features", "max_threads"):
            if name in resources:
                try:
                    value = int(resources[name])
                except (TypeError, ValueError) as error:
                    raise ResourceBudgetError(f"resource {name} must be an integer") from error
                if value <= 0:
                    raise ResourceBudgetError(f"resource {name} must be positive")
        if train_batch is not None and "max_train_rows" in resources \
                and len(train_batch.rows) > int(resources["max_train_rows"]):
            raise ResourceBudgetError("training rows exceed ModelSpec max_train_rows")
        if "max_features" in resources and len(feature_ids) > int(resources["max_features"]):
            raise ResourceBudgetError("features exceed ModelSpec max_features")
        params = dict(spec.params)
        n_jobs = params.get("n_jobs")
        if n_jobs is not None:
            try:
                n_jobs = int(n_jobs)
            except (TypeError, ValueError) as error:
                raise ResourceBudgetError("params.n_jobs must be an integer") from error
            if n_jobs == -1:
                raise ResourceBudgetError("unbounded params.n_jobs=-1 is not permitted")
            if n_jobs <= 0:
                raise ResourceBudgetError("params.n_jobs must be positive")
            if "max_threads" in resources and n_jobs > int(resources["max_threads"]):
                raise ResourceBudgetError("params.n_jobs exceeds max_threads")

    @staticmethod
    def _check_window_contract(context: FitContext) -> None:
        if _window_overlap(context.train_window, context.validation_window):
            raise LeakageError("train and validation windows overlap")
        if _window_overlap(context.train_window, context.inference_window):
            raise LeakageError("train and inference windows overlap")
        if _window_overlap(context.validation_window, context.inference_window):
            raise LeakageError("validation and inference windows overlap")

    @staticmethod
    def _prepare_fit_context(context: FitContext, spec: ModelSpec,
                             train_batch: ModelInputBatch | None) -> FitContext:
        context = ModelRunner._with_target(context, spec)
        ModelRunner._check_window_contract(context)
        metadata = dict(context.metadata)
        if train_batch is not None:
            metadata["_fit_keys"] = tuple(_key(row) for row in train_batch.rows)
        return replace(context, metadata=metadata)

    @staticmethod
    def _validate_inference_batch(spec: ModelSpec, batch: ModelInputBatch,
                                  fit_context: FitContext | None,
                                  context: PredictContext) -> None:
        batch.validate()
        if spec.input_factor_ids and tuple(batch.feature_ids) != tuple(spec.input_factor_ids):
            raise DataContractError(
                f"feature order does not match ModelSpec: expected {spec.input_factor_ids}, "
                f"got {batch.feature_ids}")
        if spec.target_field:
            leaked = [spec.target_field for row in batch.rows if spec.target_field in row]
            if leaked:
                raise LeakageError("predict input contains the training target field")
        declared_targets = context.metadata.get("target_fields", ())
        if isinstance(declared_targets, str):
            declared_targets = (declared_targets,)
        if declared_targets:
            leaked_fields = sorted({str(field) for field in declared_targets
                                    if any(field in row for row in batch.rows)})
            if leaked_fields:
                raise LeakageError(f"predict input contains declared target fields: {leaked_fields}")
        if fit_context is None:
            return
        if fit_context.target_field and any(fit_context.target_field in row for row in batch.rows):
            raise LeakageError("predict input contains the resolved target field")
        if fit_context.train_window and batch.rows:
            if min(str(row["timestamp"])[:10] for row in batch.rows) <= str(fit_context.train_window[1])[:10]:
                raise LeakageError("inference must occur after the training window")
        if spec.fit_policy == "train_per_fold":
            if not fit_context.fold_id or not context.fold_id:
                raise LeakageError("train_per_fold prediction requires a fold_id")
            if fit_context.fold_id != context.fold_id:
                raise LeakageError("fit and predict fold_id do not match")
            fit_keys = set(fit_context.metadata.get("_fit_keys", ()))
            inference_keys = {_key(row) for row in batch.rows}
            if fit_keys.intersection(inference_keys):
                raise LeakageError("training and inference rows overlap")
            if _window_overlap(fit_context.train_window, context.inference_window):
                raise LeakageError("training and inference windows overlap")
            if _window_overlap(fit_context.validation_window, context.inference_window):
                raise LeakageError("validation and inference windows overlap")

    def _virtual_manifest(self, spec: ModelSpec, adapter, context: FitContext) -> dict[str, Any]:
        """Return an explicit ephemeral manifest when no store was supplied.

        A caller that needs cross-process reproducibility must provide an
        ArtifactStore.  Ephemeral manifests still make a no-fit or in-memory
        canary's lineage explicit instead of using a magic placeholder.
        """
        base = {
            "schema": "quant-project-model-artifact-manifest-v1",
            "adapter_id": adapter.adapter_id,
            "adapter_version": adapter.adapter_version,
            "model_spec_sha256": spec.config_sha256,
            "model_identity_sha256": spec.model_identity_sha256,
            "fit_policy": spec.fit_policy,
            "files": {},
            "payload_sha256": None,
            "metadata": {
                "persistence": "ephemeral",
                "run_id": context.run_id,
                "input_data_sha256": context.input_data_sha256,
            },
        }
        artifact_id = sha256_json(base)
        manifest = base | {"artifact_id": artifact_id, "sealed": True, "status": "ephemeral"}
        manifest["manifest_sha256"] = sha256_json(manifest)
        return manifest

    def fit(self, spec: ModelSpec, train_batch: ModelInputBatch | None,
            fit_context: FitContext) -> FittedModel:
        if spec.execution_status == "reference_only":
            raise CapabilityError("reference_only ModelSpec cannot be executed")
        if not isinstance(fit_context, FitContext):
            raise ConfigError("fit_context must be a FitContext")
        context = self._prepare_fit_context(fit_context, spec, train_batch)
        feature_ids = self._feature_ids(spec, train_batch, context)
        self._check_resources(spec, train_batch, feature_ids)
        adapter = self.registry.resolve(spec.adapter)
        target_field = context.target_field or spec.target_field
        adapter.validate(spec, feature_ids, target_field)
        if spec.fit_policy == "train_per_fold":
            if not context.fold_id:
                raise ConfigError("train_per_fold requires fit_context.fold_id")
            if train_batch is None:
                raise DataContractError("train_per_fold requires a training batch")
            if not target_field:
                raise ConfigError("train_per_fold requires target_field")
            for index, row in enumerate(train_batch.rows):
                if target_field not in row or row[target_field] is None:
                    raise DataContractError(f"training row {index} missing target {target_field}")
        if spec.fit_policy == "frozen_artifact":
            if self.artifact_store is None:
                raise ArtifactError("frozen_artifact requires an ArtifactStore")
            if not spec.artifact:
                raise ArtifactError("frozen_artifact ModelSpec has no artifact reference")
            handle, manifest = self.artifact_store.load(
                artifact_ref=dict(spec.artifact), adapter=adapter, model_spec=spec)
            historical = manifest.get("metadata", {}).get("fit_context")
            if isinstance(historical, dict):
                context = FitContext(**historical)
            return FittedModel(spec, adapter.adapter_id, adapter.adapter_version, handle,
                               manifest, context)
        if train_batch is None:
            if spec.fit_policy != "no_fit":
                raise DataContractError("fit requires a batch for non-frozen models")
            # No-fit adapters must be usable for inference-only runs without a
            # label-bearing training slice.  Their fit method is still called
            # so all adapters share one lifecycle, but it receives an empty,
            # schema-checked batch and cannot observe labels.
            train_batch = ModelInputBatch.from_rows([], feature_ids)
        try:
            handle = adapter.fit(train_batch, context.with_runtime(model_spec=spec, registry=self.registry))
        except (DependencyUnavailableError, CapabilityError, ConfigError, DataContractError,
                LeakageError, ResourceBudgetError, FitError):
            raise
        except Exception as error:
            raise FitError(f"{adapter.adapter_id} fit failed: {error}") from error
        if self.artifact_store is not None:
            try:
                manifest = self.artifact_store.publish(
                    model_spec=spec,
                    adapter=adapter,
                    handle=handle,
                    metadata={
                        "run_id": context.run_id,
                        "fold_id": context.fold_id,
                        "input_data_sha256": context.input_data_sha256,
                        "feature_ids": list(feature_ids),
                        "target_field": target_field,
                        "fit_context": {
                            "run_id": context.run_id, "fold_id": context.fold_id,
                            "research_release": context.research_release, "data_mode": context.data_mode,
                            "train_window": context.train_window, "validation_window": context.validation_window,
                            "inference_window": context.inference_window, "purge_sessions": context.purge_sessions,
                            "embargo_sessions": context.embargo_sessions, "random_seed": context.random_seed,
                            "feature_order": list(feature_ids), "target_field": target_field,
                            "input_data_sha256": context.input_data_sha256,
                        },
                        "research_context": {key: value for key, value in context.metadata.items()
                                             if not key.startswith("_")},
                    },
                )
            except (ArtifactError, DependencyUnavailableError):
                raise
            except Exception as error:
                raise ArtifactError(f"could not publish model artifact: {error}") from error
        else:
            manifest = self._virtual_manifest(spec, adapter, context)
        return FittedModel(spec, adapter.adapter_id, adapter.adapter_version, handle,
                           manifest, context)

    def predict(self, fitted: FittedModel, inference_batch: ModelInputBatch,
                predict_context: PredictContext) -> tuple[ModelScore, ...]:
        if not isinstance(fitted, FittedModel):
            raise ConfigError("predict requires a FittedModel")
        if fitted.spec.execution_status == "reference_only":
            raise CapabilityError("reference_only ModelSpec cannot be executed")
        if not isinstance(predict_context, PredictContext):
            raise ConfigError("predict_context must be a PredictContext")
        adapter = self.registry.resolve(fitted.spec.adapter)
        if adapter.adapter_id != fitted.adapter_id or adapter.adapter_version != fitted.adapter_version:
            raise ArtifactError("fitted model adapter identity does not match registry")
        self._validate_inference_batch(fitted.spec, inference_batch, fitted.fit_context, predict_context)
        context = predict_context.with_runtime(model_spec=fitted.spec, registry=self.registry)
        try:
            predictions = adapter.predict(fitted.handle, inference_batch, context)
        except (DependencyUnavailableError, CapabilityError, ConfigError, DataContractError,
                LeakageError, ResourceBudgetError, PredictError):
            raise
        except Exception as error:
            raise PredictError(f"{adapter.adapter_id} predict failed: {error}") from error
        if isinstance(predictions, Mapping) or isinstance(predictions, (str, bytes)):
            raise PredictError("adapter prediction output must be a sequence of rows")
        try:
            prediction_rows = list(predictions)
        except TypeError as error:
            raise PredictError("adapter prediction output is not iterable") from error
        expected = [_key(row) for row in inference_batch.rows]
        expected_set = set(expected)
        by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
        for index, row in enumerate(prediction_rows):
            if not isinstance(row, Mapping):
                raise PredictError(f"prediction row {index} is not a mapping")
            if row.get("timestamp") is None or row.get("symbol") is None:
                raise PredictError(f"prediction row {index} requires timestamp and symbol")
            key = _key(row)
            if key in by_key:
                raise PredictError(f"duplicate prediction key: {key[0]}|{key[1]}")
            if key not in expected_set:
                raise PredictError(f"prediction contains an unexpected key: {key[0]}|{key[1]}")
            if "score" not in row:
                raise PredictError(f"prediction row {index} has no score")
            by_key[key] = row
        if len(prediction_rows) != len(expected) or set(by_key) != expected_set:
            missing = sorted(expected_set - set(by_key))
            raise PredictError(f"prediction key coverage mismatch; missing={missing}")
        manifest = dict(fitted.artifact)
        manifest_sha = manifest.get("manifest_sha256")
        if not manifest_sha:
            manifest_sha = sha256_json(manifest)
        payload_sha = manifest.get("payload_sha256")
        scores: list[ModelScore] = []
        for source_row in inference_batch.rows:
            key = _key(source_row)
            prediction = by_key[key]
            score = finite_score(prediction["score"], key=f"{key[0]}|{key[1]}")
            model_score_id = sha256_json({
                "research_run_id": predict_context.run_id,
                "fold_id": predict_context.fold_id,
                "timestamp": key[0],
                "symbol": key[1],
                "model_config_sha256": fitted.spec.config_sha256,
            })
            diagnostics = {
                name: value for name, value in prediction.items()
                if name not in {"timestamp", "symbol", "score"}
            }
            nested = diagnostics.get("diagnostics")
            if isinstance(nested, Mapping):
                diagnostics = dict(nested) | {name: value for name, value in diagnostics.items()
                                               if name != "diagnostics"}
            source_vector_id = prediction.get("source_factor_vector_id")
            if source_vector_id is None:
                source_vector_id = source_row.get("source_factor_vector_id")
            scores.append(ModelScore(
                model_score_id=model_score_id,
                research_run_id=predict_context.run_id,
                timestamp=key[0],
                symbol=key[1],
                score=score,
                fold_id=predict_context.fold_id,
                source_model_id=fitted.spec.id,
                source_factor_set_id=fitted.spec.factor_set_id,
                model_config_sha256=fitted.spec.config_sha256,
                factor_set_config_sha256=fitted.spec.factor_set_config_sha256,
                artifact_manifest_sha256=str(manifest_sha),
                model_payload_sha256=(str(payload_sha) if payload_sha is not None else None),
                source_factor_vector_id=(str(source_vector_id) if source_vector_id is not None else None),
                diagnostics=diagnostics,
            ))
        return tuple(scores)

    def fit_predict(self, spec: ModelSpec, train_batch: ModelInputBatch | None,
                    fit_context: FitContext, inference_batch: ModelInputBatch,
                    predict_context: PredictContext) -> tuple[ModelScore, ...]:
        fitted = self.fit(spec, train_batch, fit_context)
        return self.predict(fitted, inference_batch, predict_context)
