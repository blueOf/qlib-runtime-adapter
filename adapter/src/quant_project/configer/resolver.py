"""Resolve one Experiment into immutable domain objects without executing it."""
from __future__ import annotations

from .loader import ConfigLoader, resolve_research_component
from .models import (ExperimentSpec, FactorSpec, ResolvedExperiment, SignalSpec, StrategySpec,
                     canonical_hash)
from .validator import validate_resolved


def resolve_experiment(identifier, root, *, context) -> ResolvedExperiment:
    loader = ConfigLoader(root)
    experiment_path, experiment_data = loader.experiment(identifier)
    if not isinstance(context, dict):
        raise ValueError("research run context must be supplied as an object")
    experiment_data["context"] = context
    if experiment_data.get("schema_version") == "stock-experiment-v2":
        from .four_layer import ExperimentSpecV2, ModelSignalSpec, ResolvedExperimentV2
        from .research_models import FactorSetSpec
        experiment = ExperimentSpecV2.from_dict(experiment_data)
        factor_path, factor_data = loader.factor_set(experiment.factor_set)
        model_path, model_data = loader.model(experiment.model)
        factor_set = FactorSetSpec.from_mapping(resolve_research_component(factor_data, loader, "factor_set"))
        model = factor_set.bind_model(resolve_research_component(model_data, loader, "model"))
        signal_path, signal_data = loader.signal(experiment.signal, reject_duplicates=True)
        strategy_path, strategy_data = loader.strategy(experiment.strategy, reject_duplicates=True)
        hashes = {str(path): canonical_hash(data) for path, data in (
            (factor_path, factor_data), (model_path, model_data), (signal_path, signal_data),
            (strategy_path, strategy_data))}
        hashes[f"{experiment_path}#{identifier}"] = canonical_hash(
            {key: value for key, value in experiment_data.items() if key != "context"})
        return validate_resolved(ResolvedExperimentV2(experiment, factor_set, model,
            ModelSignalSpec.from_dict(signal_data), StrategySpec.from_dict(strategy_data), experiment.context, hashes))
    experiment = ExperimentSpec.from_dict(experiment_data)
    registry_path, registry = loader.factor_registry()
    factor_data = dict(registry.get(experiment.factor) or {})
    factor_data.setdefault("id", experiment.factor)
    factor = FactorSpec.from_dict(factor_data)
    signal_path, signal_data = loader.signal(experiment.signal)
    strategy_path, strategy_data = loader.strategy(experiment.strategy)
    signal, strategy = SignalSpec.from_dict(signal_data), StrategySpec.from_dict(strategy_data)
    hashes = {
        f"{experiment_path}#{identifier}": canonical_hash(
            {key: value for key, value in experiment_data.items() if key != "context"}),
        str(registry_path): canonical_hash(registry),
        str(signal_path): canonical_hash(signal_data),
        str(strategy_path): canonical_hash(strategy_data),
    }
    return validate_resolved(ResolvedExperiment(experiment=experiment, factor=factor, signal=signal,
                                                strategy=strategy, context=experiment.context,
                                                source_hashes=hashes))
