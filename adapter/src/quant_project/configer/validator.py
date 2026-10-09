"""Cross-layer semantic checks for a resolved research configuration."""
from __future__ import annotations

from datetime import date

from .models import ConfigError, ResolvedExperiment
from ..pit.time_policy import PITTimeError, normalize_as_of


def validate_resolved(resolved: ResolvedExperiment) -> ResolvedExperiment:
    from .four_layer import ResolvedExperimentV2
    if isinstance(resolved, ResolvedExperimentV2):
        if resolved.signal.source_model != resolved.model.id:
            raise ConfigError("V2 SignalSpec must reference the Experiment model")
    elif resolved.signal.source_factor != resolved.factor.id:
        raise ConfigError("SignalSpec must reference the Experiment factor")
    if not resolved.context.universe:
        raise ConfigError("BacktestContext.universe is required")
    if resolved.context.start and resolved.context.end:
        if date.fromisoformat(resolved.context.start) > date.fromisoformat(resolved.context.end):
            raise ConfigError("BacktestContext.start must not be after end")
    if not resolved.context.research_release and not resolved.context.snapshot_compatible:
        raise ConfigError("Context requires a research_release or explicit snapshot_compatible=true")
    if resolved.context.data_mode == "latest_snapshot":
        raise ConfigError("latest_snapshot is reserved for current scans; research requires snapshot_compatible or point_in_time")
    if resolved.context.data_mode == "point_in_time":
        if not resolved.context.research_release:
            raise ConfigError("point_in_time research requires a pinned research_release")
        if not resolved.context.as_of:
            raise ConfigError("point_in_time research requires an explicit as_of")
        if not resolved.context.as_of_policy:
            raise ConfigError("point_in_time research requires an explicit as_of_policy")
        try:
            normalize_as_of(resolved.context.as_of)
        except PITTimeError as error:
            raise ConfigError(str(error)) from error
    if resolved.experiment.dependencies is not None:
        required = resolved.declared_dependencies
        missing = [name for name, value in required.items()
                   if value and not resolved.experiment.dependencies.get(name, False)]
        if missing:
            raise ConfigError(f"Experiment dependencies understate component requirements: {', '.join(missing)}")
    selection = resolved.signal.selection
    if not selection.get("method"):
        raise ConfigError("SignalSpec.selection.method is required")
    eligibility = resolved.signal.eligibility
    industry_limit = eligibility.get("known_industry_max_positions")
    if industry_limit is not None:
        if type(industry_limit) is not int or industry_limit < 1:
            raise ConfigError("SignalSpec.eligibility.known_industry_max_positions must be a positive integer")
        if selection["method"] != "top_k":
            raise ConfigError("SignalSpec industry limit currently requires top_k selection")
        if not resolved.signal.dependencies["industry"]:
            raise ConfigError("SignalSpec industry limit requires an industry dependency")
        if type(eligibility.get("missing_industry_is_independent", True)) is not bool:
            raise ConfigError("SignalSpec.eligibility.missing_industry_is_independent must be boolean")
    if not resolved.strategy.portfolio.get("max_positions"):
        raise ConfigError("StrategySpec.portfolio.max_positions is required")
    if not resolved.strategy.entry.get("timing"):
        raise ConfigError("StrategySpec.entry.timing is required")
    return resolved
