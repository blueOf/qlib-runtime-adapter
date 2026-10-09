"""Deterministic research services with explicit evidence and object boundaries.

They operate only on caller-supplied FactorScore/label/price inputs.  In
particular, this module neither fetches quotes nor writes intraday overlays.
"""
from __future__ import annotations

import json
import hashlib
import math
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path
from statistics import mean, median, pstdev
from uuid import uuid4

from ..common import atomic_json
from ..adapters.qlib import compile_qlib_runtime_config
from ..configer.compiler import compile_experiment
from ..configer.persistence import persist_resolved_config
from ..paths import FORMAL_DATA_ROOT, RUNS_ROOT
from ..research_registry import register_research_run
from ..configer.resolver import resolve_experiment
from ..configer.models import ResolvedExperiment
from ..price_limits import assess_limit_market_state, derive_price_limit_rows
from .contracts import AccountSnapshot, CandidateSignal, ModelCandidateSignal, ModelScore, FactorScore, Fill, Order, Position


def _context(resolved: ResolvedExperiment):
    return resolved.context.to_dict()


def _warning_context(resolved):
    metadata = resolved.context.metadata
    scope_warnings = []
    if metadata.get("decisionEligible") is False:
        scope_warnings.append({"code": "COMPATIBILITY_ONLY", "message": "historical compatibility evidence is not eligible for research selection or promotion"})
    if metadata.get("selection_bias_risk"):
        scope_warnings.append({"code": str(metadata["selection_bias_risk"]), "message": "historical selection-bias restriction is retained"})
    if resolved.context.data_mode != "snapshot_compatible":
        return scope_warnings
    dependencies = resolved.declared_dependencies
    warnings = [{"code": "SNAPSHOT_COMPATIBLE_ONLY",
                 "message": "results use a compatible snapshot and are not point-in-time validated"}]
    if dependencies["fundamentals"]:
        warnings.append({"code": "FUNDAMENTALS_SNAPSHOT_BIAS",
                         "message": "fundamentals use a compatible snapshot without historical announcement times"})
    if dependencies["industry"]:
        warnings.append({"code": "INDUSTRY_SNAPSHOT_BIAS",
                         "message": "industry inputs use a compatible snapshot without historical effective times"})
    return warnings + scope_warnings


def _research_release_identity(resolved, dependencies=None):
    value = resolved.context.research_release
    if not value:
        if resolved.context.data_mode == "point_in_time":
            raise ValueError("point_in_time research requires a pinned Research Release")
        return None
    from ..research_release import verify_research_release

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = FORMAL_DATA_ROOT / "research-releases" / str(value)
    release = verify_research_release(path)
    identity = {"release_id": release["release_id"], "root": release["root"],
            "database_sha256": release["database_sha256"],
            "provider_sha256": release["provider_sha256"],
            "raw_manifest_sha256": release["raw_manifest_sha256"],
            "normalized_manifest_sha256": release["normalized_manifest_sha256"],
            "data_mode": resolved.context.data_mode,
            "capabilities": {name: release["capabilities"][name]
                             for name in ("market_data", "fundamentals", "industry")},
            "versions": release.get("versions", {}), "sources": release.get("sources", {}),
            "coverage": release.get("coverage", {}),
            "known_biases": release.get("known_biases", []),
            "point_in_time": release["capabilities"]["point_in_time"]}
    if dependencies is not None:
        from ..pit.capabilities import validate_release_dependencies
        validate_release_dependencies(identity, dependencies, data_mode=resolved.context.data_mode,
                                      as_of=resolved.context.as_of,
                                      as_of_policy=resolved.context.as_of_policy)
    return identity


def _research_scores(resolved, input_payload):
    dataset_ref = input_payload.get("factor_dataset")
    if not dataset_ref:
        return input_payload.get("scores", []), {"kind": "inline_scores"}
    if input_payload.get("scores"):
        raise ValueError("research input must choose factor_dataset or inline scores, not both")
    from ..factor_store import load_factor_scores, verify_factor_dataset

    dataset = verify_factor_dataset(dataset_ref)
    if dataset.get("factor_id") != resolved.factor.id:
        raise ValueError("factor dataset factor_id does not match the resolved experiment")
    if dataset.get("snapshot_compatible") and not resolved.context.snapshot_compatible:
        raise ValueError("snapshot-compatible factor dataset cannot be used by a non-snapshot context")
    dataset_release = dataset.get("research_release") or {}
    if resolved.context.research_release and dataset_release.get("release_id") \
            != str(resolved.context.research_release) \
            and dataset_release.get("root") != str(Path(resolved.context.research_release).expanduser().resolve()):
        raise ValueError("factor dataset research release does not match the resolved experiment")
    return load_factor_scores(dataset_ref), {
        "kind": "factor_dataset", "dataset_id": dataset["dataset_id"],
        "factor_id": dataset["factor_id"], "root": dataset["root"],
        "content_sha256": dataset["content_sha256"],
    }


def _has_industry_input(scores, groups):
    for value in (groups or {}).values():
        if isinstance(value, dict) and (value.get("industry") or value.get("sector")):
            return True
    return any(isinstance(row, dict) and ((row.get("groups") or {}).get("industry")
               or (row.get("groups") or {}).get("sector") or row.get("industry") or row.get("sector"))
               for row in scores)


def _point_in_time_industry_groups(resolved, release_identity, scores, groups):
    from ..pit.repositories import IndustryPITRepository

    classification = (resolved.context.metadata.get("industry_classification")
                      or resolved.experiment.metadata.get("industry_classification"))
    if not classification:
        raise ValueError("point_in_time industry dependency requires context.metadata.industry_classification")
    database = Path(release_identity["root"]) / "database" / "market.duckdb"
    repository = IndustryPITRepository(database)
    observations = {}
    for row in scores:
        if isinstance(row, dict):
            timestamp, symbol = str(row.get("timestamp", "")), str(row.get("symbol", ""))
        else:
            timestamp, symbol = row.timestamp, row.symbol
        if timestamp and symbol:
            observations.setdefault(timestamp, set()).add(symbol)
    result = dict(groups or {})
    for timestamp, symbols in sorted(observations.items()):
        rows = repository.get_industry(sorted(symbols), classification=classification, as_of=timestamp)
        found = {row["instrument"][-6:]: row for row in rows}
        if len(found) != len(symbols):
            missing = sorted(symbol for symbol in symbols if symbol[-6:] not in found)
            raise ValueError(f"PIT industry coverage is missing at {timestamp}: {', '.join(missing)}")
        for symbol in symbols:
            key = f"{timestamp}|{symbol}"
            prior = dict(result.get(key) or {})
            industry = found[symbol[-6:]]
            prior.update(industry=industry["industry_name"], industry_code=industry["industry_code"],
                         classification=industry["classification"])
            result[key] = prior
    return result


def _input_dataset_capabilities(scores, groups, prices=None):
    industry_present = _has_industry_input(scores, groups)
    return {
        "market_data": {"available": bool(scores or prices), "point_in_time": False,
                        "source": "factor_scores_or_prices", "coverage": {"factor_score_rows": len(scores),
                                                                             "price_rows": len(prices or [])}},
        "fundamentals": {"available": False, "point_in_time": False,
                         "source": None, "coverage": {}},
        "industry": {"available": industry_present, "point_in_time": False,
                     "source": "caller_supplied_groups" if industry_present else None,
                     "coverage": {"observations": sum(1 for value in (groups or {}).values()
                                                         if isinstance(value, dict) and
                                                         (value.get("industry") or value.get("sector")))}}
    }


def _as_scores(values, factor_id):
    result = []
    for value in values:
        if isinstance(value, FactorScore):
            result.append(value)
            continue
        result.append(FactorScore(timestamp=str(value["timestamp"]), symbol=str(value["symbol"]),
                                  score=float(value["score"]), source_factor_id=factor_id))
    return result


def _as_candidates(values, resolved):
    result, ids, keys = [], set(), set()
    for value in values or []:
        candidate = value if isinstance(value, CandidateSignal) else CandidateSignal(
            signal_id=str(value["signal_id"]), timestamp=str(value["timestamp"]),
            symbol=str(value["symbol"]), score=float(value["score"]), rank=int(value["rank"]),
            percentile=float(value["percentile"]), side=str(value.get("side", "LONG")).upper(),
            source_factor_id=str(value["source_factor_id"]),
            source_factor_score_id=str(value["source_factor_score_id"]))
        key = (candidate.timestamp, candidate.symbol)
        if candidate.signal_id in ids or key in keys:
            raise ValueError(f"duplicate CandidateSignal: {candidate.signal_id}/{key}")
        if not all((candidate.signal_id, candidate.timestamp, candidate.symbol,
                    candidate.source_factor_score_id)):
            raise ValueError("CandidateSignal identifiers must be non-empty")
        if candidate.source_factor_id != resolved.factor.id:
            raise ValueError("CandidateSignal source_factor_id does not match the resolved experiment")
        if candidate.side not in {"LONG", "SHORT"} or not _finite(candidate.score) or candidate.rank < 1 \
                or not _finite(candidate.percentile) or not 0 <= candidate.percentile <= 1:
            raise ValueError(f"unsupported or invalid CandidateSignal: {candidate.signal_id}")
        ids.add(candidate.signal_id)
        keys.add(key)
        result.append(candidate)
    return sorted(result, key=lambda item: (item.timestamp, item.rank, item.symbol))


def _corr(left, right):
    if len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right))
    return numerator / denominator if denominator else None


def _ranks(values):
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index, _ in ordered[start:end]:
            result[index] = float(average_rank)
        start = end
    return result


def _label_horizons(labels):
    """Normalize scalar, horizon-first, or row-first label mappings."""
    if not labels:
        return {}
    if all(isinstance(value, dict) for value in labels.values()):
        if all(str(key).startswith("h") for key in labels):
            return {str(horizon): dict(values) for horizon, values in labels.items()}
        result = defaultdict(dict)
        for observation, values in labels.items():
            for horizon, value in values.items():
                result[str(horizon)][str(observation)] = value
        return dict(result)
    return {"h1": dict(labels)}


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _observations_on_date(values, as_of):
    if not isinstance(values, dict):
        return values
    return {key: value for key, value in values.items()
            if str(key).split("|", 1)[0][:10] == as_of}


def _labels_on_date(labels, as_of):
    if not isinstance(labels, dict):
        return labels
    if labels and all(str(key).startswith("h") and isinstance(value, dict)
                      for key, value in labels.items()):
        return {horizon: _observations_on_date(values, as_of)
                for horizon, values in labels.items()}
    return _observations_on_date(labels, as_of)


def _row_session_date(row):
    timestamp = (row.get("timestamp") if isinstance(row, dict)
                 else getattr(row, "timestamp", None))
    value = str(timestamp or "")[:10]
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as error:
        raise ValueError(f"research row has an invalid timestamp: {timestamp!r}") from error


def _date_bound(value, name):
    if value is None:
        return None
    text = str(value)[:10]
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as error:
        raise ValueError(f"context.{name} must begin with an ISO date") from error


def _observations_on_dates(values, selected_dates):
    if not isinstance(values, dict):
        return values
    return {key: value for key, value in values.items()
            if str(key).split("|", 1)[0][:10] in selected_dates}


def _labels_on_dates(labels, selected_dates):
    if not isinstance(labels, dict):
        return labels
    if labels and all(str(key).startswith("h") and isinstance(value, dict)
                      for key, value in labels.items()):
        return {horizon: _observations_on_dates(values, selected_dates)
                for horizon, values in labels.items()}
    return _observations_on_dates(labels, selected_dates)


def _filter_research_window(rows, labels, groups, regimes, context, session_window=None):
    """Apply Context dates and an optional newest-first session-position slice."""
    start = _date_bound(context.start, "start")
    end = _date_bound(context.end, "end")
    if start and end and start > end:
        raise ValueError("context.start must not be after context.end")

    input_dates = sorted({_row_session_date(row) for row in rows}, reverse=True)
    bounded_dates = [stamp for stamp in input_dates
                     if (start is None or stamp >= start) and (end is None or stamp <= end)]
    if not bounded_dates:
        raise ValueError("research window contains no factor scores or candidate signals")

    selected_order = bounded_dates
    normalized_session_window = None
    if session_window is not None:
        if not isinstance(session_window, dict):
            raise ValueError("session_window must be an object with after and through")
        if set(session_window) != {"after", "through"}:
            raise ValueError("session_window may contain only after and through")
        after, through = session_window.get("after"), session_window.get("through")
        if (isinstance(after, bool) or isinstance(through, bool)
                or not isinstance(after, int) or not isinstance(through, int)
                or after < 0 or through <= after):
            raise ValueError("session_window requires integers satisfying 0 <= after < through")
        if through > len(bounded_dates):
            raise ValueError(
                f"session_window through={through} exceeds {len(bounded_dates)} available sessions"
            )
        selected_order = bounded_dates[after:through]
        normalized_session_window = {
            "after": after,
            "through": through,
            "selected_position_start": after + 1,
            "selected_position_end": through,
            "order": "newest_first",
        }

    selected_dates = set(selected_order)
    filtered_rows = [row for row in rows if _row_session_date(row) in selected_dates]
    if not filtered_rows:
        raise ValueError("research window contains no usable rows")
    window = {
        "kind": ("recent_session_positions" if normalized_session_window
                 else "context_date_range"),
        "requested_start": start,
        "requested_end": end,
        "input_session_count": len(input_dates),
        "bounded_session_count": len(bounded_dates),
        "selected_session_count": len(selected_dates),
        "selected_row_count": len(filtered_rows),
        "selected_start": min(selected_dates),
        "selected_end": max(selected_dates),
    }
    if normalized_session_window:
        window["session_window"] = normalized_session_window
    return (filtered_rows, _labels_on_dates(labels, selected_dates),
            _observations_on_dates(groups, selected_dates),
            _observations_on_dates(regimes, selected_dates), window, selected_dates)


def infer_regimes(source) -> tuple[dict, dict]:
    """Infer point-in-time regimes from a caller-supplied benchmark close series."""
    if not isinstance(source, dict) or not isinstance(source.get("prices"), list):
        raise ValueError("regime_source requires a prices array")
    window = int(source.get("window", 20))
    if window < 2:
        raise ValueError("regime_source window must be at least 2")
    bull = float(source.get("bull_threshold", 0.02))
    bear = float(source.get("bear_threshold", -0.02))
    high_volatility = float(source.get("high_volatility_threshold", 0.30))
    if not all(_finite(value) for value in (bull, bear, high_volatility)) \
            or bear >= bull or high_volatility <= 0:
        raise ValueError("regime_source thresholds are invalid")
    rows, seen = [], set()
    for row in source["prices"]:
        timestamp, close = str(row.get("timestamp", "")), row.get("close")
        if not timestamp or timestamp in seen or not _finite(close) or float(close) <= 0:
            raise ValueError(f"invalid or duplicate regime benchmark row: {timestamp!r}")
        seen.add(timestamp)
        rows.append((timestamp, float(close)))
    rows.sort()
    if len(rows) < 2:
        raise ValueError("regime_source requires at least two benchmark observations")
    evidence_rows, regimes = [], {}
    for index in range(1, len(rows)):
        start = max(0, index - window + 1)
        closes = [close for _, close in rows[start:index + 1]]
        returns = [right / left - 1 for left, right in zip(closes, closes[1:])]
        momentum = closes[-1] / closes[0] - 1
        volatility = pstdev(returns) * math.sqrt(252) if len(returns) > 1 else 0.0
        if volatility >= high_volatility:
            regime = "HIGH_VOL"
        elif momentum >= bull:
            regime = "BULL"
        elif momentum <= bear:
            regime = "BEAR"
        else:
            regime = "RANGE"
        timestamp = rows[index][0]
        regimes[timestamp] = regime
        evidence_rows.append({"timestamp": timestamp, "close": closes[-1],
                              "window_observations": len(closes), "momentum": momentum,
                              "annualized_volatility": volatility, "regime": regime})
    evidence = {
        "schema": "quant-project-regime-inference-v1",
        "method": "trailing_benchmark_momentum_volatility_v1",
        "parameters": {"window": window, "bull_threshold": bull,
                       "bear_threshold": bear,
                       "high_volatility_threshold": high_volatility},
        "source": {"benchmark_id": source.get("benchmark_id"),
                   "observation_count": len(rows), "first_timestamp": rows[0][0],
                   "last_timestamp": rows[-1][0]},
        "rows": evidence_rows,
    }
    return regimes, evidence


def _summary(values):
    finite = [float(value) for value in values if _finite(value)]
    return {
        "mean": mean(finite) if finite else None,
        "median": median(finite) if finite else None,
        "std": pstdev(finite) if len(finite) > 1 else 0.0 if finite else None,
        "count": len(finite),
        "positive_ratio": sum(value > 0 for value in finite) / len(finite) if finite else None,
    }


def _observation_dimensions(scores, groups=None):
    """Merge optional row metadata with an explicit observation-key mapping."""
    result = {}
    for row in scores:
        if isinstance(row, dict):
            key = f"{row.get('timestamp')}|{row.get('symbol')}"
            dimensions = dict(row.get("groups") or {})
            for name in ("industry", "sector", "size_bucket", "liquidity_bucket", "asset_group"):
                if row.get(name) is not None:
                    dimensions[name] = row[name]
            if dimensions:
                result[key] = {str(name): str(value) for name, value in dimensions.items()
                               if value is not None}
    for key, value in (groups or {}).items():
        if isinstance(value, dict):
            result.setdefault(str(key), {}).update({str(name): str(item)
                                                    for name, item in value.items()
                                                    if item is not None})
        elif value is not None:
            result.setdefault(str(key), {})["asset_group"] = str(value)
    return result


def _group_factor_stability(grouped, label_map, dimensions, regimes):
    grouped_dimensions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for timestamp, rows in grouped.items():
        for row in rows:
            key = f"{timestamp}|{row.symbol}"
            for dimension, value in dimensions.get(key, {}).items():
                grouped_dimensions[dimension][value][timestamp].append(row)
    asset_groups = []
    for dimension, values in sorted(grouped_dimensions.items()):
        for value, subset in sorted(values.items()):
            analysis = _factor_horizon(subset, label_map)
            asset_groups.append({"dimension": dimension, "group": value,
                                 "sample_count": analysis["valid_rows"],
                                 "rank_ic_mean": analysis["predictive"]["rank_ic_mean"],
                                 "positive_ratio": analysis["predictive"]["positive_ratio"]})
    regime_groups = defaultdict(lambda: defaultdict(list))
    for timestamp, rows in grouped.items():
        regime = (regimes or {}).get(timestamp)
        if regime is not None:
            regime_groups[str(regime)][timestamp].extend(rows)
    regime_rows = []
    for regime, subset in sorted(regime_groups.items()):
        analysis = _factor_horizon(subset, label_map)
        regime_rows.append({"regime": regime, "sample_count": analysis["valid_rows"],
                            "rank_ic_mean": analysis["predictive"]["rank_ic_mean"],
                            "positive_ratio": analysis["predictive"]["positive_ratio"]})
    return asset_groups, regime_rows


def _percentile(values, probability):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _stable_id(prefix, *parts):
    payload = "|".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _factor_horizon(grouped, label_map, *, quantiles=5):
    pearson_values, rank_values = [], []
    quantile_values = [[] for _ in range(quantiles)]
    tails = {ratio: {"top": [], "bottom": []} for ratio in (0.20, 0.10, 0.05, 0.02, 0.01)}
    time_blocks = []
    valid = 0
    for timestamp, rows in sorted(grouped.items()):
        pairs = [(row.score, label_map.get(f"{timestamp}|{row.symbol}")) for row in rows]
        pairs = [(float(score), float(label)) for score, label in pairs if _finite(label)]
        valid += len(pairs)
        if len(pairs) >= 2:
            xs, ys = zip(*pairs)
            pearson, rank_ic = _corr(xs, ys), _corr(_ranks(xs), _ranks(ys))
            if pearson is not None:
                pearson_values.append(pearson)
            if rank_ic is not None:
                rank_values.append(rank_ic)
            time_blocks.append({"block": timestamp, "sample_count": len(pairs),
                                "pearson_ic": pearson, "rank_ic": rank_ic})
        ordered = sorted(pairs, key=lambda item: item[0])
        for index, (_, label) in enumerate(ordered):
            bucket = min(quantiles - 1, int(index * quantiles / max(len(ordered), 1)))
            quantile_values[bucket].append(label)
        for ratio in tails:
            count = max(1, math.ceil(len(ordered) * ratio)) if ordered else 0
            if count:
                tails[ratio]["bottom"].extend(label for _, label in ordered[:count])
                tails[ratio]["top"].extend(label for _, label in ordered[-count:])
    pearson_summary, rank_summary = _summary(pearson_values), _summary(rank_values)
    quantile_returns = [{"quantile": index + 1, **_summary(values)}
                        for index, values in enumerate(quantile_values)]
    q_means = [item["mean"] for item in quantile_returns if item["mean"] is not None]
    monotonicity = _corr(list(range(1, len(q_means) + 1)), q_means) if len(q_means) >= 2 else None
    tail_report = {}
    for ratio, values in tails.items():
        top, bottom = _summary(values["top"]), _summary(values["bottom"])
        key = f"top_{int(ratio * 100)}"
        tail_report[key] = {"top": top, "bottom": bottom,
                            "top_minus_bottom": (top["mean"] - bottom["mean"]
                                                 if top["mean"] is not None and bottom["mean"] is not None
                                                 else None)}
    return {
        "valid_rows": valid,
        "predictive": {
            "pearson_ic_mean": pearson_summary["mean"],
            "pearson_ic_median": pearson_summary["median"],
            "pearson_ic_std": pearson_summary["std"],
            "rank_ic_mean": rank_summary["mean"],
            "rank_ic_median": rank_summary["median"],
            "rank_ic_std": rank_summary["std"],
            "icir": (rank_summary["mean"] / rank_summary["std"]
                       if rank_summary["mean"] is not None and rank_summary["std"] else None),
            "rank_ic_t_stat": (rank_summary["mean"]
                                / (rank_summary["std"] / math.sqrt(rank_summary["count"]))
                                if rank_summary["mean"] is not None and rank_summary["std"]
                                and rank_summary["count"] else None),
            "positive_ratio": rank_summary["positive_ratio"],
            "sample_count": rank_summary["count"],
        },
        "quantiles": {"quantile_returns": quantile_returns,
                      "top_minus_bottom": (q_means[-1] - q_means[0] if len(q_means) >= 2 else None),
                      "monotonicity_score": monotonicity},
        "tails": tail_report,
        "time_blocks": time_blocks,
    }


def _factor_characteristics(grouped):
    timestamps = sorted(grouped)
    dispersions, score_autocorrelation, rank_persistence, turnovers = [], [], [], []
    previous_scores, previous_top = None, None
    for timestamp in timestamps:
        rows = grouped[timestamp]
        current_scores = {row.symbol: row.score for row in rows}
        if len(current_scores) > 1:
            dispersions.append(pstdev(current_scores.values()))
        ordered = sorted(current_scores, key=lambda symbol: (-current_scores[symbol], symbol))
        current_top = set(ordered[:max(1, math.ceil(len(ordered) * 0.2))])
        if previous_scores is not None:
            common = sorted(set(previous_scores) & set(current_scores))
            if len(common) >= 2:
                score_autocorrelation.append(_corr([previous_scores[key] for key in common],
                                                   [current_scores[key] for key in common]))
                rank_persistence.append(_corr(_ranks([previous_scores[key] for key in common]),
                                              _ranks([current_scores[key] for key in common])))
            union = previous_top | current_top
            turnovers.append(1 - len(previous_top & current_top) / len(union) if union else 0.0)
        previous_scores, previous_top = current_scores, current_top
    return {"turnover": _summary(turnovers)["mean"],
            "autocorrelation": _summary(score_autocorrelation)["mean"],
            "cross_sectional_dispersion": _summary(dispersions)["mean"],
            "concentration": None,
            "rank_persistence": _summary(rank_persistence)["mean"]}


def factor_eval(resolved: ResolvedExperiment, scores, labels=None, *, groups=None,
                regimes=None) -> tuple[dict, list[FactorScore]]:
    """Evaluate cross-sectional factor data without creating CandidateSignals."""
    factor_scores = _as_scores(scores, resolved.factor.id)
    horizons = _label_horizons(labels)
    grouped = defaultdict(list)
    seen = set()
    for score in factor_scores:
        key = (score.timestamp, score.symbol)
        if key not in seen and _finite(score.score):
            grouped[score.timestamp].append(score)
            seen.add(key)
    by_horizon = {name: _factor_horizon(grouped, values)
                  for name, values in sorted(horizons.items())}
    primary = by_horizon.get("h1") or next(iter(by_horizon.values()), None)
    valid = primary["valid_rows"] if primary else 0
    predictive = {name: value["predictive"] for name, value in by_horizon.items()}
    quantile = {name: value["quantiles"] for name, value in by_horizon.items()}
    tails = {name: value["tails"] for name, value in by_horizon.items()}
    curve = [{"horizon": name, "rank_ic": value["predictive"]["rank_ic_mean"],
              "quantile_spread": value["quantiles"]["top_minus_bottom"]}
             for name, value in by_horizon.items()]
    finite_curve = [item for item in curve if item["rank_ic"] is not None]
    peak = max(finite_curve, key=lambda item: abs(item["rank_ic"]))["horizon"] if finite_curve else None
    invalid_scores = sum(not _finite(item.score) for item in factor_scores)
    dimensions = _observation_dimensions(scores, groups)
    primary_labels = horizons.get("h1") or next(iter(horizons.values()), {})
    asset_groups, regime_rows = _group_factor_stability(grouped, primary_labels,
                                                        dimensions, regimes or {})
    report = {"schema": "stock-factor-report-v1", "skill_meta": {"skill_name": "factor-eval", "skill_version": "1.0.0"},
              "context_snapshot": _context(resolved), "factor_spec_snapshot": resolved.factor.to_dict(),
              "label_spec_snapshot": {"horizons": list(horizons), "source": "caller-supplied"},
              "data_quality": {"total_rows": len(factor_scores), "valid_rows": valid,
                               "coverage": valid / len(factor_scores) if factor_scores else 0.0,
                               "missing_rate": 1 - valid / len(factor_scores) if factor_scores else 0.0,
                               "invalid_rate": invalid_scores / len(factor_scores) if factor_scores else 0.0,
                               "duplicate_count": len(factor_scores) - len({(x.timestamp, x.symbol) for x in factor_scores}),
                               "valid_score_rows": sum(len(rows) for rows in grouped.values()),
                               "cross_section_count": len(grouped)},
              "predictive_power": {"by_horizon": predictive, **({"h1": predictive["h1"]} if "h1" in predictive else {})},
              "quantile_analysis": {"by_horizon": quantile},
              "tail_analysis": {"by_horizon": tails},
              "decay_analysis": {"horizon_curve": curve, "peak_horizon": peak,
                                 "half_life_diagnostics": None},
              "stability_analysis": {"time_blocks": (primary or {}).get("time_blocks", []),
                                     "time_block_definition": "one input timestamp",
                                     "asset_groups": asset_groups, "regimes": regime_rows},
              "characteristics": _factor_characteristics(grouped),
              "warnings": _warning_context(resolved), "diagnostics": []}
    return report, factor_scores


def signal_eval(resolved: ResolvedExperiment, scores, labels=None, *, groups=None,
                regimes=None) -> tuple[dict, list[CandidateSignal]]:
    """Produce CandidateSignal objects only; capacity and fills are excluded."""
    factor_scores = _as_scores(scores, resolved.factor.id)
    dimensions = _observation_dimensions(scores, groups)
    return _signal_eval(resolved, factor_scores, labels, dimensions=dimensions, regimes=regimes)


def model_signal_eval(resolved, scores, labels=None, *, groups=None, regimes=None, compiled_plan_sha256):
    """Strict V2 handoff: normalized ModelScore -> lineage-preserving candidates."""
    from ..modeling.contracts import DataContractError
    from ..model_score_store import validate_model_scores
    if not isinstance(compiled_plan_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", compiled_plan_sha256):
        raise DataContractError("V2 Signal requires its compiled plan SHA-256")
    rows, keys = [], set()
    model_config_sha256 = resolved.model.config_sha256
    factor_set_config_sha256 = resolved.factor_set.config_sha256
    for value in scores:
        if isinstance(value, ModelScore):
            row = value
        elif isinstance(value, dict):
            try:
                row = ModelScore(**value)
            except TypeError as error:
                raise DataContractError("V2 Signal requires ModelScore objects with complete provenance") from error
        else:
            raise DataContractError("V2 Signal cannot consume raw FactorScore or top rows")
        key = row.timestamp, row.symbol
        if key in keys or isinstance(row.score, bool) or not isinstance(row.score, (int, float)) or not _finite(row.score):
            raise DataContractError("V2 ModelScore keys must be unique and scores finite")
        keys.add(key)
        if (row.source_model_id != resolved.model.id or row.model_config_sha256 != model_config_sha256
                or row.source_factor_set_id != resolved.factor_set.id
                or row.factor_set_config_sha256 != factor_set_config_sha256):
            raise DataContractError("ModelScore lineage does not match the resolved four-layer composition")
        if not all((row.model_score_id, row.research_run_id, row.fold_id, row.source_factor_vector_id,
                    row.artifact_manifest_sha256, row.model_payload_sha256)):
            raise DataContractError("V2 ModelScore provenance is incomplete")
        rows.append(row)
    validate_model_scores([row.to_dict() for row in rows])
    return _signal_eval(resolved, rows, labels, dimensions=_observation_dimensions(scores, groups),
                        regimes=regimes, model_source=True, compiled_plan_sha256=compiled_plan_sha256)


def _signal_eval(resolved, factor_scores, labels, *, dimensions, regimes=None,
                 model_source=False, compiled_plan_sha256=None):
    # Selection/statistics are shared by the two explicit score contracts.
    grouped = defaultdict(list)
    for score in factor_scores:
        grouped[score.timestamp].append(score)
    selection = resolved.signal.selection
    method = selection.get("method")
    default_side = str(resolved.signal.direction.get("side", "long")).upper()
    if default_side not in {"LONG", "SHORT"}:
        raise ValueError(f"unsupported signal direction side: {default_side}")
    descending = str(resolved.signal.ranking.get(
        "order", "descending" if model_source or resolved.factor.higher_is_better else "ascending")).lower() \
        not in {"ascending", "asc"}
    industry_limit = resolved.signal.eligibility.get("known_industry_max_positions")
    missing_industry_is_independent = resolved.signal.eligibility.get(
        "missing_industry_is_independent", True)
    candidates = []
    for timestamp, rows in sorted(grouped.items()):
        ordered = []
        seen = set()
        for row in sorted((item for item in rows if _finite(item.score)),
                          key=(lambda item: (-item.score, item.symbol)) if descending
                          else (lambda item: (item.score, item.symbol))):
            if row.symbol not in seen:
                ordered.append(row)
                seen.add(row.symbol)
        selections = []
        if method == "top_k":
            count = int(selection.get("top_k", 0))
            if count < 1:
                raise ValueError("top_k must be positive")
            if industry_limit is None:
                selected = ordered[:count]
            else:
                selected, industry_counts = [], defaultdict(int)
                for row in ordered:
                    observation = dimensions.get(f"{timestamp}|{row.symbol}", {})
                    industry = str(observation.get("industry") or observation.get("sector") or "").strip()
                    key = industry or None
                    if (industry or not missing_industry_is_independent) and industry_counts[key] >= industry_limit:
                        continue
                    if industry or not missing_industry_is_independent:
                        industry_counts[key] += 1
                    selected.append(row)
                    if len(selected) == count:
                        break
            selections.append((default_side, selected))
        elif method == "top_percentile":
            share = float(selection.get("value", selection.get("top_percentile", 0)))
            if not 0 < share <= 1:
                raise ValueError("top_percentile must be in (0, 1]")
            selections.append((default_side, ordered[:math.ceil(
                len(ordered) * share)]))
        elif method == "bottom_k":
            count = int(selection.get("bottom_k", 0))
            if count < 1:
                raise ValueError("bottom_k must be positive")
            selections.append((default_side, list(reversed(ordered[-count:]))))
        elif method == "threshold":
            threshold = float(selection["threshold"])
            operator = str(selection.get("operator", "lte" if default_side == "SHORT" else "gte")).lower()
            if operator not in {"gte", "lte"}:
                raise ValueError("threshold operator must be gte or lte")
            selections.append((default_side, [row for row in ordered
                                               if row.score >= threshold] if operator == "gte"
                              else [row for row in reversed(ordered) if row.score <= threshold]))
        elif method == "rank_range":
            minimum = int(selection.get("min_rank", 1))
            maximum = int(selection.get("max_rank", len(ordered)))
            if minimum < 1 or maximum < minimum:
                raise ValueError("rank_range requires 1 <= min_rank <= max_rank")
            selections.append((default_side, ordered[minimum - 1:maximum]))
        elif method == "dual_threshold":
            long_threshold = float(selection["long_threshold"])
            short_threshold = float(selection["short_threshold"])
            if short_threshold >= long_threshold:
                raise ValueError("dual_threshold requires short_threshold < long_threshold")
            selections.append(("LONG", [row for row in ordered if row.score >= long_threshold]))
            selections.append(("SHORT", [row for row in reversed(ordered)
                                           if row.score <= short_threshold]))
        else:
            raise ValueError(f"unsupported research signal selection method: {method}")
        for side, selected in selections:
            for rank, row in enumerate(selected, 1):
                if model_source:
                    candidates.append(ModelCandidateSignal(
                        signal_id=f"{resolved.signal.id}:{row.model_score_id}:{side}",
                        timestamp=timestamp, symbol=row.symbol, score=row.score, rank=rank,
                        percentile=rank / len(ordered), side=side, research_run_id=row.research_run_id,
                        source_model_score_id=row.model_score_id, source_model_id=row.source_model_id,
                        source_factor_set_id=row.source_factor_set_id, source_factor_vector_id=row.source_factor_vector_id,
                        fold_id=row.fold_id, model_config_sha256=row.model_config_sha256,
                        factor_set_config_sha256=row.factor_set_config_sha256,
                        artifact_manifest_sha256=row.artifact_manifest_sha256, model_payload_sha256=row.model_payload_sha256,
                        compiled_plan_sha256=compiled_plan_sha256))
                    continue
                candidates.append(CandidateSignal(
                    signal_id=f"{resolved.signal.id}:{timestamp}:{row.symbol}:{side}",
                    timestamp=timestamp, symbol=row.symbol, score=row.score, rank=rank,
                    percentile=rank / len(ordered), side=side,
                    source_factor_id=resolved.factor.id,
                    source_factor_score_id=f"{timestamp}:{row.symbol}",
                ))
    candidate_groups = defaultdict(list)
    for candidate in candidates:
        candidate_groups[candidate.timestamp].append(candidate)
    counts = [len(candidate_groups.get(timestamp, [])) for timestamp in sorted(grouped)]
    symbol_counts = defaultdict(int)
    for candidate in candidates:
        symbol_counts[candidate.symbol] += 1
    ordered_times = sorted(grouped)
    set_turnovers, overlap_ratios, rank_turnovers, persistence = [], [], [], []
    for left_time, right_time in zip(ordered_times, ordered_times[1:]):
        left = candidate_groups.get(left_time, [])
        right = candidate_groups.get(right_time, [])
        left_set = {(item.symbol, item.side) for item in left}
        right_set = {(item.symbol, item.side) for item in right}
        union, intersection = left_set | right_set, left_set & right_set
        set_turnovers.append(1 - len(intersection) / len(union) if union else 0.0)
        overlap_ratios.append(len(intersection) / min(len(left_set), len(right_set))
                              if left_set and right_set else 0.0)
        persistence.append(len(intersection) / len(left_set) if left_set else 0.0)
        if len(intersection) >= 2:
            left_rank = {(item.symbol, item.side): item.rank for item in left}
            right_rank = {(item.symbol, item.side): item.rank for item in right}
            common = sorted(intersection)
            rank_turnovers.append(1 - (_corr([left_rank[key] for key in common],
                                             [right_rank[key] for key in common]) or 0.0))
    quality = {}
    for horizon, label_map in sorted(_label_horizons(labels).items()):
        values = [float(label_map[f"{item.timestamp}|{item.symbol}"])
                  * (1 if item.side == "LONG" else -1)
                  for item in candidates if _finite(label_map.get(f"{item.timestamp}|{item.symbol}"))]
        stats = _summary(values)
        quality[horizon] = {"mean_future_return": stats["mean"],
                            "median_future_return": stats["median"],
                            "hit_rate": stats["positive_ratio"], "sample_count": stats["count"]}
    total = len(candidates)
    shares = [count / total for count in symbol_counts.values()] if total else []
    group_counts = defaultdict(lambda: defaultdict(int))
    regime_counts = defaultdict(int)
    for candidate in candidates:
        key = f"{candidate.timestamp}|{candidate.symbol}"
        for dimension, value in dimensions.get(key, {}).items():
            group_counts[dimension][value] += 1
        regime = (regimes or {}).get(candidate.timestamp)
        if regime is not None:
            regime_counts[str(regime)] += 1
    group_stability = [{"dimension": dimension, "group": group, "signal_count": count,
                        "share": count / total if total else 0.0}
                       for dimension, values in sorted(group_counts.items())
                       for group, count in sorted(values.items())]
    regime_stability = [{"regime": regime, "signal_count": count,
                         "share": count / total if total else 0.0}
                        for regime, count in sorted(regime_counts.items())]
    sector_values = group_counts.get("industry") or group_counts.get("sector") or {}
    sector_shares = [count / total for count in sector_values.values()] if total else []
    stability = []
    by_month = defaultdict(list)
    for candidate in candidates:
        by_month[candidate.timestamp[:7]].append(candidate)
    for block, rows in sorted(by_month.items()):
        stability.append({"block": block, "signal_count": len(rows),
                          "unique_symbols": len({item.symbol for item in rows})})
    source_snapshot = ({"model": resolved.model.id, "factor_set": resolved.factor_set.id,
                        "model_score_count": len(factor_scores), "higher_is_better": True,
                        "compiled_plan_sha256": compiled_plan_sha256} if model_source else
                       {"factor": resolved.factor.id, "factor_score_count": len(factor_scores)})
    report = {"schema": "stock-signal-report-v2" if model_source else "stock-signal-report-v1",
              "skill_meta": {"skill_name": "signal-eval", "skill_version": "2.0.0" if model_source else "1.0.0"},
              "context_snapshot": _context(resolved), "signal_spec_snapshot": resolved.signal.to_dict(),
              "source_snapshot": source_snapshot,
              "signal_statistics": {"total_signals": len(candidates), "timestamps_with_signal": len({x.timestamp for x in candidates}),
                                    "signals_per_timestamp_mean": mean(counts) if counts else 0.0,
                                    "signals_per_timestamp_median": median(counts) if counts else 0.0,
                                    "empty_period_ratio": sum(count == 0 for count in counts) / len(counts) if counts else 0.0,
                                    "burstiness": (pstdev(counts) / mean(counts)
                                                   if len(counts) > 1 and mean(counts) else 0.0)},
              "quality": {"by_horizon": quality,
                          "status": "computed" if quality else "labels_not_available"},
              "overlap": {"same_symbol_repeat_rate": (sum(max(0, count - 1) for count in symbol_counts.values()) / total
                                                        if total else 0.0),
                          "concurrent_signal_count_mean": mean(counts) if counts else 0.0,
                          "concurrent_signal_count_p95": _percentile(counts, 0.95),
                          "overlap_ratio": _summary(overlap_ratios)["mean"] or 0.0},
              "turnover": {"candidate_set_turnover": _summary(set_turnovers)["mean"] or 0.0,
                           "rank_turnover": _summary(rank_turnovers)["mean"],
                           "persistence": _summary(persistence)["mean"]},
              "concentration": {"top_symbol_share": max(shares) if shares else 0.0,
                                "top_sector_share": max(sector_shares) if sector_shares else None,
                                "herfindahl_index": sum(share * share for share in shares),
                                "sector_herfindahl_index": (sum(share * share for share in sector_shares)
                                                             if sector_shares else None)},
              "stability": {"time_blocks": stability, "time_block_definition": "calendar month",
                            "asset_groups": group_stability, "regimes": regime_stability},
              "warnings": _warning_context(resolved), "diagnostics": []}
    return report, candidates


def _market_bars(prices, *, touch_policy="block_on_touch", trading_calendar=None):
    price_rows = [dict(row) for row in prices]
    # Six-digit exchange securities must carry the required exchange and rule
    # metadata; generic synthetic symbols can still exercise the event engine
    # without claiming that a CN exchange rule was derived.
    rule_rows = {}
    for index, row in enumerate(price_rows):
        identity = str(row.get("code", row.get("symbol", ""))).strip().upper()
        looks_like_exchange_code = bool(re.fullmatch(
            r"(?:(?:SH|SZ|SSE|SZSE|XSHG|XSHE)\.?\d{6}|\d{6}(?:\.(?:SH|SZ|SSE|SZSE|XSHG|XSHE))?)",
            identity,
        ))
        if row.get("exchange") is not None or looks_like_exchange_code:
            rule_rows[index] = row
    derived_by_index = {}
    if rule_rows:
        derived = derive_price_limit_rows(rule_rows.values(), trading_calendar=trading_calendar)
        derived_by_index = dict(zip(rule_rows, derived))
    bars = {}
    for index, original_row in enumerate(price_rows):
        row = derived_by_index.get(index, original_row)
        timestamp_raw = row.get("timestamp", row.get("date"))
        symbol_raw = row.get("symbol", row.get("code"))
        if timestamp_raw is None or symbol_raw is None or not str(timestamp_raw) or not str(symbol_raw):
            raise ValueError("market bar timestamp and symbol must be non-empty")
        timestamp, symbol = str(timestamp_raw), str(symbol_raw)
        key = (timestamp, symbol)
        if key in bars:
            raise ValueError(f"duplicate market bar: {timestamp}|{symbol}")
        status = str(row.get("trade_status", row.get("trading_status", ""))).strip().upper()
        suspended = (bool(row.get("suspended")) if "suspended" in row
                     else status in {"0", "SUSPENDED", "HALTED", "PAUSED", "停牌", "SUSPEND"})
        fallback = row.get("price", row.get("close"))

        def value(name, default=fallback):
            raw = row.get(name, default)
            return float(raw) if _finite(raw) else None

        if suspended:
            open_price = value("open", None)
            close = value("close", row.get("previous_close", row.get("prev_close")))
            high, low = value("high", None), value("low", None)
            open_price, high, low = [item if item is None or item > 0 else None
                                     for item in (open_price, high, low)]
            close = close if close is None or close > 0 else None
        else:
            open_price, close = value("open"), value("close")
            high = value("high", max(item for item in (open_price, close) if item is not None)
                         if open_price is not None or close is not None else None)
            low = value("low", min(item for item in (open_price, close) if item is not None)
                        if open_price is not None or close is not None else None)
        volume = value("volume", None)
        prices_present = (open_price, high, low, close)
        if not suspended and any(item is None or item <= 0 for item in prices_present):
            raise ValueError(f"market bar requires finite positive OHLC: {timestamp}|{symbol}")
        if (not suspended and
                (high < max(open_price, close, low) or low > min(open_price, close, high))):
            raise ValueError(f"market bar has inconsistent OHLC range: {timestamp}|{symbol}")
        if volume is not None and volume < 0:
            raise ValueError(f"market bar volume cannot be negative: {timestamp}|{symbol}")
        upper_limit = value("upper_limit_price", None)
        lower_limit = value("lower_limit_price", None)
        ask_volume = value("ask_volume", None)
        bid_volume = value("bid_volume", None)
        for name, item in (("upper_limit_price", upper_limit),
                           ("lower_limit_price", lower_limit)):
            if item is not None and item <= 0:
                raise ValueError(f"market bar {name} must be positive: {timestamp}|{symbol}")
        for name, item in (("ask_volume", ask_volume), ("bid_volume", bid_volume)):
            if item is not None and item < 0:
                raise ValueError(f"market bar {name} cannot be negative: {timestamp}|{symbol}")
        if upper_limit is not None and lower_limit is not None and lower_limit >= upper_limit:
            raise ValueError(f"market bar limit price range is invalid: {timestamp}|{symbol}")
        market_limit_state = assess_limit_market_state(
            row | {"high": high, "low": low,
                   "upper_limit_price": upper_limit,
                   "lower_limit_price": lower_limit},
            touch_policy=touch_policy,
        )
        buy_explicit = "limit_buy_locked" in row
        sell_explicit = "limit_sell_locked" in row
        bars[key] = {"timestamp": timestamp, "symbol": symbol, "open": open_price,
                     "high": high, "low": low, "close": close, "volume": volume,
                     "suspended": suspended,
                     "limit_buy_locked": market_limit_state["block_buy"],
                     "limit_sell_locked": market_limit_state["block_sell"],
                     "limit_state": market_limit_state,
                     "upper_limit_price": upper_limit,
                     "lower_limit_price": lower_limit,
                     "price_limit_unrestricted": row.get("price_limit_unrestricted"),
                     "price_limit_rule_id": row.get("price_limit_rule_id"),
                     "price_limit_rule_version": row.get("price_limit_rule_version"),
                     "price_limit_basis": row.get("price_limit_basis"),
                     "market_state_source": {
                        "suspension": "explicit" if "suspended" in row else (
                            "trading_status" if status else "default"),
                        "buy_limit": "explicit" if buy_explicit else (
                            "shared_rules_or_provider_field" if upper_limit is not None else "default"),
                        "sell_limit": "explicit" if sell_explicit else (
                            "shared_rules_or_provider_field" if lower_limit is not None else "default"),
                     }}
    return bars


def _canonical_market_symbol(value):
    raw = str(value).strip().upper()
    match = re.fullmatch(
        r"(?:(?:SH|SZ|SSE|SZSE|XSHG|XSHE)\.?(\d{6})|(\d{6})(?:\.(?:SH|SZ|SSE|SZSE|XSHG|XSHE))?)",
        raw,
    )
    return next((item for item in match.groups() if item), raw) if match else raw


def _corporate_action_rows(corporate_actions):
    actions = defaultdict(list)
    for action in corporate_actions or []:
        timestamp_raw = action.get("timestamp") or action.get("ex_date") or action.get("exDate")
        symbol_raw = action.get("symbol") or action.get("code") or action.get("instrument")
        if timestamp_raw is None or symbol_raw is None or not str(timestamp_raw) or not str(symbol_raw):
            raise ValueError("corporate action timestamp and symbol must be non-empty")
        timestamp, symbol = str(timestamp_raw), str(symbol_raw)
        normalized = dict(action)
        for name in ("cash_dividend_per_share", "cashDividendPer10", "bonus_share_ratio",
                     "bonusSharePer10", "bonusSharesPer10", "split_multiplier",
                     "rights_share_ratio", "rightsSharePer10", "rightsSharesPer10",
                     "rights_price", "rightsPrice"):
            if name not in normalized:
                continue
            if not _finite(normalized[name]):
                raise ValueError(f"corporate action {name} must be finite: {timestamp}|{symbol}")
            normalized[name] = float(normalized[name])
        for name in ("cash_dividend_per_share", "cashDividendPer10", "bonus_share_ratio",
                     "bonusSharePer10", "bonusSharesPer10", "rights_share_ratio",
                     "rightsSharePer10", "rightsSharesPer10", "rights_price", "rightsPrice"):
            if normalized.get(name, 0) < 0:
                raise ValueError(f"corporate action {name} cannot be negative: {timestamp}|{symbol}")
        if "split_multiplier" in normalized and normalized["split_multiplier"] <= 0:
            raise ValueError(f"corporate action split_multiplier must be positive: {timestamp}|{symbol}")
        actions[(timestamp, symbol)].append(normalized)
    return actions


def _configured_rate(spec, *names):
    for name in names:
        if spec.get(name) is not None:
            value = float(spec[name])
            return value / 100 if abs(value) > 1 else value
    return None


def _signal_exit_enabled(exit_spec):
    value = exit_spec.get("signal_exit", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("enabled", True))
    raise ValueError("signal_exit must be a boolean or an object")


def _rank_exit_limit(exit_spec):
    value = exit_spec.get("rank_exit")
    if value is None or value is False:
        return None
    if isinstance(value, bool):
        raise ValueError("rank_exit requires a positive max_rank")
    if isinstance(value, dict):
        value = value.get("max_rank")
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise ValueError("rank_exit requires a positive max_rank") from None
    if limit < 1:
        raise ValueError("rank_exit max_rank must be positive")
    return limit


def _liquidity_quantity(quantity, bar, execution, lot_size):
    participation = execution.get("max_participation_rate")
    if participation is None:
        return quantity
    if bar.get("volume") is None:
        return 0
    available = int(max(0.0, bar["volume"] * float(participation)) // lot_size) * lot_size
    if available < quantity and execution.get("partial_fill_policy", "allow") == "reject":
        return 0
    return min(quantity, available)


def strategy_backtest(resolved: ResolvedExperiment, candidates, prices,
                      corporate_actions=None, *, trading_calendar=None,
                      _prebuilt_bars=None) -> tuple[dict, dict]:
    """Deterministic event backtest over caller-supplied bars.

    Event order per timestamp is corporate action -> existing-position exit ->
    scheduled entry -> mark-to-market.  Signals are never removed merely
    because the portfolio cannot trade them; every refusal is evidence.
    """
    if any(not isinstance(item, (CandidateSignal, ModelCandidateSignal)) for item in candidates):
        raise TypeError("strategy-backtest accepts CandidateSignal objects, never raw top rows")
    portfolio, execution, exit_spec = (resolved.strategy.portfolio,
                                       resolved.strategy.execution,
                                       resolved.strategy.exit)
    limit_touch_policy = str(execution.get("limit_touch_policy", "block_on_touch"))
    actions = _corporate_action_rows(corporate_actions)
    action_keys = {(timestamp[:10], _canonical_market_symbol(symbol))
                   for timestamp, symbol in actions}
    if _prebuilt_bars is None:
        limit_input_rows = []
        for raw in prices:
            row = dict(raw)
            row_day = str(row.get("timestamp", row.get("date", "")))[:10]
            row_symbol = row.get("symbol", row.get("code"))
            if row_symbol is not None and (row_day, _canonical_market_symbol(row_symbol)) in action_keys:
                row["corporate_action"] = True
            limit_input_rows.append(row)
        bars = _market_bars(limit_input_rows, touch_policy=limit_touch_policy,
                            trading_calendar=trading_calendar)
    else:
        bars = _prebuilt_bars
    calendar = sorted({timestamp for timestamp, _ in bars})
    initial_cash = float(portfolio.get("initial_capital", 1_000_000.0))
    max_positions = int(portfolio.get("max_positions", 1))
    if initial_cash <= 0 or max_positions <= 0:
        raise ValueError("strategy portfolio requires positive initial_capital and max_positions")
    weight = float(portfolio.get("per_position_weight", 1 / max_positions))
    if not 0 < weight <= 1:
        raise ValueError("per_position_weight must be in (0, 1]")
    lot_size = max(1, int(execution.get("lot_size", 1)))
    fee_bps = float(execution.get("fee_bps",
                    float(execution.get("round_trip_cost_bps", 0.0)) / 2))
    slippage_bps = float(execution.get("slippage_bps", 0.0))
    min_fee = float(execution.get("minimum_fee", 0.0))
    short_margin_rate = float(execution.get("short_margin_rate", 1.0))
    if fee_bps < 0 or slippage_bps < 0 or min_fee < 0 or short_margin_rate <= 0:
        raise ValueError("execution fees/slippage must be non-negative and short_margin_rate positive")
    timing = str(resolved.strategy.entry.get("timing", "same_close"))
    if timing not in {"same_close", "next_open", "next_close",
                      "next_session_open", "next_session_close"}:
        raise ValueError(f"unsupported research entry timing: {timing}")
    scheduled = defaultdict(list)
    scheduled_views = defaultdict(dict)
    blocked, blocked_exits = [], []
    for candidate in sorted(candidates, key=lambda item: (item.timestamp, item.rank, item.symbol)):
        if timing == "same_close":
            execution_timestamp = candidate.timestamp if candidate.timestamp in calendar else None
        else:
            index = bisect_right(calendar, candidate.timestamp)
            execution_timestamp = calendar[index] if index < len(calendar) else None
        if execution_timestamp is None:
            blocked.append({"signal_id": candidate.signal_id, "symbol": candidate.symbol,
                            "timestamp": candidate.timestamp, "reason": "SIGNAL_EXPIRED"})
        else:
            scheduled[execution_timestamp].append(candidate)
            scheduled_views[execution_timestamp][candidate.symbol] = candidate

    cash = initial_cash
    positions = {}
    orders, fills, snapshots, trades = [], [], [], []
    cumulative_realized = 0.0
    take_rate = _configured_rate(exit_spec, "take_profit_pct", "take_profit")
    stop_rate = _configured_rate(exit_spec, "stop_loss_pct", "stop_loss")
    trailing_rate = _configured_rate(exit_spec, "trailing_stop_pct", "trailing_stop")
    maximum_holding = exit_spec.get("max_holding_days", exit_spec.get("max_holding_period"))
    maximum_holding = int(maximum_holding) if maximum_holding is not None else None
    signal_exit_enabled = _signal_exit_enabled(exit_spec)
    rank_exit_limit = _rank_exit_limit(exit_spec)
    t_plus_one = bool(execution.get("t_plus_one", False))

    def blocked_row(candidate, timestamp, reason):
        blocked.append({"signal_id": candidate.signal_id, "symbol": candidate.symbol,
                        "timestamp": timestamp, "reason": reason})

    def base_price(bar, side, *, entry=False):
        if entry:
            field = "open" if timing in {"next_open", "next_session_open"} else "close"
            return bar.get(field)
        return bar.get("close")

    def fill_price(raw, side):
        direction = 1 if side in {"BUY", "COVER"} else -1
        return raw * (1 + direction * slippage_bps / 10_000)

    def fee_for(notional):
        return max(min_fee, notional * fee_bps / 10_000)

    for timestamp in calendar:
        exited_this_session = set()
        # Corporate actions are applied on the effective date before trading.
        for symbol, state in list(positions.items()):
            for action in actions.get((timestamp, symbol), []):
                cash_per_share = float(action.get("cash_dividend_per_share",
                                       float(action.get("cashDividendPer10", 0.0)) / 10))
                bonus_ratio = float(action.get("bonus_share_ratio",
                                    float(action.get("bonusSharePer10", 0.0)) / 10))
                split_multiplier = float(action.get("split_multiplier", 1.0))
                rights_ratio = float(action.get("rights_share_ratio",
                                     float(action.get("rightsSharePer10", 0.0)) / 10))
                rights_price = float(action.get("rights_price", action.get("rightsPrice", 0.0)))
                if state["side"] == "SHORT" and rights_ratio:
                    raise ValueError(f"short corporate-action rights are unsupported: {timestamp}|{symbol}")
                dividend = state["quantity"] * cash_per_share * (1 if state["side"] == "LONG" else -1)
                rights_quantity = (int(state["quantity"] * rights_ratio)
                                   if state["side"] == "LONG" else 0)
                rights_cost = rights_quantity * rights_price
                if rights_cost > cash + dividend + 1e-9:
                    blocked_exits.append({"symbol": symbol, "timestamp": timestamp,
                                          "reason": "CORPORATE_ACTION_CASH"})
                    rights_quantity = 0
                    rights_cost = 0.0
                old_quantity, old_cost = state["quantity"], state["quantity"] * state["avg_cost"]
                new_quantity = int(round(old_quantity * (1 + bonus_ratio) * split_multiplier)) + rights_quantity
                cash += dividend - rights_cost
                if new_quantity > 0:
                    state["quantity"] = new_quantity
                    state["avg_cost"] = (old_cost + rights_cost) / new_quantity
                    state["extreme_price"] *= old_quantity / new_quantity
                state["corporate_actions"].append({"timestamp": timestamp,
                    "cash_dividend": dividend, "rights_cost": rights_cost,
                    "quantity_before": old_quantity, "quantity_after": state["quantity"]})

        # Existing positions evaluate exits before new entries consume capacity.
        for symbol, state in list(positions.items()):
            bar = bars.get((timestamp, symbol))
            if bar is None:
                continue
            state["holding_period"] += 0 if timestamp == state["opened_at"] else 1
            if _finite(bar.get("high")):
                state["extreme_price"] = (max(state["extreme_price"], bar["high"])
                                          if state["side"] == "LONG"
                                          else min(state["extreme_price"], bar["low"]))
            direction = 1 if state["side"] == "LONG" else -1
            stop = (state["avg_cost"] * (1 - direction * stop_rate)
                    if stop_rate is not None else None)
            take = (state["avg_cost"] * (1 + direction * take_rate)
                    if take_rate is not None else None)
            trailing = (state["extreme_price"] * (1 - trailing_rate)
                        if trailing_rate is not None and state["side"] == "LONG"
                        else state["extreme_price"] * (1 + trailing_rate)
                        if trailing_rate is not None else None)
            reason, raw_exit = None, None
            due_maximum_holding = (maximum_holding is not None
                                   and state["holding_period"] >= maximum_holding)
            if not bar["suspended"] and _finite(bar.get("open")):
                if stop is not None and ((state["side"] == "LONG" and bar["open"] <= stop)
                                         or (state["side"] == "SHORT" and bar["open"] >= stop)):
                    reason, raw_exit = "STOP_LOSS_GAP", bar["open"]
                elif trailing is not None and ((state["side"] == "LONG" and bar["open"] <= trailing)
                                               or (state["side"] == "SHORT" and bar["open"] >= trailing)):
                    reason, raw_exit = "TRAILING_STOP_GAP", bar["open"]
                elif take is not None and ((state["side"] == "LONG" and bar["open"] >= take)
                                           or (state["side"] == "SHORT" and bar["open"] <= take)):
                    reason, raw_exit = "TAKE_PROFIT_GAP", bar["open"]
                else:
                    risk_price = bar.get("low") if state["side"] == "LONG" else bar.get("high")
                    reward_price = bar.get("high") if state["side"] == "LONG" else bar.get("low")
                    stop_hit = stop is not None and _finite(risk_price) and (
                        risk_price <= stop if state["side"] == "LONG" else risk_price >= stop)
                    trailing_hit = trailing is not None and _finite(risk_price) and (
                        risk_price <= trailing if state["side"] == "LONG" else risk_price >= trailing)
                    take_hit = take is not None and _finite(reward_price) and (
                        reward_price >= take if state["side"] == "LONG" else reward_price <= take)
                    if stop_hit or trailing_hit:
                        reason, raw_exit = ("STOP_LOSS", stop) if stop_hit else ("TRAILING_STOP", trailing)
                    elif take_hit:
                        reason, raw_exit = "TAKE_PROFIT", take
            current_signal = scheduled_views.get(timestamp, {}).get(symbol)
            if current_signal is not None and current_signal.side != state["side"]:
                current_signal = None
            if reason is None and timestamp in scheduled_views:
                if signal_exit_enabled and current_signal is None:
                    reason, raw_exit = "SIGNAL_EXIT", bar.get("close")
                elif (rank_exit_limit is not None and current_signal is not None
                      and current_signal.rank > rank_exit_limit):
                    reason, raw_exit = "RANK_EXIT", bar.get("close")
            if reason is None and due_maximum_holding:
                reason, raw_exit = "MAX_HOLDING", bar.get("close")
            if reason is None:
                continue
            if t_plus_one and timestamp == state["opened_at"]:
                blocked_exits.append({"symbol": symbol, "timestamp": timestamp,
                                      "reason": "T_PLUS_ONE", "exit_reason": reason})
                continue
            if bar["suspended"]:
                blocked_exits.append({"symbol": symbol, "timestamp": timestamp,
                                      "reason": "SUSPENDED", "exit_reason": reason})
                continue
            exit_side = "SELL" if state["side"] == "LONG" else "COVER"
            if ((exit_side == "SELL" and bar["limit_sell_locked"])
                    or (exit_side == "COVER" and bar["limit_buy_locked"])):
                limit_reason = (bar["limit_state"]["sell_block_reason"] if exit_side == "SELL"
                                else bar["limit_state"]["buy_block_reason"]) or "LIMIT_LOCKED"
                blocked_exits.append({"symbol": symbol, "timestamp": timestamp,
                                      "reason": limit_reason, "exit_reason": reason,
                                      "market_state_evidence": bar["limit_state"]})
                continue
            if not _finite(raw_exit) or raw_exit <= 0:
                blocked_exits.append({"symbol": symbol, "timestamp": timestamp,
                                      "reason": "NO_PRICE", "exit_reason": reason})
                continue
            desired = state["quantity"]
            quantity = _liquidity_quantity(desired, bar, execution, lot_size)
            if quantity < lot_size:
                blocked_exits.append({"symbol": symbol, "timestamp": timestamp,
                                      "reason": ("LIQUIDITY_DATA_MISSING"
                                                 if execution.get("max_participation_rate") is not None
                                                 and bar.get("volume") is None else "LIQUIDITY"),
                                      "exit_reason": reason})
                continue
            price = fill_price(float(raw_exit), exit_side)
            order = Order(order_id=_stable_id("order", resolved.strategy.id, symbol,
                                               timestamp, exit_side, reason),
                          timestamp=timestamp, symbol=symbol, side=exit_side, quantity=desired,
                          source_signal_id=state["source_signal_id"], reason_code=reason)
            fee = fee_for(quantity * price)
            slippage = quantity * abs(price - float(raw_exit))
            fill = Fill(fill_id=_stable_id("fill", order.order_id, price, quantity),
                        order_id=order.order_id, timestamp=timestamp, symbol=symbol,
                        side=exit_side, quantity=quantity, price=price, fee=fee,
                        slippage_cost=slippage,
                        status="FULL" if quantity == desired else "PARTIAL")
            cash += (quantity * price - fee if state["side"] == "LONG"
                     else -quantity * price - fee)
            entry_fee = state["entry_fee"] * quantity / state["quantity"]
            gross_pnl = quantity * (price - state["avg_cost"]) * direction
            net_pnl = gross_pnl - entry_fee - fee
            cumulative_realized += net_pnl
            trades.append({"symbol": symbol, "source_signal_id": state["source_signal_id"],
                           "entry_timestamp": state["opened_at"], "exit_timestamp": timestamp,
                           "side": state["side"], "entry_price": state["avg_cost"], "exit_price": price,
                           "quantity": quantity, "gross_pnl": gross_pnl, "net_pnl": net_pnl,
                           "return": (price / state["avg_cost"] - 1) * direction,
                           "holding_period": state["holding_period"], "exit_reason": reason,
                           "corporate_actions": state["corporate_actions"]})
            state["quantity"] -= quantity
            state["entry_fee"] -= entry_fee
            state["realized_pnl"] += net_pnl
            orders.append(order)
            fills.append(fill)
            if state["quantity"] <= 0:
                del positions[symbol]
                exited_this_session.add(symbol)

        for candidate in sorted(scheduled.get(timestamp, []), key=lambda item: (item.rank, item.symbol)):
            bar = bars.get((timestamp, candidate.symbol))
            if candidate.symbol in exited_this_session:
                blocked_row(candidate, timestamp, "EXITED_THIS_SESSION")
                continue
            if candidate.symbol in positions:
                blocked_row(candidate, timestamp, "ALREADY_HELD")
                continue
            if len(positions) >= max_positions:
                blocked_row(candidate, timestamp, "NO_CAPACITY")
                continue
            if bar is None:
                blocked_row(candidate, timestamp, "NO_PRICE")
                continue
            if bar["suspended"]:
                blocked_row(candidate, timestamp, "SUSPENDED")
                continue
            entry_side = "BUY" if candidate.side == "LONG" else "SHORT"
            if ((entry_side == "BUY" and bar["limit_buy_locked"])
                    or (entry_side == "SHORT" and bar["limit_sell_locked"])):
                limit_reason = (bar["limit_state"]["buy_block_reason"] if entry_side == "BUY"
                                else bar["limit_state"]["sell_block_reason"]) or "LIMIT_LOCKED"
                blocked_row(candidate, timestamp, limit_reason)
                blocked[-1]["market_state_evidence"] = bar["limit_state"]
                continue
            raw_entry = base_price(bar, entry_side, entry=True)
            if not _finite(raw_entry) or raw_entry <= 0:
                blocked_row(candidate, timestamp, "NO_PRICE")
                continue
            price = fill_price(float(raw_entry), entry_side)
            max_gross = float(portfolio.get("max_gross_exposure", 1.0)) * initial_cash
            current_market = sum(state["quantity"] * state.get("last_price", state["avg_cost"])
                                 for state in positions.values())
            buying_power = cash if candidate.side == "LONG" else cash / short_margin_rate
            target_notional = min(initial_cash * weight,
                                  max(0.0, max_gross - current_market), buying_power)
            quantity = int(target_notional // price // lot_size) * lot_size
            while (candidate.side == "LONG" and quantity >= lot_size
                   and quantity * price + fee_for(quantity * price) > cash + 1e-9):
                quantity -= lot_size
            if quantity < lot_size:
                blocked_row(candidate, timestamp, "NO_CASH")
                continue
            desired = quantity
            quantity = _liquidity_quantity(desired, bar, execution, lot_size)
            if quantity < lot_size:
                blocked_row(candidate, timestamp,
                            "LIQUIDITY_DATA_MISSING" if execution.get("max_participation_rate") is not None
                            and bar.get("volume") is None else "LIQUIDITY")
                continue
            order = Order(order_id=_stable_id("order", resolved.strategy.id,
                                               candidate.signal_id, timestamp, entry_side),
                          timestamp=timestamp, symbol=candidate.symbol, side=entry_side,
                          quantity=desired, source_signal_id=candidate.signal_id,
                          reason_code="CANDIDATE_SIGNAL")
            fee = fee_for(quantity * price)
            slippage = quantity * abs(price - float(raw_entry))
            fill = Fill(fill_id=_stable_id("fill", order.order_id, price, quantity),
                        order_id=order.order_id, timestamp=timestamp, symbol=candidate.symbol,
                        side=entry_side, quantity=quantity, price=price, fee=fee,
                        slippage_cost=slippage,
                        status="FULL" if quantity == desired else "PARTIAL")
            cash += (-quantity * price - fee if candidate.side == "LONG"
                     else quantity * price - fee)
            positions[candidate.symbol] = {
                "symbol": candidate.symbol, "quantity": quantity, "avg_cost": price,
                "side": candidate.side,
                "opened_at": timestamp, "last_updated_at": timestamp,
                "unrealized_pnl": 0.0, "realized_pnl": 0.0, "holding_period": 1,
                "extreme_price": (max(price, bar.get("high") or price)
                                  if candidate.side == "LONG"
                                  else min(price, bar.get("low") or price)),
                "last_price": bar.get("close") or price, "entry_fee": fee,
                "source_signal_id": candidate.signal_id, "corporate_actions": [],
            }
            orders.append(order)
            fills.append(fill)
            # A next-open entry can encounter an intraday stop/take on its
            # entry session.  With T+1 it is evidence of a blocked exit, not a
            # trade that may be silently moved to another timestamp.
            if timing in {"next_open", "next_session_open"} and t_plus_one:
                state = positions[candidate.symbol]
                candidate_direction = 1 if candidate.side == "LONG" else -1
                same_day_stop = (state["avg_cost"] * (1 - candidate_direction * stop_rate)
                                 if stop_rate is not None else None)
                same_day_take = (state["avg_cost"] * (1 + candidate_direction * take_rate)
                                 if take_rate is not None else None)
                stop_hit = (same_day_stop is not None
                            and ((_finite(bar.get("low")) and bar["low"] <= same_day_stop)
                                 if candidate.side == "LONG" else
                                 (_finite(bar.get("high")) and bar["high"] >= same_day_stop)))
                take_hit = (same_day_take is not None
                            and ((_finite(bar.get("high")) and bar["high"] >= same_day_take)
                                 if candidate.side == "LONG" else
                                 (_finite(bar.get("low")) and bar["low"] <= same_day_take)))
                if stop_hit or take_hit:
                    blocked_exits.append({"symbol": candidate.symbol, "timestamp": timestamp,
                                          "reason": "T_PLUS_ONE",
                                          "exit_reason": "STOP_LOSS" if stop_hit else "TAKE_PROFIT"})

        market_value, gross_market_value, unrealized = 0.0, 0.0, 0.0
        for symbol, state in positions.items():
            bar = bars.get((timestamp, symbol))
            mark = bar.get("close") if bar and _finite(bar.get("close")) else state["last_price"]
            state["last_price"], state["last_updated_at"] = mark, timestamp
            position_direction = 1 if state["side"] == "LONG" else -1
            state["unrealized_pnl"] = (state["quantity"] * (mark - state["avg_cost"])
                                       * position_direction)
            signed_value = state["quantity"] * mark * position_direction
            market_value += signed_value
            gross_market_value += abs(signed_value)
            unrealized += state["unrealized_pnl"]
        nav = cash + market_value
        snapshots.append(AccountSnapshot(
            timestamp=timestamp, cash=cash, market_value=market_value, nav=nav,
            realized_pnl=cumulative_realized, unrealized_pnl=unrealized,
            gross_exposure=gross_market_value / nav if nav else 0.0,
            net_exposure=market_value / nav if nav else 0.0,
            position_count=len(positions)))

    nav = snapshots[-1].nav if snapshots else initial_cash
    navs = [snapshot.nav for snapshot in snapshots]
    peak, max_drawdown = initial_cash, 0.0
    for value in navs:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1 if peak else 0.0)
    returns = [right / left - 1 for left, right in zip(navs, navs[1:]) if left]
    volatility = pstdev(returns) * math.sqrt(252) if len(returns) > 1 else None
    sharpe = mean(returns) / pstdev(returns) * math.sqrt(252) \
        if len(returns) > 1 and pstdev(returns) else None
    downside = [min(value, 0.0) for value in returns]
    sortino = mean(returns) / pstdev(downside) * math.sqrt(252) \
        if len(downside) > 1 and pstdev(downside) else None
    total_return = nav / initial_cash - 1
    cagr = ((nav / initial_cash) ** (252 / max(1, len(navs) - 1)) - 1
            if len(navs) > 1 and nav > 0 else None)
    total_fees = sum(item.fee for item in fills)
    total_slippage = sum(item.slippage_cost for item in fills)
    winning = [item["net_pnl"] for item in trades if item["net_pnl"] > 0]
    losing = [item["net_pnl"] for item in trades if item["net_pnl"] < 0]
    gross_profit = sum(item["gross_pnl"] for item in trades if item["gross_pnl"] > 0)
    all_blocked = blocked + blocked_exits
    reason_counts = {reason: sum(item.get("reason") == reason for item in all_blocked)
                     for reason in ("NO_CAPACITY", "NO_CASH", "NO_PRICE", "ALREADY_HELD",
                                    "EXITED_THIS_SESSION", "SIGNAL_EXPIRED", "T_PLUS_ONE", "SUSPENDED",
                                    "LIMIT_LOCKED", "LIMIT_TOUCH_CONSERVATIVE", "LIQUIDITY",
                                    "LIQUIDITY_DATA_MISSING", "RISK_LIMIT",
                                    "CORPORATE_ACTION_CASH")}
    entry_fills = sum(item.side in {"BUY", "SHORT"} for item in fills)
    occupation = (mean(snapshot.position_count / max_positions for snapshot in snapshots)
                  if snapshots else 0.0)
    utilization = (mean(snapshot.gross_exposure for snapshot in snapshots if snapshot.nav)
                  if snapshots else 0.0)
    warnings = list(_warning_context(resolved))
    if candidates and entry_fills / len(candidates) < 0.2:
        warnings.append({"code": "LOW_FILL_RATE", "message": "fewer than 20% of signals produced entry fills"})
    if candidates and reason_counts["NO_CAPACITY"] / len(candidates) > 0.5:
        warnings.append({"code": "CAPACITY_DOMINATED", "message": "more than half of signals were blocked by capacity"})
    report = {
        "schema": "stock-strategy-report-v1",
        "skill_meta": {"skill_name": "strategy-backtest", "skill_version": "1.1.0"},
        "context_snapshot": _context(resolved),
        "strategy_spec_snapshot": resolved.strategy.to_dict(),
        "signal_source_snapshot": {"signal": resolved.signal.id,
                                   "candidate_count": len(candidates)},
        "event_order": ["corporate_action", "existing_position_exit", "scheduled_entry",
                        "mark_to_market"],
        "performance": {"total_return": total_return, "cagr": cagr,
                        "volatility": volatility, "sharpe": sharpe, "sortino": sortino,
                        "max_drawdown": max_drawdown,
                        "calmar": cagr / abs(max_drawdown) if cagr is not None and max_drawdown else None},
        "trade_statistics": {"trade_count": len(trades), "entry_fill_count": entry_fills,
                             "win_rate": len(winning) / len(trades) if trades else None,
                             "profit_factor": (sum(winning) / abs(sum(losing)) if losing else None),
                             "avg_trade_return": mean(item["return"] for item in trades) if trades else None,
                             "avg_holding_period": mean(item["holding_period"] for item in trades) if trades else None},
        "execution_statistics": {"signal_count": len(candidates), "order_count": len(orders),
                                 "fill_count": len(fills), "rejected_order_count": len(blocked),
                                 "blocked_exit_count": len(blocked_exits),
                                 "unfilled_count": len(blocked) + len(blocked_exits),
                                 "fill_rate": entry_fills / len(candidates) if candidates else 0.0,
                                 "missed_signal_rate": len(blocked) / len(candidates) if candidates else 0.0},
        "capacity_statistics": {"blocked_by_capacity": reason_counts["NO_CAPACITY"],
                                "blocked_by_cash": reason_counts["NO_CASH"],
                                "blocked_by_tradability": (reason_counts["NO_PRICE"]
                                                           + reason_counts["SUSPENDED"]),
                                "blocked_by_t1": reason_counts["T_PLUS_ONE"],
                                "blocked_by_limit_state": reason_counts["LIMIT_LOCKED"],
                                "blocked_by_liquidity": (reason_counts["LIQUIDITY"]
                                                          + reason_counts["LIQUIDITY_DATA_MISSING"]),
                                "position_occupation_ratio": occupation,
                                "cash_utilization": utilization},
        "cost_statistics": {"total_fees": total_fees,
                            "total_slippage_cost": total_slippage,
                            "cost_to_gross_profit_ratio": ((total_fees + total_slippage) / gross_profit
                                                           if gross_profit > 0 else None)},
        "curves": {"equity_curve_ref": "account_snapshots.json",
                   "drawdown_curve_ref": "account_snapshots.json",
                   "exposure_curve_ref": "account_snapshots.json"},
        "diagnostics": {"missed_alpha": None,
                        "captured_alpha": sum(item["net_pnl"] for item in trades),
                        "blocked_signal_analysis": reason_counts,
                        "market_state_derivation": {
                            "trading_status_rows": sum(
                                item["market_state_source"]["suspension"] == "trading_status"
                                for item in bars.values()),
                        "exchange_limit_field_rows": sum(
                                item["market_state_source"]["buy_limit"] == "shared_rules_or_provider_field"
                                or item["market_state_source"]["sell_limit"] == "shared_rules_or_provider_field"
                                for item in bars.values()),
                        "derived_price_limit_rows": sum(item["price_limit_rule_id"] is not None
                                                         for item in bars.values()),
                        "price_limit_rule_ids": sorted({item["price_limit_rule_id"]
                                                         for item in bars.values()
                                                         if item["price_limit_rule_id"]}),
                        "price_limit_rule_version": "1.0.0",
                        "limit_touch_policy": limit_touch_policy,
                        "ohlc_proves_book_locked": False,
                        "rows_with_unknown_book_state": sum(bool(item["limit_state"]["evidence_limitations"])
                                                             for item in bars.values()),
                        "upper_limit_touched_rows": sum(item["limit_state"]["upper_limit_touched"]
                                                         for item in bars.values()),
                        "lower_limit_touched_rows": sum(item["limit_state"]["lower_limit_touched"]
                                                         for item in bars.values()),
                        }},
        "warnings": warnings, "errors": []}
    open_positions = [Position(symbol=state["symbol"], quantity=state["quantity"],
                               avg_cost=state["avg_cost"], opened_at=state["opened_at"],
                               last_updated_at=state["last_updated_at"],
                               unrealized_pnl=state["unrealized_pnl"],
                               realized_pnl=state["realized_pnl"],
                               holding_period=state["holding_period"],
                               side=state["side"]).to_dict()
                      for state in sorted(positions.values(), key=lambda item: item["symbol"])]
    evidence = {"orders": [item.to_dict() for item in orders],
                "fills": [item.to_dict() for item in fills], "positions": open_positions,
                "account_snapshots": [item.to_dict() for item in snapshots],
                "trades": trades, "blocked_signals": blocked, "blocked_exits": blocked_exits}
    return report, evidence


_SCENARIO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SCENARIO_SECTIONS = {"portfolio", "entry", "exit", "execution", "accounting"}


def _scenario_resolved(resolved, scenario):
    if not isinstance(scenario, dict):
        raise ValueError("each robustness scenario must be an object")
    scenario_id = str(scenario.get("id", ""))
    if not _SCENARIO_ID.fullmatch(scenario_id):
        raise ValueError(f"invalid robustness scenario id: {scenario_id!r}")
    overrides = scenario.get("parameter_overrides", {})
    if not isinstance(overrides, dict) or not set(overrides).issubset(_SCENARIO_SECTIONS):
        raise ValueError(f"unsupported parameter_overrides in scenario: {scenario_id}")
    strategy_values = {}
    for section in _SCENARIO_SECTIONS:
        base = dict(getattr(resolved.strategy, section))
        section_overrides = overrides.get(section, {})
        if not isinstance(section_overrides, dict):
            raise ValueError(f"scenario override {section} must be an object: {scenario_id}")
        base.update(section_overrides)
        strategy_values[section] = base
    execution = strategy_values["execution"]
    cost_multiplier = float(scenario.get("cost_multiplier", 1.0))
    slippage_multiplier = float(scenario.get("slippage_multiplier", 1.0))
    if not _finite(cost_multiplier) or cost_multiplier < 0:
        raise ValueError(f"scenario cost_multiplier must be finite and non-negative: {scenario_id}")
    if not _finite(slippage_multiplier) or slippage_multiplier < 0:
        raise ValueError(f"scenario slippage_multiplier must be finite and non-negative: {scenario_id}")
    if "fee_bps" in execution:
        execution["fee_bps"] = float(execution["fee_bps"]) * cost_multiplier
    else:
        execution["round_trip_cost_bps"] = float(
            execution.get("round_trip_cost_bps", 0.0)) * cost_multiplier
    execution["slippage_bps"] = float(execution.get("slippage_bps", 0.0)) * slippage_multiplier
    start_offset = scenario.get("start_offset", 0)
    if isinstance(start_offset, bool):
        raise ValueError(f"scenario start_offset must be a non-negative integer: {scenario_id}")
    try:
        start_offset = int(start_offset)
    except (TypeError, ValueError):
        raise ValueError(f"scenario start_offset must be a non-negative integer: {scenario_id}") from None
    if start_offset < 0:
        raise ValueError(f"scenario start_offset must be a non-negative integer: {scenario_id}")
    strategy = replace(resolved.strategy, **strategy_values)
    return scenario_id, replace(resolved, strategy=strategy), start_offset, {
        "parameter_overrides": overrides,
        "start_offset": start_offset,
        "cost_multiplier": cost_multiplier,
        "slippage_multiplier": slippage_multiplier,
    }


def strategy_robustness(resolved: ResolvedExperiment, candidates, prices, scenarios,
                        corporate_actions=None, baseline_report=None,
                        trading_calendar=None) -> tuple[dict, dict]:
    """Run deterministic, isolated StrategySpec scenarios over the same inputs."""
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("robustness scenarios must be a non-empty array")
    bars = _market_bars(prices, trading_calendar=trading_calendar)
    calendar = sorted({timestamp for timestamp, _ in bars})
    if not calendar:
        raise ValueError("robustness scenarios require market bars")
    baseline_report = baseline_report or strategy_backtest(
        resolved, candidates, prices, corporate_actions,
        trading_calendar=trading_calendar)[0]
    prepared, ids = [], set()
    for scenario in scenarios:
        scenario_id, scenario_resolved, start_offset, normalized = _scenario_resolved(
            resolved, scenario)
        if scenario_id in ids:
            raise ValueError(f"duplicate robustness scenario id: {scenario_id}")
        if start_offset >= len(calendar):
            raise ValueError(f"scenario start_offset exceeds the market calendar: {scenario_id}")
        ids.add(scenario_id)
        prepared.append((scenario_id, scenario_resolved, start_offset, normalized))

    details, rows = {}, []
    baseline_return = baseline_report["performance"]["total_return"]
    for scenario_id, scenario_resolved, start_offset, normalized in sorted(prepared):
        first_timestamp = calendar[start_offset]
        scenario_candidates = [item for item in candidates if item.timestamp >= first_timestamp]
        scenario_prices = [row for row in prices if str(row.get("timestamp")) >= first_timestamp]
        scenario_actions = [row for row in corporate_actions or []
                            if str(row.get("timestamp") or row.get("ex_date") or row.get("exDate"))
                            >= first_timestamp]
        report, evidence = strategy_backtest(
            scenario_resolved, scenario_candidates, scenario_prices, scenario_actions,
            trading_calendar=trading_calendar)
        total_return = report["performance"]["total_return"]
        rows.append({
            "scenario_id": scenario_id, "scenario": normalized,
            "performance": report["performance"],
            "trade_statistics": report["trade_statistics"],
            "execution_statistics": report["execution_statistics"],
            "capacity_statistics": report["capacity_statistics"],
            "cost_statistics": report["cost_statistics"],
            "delta_total_return": total_return - baseline_return,
        })
        details[scenario_id] = {"strategy_report": report, "evidence": evidence}
    result = {
        "schema": "stock-strategy-robustness-v1",
        "skill_meta": {"skill_name": "strategy-robustness", "skill_version": "1.0.0"},
        "context_snapshot": _context(resolved),
        "strategy_id": resolved.strategy.id,
        "baseline": {"performance": baseline_report["performance"],
                     "trade_statistics": baseline_report["trade_statistics"],
                     "cost_statistics": baseline_report["cost_statistics"]},
        "scenario_count": len(rows), "scenarios": rows,
        "warnings": _warning_context(resolved), "errors": [],
    }
    return result, details


def run_experiment(experiment_id, input_payload: dict, *, config_root, stage="full", reports_root=None,
                   run_id=None, execution_engine="deterministic", as_of=None, dry_run=False) -> dict:
    if isinstance(input_payload, dict) and input_payload.get("schema") == "quant-project-four-layer-input-v2":
        from .four_layer import FourLayerResearchService
        from ..modeling.contracts import CapabilityError
        if execution_engine != "deterministic" or as_of is not None:
            raise CapabilityError("V2 runtime/windows must be preregistered; Qlib and as_of overrides are unsupported")
        return FourLayerResearchService(config_root=config_root, reports_root=reports_root).run(
            experiment_id, input_payload, run_id=run_id, stage=stage, dry_run=dry_run)
    if stage == "model":
        from .model_research import model_eval
        from ..modeling.contracts import CapabilityError
        if execution_engine != "deterministic" or as_of is not None:
            raise CapabilityError("model stage uses its preregistered runtime and context; execution/as_of overrides are unsupported")
        return model_eval(experiment_id, input_payload, config_root=config_root,
                          reports_root=reports_root, run_id=run_id, dry_run=dry_run)
    if dry_run:
        raise ValueError("dry-run requires an independent model input or V2 four-layer input")
    if execution_engine not in {"deterministic", "qlib"}:
        raise ValueError("execution_engine must be deterministic or qlib")
    """Research orchestration only; preview and live scan must never call this."""
    if stage not in {"factor", "signal", "strategy", "full"}:
        raise ValueError("stage must be factor, signal, strategy, or full")
    if not isinstance(input_payload, dict):
        raise TypeError("research input must be an object")
    if "execution_engine" in input_payload:
        raise ValueError("execution_engine is a run option, not a research input field")
    source_names = [name for name in ("scores", "factor_dataset", "candidate_signals")
                    if input_payload.get(name) is not None]
    if len(source_names) != 1:
        raise ValueError("research input must provide exactly one of scores, factor_dataset, or candidate_signals")
    if source_names[0] == "candidate_signals" and stage != "strategy":
        raise ValueError("candidate_signals input is valid only for the strategy stage")
    if stage in {"strategy", "full"} and not input_payload.get("prices"):
        raise ValueError("strategy execution requires at least one market bar")
    run_context = input_payload.get("context")
    if not isinstance(run_context, dict):
        raise ValueError("research input must provide a context object")
    if as_of is not None:
        as_of = date.fromisoformat(str(as_of)).isoformat()
        run_context = dict(run_context) | {"start": as_of, "end": as_of, "as_of": as_of}
    resolved = resolve_experiment(experiment_id, config_root, context=run_context)
    if not isinstance(resolved, ResolvedExperiment):
        raise ValueError("V2 combinations require quant-project-four-layer-input-v2; legacy scores/candidates cannot bypass the model")
    direct_candidates = input_payload.get("candidate_signals") is not None
    dependencies = resolved.dependencies_for_stage(stage, direct_candidates=direct_candidates)
    release_identity = _research_release_identity(resolved)
    direct_candidates = (_as_candidates(input_payload.get("candidate_signals"), resolved)
                         if input_payload.get("candidate_signals") is not None else None)
    needs_scores = stage in {"factor", "signal", "full"} or direct_candidates is None
    scores, factor_source = (_research_scores(resolved, input_payload) if needs_scores
                             else ([], {"kind": "candidate_signals"}))
    labels = input_payload.get("labels")
    groups = input_payload.get("groups")
    regimes = input_payload.get("regimes")
    window_rows = direct_candidates if direct_candidates is not None else scores
    (window_rows, labels, groups, regimes, evaluation_window,
     selected_dates) = _filter_research_window(
        window_rows, labels, groups, regimes, resolved.context,
        input_payload.get("session_window"))
    if direct_candidates is not None:
        direct_candidates = window_rows
    else:
        scores = window_rows
    regime_evidence = None
    if regimes is not None and input_payload.get("regime_source") is not None:
        raise ValueError("research input must choose explicit regimes or regime_source, not both")
    if input_payload.get("regime_source") is not None:
        regimes, regime_evidence = infer_regimes(input_payload["regime_source"])
        regimes = _observations_on_dates(regimes, selected_dates)
    if _has_industry_input(scores, groups):
        dependencies["industry"] = True
    if release_identity:
        from ..pit.capabilities import validate_release_dependencies
        validate_release_dependencies(release_identity, dependencies,
                                      data_mode=resolved.context.data_mode,
                                      as_of=resolved.context.as_of,
                                      as_of_policy=resolved.context.as_of_policy)
    if resolved.context.data_mode == "point_in_time" and dependencies["fundamentals"]:
        raise ValueError("the score-only research runner cannot consume fundamental PIT records directly; use a Factor adapter backed by FundamentalPITRepository")
    if resolved.context.data_mode == "point_in_time" and dependencies["industry"]:
        groups = _point_in_time_industry_groups(resolved, release_identity, scores, groups)
    industry_present = _has_industry_input(scores, groups)
    if dependencies["industry"] and stage in {"signal", "full"} and not industry_present:
        raise ValueError("declared industry dependency requires caller-supplied or PIT-resolved industry groups")
    actual_dependencies = dict(dependencies)
    if not industry_present:
        actual_dependencies["industry"] = False
    input_capabilities = _input_dataset_capabilities(scores, groups, input_payload.get("prices"))
    if release_identity:
        dataset_capabilities = release_identity["capabilities"]
        dataset_versions = release_identity["versions"]
        dataset_sources = release_identity["sources"]
        dataset_coverage = release_identity["coverage"]
        known_biases = list(release_identity["known_biases"])
    else:
        dataset_capabilities = input_capabilities
        dataset_versions = {"market_data": factor_source.get("dataset_id")}
        dataset_sources = {name: value["source"] for name, value in input_capabilities.items()
                           if value.get("source")}
        dataset_coverage = {name: value["coverage"] for name, value in input_capabilities.items()
                           if value.get("coverage")}
        known_biases = []
    if resolved.context.data_mode == "snapshot_compatible":
        if dependencies["fundamentals"]:
            known_biases.append("fundamentals_snapshot_has_no_historical_announcement_time")
        if dependencies["industry"]:
            known_biases.append("industry_inputs_have_no_verified_effective_time")
    runtime_metadata = dict(resolved.context.metadata)
    runtime_metadata.update({
        "dataset_capabilities": dataset_capabilities,
        "dataset_versions": dataset_versions,
        "dataset_sources": dataset_sources,
        "dataset_coverage": dataset_coverage,
        "known_biases": sorted(set(known_biases)),
        "dependencies": resolved.declared_dependencies,
        "actual_dependencies": actual_dependencies,
        "evaluation_window": evaluation_window,
    })
    resolved = replace(resolved, context=replace(resolved.context, metadata=runtime_metadata))
    compiled_plan = compile_experiment(resolved, stage=stage, direct_candidates=direct_candidates is not None)
    qlib_runtime_config = compile_qlib_runtime_config(resolved)
    if execution_engine == "qlib":
        compatibility = qlib_runtime_config["compatibility"]
        if not compatibility["supported"]:
            raise ValueError("Qlib execution is incompatible: " + "; ".join(compatibility["reasons"]))
        raise ValueError("Qlib execution is not connected to the layered research entry")
    reports_root = Path(reports_root or RUNS_ROOT / "experiments").resolve()
    run_id = run_id or uuid4().hex

    # Complete every deterministic calculation before creating the run directory.
    # Invalid input must not leave a directory that resembles a completed run.
    report_payloads, evidence_payloads, robustness_details = {}, {}, {}
    if stage in {"factor", "full"}:
        report_payloads["factor_report"] = factor_eval(
            resolved, scores, labels, groups=groups, regimes=regimes)[0]
    candidates = direct_candidates or []
    if stage in {"signal", "full"} or (stage == "strategy" and direct_candidates is None):
        report_payloads["signal_report"], candidates = signal_eval(
            resolved, scores, labels, groups=groups, regimes=regimes)
    if stage in {"strategy", "full"}:
        report_payloads["strategy_report"], evidence_payloads = strategy_backtest(
            resolved, candidates, input_payload.get("prices", []),
            input_payload.get("corporate_actions"),
            trading_calendar=input_payload.get("trading_calendar"))
        if input_payload.get("scenarios") is not None:
            report_payloads["robustness_report"], robustness_details = strategy_robustness(
                resolved, candidates, input_payload.get("prices", []),
                input_payload.get("scenarios"), input_payload.get("corporate_actions"),
                baseline_report=report_payloads["strategy_report"],
                trading_calendar=input_payload.get("trading_calendar"))

    resolved_path = persist_resolved_config(resolved, run_id=run_id, reports_root=reports_root)
    output = resolved_path.parent
    atomic_json(output / "compiled_plan.json", compiled_plan)
    atomic_json(output / "qlib_runtime_config.json", qlib_runtime_config)
    reports, artifacts = {}, {"resolved_config": "resolved_config.yaml",
                              "compiled_plan": "compiled_plan.json",
                              "qlib_runtime_config": "qlib_runtime_config.json"}
    if regime_evidence is not None:
        atomic_json(output / "regime_inference.json", regime_evidence)
        artifacts["regime_inference"] = "regime_inference.json"
    for name, payload in report_payloads.items():
        atomic_json(output / f"{name}.json", payload)
        reports[name] = f"{name}.json"
    if stage in {"signal", "strategy", "full"}:
        atomic_json(output / "candidate_signals.json", [item.to_dict() for item in candidates])
        artifacts["candidate_signals"] = "candidate_signals.json"
    if stage in {"strategy", "full"}:
        for name, value in evidence_payloads.items():
            atomic_json(output / f"{name}.json", value)
        artifacts.update({name: f"{name}.json" for name in evidence_payloads})
        for scenario_id, detail in sorted(robustness_details.items()):
            report_relative = f"robustness/{scenario_id}/strategy_report.json"
            atomic_json(output / report_relative, detail["strategy_report"])
            reports[f"robustness_{scenario_id}_strategy_report"] = report_relative
            for name, value in detail["evidence"].items():
                relative = f"robustness/{scenario_id}/{name}.json"
                atomic_json(output / relative, value)
                artifacts[f"robustness_{scenario_id}_{name}"] = relative
    summary = {"schema": "stock-backtest-summary-v1", "run_id": run_id, "experiment_id": experiment_id,
               "experiment_name": resolved.experiment.metadata.get("name"),
               "stage": stage, "status": "success", "context_snapshot": _context(resolved),
               "evaluation_window": evaluation_window,
               "execution": {
                   "requested_engine": execution_engine,
                   "actual_engine": ("deterministic_event_v1" if stage in {"strategy", "full"}
                                     else "deterministic_evaluation_v1"),
                   "qlib_executed": False,
                   "qlib_runtime_config_role": "compatibility_plan",
                   "qlib_compatibility": qlib_runtime_config["compatibility"],
               },
               "data_access": {
                   "data_mode": resolved.context.data_mode,
                   "as_of": resolved.context.as_of or resolved.context.end,
                   "as_of_policy": resolved.context.as_of_policy,
                   "research_release_id": resolved.context.research_release,
                   "capabilities": dataset_capabilities,
                   "versions": dataset_versions,
                   "sources": dataset_sources,
                   "coverage": dataset_coverage,
                   "known_biases": sorted(set(known_biases)),
                   "dependencies": resolved.declared_dependencies,
                   "actual_dependencies": actual_dependencies,
               },
               "input_sources": {
                   "factor_scores": factor_source if direct_candidates is None else None,
                   "candidate_signals": ({"kind": "inline_candidate_signals",
                                          "count": len(direct_candidates)}
                                         if direct_candidates is not None else None),
                   "regimes": ({"kind": "inferred", "method": regime_evidence["method"],
                                "benchmark_id": regime_evidence["source"]["benchmark_id"]}
                               if regime_evidence is not None else
                               ({"kind": "explicit"} if regimes is not None else None)),
                   "research_release": release_identity},
               "reports": reports, "run_evidence": artifacts,
                "warnings": _warning_context(resolved),
               "note": "summary only; detailed statistics remain in referenced stage reports"}
    atomic_json(output / "summary.json", summary)
    registry = register_research_run(output, reports_root)
    return {"output": str(output), "summary": summary, "registry": registry}
