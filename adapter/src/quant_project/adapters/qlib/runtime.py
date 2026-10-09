"""Compile domain configuration into a deterministic Qlib runtime adapter.

The generic compiler intentionally remains independent of Qlib.  This module
is the boundary that records the installed runtime, translates the current
long-only layered strategy into the project's Qlib-native objects, and fails
closed for features that the native adapter cannot execute faithfully.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ...configer.compiler import compile_experiment
from ...configer.models import ResolvedExperiment
from ...paths import PROJECT_ROOT
from ...price_limits import (
    PriceLimitRuleError,
    assess_limit_market_state,
    derive_price_limit_rows,
)


SCHEMA = "quant-project-qlib-runtime-v1"
LOCK_SCHEMA = "quant-project-qlib-runtime-lock-v1"
DEFAULT_LOCK = PROJECT_ROOT / "configs" / "runtime" / "qlib-research-lock.json"


class QlibRuntimeError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_sha256() -> str:
    paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "quant_project" / "configer" / "compiler.py",
        PROJECT_ROOT / "src" / "quant_project" / "execution" / "qlib_native.py",
        PROJECT_ROOT / "src" / "quant_project" / "price_limits.py",
    ]
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_lock(path: Path) -> dict:
    if not path.is_file():
        raise QlibRuntimeError(f"Qlib dependency lock does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QlibRuntimeError(f"Qlib dependency lock is unreadable: {path}") from error
    if payload.get("schema") != LOCK_SCHEMA or not isinstance(payload.get("packages"), dict):
        raise QlibRuntimeError("Qlib dependency lock has an unsupported shape")
    actual_python = ".".join(str(item) for item in sys.version_info[:3])
    if payload.get("python") != actual_python:
        raise QlibRuntimeError(
            f"Qlib Python version differs from the lock: {actual_python} != {payload.get('python')}"
        )
    mismatches = []
    for package, expected in sorted(payload["packages"].items()):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            mismatches.append(f"{package}={actual!r} (locked {expected!r})")
    if mismatches:
        raise QlibRuntimeError("Qlib dependency lock mismatch: " + "; ".join(mismatches))
    return payload


def _rate_percent(value: Any, *, negative: bool = False) -> float | None:
    if value is None:
        return None
    number = float(value)
    percent = number if abs(number) > 1 else number * 100
    return -abs(percent) if negative else abs(percent)


def _execution_config(resolved: ResolvedExperiment) -> dict:
    portfolio = resolved.strategy.portfolio
    entry = resolved.strategy.entry
    exit_spec = resolved.strategy.exit
    execution = resolved.strategy.execution
    timing = str(entry.get("timing", "next_session_open"))
    timing = {"next_open": "next_session_open", "next_close": "next_session_close"}.get(
        timing, timing
    )
    participation = execution.get("max_participation_rate",
                                 execution.get("max_volume_participation"))
    signal_exit = exit_spec.get("signal_exit", False)
    if isinstance(signal_exit, dict):
        signal_exit = bool(signal_exit.get("enabled", True))
    rank_exit = exit_spec.get("rank_exit")
    if isinstance(rank_exit, dict):
        rank_exit = rank_exit.get("max_rank")
    if rank_exit is None or rank_exit is False:
        rank_exit = None
    else:
        if isinstance(rank_exit, bool):
            raise QlibRuntimeError("rank_exit requires a positive integer max_rank")
        try:
            converted_rank = int(rank_exit)
        except (TypeError, ValueError):
            raise QlibRuntimeError("rank_exit requires a positive integer max_rank") from None
        if isinstance(rank_exit, float) and not rank_exit.is_integer():
            raise QlibRuntimeError("rank_exit requires a positive integer max_rank")
        rank_exit = converted_rank
    return {
        "entry": timing,
        "stopPct": _rate_percent(
            exit_spec.get("stop_loss_pct", exit_spec.get("stop_loss")), negative=True
        ),
        "takePct": _rate_percent(
            exit_spec.get("take_profit_pct", exit_spec.get("take_profit"))
        ),
        "trailingPct": _rate_percent(
            exit_spec.get("trailing_stop_pct", exit_spec.get("trailing_stop"))
        ),
        "holdDays": int(exit_spec.get("max_holding_days", 1)),
        "signalExit": bool(signal_exit),
        "rankExit": rank_exit,
        "roundTripCostBps": float(execution.get("round_trip_cost_bps", 0.0)),
        "minimumFee": float(execution.get("minimum_fee", 0.0)),
        "fixedSlippageBps": float(execution.get("slippage_bps", 0.0)),
        "maxParticipationRate": float(participation) if participation is not None else None,
        "partialFillPolicy": str(execution.get("partial_fill_policy", "allow")),
        "limitTouchPolicy": str(execution.get("limit_touch_policy", "block_on_touch")),
        "slotWeight": float(portfolio.get(
            "per_position_weight", 1 / int(portfolio.get("max_positions", 1))
        )),
        "slotCount": int(portfolio.get("max_positions", 1)),
        "initialCapital": float(portfolio.get("initial_capital", 1_000_000.0)),
        "maxGrossExposure": float(portfolio.get("max_gross_exposure", 1.0)),
        "lotSize": int(execution.get("lot_size", 1)),
        "allowSameDayExit": not bool(execution.get("t_plus_one", False)),
    }


def _compatibility(resolved: ResolvedExperiment, execution: dict) -> dict:
    reasons = []
    side = str(resolved.signal.direction.get("side", "long")).upper()
    if side != "LONG":
        reasons.append("Qlib MarketFishingStrategy currently accepts LONG signals only")
    if execution["entry"] not in {"next_session_open", "next_session_close"}:
        reasons.append("Qlib adapter requires next-session open or close entry")
    if execution["rankExit"] is not None and execution["rankExit"] <= 0:
        reasons.append("rank_exit.max_rank must be positive")
    if execution["holdDays"] <= 0 or execution["slotCount"] <= 0:
        reasons.append("holding days and slot count must be positive")
    if execution["initialCapital"] <= 0 or execution["maxGrossExposure"] <= 0:
        reasons.append("initial capital and max gross exposure must be positive")
    if not 0 < execution["slotWeight"] <= 1:
        reasons.append("slot weight must be in (0, 1]")
    if execution["lotSize"] <= 0:
        reasons.append("lot size must be positive")
    if execution["fixedSlippageBps"] < 0:
        reasons.append("fixed slippage bps must be non-negative")
    if (execution["maxParticipationRate"] is not None
            and not 0 <= execution["maxParticipationRate"] <= 1):
        reasons.append("max_participation_rate must be in [0, 1]")
    if execution["partialFillPolicy"] not in {"allow", "reject"}:
        reasons.append("partial_fill_policy must be allow or reject")
    if execution["limitTouchPolicy"] not in {"block_on_touch", "allow_on_touch"}:
        reasons.append("limit_touch_policy must be block_on_touch or allow_on_touch")
    if execution["allowSameDayExit"]:
        reasons.append("Qlib adapter supports the T+1 LONG contract only")
    return {"supported": not reasons, "reasons": reasons}


def compile_qlib_runtime_config(
        resolved: ResolvedExperiment, *, dependency_lock: str | Path | None = None) -> dict:
    """Create the deterministic, public Qlib runtime configuration."""
    lock_path = Path(dependency_lock or DEFAULT_LOCK).resolve()
    lock = _runtime_lock(lock_path)
    generic = compile_experiment(resolved)
    execution = _execution_config(resolved)
    compatibility = _compatibility(resolved, execution)
    payload = {
        "schema": SCHEMA,
        "experiment_id": resolved.experiment.id,
        "resolved_config_sha256": resolved.sha256,
        "compiled_plan_sha256": generic["compiled_plan_sha256"],
        "adapter": "quant_project.adapters.qlib.v1",
        "runtime": {
            "python_version": lock["python"],
            "qlib_version": lock["packages"]["pyqlib"],
            "dependency_lock": {
                "path": lock_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256(lock_path),
            },
            "implementation_sha256": _implementation_sha256(),
        },
        "factor": generic["factor"],
        "signal": generic["signal"] | {"direction": resolved.signal.direction},
        "strategy": {
            "class": "MarketFishingStrategy",
            "module_path": "quant_project.execution.qlib_native",
            "kwargs": {"execution": execution},
        },
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
                "trade_type": "serial",
                "settle_type": "no_settlement",
            },
        },
        "exchange": {
            "class": "RawDailyExchange",
            "module_path": "quant_project.execution.qlib_native",
            "kwargs": {
                "frequency": "day",
                "deal_price": ["$raw_open", "$raw_close"],
                "adjustment_policy": "none",
                "fixed_slippage_bps": execution["fixedSlippageBps"],
                "impact_cost": 0.0,
                "volume_threshold": ({
                    "type": "participation_rate",
                    "rate": execution["maxParticipationRate"],
                    "partial_fill_policy": execution["partialFillPolicy"],
                    "lot_size": execution["lotSize"],
                } if execution["maxParticipationRate"] is not None else None),
                "price_limit_rule_set": "cn-a-mainboard-price-limits@1.0.0",
                "limit_touch_policy": execution["limitTouchPolicy"],
            },
        },
        "context": resolved.context.to_dict(),
        "compatibility": compatibility,
    }
    payload["qlib_runtime_config_sha256"] = hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()
    return payload


def _candidate_field(candidate: Any, name: str):
    return candidate.get(name) if isinstance(candidate, dict) else getattr(candidate, name, None)


def _action_event(row: dict) -> dict:
    timestamp = row.get("timestamp") or row.get("ex_date") or row.get("exDate")
    if not timestamp:
        raise QlibRuntimeError("corporate action requires timestamp/ex_date")
    return {
        "exDate": str(timestamp),
        "cashDividendPer10": float(row.get(
            "cashDividendPer10", float(row.get("cash_dividend_per_share", 0.0)) * 10
        )),
        "bonusSharePer10": float(row.get(
            "bonusSharePer10", float(row.get("bonus_share_ratio", 0.0)) * 10
        )),
        "rightsSharePer10": float(row.get(
            "rightsSharePer10", float(row.get("rights_share_ratio", 0.0)) * 10
        )),
        "rightsPrice": float(row.get("rightsPrice", row.get("rights_price", 0.0))),
    }


def _canonical_market_symbol(value: Any) -> str:
    raw = str(value).strip().upper()
    match = re.fullmatch(
        r"(?:(?:SH|SZ|SSE|SZSE|XSHG|XSHE)\.?(\d{6})|(\d{6})(?:\.(?:SH|SZ|SSE|SZSE|XSHG|XSHE))?)",
        raw,
    )
    return next((item for item in match.groups() if item), raw) if match else raw


def _build_events(candidates: Iterable[Any], prices: list[dict],
                  corporate_actions: list[dict] | None, execution: dict,
                  trading_calendar=None) -> tuple[list[dict], list[str]]:
    candidate_rows = list(candidates)
    # Reject unsupported direction before market-data compilation so a SHORT
    # request always reports the compatibility failure directly.
    for candidate in candidate_rows:
        side = str(_candidate_field(candidate, "side") or "LONG").upper()
        if side != "LONG":
            raise QlibRuntimeError("Qlib MarketFishingStrategy cannot execute SHORT CandidateSignal")
    bars: dict[str, list[dict]] = {}
    market_dates = set()
    action_rows = list(corporate_actions or [])
    action_keys = set()
    for action in action_rows:
        action_day = action.get("timestamp") or action.get("ex_date") or action.get("exDate")
        action_symbol = action.get("symbol") or action.get("code") or action.get("instrument")
        if action_day is None or action_symbol is None:
            raise QlibRuntimeError("corporate action timestamp and symbol are required")
        action_keys.add((str(action_day)[:10], _canonical_market_symbol(action_symbol)))
    limit_input_rows = []
    for raw in prices:
        row = dict(raw)
        row_day = str(row.get("timestamp", row.get("date", "")))[:10]
        row_symbol = row.get("symbol", row.get("code"))
        if row_symbol is not None and (row_day, _canonical_market_symbol(row_symbol)) in action_keys:
            row["corporate_action"] = True
        limit_input_rows.append(row)
    try:
        derived_prices = derive_price_limit_rows(limit_input_rows, trading_calendar=trading_calendar)
    except PriceLimitRuleError as error:
        raise QlibRuntimeError(f"Qlib runtime price-limit derivation failed: {error}") from error
    for row in derived_prices:
        timestamp = str(row.get("timestamp", row.get("date", "")))
        symbol = str(row.get("symbol", row.get("code", "")))
        if not timestamp or not symbol:
            raise QlibRuntimeError("every Qlib market bar requires timestamp and symbol")
        status = str(row.get("trade_status", "")).strip().upper()
        suspended = status in {"SUSPENDED", "HALTED", "PAUSED", "停牌", "SUSPEND"}

        def optional_price(name):
            try:
                value = float(row.get(name))
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) and value > 0 else None

        opening, high, low, close = [optional_price(name)
                                     for name in ("open", "high", "low", "close")]
        if not suspended and any(value is None for value in (opening, high, low, close)):
            raise QlibRuntimeError(f"Qlib market bar has missing or invalid OHLC: {timestamp}/{symbol}")
        if (not suspended and
                (high < max(opening, low, close) or low > min(opening, high, close))):
            raise QlibRuntimeError(f"Qlib market bar has inconsistent OHLC: {timestamp}/{symbol}")
        market_state = assess_limit_market_state(row, touch_policy=execution["limitTouchPolicy"])
        try:
            volume = float(row["volume"]) if row.get("volume") is not None else None
        except (TypeError, ValueError):
            volume = None
        if volume is not None and not math.isfinite(volume):
            volume = None
        bar = {"d": timestamp, "o": opening, "h": high, "l": low, "c": close,
               "volume": volume,
               "suspended": suspended, "limitState": market_state,
               "upperLimitPrice": row.get("upper_limit_price"),
               "lowerLimitPrice": row.get("lower_limit_price"),
               "priceLimitUnrestricted": row["price_limit_unrestricted"],
               "priceLimitRuleId": row["price_limit_rule_id"],
               "priceLimitRuleVersion": row["price_limit_rule_version"],
               "priceLimitBasis": row["price_limit_basis"]}
        bars.setdefault(symbol, []).append(bar)
        market_dates.add(timestamp)
    for symbol in bars:
        bars[symbol].sort(key=lambda item: item["d"])
    actions: dict[str, list[dict]] = {}
    for row in action_rows:
        symbol = str(row.get("symbol", ""))
        if not symbol:
            raise QlibRuntimeError("every corporate action requires symbol")
        actions.setdefault(symbol, []).append(_action_event(row))
    events = []
    for candidate in candidate_rows:
        side = str(_candidate_field(candidate, "side") or "LONG").upper()
        if side != "LONG":
            raise QlibRuntimeError("Qlib MarketFishingStrategy cannot execute SHORT CandidateSignal")
        timestamp = str(_candidate_field(candidate, "timestamp") or "")
        symbol = str(_candidate_field(candidate, "symbol") or "")
        if not timestamp or not symbol:
            raise QlibRuntimeError("every CandidateSignal requires timestamp and symbol")
        future = [row for row in bars.get(symbol, []) if row["d"] > timestamp]
        events.append({
            "code": symbol,
            "date": timestamp,
            "signalId": str(_candidate_field(candidate, "signal_id") or ""),
            "factorScore": float(_candidate_field(candidate, "score")),
            "dailyRank": int(_candidate_field(candidate, "rank")),
            "side": side,
               "future": future,
            "corporateActions": [row for row in actions.get(symbol, []) if row["exDate"] > timestamp],
        })
    return events, sorted(market_dates)


@dataclass
class QlibRuntimeObjects:
    """Infrastructure-independent Qlib objects plus a real execution entry."""

    config: dict
    factor: Any
    strategy: Any
    events: list[dict]
    market_dates: list[str]
    execution: dict
    initial_cash: float

    def run(self, *, exchange=None):
        """Bind Exchange/Account/Executor and run Qlib's backtest loop."""
        from ...execution.qlib_native import run_native_backtest

        return run_native_backtest(
            self.events,
            self.market_dates,
            self.execution,
            initial_cash=self.initial_cash,
            exchange=exchange,
        )


def build_qlib_runtime_objects(
        resolved: ResolvedExperiment, candidates: Iterable[Any],
        prices: list[dict], corporate_actions: list[dict] | None = None,
        *, dependency_lock: str | Path | None = None,
        trading_calendar=None) -> QlibRuntimeObjects:
    """Build a real Qlib strategy object from standard CandidateSignal input."""
    config = compile_qlib_runtime_config(resolved, dependency_lock=dependency_lock)
    if not config["compatibility"]["supported"]:
        raise QlibRuntimeError("Qlib runtime is incompatible: " + "; ".join(
            config["compatibility"]["reasons"]
        ))
    execution = config["strategy"]["kwargs"]["execution"]
    events, market_dates = _build_events(candidates, prices, corporate_actions,
                                         execution, trading_calendar=trading_calendar)
    if not market_dates:
        raise QlibRuntimeError("Qlib runtime requires at least one market date")
    engine = config["factor"]["engine"]
    if engine["kind"] == "python_callable":
        module, _, attribute = engine["implementation"].rpartition(".")
        factor = getattr(importlib.import_module(module), attribute)
    else:
        factor = engine["expression"]
    from ...execution.qlib_native import MarketFishingStrategy

    strategy = MarketFishingStrategy(events, market_dates, execution)
    return QlibRuntimeObjects(
        config=config,
        factor=factor,
        strategy=strategy,
        events=events,
        market_dates=market_dates,
        execution=execution,
        initial_cash=float(resolved.strategy.portfolio.get("initial_capital", 1_000_000.0)),
    )
