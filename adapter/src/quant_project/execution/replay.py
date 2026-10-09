"""Independent replay of the documented T+1, gap, cost and two-slot rules.

Future bars are execution/label inputs only. They never enter factor loading.
This daily-bar simulator intentionally keeps its documented limitations.
"""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP


def sequential_sum(values):
    """Keep deterministic left-to-right sums without importing factor code."""
    total = 0
    for value in values:
        total += value
    return total

DEFAULT_EXECUTION = {"stopPct": -3, "takePct": 4, "holdDays": 7, "roundTripCostBps": 20, "slotWeight": 0.30, "slotCount": 2}


def rounded(value, places):
    if not math.isfinite(value):
        return None
    return float(Decimal.from_float(float(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def simulate_trade(event, execution=None):
    c = DEFAULT_EXECUTION | (execution or {})
    future = event.get("future", [])[:c["holdDays"]]
    if len(future) < 2:
        raise ValueError("信号缺少 T+1 之后的行情")
    entry = future[0]
    price = entry["o"]
    stop = (price * (1 + c["stopPct"] / 100)
            if c.get("stopPct") is not None else None)
    take = (price * (1 + c["takePct"] / 100)
            if c.get("takePct") is not None else None)
    cost = c["roundTripCostBps"] / 100
    path = [{"date": entry["d"], "netMarkPct": (entry["c"] / price - 1) * 100 - cost / 2}]
    exit_price, reason, date = None, None, None
    for i, bar in enumerate(future[1:], 1):
        if stop is not None and bar["o"] <= stop:
            exit_price, reason = bar["o"], "跳空止损"
        elif take is not None and bar["o"] >= take:
            exit_price, reason = bar["o"], "跳空止盈"
        elif stop is not None and take is not None and bar["l"] <= stop and bar["h"] >= take:
            exit_price, reason = stop, "同日双触发按止损"
        elif stop is not None and bar["l"] <= stop:
            exit_price, reason = stop, "止损"
        elif take is not None and bar["h"] >= take:
            exit_price, reason = take, "止盈"
        elif i == len(future) - 1:
            exit_price, reason = bar["c"], "时间"
        if exit_price is not None:
            date = bar["d"]
            path.append({"date": date, "netMarkPct": (exit_price / price - 1) * 100 - cost})
            break
        path.append({"date": bar["d"], "netMarkPct": (bar["c"] / price - 1) * 100 - cost / 2})
    result = {k: event[k] for k in ("code", "name", "industry", "factorScore", "dailyRank") if k in event}
    return result | {"signalDate": event["date"], "entryDate": entry["d"], "exitDate": date,
                     "entryPrice": rounded(price, 4), "exitPrice": rounded(exit_price, 4), "exitReason": reason,
                     "netPct": rounded((exit_price / price - 1) * 100 - cost, 4), "path": path}


def summarize(trades):
    if not trades:
        return {"n": 0, "winRate": 0, "mean": 0, "median": 0, "profitFactor": 0}
    values = sorted(t["netPct"] for t in trades)
    wins, losses = [v for v in values if v > 0], [v for v in values if v <= 0]
    n, middle = len(values), len(values) // 2
    factor = sequential_sum(wins) / abs(sequential_sum(losses)) if losses and sequential_sum(losses) != 0 else math.inf
    return {"n": n, "winRate": rounded(len(wins) / n * 100, 2), "mean": rounded(sequential_sum(values) / n, 4),
            "median": rounded(values[middle] if n % 2 else (values[middle - 1] + values[middle]) / 2, 4),
            "avgWin": rounded(sequential_sum(wins) / len(wins), 4) if wins else 0,
            "avgLoss": rounded(sequential_sum(losses) / len(losses), 4) if losses else 0,
            "profitFactor": rounded(factor, 3) if losses else None}


def max_drawdown(curve):
    peak, dd = -math.inf, 0
    for row in curve:
        peak = max(peak, row["equity"])
        dd = min(dd, row["equity"] / peak - 1)
    return dd * 100


def curve_stats(curve):
    if len(curve) < 2:
        return {"finalEquity": 1, "totalReturn": 0, "cagr": 0, "maxDrawdown": 0, "sharpe": 0}
    daily = [b["equity"] / a["equity"] - 1 for a, b in zip(curve, curve[1:])]
    avg = sequential_sum(daily) / len(daily)
    variance = sequential_sum((v - avg) ** 2 for v in daily) / max(len(daily) - 1, 1)
    final, years = curve[-1]["equity"], max(len(daily) / 244, 1 / 244)
    return {"finalEquity": rounded(final, 4), "totalReturn": rounded((final - 1) * 100, 2),
            "cagr": rounded((final ** (1 / years) - 1) * 100, 2), "maxDrawdown": rounded(max_drawdown(curve), 2),
            "sharpe": rounded(avg / math.sqrt(variance) * math.sqrt(244), 3) if variance > 0 else 0}


def build_portfolio(events, market_dates, execution=None):
    c = DEFAULT_EXECUTION | (execution or {})
    candidates = sorted([simulate_trade(event, c) for event in events], key=lambda t: (t["entryDate"], t["dailyRank"], t["code"]))
    slots, skipped = [[] for _ in range(c["slotCount"])], []
    for trade in candidates:
        if any(t["entryDate"] <= trade["entryDate"] <= t["exitDate"] and t["code"] == trade["code"] for slot in slots for t in slot):
            skipped.append(trade | {"skipped": "同票仍在持仓"})
            continue
        slot = next((s for s in slots if not s or s[-1]["exitDate"] < trade["entryDate"]), None)
        if slot is None:
            skipped.append(trade | {"skipped": f"{c['slotCount']}个仓位均占用"})
        else:
            slot.append(trade)
    dates = sorted(market_dates)
    slot_curves = []
    for schedule in slots:
        values, capital = {}, c["slotWeight"]
        for trade in schedule:
            for mark in trade["path"]:
                values[mark["date"]] = capital * (1 + mark["netMarkPct"] / 100)
            capital *= 1 + trade["netPct"] / 100
        exits, cash, curve = {t["exitDate"]: t for t in schedule}, c["slotWeight"], []
        for date in dates:
            curve.append(values.get(date, cash))
            if date in exits:
                cash *= 1 + exits[date]["netPct"] / 100
        slot_curves.append(curve)
    cash_weight = 1 - c["slotCount"] * c["slotWeight"]
    curve = [{"date": date, "equity": rounded(cash_weight + sequential_sum(s[i] for s in slot_curves), 8)} for i, date in enumerate(dates)]
    accepted = sorted([t for s in slots for t in s], key=lambda t: t["entryDate"])
    years, previous = [], 1
    for year in sorted({date[:4] for date in dates}):
        year_curve = [r for r in curve if r["date"].startswith(year)]
        end = year_curve[-1]["equity"] if year_curve else previous
        years.append({"year": year, "portfolioReturn": rounded((end / previous - 1) * 100, 2),
                      "maxDrawdown": rounded(max_drawdown([{"date": year + "-00-00", "equity": previous}] + year_curve), 2) if year_curve else 0,
                      "trades": summarize([t for t in accepted if t["entryDate"].startswith(year)])})
        previous = end
    reasons = {}
    for t in accepted:
        reasons[t["exitReason"]] = reasons.get(t["exitReason"], 0) + 1
    return {"allocation": f"{c['slotCount']}个独立仓位×{math.floor(c['slotWeight'] * 100 + 0.5)}%，其余现金",
            "selectedSignals": len(events), "enteredTrades": len(accepted), "skippedSignals": len(skipped), "skipped": skipped,
            "tradeSummary": summarize(accepted), "portfolio": curve_stats(curve), "byYear": years, "byExitReason": reasons,
            "trades": accepted, "curve": curve}
