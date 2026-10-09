"""Metrics on sealed scores; evaluation labels never enter predict batches."""
from __future__ import annotations

import math
from statistics import mean, pstdev

from ..modeling.contracts import DataContractError


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result, left = [0.0] * len(values), 0
    while left < len(order):
        right = left + 1
        while right < len(order) and values[order[right]] == values[order[left]]:
            right += 1
        for index in order[left:right]:
            result[index] = (left + 1 + right) / 2
        left = right
    return result


def correlation(left, right):
    if len(left) < 2:
        return None
    lm, rm = mean(left), mean(right)
    denominator = math.sqrt(sum((v - lm) ** 2 for v in left) * sum((v - rm) ** 2 for v in right))
    return sum((x - lm) * (y - rm) for x, y in zip(left, right)) / denominator if denominator else None


def score_metrics(rows, labels, metric):
    pairs = [(float(row["score"]), float(labels[(row["timestamp"], row["symbol"])]))
             for row in rows if (row["timestamp"], row["symbol"]) in labels]
    predictions, targets = [item[0] for item in pairs], [item[1] for item in pairs]
    mse = mean((p - y) ** 2 for p, y in pairs) if pairs else None
    daily = {}
    for row in rows:
        key = row["timestamp"], row["symbol"]
        if key in labels:
            daily.setdefault(row["timestamp"], []).append((float(row["score"]), float(labels[key])))
    ics = []
    for values in daily.values():
        value = correlation(ranks([p for p, _ in values]), ranks([y for _, y in values]))
        if value is not None:
            ics.append(value)
    rank_ic = mean(ics) if ics else None
    log_loss = None
    if metric == "log_loss" and pairs:
        if any(y not in (0.0, 1.0) or not 0 <= p <= 1 for p, y in pairs):
            raise DataContractError("log_loss requires binary labels and probability scores")
        log_loss = -mean(y * math.log(max(1e-15, min(1 - 1e-15, p))) +
                         (1 - y) * math.log(max(1e-15, min(1 - 1e-15, 1 - p))) for p, y in pairs)
    return {"rows": len(rows), "evaluated_rows": len(pairs), "rank_ic": rank_ic, "mse": mse,
            "log_loss": log_loss, "rank_ic_std": pstdev(ics) if ics else None,
            "daily_rank_ic_observations": len(ics)}
