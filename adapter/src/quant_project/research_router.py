"""Fail-closed natural-language routing for the layered research entry."""
from __future__ import annotations

import re

EXPERIMENT = re.compile(r"\bEXP[0-9]+_V[1-9][0-9]*\b", re.I)


class ResearchRouteError(ValueError):
    pass


def route_research_request(request: str) -> dict:
    text = str(request or "").strip()
    if not text:
        raise ResearchRouteError("research request cannot be empty")
    experiment_ids = {value.upper() for value in EXPERIMENT.findall(text)}
    if len(experiment_ids) != 1:
        raise ResearchRouteError("research request must contain exactly one Experiment ID")
    lowered = text.lower()
    full = any(token in lowered for token in ("完整", "全流程", "full", "end-to-end", "end to end"))
    matches = {
        "factor": any(token in lowered for token in ("因子", "factor")),
        "model": any(token in lowered for token in ("模型", "model")),
        "signal": any(token in lowered for token in ("信号", "signal")),
        "strategy": any(token in lowered for token in ("策略", "strategy", "资金回测")),
    }
    selected = [stage for stage, matched in matches.items() if matched]
    if full:
        stage = "full"
    elif len(selected) == 1:
        stage = selected[0]
    elif not selected and any(token in lowered for token in ("回测", "研究", "backtest", "research")):
        stage = "full"
    else:
        raise ResearchRouteError(
            "research request is ambiguous; specify factor, model, signal, strategy, or full")
    return {"schema": "quant-project-research-route-v1", "request": text,
            "experiment_id": next(iter(experiment_ids)), "stage": stage,
            "execution_engine": "qlib" if "qlib" in lowered else "deterministic",
            "matched_intents": (["full"] if full else selected), "status": "resolved"}
