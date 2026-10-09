"""Separate immutable objects for the research pipeline hand-offs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping
from ..modeling.contracts import ModelScore


@dataclass(frozen=True)
class FactorValue:
    timestamp: str
    symbol: str
    factor_id: str
    value: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FactorVector:
    factor_vector_id: str
    timestamp: str
    symbol: str
    factor_set_id: str
    factor_set_config_sha256: str
    values: Mapping[str, float]

    def to_dict(self):
        return {"factor_vector_id": self.factor_vector_id, "timestamp": self.timestamp,
                "symbol": self.symbol, "factor_set_id": self.factor_set_id,
                "factor_set_config_sha256": self.factor_set_config_sha256, "values": dict(self.values)}


@dataclass(frozen=True)
class FactorScore:
    timestamp: str
    symbol: str
    score: float
    source_factor_id: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CandidateSignal:
    signal_id: str
    timestamp: str
    symbol: str
    score: float
    rank: int
    percentile: float
    side: str
    source_factor_id: str
    source_factor_score_id: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ModelCandidateSignal:
    """Candidate from a sealed ModelScore, never disguised as FactorScore."""
    signal_id: str
    timestamp: str
    symbol: str
    score: float
    rank: int
    percentile: float
    side: str
    research_run_id: str
    source_model_score_id: str
    source_model_id: str
    source_factor_set_id: str
    source_factor_vector_id: str
    fold_id: str
    model_config_sha256: str
    factor_set_config_sha256: str
    artifact_manifest_sha256: str
    model_payload_sha256: str
    compiled_plan_sha256: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Order:
    order_id: str
    timestamp: str
    symbol: str
    side: str
    quantity: int
    source_signal_id: str
    reason_code: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    timestamp: str
    symbol: str
    side: str
    quantity: int
    price: float
    fee: float
    slippage_cost: float
    status: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    opened_at: str
    last_updated_at: str
    unrealized_pnl: float
    realized_pnl: float
    holding_period: int
    side: str = "LONG"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AccountSnapshot:
    timestamp: str
    cash: float
    market_value: float
    nav: float
    realized_pnl: float
    unrealized_pnl: float
    gross_exposure: float
    net_exposure: float
    position_count: int

    def to_dict(self):
        return asdict(self)
