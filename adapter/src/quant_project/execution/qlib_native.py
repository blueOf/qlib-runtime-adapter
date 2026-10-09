"""Qlib-native daily portfolio backtest for the market-fishing workflow.

The project already defines the daily OHLC stop/take rules in ``replay.py``.
This module keeps those rules, but lets Qlib own the account, position,
strategy, executor, exchange, costs, and portfolio report.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import math
from typing import Any

import numpy as np
import pandas as pd
from qlib.backtest.account import Account
from qlib.backtest.backtest import backtest_loop
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.exchange import Exchange
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest.position import BasePosition
from qlib.backtest.utils import CommonInfrastructure
from qlib.data import D
from qlib.data.data import Cal, CalendarProvider, H
from qlib.strategy.base import BaseStrategy

from ..common import instrument
from .replay import curve_stats, max_drawdown, rounded, summarize


def _date(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


class _SentinelCalendarProvider(CalendarProvider):
    """Add one non-trading sentinel only for Qlib's closed daily interval."""

    def __init__(self, base, sentinel):
        self.base = base
        self.sentinel = pd.Timestamp(sentinel)

    def load_calendar(self, freq, future):
        values = list(self.base.load_calendar(freq, False))
        if future and freq == "day" and values and self.sentinel > values[-1]:
            values.append(self.sentinel)
        return values


@contextmanager
def _padded_daily_calendar(sentinel):
    """Temporarily give Qlib a right boundary for the provider's last day."""
    original_provider = Cal._provider
    cache = H["c"]
    original_cache = {key: cache.od[key] for key in list(cache.od) if key.startswith("day_")}
    for key in list(cache.od):
        if key.startswith("day_"):
            cache.pop(key)
    Cal.register(_SentinelCalendarProvider(original_provider, sentinel))
    try:
        yield
    finally:
        for key in list(cache.od):
            if key.startswith("day_"):
                cache.pop(key)
        for key, value in original_cache.items():
            cache[key] = value
        Cal.register(original_provider)


class RawDailyExchange(Exchange):
    """Qlib Exchange backed by this project's raw, unadjusted daily fields."""

    def __init__(self, *, codes, start_time, end_time, open_cost, close_cost, min_cost=0.0,
                 lot_size=1, fixed_slippage_bps=0.0, max_participation_rate=None,
                 partial_fill_policy="allow", limit_states=None, extra_quote=None):
        self._price_overrides: dict[tuple[str, str, int], float] = {}
        self.fixed_slippage_bps = float(fixed_slippage_bps)
        self.max_participation_rate = (float(max_participation_rate)
                                       if max_participation_rate is not None else None)
        self.partial_fill_policy = str(partial_fill_policy)
        self.lot_size = max(1, int(lot_size))
        self.order_block_reasons: dict[int, str] = {}
        self.limit_states: dict[tuple[str, str], dict] = {}
        self.volume_overrides: dict[tuple[str, str], float | None] = {}
        self.quote_frame_override = extra_quote.copy() if extra_quote is not None else None
        for item in limit_states or []:
            stock_id = instrument(item["code"]).upper()
            trade_date = _date(item["date"])
            state = dict(item["state"])
            self.limit_states[(stock_id, trade_date)] = state
            self.volume_overrides[(stock_id, trade_date)] = _finite(item.get("volume"))
        super().__init__(
            freq="day",
            start_time=start_time,
            end_time=end_time,
            codes=list(codes),
            deal_price=("$raw_open", "$raw_close"),
            limit_threshold=None,
            volume_threshold=None,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
            impact_cost=0.0,
            extra_quote=extra_quote,
            trade_unit=self.lot_size,
        )

    def get_quote_from_qlib(self) -> None:
        """Load raw fields and expose the aliases expected by Qlib Exchange."""
        fields = ["$raw_open", "$raw_close", "$raw_volume"]
        if self.quote_frame_override is not None:
            raw = self.quote_frame_override.copy()
            missing = [field for field in fields if field not in raw.columns]
            if missing:
                raise ValueError(f"Qlib test quote frame is missing fields: {missing}")
            raw = raw.loc[:, fields]
        elif self.codes:
            raw = D.features(self.codes, fields, self.start_time, self.end_time,
                             freq="day", disk_cache=True)
        else:
            raw = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []],
                              names=["instrument", "datetime"]), columns=fields)
        raw = raw.astype("float64")
        frame = raw.copy()
        frame["$close"] = raw["$raw_close"]
        frame["$volume"] = raw["$raw_volume"]
        frame["$factor"] = 1.0
        frame["$change"] = raw.groupby(level="instrument")["$raw_close"].pct_change(fill_method=None)
        frame["$change"] = frame["$change"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.quote_df = frame
        self.all_fields = list(frame.columns)
        self.trade_w_adj_price = False
        self._update_limit(None)

    def _update_limit(self, limit_threshold) -> None:
        """Apply the shared project's per-session tradability decisions."""
        buy, sell, suspended = [], [], []
        for instrument_id, timestamp in self.quote_df.index:
            state = self.limit_states.get((str(instrument_id).upper(), _date(timestamp)))
            buy.append(bool(state and state.get("block_buy")))
            sell.append(bool(state and state.get("block_sell")))
            suspended.append(bool(state and state.get("suspended")))
        self.quote_df["limit_buy"] = buy
        self.quote_df["limit_sell"] = sell
        suspended_series = self.quote_df["$close"].isna() | pd.Series(
            suspended, index=self.quote_df.index
        )
        self.quote_df.loc[suspended_series, "$close"] = np.nan

    def _volume_capacity(self, stock_id, start_time, end_time):
        key = (str(stock_id).upper(), _date(start_time))
        if key not in self.volume_overrides:
            return None
        volume = self.volume_overrides[key]
        if volume is None or self.max_participation_rate is None:
            return None
        return max(0.0, volume * self.max_participation_rate)

    def check_order(self, order) -> bool:
        stock_id, trade_date = str(order.stock_id).upper(), _date(order.start_time)
        state = self.limit_states.get((stock_id, trade_date))
        if state and state.get("suspended"):
            self.order_block_reasons[id(order)] = "SUSPENDED"
            return False
        if state and order.direction == Order.BUY and state.get("block_buy"):
            self.order_block_reasons[id(order)] = state.get("buy_block_reason") or "LIMIT_LOCKED"
            return False
        if state and order.direction == Order.SELL and state.get("block_sell"):
            self.order_block_reasons[id(order)] = state.get("sell_block_reason") or "LIMIT_LOCKED"
            return False
        if not super().check_order(order):
            self.order_block_reasons[id(order)] = "EXCHANGE_TRADABILITY"
            return False
        key = (stock_id, trade_date)
        if (self.max_participation_rate is not None
                and (key not in self.volume_overrides or self.volume_overrides[key] is None)):
            self.order_block_reasons[id(order)] = "LIQUIDITY_DATA_MISSING"
            return False
        capacity = self._volume_capacity(order.stock_id, order.start_time, order.end_time)
        if (capacity is not None and self.partial_fill_policy == "reject"
                and capacity + 1e-9 < float(order.amount)):
            self.order_block_reasons[id(order)] = "LIQUIDITY"
            return False
        return True

    def _clip_amount_by_volume(self, order, dealt_order_amount):
        """Clip with the project participation contract before Qlib lot rounding."""
        capacity = self._volume_capacity(order.stock_id, order.start_time, order.end_time)
        if capacity is None:
            return None
        order.deal_amount = max(0.0, min(float(order.deal_amount), capacity))
        return None

    def set_price_override(self, stock_id: str, trade_date: str, direction: OrderDir, price: float) -> None:
        self._price_overrides[(stock_id, _date(trade_date), int(direction))] = float(price)

    def get_deal_price(self, stock_id, start_time, end_time, direction, method="ts_data_last"):
        override = self._price_overrides.get((stock_id, _date(start_time), int(direction)))
        raw = (override if override is not None else
               super().get_deal_price(stock_id, start_time, end_time, direction, method))
        price = _finite(raw)
        if price is None:
            return raw
        sign = 1 if int(direction) == Order.BUY else -1
        return price * (1 + sign * self.fixed_slippage_bps / 10_000)


class MarketFishingStrategy(BaseStrategy):
    """Qlib strategy for signal-driven OHLC exits or daily-close rebalancing."""

    def __init__(self, events, market_dates, execution, *, trade_exchange=None,
                 level_infra=None, common_infra=None):
        super().__init__(level_infra=level_infra, common_infra=common_infra,
                         trade_exchange=trade_exchange)
        self.events = list(events)
        self.execution = execution
        self.market_dates = [_date(value) for value in market_dates]
        self.date_index = {value: index for index, value in enumerate(self.market_dates)}
        self.events_by_entry = defaultdict(list)
        self.signal_states_by_entry = defaultdict(dict)
        self.skipped: list[dict] = []
        self.blocked_orders: list[dict] = []
        self.blocked_exits: list[dict] = []
        self.order_records: list[dict] = []
        self.fill_records: list[dict] = []
        self._order_sequence = 0
        for event in self.events:
            future = sorted(event.get("future", []), key=lambda row: row["d"])
            signal_date = _date(event.get("date", future[0]["d"] if future else self.market_dates[0]))
            signal_index = self.date_index.get(signal_date)
            expected_entry = (self.market_dates[signal_index + 1]
                              if signal_index is not None and signal_index + 1 < len(self.market_dates)
                              else None)
            if expected_entry is None:
                self.skipped.append(self._signal_summary(event) | {
                    "skipped": "SIGNAL_EXPIRED",
                })
                continue
            entry_date = expected_entry
            self.events_by_entry[entry_date].append({
                "event": event,
                "signalDate": signal_date,
                "entryDate": entry_date,
                "bars": {_date(row["d"]): row for row in future},
                "actions": {
                    date: [action for action in event.get("corporateActions", [])
                           if _date(action["exDate"]) == date]
                    for date in sorted({_date(action["exDate"])
                                        for action in event.get("corporateActions", [])})
                },
            })
            self.signal_states_by_entry[entry_date][instrument(event["code"])] = event
        self.active: dict[str, dict] = {}
        self.pending: dict[int, dict] = {}
        self.slot_state: list[str | None] = [None] * int(execution["slotCount"])
        self.trade_records: list[dict] = []

    @staticmethod
    def _signal_summary(event: dict) -> dict:
        return {key: event[key] for key in
                ("code", "name", "industry", "factorScore", "dailyRank", "date")
                if key in event}

    def _public_skip(self, event: dict, reason: str) -> dict:
        return self._signal_summary(event) | {"skipped": reason}

    def _step_date(self) -> str:
        start, _ = self.trade_calendar.get_step_time()
        return _date(start)

    def _entry_price(self, item: dict) -> float | None:
        bar = item.get("bars", {}).get(item.get("entryDate"))
        future = [bar] if bar is not None else []
        field = "c" if self.execution.get("entry") == "next_session_close" else "o"
        return _finite(future[0].get(field)) if future else None

    @staticmethod
    def _equivalent_price(state: dict, price: float | None) -> float | None:
        value = _finite(price)
        if value is None:
            return None
        return value * float(state.get("actionScale", 1.0)) + float(state.get("actionOffset", 0.0))

    def _apply_corporate_actions(self, stock_id: str, state: dict, trade_date: str) -> None:
        actions = state.get("actions", {}).get(trade_date, [])
        if not actions or trade_date in state["appliedActionDates"]:
            return
        position = self.trade_position
        for action in actions:
            amount_before = float(position.get_stock_amount(stock_id))
            if amount_before <= 0:
                continue
            cash_per_10 = float(action["cashDividendPer10"])
            rights_price = float(action["rightsPrice"])
            bonus_per_10 = float(action["bonusSharePer10"])
            rights_per_10 = float(action["rightsSharePer10"])
            multiplier = 1.0 + (bonus_per_10 + rights_per_10) / 10.0
            if multiplier <= 0:
                raise ValueError(f"{stock_id}/{trade_date} 的公司行动产生非正持股倍率")
            gross_dividend = amount_before * cash_per_10 / 10.0
            rights_cost = amount_before * rights_per_10 * rights_price / 10.0
            available = float(position.get_cash()) + gross_dividend
            if rights_cost > available + 1e-8:
                raise ValueError(f"{stock_id}/{trade_date} 配股认购资金不足")
            net_cash = gross_dividend - rights_cost
            amount_after = amount_before * multiplier
            previous_price = _finite(position.get_stock_price(stock_id))
            position.position[stock_id]["amount"] = amount_after
            position.position["cash"] += net_cash
            state["remainingQuantity"] = amount_after
            if previous_price is not None:
                position.position[stock_id]["price"] = (
                    previous_price - (cash_per_10 - rights_per_10 * rights_price) / 10.0
                ) / multiplier
            prior_scale = float(state["actionScale"])
            state["actionOffset"] += prior_scale * (
                cash_per_10 - rights_per_10 * rights_price
            ) / 10.0
            state["actionScale"] = prior_scale * multiplier
            state["grossCashDividend"] += gross_dividend
            state["rightsSubscriptionCost"] += rights_cost
            state["corporateActionNetCash"] += net_cash
            state["corporateActionsApplied"].append({
                **action,
                "sharesBefore": rounded(amount_before, 6),
                "sharesAfter": rounded(amount_after, 6),
                "grossCashDividend": rounded(gross_dividend, 4),
                "rightsSubscriptionCost": rounded(rights_cost, 4),
                "netCash": rounded(net_cash, 4),
            })
        state["appliedActionDates"].add(trade_date)

    def _exit_for_bar(self, state: dict, trade_date: str, bar: dict):
        entry_date = state["entryDate"]
        if not self.execution.get("allowSameDayExit", False) and trade_date == entry_date:
            return None
        entry_price = state["entryPrice"]
        stop_pct = self.execution.get("stopPct")
        take_pct = self.execution.get("takePct")
        trailing_pct = self.execution.get("trailingPct")
        stop = (entry_price * (1 + stop_pct / 100)
                if stop_pct is not None else None)
        take = (entry_price * (1 + take_pct / 100)
                if take_pct is not None else None)
        closing = _finite(bar.get("c"))
        if bar.get("suspended"):
            signal_map = self.signal_states_by_entry.get(trade_date)
            current_signal = (signal_map or {}).get(state.get("stockId"))
            if current_signal is not None and str(current_signal.get("side", "LONG")).upper() != "LONG":
                current_signal = None
            reason = None
            if signal_map is not None and self.execution.get("signalExit", False) and current_signal is None:
                reason = "SIGNAL_EXIT"
            rank_exit = self.execution.get("rankExit")
            if (reason is None and rank_exit is not None and current_signal is not None
                    and int(current_signal.get("dailyRank", 10**9)) > int(rank_exit)):
                reason = "RANK_EXIT"
            entry_index = self.date_index.get(entry_date)
            current_index = self.date_index.get(trade_date)
            if (reason is None and entry_index is not None and current_index is not None
                    and current_index - entry_index + 1 >= int(self.execution["holdDays"])):
                reason = "MAX_HOLDING"
            if reason is not None:
                return closing or entry_price, reason
            if self.execution.get("rebalance") == "daily_close":
                return closing or entry_price, "DAILY_CLOSE_REBALANCE"
            return None
        equivalent_close = self._equivalent_price(state, closing)
        if self.execution.get("rebalance") == "daily_close":
            if closing is None or equivalent_close is None:
                return None
            if stop is not None and equivalent_close <= stop:
                return closing, "STOP_LOSS"
            if take is not None and equivalent_close >= take:
                return closing, "TAKE_PROFIT"
            return closing, "DAILY_CLOSE_REBALANCE"
        opening = _finite(bar.get("o"))
        high = _finite(bar.get("h"))
        low = _finite(bar.get("l"))
        if opening is None or high is None or low is None or closing is None:
            return None
        equivalent_open = self._equivalent_price(state, opening)
        equivalent_high = self._equivalent_price(state, high)
        equivalent_low = self._equivalent_price(state, low)
        scale = float(state.get("actionScale", 1.0))
        offset = float(state.get("actionOffset", 0.0))
        extreme = _finite(state.get("extremePrice")) or entry_price
        if equivalent_high is not None:
            extreme = max(extreme, equivalent_high)
        state["extremePrice"] = extreme
        trailing = (extreme * (1 - trailing_pct / 100)
                    if trailing_pct is not None else None)
        if stop is not None and equivalent_open <= stop:
            return opening, "STOP_LOSS_GAP"
        if trailing is not None and equivalent_open <= trailing:
            return opening, "TRAILING_STOP_GAP"
        if take is not None and equivalent_open >= take:
            return opening, "TAKE_PROFIT_GAP"
        stop_hit = stop is not None and equivalent_low <= stop
        trailing_hit = trailing is not None and equivalent_low <= trailing
        take_hit = take is not None and equivalent_high >= take
        if stop_hit:
            return (stop - offset) / scale, "STOP_LOSS"
        if trailing_hit:
            return (trailing - offset) / scale, "TRAILING_STOP"
        if take_hit:
            return (take - offset) / scale, "TAKE_PROFIT"

        # CandidateSignal state is aligned to the execution trade date.  No
        # signal emitted after this date can influence this decision.
        signal_map = self.signal_states_by_entry.get(trade_date)
        current_signal = (signal_map or {}).get(state.get("stockId"))
        if current_signal is not None and str(current_signal.get("side", "LONG")).upper() != "LONG":
            current_signal = None
        if signal_map is not None and self.execution.get("signalExit", False) and current_signal is None:
            return closing, "SIGNAL_EXIT"
        rank_exit = self.execution.get("rankExit")
        if (rank_exit is not None and current_signal is not None
                and int(current_signal.get("dailyRank", 10**9)) > int(rank_exit)):
            return closing, "RANK_EXIT"
        entry_index = self.date_index.get(entry_date)
        current_index = self.date_index.get(trade_date)
        if entry_index is not None and current_index is not None:
            holding_days = current_index - entry_index + 1
            if holding_days >= int(self.execution["holdDays"]):
                return closing, "MAX_HOLDING"
        return None

    def _make_order(self, stock_id: str, amount: float, direction, start, end) -> Order:
        return Order(stock_id=stock_id, amount=float(amount), direction=direction,
                     start_time=start, end_time=end)

    def _new_order_evidence(self, *, timestamp, stock_id, side, quantity,
                            event=None, reason_code=None):
        self._order_sequence += 1
        record = {
            "order_id": f"qlib-order-{self._order_sequence:06d}",
            "timestamp": timestamp,
            "symbol": stock_id,
            "side": side,
            "quantity": float(quantity),
            "fill_quantity": 0.0,
            "status": "PENDING",
            "reason_code": reason_code,
            "source_signal_id": (event or {}).get("signalId"),
        }
        self.order_records.append(record)
        return len(self.order_records) - 1, record

    def _blocked_entry(self, event, timestamp, reason, *, market_state=None):
        row = self._public_skip(event, reason) | {"timestamp": timestamp}
        if market_state is not None:
            row["market_state_evidence"] = market_state
        self.skipped.append(row)
        self.blocked_orders.append(row.copy())

    def _blocked_exit(self, stock_id, timestamp, reason, exit_reason, *, market_state=None):
        row = {"symbol": stock_id, "timestamp": timestamp,
               "reason": reason, "exit_reason": exit_reason}
        if market_state is not None:
            row["market_state_evidence"] = market_state
        self.blocked_exits.append(row)

    def generate_trade_decision(self, execute_result=None):
        trade_start, trade_end = self.trade_calendar.get_step_time()
        trade_date = _date(trade_start)
        orders = []
        scheduled_sells = {}
        estimated_sale_cash = 0.0

        # Existing positions are checked first.  This is the same event order
        # as strategy_backtest: existing-position exit, then scheduled entry.
        for stock_id, state in sorted(self.active.items()):
            self._apply_corporate_actions(stock_id, state, trade_date)
            if state.get("exitPending"):
                continue
            bar = state["bars"].get(trade_date)
            if bar is None:
                continue
            exit_info = self._exit_for_bar(state, trade_date, bar)
            if exit_info is None:
                continue
            exit_price, reason = exit_info
            if bar.get("suspended"):
                self._blocked_exit(stock_id, trade_date, "SUSPENDED", reason)
                continue
            limit_state = bar.get("limitState") or {}
            if limit_state.get("block_sell"):
                self._blocked_exit(stock_id, trade_date,
                                   limit_state.get("sell_block_reason") or "LIMIT_LOCKED",
                                   reason, market_state=limit_state)
                continue
            amount = self.trade_position.get_stock_amount(stock_id)
            if amount <= 0:
                continue
            volume = _finite(bar.get("volume"))
            rate = self.execution.get("maxParticipationRate")
            if rate is not None and volume is None:
                self._blocked_exit(stock_id, trade_date, "LIQUIDITY_DATA_MISSING", reason)
                continue
            capacity = (math.floor(max(0.0, volume * rate) / self.execution["lotSize"])
                        * self.execution["lotSize"] if volume is not None and rate is not None
                        else None)
            if capacity is not None and (
                    capacity <= 0 or (self.execution["partialFillPolicy"] == "reject"
                                      and capacity + 1e-9 < amount)):
                self._blocked_exit(stock_id, trade_date, "LIQUIDITY", reason)
                continue
            order = self._make_order(stock_id, amount, Order.SELL, trade_start, trade_end)
            order._mf_exit_reason = reason
            raw_price = float(exit_price)
            self.trade_exchange.set_price_override(stock_id, trade_date, Order.SELL, exit_price)
            state["exitPending"] = True
            order_record_index, order_record = self._new_order_evidence(
                timestamp=trade_date, stock_id=stock_id, side="SELL", quantity=amount,
                event=state.get("event"), reason_code=reason,
            )
            self.pending[id(order)] = {"kind": "sell", "stock": stock_id,
                                       "rawPrice": raw_price,
                                       "orderRecordIndex": order_record_index,
                                       "orderRecord": order_record}
            scheduled_sells[stock_id] = state
            slipped_exit = raw_price * (1 - float(self.execution.get("fixedSlippageBps", 0.0)) / 10_000)
            gross_proceeds = float(amount) * slipped_exit
            fee = max(gross_proceeds * float(self.trade_exchange.close_cost),
                      float(self.trade_exchange.min_cost))
            estimated_sale_cash += max(gross_proceeds - fee, 0.0)
            orders.append(order)

        available_cash = float(self.trade_position.get_cash()) + estimated_sale_cash
        open_cost = float(self.trade_exchange.open_cost)
        reusable_slots = {state["slot"] for state in scheduled_sells.values()}
        available_slots = [index for index, state in enumerate(self.slot_state)
                           if state is None or index in reusable_slots]
        reserved_slots = set()
        for item in sorted(self.events_by_entry.get(trade_date, []),
                           key=lambda value: (value["event"].get("dailyRank", 10**9),
                                              value["event"].get("code", ""))):
            event = item["event"]
            stock_id = instrument(event["code"])
            if stock_id in scheduled_sells:
                self._blocked_entry(event, trade_date, "EXITED_THIS_SESSION")
                continue
            if stock_id in self.active or any(
                    pending.get("kind") == "buy" and pending.get("stock") == stock_id
                    for pending in self.pending.values()):
                self._blocked_entry(event, trade_date, "ALREADY_HELD")
                continue
            slot = next((index for index in available_slots if index not in reserved_slots), None)
            if slot is None:
                self._blocked_entry(event, trade_date, "NO_CAPACITY")
                continue
            bar = item["bars"].get(trade_date)
            if bar is None:
                self._blocked_entry(event, trade_date, "NO_PRICE")
                continue
            if bar.get("suspended"):
                self._blocked_entry(event, trade_date, "SUSPENDED")
                continue
            limit_state = bar.get("limitState") or {}
            if limit_state.get("block_buy"):
                self._blocked_entry(event, trade_date,
                                    limit_state.get("buy_block_reason") or "LIMIT_LOCKED",
                                    market_state=limit_state)
                continue
            price = self._entry_price(item)
            if price is None or price <= 0:
                self._blocked_entry(event, trade_date, "NO_PRICE")
                continue
            lot_size = max(1, int(self.execution.get("lotSize", 1)))
            slipped_price = price * (1 + float(self.execution.get("fixedSlippageBps", 0.0)) / 10_000)
            current_market = 0.0
            for held_stock, held in self.active.items():
                if held_stock in scheduled_sells:
                    continue
                held_amount = float(self.trade_position.get_stock_amount(held_stock))
                held_price = _finite(self.trade_position.get_stock_price(held_stock))
                if held_price is not None:
                    current_market += held_amount * held_price
            max_gross = (float(self.execution.get("initialCapital", 1_000_000.0))
                         * float(self.execution.get("maxGrossExposure", 1.0)))
            target_value = min(
                float(self.execution.get("initialCapital", 1_000_000.0))
                * float(self.execution["slotWeight"]),
                max(0.0, max_gross - current_market),
                available_cash / (1 + open_cost),
            )
            raw_amount = target_value / slipped_price if target_value > 0 else 0
            amount = math.floor(raw_amount / lot_size) * lot_size
            if amount <= 0:
                self._blocked_entry(event, trade_date, "NO_CASH")
                continue
            volume = _finite(bar.get("volume"))
            rate = self.execution.get("maxParticipationRate")
            if rate is not None and volume is None:
                self._blocked_entry(event, trade_date, "LIQUIDITY_DATA_MISSING")
                continue
            capacity = (math.floor(max(0.0, volume * rate) / lot_size) * lot_size
                        if volume is not None and rate is not None else None)
            if capacity is not None and (
                    capacity <= 0 or (self.execution["partialFillPolicy"] == "reject"
                                      and capacity + 1e-9 < amount)):
                self._blocked_entry(event, trade_date, "LIQUIDITY")
                continue
            order = self._make_order(stock_id, amount, Order.BUY, trade_start, trade_end)
            order._mf_event = item
            order._mf_slot = slot
            self.trade_exchange.set_price_override(stock_id, trade_date, Order.BUY, price)
            self.slot_state[slot] = f"pending:{id(order)}"
            reserved_slots.add(slot)
            order_record_index, order_record = self._new_order_evidence(
                timestamp=trade_date, stock_id=stock_id, side="BUY", quantity=amount,
                event=event, reason_code="CANDIDATE_SIGNAL",
            )
            self.pending[id(order)] = {"kind": "buy", "stock": stock_id,
                                       "item": item, "slot": slot,
                                       "rawPrice": price,
                                       "orderRecordIndex": order_record_index,
                                       "orderRecord": order_record}
            available_cash -= target_value * (1 + open_cost)
            orders.append(order)
        return TradeDecisionWO(orders, self)

    def post_exe_step(self, execute_result):
        for order, trade_val, trade_cost, trade_price in execute_result or []:
            meta = self.pending.pop(id(order), None)
            if meta is None:
                continue
            order_record = meta["orderRecord"]
            record_index = meta["orderRecordIndex"]
            raw_price = float(meta["rawPrice"])
            fill_quantity = float(getattr(order, "deal_amount", 0.0) or 0.0)
            blocked_reason = self.trade_exchange.order_block_reasons.pop(id(order), None)
            if trade_val <= 1e-8 or not np.isfinite(trade_price) or fill_quantity <= 1e-8:
                reason = blocked_reason or "NO_FILL"
                order_record.update({"status": "BLOCKED", "blocked_reason": reason})
                if meta["kind"] == "buy":
                    self.slot_state[meta["slot"]] = None
                    self.blocked_orders.append({
                        "timestamp": _date(order.start_time), "symbol": order.stock_id,
                        "reason": reason,
                        "signal_id": meta["item"]["event"].get("signalId"),
                    })
                else:
                    state = self.active.get(order.stock_id)
                    if state is not None:
                        state["exitPending"] = False
                    self._blocked_exit(order.stock_id, _date(order.start_time), reason,
                                       getattr(order, "_mf_exit_reason", "UNKNOWN"))
                continue

            slippage = fill_quantity * abs(float(trade_price) - raw_price)
            fill_id = f"qlib-fill-{len(self.fill_records) + 1:06d}"
            fill = {
                "fill_id": fill_id,
                "order_id": order_record["order_id"],
                "timestamp": _date(order.start_time),
                "symbol": order.stock_id,
                "side": "BUY" if order.direction == Order.BUY else "SELL",
                "quantity": fill_quantity,
                "price": float(trade_price),
                "raw_price": raw_price,
                "fee": float(trade_cost),
                "slippage_cost": slippage,
                "status": "FULL" if fill_quantity + 1e-8 >= float(order.amount) else "PARTIAL",
            }
            self.fill_records.append(fill)
            order_record.update({
                "fill_quantity": fill_quantity,
                "fill_price": float(trade_price),
                "fee": float(trade_cost),
                "slippage_cost": slippage,
                "status": fill["status"],
            })

            if meta["kind"] == "buy":
                item = meta["item"]
                event = item["event"]
                entry_date = item["entryDate"]
                entry_bar = item["bars"].get(entry_date) or {}
                self.slot_state[meta["slot"]] = order.stock_id
                state = {
                    "event": event,
                    "stockId": order.stock_id,
                    "side": "LONG",
                    "bars": item["bars"],
                    "entryDate": entry_date,
                    "entryIndex": self.date_index.get(entry_date),
                    "entryPrice": float(trade_price),
                    "entryValue": float(trade_val),
                    "entryQuantity": fill_quantity,
                    "remainingQuantity": fill_quantity,
                    "entryCost": float(trade_cost),
                    "remainingEntryCost": float(trade_cost),
                    "entrySlippageCost": slippage,
                    "remainingEntrySlippage": slippage,
                    "slot": meta["slot"],
                    "exitPending": False,
                    "actions": item["actions"],
                    "appliedActionDates": set(),
                    "actionScale": 1.0,
                    "actionOffset": 0.0,
                    "extremePrice": max(float(trade_price),
                                         _finite(entry_bar.get("h")) or float(trade_price)),
                    "grossCashDividend": 0.0,
                    "rightsSubscriptionCost": 0.0,
                    "corporateActionNetCash": 0.0,
                    "corporateActionsApplied": [],
                }
                self.active[order.stock_id] = state
                if not self.execution.get("allowSameDayExit", False):
                    stop_pct = self.execution.get("stopPct")
                    take_pct = self.execution.get("takePct")
                    low = _finite(entry_bar.get("l"))
                    high = _finite(entry_bar.get("h"))
                    stop = (float(trade_price) * (1 + stop_pct / 100)
                            if stop_pct is not None else None)
                    take = (float(trade_price) * (1 + take_pct / 100)
                            if take_pct is not None else None)
                    stop_hit = stop is not None and low is not None and low <= stop
                    take_hit = take is not None and high is not None and high >= take
                    if stop_hit or take_hit:
                        self._blocked_exit(order.stock_id, entry_date, "T_PLUS_ONE",
                                           "STOP_LOSS" if stop_hit else "TAKE_PROFIT")
            else:
                state = self.active.get(order.stock_id)
                if state is None:
                    continue
                quantity_before = max(float(state.get("remainingQuantity", 0.0)), fill_quantity)
                entry_fee_piece = (float(state.get("remainingEntryCost", 0.0))
                                   * fill_quantity / quantity_before if quantity_before else 0.0)
                entry_slippage_piece = (float(state.get("remainingEntrySlippage", 0.0))
                                        * fill_quantity / quantity_before if quantity_before else 0.0)
                scale = float(state.get("actionScale", 1.0))
                offset = float(state.get("actionOffset", 0.0))
                current_cost = (float(state["entryPrice"]) - offset) / scale
                gross_pnl = fill_quantity * (float(trade_price) - current_cost)
                net_pnl = gross_pnl - entry_fee_piece - float(trade_cost)
                invested = max(fill_quantity * current_cost + entry_fee_piece, 1e-12)
                record = self._signal_summary(state["event"]) | {
                    "signalDate": state["event"].get("date"),
                    "entryDate": state["entryDate"],
                    "exitDate": _date(order.start_time),
                    "entryPrice": rounded(current_cost, 4),
                    "exitPrice": rounded(trade_price, 4),
                    "quantity": fill_quantity,
                    "grossPnl": rounded(gross_pnl, 4),
                    "netPnl": rounded(net_pnl, 4),
                    "exitReason": getattr(order, "_mf_exit_reason", "UNKNOWN"),
                    "entryCost": rounded(entry_fee_piece, 4),
                    "exitCost": rounded(trade_cost, 4),
                    "entrySlippageCost": rounded(entry_slippage_piece, 4),
                    "exitSlippageCost": rounded(slippage, 4),
                    "slippageCost": rounded(entry_slippage_piece + slippage, 4),
                    "grossCashDividend": rounded(state["grossCashDividend"], 4),
                    "rightsSubscriptionCost": rounded(state["rightsSubscriptionCost"], 4),
                    "corporateActionNetCash": rounded(state["corporateActionNetCash"], 4),
                    "corporateActions": state["corporateActionsApplied"],
                    "netPct": rounded(net_pnl / invested * 100, 4),
                }
                self.trade_records.append(record)
                state["remainingEntryCost"] = max(0.0, state["remainingEntryCost"] - entry_fee_piece)
                state["remainingEntrySlippage"] = max(
                    0.0, state["remainingEntrySlippage"] - entry_slippage_piece
                )
                remaining = float(self.trade_position.get_stock_amount(order.stock_id))
                state["remainingQuantity"] = remaining
                state["exitPending"] = False
                if remaining <= 1e-8:
                    self.slot_state[state["slot"]] = None
                    del self.active[order.stock_id]


def run_native_backtest(events, market_dates, execution, initial_cash=1_000_000.0,
                        exchange: RawDailyExchange | None = None):
    """Run the event stream through Qlib's strategy/executor/exchange stack."""
    execution = dict(execution)
    execution["initialCapital"] = float(initial_cash)
    codes = sorted({instrument(event["code"]) for event in events})
    if not codes:
        return {"strategy": None, "report": pd.DataFrame(), "indicator": None}
    open_close_cost = float(execution["roundTripCostBps"]) / 10000 / 2
    limit_states = []
    for event in events:
        for bar in event.get("future", []):
            limit_states.append({
                "code": event["code"], "date": bar["d"],
                "state": dict(bar.get("limitState") or {}, suspended=bool(bar.get("suspended"))),
                "volume": bar.get("volume"),
            })
    if exchange is None:
        exchange = RawDailyExchange(
            codes=codes,
            start_time=market_dates[0],
            end_time=market_dates[-1],
            open_cost=open_close_cost,
            close_cost=open_close_cost,
            min_cost=float(execution.get("minimumFee", 0.0)),
            lot_size=int(execution.get("lotSize", 1)),
            fixed_slippage_bps=float(execution.get("fixedSlippageBps", 0.0)),
            max_participation_rate=execution.get("maxParticipationRate"),
            partial_fill_policy=execution.get("partialFillPolicy", "allow"),
            limit_states=limit_states,
        )
    else:
        if list(exchange.codes) != codes:
            raise ValueError("A reused Qlib exchange must have the same instrument set")
        if (pd.Timestamp(exchange.start_time) != pd.Timestamp(market_dates[0])
                or pd.Timestamp(exchange.end_time) != pd.Timestamp(market_dates[-1])):
            raise ValueError("A reused Qlib exchange must have the same market-date window")
        if (abs(float(exchange.open_cost) - open_close_cost) > 1e-12
                or abs(float(exchange.close_cost) - open_close_cost) > 1e-12):
            raise ValueError("A reused Qlib exchange must have the same trading costs")
        if abs(float(exchange.min_cost) - float(execution.get("minimumFee", 0.0))) > 1e-12:
            raise ValueError("A reused Qlib exchange must have the same minimum fee")
        if (exchange.lot_size != int(execution.get("lotSize", 1))
                or abs(exchange.fixed_slippage_bps
                       - float(execution.get("fixedSlippageBps", 0.0))) > 1e-12
                or exchange.max_participation_rate != execution.get("maxParticipationRate")
                or exchange.partial_fill_policy != execution.get("partialFillPolicy", "allow")):
            raise ValueError("A reused Qlib exchange must match lot, slippage, and volume policy")
        exchange.limit_states = {}
        exchange.volume_overrides = {}
        for item in limit_states:
            key = (instrument(item["code"]).upper(), _date(item["date"]))
            exchange.limit_states[key] = item["state"]
            exchange.volume_overrides[key] = _finite(item.get("volume"))
        exchange._update_limit(None)
        # OHLC exit prices are parameter-dependent overrides. Drop them before
        # the next replay so one variant cannot affect another.
        exchange._price_overrides.clear()
    account = Account(
        init_cash=float(initial_cash),
        benchmark_config={"benchmark": None},
        port_metr_enabled=True,
    )
    common = CommonInfrastructure(trade_account=account, trade_exchange=exchange)
    strategy = MarketFishingStrategy(events, market_dates, execution,
                                     trade_exchange=exchange, common_infra=common)
    sentinel = pd.Timestamp(market_dates[-1]) + pd.offsets.BDay(1)
    with _padded_daily_calendar(sentinel):
        executor = SimulatorExecutor(
            time_per_step="day",
            start_time=market_dates[0],
            end_time=market_dates[-1],
            generate_portfolio_metrics=True,
            common_infra=common,
            trade_type=SimulatorExecutor.TT_SERIAL,
            settle_type=BasePosition.ST_NO,
        )
        portfolio_metrics, indicators = backtest_loop(
            market_dates[0], market_dates[-1], strategy, executor)
    account_report, _ = strategy.common_infra.get("trade_account").get_portfolio_metrics()
    report = account_report
    indicator = indicators.get("day", (None, None))[0] if indicators else None
    return {"strategy": strategy, "report": report, "indicator": indicator}


def build_native_result(strategy: MarketFishingStrategy | None, report: pd.DataFrame,
                        market_dates, execution, initial_cash=1_000_000.0):
    """Convert Qlib's native report to the project's existing JSON contract."""
    if strategy is None:
        return {
            "allocation": f"{execution['slotCount']}个独立仓位×{round(execution['slotWeight'] * 100)}%，其余现金",
            "selectedSignals": 0, "enteredTrades": 0, "skippedSignals": 0, "skipped": [],
            "tradeSummary": summarize([]), "portfolio": curve_stats([]), "byYear": [],
            "byExitReason": {}, "trades": [], "openPositions": [], "curve": [],
            "orders": [], "fills": [], "blockedOrders": [], "blockedExits": [],
            "finalCash": float(initial_cash), "finalPositions": [],
            "totalFees": 0.0, "totalSlippageCost": 0.0,
        }
    curve = []
    if not report.empty:
        for timestamp, row in report.iterrows():
            account = _finite(row.get("account"))
            if account is not None:
                curve.append({"date": _date(timestamp),
                              "equity": rounded(account / float(initial_cash), 8)})
    trades = sorted(strategy.trade_records, key=lambda trade: (trade["entryDate"], trade["code"]))
    years = []
    previous = 1.0
    for year in sorted({row["date"][:4] for row in curve}):
        year_curve = [row for row in curve if row["date"].startswith(year)]
        end = year_curve[-1]["equity"] if year_curve else previous
        years.append({
            "year": year,
            "portfolioReturn": rounded((end / previous - 1) * 100, 2),
            "maxDrawdown": rounded(max_drawdown([{"equity": previous}] + year_curve), 2),
            "trades": summarize([trade for trade in trades if trade["entryDate"].startswith(year)]),
        })
        previous = end
    reasons = {}
    for trade in trades:
        reasons[trade["exitReason"]] = reasons.get(trade["exitReason"], 0) + 1
    open_positions = []
    last_date = _date(market_dates[-1]) if market_dates else None
    last_index = strategy.date_index.get(last_date)
    for stock_id, state in sorted(strategy.active.items()):
        quotes = [(date, bar) for date, bar in state["bars"].items()
                  if last_date is not None and date <= last_date and _finite(bar.get("c")) is not None]
        if not quotes:
            continue
        mark_date, mark_bar = max(quotes, key=lambda value: value[0])
        mark_price = float(mark_bar["c"])
        shares = float(strategy.trade_position.get_stock_amount(stock_id))
        invested = float(state["entryValue"]) + float(state["entryCost"])
        action_net_cash = float(state["corporateActionNetCash"])
        position = strategy._signal_summary(state["event"]) | {
            "entryDate": state["entryDate"],
            "entryPrice": rounded(state["entryPrice"], 4),
            "markDate": mark_date,
            "markPrice": rounded(mark_price, 4),
            "shares": rounded(shares, 4),
            "marketValue": rounded(shares * mark_price, 2),
            "grossCashDividend": rounded(state["grossCashDividend"], 4),
            "rightsSubscriptionCost": rounded(state["rightsSubscriptionCost"], 4),
            "corporateActionNetCash": rounded(action_net_cash, 4),
            "corporateActions": state["corporateActionsApplied"],
            "unrealizedPct": rounded(((shares * mark_price + action_net_cash) / invested - 1) * 100, 4)
            if invested > 0 else None,
        }
        mark_index = strategy.date_index.get(mark_date)
        if last_index is not None and mark_index is not None:
            position["markAgeSessions"] = last_index - mark_index
        open_positions.append(position)
    total_fees = sum(float(item.get("fee", 0.0)) for item in strategy.fill_records)
    total_slippage = sum(float(item.get("slippage_cost", 0.0)) for item in strategy.fill_records)
    final_positions = [{
        "symbol": stock_id,
        "quantity": float(strategy.trade_position.get_stock_amount(stock_id)),
        "avg_cost": float(strategy.trade_position.get_stock_price(stock_id)),
        "entry_date": state["entryDate"],
    } for stock_id, state in sorted(strategy.active.items())]
    return {
        "allocation": f"{execution['slotCount']}个独立仓位×{round(execution['slotWeight'] * 100)}%，其余现金",
        "selectedSignals": len(strategy.events),
        "enteredTrades": len(trades),
        "skippedSignals": len(strategy.skipped),
        "skipped": strategy.skipped,
        "tradeSummary": summarize(trades),
        "portfolio": curve_stats(curve),
        "byYear": years,
        "byExitReason": reasons,
        "trades": trades,
        "openPositions": open_positions,
        "curve": curve,
        "orders": strategy.order_records,
        "fills": strategy.fill_records,
        "blockedOrders": strategy.blocked_orders,
        "blockedExits": strategy.blocked_exits,
        "finalCash": float(strategy.trade_position.get_cash()),
        "finalPositions": final_positions,
        "totalFees": total_fees,
        "totalSlippageCost": total_slippage,
    }
