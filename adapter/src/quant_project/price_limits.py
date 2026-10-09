"""Versioned, fail-closed SSE/SZSE main-board A-share price-limit rules.

The rule table is deliberately narrow: ordinary A shares on the Shanghai and
Shenzhen main boards, IPO listing sessions, and daily limit prices.  It does
not infer relisting, delisting-period, or exchange-designated exceptional
no-limit days from OHLC data.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import math
import re
from typing import Any, Iterable


RULE_SET_ID = "cn-a-mainboard-price-limits"
RULE_SET_VERSION = "1.0.0"
# Rechecked 2026-10-07 against the current SSE/SZSE 2026 rules below.
# Extend only through the latest closed session, not into future sessions.
RULES_VERIFIED_THROUGH = date(2026, 9, 30)
TICK_SIZE = Decimal("0.01")

RULE_SOURCES = {
    "sse_2006": {
        "title": "上海证券交易所交易规则（2006年）",
        "url": "https://big5.sse.com.cn/site/cht/www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/c_20230418_5720136.shtml",
        "effective_from": "2006-07-01",
        "scope": "A股10%；ST/*ST 5%；前收盘价公式；按最小价格变动单位四舍五入；IPO首日无价格涨跌幅限制。",
    },
    "szse_2006": {
        "title": "深圳证券交易所交易规则及实施通知（2006年）",
        "url": "https://www.szse.cn/disclosure/notice/general/t20060515_499577.html",
        "effective_from": "2006-07-01",
        "scope": "股票10%；ST/*ST 5%；前收盘价公式；按最小价格变动单位四舍五入；IPO首日无价格涨跌幅限制。",
        "implementation_url": "https://www.szse.cn/disclosure/notice/general/t20060630_499617.html",
    },
    "szse_sme_merge": {
        "title": "关于合并主板与中小板相关安排的通知",
        "url": "https://www.szse.cn/disclosure/notice/general/t20210331_585343.html",
        "effective_from": "2021-04-06",
        "scope": "原中小板股票的证券类别变更为主板A股，代码保持不变。",
    },
    "szse_code_map": {
        "title": "深圳证券交易所证券代码区间表（2024年12月修订）",
        "url": "https://www.szse.cn/marketServices/technicalservice/doc/P020241212550140892927.pdf",
        "effective_from": "2024-12-12",
        "scope": "主板A股代码包括000、001200-001999及002-004；001001-001199为主板存托凭证，不纳入本规则。",
    },
    "sse_2023": {
        "title": "上海证券交易所交易规则（2023年修订）",
        "url": "https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20250612_10824490.shtml",
        "effective_from": "2023-04-10",
        "scope": "主板新股上市前5个交易日无价格涨跌幅限制；普通主板10%；风险警示股票5%；0.01元A股最小变动单位及低价最小价差处理。",
    },
    "szse_2023": {
        "title": "深圳证券交易所交易规则（2023年修订）",
        "url": "https://docs.static.szse.cn/www/lawrules/rule/stock/W020230217564423808793.pdf",
        "effective_from": "2023-04-10",
        "scope": "主板新股上市前5个交易日无价格涨跌幅限制；普通主板10%；风险警示股票5%；按价格最小变动单位取整。",
    },
    "sse_2026": {
        "title": "上海证券交易所交易规则（2026年修订）",
        "url": "https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml",
        "effective_from": "2026-07-06",
        "scope": "主板风险警示股票涨跌幅比例由5%调整为10%；A股价格最小变动单位0.01元；按四舍五入取整并适用低价最小价差处理。",
    },
    "szse_2026": {
        "title": "深圳证券交易所交易规则（2026年修订）",
        "url": "https://investor.szse.cn/lawrules/rule/trade/t20260424_620190.html",
        "effective_from": "2026-07-06",
        "scope": "主板风险警示股票涨跌幅比例调整为10%，自2026-07-06起施行。",
    },
}


class PriceLimitRuleError(ValueError):
    """A required identity, input, or effective rule is outside covered scope."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _as_date(value: Any, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        raise PriceLimitRuleError("MISSING_OR_INVALID_DATE", f"{field} must be YYYY-MM-DD") from None


def _decimal(value: Any, field: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise PriceLimitRuleError("MISSING_REFERENCE_PRICE", f"{field} is required")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise PriceLimitRuleError("INVALID_REFERENCE_PRICE", f"{field} is not numeric") from None
    if not result.is_finite() or result <= 0:
        raise PriceLimitRuleError("INVALID_REFERENCE_PRICE", f"{field} must be finite and positive")
    return result


def _normalize_code(row: dict) -> tuple[str, str]:
    exchange_raw = row.get("exchange")
    raw = str(row.get("code", row.get("symbol", ""))).strip().upper()
    if not exchange_raw or not raw:
        raise PriceLimitRuleError("MISSING_SECURITY_IDENTITY", "exchange and code are required")
    exchange_text = str(exchange_raw).strip().upper()
    exchange_aliases = {
        "SH": "SSE", "XSHG": "SSE", "SHSE": "SSE", "SSE": "SSE", "上海": "SSE",
        "SZ": "SZSE", "XSHE": "SZSE", "SZSE": "SZSE", "深圳": "SZSE",
    }
    exchange = exchange_aliases.get(exchange_text)
    if exchange is None:
        raise PriceLimitRuleError("UNSUPPORTED_EXCHANGE", f"unsupported exchange {exchange_raw!r}")
    suffix_match = re.fullmatch(
        r"(?:(?:SH|SZ|SSE|SZSE|XSHG|XSHE)\.)?(\d{6})(?:\.(?:SH|SZ|SSE|SZSE|XSHG|XSHE))?",
        raw,
    )
    if suffix_match is None:
        suffix_match = re.fullmatch(r"(?:SH|SZ|SSE|SZSE|XSHG|XSHE)(\d{6})", raw)
    if suffix_match is None:
        raise PriceLimitRuleError("UNSUPPORTED_SECURITY_CODE", f"unsupported code format {raw!r}")
    code = suffix_match.group(1)
    suffix = raw.rsplit(".", 1)[-1] if "." in raw else None
    if suffix in {"SH", "SSE", "XSHG"} and exchange != "SSE":
        raise PriceLimitRuleError("EXCHANGE_CODE_MISMATCH", f"{raw} conflicts with exchange {exchange}")
    if suffix in {"SZ", "SZSE", "XSHE"} and exchange != "SZSE":
        raise PriceLimitRuleError("EXCHANGE_CODE_MISMATCH", f"{raw} conflicts with exchange {exchange}")
    return exchange, code


def _main_board(exchange: str, code: str, day: date) -> str:
    if exchange == "SSE" and code[:3] in {"600", "601", "603", "605"}:
        return "SSE_MAIN_A"
    if exchange == "SZSE":
        if code.startswith("000"):
            return "SZSE_MAIN_A"
        if day >= date(2021, 4, 6) and (
                code.startswith(("0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019"))
                or code[:3] in {"002", "003", "004"}):
            return "SZSE_MAIN_A"
        if code[:3] in {"002", "003", "004"} and day < date(2021, 4, 6):
            raise PriceLimitRuleError(
                "UNSUPPORTED_BOARD",
                f"SZSE {code} was outside the main-board scope before the 2021-04-06 board merger",
            )
    raise PriceLimitRuleError(
        "UNSUPPORTED_BOARD", f"{exchange} {code} is not a supported main-board A share"
    )


def _is_st(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"1", "TRUE", "Y", "YES", "ST", "*ST"}:
            return True
        if normalized in {"0", "FALSE", "N", "NO", "NORMAL", "非ST"}:
            return False
    raise PriceLimitRuleError("MISSING_ST_STATUS", "is_st must be an explicit boolean")


def _trade_status(row: dict) -> str:
    value = row.get("trade_status")
    if value is None or not str(value).strip():
        raise PriceLimitRuleError("MISSING_TRADE_STATUS", "trade_status is required")
    if isinstance(value, bool):
        return "TRADING" if value else "SUSPENDED"
    if isinstance(value, int) and value in (0, 1):
        return "TRADING" if value == 1 else "SUSPENDED"
    if isinstance(value, float) and value in (0.0, 1.0):
        return "TRADING" if value == 1.0 else "SUSPENDED"
    status = str(value).strip().upper()
    aliases = {
        "1": "TRADING", "0": "SUSPENDED",
        "TRADE": "TRADING", "TRADING": "TRADING", "NORMAL": "TRADING",
        "ACTIVE": "TRADING", "交易": "TRADING", "正常": "TRADING",
        "SUSPENDED": "SUSPENDED", "HALTED": "SUSPENDED", "PAUSED": "SUSPENDED",
        "停牌": "SUSPENDED", "SUSPEND": "SUSPENDED",
    }
    if status not in aliases:
        raise PriceLimitRuleError("UNSUPPORTED_TRADE_STATUS", f"unsupported trade_status {value!r}")
    return aliases[status]


def _special_price_limit_event(row: dict) -> str | None:
    value = row.get("special_price_limit_status", row.get("price_limit_event"))
    if value is None or value is False:
        return None
    if value is True:
        raise PriceLimitRuleError(
            "UNSUPPORTED_SPECIAL_PRICE_LIMIT_EVENT",
            "special price-limit event requires an explicitly supported exchange rule",
        )
    normalized = str(value).strip().upper()
    if normalized in {"", "NONE", "NORMAL", "ORDINARY", "IPO"}:
        return None
    raise PriceLimitRuleError(
        "UNSUPPORTED_SPECIAL_PRICE_LIMIT_EVENT",
        f"special price-limit event {value!r} is outside rule-set coverage",
    )


def _rule_for(exchange: str, day: date, is_st: bool) -> tuple[str, Decimal, int]:
    if day < date(2006, 7, 1):
        raise PriceLimitRuleError(
            "RULE_DATE_NOT_COVERED", f"no versioned rule is enabled before 2006-07-01: {day}"
        )
    if day > RULES_VERIFIED_THROUGH:
        raise PriceLimitRuleError(
            "RULE_DATE_NOT_COVERED",
            f"rule set {RULE_SET_VERSION} is verified only through {RULES_VERIFIED_THROUGH}",
        )
    if day >= date(2026, 7, 6):
        return (f"{exchange}-MAINBOARD-A-2026-07-06", Decimal("0.10"), 5)
    if day >= date(2023, 4, 10):
        return (f"{exchange}-MAINBOARD-A-2023-04-10", Decimal("0.05") if is_st else Decimal("0.10"), 5)
    # The 2006 rules establish 10% ordinary / 5% ST.  Later official rule
    # amendments retain those main-board ratios until the effective dates above.
    return (f"{exchange}-MAINBOARD-A-2006-07-01", Decimal("0.05") if is_st else Decimal("0.10"), 1)


def _round_limit(reference: Decimal, rate: Decimal, sign: int) -> Decimal:
    raw = reference * (Decimal(1) + rate * sign)
    rounded = raw.quantize(TICK_SIZE, rounding=ROUND_HALF_UP)
    if abs(rounded - reference) < TICK_SIZE:
        rounded = reference + TICK_SIZE * sign
    if rounded < TICK_SIZE:
        rounded = TICK_SIZE
    return rounded.quantize(TICK_SIZE, rounding=ROUND_HALF_UP)


def derive_price_limit_rows(
        rows: Iterable[dict], *, trading_calendar: Iterable[Any] | dict[str, Iterable[Any]] | None = None,
) -> list[dict]:
    """Return read-layer copies enriched with effective upper/lower prices.

    ``trading_calendar`` must cover every listing date through each row date.
    When omitted, the complete session set is inferred from the supplied row
    dates; missing listing sessions therefore fail closed.
    """
    raw_rows = [dict(row) for row in rows]
    if not raw_rows:
        return []

    normalized: list[dict] = []
    by_security: dict[tuple[str, str], list[tuple[date, int, dict]]] = {}
    calendars: dict[str, set[date]] = {"SSE": set(), "SZSE": set()}
    for index, row in enumerate(raw_rows):
        day = _as_date(row.get("date", row.get("timestamp")), "date")
        exchange, code = _normalize_code(row)
        board = _main_board(exchange, code, day)
        listing_day = _as_date(row.get("listing_date"), "listing_date")
        if day < listing_day:
            raise PriceLimitRuleError("DATE_BEFORE_LISTING", f"{code} {day} precedes listing date {listing_day}")
        st = _is_st(row.get("is_st"))
        status = _trade_status(row)
        special_event = _special_price_limit_event(row)
        item = {"date": day, "exchange": exchange, "code": code, "board": board,
                "listing_date": listing_day, "is_st": st, "trade_status": status,
                "special_price_limit_event": special_event,
                "index": index, "row": row}
        normalized.append(item)
        calendars[exchange].add(day)
        by_security.setdefault((exchange, code), []).append((day, index, row))

    if trading_calendar is not None:
        if isinstance(trading_calendar, dict):
            for exchange_raw, values in trading_calendar.items():
                exchange = {"SH": "SSE", "SSE": "SSE", "SZ": "SZSE", "SZSE": "SZSE"}.get(
                    str(exchange_raw).upper()
                )
                if exchange is None:
                    raise PriceLimitRuleError("UNSUPPORTED_EXCHANGE", f"calendar exchange {exchange_raw!r}")
                calendars[exchange].update(_as_date(value, "trading_calendar") for value in values)
        else:
            shared = {_as_date(value, "trading_calendar") for value in trading_calendar}
            for exchange in calendars:
                calendars[exchange].update(shared)

    previous_close_by_index: dict[int, Decimal] = {}
    prior_bar_by_security: dict[tuple[str, str], list[tuple[date, dict]]] = {}
    for key, values in by_security.items():
        values.sort(key=lambda item: (item[0], item[1]))
        sessions = sorted(calendars[key[0]])
        session_position = {value: index for index, value in enumerate(sessions)}
        for day, source_index, row in values:
            has_corporate_action = any(
                row.get(name) for name in ("ex_rights", "corporate_action", "corporate_actions")
            )
            if has_corporate_action:
                # Raw prior close is not necessarily the exchange's adjusted
                # reference price on an ex-date. Never silently substitute it.
                prior_raw = row.get("reference_close")
                if prior_raw is None:
                    raise PriceLimitRuleError(
                        "MISSING_ADJUSTED_REFERENCE_CLOSE",
                        f"{key[1]} {day} is marked for a corporate action and requires reference_close",
                    )
            else:
                prior_raw = next((row.get(name) for name in ("previous_close", "prev_close", "pre_close", "reference_close")
                                  if row.get(name) is not None), None)
            if prior_raw is None:
                prior_day = None
                if day in session_position and session_position[day] > 0:
                    prior_day = sessions[session_position[day] - 1]
                security_history = prior_bar_by_security.get(key, [])
                history_by_day = {candidate_day: candidate
                                  for candidate_day, candidate in security_history}
                prior_row, source_day = None, None
                if day in session_position:
                    for candidate_day in reversed(sessions[:session_position[day]]):
                        candidate = history_by_day.get(candidate_day)
                        # A missing row may be a data gap; only explicit
                        # suspension rows justify stepping back to an older
                        # traded close. Otherwise require a supplied reference.
                        if candidate is None:
                            break
                        if _trade_status(candidate) == "SUSPENDED":
                            continue
                        if candidate.get("close") is not None:
                            prior_row, source_day = candidate, candidate_day
                        break
                if prior_row is None:
                    # The row may be in an IPO no-limit session, where no
                    # previous close is part of the limit-price calculation.
                    # A restricted row fails closed in the output pass below.
                    prior_raw = None
                else:
                    prior_raw = prior_row.get("close")
                    if prior_raw is None:
                        raise PriceLimitRuleError("MISSING_PREVIOUS_CLOSE", f"prior close missing for {key[1]} {day}")
                    row["_price_limit_reference_source"] = (
                        "prior_session_close" if source_day == prior_day else "last_available_security_close"
                    )
                    row["_price_limit_reference_date"] = source_day.isoformat() if source_day else None
            else:
                row["_price_limit_reference_source"] = (
                    "reference_close" if has_corporate_action else next(
                        name for name in ("previous_close", "prev_close", "pre_close", "reference_close")
                        if row.get(name) is not None
                    )
                )
            try:
                previous_close_by_index[source_index] = _decimal(prior_raw, "previous_close")
            except PriceLimitRuleError:
                # IPO no-limit sessions do not need a previous close.  Leave
                # validation to the row-specific branch below.
                previous_close_by_index[source_index] = Decimal("0")
            prior_bar_by_security.setdefault(key, []).append((day, row))

    output = [dict(row) for row in raw_rows]
    for item in normalized:
        row = output[item["index"]]
        day, exchange, code = item["date"], item["exchange"], item["code"]
        listing_day = item["listing_date"]
        sessions = sorted(calendars[exchange])
        if day not in sessions or listing_day not in sessions:
            raise PriceLimitRuleError(
                "INCOMPLETE_LISTING_CALENDAR",
                f"{exchange} calendar must contain listing date {listing_day} and trade date {day}",
            )
        session_no = sum(listing_day <= session <= day for session in sessions)
        if session_no <= 0:
            raise PriceLimitRuleError("INCOMPLETE_LISTING_CALENDAR", f"cannot number listing sessions for {code} {day}")
        rule_id, rate, ipo_days = _rule_for(exchange, day, item["is_st"])
        # The five-session IPO rule took effect with the first registration
        # based main-board IPO on 2023-04-10; before that, only listing day
        # itself was without the ordinary daily price limit.
        if day >= date(2023, 4, 10):
            ipo_days = 5
        unrestricted = session_no <= ipo_days
        rule_source_ids = ["sse_2006" if exchange == "SSE" else "szse_2006"]
        if exchange == "SZSE":
            rule_source_ids.append("szse_code_map")
            if code[:3] in {"002", "003", "004"} and day >= date(2021, 4, 6):
                rule_source_ids.append("szse_sme_merge")
        if day >= date(2023, 4, 10):
            rule_source_ids.append("sse_2023" if exchange == "SSE" else "szse_2023")
        if day >= date(2026, 7, 6):
            rule_source_ids.append("sse_2026" if exchange == "SSE" else "szse_2026")
        basis = {
            "exchange": exchange, "code": code, "board": item["board"],
            "trade_date": day.isoformat(), "listing_date": listing_day.isoformat(),
            "listing_session_number": session_no,
            "listing_session_count_source": "exchange_calendar",
            "is_st": item["is_st"], "trade_status": item["trade_status"],
            "special_price_limit_event": item["special_price_limit_event"],
            "tick_size": str(TICK_SIZE), "price_rounding": "ROUND_HALF_UP_to_tick",
            "low_price_minimum_tick_rule": True,
            "rules_verified_through": RULES_VERIFIED_THROUGH.isoformat(),
            "rule_source_ids": rule_source_ids,
        }
        row.update({
            "trade_status": item["trade_status"],
            "upper_limit_price": None,
            "lower_limit_price": None,
            "price_limit_unrestricted": unrestricted,
            "price_limit_rule_id": rule_id,
            "price_limit_rule_set_id": RULE_SET_ID,
            "price_limit_rule_version": RULE_SET_VERSION,
            "price_limit_is_derived": True,
            "price_limit_basis": basis,
        })
        if unrestricted:
            row["price_limit_basis"] = basis | {
                "reason": "IPO_FIRST_SESSION" if ipo_days == 1 else "IPO_FIRST_FIVE_SESSIONS",
                "limit_prices": None,
            }
            continue
        reference = previous_close_by_index[item["index"]]
        if reference <= 0:
            raise PriceLimitRuleError("MISSING_PREVIOUS_CLOSE", f"{code} {day} requires previous close")
        if reference % TICK_SIZE != 0:
            raise PriceLimitRuleError(
                "INVALID_PREVIOUS_CLOSE_TICK", f"{code} {day} previous close {reference} is not on the 0.01 tick"
            )
        row["upper_limit_price"] = float(_round_limit(reference, rate, 1))
        row["lower_limit_price"] = float(_round_limit(reference, rate, -1))
        row["price_limit_basis"] = basis | {
            "previous_close": float(reference),
            "previous_close_source": row.pop("_price_limit_reference_source", "unknown"),
            "previous_close_date": row.pop("_price_limit_reference_date", None),
            "limit_rate": float(rate),
            "upper_limit_price": row["upper_limit_price"],
            "lower_limit_price": row["lower_limit_price"],
        }
    for row in output:
        row.pop("_price_limit_reference_source", None)
    return output


def assess_limit_market_state(row: dict, *, touch_policy: str = "block_on_touch") -> dict:
    """Separate price touches, visible book evidence, and the chosen policy."""
    if touch_policy not in {"block_on_touch", "allow_on_touch"}:
        raise PriceLimitRuleError("INVALID_LIMIT_TOUCH_POLICY", "use block_on_touch or allow_on_touch")
    upper = row.get("upper_limit_price")
    lower = row.get("lower_limit_price")
    high, low = row.get("high"), row.get("low")

    def finite(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    upper, lower, high, low = map(finite, (upper, lower, high, low))
    upper_touched = upper is not None and high is not None and high >= upper - 1e-9
    lower_touched = lower is not None and low is not None and low <= lower + 1e-9
    ask = finite(row.get("ask_volume"))
    bid = finite(row.get("bid_volume"))
    explicit_buy = row.get("limit_buy_locked") if "limit_buy_locked" in row else None
    explicit_sell = row.get("limit_sell_locked") if "limit_sell_locked" in row else None
    buy_locked = bool(explicit_buy) if explicit_buy is not None else (
        upper_touched and ask is not None and ask <= 0
    )
    sell_locked = bool(explicit_sell) if explicit_sell is not None else (
        lower_touched and bid is not None and bid <= 0
    )
    buy_unknown = upper_touched and ask is None and explicit_buy is None
    sell_unknown = lower_touched and bid is None and explicit_sell is None
    conservative_buy = buy_unknown and touch_policy == "block_on_touch"
    conservative_sell = sell_unknown and touch_policy == "block_on_touch"
    return {
        "upper_limit_touched": bool(upper_touched),
        "lower_limit_touched": bool(lower_touched),
        "buy_book_lock_evidence": "visible_ask_volume_zero" if buy_locked and ask is not None else (
            "explicit_lock_state" if buy_locked else "unknown_no_bid_ask" if buy_unknown else "not_locked_or_not_touched"
        ),
        "sell_book_lock_evidence": "visible_bid_volume_zero" if sell_locked and bid is not None else (
            "explicit_lock_state" if sell_locked else "unknown_no_bid_ask" if sell_unknown else "not_locked_or_not_touched"
        ),
        "limit_touch_policy": touch_policy,
        "block_buy": bool(buy_locked or conservative_buy),
        "block_sell": bool(sell_locked or conservative_sell),
        "buy_block_reason": "LIMIT_LOCKED" if buy_locked else (
            "LIMIT_TOUCH_CONSERVATIVE" if conservative_buy else None
        ),
        "sell_block_reason": "LIMIT_LOCKED" if sell_locked else (
            "LIMIT_TOUCH_CONSERVATIVE" if conservative_sell else None
        ),
        "ohlc_proves_book_locked": False,
        "evidence_limitations": [
            name for name, unknown in (("buy", buy_unknown), ("sell", sell_unknown)) if unknown
        ],
    }
