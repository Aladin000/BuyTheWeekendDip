"""Simulate the long-only strategies for the development and held-out stages.

Accepted negative-shock events are turned into trades for each strategy
account, after checking the accepted inference run and its outcome and feature
inputs. The development entry point reads development prices only; the
held-out stage reuses the same accounting with the held-out split.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src import paths
from src.acquisition import verify_manifest_checksum, write_manifest_checksum
from src.artifacts import (
    RunLog,
    build_manifest,
    matches_active_protocol,
    create_exclusive,
    create_exclusive_bytes,
    create_run_log,
    read_manifest,
    run_relative_dir,
    sha256_bytes,
    sha256_file,
)
from src.config import ConfigNode, load_config
from src.count_audit import ACCEPTED_FEATURE_MANIFEST
from src.features import FEATURE_COLUMNS
from src.inference import (
    ACCEPTED_OUTCOME_MANIFEST,
    EventRecord,
    read_event_records,
    verify_inference_inputs,
    verify_inference_inventory,
)

STAGE = "8. Development backtest"
MODE = "backtest"
DEVELOPMENT_PERIOD = "development"
HELD_OUT_PERIOD = "held_out"
UTC = timezone.utc

ACCEPTED_INFERENCE_MANIFEST = Path(
    "outputs/runs/20260908T152939.777879Z-development-0f73422e-0d444a/"
    "development-inference.manifest.json"
)
ACCEPTED_INFERENCE_MANIFEST_SHA256 = (
    "dafa13aa144ff4ab25e08fc6d5efdbef93e4de08a0dc4462601f5368c2b4be90"
)

STRATEGY_ACCOUNTS: tuple[str, ...] = (
    "all_shock",
    "weekend",
    "low_activity",
    "combined",
    "weekday_shock",
)
COST_SCENARIOS: tuple[str, ...] = ("low", "primary", "high")
CONCENTRATION_TOP_N = 5

TRADE_AUDIT_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "event_id",
    "signal_time",
    "weekend",
    "activity_percentile_90d",
    "low_activity",
    "strategy_filter_passed",
    "execution_available",
    "position_available",
    "strategy_eligible",
    "rejection_reason",
    "entry_time",
    "exit_time",
    "entry_price_raw",
    "exit_price_raw",
    "hold_hours",
)
TRADE_RESULT_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "cost_scenario",
    "event_id",
    "signal_time",
    "entry_time",
    "exit_time",
    "entry_price_raw",
    "exit_price_raw",
    "fee_rate",
    "slippage_rate",
    "entry_fill",
    "exit_fill",
    "gross_return",
    "net_return",
    "capital_before",
    "entry_units",
    "capital_after",
    "notional_turnover",
)
METRIC_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "cost_scenario",
    "trade_count",
    "gross_cumulative_return",
    "net_cumulative_return",
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "turnover",
    "time_invested_fraction",
    "average_net_trade_return",
    "median_net_trade_return",
    "net_hit_rate",
    "largest_net_winner",
    "largest_net_loser",
    "average_gross_trade_return",
    "median_gross_trade_return",
    "gross_hit_rate",
    "largest_gross_winner",
    "largest_gross_loser",
    "break_even_cost_per_side",
    "break_even_status",
    "largest_trade_share_absolute_pnl",
    "top5_trade_share_absolute_pnl",
    "elapsed_calendar_days",
    "split_hours",
    "annualised_metric_label",
)
DAILY_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "cost_scenario",
    "calendar_date",
    "equity",
    "daily_return",
)
BENCHMARK_COLUMNS: tuple[str, ...] = (
    "benchmark_name",
    "cost_scenario",
    "entry_time",
    "exit_time",
    "entry_price_raw",
    "exit_price_raw",
    "gross_cumulative_return",
    "net_cumulative_return",
    "exposure_note",
)
INVENTORY_COLUMNS: tuple[str, ...] = (
    "artifact_path",
    "sha256",
    "record_count",
    "artifact_type",
    "column_schema_json",
)


def _equity_columns() -> tuple[str, ...]:
    columns = ["mark_time"]
    for name in STRATEGY_ACCOUNTS:
        columns.extend(
            (
                f"{name}_gross_equity",
                f"{name}_primary_net_equity",
                f"{name}_primary_drawdown",
                f"{name}_position_open",
                f"{name}_mark_source",
            )
        )
    return tuple(columns)


EQUITY_COLUMNS = _equity_columns()
TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "development_trade_audit.csv": TRADE_AUDIT_COLUMNS,
    "development_trade_results.csv": TRADE_RESULT_COLUMNS,
    "development_strategy_metrics.csv": METRIC_COLUMNS,
    "development_equity_hourly.csv": EQUITY_COLUMNS,
    "development_equity_daily.csv": DAILY_COLUMNS,
    "development_benchmarks.csv": BENCHMARK_COLUMNS,
}
FIGURE_NAMES = {
    "development_equity_curves.png",
    "development_drawdowns.png",
}


class BacktestError(RuntimeError):
    """Raised when backtest inputs, accounting, or saved outputs are invalid."""


@dataclass(frozen=True)
class BacktestHour:
    open_time: datetime
    close_time: datetime
    close_time_raw: int
    segment_id: int
    open_price: float
    close_price: float


@dataclass(frozen=True)
class TradePlan:
    strategy_name: str
    event_id: str
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    exit_boundary: datetime
    entry_price: float
    exit_price: float
    weekend: bool
    activity_percentile: float | None
    low_activity: bool | None


@dataclass(frozen=True)
class TradeResult:
    plan: TradePlan
    scenario: str
    fee: float
    slippage: float
    entry_fill: float
    exit_fill: float
    gross_return: float
    net_return: float
    capital_before: float
    units: float
    capital_after: float
    notional_turnover: float


@dataclass(frozen=True)
class EquityPoint:
    mark_time: datetime
    equity: float
    position_open: bool
    source: str
    drawdown: float


@dataclass(frozen=True)
class BacktestBundle:
    audit_rows: tuple[dict[str, object], ...]
    trade_rows: tuple[dict[str, object], ...]
    metric_rows: tuple[dict[str, object], ...]
    equity_rows: tuple[dict[str, object], ...]
    daily_rows: tuple[dict[str, object], ...]
    benchmark_rows: tuple[dict[str, object], ...]
    figures: tuple[tuple[str, bytes], ...]
    negative_event_count: int
    executed_trade_counts: Mapping[str, int]


@dataclass(frozen=True)
class DevelopmentBacktestResult:
    run_id: str
    manifest_relative_path: Path
    manifest_checksum_relative_path: Path
    log_relative_path: Path
    inventory_relative_path: Path
    negative_events: int
    trade_counts: Mapping[str, int]


def _parse_utc(text: str, field: str, row_number: int) -> datetime:
    if not text.endswith("Z"):
        raise BacktestError(f"feature row {row_number} field {field} is not UTC")
    try:
        value = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        raise BacktestError(
            f"feature row {row_number} field {field} is not a timestamp"
        ) from None
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise BacktestError(f"feature row {row_number} field {field} is not UTC")
    return value


def _parse_price(text: str, field: str, row_number: int) -> float:
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        raise BacktestError(
            f"feature row {row_number} field {field} is not numeric"
        ) from None
    value = float(decimal)
    if not decimal.is_finite() or value <= 0 or not math.isfinite(value):
        raise BacktestError(f"feature row {row_number} field {field} is invalid")
    return value


def read_analysis_hours(
    relative_path: Path,
    expected_rows: int,
    config: ConfigNode,
    *,
    analysis_period: str,
) -> tuple[BacktestHour, ...]:
    """Parse prices only for the explicitly selected analysis period."""
    if analysis_period not in {DEVELOPMENT_PERIOD, HELD_OUT_PERIOD}:
        raise BacktestError("backtest reader requires development or held_out")
    source = paths.require_file(paths.resolve(relative_path))
    output: list[BacktestHour] = []
    total = 0
    previous_open: datetime | None = None
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = tuple(next(reader))
            except StopIteration:
                raise BacktestError("pre-outcome feature artifact is empty") from None
            if header != FEATURE_COLUMNS:
                raise BacktestError("feature artifact schema does not match the feature stage")
            index = {name: header.index(name) for name in header}
            for row_number, cells in enumerate(reader, start=2):
                total += 1
                if len(cells) != len(header):
                    raise BacktestError(
                        f"feature row {row_number} has {len(cells)} fields; expected {len(header)}"
                    )
                period = cells[index["analysis_period"]]
                if period not in {"warmup", "development", "held_out"}:
                    raise BacktestError(
                        f"feature row {row_number} has invalid analysis period {period!r}"
                    )
                open_time = _parse_utc(cells[index["open_time"]], "open_time", row_number)
                if previous_open is not None and open_time <= previous_open:
                    raise BacktestError("feature timestamps are not strictly chronological")
                previous_open = open_time
                if period != analysis_period:
                    continue
                close_time = _parse_utc(
                    cells[index["close_time"]], "close_time", row_number
                )
                try:
                    close_time_raw = int(cells[index["close_time_raw"]])
                    segment_id = int(cells[index["contiguous_segment_id"]])
                except ValueError:
                    raise BacktestError(
                        f"feature row {row_number} has invalid integer identity"
                    ) from None
                output.append(
                    BacktestHour(
                        open_time=open_time,
                        close_time=close_time,
                        close_time_raw=close_time_raw,
                        segment_id=segment_id,
                        open_price=_parse_price(
                            cells[index["open"]], "open", row_number
                        ),
                        close_price=_parse_price(
                            cells[index["close"]], "close", row_number
                        ),
                    )
                )
    except (OSError, UnicodeError, csv.Error) as error:
        raise BacktestError(f"cannot read feature artifact: {error}") from None
    if total != expected_rows:
        raise BacktestError(
            f"feature row count is {total}; expected {expected_rows} from its manifest"
        )
    split = getattr(config.splits, analysis_period)
    start = split.start_inclusive
    end = split.end_exclusive
    if not output or output[0].open_time != start or output[-1].open_time >= end:
        raise BacktestError(
            f"{analysis_period} price coverage does not match the locked split"
        )
    if any(not start <= row.open_time < end for row in output):
        raise BacktestError(
            f"non-{analysis_period} price entered the backtest reader"
        )
    return tuple(output)


def read_development_hours(
    relative_path: Path, expected_rows: int, config: ConfigNode
) -> tuple[BacktestHour, ...]:
    """Parse development prices; other rows are only counted and order-checked."""
    return read_analysis_hours(
        relative_path,
        expected_rows,
        config,
        analysis_period=DEVELOPMENT_PERIOD,
    )


def _strategy_filter(event: EventRecord, strategy: str) -> tuple[bool, str]:
    if strategy == "all_shock":
        return True, ""
    if strategy == "weekend":
        return (True, "") if event.weekend else (False, "weekend_filter_not_met")
    if strategy == "weekday_shock":
        return (True, "") if not event.weekend else (False, "weekday_filter_not_met")
    if event.activity_percentile is None or event.low_activity is None:
        return False, "activity_unavailable"
    if strategy == "low_activity":
        return (True, "") if event.low_activity else (False, "low_activity_filter_not_met")
    if strategy == "combined":
        if not event.weekend:
            return False, "weekend_filter_not_met"
        return (True, "") if event.low_activity else (False, "low_activity_filter_not_met")
    raise BacktestError(f"unknown strategy {strategy!r}")


def _execution_plan(
    event: EventRecord,
    strategy: str,
    hours: Sequence[BacktestHour],
    index_by_close: Mapping[datetime, int],
    config: ConfigNode,
    *,
    analysis_period: str = DEVELOPMENT_PERIOD,
) -> tuple[TradePlan | None, str]:
    source_index = index_by_close.get(event.signal_close_time)
    if source_index is None:
        raise BacktestError(
            f"event {event.event_id} does not match an {analysis_period} bar"
        )
    entry_index = source_index + config.strategy.entry_offset_bars
    exit_index = source_index + config.strategy.exit_offset_bars
    if exit_index >= len(hours):
        return None, f"crosses_{analysis_period}_boundary"
    signal = hours[source_index]
    if (
        signal.open_time != event.signal_open_time
        or signal.segment_id != event.segment_id
        or not math.isclose(
            signal.close_price,
            event.signal_close,
            rel_tol=config.reproducibility.rtol,
            abs_tol=config.reproducibility.atol,
        )
    ):
        raise BacktestError(
            f"event {event.event_id} does not reconcile with its source bar"
        )
    for offset in range(1, config.strategy.exit_offset_bars + 1):
        candidate = hours[source_index + offset]
        if (
            candidate.open_time != signal.open_time + timedelta(hours=offset)
            or candidate.segment_id != signal.segment_id
        ):
            return None, "missing_or_noncontiguous_execution_bar"
    entry = hours[entry_index]
    exit_bar = hours[exit_index]
    exit_boundary = exit_bar.open_time + timedelta(hours=1)
    split = getattr(config.splits, analysis_period)
    if exit_boundary > split.end_exclusive:
        return None, f"crosses_{analysis_period}_boundary"
    return (
        TradePlan(
            strategy_name=strategy,
            event_id=event.event_id,
            signal_time=event.signal_close_time,
            entry_time=entry.open_time,
            exit_time=exit_bar.close_time,
            exit_boundary=exit_boundary,
            entry_price=entry.open_price,
            exit_price=exit_bar.close_price,
            weekend=event.weekend,
            activity_percentile=event.activity_percentile,
            low_activity=event.low_activity,
        ),
        "",
    )


def build_trade_plans(
    events: Sequence[EventRecord],
    hours: Sequence[BacktestHour],
    config: ConfigNode,
    *,
    analysis_period: str = DEVELOPMENT_PERIOD,
) -> tuple[tuple[dict[str, object], ...], dict[str, tuple[TradePlan, ...]]]:
    if analysis_period not in {DEVELOPMENT_PERIOD, HELD_OUT_PERIOD}:
        raise BacktestError("trade plans require development or held_out")
    negative = sorted(
        (event for event in events if event.sign == "negative"),
        key=lambda event: event.signal_close_time,
    )
    if len({event.event_id for event in negative}) != len(negative):
        raise BacktestError("negative event identities are not unique")
    if tuple(config.strategy.variants) != STRATEGY_ACCOUNTS[:4]:
        raise BacktestError("configured strategy variants do not match Stage 8")
    index_by_close = {row.close_time: index for index, row in enumerate(hours)}
    if len(index_by_close) != len(hours):
        raise BacktestError(
            f"{analysis_period} bars contain duplicate close timestamps"
        )
    audits: list[dict[str, object]] = []
    accepted: dict[str, list[TradePlan]] = {name: [] for name in STRATEGY_ACCOUNTS}
    for strategy in STRATEGY_ACCOUNTS:
        active_until: datetime | None = None
        for event in negative:
            filter_passed, filter_reason = _strategy_filter(event, strategy)
            plan: TradePlan | None = None
            execution_available = False
            position_available = False
            reason = filter_reason
            if filter_passed:
                plan, reason = _execution_plan(
                    event,
                    strategy,
                    hours,
                    index_by_close,
                    config,
                    analysis_period=analysis_period,
                )
                execution_available = plan is not None
                if plan is not None:
                    position_available = (
                        active_until is None or plan.entry_time >= active_until
                    )
                    if not position_available:
                        reason = "position_already_open"
                    else:
                        active_until = plan.exit_boundary
                        accepted[strategy].append(plan)
            eligible = plan is not None and position_available
            audits.append(
                {
                    "strategy_name": strategy,
                    "event_id": event.event_id,
                    "signal_time": event.signal_close_time,
                    "weekend": event.weekend,
                    "activity_percentile_90d": event.activity_percentile,
                    "low_activity": event.low_activity,
                    "strategy_filter_passed": filter_passed,
                    "execution_available": execution_available,
                    "position_available": position_available,
                    "strategy_eligible": eligible,
                    "rejection_reason": "" if eligible else reason,
                    "entry_time": None if plan is None else plan.entry_time,
                    "exit_time": None if plan is None else plan.exit_time,
                    "entry_price_raw": None if plan is None else plan.entry_price,
                    "exit_price_raw": None if plan is None else plan.exit_price,
                    "hold_hours": config.strategy.hold_hours if eligible else None,
                }
            )
    if len(audits) != len(negative) * len(STRATEGY_ACCOUNTS):
        raise BacktestError("trade audit does not reconcile with accepted negative events")
    return tuple(audits), {name: tuple(value) for name, value in accepted.items()}


def simulate_trades(
    plans: Sequence[TradePlan],
    scenario: str,
    fee: float,
    slippage: float,
    initial_capital: float,
) -> tuple[TradeResult, ...]:
    if fee < 0 or slippage < 0 or fee >= 1 or slippage >= 1:
        raise BacktestError("cost rates must lie in [0, 1)")
    if initial_capital <= 0:
        raise BacktestError("initial capital must be positive")
    capital = float(initial_capital)
    output: list[TradeResult] = []
    previous_exit: datetime | None = None
    for plan in plans:
        if previous_exit is not None and plan.entry_time < previous_exit:
            raise BacktestError("trade plans overlap")
        previous_exit = plan.exit_boundary
        entry_fill = plan.entry_price * (1 + slippage)
        exit_fill = plan.exit_price * (1 - slippage)
        units = capital * (1 - fee) / entry_fill
        after = units * exit_fill * (1 - fee)
        if not all(
            math.isfinite(value) and value > 0
            for value in (entry_fill, exit_fill, units, after)
        ):
            raise BacktestError("trade accounting produced a non-positive value")
        output.append(
            TradeResult(
                plan=plan,
                scenario=scenario,
                fee=fee,
                slippage=slippage,
                entry_fill=entry_fill,
                exit_fill=exit_fill,
                gross_return=plan.exit_price / plan.entry_price - 1,
                net_return=after / capital - 1,
                capital_before=capital,
                units=units,
                capital_after=after,
                notional_turnover=units * entry_fill + units * exit_fill,
            )
        )
        capital = after
    return tuple(output)


def build_equity_curve(
    results: Sequence[TradeResult],
    hours: Sequence[BacktestHour],
    start: datetime,
    end: datetime,
    initial_capital: float,
) -> tuple[EquityPoint, ...]:
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise BacktestError("equity boundaries must be ordered and timezone aware")
    entries = {result.plan.entry_time: result for result in results}
    exits = {result.plan.exit_boundary: result for result in results}
    if len(entries) != len(results) or len(exits) != len(results):
        raise BacktestError("trade entries or exits are duplicated")
    hour_by_open = {row.open_time: row for row in hours}
    points: list[EquityPoint] = []
    capital = float(initial_capital)
    position: TradeResult | None = None
    running_max = capital
    total_hours = int((end - start).total_seconds() // 3600)
    for offset in range(total_hours + 1):
        moment = start + timedelta(hours=offset)
        source = "initial_boundary" if offset == 0 else "cash_calendar_mark"
        if position is not None and moment == position.plan.exit_boundary:
            capital = position.capital_after
            position = None
            source = "exit_cash"
        entering = entries.get(moment)
        if entering is not None:
            if position is not None:
                raise BacktestError("equity curve encountered concurrent positions")
            if not math.isclose(
                entering.capital_before, capital, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise BacktestError("trade capital does not reconcile with equity")
            position = entering
            equity = position.units * position.plan.entry_price
            source = "market_open"
        elif position is not None:
            marked_hour = hour_by_open.get(moment - timedelta(hours=1))
            if marked_hour is None:
                raise BacktestError("an invested hour lacks an observed price")
            equity = position.units * marked_hour.close_price
            source = "market_close"
        else:
            equity = capital
        if not math.isfinite(equity) or equity <= 0:
            raise BacktestError("equity curve contains a non-positive value")
        running_max = max(running_max, equity)
        points.append(
            EquityPoint(moment, equity, position is not None, source, equity / running_max - 1)
        )
    if position is not None:
        raise BacktestError("a trade remains open at the split end")
    expected_terminal = results[-1].capital_after if results else initial_capital
    if not math.isclose(
        points[-1].equity, expected_terminal, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise BacktestError("terminal trade capital and equity curve disagree")
    return tuple(points)


def daily_equity(curve: Sequence[EquityPoint]) -> tuple[dict[str, object], ...]:
    if len(curve) < 25:
        raise BacktestError("equity curve is too short for daily metrics")
    by_time = {point.mark_time: point for point in curve}
    start = curve[0].mark_time
    end = curve[-1].mark_time
    days = int((end - start).total_seconds() // 86400)
    prior = curve[0].equity
    output: list[dict[str, object]] = []
    for offset in range(1, days + 1):
        boundary = start + timedelta(days=offset)
        point = by_time.get(boundary)
        if point is None:
            raise BacktestError("hourly equity lacks a UTC day-end boundary")
        output.append(
            {
                "calendar_date": (boundary - timedelta(days=1)).date(),
                "equity": point.equity,
                "daily_return": point.equity / prior - 1,
            }
        )
        prior = point.equity
    return tuple(output)


def _break_even(plans: Sequence[TradePlan]) -> tuple[float | None, str]:
    if not plans:
        return None, "not_applicable_no_trades"
    gross_wealth = math.prod(plan.exit_price / plan.entry_price for plan in plans)
    if gross_wealth <= 1:
        return None, "no_positive_break_even_cost"
    cost = 1 - gross_wealth ** (-1 / (2 * len(plans)))
    return cost, "positive_solution"


def performance_record(
    strategy: str,
    scenario: str,
    plans: Sequence[TradePlan],
    gross_results: Sequence[TradeResult],
    net_results: Sequence[TradeResult],
    curve: Sequence[EquityPoint],
    daily: Sequence[Mapping[str, object]],
    config: ConfigNode,
    *,
    analysis_period: str = DEVELOPMENT_PERIOD,
) -> dict[str, object]:
    if analysis_period not in {DEVELOPMENT_PERIOD, HELD_OUT_PERIOD}:
        raise BacktestError("performance record requires development or held_out")
    n = len(plans)
    if len(gross_results) != n or len(net_results) != n:
        raise BacktestError("trade result counts do not reconcile")
    split = getattr(config.splits, analysis_period)
    days = (split.end_exclusive - split.start_inclusive).total_seconds() / 86400
    split_hours = int(days * 24)
    net_terminal = curve[-1].equity
    gross_terminal = gross_results[-1].capital_after if n else config.strategy.initial_capital
    daily_returns = np.asarray([float(row["daily_return"]) for row in daily])
    if len(daily_returns) != int(days):
        raise BacktestError("daily equity count does not match elapsed split days")
    daily_sd = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    annual_volatility = daily_sd * math.sqrt(config.performance.sharpe_periods_per_year)
    sharpe = (
        None
        if daily_sd <= config.reproducibility.atol
        else math.sqrt(config.performance.sharpe_periods_per_year)
        * float(np.mean(daily_returns) - config.performance.risk_free_rate)
        / daily_sd
    )
    net_returns = [result.net_return for result in net_results]
    gross_returns = [result.gross_return for result in gross_results]
    pnl = [result.capital_after - result.capital_before for result in net_results]
    absolute_total = sum(abs(value) for value in pnl)
    ordered_absolute = sorted((abs(value) for value in pnl), reverse=True)
    break_even, break_even_status = _break_even(plans)
    day_end_mean = float(np.mean([float(row["equity"]) for row in daily]))
    turnover = sum(result.notional_turnover for result in net_results) / day_end_mean if n else 0.0

    def mean_or_none(values: Sequence[float]) -> float | None:
        return None if not values else float(np.mean(values))

    def median_or_none(values: Sequence[float]) -> float | None:
        return None if not values else float(np.median(values))

    return {
        "strategy_name": strategy,
        "cost_scenario": scenario,
        "trade_count": n,
        "gross_cumulative_return": gross_terminal / config.strategy.initial_capital - 1,
        "net_cumulative_return": net_terminal / config.strategy.initial_capital - 1,
        "annualised_return": 0.0 if n == 0 else (net_terminal / config.strategy.initial_capital) ** (365 / days) - 1,
        "annualised_volatility": 0.0 if n == 0 else annual_volatility,
        "sharpe_ratio": None if n == 0 else sharpe,
        "maximum_drawdown": 0.0 if n == 0 else min(point.drawdown for point in curve),
        "turnover": turnover,
        "time_invested_fraction": n * config.strategy.hold_hours / split_hours,
        "average_net_trade_return": mean_or_none(net_returns),
        "median_net_trade_return": median_or_none(net_returns),
        "net_hit_rate": None if not net_returns else sum(value > 0 for value in net_returns) / n,
        "largest_net_winner": None if not net_returns else max(net_returns),
        "largest_net_loser": None if not net_returns else min(net_returns),
        "average_gross_trade_return": mean_or_none(gross_returns),
        "median_gross_trade_return": median_or_none(gross_returns),
        "gross_hit_rate": None if not gross_returns else sum(value > 0 for value in gross_returns) / n,
        "largest_gross_winner": None if not gross_returns else max(gross_returns),
        "largest_gross_loser": None if not gross_returns else min(gross_returns),
        "break_even_cost_per_side": break_even,
        "break_even_status": break_even_status,
        "largest_trade_share_absolute_pnl": None if absolute_total <= config.reproducibility.atol else ordered_absolute[0] / absolute_total,
        "top5_trade_share_absolute_pnl": None if absolute_total <= config.reproducibility.atol else sum(ordered_absolute[:CONCENTRATION_TOP_N]) / absolute_total,
        "elapsed_calendar_days": days,
        "split_hours": split_hours,
        "annualised_metric_label": "descriptive_highly_unstable" if n < config.strategy.sparse_trade_label_threshold else "reported_with_sampling_uncertainty",
    }


def _trade_result_row(result: TradeResult) -> dict[str, object]:
    return {
        "strategy_name": result.plan.strategy_name,
        "cost_scenario": result.scenario,
        "event_id": result.plan.event_id,
        "signal_time": result.plan.signal_time,
        "entry_time": result.plan.entry_time,
        "exit_time": result.plan.exit_time,
        "entry_price_raw": result.plan.entry_price,
        "exit_price_raw": result.plan.exit_price,
        "fee_rate": result.fee,
        "slippage_rate": result.slippage,
        "entry_fill": result.entry_fill,
        "exit_fill": result.exit_fill,
        "gross_return": result.gross_return,
        "net_return": result.net_return,
        "capital_before": result.capital_before,
        "entry_units": result.units,
        "capital_after": result.capital_after,
        "notional_turnover": result.notional_turnover,
    }


def _scenario_cost(config: ConfigNode, scenario: str) -> tuple[float, float]:
    if scenario == "gross":
        return 0.0, 0.0
    if scenario not in COST_SCENARIOS:
        raise BacktestError(f"unknown cost scenario {scenario!r}")
    node = getattr(config.costs.scenarios, scenario)
    return node.fee_per_side, node.slippage_per_side


def benchmark_rows(
    hours: Sequence[BacktestHour],
    config: ConfigNode,
    *,
    analysis_period: str = DEVELOPMENT_PERIOD,
) -> tuple[dict[str, object], ...]:
    if analysis_period not in {DEVELOPMENT_PERIOD, HELD_OUT_PERIOD}:
        raise BacktestError("benchmarks require development or held_out")
    split = getattr(config.splits, analysis_period)
    first = hours[0]
    last = hours[-1]
    output: list[dict[str, object]] = [
        {
            "benchmark_name": "cash",
            "cost_scenario": "not_applicable",
            "entry_time": split.start_inclusive,
            "exit_time": split.end_exclusive,
            "entry_price_raw": None,
            "exit_price_raw": None,
            "gross_cumulative_return": 0.0,
            "net_cumulative_return": 0.0,
            "exposure_note": "zero_yield_usdt_cash",
        }
    ]
    gross = last.close_price / first.open_price - 1
    for scenario in COST_SCENARIOS:
        fee, slip = _scenario_cost(config, scenario)
        factor = (1 - fee) ** 2 * last.close_price * (1 - slip) / (
            first.open_price * (1 + slip)
        )
        output.append(
            {
                "benchmark_name": "btc_buy_and_hold",
                "cost_scenario": scenario,
                "entry_time": first.open_time,
                "exit_time": last.close_time,
                "entry_price_raw": first.open_price,
                "exit_price_raw": last.close_price,
                "gross_cumulative_return": gross,
                "net_cumulative_return": factor - 1,
                "exposure_note": "different_exposure_not_risk_matched",
            }
        )
    return tuple(output)


def _format_value(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise BacktestError("cannot serialise a non-finite number")
        return format(number, ".17g")
    return str(value)


def render_records(records: Sequence[Mapping[str, object]], schema: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(schema), lineterminator="\n")
    writer.writeheader()
    for record in records:
        if set(record) != set(schema):
            raise BacktestError("output record does not match its declared schema")
        writer.writerow({name: _format_value(record[name]) for name in schema})
    return buffer.getvalue()


def render_figures(
    curves: Mapping[tuple[str, str], Sequence[EquityPoint]],
    *,
    analysis_period: str = DEVELOPMENT_PERIOD,
    label: str = "Development",
) -> tuple[tuple[str, bytes], ...]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    metadata = {"Software": f"matplotlib {matplotlib.__version__}"}
    colours = {
        "all_shock": "#335c99",
        "weekend": "#c43c39",
        "low_activity": "#d28b26",
        "combined": "#2d7f5e",
    }

    def save(name: str, figure) -> tuple[str, bytes]:
        buffer = io.BytesIO()
        figure.savefig(
            buffer, format="png", dpi=160, bbox_inches="tight", metadata=metadata
        )
        plt.close(figure)
        return name, buffer.getvalue()

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for strategy in STRATEGY_ACCOUNTS[:4]:
        curve = curves[(strategy, "primary")]
        axes[0].plot(
            [point.mark_time for point in curve],
            [point.equity for point in curve],
            label=strategy.replace("_", " ").title(),
            color=colours[strategy],
            linewidth=1.2,
        )
    axes[0].set(title=f"{label} primary-net equity", ylabel="Normalised wealth")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.2)
    for scenario, style, legend_label in (
        ("gross", "--", "Weekend gross"),
        ("primary", "-", "Weekend primary net"),
    ):
        curve = curves[("weekend", scenario)]
        axes[1].plot(
            [point.mark_time for point in curve],
            [point.equity for point in curve],
            linestyle=style,
            label=legend_label,
            color=colours["weekend"],
            linewidth=1.4,
        )
    axes[1].set(
        title="Titled strategy: effect of primary trading costs",
        ylabel="Normalised wealth",
        xlabel="UTC date",
    )
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    equity_figure = save(f"{analysis_period}_equity_curves.png", figure)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    for strategy in STRATEGY_ACCOUNTS[:4]:
        curve = curves[(strategy, "primary")]
        axis.plot(
            [point.mark_time for point in curve],
            [point.drawdown for point in curve],
            label=strategy.replace("_", " ").title(),
            color=colours[strategy],
            linewidth=1.1,
        )
    axis.set(
        title=f"{label} primary-net drawdowns",
        xlabel="UTC date",
        ylabel="Drawdown from running peak",
    )
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=0.2)
    drawdown_figure = save(f"{analysis_period}_drawdowns.png", figure)
    return equity_figure, drawdown_figure


def build_backtest_bundle(
    events: Sequence[EventRecord],
    hours: Sequence[BacktestHour],
    config: ConfigNode,
    *,
    analysis_period: str = DEVELOPMENT_PERIOD,
    label: str = "Development",
) -> BacktestBundle:
    if analysis_period not in {DEVELOPMENT_PERIOD, HELD_OUT_PERIOD}:
        raise BacktestError("backtest bundle requires development or held_out")
    split = getattr(config.splits, analysis_period)
    audits, plans_by_strategy = build_trade_plans(
        events, hours, config, analysis_period=analysis_period
    )
    curves: dict[tuple[str, str], tuple[EquityPoint, ...]] = {}
    results: dict[tuple[str, str], tuple[TradeResult, ...]] = {}
    daily_by_account: dict[tuple[str, str], tuple[dict[str, object], ...]] = {}
    trade_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for strategy in STRATEGY_ACCOUNTS:
        plans = plans_by_strategy[strategy]
        for scenario in ("gross", *COST_SCENARIOS):
            fee, slippage = _scenario_cost(config, scenario)
            trade_results = simulate_trades(
                plans, scenario, fee, slippage, config.strategy.initial_capital
            )
            curve = build_equity_curve(
                trade_results,
                hours,
                split.start_inclusive,
                split.end_exclusive,
                config.strategy.initial_capital,
            )
            daily = daily_equity(curve)
            results[(strategy, scenario)] = trade_results
            curves[(strategy, scenario)] = curve
            daily_by_account[(strategy, scenario)] = daily
            if scenario != "gross":
                trade_rows.extend(_trade_result_row(result) for result in trade_results)
                metric_rows.append(
                    performance_record(
                        strategy,
                        scenario,
                        plans,
                        results[(strategy, "gross")],
                        trade_results,
                        curve,
                        daily,
                        config,
                        analysis_period=analysis_period,
                    )
                )
            for record in daily:
                daily_rows.append(
                    {
                        "strategy_name": strategy,
                        "cost_scenario": scenario,
                        **record,
                    }
                )
    equity_rows: list[dict[str, object]] = []
    reference = curves[(STRATEGY_ACCOUNTS[0], "primary")]
    for index, point in enumerate(reference):
        row: dict[str, object] = {"mark_time": point.mark_time}
        for strategy in STRATEGY_ACCOUNTS:
            gross_point = curves[(strategy, "gross")][index]
            primary_point = curves[(strategy, "primary")][index]
            if gross_point.mark_time != point.mark_time or primary_point.mark_time != point.mark_time:
                raise BacktestError("strategy equity calendars do not align")
            row.update(
                {
                    f"{strategy}_gross_equity": gross_point.equity,
                    f"{strategy}_primary_net_equity": primary_point.equity,
                    f"{strategy}_primary_drawdown": primary_point.drawdown,
                    f"{strategy}_position_open": primary_point.position_open,
                    f"{strategy}_mark_source": primary_point.source,
                }
            )
        equity_rows.append(row)
    negative_count = sum(event.sign == "negative" for event in events)
    counts = {name: len(plans) for name, plans in plans_by_strategy.items()}
    return BacktestBundle(
        audit_rows=audits,
        trade_rows=tuple(trade_rows),
        metric_rows=tuple(metric_rows),
        equity_rows=tuple(equity_rows),
        daily_rows=tuple(daily_rows),
        benchmark_rows=benchmark_rows(
            hours, config, analysis_period=analysis_period
        ),
        figures=render_figures(
            curves, analysis_period=analysis_period, label=label
        ),
        negative_event_count=negative_count,
        executed_trade_counts=counts,
    )


def verify_backtest_inputs(
    config: ConfigNode,
    inference_manifest_relative: Path,
    *,
    require_frozen_acceptance: bool,
) -> tuple[Path, Path, int, dict[str, str]]:
    if (
        require_frozen_acceptance
        and inference_manifest_relative != ACCEPTED_INFERENCE_MANIFEST
    ):
        raise BacktestError("default Stage 8 execution must use accepted Stage 7 evidence")
    verified_stage7 = verify_inference_inventory(inference_manifest_relative)
    inference_digest = sha256_file(paths.resolve(inference_manifest_relative))
    if require_frozen_acceptance and inference_digest != ACCEPTED_INFERENCE_MANIFEST_SHA256:
        raise BacktestError("accepted Stage 7 manifest identity does not match the freeze")
    manifest = read_manifest(inference_manifest_relative)
    if manifest["code_identity"]["git_dirty"] is not False:
        raise BacktestError("Stage 7 evidence was produced from a dirty repository")
    if manifest["exclusions"].get("held_out_rows_analysed") != 0:
        raise BacktestError("Stage 7 evidence does not preserve held-out isolation")
    if not matches_active_protocol(manifest, config):
        raise BacktestError("Stage 7 evidence does not match active specification/config")
    artifacts, feature_relative, feature_rows, upstream = verify_inference_inputs(
        config,
        ACCEPTED_OUTCOME_MANIFEST,
        ACCEPTED_FEATURE_MANIFEST,
        require_frozen_acceptance=require_frozen_acceptance,
    )
    checksums = dict(upstream)
    stage7_sidecar = Path(str(inference_manifest_relative) + ".sha256")
    inventory = inference_manifest_relative.parent / "development_inference_inventory.csv"
    for relative, digest in (
        (inference_manifest_relative, inference_digest),
        (stage7_sidecar, sha256_file(paths.resolve(stage7_sidecar))),
        (inventory, sha256_file(paths.resolve(inventory))),
    ):
        checksums[relative.as_posix()] = digest
    checksums.update(verified_stage7)
    return (
        artifacts["development_event_outcomes.csv"],
        feature_relative,
        feature_rows,
        dict(sorted(checksums.items())),
    )


def _render_inventory(
    artifacts: Sequence[tuple[Path, str, int, str, Sequence[str]]]
) -> str:
    return render_records(
        [
            {
                "artifact_path": path.as_posix(),
                "sha256": digest,
                "record_count": count,
                "artifact_type": artifact_type,
                "column_schema_json": json.dumps(list(schema), separators=(",", ":")),
            }
            for path, digest, count, artifact_type, schema in artifacts
        ],
        INVENTORY_COLUMNS,
    )


def _read_rows(relative: Path, schema: Sequence[str]) -> list[dict[str, str]]:
    try:
        with paths.require_file(paths.resolve(relative)).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != tuple(schema):
                raise BacktestError(f"{relative.name} schema is invalid")
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise BacktestError(f"cannot read {relative.name}: {error}") from None


def verify_backtest_inventory(manifest_relative: Path) -> dict[str, str]:
    verify_manifest_checksum(manifest_relative)
    manifest = read_manifest(manifest_relative)
    if manifest["stage"] != STAGE or manifest["mode"] != MODE:
        raise BacktestError("manifest is not Stage 8 development-backtest evidence")
    if manifest["column_schema"] != list(INVENTORY_COLUMNS):
        raise BacktestError("Stage 8 manifest does not declare the inventory schema")
    inventory = manifest_relative.parent / "development_backtest_inventory.csv"
    if sha256_file(paths.resolve(inventory)) != manifest["output_checksum"]:
        raise BacktestError("Stage 8 inventory checksum does not match its manifest")
    rows = _read_rows(inventory, INVENTORY_COLUMNS)
    if len(rows) != manifest["row_count"]:
        raise BacktestError("Stage 8 inventory row count disagrees with its manifest")
    expected = set(TABLE_SCHEMAS) | FIGURE_NAMES
    if {Path(row["artifact_path"]).name for row in rows} != expected or len(rows) != len(expected):
        raise BacktestError("Stage 8 inventory does not list the required artifacts")
    eligibility = manifest["eligibility_changes"]
    expected_counts = {
        "development_trade_audit.csv": eligibility.get("trade_audit_records"),
        "development_trade_results.csv": eligibility.get("scenario_trade_records"),
        "development_strategy_metrics.csv": eligibility.get("metric_records"),
        "development_equity_hourly.csv": eligibility.get("hourly_equity_records"),
        "development_equity_daily.csv": eligibility.get("daily_equity_records"),
        "development_benchmarks.csv": eligibility.get("benchmark_records"),
    }
    if (
        eligibility.get("output_artifacts") != len(expected)
        or eligibility.get("generated_figures") != len(FIGURE_NAMES)
        or any(value is None for value in expected_counts.values())
    ):
        raise BacktestError("Stage 8 manifest eligibility counts are incomplete")
    verified: dict[str, str] = {}
    for row in rows:
        relative = Path(row["artifact_path"])
        if relative.parent != manifest_relative.parent:
            raise BacktestError("Stage 8 artifact is outside its run directory")
        digest = row["sha256"]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise BacktestError("Stage 8 inventory contains an invalid SHA-256")
        artifact = paths.require_file(paths.resolve(relative))
        if sha256_file(artifact) != digest:
            raise BacktestError(f"Stage 8 artifact checksum failed for {relative.name}")
        try:
            count = int(row["record_count"])
            schema = json.loads(row["column_schema_json"])
        except (ValueError, TypeError, json.JSONDecodeError):
            raise BacktestError("Stage 8 inventory metadata is malformed") from None
        if relative.name in TABLE_SCHEMAS:
            if row["artifact_type"] != "csv" or schema != list(TABLE_SCHEMAS[relative.name]):
                raise BacktestError(f"Stage 8 table metadata failed for {relative.name}")
            if len(_read_rows(relative, TABLE_SCHEMAS[relative.name])) != count:
                raise BacktestError(f"Stage 8 table row count failed for {relative.name}")
            if count != expected_counts[relative.name]:
                raise BacktestError(
                    f"Stage 8 manifest count failed for {relative.name}"
                )
        elif (
            row["artifact_type"] != "png"
            or schema != ["binary_png"]
            or count != 1
            or not artifact.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        ):
            raise BacktestError(f"Stage 8 figure metadata failed for {relative.name}")
        verified[relative.as_posix()] = digest
    return dict(sorted(verified.items()))


def _record_failure(log: RunLog, error: Exception) -> None:
    try:
        log.record(
            "development_backtest_failed",
            reason=type(error).__name__,
            detail=str(error)[:500],
            held_out_outcomes_accessed=False,
        )
    except Exception:  # pragma: no cover
        pass


def run_development_backtest(
    config: ConfigNode | None = None,
    *,
    inference_manifest_relative: Path | None = None,
) -> DevelopmentBacktestResult:
    cfg = config if config is not None else load_config()
    inference_manifest = inference_manifest_relative or ACCEPTED_INFERENCE_MANIFEST
    frozen = inference_manifest == ACCEPTED_INFERENCE_MANIFEST
    draft = build_manifest(STAGE, MODE, cfg)
    if frozen and draft.code_identity["git_dirty"] is not False:
        raise BacktestError(
            "development backtest requires a clean committed repository; "
            "offline tests must use an explicit synthetic manifest"
        )
    log = create_run_log(draft.run_id, config=cfg)
    log.record(
        "development_backtest_started",
        inference_manifest=inference_manifest.as_posix(),
        analysis_period=DEVELOPMENT_PERIOD,
        held_out_outcomes_accessed=False,
    )
    try:
        event_relative, feature_relative, feature_rows, input_checksums = (
            verify_backtest_inputs(
                cfg, inference_manifest, require_frozen_acceptance=frozen
            )
        )
        events = read_event_records(event_relative, cfg)
        hours = read_development_hours(feature_relative, feature_rows, cfg)
        bundle = build_backtest_bundle(events, hours, cfg)
        run_dir = run_relative_dir(draft.run_id, cfg)
        table_payloads = (
            ("development_trade_audit.csv", render_records(bundle.audit_rows, TRADE_AUDIT_COLUMNS), len(bundle.audit_rows)),
            ("development_trade_results.csv", render_records(bundle.trade_rows, TRADE_RESULT_COLUMNS), len(bundle.trade_rows)),
            ("development_strategy_metrics.csv", render_records(bundle.metric_rows, METRIC_COLUMNS), len(bundle.metric_rows)),
            ("development_equity_hourly.csv", render_records(bundle.equity_rows, EQUITY_COLUMNS), len(bundle.equity_rows)),
            ("development_equity_daily.csv", render_records(bundle.daily_rows, DAILY_COLUMNS), len(bundle.daily_rows)),
            ("development_benchmarks.csv", render_records(bundle.benchmark_rows, BENCHMARK_COLUMNS), len(bundle.benchmark_rows)),
        )
        inventory_rows: list[tuple[Path, str, int, str, Sequence[str]]] = []
        for name, content, count in table_payloads:
            relative = run_dir / name
            create_exclusive(relative, content)
            digest = sha256_bytes(content.encode("utf-8"))
            inventory_rows.append((relative, digest, count, "csv", TABLE_SCHEMAS[name]))
            log.record("development_backtest_table_written", path=relative.as_posix(), rows=count, sha256=digest)
        for name, payload in bundle.figures:
            relative = run_dir / name
            create_exclusive_bytes(relative, payload)
            digest = sha256_bytes(payload)
            inventory_rows.append((relative, digest, 1, "png", ("binary_png",)))
            log.record("development_backtest_figure_written", path=relative.as_posix(), sha256=digest)
        inventory_relative = run_dir / "development_backtest_inventory.csv"
        inventory_content = _render_inventory(inventory_rows)
        create_exclusive(inventory_relative, inventory_content)
        inventory_digest = sha256_bytes(inventory_content.encode("utf-8"))
        finished = replace(
            draft,
            input_checksums=input_checksums,
            exclusions={"held_out_rows_analysed": 0},
            eligibility_changes={
                "accepted_negative_events": bundle.negative_event_count,
                **{f"executed_{name}_trades": count for name, count in bundle.executed_trade_counts.items()},
                "trade_audit_records": len(bundle.audit_rows),
                "scenario_trade_records": len(bundle.trade_rows),
                "metric_records": len(bundle.metric_rows),
                "hourly_equity_records": len(bundle.equity_rows),
                "daily_equity_records": len(bundle.daily_rows),
                "benchmark_records": len(bundle.benchmark_rows),
                "generated_figures": len(bundle.figures),
                "output_artifacts": len(inventory_rows),
            },
        ).completed(inventory_digest, len(inventory_rows), list(INVENTORY_COLUMNS))
        manifest_path = finished.write_for_run("development-backtest", cfg)
        manifest_relative = manifest_path.relative_to(paths.project_root().resolve())
        sidecar_relative, manifest_digest = write_manifest_checksum(manifest_relative)
        verify_backtest_inventory(manifest_relative)
        log.record("manifest_checksum_written", path=sidecar_relative.as_posix(), sha256=manifest_digest, manifest=manifest_relative.as_posix())
        log.record(
            "development_backtest_completed",
            manifest=manifest_relative.as_posix(),
            trade_counts=dict(bundle.executed_trade_counts),
            held_out_outcomes_accessed=False,
        )
    except Exception as error:
        _record_failure(log, error)
        raise
    return DevelopmentBacktestResult(
        run_id=draft.run_id,
        manifest_relative_path=manifest_relative,
        manifest_checksum_relative_path=sidecar_relative,
        log_relative_path=log.relative_path,
        inventory_relative_path=inventory_relative,
        negative_events=bundle.negative_event_count,
        trade_counts=dict(bundle.executed_trade_counts),
    )


def summarise(result: DevelopmentBacktestResult) -> list[str]:
    return [
        f"run id                  {result.run_id}",
        f"accepted negative events {result.negative_events}",
        *(f"{name + ' trades':24}{count}" for name, count in result.trade_counts.items()),
        f"inventory               {result.inventory_relative_path.as_posix()}",
        f"manifest                {result.manifest_relative_path.as_posix()}",
        f"manifest checksum       {result.manifest_checksum_relative_path.as_posix()}",
        f"run log                 {result.log_relative_path.as_posix()}",
        "held-out outcomes       not accessed",
    ]
