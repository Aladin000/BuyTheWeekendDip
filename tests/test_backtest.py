"""Offline checks for the development backtest."""

from __future__ import annotations

import csv
import io
import math
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src import backtest, pipeline
from src.artifacts import build_manifest, read_manifest
from src.backtest import (
    BacktestBundle,
    BacktestError,
    BacktestHour,
    DevelopmentBacktestResult,
    TradePlan,
    benchmark_rows,
    build_equity_curve,
    build_trade_plans,
    daily_equity,
    performance_record,
    read_analysis_hours,
    read_development_hours,
    render_figures,
    run_development_backtest,
    simulate_trades,
    verify_backtest_inventory,
)
from src.features import FEATURE_COLUMNS
from src.inference import EventRecord

UTC = timezone.utc


def hour(index: int, *, start: datetime | None = None, segment: int = 1) -> BacktestHour:
    origin = start or datetime(2020, 1, 1, tzinfo=UTC)
    open_time = origin + timedelta(hours=index)
    return BacktestHour(
        open_time=open_time,
        close_time=open_time + timedelta(hours=1) - timedelta(milliseconds=1),
        close_time_raw=int((open_time + timedelta(hours=1) - timedelta(milliseconds=1)).timestamp() * 1000),
        segment_id=segment,
        open_price=100 + index,
        close_price=100.5 + index,
    )


def event(index: int, *, weekend: bool = False, low: bool | None = True) -> EventRecord:
    source = hour(index)
    z = -3.0
    return EventRecord(
        event_id=f"negative-{source.close_time_raw}",
        sign="negative",
        signal_open_time=source.open_time,
        signal_close_time=source.close_time,
        signal_close=source.close_price,
        segment_id=source.segment_id,
        calendar_year=2020,
        calendar_month="2020-01",
        weekend=weekend,
        crosses_into_monday_12h=False if weekend else None,
        time_block="00-05",
        shock_return=-0.03,
        shock_zscore=z,
        shock_magnitude=-z,
        absolute_shock_z=abs(z),
        activity_percentile=None if low is None else (0.2 if low else 0.7),
        low_activity_score=None if low is None else (0.8 if low else 0.3),
        low_activity=low,
        rv24=None if low is None else 0.02,
        full_feature_eligible=low is not None,
        complete_12h=True,
        complete_24h=True,
        outcome_12h=0.01,
        post_positive_12h=True,
        eligible_h1_h2=True,
        eligible_h3_h4=low is not None,
    )


def small_config(days: int = 2):
    start = datetime(2020, 1, 1, tzinfo=UTC)
    return SimpleNamespace(
        splits=SimpleNamespace(
            development=SimpleNamespace(
                start_inclusive=start, end_exclusive=start + timedelta(days=days)
            )
        ),
        strategy=SimpleNamespace(
            initial_capital=1.0,
            variants=("all_shock", "weekend", "low_activity", "combined"),
            entry_offset_bars=1,
            exit_offset_bars=12,
            hold_hours=12,
            sparse_trade_label_threshold=20,
        ),
        performance=SimpleNamespace(
            sharpe_periods_per_year=365,
            risk_free_rate=0.0,
        ),
        reproducibility=SimpleNamespace(atol=1e-12),
        costs=SimpleNamespace(
            scenarios=SimpleNamespace(
                low=SimpleNamespace(fee_per_side=0.0005, slippage_per_side=0.0001),
                primary=SimpleNamespace(fee_per_side=0.001, slippage_per_side=0.0002),
                high=SimpleNamespace(fee_per_side=0.001, slippage_per_side=0.001),
            )
        ),
    )


def plan(*, entry: int = 1, exit_price: float = 112.5) -> TradePlan:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    return TradePlan(
        strategy_name="all_shock",
        event_id="negative-1",
        signal_time=start + timedelta(hours=entry) - timedelta(milliseconds=1),
        entry_time=start + timedelta(hours=entry),
        exit_time=start + timedelta(hours=entry + 12) - timedelta(milliseconds=1),
        exit_boundary=start + timedelta(hours=entry + 12),
        entry_price=101.0,
        exit_price=exit_price,
        weekend=False,
        activity_percentile=0.2,
        low_activity=True,
    )


def test_primary_execution_indices_filters_and_audit_reconcile(config) -> None:
    hours = tuple(hour(index) for index in range(60))
    events = (event(0, weekend=True, low=True), event(24, weekend=False, low=None))
    audit, plans = build_trade_plans(events, hours, config)
    assert len(audit) == 10
    assert {name: len(rows) for name, rows in plans.items()} == {
        "all_shock": 2,
        "weekend": 1,
        "low_activity": 1,
        "combined": 1,
        "weekday_shock": 1,
    }
    first = plans["all_shock"][0]
    assert first.entry_time == hours[1].open_time
    assert first.exit_time == hours[12].close_time
    assert first.exit_boundary == hours[12].open_time + timedelta(hours=1)
    unavailable = next(
        row for row in audit if row["strategy_name"] == "low_activity" and row["event_id"] == events[1].event_id
    )
    assert unavailable["rejection_reason"] == "activity_unavailable"


def test_execution_rejects_gap_and_never_substitutes_later_bar(config) -> None:
    hours = [hour(index) for index in range(20)]
    hours[1] = replace(hours[1], open_time=hours[1].open_time + timedelta(hours=1))
    audit, plans = build_trade_plans((event(0),), tuple(hours), config)
    assert plans["all_shock"] == ()
    row = next(record for record in audit if record["strategy_name"] == "all_shock")
    assert row["rejection_reason"] == "missing_or_noncontiguous_execution_bar"


def test_event_must_reconcile_with_frozen_source_bar(config) -> None:
    hours = tuple(hour(index) for index in range(20))
    altered = replace(event(0), signal_close=999.0)
    with pytest.raises(BacktestError, match="does not reconcile"):
        build_trade_plans((altered,), hours, config)


def test_position_rule_rejects_overlap(config) -> None:
    hours = tuple(hour(index) for index in range(40))
    audit, plans = build_trade_plans((event(0), event(2)), hours, config)
    assert len(plans["all_shock"]) == 1
    rejected = [row for row in audit if row["strategy_name"] == "all_shock" and not row["strategy_eligible"]]
    assert rejected[0]["rejection_reason"] == "position_already_open"


def test_exact_multiplicative_cost_formula_and_monotonicity() -> None:
    trade = plan()
    gross = simulate_trades((trade,), "gross", 0.0, 0.0, 1.0)[0]
    low = simulate_trades((trade,), "low", 0.0005, 0.0001, 1.0)[0]
    primary = simulate_trades((trade,), "primary", 0.001, 0.0002, 1.0)[0]
    high = simulate_trades((trade,), "high", 0.001, 0.001, 1.0)[0]
    expected = (1 - 0.001) ** 2 * trade.exit_price * (1 - 0.0002) / (
        trade.entry_price * (1 + 0.0002)
    )
    assert primary.capital_after == pytest.approx(expected)
    assert primary.units == pytest.approx((1 - 0.001) / (trade.entry_price * 1.0002))
    assert gross.capital_after > low.capital_after > primary.capital_after > high.capital_after


def test_multiple_trades_compound_current_capital_without_leverage() -> None:
    first = plan()
    second = replace(
        plan(entry=25, exit_price=130.0),
        event_id="negative-2",
        entry_price=120.0,
    )
    results = simulate_trades((first, second), "primary", 0.001, 0.0002, 1.0)
    assert results[1].capital_before == pytest.approx(results[0].capital_after)
    assert results[1].capital_after == pytest.approx(
        math.prod(1 + result.net_return for result in results)
    )
    assert all(result.units * result.entry_fill < result.capital_before for result in results)


def test_hourly_equity_marks_entry_cost_intermediate_close_and_exit() -> None:
    hours = tuple(hour(index) for index in range(48))
    result = simulate_trades((plan(),), "primary", 0.001, 0.0002, 1.0)
    curve = build_equity_curve(
        result,
        hours,
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 3, tzinfo=UTC),
        1.0,
    )
    assert len(curve) == 49
    assert curve[0].source == "initial_boundary" and curve[0].equity == 1.0
    assert curve[1].source == "market_open"
    assert curve[1].equity == pytest.approx(result[0].units * result[0].plan.entry_price)
    assert curve[2].source == "market_close"
    assert curve[13].source == "exit_cash"
    assert curve[13].equity == pytest.approx(result[0].capital_after)
    assert all(point.source != "cash_calendar_mark" or not point.position_open for point in curve)


def test_invested_missing_hour_fails_instead_of_carrying_a_price() -> None:
    hours = tuple(hour(index) for index in range(48) if index != 1)
    result = simulate_trades((plan(),), "primary", 0.001, 0.0002, 1.0)
    with pytest.raises(BacktestError, match="lacks an observed price"):
        build_equity_curve(result, hours, hour(0).open_time, hour(0).open_time + timedelta(days=2), 1.0)


def test_daily_sampling_keeps_first_day_return_and_cash_days() -> None:
    hours = tuple(hour(index) for index in range(48))
    result = simulate_trades((plan(),), "primary", 0.001, 0.0002, 1.0)
    curve = build_equity_curve(result, hours, hour(0).open_time, hour(0).open_time + timedelta(days=2), 1.0)
    daily = daily_equity(curve)
    assert len(daily) == 2
    assert daily[0]["daily_return"] == pytest.approx(daily[0]["equity"] - 1)
    assert daily[1]["daily_return"] == pytest.approx(0.0)


def test_metrics_break_even_turnover_exposure_and_zero_trade_semantics() -> None:
    cfg = small_config()
    hours = tuple(hour(index) for index in range(48))
    plans = (plan(),)
    gross = simulate_trades(plans, "gross", 0.0, 0.0, 1.0)
    net = simulate_trades(plans, "primary", 0.001, 0.0002, 1.0)
    curve = build_equity_curve(net, hours, cfg.splits.development.start_inclusive, cfg.splits.development.end_exclusive, 1.0)
    daily = daily_equity(curve)
    row = performance_record("all_shock", "primary", plans, gross, net, curve, daily, cfg)
    assert row["trade_count"] == 1
    assert row["net_cumulative_return"] == pytest.approx(net[-1].capital_after - 1)
    assert row["time_invested_fraction"] == pytest.approx(12 / 48)
    expected_break_even = 1 - (plans[0].exit_price / plans[0].entry_price) ** (-0.5)
    assert row["break_even_cost_per_side"] == pytest.approx(expected_break_even)
    assert row["maximum_drawdown"] <= 0
    expected_days = 2.0
    assert row["annualised_return"] == pytest.approx(
        net[-1].capital_after ** (365 / expected_days) - 1
    )
    daily_returns = np.array([record["daily_return"] for record in daily])
    expected_sd = daily_returns.std(ddof=1)
    assert row["annualised_volatility"] == pytest.approx(expected_sd * math.sqrt(365))
    assert row["sharpe_ratio"] == pytest.approx(
        math.sqrt(365) * daily_returns.mean() / expected_sd
    )
    expected_turnover = sum(result.notional_turnover for result in net) / np.mean(
        [record["equity"] for record in daily]
    )
    assert row["turnover"] == pytest.approx(expected_turnover)
    assert row["largest_trade_share_absolute_pnl"] == pytest.approx(1.0)
    assert row["top5_trade_share_absolute_pnl"] == pytest.approx(1.0)
    empty_curve = build_equity_curve((), hours, cfg.splits.development.start_inclusive, cfg.splits.development.end_exclusive, 1.0)
    empty_daily = daily_equity(empty_curve)
    empty = performance_record("combined", "primary", (), (), (), empty_curve, empty_daily, cfg)
    assert empty["net_cumulative_return"] == empty["annualised_return"] == 0
    assert empty["maximum_drawdown"] == empty["turnover"] == empty["time_invested_fraction"] == 0
    assert empty["sharpe_ratio"] is None and empty["break_even_cost_per_side"] is None


def test_break_even_has_no_positive_solution_when_gross_loses() -> None:
    cfg = small_config()
    hours = tuple(hour(index) for index in range(48))
    losing = (plan(exit_price=90.0),)
    gross = simulate_trades(losing, "gross", 0, 0, 1)
    net = simulate_trades(losing, "primary", 0.001, 0.0002, 1)
    curve = build_equity_curve(net, hours, cfg.splits.development.start_inclusive, cfg.splits.development.end_exclusive, 1)
    row = performance_record("all_shock", "primary", losing, gross, net, curve, daily_equity(curve), cfg)
    assert row["break_even_cost_per_side"] is None
    assert row["break_even_status"] == "no_positive_break_even_cost"


def test_buy_and_hold_uses_first_open_final_close_and_same_cost_formula() -> None:
    cfg = small_config()
    hours = tuple(hour(index) for index in range(48))
    rows = benchmark_rows(hours, cfg)
    assert rows[0]["benchmark_name"] == "cash" and rows[0]["net_cumulative_return"] == 0
    primary = next(row for row in rows if row["cost_scenario"] == "primary")
    expected = (1 - 0.001) ** 2 * hours[-1].close_price * (1 - 0.0002) / (
        hours[0].open_price * (1 + 0.0002)
    ) - 1
    assert primary["entry_time"] == hours[0].open_time
    assert primary["exit_time"] == hours[-1].close_time
    assert primary["net_cumulative_return"] == pytest.approx(expected)


def _feature_csv(rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(FEATURE_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        complete = {name: "" for name in FEATURE_COLUMNS}
        complete.update(row)
        writer.writerow(complete)
    return buffer.getvalue()


def test_reader_parses_development_only_and_ignores_later_price_cells(
    config, temp_acquisition_root
) -> None:
    relative = Path("features.csv")
    start = config.splits.development.start_inclusive
    close = start + timedelta(hours=1) - timedelta(milliseconds=1)
    text = _feature_csv(
        [
            {
                "open_time": start.isoformat().replace("+00:00", "Z"),
                "close_time": close.isoformat().replace("+00:00", "Z"),
                "close_time_raw": str(int(close.timestamp() * 1000)),
                "contiguous_segment_id": "1",
                "open": "100",
                "close": "101",
                "analysis_period": "development",
            },
            {
                "open_time": "2024-01-01T00:00:00Z",
                "close_time": "2024-01-01T00:59:59.999Z",
                "close_time_raw": "not-parsed",
                "contiguous_segment_id": "2",
                "open": "not-parsed",
                "close": "not-parsed",
                "analysis_period": "held_out",
            },
        ]
    )
    (temp_acquisition_root / relative).write_text(text, encoding="utf-8")
    parsed = read_development_hours(relative, 2, config)
    assert len(parsed) == 1 and parsed[0].open_price == 100


def test_reader_can_target_held_out_without_parsing_development_prices(
    config, temp_acquisition_root
) -> None:
    relative = Path("held_out_features.csv")
    held_out_start = config.splits.held_out.start_inclusive
    held_out_close = held_out_start + timedelta(hours=1) - timedelta(milliseconds=1)
    text = _feature_csv(
        [
            {
                "open_time": "2023-12-31T23:00:00Z",
                "close_time": "2023-12-31T23:59:59.999Z",
                "close_time_raw": "not-parsed",
                "contiguous_segment_id": "1",
                "open": "not-parsed",
                "close": "not-parsed",
                "analysis_period": "development",
            },
            {
                "open_time": held_out_start.isoformat().replace("+00:00", "Z"),
                "close_time": held_out_close.isoformat().replace("+00:00", "Z"),
                "close_time_raw": str(int(held_out_close.timestamp() * 1000)),
                "contiguous_segment_id": "1",
                "open": "100",
                "close": "101",
                "analysis_period": "held_out",
            },
        ]
    )
    (temp_acquisition_root / relative).write_text(text, encoding="utf-8")
    parsed = read_analysis_hours(
        relative, 2, config, analysis_period="held_out"
    )
    assert len(parsed) == 1 and parsed[0].open_price == 100


def test_held_out_trade_plan_uses_held_out_boundary(config) -> None:
    start = config.splits.held_out.start_inclusive
    hours = tuple(hour(index, start=start) for index in range(20))
    source = hours[0]
    held_out_event = replace(
        event(0),
        event_id=f"negative-{source.close_time_raw}",
        signal_open_time=source.open_time,
        signal_close_time=source.close_time,
        signal_close=source.close_price,
        segment_id=source.segment_id,
    )
    audits, plans = build_trade_plans(
        (held_out_event,), hours, config, analysis_period="held_out"
    )
    assert len(plans["all_shock"]) == 1
    assert next(
        row for row in audits
        if row["strategy_name"] == "all_shock"
    )["strategy_eligible"] is True


def test_figures_are_deterministic_pngs() -> None:
    hours = tuple(hour(index) for index in range(48))
    curves = {}
    for strategy in backtest.STRATEGY_ACCOUNTS:
        for scenario in ("gross", "primary"):
            results = simulate_trades((replace(plan(), strategy_name=strategy),), scenario, 0 if scenario == "gross" else 0.001, 0 if scenario == "gross" else 0.0002, 1)
            curves[(strategy, scenario)] = build_equity_curve(results, hours, hour(0).open_time, hour(0).open_time + timedelta(days=2), 1)
    first = render_figures(curves)
    second = render_figures(curves)
    assert first == second
    assert {name for name, _ in first} == backtest.FIGURE_NAMES
    assert all(payload.startswith(b"\x89PNG\r\n\x1a\n") for _, payload in first)


def test_import_has_no_project_data_or_network_io(project_root) -> None:
    script = r'''
import sys
opened = []
def audit(event, args):
    if event == "open":
        path = str(args[0])
        if "BuyTheWeekendDip" in path and "outputs/runs" in path:
            opened.append(path)
sys.addaudithook(audit)
import src.backtest
assert opened == [], opened
'''
    completed = subprocess.run([sys.executable, "-c", script], cwd=project_root, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_backtest_pipeline_dispatch_is_thin(monkeypatch, capsys) -> None:
    result = DevelopmentBacktestResult(
        "run",
        Path("outputs/runs/run/development-backtest.manifest.json"),
        Path("outputs/runs/run/development-backtest.manifest.json.sha256"),
        Path("outputs/runs/run/run.log.jsonl"),
        Path("outputs/runs/run/development_backtest_inventory.csv"),
        10,
        {name: 1 for name in backtest.STRATEGY_ACCOUNTS},
    )
    monkeypatch.setattr(pipeline, "load_config", lambda: object())
    monkeypatch.setattr(backtest, "run_development_backtest", lambda config: result)
    assert pipeline.dispatch("backtest") == 0
    output = capsys.readouterr().out
    assert "held-out outcomes       not accessed" in output


def test_stage8_run_writes_and_reverifies_immutable_evidence(
    config, temp_acquisition_root, monkeypatch
) -> None:
    dummy = BacktestBundle(
        audit_rows=(),
        trade_rows=(),
        metric_rows=(),
        equity_rows=(),
        daily_rows=(),
        benchmark_rows=(),
        figures=tuple((name, b"\x89PNG\r\n\x1a\nsynthetic") for name in sorted(backtest.FIGURE_NAMES)),
        negative_event_count=12,
        executed_trade_counts={name: 0 for name in backtest.STRATEGY_ACCOUNTS},
    )
    monkeypatch.setattr(backtest, "verify_backtest_inputs", lambda *args, **kwargs: (Path("events.csv"), Path("features.csv"), 1, {}))
    monkeypatch.setattr(backtest, "read_event_records", lambda *args: ())
    monkeypatch.setattr(backtest, "read_development_hours", lambda *args: ())
    monkeypatch.setattr(backtest, "build_backtest_bundle", lambda *args: dummy)
    result = run_development_backtest(config, inference_manifest_relative=Path("synthetic.manifest.json"))
    manifest = read_manifest(result.manifest_relative_path)
    assert manifest["stage"] == backtest.STAGE
    assert manifest["exclusions"] == {"held_out_rows_analysed": 0}
    assert len(verify_backtest_inventory(result.manifest_relative_path)) == 8
    log = (temp_acquisition_root / result.log_relative_path).read_text(encoding="utf-8")
    assert '"held_out_outcomes_accessed": false' in log
    table = temp_acquisition_root / result.manifest_relative_path.parent / "development_strategy_metrics.csv"
    table.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(BacktestError, match="checksum failed"):
        verify_backtest_inventory(result.manifest_relative_path)


def test_default_stage8_refuses_dirty_tree_before_opening_log(config, monkeypatch) -> None:
    draft = build_manifest(backtest.STAGE, backtest.MODE, config)
    dirty = replace(draft, code_identity={**draft.code_identity, "git_dirty": True})
    monkeypatch.setattr(backtest, "build_manifest", lambda *args, **kwargs: dirty)
    monkeypatch.setattr(backtest, "create_run_log", lambda *args, **kwargs: pytest.fail("dirty execution opened a run log"))
    with pytest.raises(BacktestError, match="clean committed repository"):
        run_development_backtest(config)
