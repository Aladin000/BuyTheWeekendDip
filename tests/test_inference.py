"""Offline tests for the development inference stage."""

from __future__ import annotations

import csv
import io
import math
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from src import inference, pipeline
from src.artifacts import build_manifest, read_manifest
from src.events import EventError, construct_any_sign_events
from src.inference import (
    DevelopmentInferenceResult,
    EventRecord,
    HourCandidate,
    InferenceBundle,
    InferenceError,
    ModelData,
    PathRecord,
    attenuation_rows,
    build_any_sign_event_records,
    build_diagnostics,
    build_group_comparisons,
    build_inference_bundle,
    build_model_data,
    build_primary_models,
    count_cross_sign_overlaps,
    fit_hc3,
    holm_adjust,
    month_block_bootstrap,
    one_sided_greater_p,
    reconcile_stage6_records,
    render_figures,
    render_records,
    run_development_inference,
    verify_inference_inventory,
)
from src.outcomes import OutcomeInputRow
from src.outcomes import EVENT_OUTPUT_COLUMNS

UTC = timezone.utc


def event_record(
    index: int,
    *,
    sign: str = "negative",
    month: int = 1,
    weekend: bool | None = None,
    low_activity: bool | None = None,
    full_feature: bool = True,
    complete: bool = True,
    outcome: float | None = None,
) -> EventRecord:
    close_time = datetime(2020, month, 1, tzinfo=UTC) + timedelta(hours=index)
    block = inference.TIME_BLOCKS[index % 4]
    is_weekend = bool((index // 4) % 2) if weekend is None else weekend
    percentile = ((index * 17) % 97 + 1) / 100
    low = percentile < 1 / 3 if low_activity is None else low_activity
    score = 1 - percentile
    zscore = (-1 if sign == "negative" else 1) * (2.6 + (index % 7) * 0.13)
    value = (
        0.001
        + 0.002 * is_weekend
        + 0.0015 * score
        - 0.0002 * abs(zscore)
        + 0.0001 * ((index * index) % 11)
        if outcome is None
        else outcome
    )
    return EventRecord(
        event_id=f"{sign}-{int(close_time.timestamp() * 1000)}",
        sign=sign,
        signal_open_time=close_time - timedelta(hours=1) + timedelta(milliseconds=1),
        signal_close_time=close_time,
        signal_close=10000 + index,
        segment_id=1,
        calendar_year=2020,
        calendar_month=f"2020-{month:02d}",
        weekend=is_weekend,
        crosses_into_monday_12h=False if is_weekend else None,
        time_block=block,
        shock_return=zscore / 100,
        shock_zscore=zscore,
        shock_magnitude=-zscore if sign == "negative" else None,
        absolute_shock_z=abs(zscore),
        activity_percentile=percentile if full_feature else None,
        low_activity_score=score if full_feature else None,
        low_activity=low if full_feature else None,
        rv24=(0.01 + ((index * 5) % 13) / 1000) if full_feature else None,
        full_feature_eligible=full_feature,
        complete_12h=complete,
        complete_24h=complete,
        outcome_12h=value if complete else None,
        post_positive_12h=value > 0 if complete else None,
        eligible_h1_h2=sign == "negative" and complete,
        eligible_h3_h4=sign == "negative" and complete and full_feature,
    )


def rich_events() -> tuple[EventRecord, ...]:
    negative = [event_record(index, month=index % 4 + 1) for index in range(48)]
    positive = [
        event_record(index + 60, sign="positive", month=index % 4 + 1)
        for index in range(40)
    ]
    return tuple(sorted((*negative, *positive), key=lambda row: row.signal_close_time))


def outcome_input(
    moment: datetime,
    close: str | None,
    *,
    period: str,
    zscore: float | None,
) -> OutcomeInputRow:
    close_time = moment + timedelta(hours=1) - timedelta(milliseconds=1)
    development = period == "development"
    block_start = (close_time.hour // 6) * 6
    return OutcomeInputRow(
        open_time=moment,
        close_time=close_time,
        segment_id=1,
        analysis_period=period,
        close=Decimal(close) if development and close is not None else None,
        close_time_raw=int(close_time.timestamp() * 1000) if development else None,
        shock_eligible=development and zscore is not None,
        negative_event=False,
        positive_event=False,
        event_id="",
        calendar_year=close_time.year if development else None,
        calendar_month=close_time.strftime("%Y-%m") if development else "",
        weekend=(close_time.isoweekday() in (6, 7)) if development else None,
        crosses_into_monday_12h=None,
        utc_time_block=f"{block_start:02d}-{block_start + 5:02d}" if development else "",
        shock_log_return_1h=zscore / 100 if development and zscore is not None else None,
        shock_zscore=zscore if development else None,
        shock_magnitude=-zscore if development and zscore is not None and zscore < 0 else None,
        absolute_shock_z=abs(zscore) if development and zscore is not None else None,
        activity_percentile=0.2 if development else None,
        low_activity_score=0.8 if development else None,
        low_activity=True if development else None,
        rv24=0.02 if development else None,
        full_feature_eligible=development,
    )


def render_event_output_record(moment: datetime, *, low: str = "true") -> str:
    microsecond_source = moment >= datetime(2025, 1, 1, tzinfo=UTC)
    quantum = timedelta(microseconds=1) if microsecond_source else timedelta(milliseconds=1)
    close_time = moment + timedelta(hours=1) - quantum
    raw = int(close_time.timestamp() * (1_000_000 if microsecond_source else 1_000))
    record = {name: "" for name in EVENT_OUTPUT_COLUMNS}
    record.update(
        {
            "event_id": f"negative-{raw}",
            "event_sign": "negative",
            "signal_open_time": moment.isoformat().replace("+00:00", "Z"),
            "signal_close_time": close_time.isoformat().replace("+00:00", "Z"),
            "signal_close": "10000",
            "contiguous_segment_id": "1",
            "calendar_year": str(close_time.year),
            "calendar_month": close_time.strftime("%Y-%m"),
            "weekend": str(close_time.isoweekday() in (6, 7)).lower(),
            "crosses_into_monday_12h": "false",
            "utc_time_block": "00-05",
            "shock_log_return_1h": "-0.02",
            "shock_zscore": "-3.0",
            "shock_magnitude": "3.0",
            "absolute_shock_z": "3.0",
            "activity_percentile_90d": "0.2",
            "low_activity_score": "0.8",
            "low_activity": low,
            "rv24": "0.02",
            "full_feature_eligible_hour": "true",
            "complete_6h": "true",
            "post_log_return_6h": "0.005",
            "complete_12h": "true",
            "post_log_return_12h": "0.01",
            "post_positive_12h": "true",
            "complete_24h": "true",
            "post_log_return_24h": "0.02",
            "eligible_h1_h2": "true",
            "eligible_h3_h4": "true",
        }
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(EVENT_OUTPUT_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerow(record)
    return buffer.getvalue()


def test_any_sign_cooldown_is_shared_continuous_and_includes_equality() -> None:
    start = datetime(2019, 12, 31, tzinfo=UTC)
    times = [start + timedelta(hours=value) for value in (0, 12, 24, 30, 48)]
    decisions = construct_any_sign_events(
        [-3.0, 3.0, 3.1, -3.2, -3.3],
        times,
        list(range(5)),
        negative_threshold=-2.5,
        positive_threshold=2.5,
        cooldown_hours=24,
    )
    assert [decision.accepted for decision in decisions] == [True, False, True, False, True]
    assert decisions[1].rejection_reason == "any_sign_cooldown"
    assert decisions[2].event_id == "positive-2"


def test_event_reader_enforces_schema_and_derived_invariants(
    config, temp_acquisition_root
) -> None:
    relative = Path("events.csv")
    (temp_acquisition_root / relative).write_text(
        render_event_output_record(datetime(2020, 1, 4, tzinfo=UTC)), encoding="utf-8"
    )
    parsed = inference.read_event_records(relative, config)
    assert len(parsed) == 1
    assert parsed[0].low_activity is True
    (temp_acquisition_root / relative).write_text(
        render_event_output_record(datetime(2020, 1, 4, tzinfo=UTC), low="false"),
        encoding="utf-8",
    )
    with pytest.raises(InferenceError, match="low-activity flag"):
        inference.read_event_records(relative, config)


def test_event_reader_accepts_locked_microsecond_candles_from_2025(
    config, temp_acquisition_root
) -> None:
    relative = Path("events.csv")
    (temp_acquisition_root / relative).write_text(
        render_event_output_record(datetime(2025, 1, 4, tzinfo=UTC)), encoding="utf-8"
    )
    parsed = inference.read_event_records(relative, config)
    assert parsed[0].signal_close_time == datetime(
        2025, 1, 4, 0, 59, 59, 999999, tzinfo=UTC
    )


@pytest.mark.parametrize(
    "zscores,times,raw",
    [([0.0], [], [1]), ([math.nan], [datetime.now(UTC)], [1]), ([0.0], [datetime.now()], [1])],
)
def test_any_sign_cooldown_rejects_invalid_inputs(zscores, times, raw) -> None:
    with pytest.raises(EventError):
        construct_any_sign_events(
            zscores,
            times,
            raw,
            negative_threshold=-2.5,
            positive_threshold=2.5,
            cooldown_hours=24,
        )


def test_model_matrices_and_samples_match_locked_formulas() -> None:
    events = rich_events()
    models = build_primary_models(events)
    assert models["model0"].terms == ("intercept",)
    assert models["model1"].terms == ("intercept", "weekend")
    assert models["model2"].terms == (
        "intercept",
        "weekend",
        "low_activity_score",
        "shock_magnitude",
        "rv24",
        "time_block_06_11",
        "time_block_12_17",
        "time_block_18_23",
    )
    assert models["model3"].terms[3] == "weekend_x_low_activity_score"
    assert models["h5_sign"].terms[1] == "positive"
    assert len(models["model1"].rows) == 48
    assert len(models["model1_cc"].rows) == len(models["model2"].rows) == 48
    assert len(models["h5_sign"].rows) == 88
    assert np.all(models["model2"].x[:, 0] == 1)
    assert np.all(np.sum(models["model2"].x[:, 5:], axis=1) <= 1)


def test_hc3_and_finite_sample_t_match_independent_matrix_calculation() -> None:
    x = np.column_stack(
        [np.ones(12), np.arange(12, dtype=float), np.asarray([0, 1] * 6, dtype=float)]
    )
    y = np.asarray([0.2, 1.1, 1.8, 3.4, 4.2, 4.7, 6.4, 7.1, 7.7, 9.5, 10.1, 11.4])
    rows = tuple(event_record(index, outcome=float(y[index])) for index in range(12))
    data = ModelData("fixture", "fixture", ("intercept", "trend", "indicator"), rows, y, x)
    fit = fit_hc3(data)
    assert fit.status == "estimable"
    inverse = np.linalg.inv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.diag(x @ inverse @ x.T)
    meat = x.T @ np.diag((residual / (1 - leverage)) ** 2) @ x
    standard_errors = np.sqrt(np.diag(inverse @ meat @ inverse))
    t_values = beta / standard_errors
    df = len(y) - x.shape[1]
    critical = stats.t.ppf(0.975, df)
    assert fit.estimates == pytest.approx(beta, abs=1e-12)
    assert fit.standard_errors == pytest.approx(standard_errors, abs=1e-12)
    assert fit.t_values == pytest.approx(t_values, abs=1e-12)
    assert fit.ci_low == pytest.approx(beta - critical * standard_errors, abs=1e-12)
    assert fit.two_sided_p == pytest.approx(2 * stats.t.sf(abs(t_values), df), abs=1e-12)
    assert one_sided_greater_p(float(t_values[1]), df) == pytest.approx(
        stats.t.sf(t_values[1], df), abs=1e-15
    )


@pytest.mark.parametrize(
    "x,y,reason",
    [
        (np.empty((0, 1)), np.empty(0), "empty_sample"),
        (np.ones((2, 2)), np.ones(2), "insufficient_residual_degrees_of_freedom"),
        (np.ones((5, 2)), np.arange(5.0), "rank_deficient"),
    ],
)
def test_unestimable_models_fail_explicitly(x, y, reason) -> None:
    rows = tuple(event_record(index) for index in range(len(y)))
    result = fit_hc3(ModelData("fixture", "fixture", tuple(f"t{i}" for i in range(x.shape[1])), rows, y, x))
    assert result.status == "unestimable"
    assert result.reason == reason


def test_holm_preserves_four_slots_and_hides_placeholder() -> None:
    members = ("H1_model0", "H2_model1", "H3_model2", "H4_model2")
    raw = dict(zip(members, [0.01, None, 0.04, 0.20], strict=True))
    adjusted = holm_adjust(raw, members)
    assert adjusted == {
        "H1_model0": pytest.approx(0.04),
        "H2_model1": None,
        "H3_model2": pytest.approx(0.12),
        "H4_model2": pytest.approx(0.4),
    }


def test_common_sample_attenuation_and_zero_denominator() -> None:
    events = rich_events()
    negative = tuple(row for row in events if row.sign == "negative")
    data = build_model_data("model1_cc", "cc", negative)
    data2 = build_model_data("model2", "cc", negative)
    first = fit_hc3(data)
    second = fit_hc3(data2)
    rows = attenuation_rows(first, second, 1e-12)
    expected = first.coefficient("weekend") - second.coefficient("weekend")
    assert rows[0]["estimate"] == pytest.approx(expected)
    zero = replace(first, estimates=np.asarray([0.0, 0.0]))
    assert attenuation_rows(zero, second, 1e-12)[1]["status"] == "undefined_zero_denominator"


def test_month_block_bootstrap_is_deterministic_and_counts_failures(config) -> None:
    rows = tuple(event_record(index, month=index % 3 + 1) for index in range(36))
    data = build_model_data("model1", "fixture", rows)
    first = month_block_bootstrap(
        data, "weekend", config, analysis="fixture", target=80, maximum_attempts=200
    )
    second = month_block_bootstrap(
        data, "weekend", config, analysis="fixture", target=80, maximum_attempts=200
    )
    assert first == second
    assert first.valid == 80
    assert first.attempted == first.valid + first.empty + first.rank_deficient
    assert first.low is not None and first.high is not None


def test_month_bootstrap_carries_every_event_in_each_selected_block(
    config, monkeypatch
) -> None:
    rows = tuple(event_record(index, month=index % 3 + 1) for index in range(30))
    original_counts = Counter(row.calendar_month for row in rows)
    observed: list[Counter[str]] = []
    original_fit = inference._coefficient_by_lstsq

    def capture(data):
        observed.append(Counter(row.calendar_month for row in data.rows))
        return original_fit(data)

    monkeypatch.setattr(inference, "_coefficient_by_lstsq", capture)
    month_block_bootstrap(
        build_model_data("model0", "fixture", rows),
        "intercept",
        config,
        analysis="whole_block_fixture",
        target=8,
        maximum_attempts=20,
    )
    assert observed
    assert all(
        count % original_counts[month] == 0
        for draw in observed
        for month, count in draw.items()
    )


def test_group_comparisons_use_indicator_regressions_and_bootstrap(config, monkeypatch) -> None:
    original = month_block_bootstrap

    def small_bootstrap(data, term, config, *, analysis):
        return original(
            data,
            term,
            config,
            analysis=analysis,
            target=20,
            maximum_attempts=100,
        )

    monkeypatch.setattr(inference, "month_block_bootstrap", small_bootstrap)
    rows, bootstraps = build_group_comparisons(rich_events(), config)
    assert [row["comparison"] for row in rows] == [
        "weekend_vs_weekday",
        "low_vs_other_activity",
        "combined_vs_remaining",
    ]
    assert all(int(row["n_1"]) + int(row["n_0"]) > 0 for row in rows)
    assert all(result.valid == 20 for result in bootstraps)


def test_diagnostics_flag_but_do_not_delete_and_label_model3(config) -> None:
    models = build_primary_models(rich_events())
    fits = {name: fit_hc3(data) for name, data in models.items()}
    records = build_diagnostics(fits, config)
    cook = [row for row in records if row["diagnostic"] == "cooks_distance_above_4_over_n"]
    assert cook
    assert all(row["status"] == "flagged_not_deleted" for row in cook)
    interaction = [row for row in records if row["diagnostic"] == "interaction_interpretability"]
    assert interaction[0]["status"] in {"exploratory_underpowered", "quantitatively_interpretable"}
    assert any(row["diagnostic"] == "variance_inflation_factor" for row in records)


def test_cross_sign_overlap_uses_strict_twelve_hour_window() -> None:
    negative = event_record(0, outcome=0.01)
    positive_overlap = replace(
        event_record(0, sign="positive", outcome=-0.01),
        signal_close_time=negative.signal_close_time + timedelta(hours=11),
    )
    positive_boundary = replace(
        event_record(1, sign="positive", outcome=-0.01),
        signal_close_time=negative.signal_close_time + timedelta(hours=12),
    )
    assert count_cross_sign_overlaps((negative, positive_overlap, positive_boundary)) == 1


def test_stage6_event_and_balanced_path_reconciliation(config) -> None:
    event = event_record(0, outcome=0.012)
    paths = tuple(
        PathRecord(
            event.event_id,
            event.signal_close_time,
            step,
            event.outcome_12h if step == 12 else step / 1000,
        )
        for step in range(1, 25)
    )
    reconcile_stage6_records(
        (event,),
        paths,
        (inference.DriftRecord("reference-1", "2020-01", 0.0),),
        config,
    )
    with pytest.raises(InferenceError, match="step 12"):
        reconcile_stage6_records(
            (event,),
            tuple(replace(row, value=0.5) if row.step == 12 else row for row in paths),
            (inference.DriftRecord("reference-1", "2020-01", 0.0),),
            config,
        )


def test_any_sign_diagnostic_uses_warmup_and_does_not_change_primary_rows(config) -> None:
    start = datetime(2019, 12, 31, tzinfo=UTC)
    feature_rows = []
    candidates = []
    for index in range(80):
        moment = start + timedelta(hours=index)
        period = "warmup" if index < 24 else "development"
        zscore = -3.0 if index == 20 else 3.0 if index == 30 else -3.1 if index == 50 else 0.0
        feature_rows.append(outcome_input(moment, str(100 + index), period=period, zscore=zscore))
        close_time = moment + timedelta(hours=1) - timedelta(milliseconds=1)
        candidates.append(
            HourCandidate(
                period,
                moment,
                close_time,
                int(close_time.timestamp() * 1000),
                True,
                zscore,
            )
        )
    # Magnitude fields can be empty when the any-sign rule accepts an hour that
    # the primary per-sign cooldown rejected.
    feature_rows[50] = replace(
        feature_rows[50], shock_magnitude=None, absolute_shock_z=None
    )
    output = build_any_sign_event_records(feature_rows, candidates, config)
    # The positive shock ten hours after the warm-up negative is suppressed;
    # the later negative is kept.
    assert [row.sign for row in output] == ["negative"]
    assert output[0].signal_open_time == start + timedelta(hours=50)
    assert output[0].shock_magnitude == pytest.approx(3.1)
    assert not any(row.negative_event or row.positive_event for row in feature_rows)


def test_rendering_and_figures_are_deterministic_and_schema_exact() -> None:
    records = [
        {
            "section": "x",
            "comparison": "x",
            "group": "x",
            "variable": "x",
            "statistic": "x",
            "value": 0.1,
            "n": 1,
            "status": "x",
        }
    ]
    rendered = render_records(records, inference.DESCRIPTIVE_COLUMNS)
    assert rendered.splitlines()[0] == ",".join(inference.DESCRIPTIVE_COLUMNS)
    events = rich_events()
    path_summary = [
        {
            "horizon_step_hours": step,
            "event_count": 2,
            "mean_cumulative_log_return": step / 1000,
            "median_cumulative_log_return": step / 1000,
            "standard_deviation": 0.0,
            "q05": step / 1000,
            "q25": step / 1000,
            "q75": step / 1000,
            "q95": step / 1000,
        }
        for step in range(1, 25)
    ]
    hours = [
        outcome_input(
            datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=index),
            str(100 + index),
            period="development",
            zscore=0.0,
        )
        for index in range(96)
    ]
    first = render_figures(events, path_summary, hours)
    second = render_figures(events, path_summary, hours)
    assert [name for name, _ in first] == [
        "development_price_events.png",
        "development_average_path.png",
        "development_weekend_distribution.png",
        "development_activity_distribution.png",
    ]
    assert all(payload.startswith(b"\x89PNG\r\n\x1a\n") for _, payload in first)
    assert first == second


def test_complete_synthetic_inference_bundle_covers_h1_through_h5(
    config, monkeypatch
) -> None:
    events = rich_events()
    path_events = [row for row in events if row.sign == "negative"][:2]
    paths = tuple(
        PathRecord(row.event_id, row.signal_close_time, step, step / 1000 + offset)
        for offset, row in enumerate(path_events)
        for step in range(1, 25)
    )
    drift = tuple(
        inference.DriftRecord(f"reference-{index}", f"2020-{index % 4 + 1:02d}", index / 10000)
        for index in range(20)
    )
    hours = [
        outcome_input(
            datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=index),
            str(100 + index),
            period="development",
            zscore=0.0,
        )
        for index in range(96)
    ]
    original = month_block_bootstrap

    def small_bootstrap(data, term, config, *, analysis):
        return original(
            data, term, config, analysis=analysis, target=12, maximum_attempts=80
        )

    monkeypatch.setattr(inference, "month_block_bootstrap", small_bootstrap)
    bundle = build_inference_bundle(events, paths, drift, events, hours, config)
    hypotheses = {row["hypothesis"] for row in bundle.model_results}
    assert {"H1_model0", "H2_model1", "H3_model2", "H4_model2", "H5"} <= hypotheses
    assert len(bundle.bootstrap_results) == 13
    assert all(
        result.attempted == result.valid + result.empty + result.rank_deficient
        for result in bundle.bootstrap_results
    )
    assert next(result for result in bundle.bootstrap_results if result.model == "model0").valid == 12
    assert len(bundle.figures) == 4
    assert any(row["section"] == "market_drift_reference" for row in bundle.descriptives)


def test_inference_bundle_uses_confirmatory_held_out_labels_and_filenames(
    config, monkeypatch
) -> None:
    events = rich_events()
    paths = tuple(
        PathRecord(row.event_id, row.signal_close_time, step, step / 1000)
        for row in events
        if row.sign == "negative"
        for step in range(1, 25)
    )
    drift = tuple(
        inference.DriftRecord(
            f"reference-{index}", f"2020-{index % 4 + 1:02d}", index / 10000
        )
        for index in range(20)
    )
    hours = [
        replace(
            outcome_input(
                datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=index),
                str(100 + index),
                period="development",
                zscore=0.0,
            ),
            analysis_period="held_out",
        )
        for index in range(96)
    ]
    original = month_block_bootstrap

    def small_bootstrap(data, term, config, *, analysis):
        return original(
            data, term, config, analysis=analysis, target=12, maximum_attempts=80
        )

    monkeypatch.setattr(inference, "month_block_bootstrap", small_bootstrap)
    bundle = build_inference_bundle(
        events,
        paths,
        drift,
        events,
        hours,
        config,
        analysis_period="held_out",
        label="Held-out",
    )
    primary = {
        row["interpretation"]
        for row in bundle.model_results
        if row["hypothesis"] in {"H1_model0", "H2_model1", "H3_model2", "H4_model2"}
    }
    assert primary <= {"supported_in_held_out", "not_supported_in_held_out"}
    assert {name for name, _ in bundle.figures} == {
        "held_out_price_events.png",
        "held_out_average_path.png",
        "held_out_weekend_distribution.png",
        "held_out_activity_distribution.png",
    }


def test_import_has_no_project_data_io(project_root) -> None:
    script = r'''
import sys
opened = []
def audit(event, args):
    if event == "open":
        path = str(args[0])
        if "BuyTheWeekendDip" in path and "outputs/runs" in path:
            opened.append(path)
sys.addaudithook(audit)
import src.inference
assert opened == [], opened
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_development_pipeline_dispatch_is_thin(monkeypatch, capsys) -> None:
    result = DevelopmentInferenceResult(
        "run",
        Path("outputs/runs/run/development-inference.manifest.json"),
        Path("outputs/runs/run/development-inference.manifest.json.sha256"),
        Path("outputs/runs/run/run.log.jsonl"),
        Path("outputs/runs/run/development_inference_inventory.csv"),
        10,
        20,
        10,
    )
    monkeypatch.setattr(pipeline, "load_config", lambda: object())
    monkeypatch.setattr(inference, "run_development_inference", lambda config: result)
    assert pipeline.dispatch("development") == 0
    output = capsys.readouterr().out
    assert "held-out outcomes       not accessed" in output


def test_stage7_run_writes_and_reverifies_immutable_evidence(
    config, temp_acquisition_root, monkeypatch
) -> None:
    dummy = InferenceBundle(
        descriptives=(),
        group_comparisons=(),
        path_summary=(),
        model_results=(),
        bootstrap_results=(),
        diagnostics=(),
        figures=tuple(
            (name, b"\x89PNG\r\n\x1a\nsynthetic")
            for name in (
                "development_price_events.png",
                "development_average_path.png",
                "development_weekend_distribution.png",
                "development_activity_distribution.png",
            )
        ),
        event_rows=12,
        h5_any_sign_events=9,
    )
    monkeypatch.setattr(
        inference,
        "verify_inference_inputs",
        lambda *args, **kwargs: (
            {
                "development_event_outcomes.csv": Path("events.csv"),
                "development_event_paths_24h.csv": Path("paths.csv"),
                "development_drift_reference.csv": Path("drift.csv"),
            },
            Path("features.csv"),
            1,
            {},
        ),
    )
    monkeypatch.setattr(inference, "read_event_records", lambda *args: ())
    monkeypatch.setattr(inference, "read_path_records", lambda *args: ())
    monkeypatch.setattr(inference, "read_drift_records", lambda *args: ())
    monkeypatch.setattr(inference, "read_outcome_rows", lambda *args: ())
    monkeypatch.setattr(inference, "read_any_sign_candidates", lambda *args: ())
    monkeypatch.setattr(inference, "build_any_sign_event_records", lambda *args: ())
    monkeypatch.setattr(inference, "build_inference_bundle", lambda *args: dummy)
    result = run_development_inference(
        config,
        outcome_manifest_relative=Path("outcome.manifest.json"),
        feature_manifest_relative=Path("feature.manifest.json"),
    )
    manifest = read_manifest(result.manifest_relative_path)
    assert manifest["stage"] == inference.STAGE
    assert manifest["exclusions"] == {"held_out_rows_analysed": 0}
    assert len(verify_inference_inventory(result.manifest_relative_path)) == 10
    log = (temp_acquisition_root / result.log_relative_path).read_text(encoding="utf-8")
    assert '"held_out_outcomes_accessed": false' in log
    first_table = temp_acquisition_root / result.manifest_relative_path.parent / next(
        iter(inference.TABLE_SCHEMAS)
    )
    first_table.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(InferenceError, match="checksum failed"):
        verify_inference_inventory(result.manifest_relative_path)


def test_default_stage7_refuses_dirty_tree_before_opening_log(config, monkeypatch) -> None:
    draft = build_manifest(inference.STAGE, inference.MODE, config)
    dirty = replace(draft, code_identity={**draft.code_identity, "git_dirty": True})
    monkeypatch.setattr(inference, "build_manifest", lambda *args, **kwargs: dirty)
    monkeypatch.setattr(
        inference,
        "create_run_log",
        lambda *args, **kwargs: pytest.fail("dirty execution opened a run log"),
    )
    with pytest.raises(InferenceError, match="clean committed repository"):
        run_development_inference(config)


def test_accepted_and_synthetic_stage7_inputs_cannot_be_mixed(config) -> None:
    with pytest.raises(InferenceError, match="cannot be mixed"):
        run_development_inference(
            config,
            outcome_manifest_relative=inference.ACCEPTED_OUTCOME_MANIFEST,
            feature_manifest_relative=Path("synthetic.manifest.json"),
        )
