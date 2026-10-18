"""Offline checks that features use only past data."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import subprocess
import sys
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from src import features, pipeline
from src.acquisition import verify_manifest_checksum, write_manifest_checksum
from src.artifacts import build_manifest, read_manifest, run_relative_dir
from src.events import EventDecision
from src.features import (
    FEATURE_COLUMNS,
    CleanInputRow,
    FeatureError,
    FeatureResult,
    compute_feature_set,
    compute_log_returns,
    lagged_decimal_sums,
    lagged_percentile_numerators,
    lagged_rolling_mean_std,
    lagged_rv,
    standardise_current_returns,
    verify_validation_input,
)
from src.validation import CLEAN_COLUMNS

UTC = timezone.utc
START = datetime(2019, 10, 1, tzinfo=UTC)


def clean_input_rows(count: int) -> list[CleanInputRow]:
    rows = []
    for index in range(count):
        open_time = START + timedelta(hours=index)
        close_time = open_time + timedelta(hours=1, milliseconds=-1)
        rows.append(
            CleanInputRow(
                cells=tuple("" for _ in CLEAN_COLUMNS),
                open_time=open_time,
                close_time=close_time,
                close_time_raw=int(close_time.timestamp() * 1_000),
                close=Decimal(100) + Decimal(index) / Decimal(1_000),
                quote_volume=Decimal(1_000 + index % 7),
                trade_count=Decimal(10 + index % 3),
                segment_id=1,
                has_previous_contiguous_hour=index > 0,
            )
        )
    return rows


def test_g2_hourly_log_return_and_gap_fixture() -> None:
    logs, simple = compute_log_returns(
        [Decimal(100), Decimal(95), Decimal(90)], [1, 1, 2]
    )
    assert logs[0] is None
    assert logs[1] == pytest.approx(math.log(95 / 100), abs=1e-12)
    assert simple[1] == pytest.approx(-0.05, abs=1e-12)
    assert logs[2] is None
    assert simple[2] is None


def test_g2_shock_window_is_exactly_720_prior_returns_with_ddof_one() -> None:
    values = [None] + [index / 10_000 for index in range(1, 722)]
    segments = [1] * len(values)
    means, stds = lagged_rolling_mean_std(values, segments, window=720, ddof=1)
    expected = np.asarray(values[1:721], dtype=float)
    assert means[720] is None
    assert means[721] == pytest.approx(float(expected.mean()), abs=1e-12)
    assert stds[721] == pytest.approx(float(expected.std(ddof=1)), abs=1e-12)

    changed = list(values)
    changed[721] = 99.0
    changed_means, changed_stds = lagged_rolling_mean_std(
        changed, segments, window=720, ddof=1
    )
    assert changed_means[721] == means[721]
    assert changed_stds[721] == stds[721]


def test_g2_zero_variance_never_creates_an_infinite_zscore() -> None:
    zscores, eligible, reasons = standardise_current_returns(
        [0.0], [0.0], [0.0]
    )
    assert zscores == (None,)
    assert eligible == (False,)
    assert reasons == ("zero_shock_window_variance",)


def test_g2_activity_uses_previous_24_values_and_resets_at_gap() -> None:
    values = [Decimal(index) for index in range(1, 27)]
    sums = lagged_decimal_sums(values, [1] * 26, window=24)
    assert sums[23] is None
    assert sums[24] == sum(values[:24], Decimal(0))
    changed = list(values)
    changed[24] = Decimal(1_000_000)
    assert lagged_decimal_sums(changed, [1] * 26, window=24)[24] == sums[24]

    segmented = lagged_decimal_sums(values, [1] * 13 + [2] * 13, window=24)
    assert all(value is None for value in segmented)


def test_g2_activity_percentile_ties_and_one_third_boundary(config) -> None:
    tied = [Decimal(1)] * 2161
    numerators, counts = lagged_percentile_numerators(
        tied, [1] * len(tied), window=2160
    )
    assert counts[-1] == 2160
    assert numerators[-1] == 2160

    exactly_one_third = (
        [Decimal(0)] * 719
        + [Decimal(1)]
        + [Decimal(2)] * 1440
        + [Decimal(1)]
    )
    boundary, _ = lagged_percentile_numerators(
        exactly_one_third, [1] * len(exactly_one_third), window=2160
    )
    assert boundary[-1] == 720
    assert features.is_low_activity(boundary[-1], 2160, config) is False

    below = list(exactly_one_third)
    below[719] = Decimal(2)
    below_numerator, _ = lagged_percentile_numerators(
        below, [1] * len(below), window=2160
    )
    assert below_numerator[-1] == 719
    assert features.is_low_activity(below_numerator[-1], 2160, config) is True

    broken_segments = [1] * 1000 + [2] * (len(tied) - 1000)
    broken, broken_counts = lagged_percentile_numerators(
        tied, broken_segments, window=2160
    )
    assert broken[-1] is None
    assert broken_counts[-1] == len(tied) - 1001


def test_g2_rv24_excludes_current_return_and_resets_at_gap() -> None:
    returns = [None] + [0.1] * 25
    rv = lagged_rv(returns, [1] * len(returns), window=24)
    assert rv[24] is None
    assert rv[25] == pytest.approx(math.sqrt(24 * 0.1**2), abs=1e-12)
    changed = list(returns)
    changed[25] = 20.0
    assert lagged_rv(changed, [1] * len(changed), window=24)[25] == rv[25]
    assert lagged_rv(returns, [1] * 13 + [2] * 13, window=24)[-1] is None


def rows_from(start: datetime, count: int) -> list[CleanInputRow]:
    rows = clean_input_rows(count)
    return [
        replace(
            row,
            open_time=start + timedelta(hours=index),
            close_time=start + timedelta(hours=index + 1, milliseconds=-1),
            close_time_raw=int(
                (start + timedelta(hours=index + 1, milliseconds=-1)).timestamp()
                * 1_000
            ),
        )
        for index, row in enumerate(rows)
    ]


def test_g2_calendar_labels_require_a_complete_monday_crossing_horizon(config) -> None:
    sunday_rows = rows_from(datetime(2024, 12, 29, 13, tzinfo=UTC), 13)
    event = EventDecision(False, True, False, True, True, "positive-test", "")
    sunday = sunday_rows[0].close_time
    assert features._analysis_period(sunday, config) == "held_out"
    assert features._time_block(sunday.hour, config) == "12-17"
    assert features._crosses_into_monday(sunday_rows, 0, event, config) == "true"

    saturday_rows = rows_from(datetime(2024, 12, 28, 0, tzinfo=UTC), 13)
    assert features._crosses_into_monday(saturday_rows, 0, event, config) == "false"


def test_g2_monday_crossing_is_null_for_a_gap_or_incomplete_sample(config) -> None:
    event = EventDecision(True, False, True, False, True, "negative-test", "")
    rows = rows_from(datetime(2020, 6, 27, 19, tzinfo=UTC), 13)
    rows[7:] = [replace(row, segment_id=2) for row in rows[7:]]
    assert features._crosses_into_monday(rows, 0, event, config) == ""
    assert features._crosses_into_monday(rows[:12], 0, event, config) == ""


def test_g2_monday_crossing_is_null_for_weekdays_and_non_events(config) -> None:
    weekday_rows = rows_from(datetime(2024, 12, 30, 0, tzinfo=UTC), 13)
    event = EventDecision(True, False, True, False, True, "negative-test", "")
    no_event = EventDecision(False, False, False, False, False, "", "")
    assert features._crosses_into_monday(weekday_rows, 0, event, config) == ""
    weekend_rows = rows_from(datetime(2024, 12, 29, 13, tzinfo=UTC), 13)
    assert features._crosses_into_monday(weekend_rows, 0, no_event, config) == ""


@pytest.mark.parametrize(
    ("moment", "weekend"),
    [
        (datetime(2024, 12, 27, 23, 59, 59, 999000, tzinfo=UTC), False),
        (datetime(2024, 12, 28, 0, 59, 59, 999000, tzinfo=UTC), True),
        (datetime(2024, 12, 29, 23, 59, 59, 999000, tzinfo=UTC), True),
        (datetime(2024, 12, 30, 0, 59, 59, 999000, tzinfo=UTC), False),
        (datetime(2025, 1, 1, 0, 59, 59, 999999, tzinfo=UTC), False),
    ],
)
def test_g2_weekend_boundaries_use_unrounded_source_close_time(
    config, moment, weekend
) -> None:
    assert (moment.isoweekday() in config.calendar.weekend_iso_weekdays) is weekend


def test_g2_split_membership_uses_completed_close_time(config) -> None:
    assert features._analysis_period(
        datetime(2019, 12, 31, 23, 59, 59, 999000, tzinfo=UTC), config
    ) == "warmup"
    assert features._analysis_period(
        datetime(2020, 1, 1, 0, 59, 59, 999000, tzinfo=UTC), config
    ) == "development"
    assert features._analysis_period(
        datetime(2024, 1, 1, 0, 59, 59, 999000, tzinfo=UTC), config
    ) == "held_out"


def test_g2_event_magnitude_is_blank_for_a_cooldown_rejected_candidate(config) -> None:
    rows = clean_input_rows(2)
    rejected = EventDecision(True, False, False, False, False, "", "same_sign_cooldown")
    feature_set = features.FeatureSet(
        log_returns=(None, -0.1),
        simple_returns=(None, -0.09),
        rolling_means=(None, 0.0),
        rolling_stds=(None, 0.04),
        shock_zscores=(None, -2.6),
        shock_eligible=(False, True),
        shock_reasons=("insufficient_contiguous_shock_history", ""),
        quote_activity=(None, None),
        trade_activity=(None, None),
        quote_percentile_numerators=(None, None),
        trade_percentile_numerators=(None, None),
        activity_reference_counts=(0, 0),
        rv24=(None, None),
        full_feature_eligible=(False, False),
        full_feature_reasons=(
            "insufficient_contiguous_shock_history|insufficient_contiguous_activity_history|insufficient_contiguous_rv_history",
            "insufficient_contiguous_activity_history|insufficient_contiguous_rv_history",
        ),
        events=(EventDecision(False, False, False, False, False, "", ""), rejected),
    )
    records = list(csv.DictReader(io.StringIO(features.render_feature_rows(rows, feature_set, config))))
    assert records[1]["negative_candidate"] == "true"
    assert records[1]["negative_event"] == "false"
    assert records[1]["shock_magnitude"] == ""
    assert records[1]["absolute_shock_z"] == ""


def test_g2_changing_only_the_last_observation_leaves_every_earlier_field_fixed(
    config,
) -> None:
    rows = clean_input_rows(2200)
    baseline = compute_feature_set(rows, config)
    changed_rows = list(rows)
    changed_rows[-1] = replace(
        changed_rows[-1],
        close=Decimal(999),
        quote_volume=Decimal(999999),
        trade_count=Decimal(999999),
    )
    changed = compute_feature_set(changed_rows, config)
    for field in fields(baseline):
        assert getattr(baseline, field.name)[:-1] == getattr(changed, field.name)[:-1]
    assert baseline.quote_activity[-1] == changed.quote_activity[-1]
    assert baseline.full_feature_eligible[2184] is True
    assert baseline.activity_reference_counts[2184] == 2160


def clean_record(moment: datetime, source_row: int, close: str) -> dict[str, object]:
    open_raw = int(moment.timestamp()) * 1_000
    close_raw = open_raw + 3_600_000 - 1
    close_time = moment + timedelta(hours=1, milliseconds=-1)
    return {
        "open_time_raw": open_raw,
        "close_time_raw": close_raw,
        "source_time_unit": "milliseconds",
        "open_time": moment.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "close_time": close_time.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "base_volume": "10",
        "quote_volume": "1000",
        "trade_count": "5",
        "taker_buy_base_volume": "4",
        "taker_buy_quote_volume": "400",
        "ignore": "0",
        "source_period": moment.strftime("%Y-%m"),
        "source_file": "synthetic.zip",
        "source_member": "synthetic.csv",
        "source_row": source_row,
        "is_complete_candle": "true",
        "is_duplicate": "false",
        "has_previous_contiguous_hour": str(source_row > 1).lower(),
        "gap_before_hours": "",
        "contiguous_segment_id": 1,
        "ohlc_valid": "true",
        "volume_valid": "true",
        "anomaly_flag": "false",
        "anomaly_reviewed": "false",
        "anomaly_disposition": "",
        "exclusion_reason": "",
    }


def validation_fixture(config, root: Path) -> Path:
    draft = build_manifest("3. Validation and cleaning", "validate", config)
    run_dir = run_relative_dir(draft.run_id, config)
    clean_relative = run_dir / "clean_hourly.csv"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CLEAN_COLUMNS), lineterminator="\n")
    writer.writeheader()
    start = datetime(2024, 12, 1, tzinfo=UTC)
    records = [clean_record(start + timedelta(hours=i), i + 1, str(100 + i)) for i in range(3)]
    writer.writerows(records)
    content = buffer.getvalue()
    destination = root / clean_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    finished = draft.completed(
        hashlib.sha256(content.encode()).hexdigest(), len(records), list(CLEAN_COLUMNS)
    )
    manifest_path = finished.write_for_run("validation", config)
    manifest_relative = manifest_path.relative_to(root)
    write_manifest_checksum(manifest_relative)
    return manifest_relative


def test_g2_end_to_end_artifact_is_manifested_and_contains_no_outcomes(
    config, temp_acquisition_root
) -> None:
    upstream = validation_fixture(config, temp_acquisition_root)
    result = features.run_features(config, validation_manifest_relative=upstream)
    output = temp_acquisition_root / result.feature_relative_path
    with output.open(newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader))
        rows = list(reader)
    assert header == FEATURE_COLUMNS
    assert len(rows) == 3
    assert not any(
        token in column
        for column in header
        for token in ("post_", "forward", "outcome", "exit_price", "pnl")
    )
    manifest = read_manifest(result.manifest_relative_path)
    assert manifest["row_count"] == 3
    assert manifest["exclusions"] == {"rows_dropped": 0}
    assert manifest["output_checksum"] == hashlib.sha256(output.read_bytes()).hexdigest()
    verify_manifest_checksum(result.manifest_relative_path)

    repeated = features.run_features(config, validation_manifest_relative=upstream)
    assert (
        temp_acquisition_root / repeated.feature_relative_path
    ).read_bytes() == output.read_bytes()


def test_g2_tampered_clean_artifact_is_refused_before_feature_parsing(
    config, temp_acquisition_root
) -> None:
    upstream = validation_fixture(config, temp_acquisition_root)
    clean = temp_acquisition_root / upstream.parent / "clean_hourly.csv"
    clean.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FeatureError, match="checksum"):
        verify_validation_input(config, upstream)


def test_features_pipeline_dispatch_is_thin(monkeypatch, capsys) -> None:
    result = FeatureResult(
        run_id="run",
        manifest_relative_path=Path("outputs/runs/run/features.manifest.json"),
        manifest_checksum_relative_path=Path(
            "outputs/runs/run/features.manifest.json.sha256"
        ),
        log_relative_path=Path("outputs/runs/run/run.log.jsonl"),
        feature_relative_path=Path("outputs/runs/run/pre_outcome_features.csv"),
        row_count=10,
    )
    monkeypatch.setattr(pipeline, "load_config", lambda: object())
    monkeypatch.setattr(features, "run_features", lambda config: result)
    assert pipeline.dispatch("features") == 0
    printed = capsys.readouterr().out
    assert "feature rows            10" in printed
    assert "forward outcomes        not computed" in printed


_IMPORT_PROGRAM = """
import sys
opened = []
network = []
def hook(event, args):
    if event == "open": opened.append(str(args[0]))
    elif event.startswith(("socket.", "urllib.")): network.append(event)
sys.addaudithook(hook)
import src.features
import src.events
offenders = sorted(
    {path for path in opened if path.endswith(("project.yaml", ".zip", ".csv"))}
    | set(network)
)
print("OFFENDERS " + repr(offenders) if offenders else "CLEAN")
"""


def test_feature_imports_read_no_project_data_and_open_no_socket(project_root) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROGRAM],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(project_root)},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "CLEAN"
