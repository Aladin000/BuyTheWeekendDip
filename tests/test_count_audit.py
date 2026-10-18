"""Offline checks for the count-only audit."""

from __future__ import annotations

import csv
import hashlib
import io
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import count_audit, pipeline
from src.acquisition import verify_manifest_checksum, write_manifest_checksum
from src.artifacts import build_manifest, read_manifest, run_relative_dir
from src.count_audit import (
    ACCEPTED_FEATURE_MANIFEST,
    COUNT_INPUT_COLUMNS,
    COUNT_OUTPUT_COLUMNS,
    CountAuditError,
    CountAuditResult,
    CountInputRow,
    build_count_records,
    combined_strategy_interpretation_label,
    model3_interpretation_label,
    read_count_rows,
    render_count_records,
    structural_horizon_flags,
)
from src.features import FEATURE_COLUMNS

UTC = timezone.utc


def count_row(
    moment: datetime,
    *,
    segment: int = 0,
    period: str = "development",
    shock_eligible: bool = True,
    full_feature_eligible: bool = True,
    negative_candidate: bool = False,
    positive_candidate: bool = False,
    negative_event: bool = False,
    positive_event: bool = False,
    low_activity: bool | None = False,
) -> CountInputRow:
    close = moment + timedelta(hours=1) - timedelta(milliseconds=1)
    sign = "negative" if negative_event else "positive" if positive_event else ""
    return CountInputRow(
        open_time=moment,
        close_time=close,
        segment_id=segment,
        shock_eligible=shock_eligible,
        full_feature_eligible=full_feature_eligible,
        negative_candidate=negative_candidate or negative_event,
        positive_candidate=positive_candidate or positive_event,
        negative_event=negative_event,
        positive_event=positive_event,
        event_id=f"{sign}-{int(close.timestamp() * 1000)}" if sign else "",
        analysis_period=period,
        calendar_year=close.year,
        weekend=close.isoweekday() in (6, 7),
        low_activity=low_activity if full_feature_eligible else None,
    )


def find_record(records, *, scope: str, measure: str, **fields):
    matches = [
        record
        for record in records
        if record.scope == scope
        and record.measure == measure
        and all(getattr(record, name) == value for name, value in fields.items())
    ]
    assert len(matches) == 1
    return matches[0]


def feature_record(row: CountInputRow, *, price_marker: str = "not-used") -> dict[str, str]:
    record = {name: "" for name in FEATURE_COLUMNS}
    record.update(
        {
            "open_time": row.open_time.isoformat().replace("+00:00", "Z"),
            "close_time": row.close_time.isoformat().replace("+00:00", "Z"),
            "contiguous_segment_id": str(row.segment_id),
            "shock_eligible_hour": str(row.shock_eligible).lower(),
            "full_feature_eligible_hour": str(row.full_feature_eligible).lower(),
            "negative_candidate": str(row.negative_candidate).lower(),
            "positive_candidate": str(row.positive_candidate).lower(),
            "negative_event": str(row.negative_event).lower(),
            "positive_event": str(row.positive_event).lower(),
            "event_id": row.event_id,
            "analysis_period": row.analysis_period,
            "calendar_year": str(row.calendar_year),
            "weekend": str(row.weekend).lower(),
            "low_activity": (
                "" if row.low_activity is None else str(row.low_activity).lower()
            ),
        }
    )
    for name in ("open", "high", "low", "close"):
        record[name] = price_marker
    return record


def render_feature_fixture(rows: list[CountInputRow], *, price_marker: str = "not-used") -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(FEATURE_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(feature_record(row, price_marker=price_marker) for row in rows)
    return buffer.getvalue()


def feature_manifest_fixture(config, root: Path, rows: list[CountInputRow]) -> Path:
    draft = build_manifest("4. Pre-outcome features", "features", config)
    run_dir = run_relative_dir(draft.run_id, config)
    feature_relative = run_dir / "pre_outcome_features.csv"
    content = render_feature_fixture(rows)
    destination = root / feature_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    finished = draft.completed(
        hashlib.sha256(content.encode()).hexdigest(), len(rows), list(FEATURE_COLUMNS)
    )
    manifest_path = finished.write_for_run("features", config)
    manifest_relative = manifest_path.relative_to(root)
    write_manifest_checksum(manifest_relative)
    return manifest_relative


def test_g3_structural_horizons_use_only_timestamps_segments_and_splits() -> None:
    start = datetime(2023, 12, 30, tzinfo=UTC)
    rows = [count_row(start + timedelta(hours=i)) for i in range(30)]
    flags = structural_horizon_flags(rows, (6, 12, 24))
    assert flags[0] == {6: True, 12: True, 24: True}
    assert flags[18] == {6: True, 12: False, 24: False}

    gapped = list(rows)
    gapped[8] = replace(gapped[8], segment_id=1)
    for index in range(9, len(gapped)):
        gapped[index] = replace(gapped[index], segment_id=1)
    flags = structural_horizon_flags(gapped, (6, 12, 24))
    assert flags[1][6] is True
    assert flags[1][12] is False

    split = list(rows)
    for index in range(10, len(split)):
        split[index] = replace(split[index], analysis_period="held_out")
    flags = structural_horizon_flags(split, (6, 12, 24))
    assert flags[3][6] is True
    assert flags[3][12] is False


def test_g3_count_reader_ignores_every_price_cell(temp_acquisition_root) -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    source_rows = [count_row(start + timedelta(hours=i)) for i in range(2)]
    first = Path("first.csv")
    second = Path("second.csv")
    (temp_acquisition_root / first).write_text(
        render_feature_fixture(source_rows, price_marker="FORBIDDEN_A"), encoding="utf-8"
    )
    (temp_acquisition_root / second).write_text(
        render_feature_fixture(source_rows, price_marker="FORBIDDEN_B"), encoding="utf-8"
    )
    assert read_count_rows(first, 2, cooldown_hours=24) == read_count_rows(
        second, 2, cooldown_hours=24
    )
    assert set(COUNT_INPUT_COLUMNS).isdisjoint(
        {
            "open",
            "high",
            "low",
            "close",
            "log_return_1h",
            "simple_return_1h",
            "shock_zscore",
        }
    )


def test_g3_required_counts_denominators_rates_and_warmup_exclusion(config) -> None:
    start = datetime(2019, 12, 31, 22, tzinfo=UTC)
    rows = [
        count_row(start, period="warmup", negative_event=True, low_activity=True),
        count_row(start + timedelta(hours=1), period="warmup"),
    ]
    analysis_start = datetime(2020, 1, 4, tzinfo=UTC)
    rows.extend(count_row(analysis_start + timedelta(hours=i)) for i in range(40))
    rows[2] = count_row(analysis_start, negative_event=True, low_activity=True)
    rows[26] = count_row(
        analysis_start + timedelta(hours=24),
        positive_event=True,
        full_feature_eligible=False,
    )

    records = build_count_records(rows, config)
    assert find_record(
        records,
        scope="development",
        measure="accepted_events",
        sign="negative",
        weekend_group="",
        activity_group="",
    ).count == 1
    assert find_record(
        records,
        scope="development",
        measure="accepted_events",
        sign="positive",
        weekend_group="",
        activity_group="",
    ).count == 1
    assert find_record(
        records, scope="full_analysis", measure="analysis_hours"
    ).count == 40
    rate = find_record(
        records,
        scope="development",
        measure="accepted_events_per_1000_eligible_hours",
        sign="negative",
        weekend_group="weekend",
        activity_group="",
    )
    assert rate.count == 1
    assert rate.denominator_measure == "shock_eligible_hours"
    assert rate.rate_per_1000 == pytest.approx(1000 / rate.denominator_count)
    assert find_record(
        records,
        scope="development",
        measure="structurally_eligible_events",
        sign="negative",
        model_or_strategy="H1_H2",
    ).count == 1


def test_g3_interpretation_thresholds_are_exact(config) -> None:
    assert model3_interpretation_label([10, 10, 10, 10], 89, config) == (
        "exploratory_underpowered"
    )
    assert model3_interpretation_label([10, 10, 10, 10], 90, config) == (
        "quantitatively_interpretable"
    )
    assert model3_interpretation_label([9, 30, 30, 30], 99, config) == (
        "exploratory_underpowered"
    )
    assert combined_strategy_interpretation_label(19, config) == (
        "descriptive_highly_unstable"
    )
    assert combined_strategy_interpretation_label(20, config) == "standard"


def test_g3_four_joint_cells_and_strategy_filters_reconcile(config) -> None:
    start = datetime(2020, 1, 4, tzinfo=UTC)  # Saturday
    rows = [count_row(start + timedelta(hours=i)) for i in range(100)]
    for index, low_activity in ((0, True), (24, False), (48, True), (72, False)):
        rows[index] = count_row(
            start + timedelta(hours=index),
            negative_event=True,
            low_activity=low_activity,
        )
    records = build_count_records(rows, config)
    for weekend_group, activity_group in (
        ("weekend", "low"),
        ("weekend", "other"),
        ("weekday", "low"),
        ("weekday", "other"),
    ):
        assert find_record(
            records,
            scope="development",
            measure="accepted_events",
            sign="negative",
            weekend_group=weekend_group,
            activity_group=activity_group,
        ).count == 1
        assert find_record(
            records,
            scope="development",
            measure="structurally_eligible_events",
            sign="negative",
            weekend_group=weekend_group,
            activity_group=activity_group,
            model_or_strategy="H3_H4",
        ).count == 1
    expected = {"all_shock": 4, "weekend": 2, "low_activity": 2, "combined": 1}
    for strategy, count in expected.items():
        assert find_record(
            records,
            scope="development",
            measure="structurally_eligible_events",
            sign="negative",
            model_or_strategy=strategy,
        ).count == count


def test_g3_render_is_deterministic_and_contains_no_outcome_schema(config) -> None:
    start = datetime(2020, 1, 4, tzinfo=UTC)
    rows = [count_row(start + timedelta(hours=i)) for i in range(30)]
    content = render_count_records(build_count_records(rows, config))
    assert content == render_count_records(build_count_records(rows, config))
    header = tuple(next(csv.reader(io.StringIO(content))))
    assert header == COUNT_OUTPUT_COLUMNS
    assert not any(
        token in column
        for column in header
        for token in ("price", "return", "pnl", "profit", "wealth", "sharpe")
    )


def test_g3_end_to_end_manifest_and_input_tamper_guard(
    config, temp_acquisition_root
) -> None:
    start = datetime(2020, 1, 4, tzinfo=UTC)
    rows = [count_row(start + timedelta(hours=i)) for i in range(30)]
    upstream = feature_manifest_fixture(config, temp_acquisition_root, rows)
    result = count_audit.run_count_audit(
        config, feature_manifest_relative=upstream
    )
    output = temp_acquisition_root / result.counts_relative_path
    manifest = read_manifest(result.manifest_relative_path)
    assert manifest["stage"] == "5. Count-only audit"
    assert manifest["mode"] == "count-only"
    assert manifest["row_count"] == result.output_record_count
    assert manifest["column_schema"] == list(COUNT_OUTPUT_COLUMNS)
    assert manifest["output_checksum"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["exclusions"] == {"rows_dropped": 0}
    verify_manifest_checksum(result.manifest_relative_path)
    log = (temp_acquisition_root / result.log_relative_path).read_text(encoding="utf-8")
    assert '"outcome_fields_present": false' in log

    feature = temp_acquisition_root / upstream.parent / "pre_outcome_features.csv"
    feature.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(CountAuditError, match="checksum"):
        count_audit.verify_feature_input(config, upstream)


def test_g3_default_input_is_the_accepted_corrected_run() -> None:
    assert ACCEPTED_FEATURE_MANIFEST.as_posix() == (
        "outputs/runs/20260906T003831.725064Z-features-92e5ff02-878985/"
        "features.manifest.json"
    )


def test_count_only_pipeline_dispatch_is_thin(monkeypatch, capsys) -> None:
    result = CountAuditResult(
        run_id="run",
        manifest_relative_path=Path("outputs/runs/run/count-audit.manifest.json"),
        manifest_checksum_relative_path=Path(
            "outputs/runs/run/count-audit.manifest.json.sha256"
        ),
        log_relative_path=Path("outputs/runs/run/run.log.jsonl"),
        counts_relative_path=Path("outputs/runs/run/feasibility_counts.csv"),
        input_row_count=10,
        output_record_count=20,
    )
    monkeypatch.setattr(pipeline, "load_config", lambda: object())
    monkeypatch.setattr(count_audit, "run_count_audit", lambda config: result)
    assert pipeline.dispatch("count-only") == 0
    printed = capsys.readouterr().out
    assert "count records           20" in printed
    assert "future price values     not parsed, used, or computed" in printed


_IMPORT_PROGRAM = """
import sys
opened = []
network = []
def hook(event, args):
    if event == "open": opened.append(str(args[0]))
    elif event.startswith(("socket.", "urllib.")): network.append(event)
sys.addaudithook(hook)
import src.count_audit
offenders = sorted(
    {path for path in opened if path.endswith(("project.yaml", ".zip", ".csv"))}
    | set(network)
)
print("OFFENDERS " + repr(offenders) if offenders else "CLEAN")
"""


def test_count_audit_import_reads_no_project_data_and_opens_no_socket(project_root) -> None:
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
