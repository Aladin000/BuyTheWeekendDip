"""Offline checks for development-only outcomes."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src import outcomes, pipeline
from src.acquisition import verify_manifest_checksum
from src.artifacts import build_manifest, read_manifest
from src.count_audit import COUNT_OUTPUT_COLUMNS
from src.features import FEATURE_COLUMNS
from src.outcomes import (
    DRIFT_OUTPUT_COLUMNS,
    EVENT_OUTPUT_COLUMNS,
    INVENTORY_COLUMNS,
    PATH_OUTPUT_COLUMNS,
    DevelopmentOutcomeResult,
    Gate3Expectations,
    OutcomeError,
    OutcomeInputRow,
    build_outcomes,
    build_development_outcomes,
    read_gate3_expectations,
    read_outcome_rows,
    reconcile_with_gate3,
    render_drift_reference,
    render_event_outcomes,
    render_inventory,
    render_paths,
    verify_development_inventory,
)

UTC = timezone.utc


def input_row(
    moment: datetime,
    close: str | None,
    *,
    segment: int = 1,
    period: str = "development",
    sign: str = "",
    shock_zscore: float = 0.0,
    shock_eligible: bool = True,
    full_feature: bool = True,
) -> OutcomeInputRow:
    close_time = moment + timedelta(hours=1) - timedelta(milliseconds=1)
    development = period == "development"
    event_id = f"{sign}-{int(close_time.timestamp() * 1000)}" if sign else ""
    block_start = (close_time.hour // 6) * 6
    return OutcomeInputRow(
        open_time=moment,
        close_time=close_time,
        segment_id=segment,
        analysis_period=period,
        close=Decimal(close) if development and close is not None else None,
        close_time_raw=int(close_time.timestamp() * 1000) if development else None,
        shock_eligible=shock_eligible if development else False,
        negative_event=development and sign == "negative",
        positive_event=development and sign == "positive",
        event_id=event_id if development else "",
        calendar_year=close_time.year if development else None,
        calendar_month=close_time.strftime("%Y-%m") if development else "",
        weekend=(close_time.isoweekday() in (6, 7)) if development else None,
        crosses_into_monday_12h=None,
        utc_time_block=f"{block_start:02d}-{block_start + 5:02d}" if development else "",
        shock_log_return_1h=(
            -0.02 if sign == "negative" else 0.02 if sign == "positive" else 0.0
        )
        if development
        else None,
        shock_zscore=shock_zscore if development else None,
        shock_magnitude=-shock_zscore if development and sign == "negative" else None,
        absolute_shock_z=abs(shock_zscore) if development and sign else None,
        activity_percentile=0.5 if development and full_feature else None,
        low_activity_score=0.5 if development and full_feature else None,
        low_activity=False if development and full_feature else None,
        rv24=0.1 if development and full_feature else None,
        full_feature_eligible=development and full_feature,
    )


def expected_one_each() -> Gate3Expectations:
    return Gate3Expectations(
        accepted_negative=1,
        accepted_positive=1,
        structurally_complete={
            (sign, horizon): 1
            for sign in ("negative", "positive")
            for horizon in (6, 12, 24)
        },
        eligible_h1_h2=1,
        eligible_h3_h4=1,
    )


def feature_record(row: OutcomeInputRow, held_out_close: str = "DO_NOT_PARSE") -> dict[str, str]:
    record = {name: "" for name in FEATURE_COLUMNS}
    development = row.analysis_period == "development"
    block_start = (row.close_time.hour // 6) * 6
    record.update(
        {
            "open_time": row.open_time.isoformat().replace("+00:00", "Z"),
            "close_time": row.close_time.isoformat().replace("+00:00", "Z"),
            "close_time_raw": str(int(row.close_time.timestamp() * 1000)),
            "close": (
                format(row.close, "f")
                if development and row.close is not None
                else held_out_close
            ),
            "contiguous_segment_id": str(row.segment_id),
            "analysis_period": row.analysis_period,
            "shock_eligible_hour": str(row.shock_eligible).lower(),
            "negative_event": str(row.negative_event).lower(),
            "positive_event": str(row.positive_event).lower(),
            "event_id": row.event_id,
            "calendar_year": str(row.close_time.year),
            "calendar_month": row.close_time.strftime("%Y-%m"),
            "weekend": str(row.close_time.isoweekday() in (6, 7)).lower(),
            "utc_time_block": f"{block_start:02d}-{block_start + 5:02d}",
            "log_return_1h": (
                "" if row.shock_log_return_1h is None else str(row.shock_log_return_1h)
            ),
            "shock_zscore": "" if row.shock_zscore is None else str(row.shock_zscore),
            "shock_magnitude": "" if row.shock_magnitude is None else str(row.shock_magnitude),
            "absolute_shock_z": "" if row.absolute_shock_z is None else str(row.absolute_shock_z),
            "activity_percentile_90d": (
                "" if row.activity_percentile is None else str(row.activity_percentile)
            ),
            "low_activity_score": (
                "" if row.low_activity_score is None else str(row.low_activity_score)
            ),
            "low_activity": "" if row.low_activity is None else str(row.low_activity).lower(),
            "rv24": "" if row.rv24 is None else str(row.rv24),
            "full_feature_eligible_hour": str(row.full_feature_eligible).lower(),
        }
    )
    return record


def render_feature_rows(rows: list[OutcomeInputRow], held_out_close: str = "DO_NOT_PARSE") -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(FEATURE_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(feature_record(row, held_out_close) for row in rows)
    return buffer.getvalue()


def test_exact_horizons_balanced_paths_and_gate3_reconciliation(config) -> None:
    start = datetime(2020, 1, 4, tzinfo=UTC)
    rows = [input_row(start + timedelta(hours=i), str(100 + i)) for i in range(60)]
    rows[0] = input_row(start, "100", sign="negative", shock_zscore=-3.0)
    rows[30] = input_row(start + timedelta(hours=30), "130", sign="positive", shock_zscore=3.0)
    bundle = build_development_outcomes(rows, config)
    reconcile_with_gate3(bundle, expected_one_each())
    negative = bundle.events[0]
    assert negative.row.event_id == rows[0].event_id
    for horizon in (6, 12, 24):
        expected = float(Decimal(str(100 + horizon)).ln() - Decimal("100").ln())
        assert negative.horizons[horizon].value == expected
        assert negative.horizons[horizon].value == pytest.approx(
            math.log((100 + horizon) / 100), abs=1e-12
        )
    assert len(bundle.paths) == 24
    assert {record.event_sign for record in bundle.paths} == {"negative"}
    assert [record.step for record in bundle.paths] == list(range(1, 25))
    assert bundle.paths[23].value == negative.horizons[24].value


def test_gap_and_split_boundaries_keep_event_identity(config) -> None:
    start = datetime(2020, 1, 4, tzinfo=UTC)
    gapped = [input_row(start + timedelta(hours=i), "100") for i in range(6)]
    gapped += [input_row(start + timedelta(hours=i + 1), "100", segment=2) for i in range(6, 31)]
    gapped[0] = input_row(start, "100", sign="negative", shock_zscore=-3.0)
    event = build_development_outcomes(gapped, config).events[0]
    assert event.row.event_id == gapped[0].event_id
    assert event.horizons[6].value is None
    assert event.horizons[6].reason == "missing_or_noncontiguous_future_bar"

    boundary = [input_row(start + timedelta(hours=i), "100") for i in range(10)]
    boundary += [
        input_row(start + timedelta(hours=i), None, period="held_out")
        for i in range(10, 30)
    ]
    boundary[0] = input_row(start, "100", sign="negative", shock_zscore=-3.0)
    event = build_development_outcomes(boundary, config).events[0]
    assert event.horizons[6].complete is True
    assert event.horizons[12].reason == "crosses_development_boundary"

    sample_end = [input_row(start + timedelta(hours=i), "100") for i in range(6)]
    sample_end[0] = input_row(start, "100", sign="negative", shock_zscore=-3.0)
    event = build_development_outcomes(sample_end, config).events[0]
    assert event.horizons[6].reason == "end_of_available_timestamps"


def test_model_eligibility_is_separate_from_event_and_strategy(config) -> None:
    start = datetime(2020, 1, 4, tzinfo=UTC)
    rows = [input_row(start + timedelta(hours=i), "100") for i in range(30)]
    rows[0] = input_row(start, "100", sign="negative", shock_zscore=-3.0, full_feature=False)
    event = build_development_outcomes(rows, config).events[0]
    assert event.eligible_h1_h2 is True
    assert event.eligible_h3_h4 is False
    assert event.model_reason == "incomplete_model_covariates"
    assert not any("strategy" in column or "trade" in column for column in EVENT_OUTPUT_COLUMNS)


def test_event_output_carries_the_frozen_monday_crossing_label(config) -> None:
    start = datetime(2020, 1, 5, 18, tzinfo=UTC)
    rows = [input_row(start + timedelta(hours=i), "100") for i in range(30)]
    rows[0] = replace(
        input_row(start, "100", sign="negative", shock_zscore=-3.0),
        crosses_into_monday_12h=True,
    )
    rendered = list(
        csv.DictReader(
            io.StringIO(
                render_event_outcomes(build_development_outcomes(rows, config).events)
            )
        )
    )
    assert rendered[0]["crosses_into_monday_12h"] == "true"


def test_drift_reference_uses_locked_non_candidate_rule(config) -> None:
    start = datetime(2020, 1, 4, tzinfo=UTC)
    rows = [input_row(start + timedelta(hours=i), "100") for i in range(15)]
    rows[0] = replace(rows[0], shock_zscore=-2.5)
    rows[1] = replace(rows[1], shock_zscore=-2.5001)
    rows[2] = replace(rows[2], shock_zscore=3.0)
    assert [
        record.source_index
        for record in build_development_outcomes(rows, config).drift_reference
    ] == [0, 2]


def test_reader_does_not_parse_held_out_prices(config, temp_acquisition_root) -> None:
    start = datetime(2023, 12, 31, 18, tzinfo=UTC)
    rows = [input_row(start + timedelta(hours=i), "100") for i in range(6)]
    rows[0] = input_row(start, "100", sign="negative", shock_zscore=-3.0)
    rows += [
        input_row(start + timedelta(hours=i), None, period="held_out")
        for i in range(6, 9)
    ]
    rendered: list[str] = []
    for filename, poison in (("first.csv", "NOT_A_NUMBER_A"), ("second.csv", "NOT_A_NUMBER_B")):
        relative = Path(filename)
        (temp_acquisition_root / relative).write_text(
            render_feature_rows(rows, poison), encoding="utf-8"
        )
        parsed = read_outcome_rows(relative, len(rows), config)
        assert [row.close for row in parsed[-3:]] == [None, None, None]
        rendered.append(render_event_outcomes(build_development_outcomes(parsed, config).events))
    assert rendered[0] == rendered[1]


def test_reader_and_builder_can_target_held_out_without_parsing_development(
    config, temp_acquisition_root
) -> None:
    development = input_row(
        datetime(2023, 12, 31, 23, tzinfo=UTC),
        "100",
    )
    held_out_start = config.splits.held_out.start_inclusive
    held_out = [
        replace(
            input_row(
                held_out_start + timedelta(hours=index),
                str(100 + index),
                sign="negative" if index == 0 else "",
                shock_zscore=-3.0 if index == 0 else 0.0,
            ),
            analysis_period="held_out",
        )
        for index in range(30)
    ]
    records = [feature_record(development), *[
        feature_record(row, str(100 + index))
        for index, row in enumerate(held_out)
    ]]
    records[0]["close"] = "DO_NOT_PARSE"
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(FEATURE_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(records)
    relative = Path("held_out_features.csv")
    (temp_acquisition_root / relative).write_text(
        buffer.getvalue(), encoding="utf-8"
    )

    parsed = read_outcome_rows(
        relative, len(records), config, analysis_period="held_out"
    )
    assert parsed[0].close is None
    assert parsed[1].close == Decimal("100")
    bundle = build_outcomes(parsed, config, analysis_period="held_out")
    assert len(bundle.events) == 1
    assert bundle.events[0].row.event_id.startswith("negative-")
    assert bundle.events[0].horizons[24].complete is True


def test_reader_requires_the_exact_stable_event_id(config, temp_acquisition_root) -> None:
    row = input_row(
        datetime(2020, 1, 4, tzinfo=UTC),
        "100",
        sign="negative",
        shock_zscore=-3.0,
    )
    record = feature_record(row)
    record["event_id"] = "negative-wrong-timestamp"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(FEATURE_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerow(record)
    relative = Path("wrong_event_id.csv")
    (temp_acquisition_root / relative).write_text(buffer.getvalue(), encoding="utf-8")
    with pytest.raises(OutcomeError, match="wrong negative event ID"):
        read_outcome_rows(relative, 1, config)


def test_reader_rejects_an_event_that_is_not_shock_eligible(
    config, temp_acquisition_root
) -> None:
    row = input_row(
        datetime(2020, 1, 4, tzinfo=UTC),
        "100",
        sign="negative",
        shock_zscore=-3.0,
        shock_eligible=False,
    )
    relative = Path("ineligible_event.csv")
    (temp_acquisition_root / relative).write_text(
        render_feature_rows([row]), encoding="utf-8"
    )
    with pytest.raises(OutcomeError, match="ineligible accepted event"):
        read_outcome_rows(relative, 1, config)


def test_gate3_expectations_are_read_by_exact_identity(temp_acquisition_root) -> None:
    relative = Path("counts.csv")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(COUNT_OUTPUT_COLUMNS), lineterminator="\n")
    writer.writeheader()

    def emit(measure: str, sign: str, count: int, horizon: str = "", model: str = "") -> None:
        record = {name: "" for name in COUNT_OUTPUT_COLUMNS}
        record.update(
            {
                "scope_type": "split",
                "scope": "development",
                "sign": sign,
                "horizon_hours": horizon,
                "model_or_strategy": model,
                "measure": measure,
                "count": str(count),
            }
        )
        writer.writerow(record)

    emit("accepted_events", "negative", 1)
    emit("accepted_events", "positive", 1)
    for sign in ("negative", "positive"):
        for horizon in (6, 12, 24):
            emit("structurally_complete_events", sign, 1, str(horizon))
    emit("structurally_eligible_events", "negative", 1, "12", "H1_H2")
    emit("structurally_eligible_events", "negative", 1, "12", "H3_H4")
    (temp_acquisition_root / relative).write_text(buffer.getvalue(), encoding="utf-8")
    assert read_gate3_expectations(relative) == expected_one_each()


def test_renderers_and_inventory_are_deterministic(config) -> None:
    start = datetime(2020, 1, 4, tzinfo=UTC)
    rows = [input_row(start + timedelta(hours=i), str(100 + i)) for i in range(30)]
    rows[0] = input_row(start, "100", sign="negative", shock_zscore=-3.0)
    bundle = build_development_outcomes(rows, config)
    rendered = (
        (render_event_outcomes(bundle.events), EVENT_OUTPUT_COLUMNS),
        (render_paths(bundle.paths), PATH_OUTPUT_COLUMNS),
        (render_drift_reference(bundle.drift_reference), DRIFT_OUTPUT_COLUMNS),
    )
    for content, schema in rendered:
        assert tuple(next(csv.reader(io.StringIO(content)))) == schema
    event_content = rendered[0][0]
    inventory = render_inventory(
        [
            (
                Path("events.csv"),
                hashlib.sha256(event_content.encode()).hexdigest(),
                1,
                EVENT_OUTPUT_COLUMNS,
            )
        ]
    )
    parsed = list(csv.DictReader(io.StringIO(inventory)))
    assert tuple(parsed[0]) == INVENTORY_COLUMNS
    assert json.loads(parsed[0]["column_schema_json"]) == list(EVENT_OUTPUT_COLUMNS)


def test_end_to_end_writes_checksum_bound_inventory(
    config, temp_acquisition_root, monkeypatch
) -> None:
    start = datetime(2020, 1, 4, tzinfo=UTC)
    rows = [input_row(start + timedelta(hours=i), str(100 + i)) for i in range(60)]
    rows[0] = input_row(start, "100", sign="negative", shock_zscore=-3.0)
    rows[30] = input_row(start + timedelta(hours=30), "130", sign="positive", shock_zscore=3.0)
    feature_relative = Path("synthetic_features.csv")
    (temp_acquisition_root / feature_relative).write_text(
        render_feature_rows(rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        outcomes,
        "verify_development_inputs",
        lambda *args, **kwargs: (
            feature_relative,
            len(rows),
            expected_one_each(),
            {},
        ),
    )
    result = outcomes.run_development_outcomes(
        config,
        feature_manifest_relative=Path("feature.manifest.json"),
        count_manifest_relative=Path("count.manifest.json"),
    )
    manifest = read_manifest(result.manifest_relative_path)
    verify_manifest_checksum(result.manifest_relative_path)
    inventory_path = temp_acquisition_root / result.inventory_relative_path
    assert manifest["output_checksum"] == hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    with inventory_path.open(newline="") as handle:
        inventory = list(csv.DictReader(handle))
    assert len(inventory) == 3
    for record in inventory:
        artifact = temp_acquisition_root / record["artifact_path"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == record["sha256"]
        with artifact.open(newline="") as handle:
            reader = csv.reader(handle)
            assert next(reader) == json.loads(record["column_schema_json"])
            assert sum(1 for _ in reader) == int(record["row_count"])
    assert len(verify_development_inventory(result.manifest_relative_path)) == 3
    assert '"held_out_outcomes_accessed": false' in (
        temp_acquisition_root / result.log_relative_path
    ).read_text()
    (temp_acquisition_root / result.event_relative_path).write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(OutcomeError, match="checksum failed"):
        verify_development_inventory(result.manifest_relative_path)


def test_default_development_run_refuses_a_dirty_tree_before_opening_a_log(
    config, monkeypatch
) -> None:
    draft = build_manifest(outcomes.STAGE, outcomes.MODE, config)
    dirty = replace(
        draft,
        code_identity={**draft.code_identity, "git_dirty": True},
    )
    monkeypatch.setattr(outcomes, "build_manifest", lambda *args, **kwargs: dirty)
    monkeypatch.setattr(
        outcomes,
        "create_run_log",
        lambda *args, **kwargs: pytest.fail("dirty execution opened a run log"),
    )
    with pytest.raises(OutcomeError, match="clean committed repository"):
        outcomes.run_development_outcomes(config)


def test_accepted_and_synthetic_inputs_cannot_be_mixed(config) -> None:
    with pytest.raises(OutcomeError, match="cannot be mixed"):
        outcomes.run_development_outcomes(
            config,
            feature_manifest_relative=outcomes.ACCEPTED_FEATURE_MANIFEST,
            count_manifest_relative=Path("synthetic-count.manifest.json"),
        )


def test_stage6_summary_remains_available_after_pipeline_advances(capsys) -> None:
    result = DevelopmentOutcomeResult(
        "run",
        Path("outputs/runs/run/development-outcomes.manifest.json"),
        Path("outputs/runs/run/development-outcomes.manifest.json.sha256"),
        Path("outputs/runs/run/run.log.jsonl"),
        Path("outputs/runs/run/development_output_inventory.csv"),
        Path("outputs/runs/run/development_event_outcomes.csv"),
        Path("outputs/runs/run/development_event_paths_24h.csv"),
        Path("outputs/runs/run/development_drift_reference.csv"),
        2,
        24,
        10,
    )
    print("\n".join(outcomes.summarise(result)))
    assert "held-out outcomes       not accessed" in capsys.readouterr().out


def test_import_hygiene_and_held_out_refusal(project_root, monkeypatch) -> None:
    program = """
import sys
opened=[]
network=[]
def hook(event,args):
    if event == 'open': opened.append(str(args[0]))
    elif event.startswith(('socket.','urllib.')): network.append(event)
sys.addaudithook(hook)
import src.outcomes
bad=sorted({p for p in opened if p.endswith(('project.yaml','.zip','.csv'))}|set(network))
print('BAD '+repr(bad) if bad else 'CLEAN')
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(project_root)},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "CLEAN"
    monkeypatch.setattr(pipeline, "held_out_authorised", lambda: False)
    with pytest.raises(pipeline.AuthorisationError):
        pipeline.dispatch("held-out")


def test_gate3_reconciliation_fails_closed(config) -> None:
    start = datetime(2020, 1, 4, tzinfo=UTC)
    rows = [input_row(start + timedelta(hours=i), "100") for i in range(30)]
    rows[0] = input_row(start, "100", sign="negative", shock_zscore=-3.0)
    with pytest.raises(OutcomeError, match="event counts"):
        reconcile_with_gate3(build_development_outcomes(rows, config), expected_one_each())
