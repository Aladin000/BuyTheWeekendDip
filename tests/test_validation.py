"""Offline tests for source validation and cleaning."""

from __future__ import annotations

import csv
import hashlib
import io
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from src import acquisition, pipeline, validation
from src.acquisition import month_targets
from src.artifacts import read_manifest
from src.config import ConfigNode
from src.validation import (
    CLEAN_COLUMNS,
    DOCUMENTED_SOURCE_COLUMNS,
    AnomalyCandidate,
    ReviewDecision,
    ValidationError,
    ValidationResult,
    assign_segments_and_gaps,
    build_anomaly_candidates,
    detect_timestamp_unit,
    parse_month,
    parse_source_row,
    render_anomaly_candidates,
    render_clean_rows,
    validate_review_ledger,
    verify_acquisition_inputs,
)

UTC = timezone.utc
DECEMBER_2024 = datetime(2024, 12, 1, tzinfo=UTC)
JANUARY_2025 = datetime(2025, 1, 1, tzinfo=UTC)


def _node(value):
    if isinstance(value, dict):
        return ConfigNode({key: _node(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_node(item) for item in value)
    return value


def one_month_config(config, *, active_schema: bool = False) -> ConfigNode:
    data = config.to_dict()
    data["coverage"]["raw_start_inclusive"] = DECEMBER_2024
    data["coverage"]["raw_end_exclusive"] = JANUARY_2025
    data["source"]["first_month"] = (2024, 12)
    data["source"]["last_month"] = (2024, 12)
    data["source"]["schema_verified"] = active_schema
    data["source"]["expected_column_names"] = (
        DOCUMENTED_SOURCE_COLUMNS if active_schema else None
    )
    return _node(data)


def row_at(moment: datetime, *, unit: str = "milliseconds", **replacements) -> list[str]:
    scale = 1_000 if unit == "milliseconds" else 1_000_000
    open_raw = int(moment.timestamp()) * scale
    hour = 3_600 * scale
    row = [
        str(open_raw),
        "100.0",
        "102.0",
        "98.0",
        "101.0",
        "10.0",
        str(open_raw + hour - 1),
        "1000.0",
        "5",
        "4.0",
        "400.0",
        "0",
    ]
    for key, value in replacements.items():
        row[DOCUMENTED_SOURCE_COLUMNS.index(key)] = str(value)
    return row


def csv_bytes(records) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8")


def zip_bytes(member: str, records) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, csv_bytes(records))
    return buffer.getvalue()


def write_month_archive(root: Path, target, records) -> None:
    destination = root / target.zip_relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(zip_bytes(target.expected_csv_member, records))


def parsed(config, moment: datetime, **replacements):
    target = month_targets(config)[0]
    return parse_source_row(
        row_at(moment, unit=target.expected_timestamp_unit, **replacements),
        target,
        1,
        config.coverage.raw_end_exclusive,
    )


@pytest.mark.parametrize("header", (None, DOCUMENTED_SOURCE_COLUMNS))
def test_g1_exact_zero_or_one_header_is_accepted(
    config, temp_acquisition_root, header
) -> None:
    cfg = one_month_config(config, active_schema=header is not None)
    target = month_targets(cfg)[0]
    records = [row_at(DECEMBER_2024)]
    if header is not None:
        records.insert(0, header)
    write_month_archive(temp_acquisition_root, target, records)
    rows, audit, exclusions = parse_month(cfg, target)
    assert len(rows) == 1
    assert audit.header_present is (header is not None)
    assert audit.physical_records == 1 + int(header is not None)
    assert audit.parsed_data_rows == 1
    assert audit.duplicate_rows == 0
    assert audit.invalid_rows == 0
    assert exclusions == []


def test_g1_preflight_records_header_but_final_requires_exact_tokens(
    config, temp_acquisition_root
) -> None:
    inactive = one_month_config(config)
    target = month_targets(inactive)[0]
    wrong = list(DOCUMENTED_SOURCE_COLUMNS)
    wrong[-1] = "unexpected"
    write_month_archive(temp_acquisition_root, target, [wrong, row_at(DECEMBER_2024)])
    _, audit, _ = parse_month(inactive, target)
    assert audit.header_tokens == tuple(wrong)
    with pytest.raises(ValidationError, match="header does not exactly match"):
        parse_month(one_month_config(config, active_schema=True), target)


def test_g1_activated_schema_cannot_name_the_positions_arbitrarily(
    config, temp_acquisition_root
) -> None:
    cfg = one_month_config(config, active_schema=True)
    data = cfg.to_dict()
    names = list(DOCUMENTED_SOURCE_COLUMNS)
    names[-1] = "arbitrary_unused_field"
    data["source"]["expected_column_names"] = names
    altered = _node(data)
    target = month_targets(altered)[0]
    write_month_archive(temp_acquisition_root, target, [row_at(DECEMBER_2024)])
    with pytest.raises(ValidationError, match="documented 12-field"):
        parse_month(altered, target)


@pytest.mark.parametrize(
    ("records", "message"),
    (
        ([row_at(DECEMBER_2024)[:-1]], "11 columns"),
        ([row_at(DECEMBER_2024) + ["extra"]], "13 columns"),
        ([[]], "blank"),
        ([row_at(DECEMBER_2024), list(DOCUMENTED_SOURCE_COLUMNS)], "unexpected header"),
    ),
)
def test_g1_malformed_source_records_fail_closed(
    config, temp_acquisition_root, records, message
) -> None:
    cfg = one_month_config(config)
    target = month_targets(cfg)[0]
    write_month_archive(temp_acquisition_root, target, records)
    with pytest.raises(ValidationError, match=message):
        parse_month(cfg, target)


def test_g1_timestamp_units_switch_exactly_at_2025(config) -> None:
    targets = {target.period: target for target in month_targets(config)}
    before = parse_source_row(
        row_at(datetime(2024, 12, 31, 23, tzinfo=UTC)),
        targets["2024-12"],
        1,
        config.coverage.raw_end_exclusive,
    )
    after = parse_source_row(
        row_at(JANUARY_2025, unit="microseconds"),
        targets["2025-01"],
        1,
        config.coverage.raw_end_exclusive,
    )
    assert before.source_time_unit == "milliseconds"
    assert after.source_time_unit == "microseconds"
    assert before.close_time.microsecond == 999_000
    assert after.close_time.microsecond == 999_999
    assert detect_timestamp_unit(before.open_time_raw) == "milliseconds"
    assert detect_timestamp_unit(after.open_time_raw) == "microseconds"


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        ({"close_time": "1733014799999999"}, "mixes timestamp units"),
        ({"open_time": "1733011200000000", "close_time": "1733014799999999"}, "uses microseconds"),
        ({"open_time": "1733011200001", "close_time": "1733014799999"}, "not aligned"),
        ({"close_time": "1733014799998"}, "does not equal"),
        ({"open_time": "1735689600000", "close_time": "1735693199999"}, "does not belong"),
    ),
)
def test_g1_invalid_timestamp_relations_fail(config, mutate, message) -> None:
    cfg = one_month_config(config)
    target = month_targets(cfg)[0]
    cells = row_at(DECEMBER_2024)
    for name, value in mutate.items():
        cells[DOCUMENTED_SOURCE_COLUMNS.index(name)] = value
    with pytest.raises(ValidationError, match=message):
        parse_source_row(cells, target, 1, cfg.coverage.raw_end_exclusive)


def test_g1_nonstandard_close_row_is_excluded_as_an_explicit_gap(
    config, temp_acquisition_root
) -> None:
    cfg = one_month_config(config, active_schema=True)
    target = month_targets(cfg)[0]
    records = [
        row_at(DECEMBER_2024),
        row_at(
            DECEMBER_2024 + timedelta(hours=1),
            close_time="1733018398000",
        ),
        row_at(DECEMBER_2024 + timedelta(hours=2)),
    ]
    write_month_archive(temp_acquisition_root, target, records)
    rows, audit, exclusions = parse_month(cfg, target)
    assert [row.open_time for row in rows] == [
        DECEMBER_2024,
        DECEMBER_2024 + timedelta(hours=2),
    ]
    assert audit.nonstandard_close_rows_excluded == 1
    assert exclusions[0].source_row == 2
    assert exclusions[0].exclusion_reason == "nonstandard_close_time_relation"
    _, gaps = assign_segments_and_gaps(
        rows, DECEMBER_2024, DECEMBER_2024 + timedelta(hours=3)
    )
    assert [gap.missing_open_time for gap in gaps] == [
        DECEMBER_2024 + timedelta(hours=1)
    ]


def test_g1_nonstandard_close_still_has_to_belong_to_its_archive_month(
    config, temp_acquisition_root
) -> None:
    cfg = one_month_config(config, active_schema=True)
    target = month_targets(cfg)[0]
    records = [
        row_at(
            DECEMBER_2024,
            close_time="1733011199999",
        )
    ]
    write_month_archive(temp_acquisition_root, target, records)
    with pytest.raises(ValidationError, match="close_time does not belong"):
        parse_month(cfg, target)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("open", "NaN", "finite"),
        ("close", "0", "positive"),
        ("high", "99", "below"),
        ("low", "102", "above"),
        ("volume", "-1", "non-negative"),
        ("quote_asset_volume", "Infinity", "finite"),
        ("number_of_trades", "1.5", "integer-compatible"),
        ("taker_buy_base_asset_volume", "-0.1", "non-negative"),
    ),
)
def test_g1_numeric_invariants_fail_closed(config, field, value, message) -> None:
    cfg = one_month_config(config)
    target = month_targets(cfg)[0]
    with pytest.raises(ValidationError, match=message):
        parse_source_row(
            row_at(DECEMBER_2024, **{field: value}),
            target,
            1,
            cfg.coverage.raw_end_exclusive,
        )


@pytest.mark.parametrize("offsets", ((0, 0), (1, 0)))
def test_g1_duplicate_or_disordered_rows_fail(
    config, temp_acquisition_root, offsets
) -> None:
    cfg = one_month_config(config)
    target = month_targets(cfg)[0]
    records = [
        row_at(DECEMBER_2024 + timedelta(hours=offsets[0])),
        row_at(DECEMBER_2024 + timedelta(hours=offsets[1])),
    ]
    write_month_archive(temp_acquisition_root, target, records)
    with pytest.raises(ValidationError, match="duplicate|disorder"):
        parse_month(cfg, target)


def test_g1_gaps_are_explicit_and_segment_safe(config) -> None:
    cfg = one_month_config(config)
    start = cfg.coverage.raw_start_inclusive
    rows = [parsed(cfg, start), parsed(cfg, start + timedelta(hours=1)), parsed(cfg, start + timedelta(hours=3))]
    annotated, gaps = assign_segments_and_gaps(rows, start, start + timedelta(hours=4))
    assert [row.contiguous_segment_id for row in annotated] == [1, 1, 2]
    assert [row.has_previous_contiguous_hour for row in annotated] == [False, True, False]
    assert annotated[-1].gap_before_hours == 1
    assert [gap.missing_open_time for gap in gaps] == [start + timedelta(hours=2)]
    candidates = build_anomaly_candidates(annotated, cfg)
    by_time = {candidate.open_time: candidate for candidate in candidates}
    assert by_time[start + timedelta(hours=1)].diagnostic_log_return_1h is not None
    assert start + timedelta(hours=3) not in by_time


def test_g1_anomaly_union_is_deterministic_and_keeps_zero_activity(config) -> None:
    cfg = one_month_config(config)
    start = cfg.coverage.raw_start_inclusive
    raw_rows = []
    price = 100
    for hour in range(25):
        next_price = price + hour + 1
        replacements = {
            "open": str(price),
            "high": str(next_price + 1),
            "low": str(price - 1),
            "close": str(next_price),
        }
        if hour == 24:
            replacements.update(
                {
                    "open": str(price + 0.5),
                    "volume": "0",
                    "quote_asset_volume": "0",
                    "number_of_trades": "0",
                }
            )
        raw_rows.append(parsed(cfg, start + timedelta(hours=hour), **replacements))
        price = next_price
    annotated, _ = assign_segments_and_gaps(raw_rows, start, start + timedelta(hours=25))
    first = build_anomaly_candidates(annotated, cfg)
    second = build_anomaly_candidates(annotated, cfg)
    assert first == second
    assert len(first) == 20
    final = next(candidate for candidate in first if candidate.open_time == start + timedelta(hours=24))
    assert "zero_base_volume" in final.flag_types
    assert "zero_quote_volume" in final.flag_types
    assert "zero_trade_count" in final.flag_types
    assert "top_20_absolute_log_open_gap" in final.flag_types
    assert "top_20_absolute_log_return" in final.flag_types


def test_g1_only_twenty_largest_nonzero_open_gaps_are_flagged(config) -> None:
    cfg = one_month_config(config)
    start = cfg.coverage.raw_start_inclusive
    raw_rows = []
    for hour in range(26):
        price = 100 + hour
        raw_rows.append(
            parsed(
                cfg,
                start + timedelta(hours=hour),
                open=str(price),
                high=str(price),
                low=str(price),
                close=str(price),
            )
        )
    annotated, _ = assign_segments_and_gaps(
        raw_rows, start, start + timedelta(hours=26)
    )
    candidates = build_anomaly_candidates(annotated, cfg)
    open_gap_candidates = [
        candidate
        for candidate in candidates
        if "top_20_absolute_log_open_gap" in candidate.flag_types
    ]
    assert len(open_gap_candidates) == 20
    assert all(candidate.diagnostic_log_open_gap is not None for candidate in open_gap_candidates)


def test_g1_review_ledger_must_match_every_candidate(config) -> None:
    cfg = one_month_config(config)
    row = replace_for_segment(parsed(cfg, DECEMBER_2024))
    candidate = AnomalyCandidate(
        "dq-test", row.open_time, row.source_file, row.source_row,
        ("zero_base_volume",), None, None, "", row.open_text,
    )
    content = render_anomaly_candidates([candidate])
    digest = hashlib.sha256(content.encode()).hexdigest()
    decisions = {"dq-test": ReviewDecision("dq-test", "retain", "Source row verified.")}
    validate_review_ledger([candidate], content, digest, decisions)
    with pytest.raises(ValidationError, match="different candidate table"):
        validate_review_ledger([candidate], content, "0" * 64, decisions)
    with pytest.raises(ValidationError, match="blocks cleaning"):
        validate_review_ledger(
            [candidate], content, digest,
            {"dq-test": ReviewDecision("dq-test", "block_gate", "Needs resolution.")},
        )


def replace_for_segment(row):
    from dataclasses import replace

    return replace(row, contiguous_segment_id=1)


def test_g1_clean_output_contains_no_analytical_or_forward_fields(config) -> None:
    cfg = one_month_config(config)
    row = replace_for_segment(parsed(cfg, DECEMBER_2024))
    content = render_clean_rows([row], [], {})
    header = content.splitlines()[0].split(",")
    assert tuple(header) == CLEAN_COLUMNS
    forbidden = ("shock", "candidate", "event", "forward", "post_", "pnl", "return_12h")
    assert not any(any(token in column for token in forbidden) for column in header)


def test_g1_raw_readme_is_invariant_when_schema_is_activated(config) -> None:
    inactive = one_month_config(config)
    active = one_month_config(config, active_schema=True)
    assert acquisition.render_raw_readme(inactive) == acquisition.render_raw_readme(active)


def full_month_rows() -> list[list[str]]:
    rows = []
    cursor = DECEMBER_2024
    while cursor < JANUARY_2025:
        rows.append(row_at(cursor, open="100", high="100", low="100", close="100"))
        cursor += timedelta(hours=1)
    return rows


def acquire_synthetic_month(config, records=None):
    target = month_targets(config)[0]
    archive = zip_bytes(
        target.expected_csv_member,
        full_month_rows() if records is None else records,
    )
    checksum = f"{hashlib.sha256(archive).hexdigest()}  {target.zip_filename}\n".encode()
    payloads = {target.zip_url: archive, target.checksum_url: checksum}
    return acquisition.run_acquisition(config, fetch=payloads.__getitem__)


def test_g1_run_manifests_nonstandard_close_exclusion(
    config, temp_acquisition_root
) -> None:
    cfg = one_month_config(config, active_schema=True)
    records = full_month_rows()
    open_raw = int(records[10][0])
    records[10][6] = str(open_raw + 1_000)
    acquired = acquire_synthetic_month(cfg, records)
    result = validation.run_validation(
        cfg, acquisition_manifest_relative=acquired.manifest_relative_path
    )
    assert result.status == "review_required"
    assert result.row_count == 743
    assert result.excluded_row_count == 1
    assert result.gap_count == 1
    run_dir = temp_acquisition_root / result.manifest_relative_path.parent
    excluded = list(
        csv.DictReader(
            io.StringIO((run_dir / "excluded_rows.csv").read_text(encoding="utf-8"))
        )
    )
    assert len(excluded) == 1
    assert excluded[0]["source_row"] == "11"
    assert excluded[0]["exclusion_reason"] == "nonstandard_close_time_relation"
    manifest = read_manifest(result.manifest_relative_path)
    assert manifest["exclusions"]["nonstandard_close_time_rows"] == 1


def test_g1_preflight_and_reviewed_final_run_are_reproducible_offline(
    config, temp_acquisition_root
) -> None:
    inactive = one_month_config(config)
    acquired = acquire_synthetic_month(inactive)
    preflight = validation.run_validation(
        inactive, acquisition_manifest_relative=acquired.manifest_relative_path
    )
    assert preflight.status == "review_required"
    assert preflight.clean_relative_path is None
    manifest = read_manifest(preflight.manifest_relative_path)
    assert manifest["row_count"] == 20
    reconciliation_path = (
        preflight.manifest_relative_path.parent / "file_reconciliation.csv"
    )
    reconciliation = list(
        csv.DictReader(
            io.StringIO(
                (temp_acquisition_root / reconciliation_path).read_text(encoding="utf-8")
            )
        )
    )
    assert len(reconciliation) == 1
    assert reconciliation[0]["accepted_complete_rows"] == "744"
    assert reconciliation[0]["duplicate_rows"] == "0"
    assert reconciliation[0]["invalid_rows"] == "0"
    assert reconciliation[0]["excluded_rows_total"] == "0"
    assert reconciliation[0]["missing_hours"] == "0"
    candidate_path = preflight.manifest_relative_path.parent / "anomaly_candidates.csv"
    candidate_bytes = (temp_acquisition_root / candidate_path).read_bytes()
    candidates = list(csv.DictReader(io.StringIO(candidate_bytes.decode())))
    review = {
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "reviews": [
            {
                "candidate_id": candidate["candidate_id"],
                "disposition": "retain",
                "reason": "Synthetic source row verified.",
            }
            for candidate in candidates
        ],
    }
    review_path = temp_acquisition_root / validation.REVIEW_RELATIVE_PATH
    review_path.write_text(yaml.safe_dump(review, sort_keys=True), encoding="utf-8")
    active = one_month_config(config, active_schema=True)
    final = validation.run_validation(
        active, acquisition_manifest_relative=acquired.manifest_relative_path
    )
    assert final.status == "complete"
    assert final.row_count == 744
    assert final.gap_count == 0
    assert final.clean_relative_path is not None
    clean = (temp_acquisition_root / final.clean_relative_path).read_text(encoding="utf-8")
    assert len(clean.splitlines()) == 745
    assert read_manifest(final.manifest_relative_path)["output_checksum"] == hashlib.sha256(
        clean.encode()
    ).hexdigest()
    assert len(read_manifest(final.manifest_relative_path)["input_checksums"]) == 6
    acquisition.verify_manifest_checksum(final.manifest_relative_path)

    repeated = validation.run_validation(
        active, acquisition_manifest_relative=acquired.manifest_relative_path
    )
    repeated_clean = (temp_acquisition_root / repeated.clean_relative_path).read_text(
        encoding="utf-8"
    )
    assert repeated_clean == clean


def test_g1_tampered_raw_input_is_refused_before_parsing(config, temp_acquisition_root) -> None:
    cfg = one_month_config(config)
    acquired = acquire_synthetic_month(cfg)
    target = month_targets(cfg)[0]
    (temp_acquisition_root / target.zip_relative_path).write_bytes(b"tampered")
    with pytest.raises(ValidationError, match="raw input checksum mismatch"):
        verify_acquisition_inputs(cfg, acquired.manifest_relative_path)


def test_validate_pipeline_dispatch_is_thin(monkeypatch, capsys) -> None:
    result = ValidationResult(
        run_id="run",
        status="review_required",
        manifest_relative_path=Path("outputs/runs/run/validation-preflight.manifest.json"),
        manifest_checksum_relative_path=Path(
            "outputs/runs/run/validation-preflight.manifest.json.sha256"
        ),
        log_relative_path=Path("outputs/runs/run/run.log.jsonl"),
        clean_relative_path=None,
        row_count=10,
        excluded_row_count=1,
        gap_count=1,
        anomaly_count=2,
    )
    monkeypatch.setattr(pipeline, "load_config", lambda: object())
    monkeypatch.setattr(validation, "run_validation", lambda config: result)
    assert pipeline.dispatch("validate") == 0
    printed = capsys.readouterr().out
    assert "review_required" in printed
    assert "pending schema/manual review" in printed


_IMPORT_PROGRAM = """
import sys
opened = []
network = []
def hook(event, args):
    if event == "open": opened.append(str(args[0]))
    elif event.startswith(("socket.", "urllib.")): network.append(event)
sys.addaudithook(hook)
import src.validation
offenders = sorted(
    {path for path in opened if path.endswith(("project.yaml", ".zip", ".csv"))}
    | set(network)
)
print("OFFENDERS " + repr(offenders) if offenders else "CLEAN")
"""


def test_validation_import_reads_no_project_data_and_opens_no_socket(project_root) -> None:
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
