"""Parse, check, and clean the hourly Binance source data.

Checks the acquisition run before opening any CSV, parses the 12-field kline
rows, validates timestamps and prices, and records gaps without filling them.
No features, events, outcomes, or trades are built here.

Runs in two passes. The first records the header tokens and the anomaly
candidates to review by hand. The clean file is written only once the column
names are set in the configuration and every candidate has a review entry.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from src import paths
from src.acquisition import (
    MonthTarget,
    month_targets,
    verify_manifest_checksum,
    write_manifest_checksum,
)
from src.artifacts import (
    CREATED_AT_FORMAT,
    RunLog,
    build_manifest,
    create_exclusive,
    create_run_log,
    read_manifest,
    run_relative_dir,
    sha256_bytes,
    sha256_file,
)
from src.config import ConfigNode, load_config

STAGE = "3. Validation and cleaning"
MODE = "validate"
REVIEW_RELATIVE_PATH = Path("config/anomaly_reviews.yaml")

# Meaning of each column position, used even when an archive has no header.
# The actual header tokens are observed in the first pass and set in
# config/project.yaml, not guessed here.
DOCUMENTED_SOURCE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
)

CLEAN_COLUMNS: tuple[str, ...] = (
    "open_time_raw",
    "close_time_raw",
    "source_time_unit",
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
    "source_period",
    "source_file",
    "source_member",
    "source_row",
    "is_complete_candle",
    "is_duplicate",
    "has_previous_contiguous_hour",
    "gap_before_hours",
    "contiguous_segment_id",
    "ohlc_valid",
    "volume_valid",
    "anomaly_flag",
    "anomaly_reviewed",
    "anomaly_disposition",
    "exclusion_reason",
)

SCHEMA_COLUMNS: tuple[str, ...] = (
    "period",
    "source_member",
    "header_present",
    "header_tokens",
    "data_rows",
)

RECONCILIATION_COLUMNS: tuple[str, ...] = (
    "period",
    "source_member",
    "physical_records",
    "header_rows",
    "parsed_data_rows",
    "accepted_complete_rows",
    "incomplete_rows_excluded",
    "nonstandard_close_rows_excluded",
    "duplicate_rows",
    "invalid_rows",
    "excluded_rows_total",
    "expected_hours",
    "missing_hours",
)

EXCLUSION_COLUMNS: tuple[str, ...] = (
    "period",
    "source_file",
    "source_member",
    "source_row",
    "open_time_raw",
    "close_time_raw",
    "source_time_unit",
    "open_time",
    "close_time",
    "exclusion_reason",
)

GAP_COLUMNS: tuple[str, ...] = (
    "missing_open_time",
    "source_period_expected",
    "previous_observed_open_time",
    "next_observed_open_time",
)

ANOMALY_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "open_time",
    "source_file",
    "source_row",
    "flag_types",
    "diagnostic_log_return_1h",
    "diagnostic_log_open_gap",
    "previous_close",
    "current_open",
)

REVIEW_COLUMNS: tuple[str, ...] = ANOMALY_COLUMNS + (
    "anomaly_reviewed",
    "anomaly_disposition",
    "review_reason",
)

_INTEGER_RE = re.compile(r"^[0-9]+$")
_ALLOWED_REVIEW_DISPOSITIONS = frozenset({"retain", "block_gate"})
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class ValidationError(RuntimeError):
    """Raised when the source data fails a validation check."""


@dataclass(frozen=True)
class ParsedCandle:
    """One valid source row, before gap and review fields are added."""

    open_time_raw: int
    close_time_raw: int
    source_time_unit: str
    open_time: datetime
    close_time: datetime
    open_text: str
    high_text: str
    low_text: str
    close_text: str
    base_volume_text: str
    quote_volume_text: str
    trade_count_text: str
    taker_buy_base_volume_text: str
    taker_buy_quote_volume_text: str
    ignore_text: str
    open_value: Decimal
    high_value: Decimal
    low_value: Decimal
    close_value: Decimal
    base_volume_value: Decimal
    quote_volume_value: Decimal
    trade_count_value: Decimal
    taker_buy_base_volume_value: Decimal
    taker_buy_quote_volume_value: Decimal
    source_period: str
    source_file: str
    source_member: str
    source_row: int
    is_complete_candle: bool
    has_previous_contiguous_hour: bool = False
    gap_before_hours: int | None = None
    contiguous_segment_id: int = 0


@dataclass(frozen=True)
class FileAudit:
    """Header and row counts for one monthly file."""

    target: MonthTarget
    header_present: bool
    header_tokens: tuple[str, ...]
    physical_records: int
    parsed_data_rows: int
    accepted_complete_rows: int
    incomplete_rows_excluded: int
    nonstandard_close_rows_excluded: int
    expected_hours: int
    duplicate_rows: int = 0
    invalid_rows: int = 0
    missing_hours: int = 0


@dataclass(frozen=True)
class GapRecord:
    missing_open_time: datetime
    source_period_expected: str
    previous_observed_open_time: datetime | None
    next_observed_open_time: datetime | None


@dataclass(frozen=True)
class ExcludedRow:
    """One source row excluded because its close time is nonstandard."""

    source_period: str
    source_file: str
    source_member: str
    source_row: int
    open_time_raw: int
    close_time_raw: int
    source_time_unit: str
    open_time: datetime
    close_time: datetime
    exclusion_reason: str


@dataclass(frozen=True)
class AnomalyCandidate:
    candidate_id: str
    open_time: datetime
    source_file: str
    source_row: int
    flag_types: tuple[str, ...]
    diagnostic_log_return_1h: float | None
    diagnostic_log_open_gap: float | None
    previous_close: str
    current_open: str


@dataclass(frozen=True)
class ReviewDecision:
    candidate_id: str
    disposition: str
    reason: str


@dataclass(frozen=True)
class ValidationResult:
    run_id: str
    status: str
    manifest_relative_path: Path
    manifest_checksum_relative_path: Path
    log_relative_path: Path
    clean_relative_path: Path | None
    row_count: int
    excluded_row_count: int
    gap_count: int
    anomaly_count: int


def _period_bounds(target: MonthTarget) -> tuple[datetime, datetime]:
    start = datetime(target.year, target.month, 1, tzinfo=timezone.utc)
    if target.month == 12:
        end = datetime(target.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(target.year, target.month + 1, 1, tzinfo=timezone.utc)
    return start, end


def _expected_hours(target: MonthTarget) -> int:
    start, end = _period_bounds(target)
    return int((end - start).total_seconds() // 3600)


def _format_utc(moment: datetime, unit: str | None = None) -> str:
    utc = moment.astimezone(timezone.utc)
    if unit == "milliseconds":
        return utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if unit == "microseconds":
        return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return utc.strftime(CREATED_AT_FORMAT)


def _month_label(moment: datetime) -> str:
    return f"{moment.year:04d}-{moment.month:02d}"


def _strict_integer(text: str, field: str, target: MonthTarget, source_row: int) -> int:
    if text != text.strip() or _INTEGER_RE.fullmatch(text) is None:
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} field {field} must be an "
            f"unsigned base-10 integer, got {text!r}"
        )
    return int(text)


def _finite_decimal(text: str, field: str, target: MonthTarget, source_row: int) -> Decimal:
    if text != text.strip() or not text:
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} field {field} is blank or padded"
        )
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} field {field} is not numeric: "
            f"{text!r}"
        ) from None
    if not value.is_finite():
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} field {field} must be finite, "
            f"got {text!r}"
        )
    return value


def detect_timestamp_unit(raw_value: int) -> str:
    """Tell milliseconds from microseconds by the number of digits."""
    digits = len(str(raw_value))
    if digits == 13:
        return "milliseconds"
    if digits == 16:
        return "microseconds"
    raise ValidationError(
        f"timestamp integer {raw_value!r} has {digits} digits; expected 13-digit "
        "milliseconds or 16-digit microseconds"
    )


def convert_timestamp(raw_value: int, unit: str) -> datetime:
    """Convert an epoch integer to UTC using integer microseconds, with no rounding."""
    if unit == "milliseconds":
        microseconds = raw_value * 1_000
    elif unit == "microseconds":
        microseconds = raw_value
    else:
        raise ValidationError(f"unsupported timestamp unit {unit!r}")
    try:
        return _EPOCH + timedelta(microseconds=microseconds)
    except OverflowError:
        raise ValidationError(f"timestamp integer {raw_value!r} is outside datetime range") from None


def _parse_timestamp_pair(
    open_text: str,
    close_text: str,
    target: MonthTarget,
    source_row: int,
    *,
    require_standard_close: bool = True,
) -> tuple[int, int, str, datetime, datetime]:
    open_raw = _strict_integer(open_text, "open_time", target, source_row)
    close_raw = _strict_integer(close_text, "close_time", target, source_row)
    open_unit = detect_timestamp_unit(open_raw)
    close_unit = detect_timestamp_unit(close_raw)
    if open_unit != close_unit:
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} mixes timestamp units: "
            f"open is {open_unit}, close is {close_unit}"
        )
    if open_unit != target.expected_timestamp_unit:
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} uses {open_unit}, expected "
            f"{target.expected_timestamp_unit} for {target.period}"
        )
    quantum_per_hour = 3_600_000 if open_unit == "milliseconds" else 3_600_000_000
    if open_raw % quantum_per_hour:
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} open_time is not aligned to a "
            "whole UTC hour"
        )
    expected_close = open_raw + quantum_per_hour - 1
    if require_standard_close and close_raw != expected_close:
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} close_time {close_raw} does not "
            f"equal open_time + 1 hour - 1 source quantum ({expected_close})"
        )
    open_time = convert_timestamp(open_raw, open_unit)
    close_time = convert_timestamp(close_raw, open_unit)
    period_start, period_end = _period_bounds(target)
    if not (period_start <= open_time < period_end):
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} open_time does not belong to "
            f"archive period {target.period}"
        )
    if not (period_start <= close_time < period_end):
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} close_time does not belong to "
            f"archive period {target.period}"
        )
    return open_raw, close_raw, open_unit, open_time, close_time


def parse_source_row(
    cells: Sequence[str],
    target: MonthTarget,
    source_row: int,
    raw_end_exclusive: datetime,
    *,
    require_standard_close: bool = True,
) -> ParsedCandle:
    """Parse and check one 12-field source row."""
    if len(cells) != len(DOCUMENTED_SOURCE_COLUMNS):
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} has {len(cells)} columns; "
            f"expected {len(DOCUMENTED_SOURCE_COLUMNS)}"
        )
    open_raw, close_raw, unit, open_time, close_time = _parse_timestamp_pair(
        cells[0],
        cells[6],
        target,
        source_row,
        require_standard_close=require_standard_close,
    )
    named = dict(zip(DOCUMENTED_SOURCE_COLUMNS, cells))
    prices = {
        name: _finite_decimal(named[name], name, target, source_row)
        for name in ("open", "high", "low", "close")
    }
    for name, value in prices.items():
        if value <= 0:
            raise ValidationError(
                f"{target.expected_csv_member} row {source_row} field {name} must be positive"
            )
    volumes = {
        name: _finite_decimal(named[name], name, target, source_row)
        for name in (
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        )
    }
    for name, value in volumes.items():
        if value < 0:
            raise ValidationError(
                f"{target.expected_csv_member} row {source_row} field {name} must be non-negative"
            )
    if volumes["number_of_trades"] != volumes["number_of_trades"].to_integral_value():
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} number_of_trades must be "
            "integer-compatible"
        )
    if prices["high"] < max(prices["open"], prices["close"]):
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} high is below open or close"
        )
    if prices["low"] > min(prices["open"], prices["close"]):
        raise ValidationError(
            f"{target.expected_csv_member} row {source_row} low is above open or close"
        )
    return ParsedCandle(
        open_time_raw=open_raw,
        close_time_raw=close_raw,
        source_time_unit=unit,
        open_time=open_time,
        close_time=close_time,
        open_text=named["open"],
        high_text=named["high"],
        low_text=named["low"],
        close_text=named["close"],
        base_volume_text=named["volume"],
        quote_volume_text=named["quote_asset_volume"],
        trade_count_text=named["number_of_trades"],
        taker_buy_base_volume_text=named["taker_buy_base_asset_volume"],
        taker_buy_quote_volume_text=named["taker_buy_quote_asset_volume"],
        ignore_text=named["ignore"],
        open_value=prices["open"],
        high_value=prices["high"],
        low_value=prices["low"],
        close_value=prices["close"],
        base_volume_value=volumes["volume"],
        quote_volume_value=volumes["quote_asset_volume"],
        trade_count_value=volumes["number_of_trades"],
        taker_buy_base_volume_value=volumes["taker_buy_base_asset_volume"],
        taker_buy_quote_volume_value=volumes["taker_buy_quote_asset_volume"],
        source_period=target.period,
        source_file=target.zip_filename,
        source_member=target.expected_csv_member,
        source_row=source_row,
        is_complete_candle=close_time < raw_end_exclusive,
    )


def _looks_like_data_row(cells: Sequence[str]) -> bool:
    return bool(cells) and _INTEGER_RE.fullmatch(cells[0]) is not None


def parse_month(
    config: ConfigNode,
    target: MonthTarget,
) -> tuple[list[ParsedCandle], FileAudit, list[ExcludedRow]]:
    """Read and check one monthly CSV straight from its ZIP, without extracting it."""
    if config.source.schema_verified:
        configured_names = config.source.expected_column_names
        if configured_names is None or tuple(configured_names) != DOCUMENTED_SOURCE_COLUMNS:
            raise ValidationError(
                "source.expected_column_names does not match the documented 12-field "
                "Binance kline order"
            )
    archive_path = paths.require_file(paths.resolve(target.zip_relative_path))
    rows: list[ParsedCandle] = []
    exclusions: list[ExcludedRow] = []
    header_present = False
    header_tokens: tuple[str, ...] = ()
    physical_records = 0
    incomplete = 0
    previous_open: datetime | None = None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != target.expected_csv_member:
                raise ValidationError(
                    f"{target.zip_filename} no longer contains exactly the expected member "
                    f"{target.expected_csv_member!r}"
                )
            with archive.open(infos[0], "r") as binary, io.TextIOWrapper(
                binary, encoding="utf-8", newline=""
            ) as text:
                reader = csv.reader(text, strict=True)
                for record_number, cells in enumerate(reader, start=1):
                    physical_records += 1
                    source_row = reader.line_num
                    if not cells or all(cell == "" for cell in cells):
                        raise ValidationError(
                            f"{target.expected_csv_member} row {source_row} is blank"
                        )
                    if record_number == 1 and not _looks_like_data_row(cells):
                        if len(cells) != len(DOCUMENTED_SOURCE_COLUMNS):
                            raise ValidationError(
                                f"{target.expected_csv_member} header has {len(cells)} columns; "
                                f"expected {len(DOCUMENTED_SOURCE_COLUMNS)}"
                            )
                        header_present = True
                        header_tokens = tuple(cells)
                        if config.source.schema_verified:
                            expected = tuple(config.source.expected_column_names)
                            if header_tokens != expected:
                                raise ValidationError(
                                    f"{target.expected_csv_member} header does not exactly match "
                                    "source.expected_column_names"
                                )
                        continue
                    if not _looks_like_data_row(cells):
                        raise ValidationError(
                            f"{target.expected_csv_member} row {source_row} is an unexpected "
                            "header or has a non-integer open_time"
                        )
                    candle = parse_source_row(
                        cells,
                        target,
                        source_row,
                        config.coverage.raw_end_exclusive,
                        require_standard_close=False,
                    )
                    if previous_open is not None and candle.open_time <= previous_open:
                        relation = "duplicate" if candle.open_time == previous_open else "disorder"
                        raise ValidationError(
                            f"{target.expected_csv_member} row {source_row} creates timestamp "
                            f"{relation} within the file"
                        )
                    previous_open = candle.open_time
                    quantum_per_hour = (
                        3_600_000
                        if candle.source_time_unit == "milliseconds"
                        else 3_600_000_000
                    )
                    expected_close = candle.open_time_raw + quantum_per_hour - 1
                    if candle.close_time_raw != expected_close:
                        exclusions.append(
                            ExcludedRow(
                                source_period=candle.source_period,
                                source_file=candle.source_file,
                                source_member=candle.source_member,
                                source_row=candle.source_row,
                                open_time_raw=candle.open_time_raw,
                                close_time_raw=candle.close_time_raw,
                                source_time_unit=candle.source_time_unit,
                                open_time=candle.open_time,
                                close_time=candle.close_time,
                                exclusion_reason="nonstandard_close_time_relation",
                            )
                        )
                        continue
                    if candle.is_complete_candle:
                        rows.append(candle)
                    else:
                        incomplete += 1
    except (UnicodeDecodeError, csv.Error, zipfile.BadZipFile) as error:
        raise ValidationError(
            f"could not parse {target.expected_csv_member}: {type(error).__name__}: {error}"
        ) from None
    if not rows and incomplete == 0 and not exclusions:
        raise ValidationError(f"{target.expected_csv_member} contains no data rows")
    audit = FileAudit(
        target=target,
        header_present=header_present,
        header_tokens=header_tokens,
        physical_records=physical_records,
        parsed_data_rows=len(rows) + incomplete + len(exclusions),
        accepted_complete_rows=len(rows),
        incomplete_rows_excluded=incomplete,
        nonstandard_close_rows_excluded=len(exclusions),
        expected_hours=_expected_hours(target),
    )
    if audit.physical_records != int(audit.header_present) + audit.parsed_data_rows:
        raise ValidationError(f"row reconciliation failed for {target.expected_csv_member}")
    return rows, audit, exclusions


def assign_segments_and_gaps(
    rows: Sequence[ParsedCandle],
    raw_start_inclusive: datetime,
    raw_end_exclusive: datetime,
) -> tuple[list[ParsedCandle], list[GapRecord]]:
    """Check row order, number the gap-free segments, and list every missing hour."""
    if not rows:
        raise ValidationError("the source snapshot contains no accepted rows")
    annotated: list[ParsedCandle] = []
    observed: dict[datetime, ParsedCandle] = {}
    previous: ParsedCandle | None = None
    segment = 1
    for row in rows:
        if not (raw_start_inclusive <= row.open_time < raw_end_exclusive):
            raise ValidationError(
                f"observed open_time {_format_utc(row.open_time)} lies outside raw coverage"
            )
        if row.open_time in observed:
            raise ValidationError(f"duplicate global open_time {_format_utc(row.open_time)}")
        if previous is not None and row.open_time <= previous.open_time:
            raise ValidationError(
                f"global source rows are not strictly chronological at "
                f"{_format_utc(row.open_time)}"
            )
        contiguous = previous is not None and row.open_time - previous.open_time == timedelta(hours=1)
        gap_before = None
        if previous is not None and not contiguous:
            delta_hours = int((row.open_time - previous.open_time).total_seconds() // 3600)
            if delta_hours < 1 or previous.open_time + timedelta(hours=delta_hours) != row.open_time:
                raise ValidationError("observed timestamps do not lie on the hourly UTC grid")
            gap_before = delta_hours - 1
            segment += 1
        annotated_row = replace(
            row,
            has_previous_contiguous_hour=contiguous,
            gap_before_hours=gap_before,
            contiguous_segment_id=segment,
        )
        annotated.append(annotated_row)
        observed[row.open_time] = annotated_row
        previous = row

    gaps: list[GapRecord] = []
    observed_times = list(observed)
    observed_index = 0
    cursor = raw_start_inclusive
    previous_observed: datetime | None = None
    while cursor < raw_end_exclusive:
        if observed_index < len(observed_times) and observed_times[observed_index] == cursor:
            previous_observed = cursor
            observed_index += 1
        else:
            next_observed = (
                observed_times[observed_index]
                if observed_index < len(observed_times)
                else None
            )
            gaps.append(
                GapRecord(
                    missing_open_time=cursor,
                    source_period_expected=_month_label(cursor),
                    previous_observed_open_time=previous_observed,
                    next_observed_open_time=next_observed,
                )
            )
        cursor += timedelta(hours=1)
    if len(annotated) + len(gaps) != int(
        (raw_end_exclusive - raw_start_inclusive).total_seconds() // 3600
    ):
        raise ValidationError("global expected-hour reconciliation failed")
    return annotated, gaps


def _candidate_id(row: ParsedCandle) -> str:
    payload = f"{row.source_file}|{row.source_row}|{row.open_time_raw}".encode("utf-8")
    return f"dq-{hashlib.sha256(payload).hexdigest()[:16]}"


def build_anomaly_candidates(
    rows: Sequence[ParsedCandle], config: ConfigNode
) -> list[AnomalyCandidate]:
    """List rows to review by hand: zero-activity hours and the largest moves and open gaps."""
    returns: dict[int, float] = {}
    open_gaps: dict[int, float] = {}
    flags: dict[int, set[str]] = {index: set() for index in range(len(rows))}
    for index, row in enumerate(rows):
        if config.data_quality.review_zero_base_volume and row.base_volume_value == 0:
            flags[index].add("zero_base_volume")
        if config.data_quality.review_zero_quote_volume and row.quote_volume_value == 0:
            flags[index].add("zero_quote_volume")
        if config.data_quality.review_zero_trade_count and row.trade_count_value == 0:
            flags[index].add("zero_trade_count")
        if index == 0 or not row.has_previous_contiguous_hour:
            continue
        previous = rows[index - 1]
        # Subtracting logs avoids overflow from a large price ratio.
        value = float(row.close_value.ln() - previous.close_value.ln())
        if not math.isfinite(value):
            raise ValidationError(
                f"non-finite diagnostic return at {_format_utc(row.open_time)}"
            )
        returns[index] = value
        if row.open_value != previous.close_value:
            gap = float(row.open_value.ln() - previous.close_value.ln())
            if not math.isfinite(gap):
                raise ValidationError(
                    f"non-finite diagnostic open gap at {_format_utc(row.open_time)}"
                )
            open_gaps[index] = gap
    ranked_returns = sorted(
        returns,
        key=lambda index: (-abs(returns[index]), rows[index].open_time, rows[index].source_row),
    )[: config.data_quality.largest_return_audit_count]
    for index in ranked_returns:
        flags[index].add("top_20_absolute_log_return")
    ranked_open_gaps = sorted(
        open_gaps,
        key=lambda index: (
            -abs(open_gaps[index]),
            rows[index].open_time,
            rows[index].source_row,
        ),
    )[: config.data_quality.largest_open_gap_audit_count]
    for index in ranked_open_gaps:
        flags[index].add("top_20_absolute_log_open_gap")

    candidates: list[AnomalyCandidate] = []
    for index, row in enumerate(rows):
        if not flags[index]:
            continue
        previous = rows[index - 1] if index > 0 and row.has_previous_contiguous_hour else None
        candidates.append(
            AnomalyCandidate(
                candidate_id=_candidate_id(row),
                open_time=row.open_time,
                source_file=row.source_file,
                source_row=row.source_row,
                flag_types=tuple(sorted(flags[index])),
                diagnostic_log_return_1h=returns.get(index),
                diagnostic_log_open_gap=open_gaps.get(index),
                previous_close=previous.close_text if previous is not None else "",
                current_open=row.open_text,
            )
        )
    return candidates


def _render_csv(columns: Sequence[str], records: Iterable[Mapping[str, object]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(record)
    return buffer.getvalue()


def render_schema_observations(audits: Sequence[FileAudit]) -> str:
    return _render_csv(
        SCHEMA_COLUMNS,
        (
            {
                "period": audit.target.period,
                "source_member": audit.target.expected_csv_member,
                "header_present": str(audit.header_present).lower(),
                "header_tokens": "|".join(audit.header_tokens),
                "data_rows": audit.parsed_data_rows,
            }
            for audit in audits
        ),
    )


def render_reconciliation(audits: Sequence[FileAudit]) -> str:
    return _render_csv(
        RECONCILIATION_COLUMNS,
        (
            {
                "period": audit.target.period,
                "source_member": audit.target.expected_csv_member,
                "physical_records": audit.physical_records,
                "header_rows": int(audit.header_present),
                "parsed_data_rows": audit.parsed_data_rows,
                "accepted_complete_rows": audit.accepted_complete_rows,
                "incomplete_rows_excluded": audit.incomplete_rows_excluded,
                "nonstandard_close_rows_excluded": (
                    audit.nonstandard_close_rows_excluded
                ),
                "duplicate_rows": audit.duplicate_rows,
                "invalid_rows": audit.invalid_rows,
                "excluded_rows_total": (
                    audit.incomplete_rows_excluded
                    + audit.nonstandard_close_rows_excluded
                    + audit.duplicate_rows
                    + audit.invalid_rows
                ),
                "expected_hours": audit.expected_hours,
                "missing_hours": audit.missing_hours,
            }
            for audit in audits
        ),
    )


def render_exclusions(exclusions: Sequence[ExcludedRow]) -> str:
    return _render_csv(
        EXCLUSION_COLUMNS,
        (
            {
                "period": row.source_period,
                "source_file": row.source_file,
                "source_member": row.source_member,
                "source_row": row.source_row,
                "open_time_raw": row.open_time_raw,
                "close_time_raw": row.close_time_raw,
                "source_time_unit": row.source_time_unit,
                "open_time": _format_utc(row.open_time, row.source_time_unit),
                "close_time": _format_utc(row.close_time, row.source_time_unit),
                "exclusion_reason": row.exclusion_reason,
            }
            for row in exclusions
        ),
    )


def render_gaps(gaps: Sequence[GapRecord]) -> str:
    return _render_csv(
        GAP_COLUMNS,
        (
            {
                "missing_open_time": _format_utc(gap.missing_open_time),
                "source_period_expected": gap.source_period_expected,
                "previous_observed_open_time": (
                    _format_utc(gap.previous_observed_open_time)
                    if gap.previous_observed_open_time is not None
                    else ""
                ),
                "next_observed_open_time": (
                    _format_utc(gap.next_observed_open_time)
                    if gap.next_observed_open_time is not None
                    else ""
                ),
            }
            for gap in gaps
        ),
    )


def _anomaly_row(candidate: AnomalyCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "open_time": _format_utc(candidate.open_time),
        "source_file": candidate.source_file,
        "source_row": candidate.source_row,
        "flag_types": "|".join(candidate.flag_types),
        "diagnostic_log_return_1h": (
            format(candidate.diagnostic_log_return_1h, ".17g")
            if candidate.diagnostic_log_return_1h is not None
            else ""
        ),
        "diagnostic_log_open_gap": (
            format(candidate.diagnostic_log_open_gap, ".17g")
            if candidate.diagnostic_log_open_gap is not None
            else ""
        ),
        "previous_close": candidate.previous_close,
        "current_open": candidate.current_open,
    }


def render_anomaly_candidates(candidates: Sequence[AnomalyCandidate]) -> str:
    return _render_csv(ANOMALY_COLUMNS, (_anomaly_row(item) for item in candidates))


def load_review_ledger(relative_path: Path = REVIEW_RELATIVE_PATH) -> tuple[str, dict[str, ReviewDecision]]:
    """Load the hand-written anomaly review file, rejecting any unexpected layout."""
    target = paths.require_file(paths.resolve(relative_path))
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValidationError(f"could not parse anomaly review ledger: {error}") from None
    if not isinstance(payload, dict) or set(payload) != {"candidate_sha256", "reviews"}:
        raise ValidationError(
            "anomaly review ledger must contain exactly candidate_sha256 and reviews"
        )
    digest = payload["candidate_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValidationError("anomaly review ledger candidate_sha256 is invalid")
    reviews = payload["reviews"]
    if not isinstance(reviews, list):
        raise ValidationError("anomaly review ledger reviews must be a list")
    decisions: dict[str, ReviewDecision] = {}
    for index, item in enumerate(reviews):
        if not isinstance(item, dict) or set(item) != {"candidate_id", "disposition", "reason"}:
            raise ValidationError(f"anomaly review entry {index} has the wrong fields")
        candidate_id = item["candidate_id"]
        disposition = item["disposition"]
        reason = item["reason"]
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValidationError(f"anomaly review entry {index} has an invalid candidate_id")
        if candidate_id in decisions:
            raise ValidationError(f"anomaly review repeats candidate {candidate_id!r}")
        if disposition not in _ALLOWED_REVIEW_DISPOSITIONS:
            raise ValidationError(
                f"anomaly review for {candidate_id!r} has invalid disposition {disposition!r}"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValidationError(f"anomaly review for {candidate_id!r} needs a reason")
        decisions[candidate_id] = ReviewDecision(candidate_id, disposition, reason.strip())
    return digest, decisions


def validate_review_ledger(
    candidates: Sequence[AnomalyCandidate],
    candidate_content: str,
    recorded_digest: str,
    decisions: Mapping[str, ReviewDecision],
) -> None:
    actual_digest = sha256_bytes(candidate_content.encode("utf-8"))
    if recorded_digest != actual_digest:
        raise ValidationError(
            "anomaly review ledger belongs to a different candidate table: "
            f"recorded {recorded_digest}, current {actual_digest}"
        )
    expected_ids = {candidate.candidate_id for candidate in candidates}
    recorded_ids = set(decisions)
    if expected_ids != recorded_ids:
        missing = sorted(expected_ids - recorded_ids)
        extra = sorted(recorded_ids - expected_ids)
        raise ValidationError(
            f"anomaly reviews do not reconcile; missing={missing}, extra={extra}"
        )
    blocked = sorted(
        decision.candidate_id
        for decision in decisions.values()
        if decision.disposition == "block_gate"
    )
    if blocked:
        raise ValidationError(f"manual anomaly review blocks cleaning for {blocked}")


def render_anomaly_review(
    candidates: Sequence[AnomalyCandidate], decisions: Mapping[str, ReviewDecision]
) -> str:
    records = []
    for candidate in candidates:
        base = _anomaly_row(candidate)
        decision = decisions[candidate.candidate_id]
        base.update(
            {
                "anomaly_reviewed": "true",
                "anomaly_disposition": decision.disposition,
                "review_reason": decision.reason,
            }
        )
        records.append(base)
    return _render_csv(REVIEW_COLUMNS, records)


def render_clean_rows(
    rows: Sequence[ParsedCandle],
    candidates: Sequence[AnomalyCandidate],
    decisions: Mapping[str, ReviewDecision],
) -> str:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    candidate_id_by_key = {
        (candidate.source_file, candidate.source_row): candidate.candidate_id
        for candidate in candidates
    }
    records = []
    for row in rows:
        candidate_id = candidate_id_by_key.get((row.source_file, row.source_row))
        decision = decisions.get(candidate_id) if candidate_id is not None else None
        records.append(
            {
                "open_time_raw": row.open_time_raw,
                "close_time_raw": row.close_time_raw,
                "source_time_unit": row.source_time_unit,
                "open_time": _format_utc(row.open_time, row.source_time_unit),
                "close_time": _format_utc(row.close_time, row.source_time_unit),
                "open": row.open_text,
                "high": row.high_text,
                "low": row.low_text,
                "close": row.close_text,
                "base_volume": row.base_volume_text,
                "quote_volume": row.quote_volume_text,
                "trade_count": row.trade_count_text,
                "taker_buy_base_volume": row.taker_buy_base_volume_text,
                "taker_buy_quote_volume": row.taker_buy_quote_volume_text,
                "ignore": row.ignore_text,
                "source_period": row.source_period,
                "source_file": row.source_file,
                "source_member": row.source_member,
                "source_row": row.source_row,
                "is_complete_candle": "true",
                "is_duplicate": "false",
                "has_previous_contiguous_hour": str(
                    row.has_previous_contiguous_hour
                ).lower(),
                "gap_before_hours": (
                    row.gap_before_hours if row.gap_before_hours is not None else ""
                ),
                "contiguous_segment_id": row.contiguous_segment_id,
                "ohlc_valid": "true",
                "volume_valid": "true",
                "anomaly_flag": str(candidate_id is not None).lower(),
                "anomaly_reviewed": str(decision is not None).lower(),
                "anomaly_disposition": decision.disposition if decision is not None else "",
                "exclusion_reason": "",
            }
        )
    if set(candidate_by_id) != set(decisions):
        raise ValidationError("candidate and review IDs disagree while rendering clean rows")
    return _render_csv(CLEAN_COLUMNS, records)


def find_acquisition_manifest(config: ConfigNode) -> Path:
    root = paths.resolve(config.paths.outputs_runs)
    candidates = sorted(
        path.relative_to(paths.project_root().resolve())
        for path in root.glob("*/acquisition.manifest.json")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise ValidationError(
            f"expected exactly one accepted acquisition manifest, found {len(candidates)}; "
            "pass the intended manifest explicitly"
        )
    return candidates[0]


def verify_acquisition_inputs(
    config: ConfigNode,
    manifest_relative: Path,
) -> dict[str, str]:
    """Check the acquisition manifest, its inventory, and every raw file's checksum."""
    verify_manifest_checksum(manifest_relative)
    manifest = read_manifest(manifest_relative)
    if manifest["stage"] != "2. Acquisition" or manifest["mode"] != "acquire":
        raise ValidationError("upstream manifest is not Stage 2 acquisition evidence")
    targets = month_targets(config)
    expected_paths = {
        path.as_posix()
        for target in targets
        for path in (target.zip_relative_path, target.checksum_relative_path)
    }
    recorded = manifest["input_checksums"]
    if set(recorded) != expected_paths or manifest["row_count"] != len(targets):
        raise ValidationError(
            "acquisition manifest does not cover exactly the configured 82 monthly pairs"
        )
    for relative, digest in recorded.items():
        if sha256_file(paths.resolve(relative)) != digest:
            raise ValidationError(f"raw input checksum mismatch for {relative}")
    inventory_relative = manifest_relative.parent / "raw_inventory.csv"
    inventory_digest = sha256_file(paths.resolve(inventory_relative))
    if inventory_digest != manifest["output_checksum"]:
        raise ValidationError("acquisition inventory checksum does not match its manifest")
    sidecar_relative = manifest_relative.parent / f"{manifest_relative.name}.sha256"
    return dict(
        sorted(
            {
                **recorded,
                manifest_relative.as_posix(): sha256_file(paths.resolve(manifest_relative)),
                sidecar_relative.as_posix(): sha256_file(paths.resolve(sidecar_relative)),
                inventory_relative.as_posix(): inventory_digest,
            }.items()
        )
    )


def _record_failure(log: RunLog, error: Exception) -> None:
    try:
        log.record(
            "validation_failed",
            reason=type(error).__name__,
            detail=str(error)[:500],
        )
    except Exception:  # pragma: no cover - logging cannot replace the cause
        pass


def _write_artifact(log: RunLog, relative: Path, content: str, rows: int) -> str:
    create_exclusive(relative, content)
    digest = sha256_bytes(content.encode("utf-8"))
    log.record("validation_artifact_written", path=relative.as_posix(), rows=rows, sha256=digest)
    return digest


def run_validation(
    config: ConfigNode | None = None,
    *,
    acquisition_manifest_relative: Path | None = None,
    review_relative: Path = REVIEW_RELATIVE_PATH,
) -> ValidationResult:
    """Run the first (review) pass or the final cleaning pass."""
    cfg = config if config is not None else load_config()
    upstream = (
        acquisition_manifest_relative
        if acquisition_manifest_relative is not None
        else find_acquisition_manifest(cfg)
    )
    draft = build_manifest(STAGE, MODE, cfg)
    log = create_run_log(draft.run_id, config=cfg)
    log.record("validation_started", acquisition_manifest=upstream.as_posix())
    try:
        input_checksums = verify_acquisition_inputs(cfg, upstream)
        targets = month_targets(cfg)
        rows: list[ParsedCandle] = []
        audits: list[FileAudit] = []
        exclusions: list[ExcludedRow] = []
        observed_headers: set[tuple[str, ...]] = set()
        for target in targets:
            month_rows, audit, month_exclusions = parse_month(cfg, target)
            rows.extend(month_rows)
            audits.append(audit)
            exclusions.extend(month_exclusions)
            if audit.header_present:
                observed_headers.add(audit.header_tokens)
        if len(observed_headers) > 1:
            raise ValidationError("source archives contain inconsistent header token sets")
        annotated, gaps = assign_segments_and_gaps(
            rows, cfg.coverage.raw_start_inclusive, cfg.coverage.raw_end_exclusive
        )
        missing_by_period: dict[str, int] = {}
        for gap in gaps:
            missing_by_period[gap.source_period_expected] = (
                missing_by_period.get(gap.source_period_expected, 0) + 1
            )
        audits = [
            replace(audit, missing_hours=missing_by_period.get(audit.target.period, 0))
            for audit in audits
        ]
        for audit in audits:
            if audit.accepted_complete_rows + audit.missing_hours != audit.expected_hours:
                raise ValidationError(
                    f"expected-hour reconciliation failed for {audit.target.period}: "
                    f"accepted {audit.accepted_complete_rows} + missing {audit.missing_hours} "
                    f"!= expected {audit.expected_hours}"
                )
        candidates = build_anomaly_candidates(annotated, cfg)
        candidate_content = render_anomaly_candidates(candidates)
        run_dir = run_relative_dir(draft.run_id, cfg)
        common = {
            "schema_observations.csv": (
                render_schema_observations(audits),
                len(audits),
            ),
            "file_reconciliation.csv": (render_reconciliation(audits), len(audits)),
            "gaps.csv": (render_gaps(gaps), len(gaps)),
            "excluded_rows.csv": (render_exclusions(exclusions), len(exclusions)),
            "anomaly_candidates.csv": (candidate_content, len(candidates)),
        }
        artifact_digests: dict[str, str] = {}
        for name, (content, count) in common.items():
            relative = run_dir / name
            artifact_digests[name] = _write_artifact(log, relative, content, count)

        review_exists = paths.resolve(review_relative).is_file()
        final_allowed = cfg.source.schema_verified and review_exists
        decisions: dict[str, ReviewDecision] = {}
        clean_relative: Path | None = None
        if final_allowed:
            recorded_digest, decisions = load_review_ledger(review_relative)
            validate_review_ledger(candidates, candidate_content, recorded_digest, decisions)
            review_content = render_anomaly_review(candidates, decisions)
            clean_content = render_clean_rows(annotated, candidates, decisions)
            review_path = run_dir / "anomaly_review.csv"
            clean_relative = run_dir / "clean_hourly.csv"
            artifact_digests[review_path.name] = _write_artifact(
                log, review_path, review_content, len(candidates)
            )
            artifact_digests[clean_relative.name] = _write_artifact(
                log, clean_relative, clean_content, len(annotated)
            )
            input_checksums[review_relative.as_posix()] = sha256_file(
                paths.resolve(review_relative)
            )
            status = "complete"
            manifest_slug = "validation"
            primary_content = clean_content
            primary_rows = len(annotated)
            primary_columns = list(CLEAN_COLUMNS)
        else:
            status = "review_required"
            manifest_slug = "validation-preflight"
            primary_content = candidate_content
            primary_rows = len(candidates)
            primary_columns = list(ANOMALY_COLUMNS)

        summary_payload = {
            "status": status,
            "source_files": len(audits),
            "parsed_data_rows": sum(audit.parsed_data_rows for audit in audits),
            "accepted_complete_rows": len(annotated),
            "incomplete_rows_excluded": sum(
                audit.incomplete_rows_excluded for audit in audits
            ),
            "nonstandard_close_rows_excluded": len(exclusions),
            "duplicate_rows": sum(audit.duplicate_rows for audit in audits),
            "invalid_rows": sum(audit.invalid_rows for audit in audits),
            "missing_hours": len(gaps),
            "contiguous_segments": max(row.contiguous_segment_id for row in annotated),
            "anomaly_candidates": len(candidates),
            "schema_verified_in_config": cfg.source.schema_verified,
            "manual_reviews_complete": final_allowed,
            "artifact_sha256": dict(sorted(artifact_digests.items())),
        }
        summary_content = json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
        summary_relative = run_dir / "data_quality_summary.json"
        _write_artifact(log, summary_relative, summary_content, 1)

        finished = replace(
            draft,
            input_checksums=dict(sorted(input_checksums.items())),
            exclusions={
                "incomplete_final_candles": sum(
                    audit.incomplete_rows_excluded for audit in audits
                ),
                "nonstandard_close_time_rows": len(exclusions),
                "duplicate_rows": sum(audit.duplicate_rows for audit in audits),
                "invalid_rows": sum(audit.invalid_rows for audit in audits),
            },
            eligibility_changes={
                "parsed_data_rows": sum(audit.parsed_data_rows for audit in audits),
                "accepted_complete_rows": len(annotated),
                "missing_expected_hours": len(gaps),
            },
        ).completed(
            sha256_bytes(primary_content.encode("utf-8")),
            primary_rows,
            primary_columns,
        )
        manifest_relative = finished.write_for_run(manifest_slug, cfg).relative_to(
            paths.project_root().resolve()
        )
        sidecar_relative, manifest_digest = write_manifest_checksum(manifest_relative)
        log.record(
            "manifest_checksum_written",
            path=sidecar_relative.as_posix(),
            sha256=manifest_digest,
            manifest=manifest_relative.as_posix(),
        )
        log.record(
            "validation_completed" if final_allowed else "validation_review_required",
            manifest=manifest_relative.as_posix(),
            accepted_rows=len(annotated),
            gaps=len(gaps),
            anomaly_candidates=len(candidates),
        )
    except Exception as error:
        _record_failure(log, error)
        raise
    return ValidationResult(
        run_id=draft.run_id,
        status=status,
        manifest_relative_path=manifest_relative,
        manifest_checksum_relative_path=sidecar_relative,
        log_relative_path=log.relative_path,
        clean_relative_path=clean_relative,
        row_count=len(annotated),
        excluded_row_count=len(exclusions),
        gap_count=len(gaps),
        anomaly_count=len(candidates),
    )


def summarise(result: ValidationResult) -> list[str]:
    lines = [
        f"run id                  {result.run_id}",
        f"validation status       {result.status}",
        f"accepted hourly rows    {result.row_count}",
        f"excluded source rows    {result.excluded_row_count}",
        f"missing hours           {result.gap_count}",
        f"anomaly candidates      {result.anomaly_count}",
        f"manifest                {result.manifest_relative_path.as_posix()}",
        f"manifest checksum       {result.manifest_checksum_relative_path.as_posix()}",
        f"run log                 {result.log_relative_path.as_posix()}",
    ]
    if result.clean_relative_path is not None:
        lines.insert(5, f"clean hourly data       {result.clean_relative_path.as_posix()}")
    else:
        lines.insert(5, "clean hourly data       pending schema/manual review")
    return lines
