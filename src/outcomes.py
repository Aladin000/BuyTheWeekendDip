"""Build forward returns for accepted events (development and held-out stages).

This is the first stage allowed to compute forward returns. It reads the
accepted feature file only after the accepted count audit is verified, and
parses prices only for the requested split. The development entry point is
fixed to development; the held-out stage selects held_out only after its own
guards. Event identity comes unchanged from the feature stage.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

from src import paths
from src.acquisition import verify_manifest_checksum, write_manifest_checksum
from src.artifacts import (
    RunLog,
    build_manifest,
    matches_active_protocol,
    create_exclusive,
    create_run_log,
    read_manifest,
    run_relative_dir,
    sha256_bytes,
    sha256_file,
)
from src.config import ConfigNode, load_config
from src.count_audit import (
    ACCEPTED_FEATURE_MANIFEST,
    COUNT_OUTPUT_COLUMNS,
    verify_feature_input,
)
from src.features import FEATURE_COLUMNS

STAGE = "6. Development outcomes"
MODE = "development"
DEVELOPMENT_PERIOD = "development"
HELD_OUT_PERIOD = "held_out"

ACCEPTED_COUNT_MANIFEST = Path(
    "outputs/runs/20260906T011026.732695Z-count-only-5b9fb859-11651d/"
    "count-audit.manifest.json"
)
ACCEPTED_COUNT_MANIFEST_SHA256 = (
    "9d04cd5646089bf5f8cfd6195f4362a2bcf96ae29abac5d0e6c675791f5149bb"
)
ACCEPTED_COUNT_OUTPUT_SHA256 = (
    "2290033a300039c05a377e7de1f49eb7ff8006f402a2e4b424b04de8ded096ed"
)

EVENT_OUTPUT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "event_sign",
    "signal_open_time",
    "signal_close_time",
    "signal_close",
    "contiguous_segment_id",
    "calendar_year",
    "calendar_month",
    "weekend",
    "crosses_into_monday_12h",
    "utc_time_block",
    "shock_log_return_1h",
    "shock_zscore",
    "shock_magnitude",
    "absolute_shock_z",
    "activity_percentile_90d",
    "low_activity_score",
    "low_activity",
    "rv24",
    "full_feature_eligible_hour",
    "complete_6h",
    "post_log_return_6h",
    "outcome_eligibility_reason_6h",
    "complete_12h",
    "post_log_return_12h",
    "post_positive_12h",
    "outcome_eligibility_reason_12h",
    "complete_24h",
    "post_log_return_24h",
    "outcome_eligibility_reason_24h",
    "eligible_h1_h2",
    "eligible_h3_h4",
    "model_eligibility_reason",
)

PATH_OUTPUT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "event_sign",
    "signal_close_time",
    "horizon_step_hours",
    "cumulative_post_log_return",
)

DRIFT_OUTPUT_COLUMNS: tuple[str, ...] = (
    "reference_id",
    "signal_open_time",
    "signal_close_time",
    "signal_close",
    "contiguous_segment_id",
    "calendar_year",
    "calendar_month",
    "weekend",
    "shock_zscore",
    "post_log_return_12h",
)

INVENTORY_COLUMNS: tuple[str, ...] = (
    "artifact_path",
    "sha256",
    "row_count",
    "column_schema_json",
)

_ANALYTICAL_SCHEMAS: dict[str, tuple[str, ...]] = {
    "development_event_outcomes.csv": EVENT_OUTPUT_COLUMNS,
    "development_event_paths_24h.csv": PATH_OUTPUT_COLUMNS,
    "development_drift_reference.csv": DRIFT_OUTPUT_COLUMNS,
}


class OutcomeError(RuntimeError):
    """Raised when outcomes cannot be constructed unambiguously."""


@dataclass(frozen=True)
class OutcomeInputRow:
    """One feature row, as needed for outcomes.

    ``close`` and the analysis fields are filled only for the selected split.
    Every row keeps its timestamps, segment and split, so a horizon can be
    rejected at a gap or split boundary without reading a price from another
    split.
    """

    open_time: datetime
    close_time: datetime
    segment_id: int
    analysis_period: str
    close: Decimal | None
    close_time_raw: int | None
    shock_eligible: bool
    negative_event: bool
    positive_event: bool
    event_id: str
    calendar_year: int | None
    calendar_month: str
    weekend: bool | None
    crosses_into_monday_12h: bool | None
    utc_time_block: str
    shock_log_return_1h: float | None
    shock_zscore: float | None
    shock_magnitude: float | None
    absolute_shock_z: float | None
    activity_percentile: float | None
    low_activity_score: float | None
    low_activity: bool | None
    rv24: float | None
    full_feature_eligible: bool


@dataclass(frozen=True)
class HorizonOutcome:
    complete: bool
    value: float | None
    reason: str


@dataclass(frozen=True)
class EventOutcome:
    source_index: int
    row: OutcomeInputRow
    sign: str
    horizons: dict[int, HorizonOutcome]
    eligible_h1_h2: bool
    eligible_h3_h4: bool
    model_reason: str


@dataclass(frozen=True)
class PathRecord:
    event_id: str
    event_sign: str
    signal_close_time: datetime
    step: int
    value: float


@dataclass(frozen=True)
class DriftRecord:
    source_index: int
    row: OutcomeInputRow
    value_12h: float


@dataclass(frozen=True)
class Gate3Expectations:
    accepted_negative: int
    accepted_positive: int
    structurally_complete: dict[tuple[str, int], int]
    eligible_h1_h2: int
    eligible_h3_h4: int


@dataclass(frozen=True)
class DevelopmentOutcomeBundle:
    events: tuple[EventOutcome, ...]
    paths: tuple[PathRecord, ...]
    drift_reference: tuple[DriftRecord, ...]


@dataclass(frozen=True)
class DevelopmentOutcomeResult:
    run_id: str
    manifest_relative_path: Path
    manifest_checksum_relative_path: Path
    log_relative_path: Path
    inventory_relative_path: Path
    event_relative_path: Path
    path_relative_path: Path
    drift_relative_path: Path
    event_count: int
    path_record_count: int
    drift_record_count: int


def _parse_utc(text: str, field: str, source_row: int) -> datetime:
    if not text.endswith("Z"):
        raise OutcomeError(f"feature row {source_row} field {field} is not UTC")
    try:
        value = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        raise OutcomeError(
            f"feature row {source_row} field {field} is not an ISO UTC timestamp"
        ) from None
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise OutcomeError(f"feature row {source_row} field {field} is not UTC")
    return value.astimezone(timezone.utc)


def _parse_bool(text: str, field: str, source_row: int) -> bool:
    if text == "true":
        return True
    if text == "false":
        return False
    raise OutcomeError(f"feature row {source_row} field {field} is not true or false")


def _parse_optional_bool(text: str, field: str, source_row: int) -> bool | None:
    return None if text == "" else _parse_bool(text, field, source_row)


def _parse_optional_float(text: str, field: str, source_row: int) -> float | None:
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        raise OutcomeError(f"feature row {source_row} field {field} is not numeric") from None
    if not math.isfinite(value):
        raise OutcomeError(f"feature row {source_row} field {field} is non-finite")
    return value


def _parse_close(text: str, source_row: int) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise OutcomeError(f"feature row {source_row} close is not numeric") from None
    if not value.is_finite() or value <= 0:
        raise OutcomeError(f"feature row {source_row} close must be positive and finite")
    return value


def read_outcome_rows(
    relative_path: Path,
    expected_rows: int,
    config: ConfigNode,
    *,
    analysis_period: str = DEVELOPMENT_PERIOD,
) -> list[OutcomeInputRow]:
    """Read prices only for the explicitly selected analysis period."""
    if analysis_period not in {DEVELOPMENT_PERIOD, HELD_OUT_PERIOD}:
        raise OutcomeError("outcome reader requires development or held_out")
    source = paths.require_file(paths.resolve(relative_path))
    rows: list[OutcomeInputRow] = []
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = tuple(next(reader))
            except StopIteration:
                raise OutcomeError("pre-outcome feature artifact is empty") from None
            if header != FEATURE_COLUMNS:
                raise OutcomeError("feature artifact schema does not match the feature stage")
            index = {name: header.index(name) for name in header}
            for source_row, cells in enumerate(reader, start=2):
                if len(cells) != len(header):
                    raise OutcomeError(
                        f"feature row {source_row} has {len(cells)} fields; expected {len(header)}"
                    )
                period = cells[index["analysis_period"]]
                if period not in {"warmup", "development", "held_out"}:
                    raise OutcomeError(
                        f"feature row {source_row} has unknown analysis period {period!r}"
                    )
                open_time = _parse_utc(cells[index["open_time"]], "open_time", source_row)
                close_time = _parse_utc(cells[index["close_time"]], "close_time", source_row)
                try:
                    segment_id = int(cells[index["contiguous_segment_id"]])
                except ValueError:
                    raise OutcomeError(
                        f"feature row {source_row} has a non-integer segment ID"
                    ) from None

                if period == analysis_period:
                    try:
                        close_time_raw = int(cells[index["close_time_raw"]])
                        calendar_year = int(cells[index["calendar_year"]])
                    except ValueError:
                        raise OutcomeError(
                            f"feature row {source_row} has a non-integer timestamp or year"
                        ) from None
                    row = OutcomeInputRow(
                        open_time=open_time,
                        close_time=close_time,
                        segment_id=segment_id,
                        analysis_period=period,
                        close=_parse_close(cells[index["close"]], source_row),
                        close_time_raw=close_time_raw,
                        shock_eligible=_parse_bool(
                            cells[index["shock_eligible_hour"]],
                            "shock_eligible_hour",
                            source_row,
                        ),
                        negative_event=_parse_bool(
                            cells[index["negative_event"]], "negative_event", source_row
                        ),
                        positive_event=_parse_bool(
                            cells[index["positive_event"]], "positive_event", source_row
                        ),
                        event_id=cells[index["event_id"]],
                        calendar_year=calendar_year,
                        calendar_month=cells[index["calendar_month"]],
                        weekend=_parse_bool(cells[index["weekend"]], "weekend", source_row),
                        crosses_into_monday_12h=_parse_optional_bool(
                            cells[index["crosses_into_monday_12h"]],
                            "crosses_into_monday_12h",
                            source_row,
                        ),
                        utc_time_block=cells[index["utc_time_block"]],
                        shock_log_return_1h=_parse_optional_float(
                            cells[index["log_return_1h"]], "log_return_1h", source_row
                        ),
                        shock_zscore=_parse_optional_float(
                            cells[index["shock_zscore"]], "shock_zscore", source_row
                        ),
                        shock_magnitude=_parse_optional_float(
                            cells[index["shock_magnitude"]], "shock_magnitude", source_row
                        ),
                        absolute_shock_z=_parse_optional_float(
                            cells[index["absolute_shock_z"]], "absolute_shock_z", source_row
                        ),
                        activity_percentile=_parse_optional_float(
                            cells[index["activity_percentile_90d"]],
                            "activity_percentile_90d",
                            source_row,
                        ),
                        low_activity_score=_parse_optional_float(
                            cells[index["low_activity_score"]],
                            "low_activity_score",
                            source_row,
                        ),
                        low_activity=_parse_optional_bool(
                            cells[index["low_activity"]], "low_activity", source_row
                        ),
                        rv24=_parse_optional_float(cells[index["rv24"]], "rv24", source_row),
                        full_feature_eligible=_parse_bool(
                            cells[index["full_feature_eligible_hour"]],
                            "full_feature_eligible_hour",
                            source_row,
                        ),
                    )
                    _validate_analysis_row(row, source_row, config, analysis_period)
                else:
                    # Rows outside the selected split keep only their timestamps
                    # and segment; no price, event, return or covariate is read.
                    row = OutcomeInputRow(
                        open_time=open_time,
                        close_time=close_time,
                        segment_id=segment_id,
                        analysis_period=period,
                        close=None,
                        close_time_raw=None,
                        shock_eligible=False,
                        negative_event=False,
                        positive_event=False,
                        event_id="",
                        calendar_year=None,
                        calendar_month="",
                        weekend=None,
                        crosses_into_monday_12h=None,
                        utc_time_block="",
                        shock_log_return_1h=None,
                        shock_zscore=None,
                        shock_magnitude=None,
                        absolute_shock_z=None,
                        activity_percentile=None,
                        low_activity_score=None,
                        low_activity=None,
                        rv24=None,
                        full_feature_eligible=False,
                    )
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as error:
        raise OutcomeError(f"cannot read feature artifact: {error}") from None

    if len(rows) != expected_rows:
        raise OutcomeError(
            f"feature row count {len(rows)} does not match manifest {expected_rows}"
        )
    _validate_order(rows)
    return rows


def _validate_analysis_row(
    row: OutcomeInputRow,
    source_row: int,
    config: ConfigNode,
    analysis_period: str,
) -> None:
    if row.segment_id < 0:
        raise OutcomeError(f"feature row {source_row} has a negative segment ID")
    split = getattr(config.splits, analysis_period)
    if not split.start_inclusive <= row.close_time < split.end_exclusive:
        raise OutcomeError(
            f"feature row {source_row} {analysis_period} label disagrees with its close time"
        )
    if row.calendar_year != row.close_time.year:
        raise OutcomeError(f"feature row {source_row} calendar year is inconsistent")
    if row.calendar_month != row.close_time.strftime("%Y-%m"):
        raise OutcomeError(f"feature row {source_row} calendar month is inconsistent")
    expected_weekend = row.close_time.isoweekday() in set(
        config.calendar.weekend_iso_weekdays
    )
    if row.weekend != expected_weekend:
        raise OutcomeError(f"feature row {source_row} weekend flag is inconsistent")
    if row.negative_event and row.positive_event:
        raise OutcomeError(f"feature row {source_row} has both event signs")
    is_event = row.negative_event or row.positive_event
    if is_event != bool(row.event_id):
        raise OutcomeError(f"feature row {source_row} event ID presence is inconsistent")
    if is_event and not row.shock_eligible:
        raise OutcomeError(f"feature row {source_row} has an ineligible accepted event")
    if row.negative_event:
        if row.event_id != f"negative-{row.close_time_raw}":
            raise OutcomeError(f"feature row {source_row} has the wrong negative event ID")
        if row.shock_zscore is None or not row.shock_zscore < config.shock.negative_threshold:
            raise OutcomeError(f"feature row {source_row} negative event violates its threshold")
    if row.positive_event:
        if row.event_id != f"positive-{row.close_time_raw}":
            raise OutcomeError(f"feature row {source_row} has the wrong positive event ID")
        if row.shock_zscore is None or not row.shock_zscore > config.shock.positive_threshold:
            raise OutcomeError(f"feature row {source_row} positive event violates its threshold")
    if row.full_feature_eligible:
        required = (
            row.activity_percentile,
            row.low_activity_score,
            row.low_activity,
            row.rv24,
        )
        if any(value is None for value in required):
            raise OutcomeError(
                f"feature row {source_row} is full-feature eligible with a missing covariate"
            )


def _validate_order(rows: Sequence[OutcomeInputRow]) -> None:
    if not rows:
        raise OutcomeError("feature artifact contains no rows")
    previous: OutcomeInputRow | None = None
    for row in rows:
        if previous is not None:
            if row.open_time <= previous.open_time:
                raise OutcomeError("feature rows are not strictly chronological")
            same_segment = row.segment_id == previous.segment_id
            contiguous = row.open_time - previous.open_time == timedelta(hours=1)
            if same_segment != contiguous:
                raise OutcomeError(
                    "feature segment identity disagrees with timestamp contiguity"
                )
            if not same_segment and row.segment_id <= previous.segment_id:
                raise OutcomeError("feature segment IDs do not increase after a gap")
        previous = row


def _horizon_reason(
    rows: Sequence[OutcomeInputRow],
    index: int,
    horizon: int,
    analysis_period: str = DEVELOPMENT_PERIOD,
) -> str:
    row = rows[index]
    if row.analysis_period != analysis_period:
        return f"not_{analysis_period}"
    for step in range(1, horizon + 1):
        if index + step >= len(rows):
            return "end_of_available_timestamps"
        future = rows[index + step]
        if (
            future.open_time != row.open_time + timedelta(hours=step)
            or future.segment_id != row.segment_id
        ):
            return "missing_or_noncontiguous_future_bar"
        if future.analysis_period != analysis_period:
            return f"crosses_{analysis_period}_boundary"
        if future.close is None:
            raise OutcomeError(f"an {analysis_period} future row has no parsed close")
    return ""


def _log_return(current: Decimal, future: Decimal) -> float:
    value = float(future.ln() - current.ln())
    if not math.isfinite(value):
        raise OutcomeError("outcome calculation produced a non-finite value")
    return value


def build_outcomes(
    rows: Sequence[OutcomeInputRow],
    config: ConfigNode,
    *,
    analysis_period: str,
) -> DevelopmentOutcomeBundle:
    """Construct one split's outcomes without changing accepted event identity."""
    if analysis_period not in {DEVELOPMENT_PERIOD, HELD_OUT_PERIOD}:
        raise OutcomeError("outcome construction requires development or held_out")
    horizons = tuple(
        sorted(
            {
                config.outcomes.primary_horizon_hours,
                *config.outcomes.secondary_horizon_hours,
            }
        )
    )
    if horizons != (6, 12, 24):
        raise OutcomeError("outcomes require the locked 6/12/24-hour horizons")

    events: list[EventOutcome] = []
    paths: list[PathRecord] = []
    drift: list[DriftRecord] = []
    for index, row in enumerate(rows):
        if row.analysis_period != analysis_period:
            continue
        if row.close is None:
            raise OutcomeError(f"{analysis_period} row has no parsed close")

        is_event = row.negative_event or row.positive_event
        is_drift_reference = (
            row.shock_eligible
            and row.shock_zscore is not None
            and row.shock_zscore >= config.shock.negative_threshold
        )
        if not is_event and not is_drift_reference:
            continue

        horizon_values: dict[int, HorizonOutcome] = {}
        required_horizons = horizons if is_event else (12,)
        for horizon in required_horizons:
            reason = _horizon_reason(rows, index, horizon, analysis_period)
            value = None
            if not reason:
                future_close = rows[index + horizon].close
                if future_close is None:
                    raise OutcomeError(
                        f"complete {analysis_period} horizon has no endpoint close"
                    )
                value = _log_return(row.close, future_close)
            horizon_values[horizon] = HorizonOutcome(
                complete=not reason, value=value, reason=reason
            )

        if is_event:
            sign = "negative" if row.negative_event else "positive"
            eligible_h1_h2 = row.negative_event and horizon_values[12].complete
            eligible_h3_h4 = eligible_h1_h2 and row.full_feature_eligible
            if row.positive_event:
                model_reason = "not_applicable_positive_event"
            elif not horizon_values[12].complete:
                model_reason = "incomplete_12h_outcome"
            elif not row.full_feature_eligible:
                model_reason = "incomplete_model_covariates"
            else:
                model_reason = ""
            event = EventOutcome(
                source_index=index,
                row=row,
                sign=sign,
                horizons=horizon_values,
                eligible_h1_h2=eligible_h1_h2,
                eligible_h3_h4=eligible_h3_h4,
                model_reason=model_reason,
            )
            events.append(event)
            # Only negative events get a 24-hour path; positive events are
            # used only in the separate H5 comparison.
            if row.negative_event and horizon_values[24].complete:
                for step in range(1, 25):
                    future_close = rows[index + step].close
                    if future_close is None:
                        raise OutcomeError(
                            f"complete path has no {analysis_period} close"
                        )
                    paths.append(
                        PathRecord(
                            event_id=row.event_id,
                            event_sign=sign,
                            signal_close_time=row.close_time,
                            step=step,
                            value=_log_return(row.close, future_close),
                        )
                    )

        # The drift reference is every shock-eligible hour that is not a
        # negative candidate and has a complete 12-hour horizon.
        if (
            is_drift_reference
            and horizon_values[12].complete
        ):
            value = horizon_values[12].value
            if value is None:
                raise OutcomeError("complete drift-reference horizon has no value")
            drift.append(DriftRecord(source_index=index, row=row, value_12h=value))

    event_ids = [event.row.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise OutcomeError(f"{analysis_period} event IDs are not unique")
    return DevelopmentOutcomeBundle(tuple(events), tuple(paths), tuple(drift))


def build_development_outcomes(
    rows: Sequence[OutcomeInputRow], config: ConfigNode
) -> DevelopmentOutcomeBundle:
    """Build development outcomes."""
    return build_outcomes(rows, config, analysis_period=DEVELOPMENT_PERIOD)


def _find_count(
    records: Sequence[dict[str, str]],
    *,
    measure: str,
    sign: str,
    horizon: str = "",
    model: str = "",
    analysis_period: str = DEVELOPMENT_PERIOD,
) -> int:
    matches = [
        record
        for record in records
        if record["scope_type"] == "split"
        and record["scope"] == analysis_period
        and record["measure"] == measure
        and record["sign"] == sign
        and record["horizon_hours"] == horizon
        and record["model_or_strategy"] == model
        and record["weekend_group"] == ""
        and record["activity_group"] == ""
    ]
    if len(matches) != 1:
        raise OutcomeError(
            f"count audit has {len(matches)} records for {measure}/{sign}/{horizon}/{model}"
        )
    try:
        return int(matches[0]["count"])
    except ValueError:
        raise OutcomeError("count audit value is not an integer") from None


def read_gate3_expectations(
    relative_path: Path, *, analysis_period: str = DEVELOPMENT_PERIOD
) -> Gate3Expectations:
    if analysis_period not in {DEVELOPMENT_PERIOD, HELD_OUT_PERIOD}:
        raise OutcomeError("count audit expectations require development or held_out")
    source = paths.require_file(paths.resolve(relative_path))
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != COUNT_OUTPUT_COLUMNS:
                raise OutcomeError("count audit schema is not the accepted schema")
            records = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise OutcomeError(f"cannot read count audit: {error}") from None
    complete = {
        (sign, horizon): _find_count(
            records,
            measure="structurally_complete_events",
            sign=sign,
            horizon=str(horizon),
            analysis_period=analysis_period,
        )
        for sign in ("negative", "positive")
        for horizon in (6, 12, 24)
    }
    return Gate3Expectations(
        accepted_negative=_find_count(
            records, measure="accepted_events", sign="negative",
            analysis_period=analysis_period,
        ),
        accepted_positive=_find_count(
            records, measure="accepted_events", sign="positive",
            analysis_period=analysis_period,
        ),
        structurally_complete=complete,
        eligible_h1_h2=_find_count(
            records,
            measure="structurally_eligible_events",
            sign="negative",
            horizon="12",
            model="H1_H2",
            analysis_period=analysis_period,
        ),
        eligible_h3_h4=_find_count(
            records,
            measure="structurally_eligible_events",
            sign="negative",
            horizon="12",
            model="H3_H4",
            analysis_period=analysis_period,
        ),
    )


def reconcile_with_gate3(
    bundle: DevelopmentOutcomeBundle, expected: Gate3Expectations
) -> None:
    by_sign = {
        sign: sum(event.sign == sign for event in bundle.events)
        for sign in ("negative", "positive")
    }
    if by_sign != {
        "negative": expected.accepted_negative,
        "positive": expected.accepted_positive,
    }:
        raise OutcomeError("event counts do not reconcile with the count audit")
    for sign in ("negative", "positive"):
        for horizon in (6, 12, 24):
            actual = sum(
                event.sign == sign and event.horizons[horizon].complete
                for event in bundle.events
            )
            if actual != expected.structurally_complete[(sign, horizon)]:
                raise OutcomeError(
                    f"{sign} {horizon}h completeness does not reconcile with the count audit"
                )
    if sum(event.eligible_h1_h2 for event in bundle.events) != expected.eligible_h1_h2:
        raise OutcomeError("H1/H2 eligibility does not reconcile with the count audit")
    if sum(event.eligible_h3_h4 for event in bundle.events) != expected.eligible_h3_h4:
        raise OutcomeError("H3/H4 eligibility does not reconcile with the count audit")


def _format_time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        raise OutcomeError("refusing to render a non-finite outcome")
    return format(value, ".17g")


def _format_bool(value: bool | None) -> str:
    return "" if value is None else str(value).lower()


def render_event_outcomes(events: Sequence[EventOutcome]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(EVENT_OUTPUT_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for event in events:
        row = event.row
        writer.writerow(
            {
                "event_id": row.event_id,
                "event_sign": event.sign,
                "signal_open_time": _format_time(row.open_time),
                "signal_close_time": _format_time(row.close_time),
                "signal_close": format(row.close, "f") if row.close is not None else "",
                "contiguous_segment_id": row.segment_id,
                "calendar_year": row.calendar_year,
                "calendar_month": row.calendar_month,
                "weekend": _format_bool(row.weekend),
                "crosses_into_monday_12h": _format_bool(
                    row.crosses_into_monday_12h
                ),
                "utc_time_block": row.utc_time_block,
                "shock_log_return_1h": _format_float(row.shock_log_return_1h),
                "shock_zscore": _format_float(row.shock_zscore),
                "shock_magnitude": _format_float(row.shock_magnitude),
                "absolute_shock_z": _format_float(row.absolute_shock_z),
                "activity_percentile_90d": _format_float(row.activity_percentile),
                "low_activity_score": _format_float(row.low_activity_score),
                "low_activity": _format_bool(row.low_activity),
                "rv24": _format_float(row.rv24),
                "full_feature_eligible_hour": _format_bool(row.full_feature_eligible),
                "complete_6h": _format_bool(event.horizons[6].complete),
                "post_log_return_6h": _format_float(event.horizons[6].value),
                "outcome_eligibility_reason_6h": event.horizons[6].reason,
                "complete_12h": _format_bool(event.horizons[12].complete),
                "post_log_return_12h": _format_float(event.horizons[12].value),
                "post_positive_12h": (
                    ""
                    if event.horizons[12].value is None
                    else _format_bool(event.horizons[12].value > 0)
                ),
                "outcome_eligibility_reason_12h": event.horizons[12].reason,
                "complete_24h": _format_bool(event.horizons[24].complete),
                "post_log_return_24h": _format_float(event.horizons[24].value),
                "outcome_eligibility_reason_24h": event.horizons[24].reason,
                "eligible_h1_h2": _format_bool(event.eligible_h1_h2),
                "eligible_h3_h4": _format_bool(event.eligible_h3_h4),
                "model_eligibility_reason": event.model_reason,
            }
        )
    return buffer.getvalue()


def render_paths(paths_: Sequence[PathRecord]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(PATH_OUTPUT_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for record in paths_:
        writer.writerow(
            {
                "event_id": record.event_id,
                "event_sign": record.event_sign,
                "signal_close_time": _format_time(record.signal_close_time),
                "horizon_step_hours": record.step,
                "cumulative_post_log_return": _format_float(record.value),
            }
        )
    return buffer.getvalue()


def render_drift_reference(records: Sequence[DriftRecord]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(DRIFT_OUTPUT_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for record in records:
        row = record.row
        writer.writerow(
            {
                "reference_id": f"reference-{row.close_time_raw}",
                "signal_open_time": _format_time(row.open_time),
                "signal_close_time": _format_time(row.close_time),
                "signal_close": format(row.close, "f") if row.close is not None else "",
                "contiguous_segment_id": row.segment_id,
                "calendar_year": row.calendar_year,
                "calendar_month": row.calendar_month,
                "weekend": _format_bool(row.weekend),
                "shock_zscore": _format_float(row.shock_zscore),
                "post_log_return_12h": _format_float(record.value_12h),
            }
        )
    return buffer.getvalue()


def render_inventory(
    artifacts: Sequence[tuple[Path, str, int, Sequence[str]]]
) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(INVENTORY_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for path, digest, row_count, schema in artifacts:
        writer.writerow(
            {
                "artifact_path": path.as_posix(),
                "sha256": digest,
                "row_count": row_count,
                "column_schema_json": json.dumps(list(schema), separators=(",", ":")),
            }
        )
    return buffer.getvalue()


def verify_development_inventory(manifest_relative: Path) -> dict[str, str]:
    """Verify the manifest-to-inventory-to-artifact checksum chain."""
    verify_manifest_checksum(manifest_relative)
    manifest = read_manifest(manifest_relative)
    if manifest["stage"] != STAGE or manifest["mode"] != MODE:
        raise OutcomeError("manifest is not a development-outcome manifest")
    if manifest["column_schema"] != list(INVENTORY_COLUMNS):
        raise OutcomeError("development manifest does not declare the inventory schema")
    inventory_relative = manifest_relative.parent / "development_output_inventory.csv"
    if sha256_file(paths.resolve(inventory_relative)) != manifest["output_checksum"]:
        raise OutcomeError("development inventory checksum does not match its manifest")
    try:
        with paths.require_file(paths.resolve(inventory_relative)).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != INVENTORY_COLUMNS:
                raise OutcomeError("development inventory schema is invalid")
            records = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise OutcomeError(f"cannot read development inventory: {error}") from None
    if len(records) != manifest["row_count"]:
        raise OutcomeError("development inventory row count disagrees with its manifest")
    if sorted(Path(record["artifact_path"]).name for record in records) != sorted(
        _ANALYTICAL_SCHEMAS
    ):
        raise OutcomeError("development inventory does not name the three required artifacts")

    verified: dict[str, str] = {}
    for record in records:
        relative = Path(record["artifact_path"])
        if relative.parent != manifest_relative.parent:
            raise OutcomeError("development artifact is outside its run directory")
        digest = record["sha256"]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise OutcomeError("development inventory contains an invalid SHA-256")
        artifact = paths.require_file(paths.resolve(relative))
        if sha256_file(artifact) != digest:
            raise OutcomeError(f"development artifact checksum failed for {relative.name}")
        try:
            expected_schema = json.loads(record["column_schema_json"])
            expected_rows = int(record["row_count"])
        except (json.JSONDecodeError, TypeError, ValueError):
            raise OutcomeError("development inventory metadata is malformed") from None
        if (
            not isinstance(expected_schema, list)
            or not expected_schema
            or any(not isinstance(name, str) or not name for name in expected_schema)
            or len(expected_schema) != len(set(expected_schema))
            or expected_rows < 0
        ):
            raise OutcomeError("development inventory schema or row count is invalid")
        if expected_schema != list(_ANALYTICAL_SCHEMAS[relative.name]):
            raise OutcomeError(
                f"development inventory declares the wrong schema for {relative.name}"
            )
        try:
            with artifact.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, strict=True)
                header = next(reader, None)
                row_count = sum(1 for _ in reader)
        except (OSError, UnicodeError, csv.Error) as error:
            raise OutcomeError(f"cannot verify development artifact: {error}") from None
        if header != expected_schema or row_count != expected_rows:
            raise OutcomeError(
                f"development artifact schema/row count failed for {relative.name}"
            )
        verified[relative.as_posix()] = digest
    return dict(sorted(verified.items()))


def verify_count_input(
    config: ConfigNode,
    manifest_relative: Path,
    *,
    require_frozen_acceptance: bool = False,
    analysis_period: str = DEVELOPMENT_PERIOD,
) -> tuple[Path, Gate3Expectations, dict[str, str]]:
    if require_frozen_acceptance and manifest_relative != ACCEPTED_COUNT_MANIFEST:
        raise OutcomeError("development execution must use the accepted count-audit manifest")
    verify_manifest_checksum(manifest_relative)
    manifest_digest = sha256_file(paths.resolve(manifest_relative))
    manifest = read_manifest(manifest_relative)
    if manifest["stage"] != "5. Count-only audit" or manifest["mode"] != "count-only":
        raise OutcomeError("upstream manifest is not a count-audit manifest")
    if manifest["column_schema"] != list(COUNT_OUTPUT_COLUMNS):
        raise OutcomeError("upstream manifest does not declare the count-audit schema")
    if not matches_active_protocol(manifest, config):
        raise OutcomeError("count audit does not match active specification/config")
    if manifest["code_identity"]["git_dirty"] is True:
        raise OutcomeError("count audit was produced from a dirty repository")
    count_relative = manifest_relative.parent / "feasibility_counts.csv"
    count_digest = sha256_file(paths.resolve(count_relative))
    if count_digest != manifest["output_checksum"]:
        raise OutcomeError("count audit checksum does not match its manifest")
    if require_frozen_acceptance and (
        manifest_digest != ACCEPTED_COUNT_MANIFEST_SHA256
        or count_digest != ACCEPTED_COUNT_OUTPUT_SHA256
    ):
        raise OutcomeError("accepted count audit does not match the frozen run")
    sidecar_relative = manifest_relative.parent / f"{manifest_relative.name}.sha256"
    checksums = {
        count_relative.as_posix(): count_digest,
        manifest_relative.as_posix(): manifest_digest,
        sidecar_relative.as_posix(): sha256_file(paths.resolve(sidecar_relative)),
    }
    return (
        count_relative,
        read_gate3_expectations(
            count_relative, analysis_period=analysis_period
        ),
        dict(sorted(checksums.items())),
    )


def verify_development_inputs(
    config: ConfigNode,
    feature_manifest_relative: Path,
    count_manifest_relative: Path,
    *,
    require_frozen_acceptance: bool = False,
) -> tuple[Path, int, Gate3Expectations, dict[str, str]]:
    feature_relative, expected_rows, feature_checksums = verify_feature_input(
        config,
        feature_manifest_relative,
        require_frozen_acceptance=require_frozen_acceptance,
    )
    _, expectations, count_checksums = verify_count_input(
        config,
        count_manifest_relative,
        require_frozen_acceptance=require_frozen_acceptance,
    )
    overlap = set(feature_checksums) & set(count_checksums)
    if overlap:
        raise OutcomeError(f"duplicate upstream checksum paths: {sorted(overlap)}")
    return (
        feature_relative,
        expected_rows,
        expectations,
        dict(sorted({**feature_checksums, **count_checksums}.items())),
    )


def _record_failure(log: RunLog, error: Exception) -> None:
    try:
        log.record(
            "development_outcomes_failed",
            reason=type(error).__name__,
            detail=str(error)[:500],
        )
    except Exception:  # pragma: no cover - logging cannot replace the cause
        pass


def run_development_outcomes(
    config: ConfigNode | None = None,
    *,
    feature_manifest_relative: Path | None = None,
    count_manifest_relative: Path | None = None,
) -> DevelopmentOutcomeResult:
    """Persist development-only outcomes and their checksum inventory."""
    cfg = config if config is not None else load_config()
    feature_manifest = feature_manifest_relative or ACCEPTED_FEATURE_MANIFEST
    count_manifest = count_manifest_relative or ACCEPTED_COUNT_MANIFEST
    custom_input = feature_manifest_relative is not None or count_manifest_relative is not None
    if custom_input and (feature_manifest_relative is None or count_manifest_relative is None):
        raise OutcomeError("custom development input requires both upstream manifests")
    feature_is_accepted = feature_manifest == ACCEPTED_FEATURE_MANIFEST
    count_is_accepted = count_manifest == ACCEPTED_COUNT_MANIFEST
    if feature_is_accepted != count_is_accepted:
        raise OutcomeError("accepted and synthetic development inputs cannot be mixed")
    frozen_inputs = feature_is_accepted and count_is_accepted

    draft = build_manifest(STAGE, MODE, cfg)
    if frozen_inputs and draft.code_identity["git_dirty"] is not False:
        raise OutcomeError(
            "development outcomes require a clean committed repository; "
            "offline tests must use explicit synthetic manifests"
        )
    log = create_run_log(draft.run_id, config=cfg)
    log.record(
        "development_outcomes_started",
        feature_manifest=feature_manifest.as_posix(),
        count_manifest=count_manifest.as_posix(),
        analysis_period=DEVELOPMENT_PERIOD,
        held_out_outcomes_accessed=False,
    )
    try:
        feature_relative, expected_rows, gate3, input_checksums = verify_development_inputs(
            cfg,
            feature_manifest,
            count_manifest,
            require_frozen_acceptance=frozen_inputs,
        )
        rows = read_outcome_rows(feature_relative, expected_rows, cfg)
        bundle = build_development_outcomes(rows, cfg)
        reconcile_with_gate3(bundle, gate3)

        run_dir = run_relative_dir(draft.run_id, cfg)
        event_relative = run_dir / "development_event_outcomes.csv"
        path_relative = run_dir / "development_event_paths_24h.csv"
        drift_relative = run_dir / "development_drift_reference.csv"
        inventory_relative = run_dir / "development_output_inventory.csv"
        payloads = (
            (
                event_relative,
                render_event_outcomes(bundle.events),
                len(bundle.events),
                EVENT_OUTPUT_COLUMNS,
            ),
            (path_relative, render_paths(bundle.paths), len(bundle.paths), PATH_OUTPUT_COLUMNS),
            (
                drift_relative,
                render_drift_reference(bundle.drift_reference),
                len(bundle.drift_reference),
                DRIFT_OUTPUT_COLUMNS,
            ),
        )
        inventory_rows: list[tuple[Path, str, int, Sequence[str]]] = []
        for relative, content, row_count, schema in payloads:
            create_exclusive(relative, content)
            digest = sha256_bytes(content.encode("utf-8"))
            inventory_rows.append((relative, digest, row_count, schema))
            log.record(
                "development_artifact_written",
                path=relative.as_posix(),
                rows=row_count,
                sha256=digest,
                held_out_outcomes_accessed=False,
            )
        inventory_content = render_inventory(inventory_rows)
        create_exclusive(inventory_relative, inventory_content)
        inventory_digest = sha256_bytes(inventory_content.encode("utf-8"))
        log.record(
            "development_inventory_written",
            path=inventory_relative.as_posix(),
            rows=len(inventory_rows),
            sha256=inventory_digest,
        )

        event_counts = {
            sign: sum(event.sign == sign for event in bundle.events)
            for sign in ("negative", "positive")
        }
        complete_counts = {
            (sign, horizon): sum(
                event.sign == sign and event.horizons[horizon].complete
                for event in bundle.events
            )
            for sign in ("negative", "positive")
            for horizon in (6, 12, 24)
        }
        finished = replace(
            draft,
            input_checksums=input_checksums,
            exclusions={
                f"{sign}_incomplete_{horizon}h": event_counts[sign]
                - complete_counts[(sign, horizon)]
                for sign in ("negative", "positive")
                for horizon in (6, 12, 24)
            },
            eligibility_changes={
                "input_feature_rows": len(rows),
                "development_hour_rows": sum(
                    row.analysis_period == DEVELOPMENT_PERIOD for row in rows
                ),
                "development_event_rows": len(bundle.events),
                "negative_event_rows": event_counts["negative"],
                "positive_event_rows": event_counts["positive"],
                **{
                    f"{sign}_complete_{horizon}h": count
                    for (sign, horizon), count in complete_counts.items()
                },
                "eligible_h1_h2_rows": sum(
                    event.eligible_h1_h2 for event in bundle.events
                ),
                "eligible_h3_h4_rows": sum(
                    event.eligible_h3_h4 for event in bundle.events
                ),
                "complete_24h_path_events": len(bundle.paths) // 24,
                "path_output_rows": len(bundle.paths),
                "drift_reference_rows": len(bundle.drift_reference),
                "analytical_output_files": len(inventory_rows),
            },
        ).completed(inventory_digest, len(inventory_rows), list(INVENTORY_COLUMNS))
        manifest_path = finished.write_for_run("development-outcomes", cfg)
        manifest_relative = manifest_path.relative_to(paths.project_root().resolve())
        sidecar_relative, manifest_digest = write_manifest_checksum(manifest_relative)
        verify_development_inventory(manifest_relative)
        log.record(
            "manifest_checksum_written",
            path=sidecar_relative.as_posix(),
            sha256=manifest_digest,
            manifest=manifest_relative.as_posix(),
        )
        log.record(
            "development_outcomes_completed",
            manifest=manifest_relative.as_posix(),
            development_events=len(bundle.events),
            held_out_outcomes_accessed=False,
        )
    except Exception as error:
        _record_failure(log, error)
        raise

    return DevelopmentOutcomeResult(
        run_id=draft.run_id,
        manifest_relative_path=manifest_relative,
        manifest_checksum_relative_path=sidecar_relative,
        log_relative_path=log.relative_path,
        inventory_relative_path=inventory_relative,
        event_relative_path=event_relative,
        path_relative_path=path_relative,
        drift_relative_path=drift_relative,
        event_count=len(bundle.events),
        path_record_count=len(bundle.paths),
        drift_record_count=len(bundle.drift_reference),
    )


def summarise(result: DevelopmentOutcomeResult) -> list[str]:
    return [
        f"run id                  {result.run_id}",
        f"development events      {result.event_count}",
        f"24h path records        {result.path_record_count}",
        f"drift-reference rows    {result.drift_record_count}",
        f"output inventory        {result.inventory_relative_path.as_posix()}",
        f"manifest                {result.manifest_relative_path.as_posix()}",
        f"manifest checksum       {result.manifest_checksum_relative_path.as_posix()}",
        f"run log                 {result.log_relative_path.as_posix()}",
        "held-out outcomes       not accessed",
        "inference/backtest      not run",
    ]
