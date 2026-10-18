"""Count events and eligible hours before any outcome is looked at.

Reads the accepted feature file but parses only timestamps, segment IDs,
eligibility and event flags, and the weekend and activity labels. Price
columns are never read. Whether a future window is complete is decided from
timestamps, segments, and split alone; no price or return is computed.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

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
from src.features import FEATURE_COLUMNS

STAGE = "5. Count-only audit"
MODE = "count-only"

# The accepted (corrected) feature run. An earlier, superseded run also
# exists, so the default run pins this one instead of guessing.
ACCEPTED_FEATURE_MANIFEST = Path(
    "outputs/runs/20260906T003831.725064Z-features-92e5ff02-878985/"
    "features.manifest.json"
)
ACCEPTED_FEATURE_MANIFEST_SHA256 = (
    "022b4e1c0a6a3f23f7aad737302b17d6d178b0f987ba6488e44a95434aa87685"
)
ACCEPTED_FEATURE_OUTPUT_SHA256 = (
    "5ce67b4d4363b1c91bc1c79820a7332ea2590c6f3697efa1132f270994f0c2d4"
)

ANALYSIS_PERIODS: tuple[str, ...] = ("development", "held_out")
MODEL3_COEFFICIENT_COUNT = 9

# The only feature columns read here; no price, volume, return, or z-score.
COUNT_INPUT_COLUMNS: tuple[str, ...] = (
    "open_time",
    "close_time",
    "contiguous_segment_id",
    "shock_eligible_hour",
    "full_feature_eligible_hour",
    "negative_candidate",
    "positive_candidate",
    "negative_event",
    "positive_event",
    "event_id",
    "analysis_period",
    "calendar_year",
    "weekend",
    "low_activity",
)

COUNT_OUTPUT_COLUMNS: tuple[str, ...] = (
    "scope_type",
    "scope",
    "sign",
    "weekend_group",
    "activity_group",
    "horizon_hours",
    "model_or_strategy",
    "measure",
    "count",
    "denominator_measure",
    "denominator_count",
    "rate_per_1000",
    "interpretation_label",
)

_FORBIDDEN_COUNT_TOKENS: tuple[str, ...] = (
    "price",
    "return",
    "pnl",
    "profit",
    "wealth",
    "sharpe",
    "drawdown",
    "hit_rate",
)


class CountAuditError(RuntimeError):
    """Raised when the count audit cannot be completed cleanly."""


@dataclass(frozen=True)
class CountInputRow:
    """The fields kept from one feature row."""

    open_time: datetime
    close_time: datetime
    segment_id: int
    shock_eligible: bool
    full_feature_eligible: bool
    negative_candidate: bool
    positive_candidate: bool
    negative_event: bool
    positive_event: bool
    event_id: str
    analysis_period: str
    calendar_year: int
    weekend: bool
    low_activity: bool | None


@dataclass(frozen=True)
class CountRecord:
    """One count, rate, or interpretation label."""

    scope_type: str
    scope: str
    sign: str = ""
    weekend_group: str = ""
    activity_group: str = ""
    horizon_hours: str = ""
    model_or_strategy: str = ""
    measure: str = ""
    count: int = 0
    denominator_measure: str = ""
    denominator_count: int | None = None
    rate_per_1000: float | None = None
    interpretation_label: str = ""


@dataclass(frozen=True)
class CountAuditResult:
    run_id: str
    manifest_relative_path: Path
    manifest_checksum_relative_path: Path
    log_relative_path: Path
    counts_relative_path: Path
    input_row_count: int
    output_record_count: int


def _parse_utc(text: str, field: str, source_row: int) -> datetime:
    if not text.endswith("Z"):
        raise CountAuditError(f"feature row {source_row} field {field} is not UTC")
    try:
        value = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        raise CountAuditError(
            f"feature row {source_row} field {field} is not an ISO UTC timestamp"
        ) from None
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise CountAuditError(f"feature row {source_row} field {field} is not UTC")
    return value.astimezone(timezone.utc)


def _parse_bool(text: str, field: str, source_row: int) -> bool:
    if text == "true":
        return True
    if text == "false":
        return False
    raise CountAuditError(
        f"feature row {source_row} field {field} is not true or false"
    )


def _parse_optional_bool(text: str, field: str, source_row: int) -> bool | None:
    if text == "":
        return None
    return _parse_bool(text, field, source_row)


def read_count_rows(
    relative_path: Path, expected_rows: int, *, cooldown_hours: int
) -> list[CountInputRow]:
    """Read only the outcome-free columns the counts need.

    The full feature header is required, but other columns are never parsed
    or kept, so price values cannot affect the counts.
    """
    source = paths.require_file(paths.resolve(relative_path))
    rows: list[CountInputRow] = []
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = tuple(next(reader))
            except StopIteration:
                raise CountAuditError("pre-outcome feature artifact is empty") from None
            if header != FEATURE_COLUMNS:
                raise CountAuditError("feature artifact columns do not match the feature schema")
            indices = {name: header.index(name) for name in COUNT_INPUT_COLUMNS}
            for source_row, cells in enumerate(reader, start=2):
                if len(cells) != len(header):
                    raise CountAuditError(
                        f"feature row {source_row} has {len(cells)} fields; "
                        f"expected {len(header)}"
                    )
                try:
                    segment_id = int(cells[indices["contiguous_segment_id"]])
                    calendar_year = int(cells[indices["calendar_year"]])
                except ValueError:
                    raise CountAuditError(
                        f"feature row {source_row} has a non-integer segment or year"
                    ) from None
                row = CountInputRow(
                    open_time=_parse_utc(
                        cells[indices["open_time"]], "open_time", source_row
                    ),
                    close_time=_parse_utc(
                        cells[indices["close_time"]], "close_time", source_row
                    ),
                    segment_id=segment_id,
                    shock_eligible=_parse_bool(
                        cells[indices["shock_eligible_hour"]],
                        "shock_eligible_hour",
                        source_row,
                    ),
                    full_feature_eligible=_parse_bool(
                        cells[indices["full_feature_eligible_hour"]],
                        "full_feature_eligible_hour",
                        source_row,
                    ),
                    negative_candidate=_parse_bool(
                        cells[indices["negative_candidate"]],
                        "negative_candidate",
                        source_row,
                    ),
                    positive_candidate=_parse_bool(
                        cells[indices["positive_candidate"]],
                        "positive_candidate",
                        source_row,
                    ),
                    negative_event=_parse_bool(
                        cells[indices["negative_event"]], "negative_event", source_row
                    ),
                    positive_event=_parse_bool(
                        cells[indices["positive_event"]], "positive_event", source_row
                    ),
                    event_id=cells[indices["event_id"]],
                    analysis_period=cells[indices["analysis_period"]],
                    calendar_year=calendar_year,
                    weekend=_parse_bool(
                        cells[indices["weekend"]], "weekend", source_row
                    ),
                    low_activity=_parse_optional_bool(
                        cells[indices["low_activity"]], "low_activity", source_row
                    ),
                )
                _validate_row(row, source_row)
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as error:
        raise CountAuditError(f"cannot read feature artifact: {error}") from None

    if len(rows) != expected_rows:
        raise CountAuditError(
            f"feature row count {len(rows)} does not match manifest {expected_rows}"
        )
    _validate_order_and_cooldown(rows, cooldown_hours)
    return rows


def _validate_row(row: CountInputRow, source_row: int) -> None:
    if row.segment_id < 0:
        raise CountAuditError(f"feature row {source_row} has a negative segment ID")
    if row.analysis_period not in {"warmup", *ANALYSIS_PERIODS}:
        raise CountAuditError(
            f"feature row {source_row} has unknown analysis period {row.analysis_period!r}"
        )
    if row.calendar_year != row.close_time.year:
        raise CountAuditError(f"feature row {source_row} calendar year is inconsistent")
    if row.weekend != (row.close_time.isoweekday() in (6, 7)):
        raise CountAuditError(f"feature row {source_row} weekend flag is inconsistent")
    if row.full_feature_eligible and not row.shock_eligible:
        raise CountAuditError(
            f"feature row {source_row} is full-feature eligible but not shock eligible"
        )
    if row.full_feature_eligible != (row.low_activity is not None):
        raise CountAuditError(
            f"feature row {source_row} activity flag disagrees with full-feature eligibility"
        )
    if row.negative_candidate and row.positive_candidate:
        raise CountAuditError(f"feature row {source_row} has both candidate signs")
    if row.negative_event and not row.negative_candidate:
        raise CountAuditError(f"feature row {source_row} negative event is not a candidate")
    if row.positive_event and not row.positive_candidate:
        raise CountAuditError(f"feature row {source_row} positive event is not a candidate")
    if (row.negative_candidate or row.positive_candidate) and not row.shock_eligible:
        raise CountAuditError(f"feature row {source_row} candidate is shock-ineligible")
    is_event = row.negative_event or row.positive_event
    if is_event != bool(row.event_id):
        raise CountAuditError(f"feature row {source_row} event ID presence is inconsistent")
    if row.negative_event and not row.event_id.startswith("negative-"):
        raise CountAuditError(f"feature row {source_row} has the wrong negative event ID")
    if row.positive_event and not row.event_id.startswith("positive-"):
        raise CountAuditError(f"feature row {source_row} has the wrong positive event ID")


def _validate_order_and_cooldown(
    rows: Sequence[CountInputRow], cooldown_hours: int
) -> None:
    if type(cooldown_hours) is not int or cooldown_hours <= 0:
        raise CountAuditError("cooldown hours must be a positive integer")
    previous: CountInputRow | None = None
    event_ids: set[str] = set()
    last_event: dict[str, datetime | None] = {"negative": None, "positive": None}
    for row in rows:
        if previous is not None:
            if row.open_time <= previous.open_time:
                raise CountAuditError("feature rows are not strictly chronological")
            same_segment = row.segment_id == previous.segment_id
            contiguous = row.open_time - previous.open_time == timedelta(hours=1)
            if same_segment != contiguous:
                raise CountAuditError(
                    "feature segment identity disagrees with timestamp contiguity"
                )
            if not same_segment and row.segment_id <= previous.segment_id:
                raise CountAuditError("feature segment IDs do not increase after a gap")
        previous = row
        if row.event_id:
            if row.event_id in event_ids:
                raise CountAuditError(f"duplicate event ID {row.event_id!r}")
            event_ids.add(row.event_id)
        for sign, accepted in (
            ("negative", row.negative_event),
            ("positive", row.positive_event),
        ):
            if accepted:
                prior = last_event[sign]
                if prior is not None and row.close_time - prior < timedelta(
                    hours=cooldown_hours
                ):
                    raise CountAuditError(
                        f"accepted {sign} events violate the configured cooldown"
                    )
                last_event[sign] = row.close_time


def structural_horizon_flags(
    rows: Sequence[CountInputRow], horizons: Iterable[int]
) -> tuple[dict[int, bool], ...]:
    """Flag whether each row's next ``h`` hours exist with no gap or split change."""
    requested = tuple(horizons)
    if not requested or any(type(value) is not int or value <= 0 for value in requested):
        raise CountAuditError("structural horizons must be positive integers")
    flags: list[dict[int, bool]] = []
    for index, row in enumerate(rows):
        current: dict[int, bool] = {}
        for horizon in requested:
            complete = row.analysis_period in ANALYSIS_PERIODS and index + horizon < len(rows)
            if complete:
                for step in range(1, horizon + 1):
                    future = rows[index + step]
                    if (
                        future.open_time != row.open_time + timedelta(hours=step)
                        or future.segment_id != row.segment_id
                        or future.analysis_period != row.analysis_period
                    ):
                        complete = False
                        break
            current[horizon] = complete
        flags.append(current)
    return tuple(flags)


def _period_rows(
    rows: Sequence[CountInputRow], period: str
) -> list[tuple[int, CountInputRow]]:
    if period == "full_analysis":
        return [
            (index, row)
            for index, row in enumerate(rows)
            if row.analysis_period in ANALYSIS_PERIODS
        ]
    return [
        (index, row)
        for index, row in enumerate(rows)
        if row.analysis_period == period
    ]


def _event(row: CountInputRow, sign: str) -> bool:
    return row.negative_event if sign == "negative" else row.positive_event


def _candidate(row: CountInputRow, sign: str) -> bool:
    return row.negative_candidate if sign == "negative" else row.positive_candidate


def _activity_group(row: CountInputRow) -> str:
    if row.low_activity is None:
        return "unavailable"
    return "low" if row.low_activity else "other"


def model3_interpretation_label(
    cell_counts: Sequence[int], complete_case_n: int, config: ConfigNode
) -> str:
    """Apply the two locked Model 3 sample-size conditions."""
    if len(cell_counts) != 4 or any(type(value) is not int or value < 0 for value in cell_counts):
        raise CountAuditError("Model 3 requires four non-negative integer cell counts")
    if type(complete_case_n) is not int or complete_case_n < 0:
        raise CountAuditError("Model 3 complete-case N must be a non-negative integer")
    required_n = (
        config.inference.model3_min_n_per_coefficient * MODEL3_COEFFICIENT_COUNT
    )
    if (
        min(cell_counts) >= config.inference.model3_min_events_per_cell
        and complete_case_n >= required_n
    ):
        return "quantitatively_interpretable"
    return "exploratory_underpowered"


def combined_strategy_interpretation_label(count: int, config: ConfigNode) -> str:
    """Label the combined strategy as standard or too sparse."""
    if type(count) is not int or count < 0:
        raise CountAuditError("combined-strategy count must be a non-negative integer")
    if count >= config.strategy.sparse_trade_label_threshold:
        return "standard"
    return "descriptive_highly_unstable"


def _record(
    scope_type: str,
    scope: str,
    measure: str,
    count: int,
    *,
    sign: str = "",
    weekend_group: str = "",
    activity_group: str = "",
    horizon: int | None = None,
    model_or_strategy: str = "",
    denominator_measure: str = "",
    denominator_count: int | None = None,
    label: str = "",
) -> CountRecord:
    rate = None
    if denominator_count is not None:
        rate = 1000.0 * count / denominator_count if denominator_count else None
    return CountRecord(
        scope_type=scope_type,
        scope=scope,
        sign=sign,
        weekend_group=weekend_group,
        activity_group=activity_group,
        horizon_hours="" if horizon is None else str(horizon),
        model_or_strategy=model_or_strategy,
        measure=measure,
        count=count,
        denominator_measure=denominator_measure,
        denominator_count=denominator_count,
        rate_per_1000=rate,
        interpretation_label=label,
    )


def _scope_records(
    rows: Sequence[CountInputRow],
    flags: Sequence[dict[int, bool]],
    scope: str,
    config: ConfigNode,
    horizons: Sequence[int],
) -> list[CountRecord]:
    selected = _period_rows(rows, scope)
    output: list[CountRecord] = []
    scope_type = "analysis" if scope == "full_analysis" else "split"

    output.extend(
        [
            _record(scope_type, scope, "analysis_hours", len(selected)),
            _record(
                scope_type,
                scope,
                "shock_eligible_hours",
                sum(row.shock_eligible for _, row in selected),
            ),
            _record(
                scope_type,
                scope,
                "shock_ineligible_hours",
                sum(not row.shock_eligible for _, row in selected),
            ),
            _record(
                scope_type,
                scope,
                "full_feature_eligible_hours",
                sum(row.full_feature_eligible for _, row in selected),
            ),
            _record(
                scope_type,
                scope,
                "full_feature_ineligible_hours",
                sum(not row.full_feature_eligible for _, row in selected),
            ),
        ]
    )

    for weekend_value, weekend_name in ((True, "weekend"), (False, "weekday")):
        shock_denominator = sum(
            row.shock_eligible and row.weekend == weekend_value for _, row in selected
        )
        output.append(
            _record(
                scope_type,
                scope,
                "shock_eligible_hours",
                shock_denominator,
                weekend_group=weekend_name,
            )
        )
        numerator = sum(
            row.negative_event and row.weekend == weekend_value for _, row in selected
        )
        output.append(
            _record(
                scope_type,
                scope,
                "accepted_events_per_1000_eligible_hours",
                numerator,
                sign="negative",
                weekend_group=weekend_name,
                denominator_measure="shock_eligible_hours",
                denominator_count=shock_denominator,
            )
        )

    for weekend_value, weekend_name in ((True, "weekend"), (False, "weekday")):
        for low_value, activity_name in ((True, "low"), (False, "other")):
            denominator = sum(
                row.full_feature_eligible
                and row.weekend == weekend_value
                and row.low_activity is low_value
                for _, row in selected
            )
            numerator = sum(
                row.negative_event
                and row.full_feature_eligible
                and row.weekend == weekend_value
                and row.low_activity is low_value
                for _, row in selected
            )
            output.append(
                _record(
                    scope_type,
                    scope,
                    "full_feature_eligible_hours",
                    denominator,
                    weekend_group=weekend_name,
                    activity_group=activity_name,
                )
            )
            output.append(
                _record(
                    scope_type,
                    scope,
                    "accepted_events_per_1000_eligible_hours",
                    numerator,
                    sign="negative",
                    weekend_group=weekend_name,
                    activity_group=activity_name,
                    denominator_measure="full_feature_eligible_hours",
                    denominator_count=denominator,
                )
            )

    for sign in ("negative", "positive"):
        output.append(
            _record(
                scope_type,
                scope,
                "candidates",
                sum(_candidate(row, sign) for _, row in selected),
                sign=sign,
            )
        )
        output.append(
            _record(
                scope_type,
                scope,
                "accepted_events",
                sum(_event(row, sign) for _, row in selected),
                sign=sign,
            )
        )
        for weekend_value, weekend_name in ((True, "weekend"), (False, "weekday")):
            output.append(
                _record(
                    scope_type,
                    scope,
                    "accepted_events",
                    sum(
                        _event(row, sign) and row.weekend == weekend_value
                        for _, row in selected
                    ),
                    sign=sign,
                    weekend_group=weekend_name,
                )
            )
        for activity_name in ("low", "other", "unavailable"):
            output.append(
                _record(
                    scope_type,
                    scope,
                    "accepted_events",
                    sum(
                        _event(row, sign) and _activity_group(row) == activity_name
                        for _, row in selected
                    ),
                    sign=sign,
                    activity_group=activity_name,
                )
            )
        for weekend_value, weekend_name in ((True, "weekend"), (False, "weekday")):
            for activity_name in ("low", "other"):
                output.append(
                    _record(
                        scope_type,
                        scope,
                        "accepted_events",
                        sum(
                            _event(row, sign)
                            and row.weekend == weekend_value
                            and _activity_group(row) == activity_name
                            for _, row in selected
                        ),
                        sign=sign,
                        weekend_group=weekend_name,
                        activity_group=activity_name,
                    )
                )
        for horizon in horizons:
            output.append(
                _record(
                    scope_type,
                    scope,
                    "structurally_complete_events",
                    sum(_event(row, sign) and flags[index][horizon] for index, row in selected),
                    sign=sign,
                    horizon=horizon,
                )
            )

    h1_h2 = [
        (index, row)
        for index, row in selected
        if row.negative_event and flags[index][config.outcomes.primary_horizon_hours]
    ]
    h3_h4 = [
        (index, row) for index, row in h1_h2 if row.full_feature_eligible
    ]
    output.append(
        _record(
            scope_type,
            scope,
            "structurally_eligible_events",
            len(h1_h2),
            sign="negative",
            horizon=config.outcomes.primary_horizon_hours,
            model_or_strategy="H1_H2",
        )
    )
    output.append(
        _record(
            scope_type,
            scope,
            "structurally_eligible_events",
            len(h3_h4),
            sign="negative",
            horizon=config.outcomes.primary_horizon_hours,
            model_or_strategy="H3_H4",
        )
    )

    model3_cells: list[int] = []
    for weekend_value, weekend_name in ((True, "weekend"), (False, "weekday")):
        for low_value, activity_name in ((True, "low"), (False, "other")):
            cell_count = sum(
                row.weekend == weekend_value and row.low_activity is low_value
                for _, row in h3_h4
            )
            model3_cells.append(cell_count)
            output.append(
                _record(
                    scope_type,
                    scope,
                    "structurally_eligible_events",
                    cell_count,
                    sign="negative",
                    weekend_group=weekend_name,
                    activity_group=activity_name,
                    horizon=config.outcomes.primary_horizon_hours,
                    model_or_strategy="H3_H4",
                )
            )

    strategy_filters = {
        "all_shock": lambda row: True,
        "weekend": lambda row: row.weekend,
        "low_activity": lambda row: row.full_feature_eligible and row.low_activity is True,
        "combined": lambda row: (
            row.weekend and row.full_feature_eligible and row.low_activity is True
        ),
    }
    strategy_counts: dict[str, int] = {}
    for name in config.strategy.variants:
        eligible = sum(strategy_filters[name](row) for _, row in h1_h2)
        strategy_counts[name] = eligible
        output.append(
            _record(
                scope_type,
                scope,
                "structurally_eligible_events",
                eligible,
                sign="negative",
                horizon=config.strategy.hold_hours,
                model_or_strategy=name,
            )
        )

    if scope in ANALYSIS_PERIODS:
        output.append(
            _record(
                scope_type,
                scope,
                "model3_complete_case_events",
                len(h3_h4),
                sign="negative",
                horizon=config.outcomes.primary_horizon_hours,
                model_or_strategy="Model_3",
                label=model3_interpretation_label(model3_cells, len(h3_h4), config),
            )
        )
        combined = strategy_counts["combined"]
        output.append(
            _record(
                scope_type,
                scope,
                "combined_structurally_eligible_trades",
                combined,
                sign="negative",
                horizon=config.strategy.hold_hours,
                model_or_strategy="combined",
                label=combined_strategy_interpretation_label(combined, config),
            )
        )
    return output


def build_count_records(
    rows: Sequence[CountInputRow], config: ConfigNode
) -> tuple[CountRecord, ...]:
    """Build every feasibility count without reading a price."""
    horizons = tuple(
        sorted(
            {
                config.outcomes.primary_horizon_hours,
                *config.outcomes.secondary_horizon_hours,
            }
        )
    )
    flags = structural_horizon_flags(rows, horizons)
    records: list[CountRecord] = []
    for scope in (*ANALYSIS_PERIODS, "full_analysis"):
        records.extend(_scope_records(rows, flags, scope, config, horizons))

    years = sorted(
        {row.calendar_year for row in rows if row.analysis_period in ANALYSIS_PERIODS}
    )
    for year in years:
        selected = [
            row
            for row in rows
            if row.analysis_period in ANALYSIS_PERIODS and row.calendar_year == year
        ]
        for sign in ("negative", "positive"):
            records.append(
                _record(
                    "year",
                    str(year),
                    "candidates",
                    sum(_candidate(row, sign) for row in selected),
                    sign=sign,
                )
            )
            records.append(
                _record(
                    "year",
                    str(year),
                    "accepted_events",
                    sum(_event(row, sign) for row in selected),
                    sign=sign,
                )
            )
    return tuple(records)


def render_count_records(records: Sequence[CountRecord]) -> str:
    """Write the count records as CSV text."""
    if any(
        token in column for column in COUNT_OUTPUT_COLUMNS for token in _FORBIDDEN_COUNT_TOKENS
    ):
        raise CountAuditError("count-only output schema contains a forbidden field")
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(COUNT_OUTPUT_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "scope_type": record.scope_type,
                "scope": record.scope,
                "sign": record.sign,
                "weekend_group": record.weekend_group,
                "activity_group": record.activity_group,
                "horizon_hours": record.horizon_hours,
                "model_or_strategy": record.model_or_strategy,
                "measure": record.measure,
                "count": str(record.count),
                "denominator_measure": record.denominator_measure,
                "denominator_count": (
                    "" if record.denominator_count is None else str(record.denominator_count)
                ),
                "rate_per_1000": (
                    "" if record.rate_per_1000 is None else f"{record.rate_per_1000:.6f}"
                ),
                "interpretation_label": record.interpretation_label,
            }
        )
    return buffer.getvalue()


def verify_feature_input(
    config: ConfigNode,
    manifest_relative: Path,
    *,
    require_frozen_acceptance: bool = False,
) -> tuple[Path, int, dict[str, str]]:
    """Check the feature run and return its file path, row count, and input checksums."""
    if require_frozen_acceptance and manifest_relative != ACCEPTED_FEATURE_MANIFEST:
        raise CountAuditError(
            "default count-only execution must use the accepted corrected feature manifest"
        )
    verify_manifest_checksum(manifest_relative)
    manifest_digest = sha256_file(paths.resolve(manifest_relative))
    manifest = read_manifest(manifest_relative)
    if manifest["stage"] != "4. Pre-outcome features" or manifest["mode"] != "features":
        raise CountAuditError("upstream manifest is not Stage 4 feature evidence")
    if manifest["column_schema"] != list(FEATURE_COLUMNS):
        raise CountAuditError("upstream manifest does not declare the feature schema")
    if not matches_active_protocol(manifest, config):
        raise CountAuditError("upstream feature evidence does not match active specification/config")
    if manifest["code_identity"]["git_dirty"] is True:
        raise CountAuditError("upstream feature evidence was produced from a dirty repository")
    feature_relative = manifest_relative.parent / "pre_outcome_features.csv"
    feature_digest = sha256_file(paths.resolve(feature_relative))
    if feature_digest != manifest["output_checksum"]:
        raise CountAuditError("feature checksum does not match the Stage 4 manifest")
    if require_frozen_acceptance and (
        manifest_digest != ACCEPTED_FEATURE_MANIFEST_SHA256
        or feature_digest != ACCEPTED_FEATURE_OUTPUT_SHA256
    ):
        raise CountAuditError("accepted feature run checksums do not match the pinned values")
    sidecar_relative = manifest_relative.parent / f"{manifest_relative.name}.sha256"
    return (
        feature_relative,
        manifest["row_count"],
        dict(
            sorted(
                {
                    feature_relative.as_posix(): feature_digest,
                    manifest_relative.as_posix(): manifest_digest,
                    sidecar_relative.as_posix(): sha256_file(paths.resolve(sidecar_relative)),
                }.items()
            )
        ),
    )


def _record_failure(log: RunLog, error: Exception) -> None:
    try:
        log.record(
            "count_audit_failed", reason=type(error).__name__, detail=str(error)[:500]
        )
    except Exception:  # pragma: no cover - logging cannot replace the cause
        pass


def run_count_audit(
    config: ConfigNode | None = None,
    *,
    feature_manifest_relative: Path | None = None,
) -> CountAuditResult:
    """Run the count-only audit and save its outputs."""
    cfg = config if config is not None else load_config()
    upstream = (
        feature_manifest_relative
        if feature_manifest_relative is not None
        else ACCEPTED_FEATURE_MANIFEST
    )
    draft = build_manifest(STAGE, MODE, cfg)
    log = create_run_log(draft.run_id, config=cfg)
    log.record("count_audit_started", feature_manifest=upstream.as_posix())
    try:
        feature_relative, expected_rows, input_checksums = verify_feature_input(
            cfg,
            upstream,
            require_frozen_acceptance=feature_manifest_relative is None,
        )
        rows = read_count_rows(
            feature_relative,
            expected_rows,
            cooldown_hours=cfg.events.cooldown_hours,
        )
        records = build_count_records(rows, cfg)
        content = render_count_records(records)
        run_dir = run_relative_dir(draft.run_id, cfg)
        counts_relative = run_dir / "feasibility_counts.csv"
        create_exclusive(counts_relative, content)
        output_digest = sha256_bytes(content.encode("utf-8"))
        log.record(
            "count_artifact_written",
            path=counts_relative.as_posix(),
            records=len(records),
            sha256=output_digest,
            outcome_fields_present=False,
            structural_horizon_flags_used=True,
        )
        analysis_rows = sum(row.analysis_period in ANALYSIS_PERIODS for row in rows)
        warmup_rows = len(rows) - analysis_rows
        finished = replace(
            draft,
            input_checksums=input_checksums,
            exclusions={"rows_dropped": 0},
            eligibility_changes={
                "input_feature_rows": len(rows),
                "analysis_rows": analysis_rows,
                "warmup_rows_excluded_from_analysis": warmup_rows,
                "output_count_records": len(records),
                "accepted_negative_analysis_events": sum(
                    row.negative_event and row.analysis_period in ANALYSIS_PERIODS
                    for row in rows
                ),
                "accepted_positive_analysis_events": sum(
                    row.positive_event and row.analysis_period in ANALYSIS_PERIODS
                    for row in rows
                ),
            },
        ).completed(output_digest, len(records), list(COUNT_OUTPUT_COLUMNS))
        manifest_path = finished.write_for_run("count-audit", cfg)
        manifest_relative = manifest_path.relative_to(paths.project_root().resolve())
        sidecar_relative, manifest_digest = write_manifest_checksum(manifest_relative)
        log.record(
            "manifest_checksum_written",
            path=sidecar_relative.as_posix(),
            sha256=manifest_digest,
            manifest=manifest_relative.as_posix(),
        )
        log.record(
            "count_audit_completed",
            manifest=manifest_relative.as_posix(),
            records=len(records),
            outcome_fields_present=False,
        )
    except Exception as error:
        _record_failure(log, error)
        raise
    return CountAuditResult(
        run_id=draft.run_id,
        manifest_relative_path=manifest_relative,
        manifest_checksum_relative_path=sidecar_relative,
        log_relative_path=log.relative_path,
        counts_relative_path=counts_relative,
        input_row_count=len(rows),
        output_record_count=len(records),
    )


def summarise(result: CountAuditResult) -> list[str]:
    return [
        f"run id                  {result.run_id}",
        f"input feature rows      {result.input_row_count}",
        f"count records           {result.output_record_count}",
        f"feasibility artifact    {result.counts_relative_path.as_posix()}",
        f"manifest                {result.manifest_relative_path.as_posix()}",
        f"manifest checksum       {result.manifest_checksum_relative_path.as_posix()}",
        f"run log                 {result.log_relative_path.as_posix()}",
        "future price values     not parsed, used, or computed",
    ]
