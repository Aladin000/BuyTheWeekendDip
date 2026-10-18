"""Feature stage: past-only features and events.

Reads the clean hourly file and keeps every row. Rolling windows restart at
each gap and leave out the current hour where a lag is required. After events
are fixed, a timestamp-only flag records whether the 12-hour window reaches
Monday, set only when that window is complete and has no gap. The output has
no future price, outcome, trade, P&L or performance field.
"""

from __future__ import annotations

import bisect
import csv
import io
import math
from collections import deque
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
from src.config import ConfigNode, is_low_activity, load_config
from src.events import EventDecision, construct_sign_specific_events
from src.validation import CLEAN_COLUMNS

STAGE = "4. Pre-outcome features"
MODE = "features"

DERIVED_COLUMNS: tuple[str, ...] = (
    "log_return_1h",
    "simple_return_1h",
    "rolling_mean_720h",
    "rolling_std_720h",
    "shock_zscore",
    "shock_eligible_hour",
    "shock_ineligibility_reason",
    "negative_candidate",
    "positive_candidate",
    "quote_activity_24h",
    "trade_activity_24h",
    "activity_percentile_90d",
    "low_activity_score",
    "low_activity",
    "trade_activity_percentile_90d",
    "trade_low_activity_score",
    "trade_low_activity",
    "activity_reference_count",
    "rv24",
    "full_feature_eligible_hour",
    "full_feature_ineligibility_reason",
    "shock_magnitude",
    "absolute_shock_z",
    "utc_weekday",
    "weekend",
    "utc_hour",
    "utc_time_block",
    "analysis_period",
    "calendar_year",
    "calendar_month",
    "negative_event",
    "positive_event",
    "cooldown_accepted",
    "event_id",
    "event_rejection_reason",
    "crosses_into_monday_12h",
)

FEATURE_COLUMNS: tuple[str, ...] = CLEAN_COLUMNS + DERIVED_COLUMNS

_FORBIDDEN_OUTPUT_TOKENS: tuple[str, ...] = (
    "post_",
    "forward",
    "outcome",
    "exit_price",
    "entry_price",
    "pnl",
    "profit",
    "strategy_return",
)


class FeatureError(RuntimeError):
    """Raised when the feature stage cannot build a valid past-only file."""


@dataclass(frozen=True)
class CleanInputRow:
    """Parsed fields of one clean row, plus its unchanged CSV cells."""

    cells: tuple[str, ...]
    open_time: datetime
    close_time: datetime
    close_time_raw: int
    close: Decimal
    quote_volume: Decimal
    trade_count: Decimal
    segment_id: int
    has_previous_contiguous_hour: bool


@dataclass(frozen=True)
class FeatureSet:
    """One aligned tuple per derived numerical or event field."""

    log_returns: tuple[float | None, ...]
    simple_returns: tuple[float | None, ...]
    rolling_means: tuple[float | None, ...]
    rolling_stds: tuple[float | None, ...]
    shock_zscores: tuple[float | None, ...]
    shock_eligible: tuple[bool, ...]
    shock_reasons: tuple[str, ...]
    quote_activity: tuple[Decimal | None, ...]
    trade_activity: tuple[Decimal | None, ...]
    quote_percentile_numerators: tuple[int | None, ...]
    trade_percentile_numerators: tuple[int | None, ...]
    activity_reference_counts: tuple[int, ...]
    rv24: tuple[float | None, ...]
    full_feature_eligible: tuple[bool, ...]
    full_feature_reasons: tuple[str, ...]
    events: tuple[EventDecision, ...]


@dataclass(frozen=True)
class FeatureResult:
    run_id: str
    manifest_relative_path: Path
    manifest_checksum_relative_path: Path
    log_relative_path: Path
    feature_relative_path: Path
    row_count: int


def _parse_bool(text: str, field: str, source_row: int) -> bool:
    if text == "true":
        return True
    if text == "false":
        return False
    raise FeatureError(f"clean row {source_row} field {field} is not true or false")


def _parse_utc(text: str, field: str, source_row: int) -> datetime:
    if not text.endswith("Z"):
        raise FeatureError(f"clean row {source_row} field {field} is not UTC")
    try:
        value = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        raise FeatureError(
            f"clean row {source_row} field {field} is not an ISO UTC timestamp"
        ) from None
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise FeatureError(f"clean row {source_row} field {field} is not UTC")
    return value.astimezone(timezone.utc)


def _parse_decimal(text: str, field: str, source_row: int) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise FeatureError(
            f"clean row {source_row} field {field} is not numeric"
        ) from None
    if not value.is_finite():
        raise FeatureError(f"clean row {source_row} field {field} is not finite")
    return value


def read_clean_rows(relative_path: Path, expected_rows: int) -> list[CleanInputRow]:
    """Read the checksum-verified clean hourly file and recheck its structure."""
    source = paths.require_file(paths.resolve(relative_path))
    rows: list[CleanInputRow] = []
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = tuple(next(reader))
            except StopIteration:
                raise FeatureError("clean hourly artifact is empty") from None
            if header != CLEAN_COLUMNS:
                raise FeatureError("clean hourly artifact has unexpected columns")
            index = {name: header.index(name) for name in header}
            for source_row, cells in enumerate(reader, start=2):
                if len(cells) != len(header):
                    raise FeatureError(
                        f"clean row {source_row} has {len(cells)} fields; expected {len(header)}"
                    )
                open_time = _parse_utc(cells[index["open_time"]], "open_time", source_row)
                close_time = _parse_utc(cells[index["close_time"]], "close_time", source_row)
                try:
                    close_time_raw = int(cells[index["close_time_raw"]])
                    segment_id = int(cells[index["contiguous_segment_id"]])
                except ValueError:
                    raise FeatureError(
                        f"clean row {source_row} has a non-integer timestamp or segment"
                    ) from None
                close = _parse_decimal(cells[index["close"]], "close", source_row)
                quote_volume = _parse_decimal(
                    cells[index["quote_volume"]], "quote_volume", source_row
                )
                trade_count = _parse_decimal(
                    cells[index["trade_count"]], "trade_count", source_row
                )
                if close <= 0 or quote_volume < 0 or trade_count < 0:
                    raise FeatureError(f"clean row {source_row} has a value out of range")
                if trade_count != trade_count.to_integral_value():
                    raise FeatureError(f"clean row {source_row} trade_count is not an integer")
                rows.append(
                    CleanInputRow(
                        cells=tuple(cells),
                        open_time=open_time,
                        close_time=close_time,
                        close_time_raw=close_time_raw,
                        close=close,
                        quote_volume=quote_volume,
                        trade_count=trade_count,
                        segment_id=segment_id,
                        has_previous_contiguous_hour=_parse_bool(
                            cells[index["has_previous_contiguous_hour"]],
                            "has_previous_contiguous_hour",
                            source_row,
                        ),
                    )
                )
    except (UnicodeDecodeError, csv.Error) as error:
        raise FeatureError(f"could not parse clean hourly artifact: {error}") from None
    if len(rows) != expected_rows:
        raise FeatureError(
            f"clean hourly row count {len(rows)} does not match manifest {expected_rows}"
        )
    _validate_clean_order(rows)
    return rows


def _validate_clean_order(rows: Sequence[CleanInputRow]) -> None:
    if not rows:
        raise FeatureError("clean hourly artifact contains no data rows")
    if rows[0].segment_id != 1 or rows[0].has_previous_contiguous_hour:
        raise FeatureError("the first clean row must start contiguous segment 1")
    for previous, current in zip(rows, rows[1:]):
        if current.open_time <= previous.open_time:
            raise FeatureError("clean hourly timestamps are not strictly increasing")
        same_segment = current.segment_id == previous.segment_id
        if same_segment:
            if (
                current.open_time - previous.open_time != timedelta(hours=1)
                or not current.has_previous_contiguous_hour
            ):
                raise FeatureError("clean contiguous-segment annotations are inconsistent")
        elif (
            current.segment_id != previous.segment_id + 1
            or current.has_previous_contiguous_hour
            or current.open_time - previous.open_time <= timedelta(hours=1)
        ):
            raise FeatureError("clean segment-boundary annotations are inconsistent")


def compute_log_returns(
    closes: Sequence[Decimal], segment_ids: Sequence[int]
) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
    """Log and simple close-to-close returns; none across a gap."""
    if len(closes) != len(segment_ids):
        raise FeatureError("close and segment inputs have different lengths")
    log_returns: list[float | None] = [None] * len(closes)
    simple_returns: list[float | None] = [None] * len(closes)
    for index in range(1, len(closes)):
        if segment_ids[index] != segment_ids[index - 1]:
            continue
        if closes[index] <= 0 or closes[index - 1] <= 0:
            raise FeatureError("return calculation requires positive closes")
        log_value = float(closes[index].ln() - closes[index - 1].ln())
        simple_value = float(closes[index] / closes[index - 1] - Decimal(1))
        if not math.isfinite(log_value) or not math.isfinite(simple_value):
            raise FeatureError("return calculation produced a non-finite value")
        log_returns[index] = log_value
        simple_returns[index] = simple_value
    return tuple(log_returns), tuple(simple_returns)


def lagged_rolling_mean_std(
    values: Sequence[float | None],
    segment_ids: Sequence[int],
    *,
    window: int,
    ddof: int,
) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
    """Two-pass mean and standard deviation over the previous ``window`` values."""
    if len(values) != len(segment_ids):
        raise FeatureError("rolling-statistic inputs have different lengths")
    if window <= ddof or ddof < 0:
        raise FeatureError("rolling-statistic window and ddof are invalid")
    means: list[float | None] = [None] * len(values)
    stds: list[float | None] = [None] * len(values)
    history: deque[float] = deque(maxlen=window)
    previous_segment: int | None = None
    for index, (value, segment) in enumerate(zip(values, segment_ids)):
        if segment != previous_segment:
            history.clear()
            previous_segment = segment
        if value is not None and len(history) == window:
            mean = math.fsum(history) / window
            variance = math.fsum((item - mean) ** 2 for item in history) / (window - ddof)
            means[index] = mean
            stds[index] = math.sqrt(variance)
        if value is None:
            history.clear()
        else:
            history.append(value)
    return tuple(means), tuple(stds)


def standardise_current_returns(
    returns: Sequence[float | None],
    means: Sequence[float | None],
    stds: Sequence[float | None],
) -> tuple[tuple[float | None, ...], tuple[bool, ...], tuple[str, ...]]:
    """Shock z-scores, with a reason for each hour that has none."""
    if not (len(returns) == len(means) == len(stds)):
        raise FeatureError("shock-standardisation inputs have different lengths")
    zscores: list[float | None] = []
    eligible: list[bool] = []
    reasons: list[str] = []
    for value, mean, std in zip(returns, means, stds):
        if value is None:
            zscores.append(None)
            eligible.append(False)
            reasons.append("no_contiguous_current_return")
        elif mean is None or std is None:
            zscores.append(None)
            eligible.append(False)
            reasons.append("insufficient_contiguous_shock_history")
        elif std == 0:
            zscores.append(None)
            eligible.append(False)
            reasons.append("zero_shock_window_variance")
        else:
            z = (value - mean) / std
            if not math.isfinite(z):
                raise FeatureError("shock z-score is non-finite")
            zscores.append(z)
            eligible.append(True)
            reasons.append("")
    return tuple(zscores), tuple(eligible), tuple(reasons)


def lagged_decimal_sums(
    values: Sequence[Decimal], segment_ids: Sequence[int], *, window: int
) -> tuple[Decimal | None, ...]:
    """Exact sum of the previous ``window`` values in the same segment."""
    if len(values) != len(segment_ids) or window <= 0:
        raise FeatureError("rolling-sum inputs or window are invalid")
    output: list[Decimal | None] = [None] * len(values)
    history: deque[Decimal] = deque()
    total = Decimal(0)
    previous_segment: int | None = None
    for index, (value, segment) in enumerate(zip(values, segment_ids)):
        if segment != previous_segment:
            history.clear()
            total = Decimal(0)
            previous_segment = segment
        if len(history) == window:
            output[index] = total
        history.append(value)
        total += value
        if len(history) > window:
            total -= history.popleft()
    return tuple(output)


def lagged_percentile_numerators(
    values: Sequence[Decimal | None],
    segment_ids: Sequence[int],
    *,
    window: int,
) -> tuple[tuple[int | None, ...], tuple[int, ...]]:
    """Count previous values at or below the current one; windows stop at gaps."""
    if len(values) != len(segment_ids) or window <= 0:
        raise FeatureError("percentile inputs or window are invalid")
    numerators: list[int | None] = [None] * len(values)
    reference_counts: list[int] = [0] * len(values)
    history: deque[Decimal] = deque()
    ordered: list[Decimal] = []
    previous_segment: int | None = None
    for index, (value, segment) in enumerate(zip(values, segment_ids)):
        if segment != previous_segment:
            history.clear()
            ordered.clear()
            previous_segment = segment
        reference_counts[index] = len(history)
        if value is not None and len(history) == window:
            numerators[index] = bisect.bisect_right(ordered, value)
        if value is None:
            history.clear()
            ordered.clear()
            continue
        history.append(value)
        bisect.insort_right(ordered, value)
        if len(history) > window:
            removed = history.popleft()
            position = bisect.bisect_left(ordered, removed)
            if position == len(ordered) or ordered[position] != removed:
                raise FeatureError("rolling percentile state lost a reference value")
            ordered.pop(position)
    return tuple(numerators), tuple(reference_counts)


def lagged_rv(
    returns: Sequence[float | None], segment_ids: Sequence[int], *, window: int
) -> tuple[float | None, ...]:
    """Square root of the sum of squared returns over the previous window."""
    if len(returns) != len(segment_ids) or window <= 0:
        raise FeatureError("RV inputs or window are invalid")
    output: list[float | None] = [None] * len(returns)
    history: deque[float] = deque()
    total = 0.0
    previous_segment: int | None = None
    for index, (value, segment) in enumerate(zip(returns, segment_ids)):
        if segment != previous_segment:
            history.clear()
            total = 0.0
            previous_segment = segment
        if value is not None and len(history) == window:
            output[index] = math.sqrt(max(total, 0.0))
        if value is None:
            history.clear()
            total = 0.0
            continue
        squared = value * value
        history.append(squared)
        total += squared
        if len(history) > window:
            total -= history.popleft()
    return tuple(output)


def compute_feature_set(
    rows: Sequence[CleanInputRow], config: ConfigNode
) -> FeatureSet:
    """Compute every feature field from current and past data only."""
    closes = [row.close for row in rows]
    segments = [row.segment_id for row in rows]
    log_returns, simple_returns = compute_log_returns(closes, segments)
    rolling_means, rolling_stds = lagged_rolling_mean_std(
        log_returns,
        segments,
        window=config.shock.window_hours,
        ddof=config.shock.ddof,
    )
    zscores, shock_eligible, shock_reasons = standardise_current_returns(
        log_returns, rolling_means, rolling_stds
    )

    quote_activity = lagged_decimal_sums(
        [row.quote_volume for row in rows],
        segments,
        window=config.activity.window_hours,
    )
    trade_activity = lagged_decimal_sums(
        [row.trade_count for row in rows],
        segments,
        window=config.activity.window_hours,
    )
    quote_numerators, reference_counts = lagged_percentile_numerators(
        quote_activity,
        segments,
        window=config.activity.reference_window,
    )
    trade_numerators, trade_reference_counts = lagged_percentile_numerators(
        trade_activity,
        segments,
        window=config.activity.reference_window,
    )
    if reference_counts != trade_reference_counts:
        raise FeatureError("quote-volume and trade-count reference windows disagree")
    rv24 = lagged_rv(log_returns, segments, window=config.controls.rv_window_hours)

    full_eligible: list[bool] = []
    full_reasons: list[str] = []
    for shock_ok, shock_reason, numerator, rv in zip(
        shock_eligible, shock_reasons, quote_numerators, rv24
    ):
        reasons: list[str] = []
        if not shock_ok:
            reasons.append(shock_reason)
        if numerator is None:
            reasons.append("insufficient_contiguous_activity_history")
        if rv is None:
            reasons.append("insufficient_contiguous_rv_history")
        full_eligible.append(not reasons)
        full_reasons.append("|".join(reasons))

    events = construct_sign_specific_events(
        zscores,
        [row.close_time for row in rows],
        [row.close_time_raw for row in rows],
        negative_threshold=config.shock.negative_threshold,
        positive_threshold=config.shock.positive_threshold,
        cooldown_hours=config.events.cooldown_hours,
    )
    return FeatureSet(
        log_returns=log_returns,
        simple_returns=simple_returns,
        rolling_means=rolling_means,
        rolling_stds=rolling_stds,
        shock_zscores=zscores,
        shock_eligible=shock_eligible,
        shock_reasons=shock_reasons,
        quote_activity=quote_activity,
        trade_activity=trade_activity,
        quote_percentile_numerators=quote_numerators,
        trade_percentile_numerators=trade_numerators,
        activity_reference_counts=reference_counts,
        rv24=rv24,
        full_feature_eligible=tuple(full_eligible),
        full_feature_reasons=tuple(full_reasons),
        events=events,
    )


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        raise FeatureError("refusing to render a non-finite feature")
    return format(value, ".17g")


def _format_decimal(value: Decimal | None) -> str:
    return "" if value is None else format(value, "f")


def _analysis_period(moment: datetime, config: ConfigNode) -> str:
    for name in ("warmup", "development", "held_out"):
        split = config.splits[name]
        if split.start_inclusive <= moment < split.end_exclusive:
            return name
    raise FeatureError(f"timestamp {moment.isoformat()} belongs to no configured split")


def _time_block(hour: int, config: ConfigNode) -> str:
    matches = [
        block.name
        for block in config.calendar.time_blocks
        if block.start_hour <= hour <= block.end_hour
    ]
    if len(matches) != 1:
        raise FeatureError(f"UTC hour {hour} belongs to {len(matches)} time blocks")
    return matches[0]


def _crosses_into_monday(
    rows: Sequence[CleanInputRow],
    index: int,
    event: EventDecision,
    config: ConfigNode,
) -> str:
    """Classify a weekend event only when its 12-hour path is complete.

    Uses timestamps and segment IDs only. It runs after the cooldown, so it
    never reads a future price or changes which events exist.
    """
    row = rows[index]
    if (
        row.close_time.isoweekday() not in config.calendar.weekend_iso_weekdays
        or not (event.negative_event or event.positive_event)
    ):
        return ""
    horizon = config.outcomes.primary_horizon_hours
    future = rows[index + 1 : index + horizon + 1]
    if len(future) != horizon or any(
        future_row.segment_id != row.segment_id for future_row in future
    ):
        return ""
    return str(
        any(
            future_row.close_time.isoweekday() == config.calendar.monday_iso_weekday
            for future_row in future
        )
    ).lower()


def render_feature_rows(
    rows: Sequence[CleanInputRow], feature_set: FeatureSet, config: ConfigNode
) -> str:
    """Render the pre-outcome feature CSV."""
    lengths = {
        len(feature_set.log_returns),
        len(feature_set.simple_returns),
        len(feature_set.rolling_means),
        len(feature_set.rolling_stds),
        len(feature_set.shock_zscores),
        len(feature_set.shock_eligible),
        len(feature_set.shock_reasons),
        len(feature_set.quote_activity),
        len(feature_set.trade_activity),
        len(feature_set.quote_percentile_numerators),
        len(feature_set.trade_percentile_numerators),
        len(feature_set.activity_reference_counts),
        len(feature_set.rv24),
        len(feature_set.full_feature_eligible),
        len(feature_set.full_feature_reasons),
        len(feature_set.events),
    }
    if lengths != {len(rows)}:
        raise FeatureError("feature arrays do not align with clean rows")
    if any(
        token in column
        for column in FEATURE_COLUMNS
        for token in _FORBIDDEN_OUTPUT_TOKENS
    ):
        raise FeatureError("pre-outcome schema contains a forbidden analytical field")

    reference_window = config.activity.reference_window
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(FEATURE_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    for index, row in enumerate(rows):
        event = feature_set.events[index]
        quote_numerator = feature_set.quote_percentile_numerators[index]
        trade_numerator = feature_set.trade_percentile_numerators[index]
        quote_percentile = (
            quote_numerator / reference_window if quote_numerator is not None else None
        )
        trade_percentile = (
            trade_numerator / reference_window if trade_numerator is not None else None
        )
        low_activity = (
            is_low_activity(quote_numerator, reference_window, config)
            if quote_numerator is not None
            else None
        )
        trade_low_activity = (
            is_low_activity(trade_numerator, reference_window, config)
            if trade_numerator is not None
            else None
        )
        weekday = row.close_time.isoweekday()
        weekend = weekday in set(config.calendar.weekend_iso_weekdays)
        zscore = feature_set.shock_zscores[index]
        record = dict(zip(CLEAN_COLUMNS, row.cells))
        record.update(
            {
                "log_return_1h": _format_float(feature_set.log_returns[index]),
                "simple_return_1h": _format_float(feature_set.simple_returns[index]),
                "rolling_mean_720h": _format_float(feature_set.rolling_means[index]),
                "rolling_std_720h": _format_float(feature_set.rolling_stds[index]),
                "shock_zscore": _format_float(zscore),
                "shock_eligible_hour": str(feature_set.shock_eligible[index]).lower(),
                "shock_ineligibility_reason": feature_set.shock_reasons[index],
                "negative_candidate": str(event.negative_candidate).lower(),
                "positive_candidate": str(event.positive_candidate).lower(),
                "quote_activity_24h": _format_decimal(
                    feature_set.quote_activity[index]
                ),
                "trade_activity_24h": _format_decimal(
                    feature_set.trade_activity[index]
                ),
                "activity_percentile_90d": _format_float(quote_percentile),
                "low_activity_score": _format_float(
                    None if quote_percentile is None else 1.0 - quote_percentile
                ),
                "low_activity": "" if low_activity is None else str(low_activity).lower(),
                "trade_activity_percentile_90d": _format_float(trade_percentile),
                "trade_low_activity_score": _format_float(
                    None if trade_percentile is None else 1.0 - trade_percentile
                ),
                "trade_low_activity": (
                    "" if trade_low_activity is None else str(trade_low_activity).lower()
                ),
                "activity_reference_count": feature_set.activity_reference_counts[index],
                "rv24": _format_float(feature_set.rv24[index]),
                "full_feature_eligible_hour": str(
                    feature_set.full_feature_eligible[index]
                ).lower(),
                "full_feature_ineligibility_reason": (
                    feature_set.full_feature_reasons[index]
                ),
                "shock_magnitude": _format_float(
                    -zscore if event.negative_event and zscore is not None else None
                ),
                "absolute_shock_z": _format_float(
                    abs(zscore)
                    if (event.negative_event or event.positive_event)
                    and zscore is not None
                    else None
                ),
                "utc_weekday": weekday,
                "weekend": str(weekend).lower(),
                "utc_hour": row.close_time.hour,
                "utc_time_block": _time_block(row.close_time.hour, config),
                "analysis_period": _analysis_period(row.close_time, config),
                "calendar_year": row.close_time.year,
                "calendar_month": row.close_time.strftime("%Y-%m"),
                "negative_event": str(event.negative_event).lower(),
                "positive_event": str(event.positive_event).lower(),
                "cooldown_accepted": str(event.cooldown_accepted).lower(),
                "event_id": event.event_id,
                "event_rejection_reason": event.event_rejection_reason,
                "crosses_into_monday_12h": _crosses_into_monday(
                    rows, index, event, config
                ),
            }
        )
        writer.writerow(record)
    return buffer.getvalue()


def find_validation_manifest(config: ConfigNode) -> Path:
    root = paths.resolve(config.paths.outputs_runs)
    candidates = sorted(
        path.relative_to(paths.project_root().resolve())
        for path in root.glob("*/validation.manifest.json")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise FeatureError(
            f"expected exactly one accepted validation manifest, found {len(candidates)}; "
            "pass the intended manifest explicitly"
        )
    return candidates[0]


def verify_validation_input(
    config: ConfigNode, manifest_relative: Path
) -> tuple[Path, int, dict[str, str]]:
    """Check the validation run and return the clean file path, row count and checksums."""
    verify_manifest_checksum(manifest_relative)
    manifest = read_manifest(manifest_relative)
    if manifest["stage"] != "3. Validation and cleaning" or manifest["mode"] != "validate":
        raise FeatureError("upstream manifest is not from the validation stage")
    if manifest["column_schema"] != list(CLEAN_COLUMNS):
        raise FeatureError("upstream manifest does not declare the clean schema")
    if not matches_active_protocol(manifest, config):
        raise FeatureError("upstream validation evidence does not match active specification/config")
    if manifest["code_identity"]["git_dirty"] is True:
        raise FeatureError("upstream validation evidence was produced from a dirty repository")
    clean_relative = manifest_relative.parent / "clean_hourly.csv"
    clean_digest = sha256_file(paths.resolve(clean_relative))
    if clean_digest != manifest["output_checksum"]:
        raise FeatureError("clean hourly checksum does not match validation manifest")
    sidecar_relative = manifest_relative.parent / f"{manifest_relative.name}.sha256"
    return (
        clean_relative,
        manifest["row_count"],
        dict(
            sorted(
                {
                    clean_relative.as_posix(): clean_digest,
                    manifest_relative.as_posix(): sha256_file(
                        paths.resolve(manifest_relative)
                    ),
                    sidecar_relative.as_posix(): sha256_file(
                        paths.resolve(sidecar_relative)
                    ),
                }.items()
            )
        ),
    )


def _record_failure(log: RunLog, error: Exception) -> None:
    try:
        log.record(
            "features_failed", reason=type(error).__name__, detail=str(error)[:500]
        )
    except Exception:  # pragma: no cover - logging cannot replace the cause
        pass


def run_features(
    config: ConfigNode | None = None,
    *,
    validation_manifest_relative: Path | None = None,
) -> FeatureResult:
    """Build the pre-outcome feature file and its manifest."""
    cfg = config if config is not None else load_config()
    upstream = (
        validation_manifest_relative
        if validation_manifest_relative is not None
        else find_validation_manifest(cfg)
    )
    draft = build_manifest(STAGE, MODE, cfg)
    log = create_run_log(draft.run_id, config=cfg)
    log.record("features_started", validation_manifest=upstream.as_posix())
    try:
        clean_relative, expected_rows, input_checksums = verify_validation_input(
            cfg, upstream
        )
        rows = read_clean_rows(clean_relative, expected_rows)
        feature_set = compute_feature_set(rows, cfg)
        content = render_feature_rows(rows, feature_set, cfg)
        run_dir = run_relative_dir(draft.run_id, cfg)
        feature_relative = run_dir / "pre_outcome_features.csv"
        create_exclusive(feature_relative, content)
        output_digest = sha256_bytes(content.encode("utf-8"))
        log.record(
            "feature_artifact_written",
            path=feature_relative.as_posix(),
            rows=len(rows),
            sha256=output_digest,
        )
        finished = replace(
            draft,
            input_checksums=input_checksums,
            exclusions={"rows_dropped": 0},
            eligibility_changes={
                "input_clean_rows": len(rows),
                "output_feature_rows": len(rows),
                "valid_return_rows": sum(
                    value is not None for value in feature_set.log_returns
                ),
                "shock_eligible_rows": sum(feature_set.shock_eligible),
                "full_feature_eligible_rows": sum(
                    feature_set.full_feature_eligible
                ),
            },
        ).completed(output_digest, len(rows), list(FEATURE_COLUMNS))
        manifest_path = finished.write_for_run("features", cfg)
        manifest_relative = manifest_path.relative_to(paths.project_root().resolve())
        sidecar_relative, manifest_digest = write_manifest_checksum(manifest_relative)
        log.record(
            "manifest_checksum_written",
            path=sidecar_relative.as_posix(),
            sha256=manifest_digest,
            manifest=manifest_relative.as_posix(),
        )
        log.record(
            "features_completed",
            manifest=manifest_relative.as_posix(),
            rows=len(rows),
        )
    except Exception as error:
        _record_failure(log, error)
        raise
    return FeatureResult(
        run_id=draft.run_id,
        manifest_relative_path=manifest_relative,
        manifest_checksum_relative_path=sidecar_relative,
        log_relative_path=log.relative_path,
        feature_relative_path=feature_relative,
        row_count=len(rows),
    )


def summarise(result: FeatureResult) -> list[str]:
    return [
        f"run id                  {result.run_id}",
        f"feature rows            {result.row_count}",
        f"pre-outcome artifact    {result.feature_relative_path.as_posix()}",
        f"manifest                {result.manifest_relative_path.as_posix()}",
        f"manifest checksum       {result.manifest_checksum_relative_path.as_posix()}",
        f"run log                 {result.log_relative_path.as_posix()}",
        "forward outcomes        not computed",
    ]
