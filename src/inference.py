"""Statistical inference for the development and held-out stages.

Inputs are the checksum-verified outcome run and, for two uses only, the
accepted feature file: the price/event figure and the H5 any-sign cooldown
diagnostic, whose cooldown starts from warm-up z-scores. The development run
never reads held-out fields or prices.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import paths
from src.acquisition import verify_manifest_checksum, write_manifest_checksum
from src.artifacts import (
    RunLog,
    build_manifest,
    matches_active_protocol,
    create_exclusive,
    create_exclusive_bytes,
    create_run_log,
    read_manifest,
    run_relative_dir,
    sha256_bytes,
    sha256_file,
)
from src.config import ConfigNode, load_config, source_timestamp_unit
from src.count_audit import ACCEPTED_FEATURE_MANIFEST, verify_feature_input
from src.events import construct_any_sign_events
from src.features import FEATURE_COLUMNS
from src.outcomes import (
    DRIFT_OUTPUT_COLUMNS,
    EVENT_OUTPUT_COLUMNS,
    PATH_OUTPUT_COLUMNS,
    DevelopmentOutcomeBundle,
    EventOutcome,
    OutcomeInputRow,
    build_outcomes,
    read_outcome_rows,
    verify_development_inventory,
)
from src.rng import make_rng

STAGE = "7. Development inference"
MODE = "development"
DEVELOPMENT_PERIOD = "development"
HELD_OUT_PERIOD = "held_out"

ACCEPTED_OUTCOME_MANIFEST = Path(
    "outputs/runs/20260907T231307.800702Z-development-5b472caa-33cd7d/"
    "development-outcomes.manifest.json"
)
ACCEPTED_OUTCOME_MANIFEST_SHA256 = (
    "1e8cb8100c374cd56aef596f03f8b908198c3ad55d34a1d57a85f26603c032a7"
)

TIME_BLOCKS: tuple[str, ...] = ("00-05", "06-11", "12-17", "18-23")
TIME_BLOCK_REFERENCE = "00-05"
TIME_BLOCK_TERMS: tuple[str, ...] = (
    "time_block_06_11",
    "time_block_12_17",
    "time_block_18_23",
)

DESCRIPTIVE_COLUMNS: tuple[str, ...] = (
    "section",
    "comparison",
    "group",
    "variable",
    "statistic",
    "value",
    "n",
    "status",
)
GROUP_COMPARISON_COLUMNS: tuple[str, ...] = (
    "comparison",
    "group_1",
    "group_0",
    "n_1",
    "n_0",
    "difference_1_minus_0",
    "hc3_standard_error",
    "residual_df",
    "t_statistic",
    "ci_95_low",
    "ci_95_high",
    "two_sided_p_value",
    "status",
)
PATH_SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon_step_hours",
    "event_count",
    "mean_cumulative_log_return",
    "median_cumulative_log_return",
    "standard_deviation",
    "q05",
    "q25",
    "q75",
    "q95",
)
MODEL_RESULT_COLUMNS: tuple[str, ...] = (
    "analysis",
    "model",
    "sample",
    "term",
    "hypothesis",
    "n",
    "parameters",
    "residual_df",
    "estimate",
    "hc3_standard_error",
    "t_statistic",
    "ci_95_low",
    "ci_95_high",
    "raw_p_value",
    "holm_adjusted_p_value",
    "test_side",
    "status",
    "interpretation",
)
BOOTSTRAP_COLUMNS: tuple[str, ...] = (
    "analysis",
    "model",
    "term",
    "sample_n",
    "event_months",
    "target_valid_replicates",
    "maximum_attempts",
    "attempted",
    "valid",
    "empty",
    "rank_deficient",
    "percentile_95_low",
    "percentile_95_high",
    "quantile_method",
    "rng_stream",
    "status",
)
DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "model",
    "diagnostic",
    "term",
    "value",
    "minimum",
    "maximum",
    "count",
    "status",
    "details",
)
INVENTORY_COLUMNS: tuple[str, ...] = (
    "artifact_path",
    "sha256",
    "record_count",
    "artifact_type",
    "column_schema_json",
)

TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "development_descriptives.csv": DESCRIPTIVE_COLUMNS,
    "development_group_comparisons.csv": GROUP_COMPARISON_COLUMNS,
    "development_average_path.csv": PATH_SUMMARY_COLUMNS,
    "development_model_results.csv": MODEL_RESULT_COLUMNS,
    "development_bootstrap_results.csv": BOOTSTRAP_COLUMNS,
    "development_diagnostics.csv": DIAGNOSTIC_COLUMNS,
}


class InferenceError(RuntimeError):
    """Raised when inference inputs or statistical steps are invalid."""


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    sign: str
    signal_open_time: datetime
    signal_close_time: datetime
    signal_close: float
    segment_id: int
    calendar_year: int
    calendar_month: str
    weekend: bool
    crosses_into_monday_12h: bool | None
    time_block: str
    shock_return: float
    shock_zscore: float
    shock_magnitude: float | None
    absolute_shock_z: float
    activity_percentile: float | None
    low_activity_score: float | None
    low_activity: bool | None
    rv24: float | None
    full_feature_eligible: bool
    complete_12h: bool
    complete_24h: bool
    outcome_12h: float | None
    post_positive_12h: bool | None
    eligible_h1_h2: bool
    eligible_h3_h4: bool


@dataclass(frozen=True)
class PathRecord:
    event_id: str
    close_time: datetime
    step: int
    value: float


@dataclass(frozen=True)
class DriftRecord:
    reference_id: str
    calendar_month: str
    value: float


@dataclass(frozen=True)
class HourCandidate:
    analysis_period: str
    open_time: datetime
    close_time: datetime
    close_time_raw: int
    shock_eligible: bool
    zscore: float | None


@dataclass(frozen=True)
class ModelData:
    name: str
    sample: str
    terms: tuple[str, ...]
    rows: tuple[EventRecord, ...]
    y: np.ndarray
    x: np.ndarray


@dataclass(frozen=True)
class FitResult:
    data: ModelData
    status: str
    reason: str
    estimates: np.ndarray | None = None
    standard_errors: np.ndarray | None = None
    t_values: np.ndarray | None = None
    ci_low: np.ndarray | None = None
    ci_high: np.ndarray | None = None
    two_sided_p: np.ndarray | None = None
    df_resid: float | None = None
    residuals: np.ndarray | None = None
    cooks_distance: np.ndarray | None = None

    def term_index(self, term: str) -> int:
        try:
            return self.data.terms.index(term)
        except ValueError:
            raise InferenceError(f"model {self.data.name} has no term {term!r}") from None

    def coefficient(self, term: str) -> float | None:
        if self.estimates is None:
            return None
        return float(self.estimates[self.term_index(term)])


@dataclass(frozen=True)
class BootstrapResult:
    analysis: str
    model: str
    term: str
    sample_n: int
    event_months: int
    target: int
    maximum_attempts: int
    attempted: int
    valid: int
    empty: int
    rank_deficient: int
    low: float | None
    high: float | None
    stream: str
    status: str


@dataclass(frozen=True)
class InferenceBundle:
    descriptives: tuple[dict[str, object], ...]
    group_comparisons: tuple[dict[str, object], ...]
    path_summary: tuple[dict[str, object], ...]
    model_results: tuple[dict[str, object], ...]
    bootstrap_results: tuple[BootstrapResult, ...]
    diagnostics: tuple[dict[str, object], ...]
    figures: tuple[tuple[str, bytes], ...]
    event_rows: int
    h5_any_sign_events: int


@dataclass(frozen=True)
class DevelopmentInferenceResult:
    run_id: str
    manifest_relative_path: Path
    manifest_checksum_relative_path: Path
    log_relative_path: Path
    inventory_relative_path: Path
    event_count: int
    model_rows: int
    bootstrap_rows: int


def _parse_utc(text: str, field: str, row_number: int) -> datetime:
    if not text.endswith("Z"):
        raise InferenceError(f"row {row_number} field {field} is not UTC Z-formatted")
    try:
        value = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        raise InferenceError(f"row {row_number} field {field} is not a timestamp") from None
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise InferenceError(f"row {row_number} field {field} is not UTC")
    return value


def _parse_bool(text: str, field: str, row_number: int) -> bool:
    if text == "true":
        return True
    if text == "false":
        return False
    raise InferenceError(f"row {row_number} field {field} is not true or false")


def _parse_optional_bool(text: str, field: str, row_number: int) -> bool | None:
    return None if text == "" else _parse_bool(text, field, row_number)


def _parse_float(text: str, field: str, row_number: int) -> float:
    try:
        value = float(text)
    except ValueError:
        raise InferenceError(f"row {row_number} field {field} is not numeric") from None
    if not math.isfinite(value):
        raise InferenceError(f"row {row_number} field {field} is non-finite")
    return value


def _parse_optional_float(text: str, field: str, row_number: int) -> float | None:
    return None if text == "" else _parse_float(text, field, row_number)


def _read_dict_rows(relative: Path, schema: Sequence[str]) -> list[dict[str, str]]:
    source = paths.require_file(paths.resolve(relative))
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != tuple(schema):
                raise InferenceError(f"{relative.name} has the wrong schema")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise InferenceError(f"cannot read {relative.name}: {error}") from None
    return rows


def read_event_records(
    relative: Path, config: ConfigNode | None = None
) -> tuple[EventRecord, ...]:
    cfg = config if config is not None else load_config()
    records: list[EventRecord] = []
    previous: datetime | None = None
    seen: set[str] = set()
    for row_number, row in enumerate(_read_dict_rows(relative, EVENT_OUTPUT_COLUMNS), start=2):
        event_id = row["event_id"]
        sign = row["event_sign"]
        if sign not in {"negative", "positive"}:
            raise InferenceError(f"event row {row_number} has invalid sign")
        close_time = _parse_utc(row["signal_close_time"], "signal_close_time", row_number)
        if previous is not None and close_time <= previous:
            raise InferenceError("event rows are not strictly chronological")
        previous = close_time
        if event_id in seen or not event_id.startswith(f"{sign}-"):
            raise InferenceError(f"event row {row_number} has invalid or duplicate identity")
        seen.add(event_id)
        block = row["utc_time_block"]
        if block not in TIME_BLOCKS:
            raise InferenceError(f"event row {row_number} has invalid UTC time block")
        complete = _parse_bool(row["complete_12h"], "complete_12h", row_number)
        complete_24h = _parse_bool(row["complete_24h"], "complete_24h", row_number)
        outcome = _parse_optional_float(
            row["post_log_return_12h"], "post_log_return_12h", row_number
        )
        if complete != (outcome is not None):
            raise InferenceError(f"event row {row_number} has inconsistent 12-hour outcome")
        post_positive = _parse_optional_bool(
            row["post_positive_12h"], "post_positive_12h", row_number
        )
        if post_positive is not None and post_positive != (outcome is not None and outcome > 0):
            raise InferenceError(f"event row {row_number} has inconsistent hit indicator")
        eligible_12 = _parse_bool(row["eligible_h1_h2"], "eligible_h1_h2", row_number)
        eligible_controls = _parse_bool(
            row["eligible_h3_h4"], "eligible_h3_h4", row_number
        )
        full_feature = _parse_bool(
            row["full_feature_eligible_hour"], "full_feature_eligible_hour", row_number
        )
        if eligible_12 != (sign == "negative" and complete):
            raise InferenceError(f"event row {row_number} has inconsistent H1/H2 eligibility")
        if eligible_controls != (eligible_12 and full_feature):
            raise InferenceError(f"event row {row_number} has inconsistent H3/H4 eligibility")
        activity_text = row["activity_percentile_90d"]
        activity = _parse_optional_float(
            activity_text, "activity_percentile_90d", row_number
        )
        score = _parse_optional_float(
            row["low_activity_score"], "low_activity_score", row_number
        )
        low = _parse_optional_bool(row["low_activity"], "low_activity", row_number)
        rv24 = _parse_optional_float(row["rv24"], "rv24", row_number)
        if full_feature and any(value is None for value in (activity, score, low, rv24)):
            raise InferenceError(f"event row {row_number} lacks required full-feature values")
        if activity is None and (score is not None or low is not None):
            raise InferenceError(f"event row {row_number} has partial activity fields")
        if activity is not None:
            if not 0 <= activity <= 1 or score is None or not 0 <= score <= 1:
                raise InferenceError(f"event row {row_number} has invalid activity values")
            if not math.isclose(
                score,
                1.0 - activity,
                rel_tol=cfg.reproducibility.rtol,
                abs_tol=cfg.reproducibility.atol,
            ):
                raise InferenceError(f"event row {row_number} has inconsistent activity score")
            try:
                activity_decimal = Decimal(activity_text)
            except InvalidOperation:
                raise InferenceError(
                    f"event row {row_number} has invalid activity percentile"
                ) from None
            threshold = Fraction(
                cfg.activity.low_activity_threshold.numerator,
                cfg.activity.low_activity_threshold.denominator,
            )
            expected_low = activity_decimal * threshold.denominator < threshold.numerator
            if low != expected_low:
                raise InferenceError(f"event row {row_number} has inconsistent low-activity flag")
        zscore = _parse_float(row["shock_zscore"], "shock_zscore", row_number)
        magnitude = _parse_optional_float(
            row["shock_magnitude"], "shock_magnitude", row_number
        )
        absolute_z = _parse_float(row["absolute_shock_z"], "absolute_shock_z", row_number)
        if sign == "negative" and not zscore < cfg.shock.negative_threshold:
            raise InferenceError(f"event row {row_number} violates the negative threshold")
        if sign == "positive" and not zscore > cfg.shock.positive_threshold:
            raise InferenceError(f"event row {row_number} violates the positive threshold")
        if sign == "positive" and magnitude is not None:
            raise InferenceError(f"event row {row_number} gives a positive event shock magnitude")
        if sign == "negative" and (
            magnitude is None
            or not math.isclose(
                magnitude,
                -zscore,
                rel_tol=cfg.reproducibility.rtol,
                abs_tol=cfg.reproducibility.atol,
            )
        ):
            raise InferenceError(f"event row {row_number} has invalid shock magnitude")
        if not math.isclose(
            absolute_z,
            abs(zscore),
            rel_tol=cfg.reproducibility.rtol,
            abs_tol=cfg.reproducibility.atol,
        ):
            raise InferenceError(f"event row {row_number} has invalid absolute z-score")
        record = EventRecord(
            event_id=event_id,
            sign=sign,
            signal_open_time=_parse_utc(
                row["signal_open_time"], "signal_open_time", row_number
            ),
            signal_close_time=close_time,
            signal_close=_parse_float(row["signal_close"], "signal_close", row_number),
            segment_id=int(row["contiguous_segment_id"]),
            calendar_year=int(row["calendar_year"]),
            calendar_month=row["calendar_month"],
            weekend=_parse_bool(row["weekend"], "weekend", row_number),
            crosses_into_monday_12h=_parse_optional_bool(
                row["crosses_into_monday_12h"], "crosses_into_monday_12h", row_number
            ),
            time_block=block,
            shock_return=_parse_float(
                row["shock_log_return_1h"], "shock_log_return_1h", row_number
            ),
            shock_zscore=zscore,
            shock_magnitude=magnitude,
            absolute_shock_z=absolute_z,
            activity_percentile=activity,
            low_activity_score=score,
            low_activity=low,
            rv24=rv24,
            full_feature_eligible=full_feature,
            complete_12h=complete,
            complete_24h=complete_24h,
            outcome_12h=outcome,
            post_positive_12h=post_positive,
            eligible_h1_h2=eligible_12,
            eligible_h3_h4=eligible_controls,
        )
        if record.signal_close <= 0:
            raise InferenceError(f"event row {row_number} has non-positive signal close")
        timestamp_unit = source_timestamp_unit(cfg, record.signal_open_time)
        source_quantum = (
            timedelta(milliseconds=1)
            if timestamp_unit == "milliseconds"
            else timedelta(microseconds=1)
        )
        if record.signal_close_time != (
            record.signal_open_time + timedelta(hours=1) - source_quantum
        ):
            raise InferenceError(f"event row {row_number} has inconsistent signal timestamps")
        if record.weekend != (close_time.isoweekday() in set(cfg.calendar.weekend_iso_weekdays)):
            raise InferenceError(f"event row {row_number} has inconsistent weekend label")
        expected_block = next(
            block.name
            for block in cfg.calendar.time_blocks
            if block.start_hour <= close_time.hour <= block.end_hour
        )
        if record.time_block != expected_block:
            raise InferenceError(f"event row {row_number} has inconsistent UTC time block")
        if not record.weekend and record.crosses_into_monday_12h is not None:
            raise InferenceError(f"event row {row_number} gives a weekday Monday-crossing flag")
        if record.weekend and record.complete_12h:
            expected_crossing = any(
                (record.signal_close_time + timedelta(hours=step)).isoweekday() == 1
                for step in range(1, 13)
            )
            if record.crosses_into_monday_12h != expected_crossing:
                raise InferenceError(f"event row {row_number} has inconsistent Monday crossing")
        if record.calendar_year != close_time.year or record.calendar_month != close_time.strftime(
            "%Y-%m"
        ):
            raise InferenceError(f"event row {row_number} has inconsistent calendar fields")
        records.append(record)
    if not records:
        raise InferenceError("development event artifact is empty")
    return tuple(records)


def read_path_records(relative: Path) -> tuple[PathRecord, ...]:
    output: list[PathRecord] = []
    by_event: dict[str, list[int]] = {}
    for row_number, row in enumerate(_read_dict_rows(relative, PATH_OUTPUT_COLUMNS), start=2):
        if row["event_sign"] != "negative":
            raise InferenceError(f"path row {row_number} is not a negative event")
        step = int(row["horizon_step_hours"])
        if not 1 <= step <= 24:
            raise InferenceError(f"path row {row_number} has invalid step")
        event_id = row["event_id"]
        by_event.setdefault(event_id, []).append(step)
        output.append(
            PathRecord(
                event_id=event_id,
                close_time=_parse_utc(
                    row["signal_close_time"], "signal_close_time", row_number
                ),
                step=step,
                value=_parse_float(
                    row["cumulative_post_log_return"],
                    "cumulative_post_log_return",
                    row_number,
                ),
            )
        )
    if not output:
        raise InferenceError("development path artifact is empty")
    if any(steps != list(range(1, 25)) for steps in by_event.values()):
        raise InferenceError("development path is not a balanced 1-24-hour sample")
    return tuple(output)


def read_drift_records(relative: Path) -> tuple[DriftRecord, ...]:
    output: list[DriftRecord] = []
    for row_number, row in enumerate(_read_dict_rows(relative, DRIFT_OUTPUT_COLUMNS), start=2):
        output.append(
            DriftRecord(
                reference_id=row["reference_id"],
                calendar_month=row["calendar_month"],
                value=_parse_float(
                    row["post_log_return_12h"], "post_log_return_12h", row_number
                ),
            )
        )
    if not output:
        raise InferenceError("development drift-reference artifact is empty")
    return tuple(output)


def reconcile_stage6_records(
    events: Sequence[EventRecord],
    path_records: Sequence[PathRecord],
    drift: Sequence[DriftRecord],
    config: ConfigNode,
) -> None:
    """Check that the event, path, and drift-reference outputs agree."""
    event_by_id = {row.event_id: row for row in events}
    if len(event_by_id) != len(events):
        raise InferenceError("Stage 6 events contain duplicate identities")
    path_ids = {row.event_id for row in path_records}
    expected_path_ids = {
        row.event_id for row in events if row.sign == "negative" and row.complete_24h
    }
    if path_ids != expected_path_ids:
        raise InferenceError("Stage 6 balanced-path identities do not reconcile with events")
    for path_row in path_records:
        event = event_by_id.get(path_row.event_id)
        if event is None or event.sign != "negative":
            raise InferenceError("Stage 6 path refers to an unknown or non-negative event")
        if path_row.close_time != event.signal_close_time:
            raise InferenceError("Stage 6 path timestamp disagrees with its event")
        if path_row.step == config.outcomes.primary_horizon_hours:
            if event.outcome_12h is None or not math.isclose(
                path_row.value,
                event.outcome_12h,
                rel_tol=config.reproducibility.rtol,
                abs_tol=config.reproducibility.atol,
            ):
                raise InferenceError("Stage 6 path step 12 disagrees with the primary outcome")
    reference_ids = [row.reference_id for row in drift]
    if len(reference_ids) != len(set(reference_ids)):
        raise InferenceError("Stage 6 drift reference contains duplicate identities")


def _model_terms(model: str) -> tuple[str, ...]:
    if model in {"model0", "positive_model0"}:
        return ("intercept",)
    if model in {"model1", "model1_cc"}:
        return ("intercept", "weekend")
    if model == "group_indicator":
        return ("intercept", "group_indicator")
    if model in {"model2", "negative_controlled"}:
        return (
            "intercept",
            "weekend",
            "low_activity_score",
            "shock_magnitude",
            "rv24",
            *TIME_BLOCK_TERMS,
        )
    if model == "positive_controlled":
        return (
            "intercept",
            "weekend",
            "low_activity_score",
            "absolute_shock_z",
            "rv24",
            *TIME_BLOCK_TERMS,
        )
    if model == "model3":
        return (
            "intercept",
            "weekend",
            "low_activity_score",
            "weekend_x_low_activity_score",
            "shock_magnitude",
            "rv24",
            *TIME_BLOCK_TERMS,
        )
    if model in {"h5_sign", "h5_any_sign"}:
        return (
            "intercept",
            "positive",
            "weekend",
            "low_activity_score",
            "absolute_shock_z",
            "rv24",
            *TIME_BLOCK_TERMS,
        )
    raise InferenceError(f"unknown model {model!r}")


def _term_value(row: EventRecord, term: str) -> float:
    if term == "intercept":
        return 1.0
    if term == "positive":
        return float(row.sign == "positive")
    if term == "weekend":
        return float(row.weekend)
    if term == "group_indicator":
        return float(row.weekend)
    if term == "low_activity_score":
        if row.low_activity_score is None:
            raise InferenceError(f"event {row.event_id} lacks low-activity score")
        return row.low_activity_score
    if term == "weekend_x_low_activity_score":
        if row.low_activity_score is None:
            raise InferenceError(f"event {row.event_id} lacks low-activity score")
        return float(row.weekend) * row.low_activity_score
    if term == "shock_magnitude":
        if row.shock_magnitude is None:
            raise InferenceError(f"event {row.event_id} lacks negative shock magnitude")
        return row.shock_magnitude
    if term == "absolute_shock_z":
        return row.absolute_shock_z
    if term == "rv24":
        if row.rv24 is None:
            raise InferenceError(f"event {row.event_id} lacks RV24")
        return row.rv24
    block_lookup = {
        "time_block_06_11": "06-11",
        "time_block_12_17": "12-17",
        "time_block_18_23": "18-23",
    }
    if term in block_lookup:
        return float(row.time_block == block_lookup[term])
    raise InferenceError(f"unknown model term {term!r}")


def build_model_data(
    name: str, sample: str, rows: Sequence[EventRecord]
) -> ModelData:
    terms = _model_terms(name)
    selected = tuple(rows)
    y_values: list[float] = []
    x_values: list[list[float]] = []
    for row in selected:
        if row.outcome_12h is None:
            raise InferenceError(f"model row {row.event_id} lacks the 12-hour outcome")
        y_values.append(row.outcome_12h)
        x_values.append([_term_value(row, term) for term in terms])
    y = np.asarray(y_values, dtype=np.float64)
    x = np.asarray(x_values, dtype=np.float64)
    if x.ndim != 2:
        x = np.empty((0, len(terms)), dtype=np.float64)
    return ModelData(name, sample, terms, selected, y, x)


def fit_hc3(data: ModelData, confidence_level: float = 0.95) -> FitResult:
    n, k = data.x.shape
    if n == 0:
        return FitResult(data, "unestimable", "empty_sample")
    if n <= k:
        return FitResult(data, "unestimable", "insufficient_residual_degrees_of_freedom")
    if np.linalg.matrix_rank(data.x) < k:
        return FitResult(data, "unestimable", "rank_deficient")
    try:
        ordinary = sm.OLS(data.y, data.x, hasconst=True).fit()
        robust = ordinary.get_robustcov_results(cov_type="HC3", use_t=True)
        estimates = np.asarray(robust.params, dtype=np.float64)
        standard_errors = np.asarray(robust.bse, dtype=np.float64)
        t_values = np.asarray(robust.tvalues, dtype=np.float64)
        two_sided = np.asarray(robust.pvalues, dtype=np.float64)
        alpha = 1.0 - confidence_level
        interval = np.asarray(robust.conf_int(alpha=alpha), dtype=np.float64)
        cooks = np.asarray(ordinary.get_influence().cooks_distance[0], dtype=np.float64)
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as error:
        return FitResult(data, "unestimable", f"fit_failed:{type(error).__name__}")
    arrays = (estimates, standard_errors, t_values, two_sided, interval, cooks)
    if any(not np.all(np.isfinite(values)) for values in arrays):
        return FitResult(data, "unestimable", "non_finite_inference")
    return FitResult(
        data=data,
        status="estimable",
        reason="",
        estimates=estimates,
        standard_errors=standard_errors,
        t_values=t_values,
        ci_low=interval[:, 0],
        ci_high=interval[:, 1],
        two_sided_p=two_sided,
        df_resid=float(ordinary.df_resid),
        residuals=np.asarray(ordinary.resid, dtype=np.float64),
        cooks_distance=cooks,
    )


def one_sided_greater_p(t_value: float, df_resid: float) -> float:
    if not math.isfinite(t_value) or not math.isfinite(df_resid) or df_resid <= 0:
        raise InferenceError("one-sided t probability requires finite t and positive df")
    return float(stats.t.sf(t_value, df_resid))


def holm_adjust(
    raw_p_values: Mapping[str, float | None], ordered_members: Sequence[str]
) -> dict[str, float | None]:
    if tuple(raw_p_values) != tuple(ordered_members):
        raise InferenceError("Holm inputs must preserve the declared family order")
    family: list[tuple[str, float]] = []
    for member in ordered_members:
        value = raw_p_values[member]
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise InferenceError(f"invalid raw p-value for {member}")
        family.append((member, 1.0 if value is None else value))
    ranked = sorted(enumerate(family), key=lambda item: (item[1][1], item[0]))
    running = 0.0
    adjusted_hidden: dict[str, float] = {}
    m = len(family)
    for rank, (_, (member, value)) in enumerate(ranked):
        running = max(running, (m - rank) * value)
        adjusted_hidden[member] = min(1.0, running)
    return {
        member: None if raw_p_values[member] is None else adjusted_hidden[member]
        for member in ordered_members
    }


def _coefficient_by_lstsq(data: ModelData) -> np.ndarray | None:
    n, k = data.x.shape
    if n <= k or np.linalg.matrix_rank(data.x) < k:
        return None
    try:
        estimate, _, rank, _ = np.linalg.lstsq(data.x, data.y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank != k or not np.all(np.isfinite(estimate)):
        return None
    return estimate


def month_block_bootstrap(
    data: ModelData,
    term: str,
    config: ConfigNode,
    *,
    analysis: str,
    target: int | None = None,
    maximum_attempts: int | None = None,
) -> BootstrapResult:
    term_index = data.terms.index(term)
    target_count = config.bootstrap.target_valid_replicates if target is None else target
    maximum = config.bootstrap.max_attempts if maximum_attempts is None else maximum_attempts
    if type(target_count) is not int or type(maximum) is not int or target_count <= 0:
        raise InferenceError("bootstrap target and maximum must be positive integers")
    if target_count > maximum:
        raise InferenceError("bootstrap target cannot exceed maximum attempts")
    months = tuple(sorted({row.calendar_month for row in data.rows}))
    stream = f"stage7:{analysis}:{data.name}:{term}"
    if not months:
        return BootstrapResult(
            analysis,
            data.name,
            term,
            len(data.rows),
            0,
            target_count,
            maximum,
            0,
            0,
            0,
            0,
            None,
            None,
            stream,
            "inconclusive_empty",
        )
    month_indices = {
        month: np.asarray(
            [index for index, row in enumerate(data.rows) if row.calendar_month == month],
            dtype=np.int64,
        )
        for month in months
    }
    rng = make_rng(stream, config)
    values: list[float] = []
    attempted = empty = rank_deficient = 0
    while len(values) < target_count and attempted < maximum:
        attempted += 1
        selected_months = rng.choice(months, size=len(months), replace=True)
        # A month drawn twice contributes all its events twice; copy_id keeps
        # the copies distinct.
        copies = [
            (copy_id, month_indices[str(month)])
            for copy_id, month in enumerate(selected_months)
        ]
        indices = np.concatenate([indices for _, indices in copies]) if copies else np.array([])
        if indices.size == 0:
            empty += 1
            continue
        resampled_rows = tuple(data.rows[int(index)] for index in indices)
        resampled = build_model_data(data.name, data.sample, resampled_rows)
        coefficients = _coefficient_by_lstsq(resampled)
        if coefficients is None:
            rank_deficient += 1
            continue
        values.append(float(coefficients[term_index]))
    status = "complete" if len(values) == target_count else "inconclusive_insufficient_valid"
    if values:
        low, high = np.quantile(np.asarray(values), [0.025, 0.975], method="linear")
    else:
        low = high = None
    return BootstrapResult(
        analysis=analysis,
        model=data.name,
        term=term,
        sample_n=len(data.rows),
        event_months=len(months),
        target=target_count,
        maximum_attempts=maximum,
        attempted=attempted,
        valid=len(values),
        empty=empty,
        rank_deficient=rank_deficient,
        low=None if low is None else float(low),
        high=None if high is None else float(high),
        stream=stream,
        status=status,
    )


def _eligible_model_rows(
    events: Sequence[EventRecord], model: str
) -> tuple[EventRecord, ...]:
    if model in {"model0", "model1"}:
        return tuple(row for row in events if row.sign == "negative" and row.eligible_h1_h2)
    if model in {"model1_cc", "model2", "model3", "negative_controlled"}:
        return tuple(row for row in events if row.sign == "negative" and row.eligible_h3_h4)
    if model == "positive_controlled":
        return tuple(
            row
            for row in events
            if row.sign == "positive" and row.complete_12h and row.full_feature_eligible
        )
    if model == "positive_model0":
        return tuple(row for row in events if row.sign == "positive" and row.complete_12h)
    if model in {"h5_sign", "h5_any_sign"}:
        return tuple(
            row for row in events if row.complete_12h and row.full_feature_eligible
        )
    raise InferenceError(f"unknown model sample {model!r}")


def build_primary_models(events: Sequence[EventRecord]) -> dict[str, ModelData]:
    return {
        name: build_model_data(name, sample, _eligible_model_rows(events, name))
        for name, sample in (
            ("model0", "negative_maximal_12h"),
            ("model1", "negative_maximal_12h"),
            ("model1_cc", "negative_model2_complete_case"),
            ("model2", "negative_model2_complete_case"),
            ("model3", "negative_model2_complete_case"),
            ("positive_model0", "positive_maximal_12h"),
            ("positive_controlled", "positive_complete_case"),
            ("h5_sign", "pooled_sign_specific_complete_case"),
        )
    }


def attenuation_rows(
    model1_cc: FitResult, model2: FitResult, tolerance: float
) -> list[dict[str, object]]:
    if model1_cc.status != "estimable" or model2.status != "estimable":
        return [
            {
                "analysis": "attenuation",
                "model": "model1_cc_vs_model2",
                "sample": "negative_model2_complete_case",
                "term": "weekend_absolute_change",
                "hypothesis": "",
                "n": len(model2.data.rows),
                "parameters": "",
                "residual_df": "",
                "estimate": "",
                "hc3_standard_error": "",
                "t_statistic": "",
                "ci_95_low": "",
                "ci_95_high": "",
                "raw_p_value": "",
                "holm_adjusted_p_value": "",
                "test_side": "descriptive",
                "status": "undefined_unestimable_input",
                "interpretation": "descriptive_noncausal",
            }
        ]
    before = model1_cc.coefficient("weekend")
    after = model2.coefficient("weekend")
    assert before is not None and after is not None
    change = before - after
    percentage = None if abs(before) <= tolerance else change / abs(before)
    label = (
        "undefined_zero_denominator"
        if percentage is None
        else "attenuation"
        if percentage > tolerance
        else "amplification"
        if percentage < -tolerance
        else "unchanged_within_tolerance"
    )
    common = {
        "analysis": "attenuation",
        "model": "model1_cc_vs_model2",
        "sample": "negative_model2_complete_case",
        "hypothesis": "",
        "n": len(model2.data.rows),
        "parameters": "",
        "residual_df": "",
        "hc3_standard_error": "",
        "t_statistic": "",
        "ci_95_low": "",
        "ci_95_high": "",
        "raw_p_value": "",
        "holm_adjusted_p_value": "",
        "test_side": "descriptive",
        "interpretation": "descriptive_noncausal",
    }
    return [
        {
            **common,
            "term": "weekend_absolute_change",
            "estimate": change,
            "status": label,
        },
        {
            **common,
            "term": "weekend_attenuation_fraction",
            "estimate": "" if percentage is None else percentage,
            "status": label,
        },
    ]


def _summary_statistics(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95], method="linear")
    result = {
        "mean": float(np.mean(array)),
        "median": float(quantiles[2]),
        "standard_deviation": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }
    return result


def _descriptive_rows(
    events: Sequence[EventRecord], paths_: Sequence[PathRecord], drift: Sequence[DriftRecord]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    negative = [row for row in events if row.sign == "negative" and row.eligible_h1_h2]
    controlled = [row for row in negative if row.eligible_h3_h4]
    positive = [row for row in events if row.sign == "positive"]
    groups: list[tuple[str, str, list[EventRecord]]] = [
        ("negative_events", "all", negative),
        ("weekend", "weekend", [row for row in negative if row.weekend]),
        ("weekend", "weekday", [row for row in negative if not row.weekend]),
        ("activity", "low_activity", [row for row in controlled if row.low_activity]),
        ("activity", "other_activity", [row for row in controlled if not row.low_activity]),
        (
            "combined",
            "weekend_low_activity",
            [row for row in controlled if row.weekend and row.low_activity],
        ),
        (
            "combined",
            "remaining",
            [row for row in controlled if not (row.weekend and row.low_activity)],
        ),
        ("positive_events", "all", positive),
    ]
    variables: tuple[tuple[str, Callable[[EventRecord], float | None]], ...] = (
        ("post_log_return_12h", lambda row: row.outcome_12h),
        ("shock_log_return_1h", lambda row: row.shock_return),
        ("shock_zscore", lambda row: row.shock_zscore),
        ("absolute_shock_z", lambda row: row.absolute_shock_z),
        ("activity_percentile_90d", lambda row: row.activity_percentile),
        ("rv24", lambda row: row.rv24),
    )
    output: list[dict[str, object]] = []
    for comparison, group, rows in groups:
        for variable, extractor in variables:
            values = [value for row in rows if (value := extractor(row)) is not None]
            for statistic, value in _summary_statistics(values).items():
                output.append(
                    {
                        "section": "event_descriptives",
                        "comparison": comparison,
                        "group": group,
                        "variable": variable,
                        "statistic": statistic,
                        "value": value,
                        "n": len(values),
                        "status": "descriptive",
                    }
                )
        outcomes = [row.outcome_12h for row in rows if row.outcome_12h is not None]
        hits = sum(value > 0 for value in outcomes)
        output.append(
            {
                "section": "event_descriptives",
                "comparison": comparison,
                "group": group,
                "variable": "post_log_return_12h",
                "statistic": "hit_rate",
                "value": "" if not outcomes else hits / len(outcomes),
                "n": len(outcomes),
                "status": "descriptive",
            }
        )
        if outcomes:
            interval = stats.binomtest(hits, len(outcomes)).proportion_ci(
                confidence_level=0.95, method="wilson"
            )
            for side, value in (("hit_rate_wilson_low", interval.low), ("hit_rate_wilson_high", interval.high)):
                output.append(
                    {
                        "section": "event_descriptives",
                        "comparison": comparison,
                        "group": group,
                        "variable": "post_log_return_12h",
                        "statistic": side,
                        "value": float(value),
                        "n": len(outcomes),
                        "status": "descriptive",
                    }
                )
    drift_values = [record.value for record in drift]
    for statistic in ("mean", "median"):
        output.append(
            {
                "section": "market_drift_reference",
                "comparison": "non_candidate_overlapping_hours",
                "group": "context_only",
                "variable": "post_log_return_12h",
                "statistic": statistic,
                "value": _summary_statistics(drift_values)[statistic],
                "n": len(drift_values),
                "status": "not_a_matched_or_formal_benchmark",
            }
        )
    output.append(
        {
            "section": "market_drift_reference",
            "comparison": "non_candidate_overlapping_hours",
            "group": "context_only",
            "variable": "post_log_return_12h",
            "statistic": "count",
            "value": len(drift_values),
            "n": len(drift_values),
            "status": "not_a_matched_or_formal_benchmark",
        }
    )
    for year in sorted({row.calendar_year for row in events}):
        for sign in ("negative", "positive"):
            count = sum(row.calendar_year == year and row.sign == sign for row in events)
            output.append(
                {
                    "section": "event_counts",
                    "comparison": "calendar_year",
                    "group": f"{year}:{sign}",
                    "variable": "accepted_events",
                    "statistic": "count",
                    "value": count,
                    "n": count,
                    "status": "descriptive",
                }
            )
    for sign in ("negative", "positive"):
        accepted = [row for row in events if row.sign == sign]
        complete = [row for row in accepted if row.complete_12h]
        controlled_count = sum(row.full_feature_eligible for row in complete)
        for statistic, value in (
            ("accepted_count", len(accepted)),
            ("complete_12h_count", len(complete)),
            ("controlled_complete_case_count", controlled_count),
        ):
            output.append(
                {
                    "section": "event_counts",
                    "comparison": "overall",
                    "group": sign,
                    "variable": "events",
                    "statistic": statistic,
                    "value": value,
                    "n": len(accepted),
                    "status": "descriptive",
                }
            )
    for rank_name, ordered in (
        ("largest_winners", sorted(negative, key=lambda row: row.outcome_12h or 0, reverse=True)[:5]),
        ("largest_losers", sorted(negative, key=lambda row: row.outcome_12h or 0)[:5]),
    ):
        for rank, row in enumerate(ordered, start=1):
            output.append(
                {
                    "section": "extreme_outcomes",
                    "comparison": rank_name,
                    "group": row.event_id,
                    "variable": "post_log_return_12h",
                    "statistic": f"rank_{rank}",
                    "value": row.outcome_12h,
                    "n": len(negative),
                    "status": "retained_not_deleted",
                }
            )
    path_output: list[dict[str, object]] = []
    for step in range(1, 25):
        values = [row.value for row in paths_ if row.step == step]
        summary = _summary_statistics(values)
        path_output.append(
            {
                "horizon_step_hours": step,
                "event_count": len(values),
                "mean_cumulative_log_return": summary["mean"],
                "median_cumulative_log_return": summary["median"],
                "standard_deviation": summary["standard_deviation"],
                "q05": summary["q05"],
                "q25": summary["q25"],
                "q75": summary["q75"],
                "q95": summary["q95"],
            }
        )
    if len({row["event_count"] for row in path_output}) != 1:
        raise InferenceError("average path does not use one balanced event sample")
    return output, path_output


def _group_comparison_data(
    events: Sequence[EventRecord], name: str
) -> tuple[ModelData, str, str]:
    if name == "weekend_vs_weekday":
        rows = _eligible_model_rows(events, "model1")
        return build_model_data("group_indicator", name, rows), "weekend", "weekday"
    controlled = _eligible_model_rows(events, "model2")
    if name == "low_vs_other_activity":
        relabelled = tuple(replace(row, weekend=bool(row.low_activity)) for row in controlled)
        return build_model_data("group_indicator", name, relabelled), "low_activity", "other_activity"
    if name == "combined_vs_remaining":
        relabelled = tuple(
            replace(row, weekend=bool(row.weekend and row.low_activity)) for row in controlled
        )
        return (
            build_model_data("group_indicator", name, relabelled),
            "weekend_low_activity",
            "remaining",
        )
    raise InferenceError(f"unknown group comparison {name!r}")


def build_group_comparisons(
    events: Sequence[EventRecord], config: ConfigNode
) -> tuple[list[dict[str, object]], list[BootstrapResult]]:
    rows: list[dict[str, object]] = []
    bootstraps: list[BootstrapResult] = []
    for name in ("weekend_vs_weekday", "low_vs_other_activity", "combined_vs_remaining"):
        data, group_1, group_0 = _group_comparison_data(events, name)
        fit = fit_hc3(data, config.inference.confidence_level)
        indicator = data.x[:, 1] if len(data.rows) else np.asarray([])
        index = 1
        rows.append(
            {
                "comparison": name,
                "group_1": group_1,
                "group_0": group_0,
                "n_1": int(np.sum(indicator == 1)),
                "n_0": int(np.sum(indicator == 0)),
                "difference_1_minus_0": "" if fit.estimates is None else fit.estimates[index],
                "hc3_standard_error": "" if fit.standard_errors is None else fit.standard_errors[index],
                "residual_df": "" if fit.df_resid is None else fit.df_resid,
                "t_statistic": "" if fit.t_values is None else fit.t_values[index],
                "ci_95_low": "" if fit.ci_low is None else fit.ci_low[index],
                "ci_95_high": "" if fit.ci_high is None else fit.ci_high[index],
                "two_sided_p_value": "" if fit.two_sided_p is None else fit.two_sided_p[index],
                "status": fit.status if fit.status == "estimable" else fit.reason,
            }
        )
        bootstraps.append(
            month_block_bootstrap(data, "group_indicator", config, analysis=name)
        )
    return rows, bootstraps


def _model_result_rows(
    fits: Mapping[str, FitResult], config: ConfigNode, *, analysis_period: str = DEVELOPMENT_PERIOD
) -> list[dict[str, object]]:
    hypothesis_slots = {
        "H1_model0": ("model0", "intercept"),
        "H2_model1": ("model1", "weekend"),
        "H3_model2": ("model2", "low_activity_score"),
        "H4_model2": ("model2", "weekend"),
    }
    raw: dict[str, float | None] = {}
    for hypothesis, (model, term) in hypothesis_slots.items():
        fit = fits[model]
        if fit.t_values is None or fit.df_resid is None:
            raw[hypothesis] = None
        else:
            raw[hypothesis] = one_sided_greater_p(
                float(fit.t_values[fit.term_index(term)]), fit.df_resid
            )
    members = tuple(config.inference.holm.members)
    if members != tuple(hypothesis_slots):
        raise InferenceError("configured Holm family does not match Models 0-2")
    adjusted = holm_adjust(raw, members)
    term_hypothesis = {(model, term): hypothesis for hypothesis, (model, term) in hypothesis_slots.items()}
    output: list[dict[str, object]] = []
    for name in (
        "model0",
        "model1",
        "model1_cc",
        "model2",
        "model3",
        "positive_model0",
        "positive_controlled",
        "h5_sign",
        "h5_any_sign",
    ):
        fit = fits[name]
        for index, term in enumerate(fit.data.terms):
            hypothesis = term_hypothesis.get(
                (name, term),
                "H5"
                if name == "h5_sign" and term == "positive"
                else "H5_diagnostic"
                if name == "h5_any_sign" and term == "positive"
                else "",
            )
            if hypothesis in raw:
                p_value = raw[hypothesis]
                holm = adjusted[hypothesis]
                side = "one_sided_greater"
            elif hypothesis in {"H5", "H5_diagnostic"} and fit.two_sided_p is not None:
                p_value = float(fit.two_sided_p[index])
                holm = None
                side = "two_sided_outside_holm"
            else:
                p_value = float(fit.two_sided_p[index]) if fit.two_sided_p is not None else None
                holm = None
                side = "two_sided_descriptive"
            if fit.status != "estimable":
                interpretation = "inconclusive_unestimable"
            elif hypothesis in raw:
                estimate = float(fit.estimates[index]) if fit.estimates is not None else 0.0
                meets = estimate > 0 and holm is not None and holm < config.inference.alpha
                if analysis_period == HELD_OUT_PERIOD:
                    interpretation = "supported_in_held_out" if meets else "not_supported_in_held_out"
                else:
                    interpretation = (
                        "development_association_meets_rule_nonconfirmatory"
                        if meets else "development_association_does_not_meet_rule"
                    )
            elif hypothesis == "H5":
                distinguishable = p_value is not None and p_value < config.inference.alpha
                if analysis_period == HELD_OUT_PERIOD:
                    interpretation = (
                        "held_out_asymmetry_evidence_secondary" if distinguishable
                        else "held_out_asymmetry_not_statistically_distinguishable"
                    )
                else:
                    interpretation = (
                        "development_asymmetry_evidence_nonconfirmatory" if distinguishable
                        else "development_asymmetry_not_statistically_distinguishable"
                    )
            elif hypothesis == "H5_diagnostic":
                interpretation = "h5_diagnostic_primary_sign_specific_events_unchanged"
            else:
                interpretation = "descriptive_noncausal"
            output.append(
                {
                    "analysis": "primary" if name.startswith("model") else "positive_shock_comparison",
                    "model": name,
                    "sample": fit.data.sample,
                    "term": term,
                    "hypothesis": hypothesis,
                    "n": len(fit.data.rows),
                    "parameters": len(fit.data.terms),
                    "residual_df": "" if fit.df_resid is None else fit.df_resid,
                    "estimate": "" if fit.estimates is None else fit.estimates[index],
                    "hc3_standard_error": "" if fit.standard_errors is None else fit.standard_errors[index],
                    "t_statistic": "" if fit.t_values is None else fit.t_values[index],
                    "ci_95_low": "" if fit.ci_low is None else fit.ci_low[index],
                    "ci_95_high": "" if fit.ci_high is None else fit.ci_high[index],
                    "raw_p_value": "" if p_value is None else p_value,
                    "holm_adjusted_p_value": "" if holm is None else holm,
                    "test_side": side,
                    "status": fit.status if fit.status == "estimable" else fit.reason,
                    "interpretation": interpretation,
                }
            )
    output.extend(
        attenuation_rows(
            fits["model1_cc"], fits["model2"], config.reproducibility.atol
        )
    )
    return output


def build_diagnostics(fits: Mapping[str, FitResult], config: ConfigNode) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for name in ("model0", "model1", "model2", "model3", "h5_sign"):
        fit = fits[name]
        if fit.status != "estimable" or fit.residuals is None or fit.cooks_distance is None:
            output.append(
                {
                    "model": name,
                    "diagnostic": "model_status",
                    "term": "",
                    "value": "",
                    "minimum": "",
                    "maximum": "",
                    "count": 0,
                    "status": fit.reason,
                    "details": "",
                }
            )
            continue
        residuals = fit.residuals
        for label, value in (
            ("residual_mean", np.mean(residuals)),
            ("residual_standard_deviation", np.std(residuals, ddof=1)),
            ("residual_skewness", stats.skew(residuals, bias=False)),
            ("residual_excess_kurtosis", stats.kurtosis(residuals, fisher=True, bias=False)),
            ("maximum_absolute_residual", np.max(np.abs(residuals))),
        ):
            output.append(
                {
                    "model": name,
                    "diagnostic": label,
                    "term": "",
                    "value": float(value),
                    "minimum": "",
                    "maximum": "",
                    "count": len(residuals),
                    "status": "diagnostic_not_selection_rule",
                    "details": "",
                }
            )
        threshold = 4.0 / len(fit.data.rows)
        flagged = np.flatnonzero(fit.cooks_distance > threshold)
        flagged_ids = [fit.data.rows[int(index)].event_id for index in flagged]
        output.append(
            {
                "model": name,
                "diagnostic": "cooks_distance_above_4_over_n",
                "term": "",
                "value": threshold,
                "minimum": "",
                "maximum": float(np.max(fit.cooks_distance)),
                "count": len(flagged_ids),
                "status": "flagged_not_deleted",
                "details": json.dumps(flagged_ids, separators=(",", ":")),
            }
        )
        for term_index, term in enumerate(fit.data.terms):
            if term == "intercept" or not flagged_ids:
                continue
            leave_one: list[float] = []
            for index in flagged:
                retained = tuple(
                    row for row_index, row in enumerate(fit.data.rows) if row_index != int(index)
                )
                coefficients = _coefficient_by_lstsq(
                    build_model_data(fit.data.name, fit.data.sample, retained)
                )
                if coefficients is not None:
                    leave_one.append(float(coefficients[term_index]))
            output.append(
                {
                    "model": name,
                    "diagnostic": "leave_one_flagged_event_out_coefficient_range",
                    "term": term,
                    "value": "",
                    "minimum": "" if not leave_one else min(leave_one),
                    "maximum": "" if not leave_one else max(leave_one),
                    "count": len(leave_one),
                    "status": "complete" if len(leave_one) == len(flagged) else "partly_unestimable",
                    "details": "fixed_primary_event_identities",
                }
            )
        if fit.data.x.shape[1] > 1:
            predictors = fit.data.x[:, 1:]
            means = np.mean(predictors, axis=0)
            scales = np.std(predictors, axis=0, ddof=0)
            if np.any(scales <= config.reproducibility.atol):
                condition = math.inf
            else:
                standardised = np.column_stack(
                    [np.ones(len(predictors)), (predictors - means) / scales]
                )
                condition = float(np.linalg.cond(standardised))
            output.append(
                {
                    "model": name,
                    "diagnostic": "standardised_design_condition_number",
                    "term": "",
                    "value": condition,
                    "minimum": "",
                    "maximum": "",
                    "count": fit.data.x.shape[1],
                    "status": "diagnostic_not_selection_rule",
                    "details": "predictors_centered_and_scaled",
                }
            )
            for index, term in enumerate(fit.data.terms[1:], start=1):
                try:
                    vif = float(variance_inflation_factor(fit.data.x, index))
                except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                    vif = math.inf
                output.append(
                    {
                        "model": name,
                        "diagnostic": "variance_inflation_factor",
                        "term": term,
                        "value": vif,
                        "minimum": "",
                        "maximum": "",
                        "count": len(fit.data.rows),
                        "status": "diagnostic_not_selection_rule",
                        "details": "no_control_selection",
                    }
                )
    model3 = fits["model3"]
    cells = {
        (weekend, low): sum(
            row.weekend == weekend and row.low_activity == low for row in model3.data.rows
        )
        for weekend in (False, True)
        for low in (False, True)
    }
    minimum_cell = min(cells.values()) if cells else 0
    required_n = config.inference.model3_min_n_per_coefficient * len(model3.data.terms)
    interpretable = (
        minimum_cell >= config.inference.model3_min_events_per_cell
        and len(model3.data.rows) >= required_n
    )
    output.append(
        {
            "model": "model3",
            "diagnostic": "interaction_interpretability",
            "term": "weekend_x_low_activity_score",
            "value": minimum_cell,
            "minimum": minimum_cell,
            "maximum": max(cells.values()) if cells else 0,
            "count": len(model3.data.rows),
            "status": "quantitatively_interpretable" if interpretable else "exploratory_underpowered",
            "details": json.dumps(
                {f"weekend_{int(key[0])}_low_{int(key[1])}": value for key, value in cells.items()},
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    )
    return output


def count_cross_sign_overlaps(events: Sequence[EventRecord]) -> int:
    negative = [row for row in events if row.sign == "negative" and row.complete_12h]
    positive = [row for row in events if row.sign == "positive" and row.complete_12h]
    horizon = timedelta(hours=12)
    return sum(
        abs(negative_row.signal_close_time - positive_row.signal_close_time) < horizon
        for negative_row in negative
        for positive_row in positive
    )


def read_any_sign_candidates(
    relative: Path, *, analysis_period: str = DEVELOPMENT_PERIOD
) -> tuple[HourCandidate, ...]:
    if analysis_period not in {DEVELOPMENT_PERIOD, HELD_OUT_PERIOD}:
        raise InferenceError("any-sign candidates require development or held_out")
    rows = _read_dict_rows(relative, FEATURE_COLUMNS)
    output: list[HourCandidate] = []
    for row_number, row in enumerate(rows, start=2):
        period = row["analysis_period"]
        if analysis_period == DEVELOPMENT_PERIOD and period == HELD_OUT_PERIOD:
            continue
        if period not in {"warmup", DEVELOPMENT_PERIOD, HELD_OUT_PERIOD}:
            raise InferenceError(f"feature row {row_number} has invalid analysis period")
        shock_eligible = _parse_bool(
            row["shock_eligible_hour"], "shock_eligible_hour", row_number
        )
        zscore = _parse_optional_float(row["shock_zscore"], "shock_zscore", row_number)
        if (zscore is not None) != shock_eligible:
            raise InferenceError(
                f"feature row {row_number} shock eligibility and z-score disagree"
            )
        output.append(
            HourCandidate(
                analysis_period=period,
                open_time=_parse_utc(row["open_time"], "open_time", row_number),
                close_time=_parse_utc(row["close_time"], "close_time", row_number),
                close_time_raw=int(row["close_time_raw"]),
                shock_eligible=shock_eligible,
                zscore=zscore,
            )
        )
    if not output or not any(row.analysis_period == analysis_period for row in output):
        raise InferenceError(f"any-sign diagnostic has no {analysis_period} rows")
    return tuple(output)


def build_any_sign_event_records(
    feature_rows: Sequence[OutcomeInputRow],
    candidates: Sequence[HourCandidate],
    config: ConfigNode,
    *,
    analysis_period: str = DEVELOPMENT_PERIOD,
) -> tuple[EventRecord, ...]:
    if analysis_period not in {DEVELOPMENT_PERIOD, HELD_OUT_PERIOD}:
        raise InferenceError("any-sign events require development or held_out")
    decisions = construct_any_sign_events(
        [row.zscore for row in candidates],
        [row.close_time for row in candidates],
        [row.close_time_raw for row in candidates],
        negative_threshold=config.shock.negative_threshold,
        positive_threshold=config.shock.positive_threshold,
        cooldown_hours=config.events.cooldown_hours,
    )
    accepted = {
        decision.event_id: decision
        for row, decision in zip(candidates, decisions, strict=True)
        if row.analysis_period == analysis_period and decision.accepted
    }
    replaced_rows: list[OutcomeInputRow] = []
    for row in feature_rows:
        if row.analysis_period != analysis_period or row.close_time_raw is None:
            replaced_rows.append(row)
            continue
        negative_id = f"negative-{row.close_time_raw}"
        positive_id = f"positive-{row.close_time_raw}"
        event_id = negative_id if negative_id in accepted else positive_id if positive_id in accepted else ""
        diagnostic_z = row.shock_zscore if event_id else None
        replaced_rows.append(
            replace(
                row,
                negative_event=event_id.startswith("negative-"),
                positive_event=event_id.startswith("positive-"),
                event_id=event_id,
                shock_magnitude=(
                    -diagnostic_z
                    if event_id.startswith("negative-") and diagnostic_z is not None
                    else None
                ),
                absolute_shock_z=(
                    abs(diagnostic_z) if diagnostic_z is not None else None
                ),
            )
        )
    bundle: DevelopmentOutcomeBundle = build_outcomes(
        replaced_rows, config, analysis_period=analysis_period
    )
    return event_records_from_outcomes(bundle.events)


def event_records_from_outcomes(
    events: Sequence[EventOutcome],
) -> tuple[EventRecord, ...]:
    """Convert validated outcome records into event records."""
    output: list[EventRecord] = []
    for event in events:
        horizon = event.horizons[12]
        row = event.row
        output.append(
            EventRecord(
                event_id=row.event_id,
                sign=event.sign,
                signal_open_time=row.open_time,
                signal_close_time=row.close_time,
                signal_close=float(row.close) if row.close is not None else math.nan,
                segment_id=row.segment_id,
                calendar_year=int(row.calendar_year) if row.calendar_year is not None else 0,
                calendar_month=row.calendar_month,
                weekend=bool(row.weekend),
                crosses_into_monday_12h=row.crosses_into_monday_12h,
                time_block=row.utc_time_block,
                shock_return=float(row.shock_log_return_1h),
                shock_zscore=float(row.shock_zscore),
                shock_magnitude=row.shock_magnitude,
                absolute_shock_z=float(row.absolute_shock_z),
                activity_percentile=row.activity_percentile,
                low_activity_score=row.low_activity_score,
                low_activity=row.low_activity,
                rv24=row.rv24,
                full_feature_eligible=row.full_feature_eligible,
                complete_12h=horizon.complete,
                complete_24h=event.horizons[24].complete,
                outcome_12h=horizon.value,
                post_positive_12h=None if horizon.value is None else horizon.value > 0,
                eligible_h1_h2=event.sign == "negative" and horizon.complete,
                eligible_h3_h4=(
                    event.sign == "negative" and horizon.complete and row.full_feature_eligible
                ),
            )
        )
    return tuple(output)


def _format_value(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "not_finite"
        return format(number, ".17g")
    return str(value)


def render_records(records: Sequence[Mapping[str, object]], schema: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(schema), lineterminator="\n")
    writer.writeheader()
    for record in records:
        if set(record) != set(schema):
            raise InferenceError("output record does not match its declared schema")
        writer.writerow({name: _format_value(record[name]) for name in schema})
    return buffer.getvalue()


def render_bootstraps(results: Sequence[BootstrapResult]) -> str:
    records = [
        {
            "analysis": result.analysis,
            "model": result.model,
            "term": result.term,
            "sample_n": result.sample_n,
            "event_months": result.event_months,
            "target_valid_replicates": result.target,
            "maximum_attempts": result.maximum_attempts,
            "attempted": result.attempted,
            "valid": result.valid,
            "empty": result.empty,
            "rank_deficient": result.rank_deficient,
            "percentile_95_low": result.low,
            "percentile_95_high": result.high,
            "quantile_method": "numpy_linear",
            "rng_stream": result.stream,
            "status": result.status,
        }
        for result in results
    ]
    return render_records(records, BOOTSTRAP_COLUMNS)


def _ecdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return ordered, np.arange(1, len(ordered) + 1, dtype=np.float64) / len(ordered)


def render_figures(
    events: Sequence[EventRecord],
    path_summary: Sequence[Mapping[str, object]],
    development_hours: Sequence[OutcomeInputRow],
    *,
    analysis_period: str = DEVELOPMENT_PERIOD,
    label: str = "Development",
) -> tuple[tuple[str, bytes], ...]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    metadata = {"Software": f"matplotlib {matplotlib.__version__}"}
    figures: list[tuple[str, bytes]] = []

    def save(name: str, figure) -> None:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight", metadata=metadata)
        plt.close(figure)
        figures.append((name, buffer.getvalue()))

    figure, axis = plt.subplots(figsize=(10, 4.5))
    hours = [row for row in development_hours if row.analysis_period == analysis_period]
    axis.plot([row.close_time for row in hours], [float(row.close) for row in hours], linewidth=0.7)
    for sign, colour, marker in (("negative", "#c43c39", "v"), ("positive", "#2d7f5e", "^")):
        selected = [row for row in events if row.sign == sign]
        axis.scatter(
            [row.signal_close_time for row in selected],
            [row.signal_close for row in selected],
            s=12,
            c=colour,
            marker=marker,
            label=f"Accepted {sign} events",
        )
    axis.set(title=f"BTC/USDT {label.lower()} sample and accepted events", ylabel="USDT per BTC")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    save(f"{analysis_period}_price_events.png", figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    steps = [int(row["horizon_step_hours"]) for row in path_summary]
    means = [float(row["mean_cumulative_log_return"]) for row in path_summary]
    axis.plot(steps, means, color="#335c99", linewidth=2)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        title=f"Average cumulative return after negative shocks (N={path_summary[0]['event_count']})",
        xlabel="Hours after completed shock candle",
        ylabel="Mean cumulative log return",
        xticks=[1, 6, 12, 18, 24],
    )
    axis.grid(alpha=0.2)
    save(f"{analysis_period}_average_path.png", figure)

    negative = _eligible_model_rows(events, "model1")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for group_label, predicate, colour in (
        ("Weekend", lambda row: row.weekend, "#c43c39"),
        ("Weekday", lambda row: not row.weekend, "#335c99"),
    ):
        values = [float(row.outcome_12h) for row in negative if predicate(row)]
        x, y = _ecdf(values)
        axis.step(
            x,
            y,
            where="post",
            label=f"{group_label} (N={len(values)})",
            color=colour,
        )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(title="Weekend versus weekday 12-hour outcomes", xlabel="12-hour log return", ylabel="Empirical CDF")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    save(f"{analysis_period}_weekend_distribution.png", figure)

    controlled = _eligible_model_rows(events, "model2")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for group_label, predicate, colour in (
        ("Low activity", lambda row: bool(row.low_activity), "#d28b26"),
        ("Other activity", lambda row: not bool(row.low_activity), "#5b6f8f"),
    ):
        values = [float(row.outcome_12h) for row in controlled if predicate(row)]
        x, y = _ecdf(values)
        axis.step(
            x,
            y,
            where="post",
            label=f"{group_label} (N={len(values)})",
            color=colour,
        )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(title="Outcome by pre-shock activity group", xlabel="12-hour log return", ylabel="Empirical CDF")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    save(f"{analysis_period}_activity_distribution.png", figure)
    return tuple(figures)


def build_inference_bundle(
    events: Sequence[EventRecord],
    paths_: Sequence[PathRecord],
    drift: Sequence[DriftRecord],
    any_sign_events: Sequence[EventRecord],
    development_hours: Sequence[OutcomeInputRow],
    config: ConfigNode,
    *,
    analysis_period: str = DEVELOPMENT_PERIOD,
    label: str = "Development",
) -> InferenceBundle:
    if analysis_period not in {DEVELOPMENT_PERIOD, HELD_OUT_PERIOD}:
        raise InferenceError("inference bundle requires development or held_out")
    models = build_primary_models(events)
    models["h5_any_sign"] = build_model_data(
        "h5_any_sign",
        "pooled_any_sign_cooldown_complete_case",
        _eligible_model_rows(any_sign_events, "h5_any_sign"),
    )
    fits = {
        name: fit_hc3(data, config.inference.confidence_level)
        for name, data in models.items()
    }
    descriptives, path_summary = _descriptive_rows(events, paths_, drift)
    overlap = count_cross_sign_overlaps(events)
    descriptives.append(
        {
            "section": "positive_shock_comparison",
            "comparison": "cross_sign_overlap",
            "group": "complete_12h_sign_specific_events",
            "variable": "overlapping_12h_window_pairs",
            "statistic": "count",
            "value": overlap,
            "n": len(events),
            "status": "diagnostic",
        }
    )
    for sign in ("negative", "positive"):
        sign_rows = [row for row in any_sign_events if row.sign == sign]
        complete_values = [
            float(row.outcome_12h)
            for row in sign_rows
            if row.complete_12h and row.outcome_12h is not None
        ]
        for statistic, value, n in (
            ("accepted_count", len(sign_rows), len(sign_rows)),
            ("complete_12h_count", len(complete_values), len(sign_rows)),
            (
                "mean_post_log_return_12h",
                "" if not complete_values else float(np.mean(complete_values)),
                len(complete_values),
            ),
            (
                "median_post_log_return_12h",
                "" if not complete_values else float(np.median(complete_values)),
                len(complete_values),
            ),
        ):
            descriptives.append(
                {
                    "section": "positive_shock_comparison",
                    "comparison": "any_sign_cooldown",
                    "group": sign,
                    "variable": "events_or_post_log_return_12h",
                    "statistic": statistic,
                    "value": value,
                    "n": n,
                    "status": "h5_diagnostic_primary_events_unchanged",
                }
            )
    descriptives.append(
        {
            "section": "positive_shock_comparison",
            "comparison": "any_sign_cooldown",
            "group": f"accepted_{analysis_period}_events",
            "variable": "event_count",
            "statistic": "count",
            "value": len(any_sign_events),
            "n": len(any_sign_events),
            "status": "h5_diagnostic_primary_events_unchanged",
        }
    )
    group_rows, group_bootstraps = build_group_comparisons(events, config)
    model_rows = _model_result_rows(fits, config, analysis_period=analysis_period)
    bootstrap_results = list(group_bootstraps)
    for model, term in (
        ("model0", "intercept"),
        ("model1", "weekend"),
        ("model2", "low_activity_score"),
        ("model2", "weekend"),
        ("model3", "weekend_x_low_activity_score"),
        ("positive_model0", "intercept"),
        ("positive_controlled", "weekend"),
        ("positive_controlled", "low_activity_score"),
        ("h5_sign", "positive"),
        ("h5_any_sign", "positive"),
    ):
        bootstrap_results.append(
            month_block_bootstrap(models[model], term, config, analysis="model_inference")
        )
    diagnostics = build_diagnostics(fits, config)
    figures = render_figures(
        events, path_summary, development_hours,
        analysis_period=analysis_period, label=label,
    )
    return InferenceBundle(
        descriptives=tuple(descriptives),
        group_comparisons=tuple(group_rows),
        path_summary=tuple(path_summary),
        model_results=tuple(model_rows),
        bootstrap_results=tuple(bootstrap_results),
        diagnostics=tuple(diagnostics),
        figures=figures,
        event_rows=len(events),
        h5_any_sign_events=len(any_sign_events),
    )


def verify_inference_inputs(
    config: ConfigNode,
    outcome_manifest_relative: Path,
    feature_manifest_relative: Path,
    *,
    require_frozen_acceptance: bool,
) -> tuple[dict[str, Path], Path, int, dict[str, str]]:
    if require_frozen_acceptance and outcome_manifest_relative != ACCEPTED_OUTCOME_MANIFEST:
        raise InferenceError("default Stage 7 execution must use the accepted Stage 6 run")
    verified_outputs = verify_development_inventory(outcome_manifest_relative)
    verify_manifest_checksum(outcome_manifest_relative)
    manifest_path = paths.resolve(outcome_manifest_relative)
    manifest_digest = sha256_file(manifest_path)
    manifest = read_manifest(outcome_manifest_relative)
    if not matches_active_protocol(manifest, config):
        raise InferenceError("Stage 6 evidence does not match active specification/config")
    if manifest["code_identity"]["git_dirty"] is True:
        raise InferenceError("Stage 6 evidence was produced from a dirty repository")
    if require_frozen_acceptance and manifest_digest != ACCEPTED_OUTCOME_MANIFEST_SHA256:
        raise InferenceError("accepted Stage 6 manifest identity does not match the freeze")
    artifact_paths = {
        name: outcome_manifest_relative.parent / name
        for name in (
            "development_event_outcomes.csv",
            "development_event_paths_24h.csv",
            "development_drift_reference.csv",
        )
    }
    inventory_relative = outcome_manifest_relative.parent / "development_output_inventory.csv"
    sidecar_relative = outcome_manifest_relative.parent / f"{outcome_manifest_relative.name}.sha256"
    checksums = {
        **verified_outputs,
        outcome_manifest_relative.as_posix(): manifest_digest,
        sidecar_relative.as_posix(): sha256_file(paths.resolve(sidecar_relative)),
        inventory_relative.as_posix(): sha256_file(paths.resolve(inventory_relative)),
    }
    feature_relative, feature_rows, feature_checksums = verify_feature_input(
        config,
        feature_manifest_relative,
        require_frozen_acceptance=require_frozen_acceptance,
    )
    overlap = set(checksums) & set(feature_checksums)
    if overlap:
        raise InferenceError(f"duplicate Stage 7 input checksum paths: {sorted(overlap)}")
    checksums.update(feature_checksums)
    return artifact_paths, feature_relative, feature_rows, dict(sorted(checksums.items()))


def _render_inventory(
    artifacts: Sequence[tuple[Path, str, int, str, Sequence[str]]]
) -> str:
    records = [
        {
            "artifact_path": path.as_posix(),
            "sha256": digest,
            "record_count": count,
            "artifact_type": artifact_type,
            "column_schema_json": json.dumps(list(schema), separators=(",", ":")),
        }
        for path, digest, count, artifact_type, schema in artifacts
    ]
    return render_records(records, INVENTORY_COLUMNS)


def verify_inference_inventory(manifest_relative: Path) -> dict[str, str]:
    """Verify the inference manifest, inventory, tables, and figures."""
    verify_manifest_checksum(manifest_relative)
    manifest = read_manifest(manifest_relative)
    if manifest["stage"] != STAGE or manifest["mode"] != MODE:
        raise InferenceError("manifest is not Stage 7 development-inference evidence")
    if manifest["column_schema"] != list(INVENTORY_COLUMNS):
        raise InferenceError("Stage 7 manifest does not declare the inventory schema")
    inventory_relative = manifest_relative.parent / "development_inference_inventory.csv"
    if sha256_file(paths.resolve(inventory_relative)) != manifest["output_checksum"]:
        raise InferenceError("Stage 7 inventory checksum does not match its manifest")
    records = _read_dict_rows(inventory_relative, INVENTORY_COLUMNS)
    if len(records) != manifest["row_count"]:
        raise InferenceError("Stage 7 inventory row count disagrees with its manifest")
    figure_names = {
        "development_price_events.png",
        "development_average_path.png",
        "development_weekend_distribution.png",
        "development_activity_distribution.png",
    }
    expected_names = set(TABLE_SCHEMAS) | figure_names
    if {Path(record["artifact_path"]).name for record in records} != expected_names:
        raise InferenceError("Stage 7 inventory does not list the required artifacts")
    if len(records) != len(expected_names):
        raise InferenceError("Stage 7 inventory contains duplicate artifact names")
    verified: dict[str, str] = {}
    for record in records:
        relative = Path(record["artifact_path"])
        if relative.parent != manifest_relative.parent:
            raise InferenceError("Stage 7 artifact is outside its run directory")
        digest = record["sha256"]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise InferenceError("Stage 7 inventory contains an invalid SHA-256")
        artifact = paths.require_file(paths.resolve(relative))
        if sha256_file(artifact) != digest:
            raise InferenceError(f"Stage 7 artifact checksum failed for {relative.name}")
        try:
            count = int(record["record_count"])
            schema = json.loads(record["column_schema_json"])
        except (ValueError, TypeError, json.JSONDecodeError):
            raise InferenceError("Stage 7 inventory metadata is malformed") from None
        if count < 0 or not isinstance(schema, list) or not schema:
            raise InferenceError("Stage 7 inventory count or schema is invalid")
        if relative.name in TABLE_SCHEMAS:
            expected_schema = list(TABLE_SCHEMAS[relative.name])
            if record["artifact_type"] != "csv" or schema != expected_schema:
                raise InferenceError(f"Stage 7 table metadata failed for {relative.name}")
            rows = _read_dict_rows(relative, expected_schema)
            if len(rows) != count:
                raise InferenceError(f"Stage 7 table row count failed for {relative.name}")
        else:
            if (
                record["artifact_type"] != "png"
                or schema != ["binary_png"]
                or count != 1
                or not artifact.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            ):
                raise InferenceError(f"Stage 7 figure metadata failed for {relative.name}")
        verified[relative.as_posix()] = digest
    return dict(sorted(verified.items()))


def _record_failure(log: RunLog, error: Exception) -> None:
    try:
        log.record(
            "development_inference_failed",
            reason=type(error).__name__,
            detail=str(error)[:500],
            held_out_outcomes_accessed=False,
        )
    except Exception:  # pragma: no cover - never hide the original error
        pass


def run_development_inference(
    config: ConfigNode | None = None,
    *,
    outcome_manifest_relative: Path | None = None,
    feature_manifest_relative: Path | None = None,
) -> DevelopmentInferenceResult:
    cfg = config if config is not None else load_config()
    outcome_manifest = outcome_manifest_relative or ACCEPTED_OUTCOME_MANIFEST
    feature_manifest = feature_manifest_relative or ACCEPTED_FEATURE_MANIFEST
    custom_input = outcome_manifest_relative is not None or feature_manifest_relative is not None
    if custom_input and (outcome_manifest_relative is None or feature_manifest_relative is None):
        raise InferenceError("custom Stage 7 input requires both outcome and feature manifests")
    outcome_is_accepted = outcome_manifest == ACCEPTED_OUTCOME_MANIFEST
    feature_is_accepted = feature_manifest == ACCEPTED_FEATURE_MANIFEST
    if outcome_is_accepted != feature_is_accepted:
        raise InferenceError("accepted and synthetic Stage 7 inputs cannot be mixed")
    frozen_inputs = outcome_is_accepted and feature_is_accepted
    draft = build_manifest(STAGE, MODE, cfg)
    if frozen_inputs and draft.code_identity["git_dirty"] is not False:
        raise InferenceError(
            "development inference requires a clean committed repository; "
            "offline tests must use explicit synthetic manifests"
        )
    log = create_run_log(draft.run_id, config=cfg)
    log.record(
        "development_inference_started",
        outcome_manifest=outcome_manifest.as_posix(),
        feature_manifest=feature_manifest.as_posix(),
        analysis_period=DEVELOPMENT_PERIOD,
        held_out_outcomes_accessed=False,
    )
    try:
        artifact_paths, feature_relative, expected_feature_rows, input_checksums = (
            verify_inference_inputs(
                cfg,
                outcome_manifest,
                feature_manifest,
                require_frozen_acceptance=frozen_inputs,
            )
        )
        events = read_event_records(artifact_paths["development_event_outcomes.csv"], cfg)
        path_records = read_path_records(artifact_paths["development_event_paths_24h.csv"])
        drift = read_drift_records(artifact_paths["development_drift_reference.csv"])
        reconcile_stage6_records(events, path_records, drift, cfg)
        feature_rows = read_outcome_rows(feature_relative, expected_feature_rows, cfg)
        candidates = read_any_sign_candidates(feature_relative)
        any_sign_events = build_any_sign_event_records(feature_rows, candidates, cfg)
        bundle = build_inference_bundle(
            events, path_records, drift, any_sign_events, feature_rows, cfg
        )
        run_dir = run_relative_dir(draft.run_id, cfg)
        table_payloads = (
            (
                "development_descriptives.csv",
                render_records(bundle.descriptives, DESCRIPTIVE_COLUMNS),
                len(bundle.descriptives),
            ),
            (
                "development_group_comparisons.csv",
                render_records(bundle.group_comparisons, GROUP_COMPARISON_COLUMNS),
                len(bundle.group_comparisons),
            ),
            (
                "development_average_path.csv",
                render_records(bundle.path_summary, PATH_SUMMARY_COLUMNS),
                len(bundle.path_summary),
            ),
            (
                "development_model_results.csv",
                render_records(bundle.model_results, MODEL_RESULT_COLUMNS),
                len(bundle.model_results),
            ),
            (
                "development_bootstrap_results.csv",
                render_bootstraps(bundle.bootstrap_results),
                len(bundle.bootstrap_results),
            ),
            (
                "development_diagnostics.csv",
                render_records(bundle.diagnostics, DIAGNOSTIC_COLUMNS),
                len(bundle.diagnostics),
            ),
        )
        inventory_rows: list[tuple[Path, str, int, str, Sequence[str]]] = []
        for name, content, count in table_payloads:
            relative = run_dir / name
            create_exclusive(relative, content)
            digest = sha256_bytes(content.encode("utf-8"))
            inventory_rows.append((relative, digest, count, "csv", TABLE_SCHEMAS[name]))
            log.record("development_inference_table_written", path=relative.as_posix(), rows=count, sha256=digest)
        for name, payload in bundle.figures:
            relative = run_dir / name
            create_exclusive_bytes(relative, payload)
            digest = sha256_bytes(payload)
            inventory_rows.append((relative, digest, 1, "png", ("binary_png",)))
            log.record("development_inference_figure_written", path=relative.as_posix(), sha256=digest)
        inventory_relative = run_dir / "development_inference_inventory.csv"
        inventory_content = _render_inventory(inventory_rows)
        create_exclusive(inventory_relative, inventory_content)
        inventory_digest = sha256_bytes(inventory_content.encode("utf-8"))
        finished = replace(
            draft,
            input_checksums=input_checksums,
            exclusions={"held_out_rows_analysed": 0},
            eligibility_changes={
                "stage6_event_rows": bundle.event_rows,
                "h5_any_sign_diagnostic_event_rows": bundle.h5_any_sign_events,
                "descriptive_records": len(bundle.descriptives),
                "model_records": len(bundle.model_results),
                "bootstrap_records": len(bundle.bootstrap_results),
                "diagnostic_records": len(bundle.diagnostics),
                "generated_figures": len(bundle.figures),
                "output_artifacts": len(inventory_rows),
            },
        ).completed(inventory_digest, len(inventory_rows), list(INVENTORY_COLUMNS))
        manifest_path = finished.write_for_run("development-inference", cfg)
        manifest_relative = manifest_path.relative_to(paths.project_root().resolve())
        sidecar_relative, manifest_digest = write_manifest_checksum(manifest_relative)
        verify_inference_inventory(manifest_relative)
        log.record(
            "manifest_checksum_written",
            path=sidecar_relative.as_posix(),
            sha256=manifest_digest,
            manifest=manifest_relative.as_posix(),
        )
        log.record(
            "development_inference_completed",
            manifest=manifest_relative.as_posix(),
            stage6_events=bundle.event_rows,
            held_out_outcomes_accessed=False,
            scipy_version=scipy.__version__,
            statsmodels_version=statsmodels.__version__,
        )
    except Exception as error:
        _record_failure(log, error)
        raise
    return DevelopmentInferenceResult(
        run_id=draft.run_id,
        manifest_relative_path=manifest_relative,
        manifest_checksum_relative_path=sidecar_relative,
        log_relative_path=log.relative_path,
        inventory_relative_path=inventory_relative,
        event_count=bundle.event_rows,
        model_rows=len(bundle.model_results),
        bootstrap_rows=len(bundle.bootstrap_results),
    )


def summarise(result: DevelopmentInferenceResult) -> list[str]:
    return [
        f"run id                  {result.run_id}",
        f"development events      {result.event_count}",
        f"model result rows       {result.model_rows}",
        f"bootstrap summaries     {result.bootstrap_rows}",
        f"inventory               {result.inventory_relative_path.as_posix()}",
        f"manifest                {result.manifest_relative_path.as_posix()}",
        f"manifest checksum       {result.manifest_checksum_relative_path.as_posix()}",
        f"run log                 {result.log_relative_path.as_posix()}",
        "held-out outcomes       not accessed",
    ]
