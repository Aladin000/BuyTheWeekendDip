"""Central configuration loading and strict validation.

``config/project.yaml`` holds the locked parameters. This module parses it,
checks it against project spec v1.5, and returns a read-only object.

Loading stops with an error on a missing or unknown key, a wrong type, an
out-of-range value, a conflict between fields, or a locked value that differs
from the project spec.

``LOCKED_PRIMARIES`` is only a guard; analysis code reads values from the
configuration object. It stops a locked parameter from being changed silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from src import paths

EXPECTED_SPEC_VERSION = "1.5"

_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")
_SPEC_VERSION_PATTERN = re.compile(r"^\*\*Specification version:\*\*\s*(\S+)\s*$", re.MULTILINE)

HOUR_SECONDS = 3600


class ConfigError(ValueError):
    """Raised for any invalid, missing, or conflicting configuration value."""


# Read-only configuration container


class ConfigNode(Mapping[str, Any]):
    """Read-only mapping with attribute access."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("configuration is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("configuration is immutable")

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"no configuration key {name!r}") from None

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ConfigNode):
            return self._data == other._data
        return NotImplemented

    def __hash__(self) -> int:  # pragma: no cover - containers are compared, not hashed
        raise TypeError("ConfigNode is unhashable")

    def __repr__(self) -> str:
        return f"ConfigNode({sorted(self._data)})"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dictionary copy."""
        out: dict[str, Any] = {}
        for key, value in self._data.items():
            out[key] = _plain(value)
        return out


def _plain(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


# Schema declaration


@dataclass(frozen=True)
class Scalar:
    kind: str
    choices: tuple[Any, ...] | None = None
    gt: float | None = None
    ge: float | None = None
    lt: float | None = None
    le: float | None = None
    allow_none: bool = False


@dataclass(frozen=True)
class Seq:
    item: Any
    length: int | None = None
    min_length: int | None = None
    unique: bool = False
    allow_none: bool = False


SCHEMA: dict[str, Any] = {
    "meta": {
        "project": Scalar("str"),
        "spec_version": Scalar("str"),
        "spec_frozen_date": Scalar("date"),
        "spec_path": Scalar("relpath"),
        "config_version": Scalar("str"),
    },
    "market": {
        "exchange": Scalar("str"),
        "market_type": Scalar("str"),
        "symbol": Scalar("str"),
        "interval": Scalar("str"),
        "timezone": Scalar("str", choices=("UTC",)),
    },
    "source": {
        "base_url": Scalar("str"),
        "path_pattern": Scalar("str"),
        "checksum_suffix": Scalar("str"),
        "first_month": Scalar("month"),
        "last_month": Scalar("month"),
        "local_digest_algorithm": Scalar("str", choices=("sha256",)),
        "allow_rest_fallback": Scalar("bool"),
        "expected_column_count": Scalar("int", gt=0),
        "expected_column_names": Seq(Scalar("str"), length=12, unique=True, allow_none=True),
        "schema_verified": Scalar("bool"),
        "max_header_rows": Scalar("int", ge=0, le=1),
    },
    "timestamps": {
        "unit_change_at": Scalar("timestamp"),
        "unit_before_change": Scalar("str", choices=("milliseconds",)),
        "unit_from_change": Scalar("str", choices=("microseconds",)),
        "close_time_rule": Scalar("str"),
        "assign_calendar_from": Scalar("str", choices=("close_time",)),
    },
    "data_quality": {
        "nonstandard_close_action": Scalar("str", choices=("exclude_as_gap",)),
        "largest_return_audit_count": Scalar("int", gt=0),
        "largest_open_gap_audit_count": Scalar("int", gt=0),
        "review_zero_base_volume": Scalar("bool"),
        "review_zero_quote_volume": Scalar("bool"),
        "review_zero_trade_count": Scalar("bool"),
    },
    "coverage": {
        "raw_start_inclusive": Scalar("timestamp"),
        "raw_end_exclusive": Scalar("timestamp"),
    },
    "splits": {
        "warmup": {
            "start_inclusive": Scalar("timestamp"),
            "end_exclusive": Scalar("timestamp"),
        },
        "development": {
            "start_inclusive": Scalar("timestamp"),
            "end_exclusive": Scalar("timestamp"),
        },
        "held_out": {
            "start_inclusive": Scalar("timestamp"),
            "end_exclusive": Scalar("timestamp"),
        },
        "membership_timestamp": Scalar("str", choices=("close_time",)),
        "containment": Scalar("str", choices=("signal_and_full_horizon_within_one_split",)),
    },
    "calendar": {
        "weekend_iso_weekdays": Seq(Scalar("int", ge=1, le=7), length=2, unique=True),
        "monday_iso_weekday": Scalar("int", ge=1, le=7),
        "time_blocks": Seq(
            {
                "name": Scalar("str"),
                "start_hour": Scalar("int", ge=0, le=23),
                "end_hour": Scalar("int", ge=0, le=23),
            },
            length=4,
        ),
        "time_block_reference": Scalar("str"),
        "time_block_source": Scalar("str", choices=("close_time",)),
    },
    "returns": {
        "type": Scalar("str", choices=("log",)),
        "require_contiguous_previous_hour": Scalar("bool"),
    },
    "shock": {
        "window_hours": Scalar("int", gt=0),
        "exclude_current_return": Scalar("bool"),
        "ddof": Scalar("int", ge=0),
        "negative_threshold": Scalar("float", lt=0.0),
        "positive_threshold": Scalar("float", gt=0.0),
        "strict_inequality": Scalar("bool"),
    },
    "events": {
        "cooldown_hours": Scalar("int", gt=0),
        "cooldown_boundary_eligible": Scalar("bool"),
        "retain": Scalar("str", choices=("first_candidate",)),
        "sign_specific": Scalar("bool"),
        "elapsed_time_basis": Scalar("str", choices=("close_time",)),
        "continuous_through_warmup_and_splits": Scalar("bool"),
        "acceptance_ignores_future_information": Scalar("bool"),
    },
    "activity": {
        "primary_measure": Scalar("str", choices=("quote_volume",)),
        "secondary_measure": Scalar("str", choices=("trade_count",)),
        "window_hours": Scalar("int", gt=0),
        "exclude_shock_hour": Scalar("bool"),
        "reference_window": Scalar("int", gt=0),
        "reference_same_segment_consecutive": Scalar("bool"),
        "tie_rule": Scalar("str", choices=("less_than_or_equal",)),
        "required_history_hours": Scalar("int", gt=0),
        "low_activity_threshold": {
            "numerator": Scalar("int", gt=0),
            "denominator": Scalar("int", gt=0),
        },
    },
    "controls": {
        "rv_window_hours": Scalar("int", gt=0),
        "rv_excludes_shock_return": Scalar("bool"),
        "shock_magnitude_definition": Scalar("str", choices=("negative_z",)),
    },
    "outcomes": {
        "primary_horizon_hours": Scalar("int", gt=0),
        "secondary_horizon_hours": Seq(Scalar("int", gt=0), length=2, unique=True),
        "require_contiguous_path": Scalar("bool"),
        "cumulative_path_hours": Scalar("int", gt=0),
    },
    "inference": {
        "alpha": Scalar("float", gt=0.0, lt=1.0),
        "covariance": Scalar("str", choices=("HC3",)),
        "h1_h4_side": Scalar("str", choices=("one_sided_greater",)),
        "confidence_level": Scalar("float", gt=0.0, lt=1.0),
        "confidence_side": Scalar("str", choices=("two_sided",)),
        "t_reference": Scalar("str", choices=("finite_sample_residual_df",)),
        "holm": {
            "method": Scalar("str", choices=("holm",)),
            "family_size": Scalar("int", gt=0),
            "members": Seq(Scalar("str"), length=4, unique=True),
            "unestimable_placeholder_p": Scalar("float", ge=0.0, le=1.0),
        },
        "h5": {
            "side": Scalar("str", choices=("two_sided",)),
            "in_holm_family": Scalar("bool"),
        },
        "model3_min_events_per_cell": Scalar("int", gt=0),
        "model3_min_n_per_coefficient": Scalar("int", gt=0),
        "cooks_distance_rule": Scalar("str", choices=("4/N",)),
    },
    "bootstrap": {
        "method": Scalar("str", choices=("calendar_month_block_pairs",)),
        "block_unit": Scalar("str", choices=("utc_month",)),
        "target_valid_replicates": Scalar("int", gt=0),
        "max_attempts": Scalar("int", gt=0),
        "with_replacement": Scalar("bool"),
    },
    "strategy": {
        "initial_capital": Scalar("float", gt=0.0),
        "variants": Seq(Scalar("str"), length=4, unique=True),
        "titled_primary": Scalar("str"),
        "principal_benchmark": Scalar("str"),
        "entry_offset_bars": Scalar("int", gt=0),
        "exit_offset_bars": Scalar("int", gt=0),
        "hold_hours": Scalar("int", gt=0),
        "delayed_entry_offset_bars": Scalar("int", gt=0),
        "delayed_exit_offset_bars": Scalar("int", gt=0),
        "max_concurrent_positions": Scalar("int", gt=0),
        "allocation_fraction": Scalar("float", gt=0.0, le=1.0),
        "allow_leverage": Scalar("bool"),
        "allow_short": Scalar("bool"),
        "use_stops": Scalar("bool"),
        "sparse_trade_label_threshold": Scalar("int", gt=0),
        "benchmarks": Seq(Scalar("str"), length=4, unique=True),
    },
    "costs": {
        "scenarios": {
            "low": {
                "fee_per_side": Scalar("float", ge=0.0),
                "slippage_per_side": Scalar("float", ge=0.0),
            },
            "primary": {
                "fee_per_side": Scalar("float", ge=0.0),
                "slippage_per_side": Scalar("float", ge=0.0),
            },
            "high": {
                "fee_per_side": Scalar("float", ge=0.0),
                "slippage_per_side": Scalar("float", ge=0.0),
            },
        },
        "active_scenario": Scalar("str"),
    },
    "performance": {
        "sharpe_periods_per_year": Scalar("int", gt=0),
        "risk_free_rate": Scalar("float", ge=0.0),
        "annualisation_basis": Scalar("str", choices=("exact_calendar_days",)),
        "daily_mark": Scalar("str", choices=("last_utc_hourly_mark",)),
    },
    "robustness": {
        "thresholds": Seq(Scalar("float", lt=0.0), length=3, unique=True),
        "horizons": Seq(Scalar("int", gt=0), length=3, unique=True),
        "activity_measures": Seq(Scalar("str"), length=2, unique=True),
        "execution": Seq(Scalar("str"), length=2, unique=True),
        "cost_scenarios": Seq(Scalar("str"), length=3, unique=True),
        "inference": Seq(Scalar("str"), length=2, unique=True),
        "positive_comparison_threshold": Scalar("float", gt=0.0),
        "diagnostic_cooldown_hours": Seq(Scalar("int", gt=0), length=2, unique=True),
        "diagnostics": Seq(Scalar("str"), length=4, unique=True),
        "one_family_at_a_time": Scalar("bool"),
    },
    "reproducibility": {
        "seed": Scalar("int", ge=0),
        "bit_generator": Scalar("str", choices=("PCG64",)),
        "record_numpy_version": Scalar("bool"),
        "atol": Scalar("float", gt=0.0),
        "rtol": Scalar("float", gt=0.0),
        "deterministic_sorting": Scalar("bool"),
    },
    "paths": {
        "data_raw": Scalar("relpath"),
        "data_interim": Scalar("relpath"),
        "data_processed": Scalar("relpath"),
        "outputs_figures": Scalar("relpath"),
        "outputs_tables": Scalar("relpath"),
        "outputs_runs": Scalar("relpath"),
    },
}


# Locked values from project spec v1.5. A configuration that differs from any
# of these is rejected.
LOCKED_PRIMARIES: dict[str, Any] = {
    "market.exchange": "binance",
    "market.market_type": "spot",
    "market.symbol": "BTCUSDT",
    "market.interval": "1h",
    "source.expected_column_count": 12,
    "source.expected_column_names": [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore",
    ],
    "source.schema_verified": True,
    "data_quality.nonstandard_close_action": "exclude_as_gap",
    "data_quality.largest_return_audit_count": 20,
    "data_quality.largest_open_gap_audit_count": 20,
    "data_quality.review_zero_base_volume": True,
    "data_quality.review_zero_quote_volume": True,
    "data_quality.review_zero_trade_count": True,
    "coverage.raw_start_inclusive": "2019-10-01T00:00:00Z",
    "coverage.raw_end_exclusive": "2026-08-01T00:00:00Z",
    "splits.development.start_inclusive": "2020-01-01T00:00:00Z",
    "splits.development.end_exclusive": "2024-01-01T00:00:00Z",
    "splits.held_out.start_inclusive": "2024-01-01T00:00:00Z",
    "splits.held_out.end_exclusive": "2026-08-01T00:00:00Z",
    "shock.window_hours": 720,
    "shock.ddof": 1,
    "shock.negative_threshold": -2.5,
    "shock.positive_threshold": 2.5,
    "events.cooldown_hours": 24,
    "activity.window_hours": 24,
    "activity.reference_window": 2160,
    "activity.required_history_hours": 2184,
    "activity.low_activity_threshold.numerator": 1,
    "activity.low_activity_threshold.denominator": 3,
    "controls.rv_window_hours": 24,
    "outcomes.primary_horizon_hours": 12,
    "outcomes.cumulative_path_hours": 24,
    "inference.alpha": 0.05,
    "inference.confidence_level": 0.95,
    "inference.holm.family_size": 4,
    "inference.holm.unestimable_placeholder_p": 1.0,
    "inference.model3_min_events_per_cell": 10,
    "inference.model3_min_n_per_coefficient": 10,
    "bootstrap.target_valid_replicates": 10000,
    "bootstrap.max_attempts": 20000,
    "strategy.initial_capital": 1.0,
    "strategy.entry_offset_bars": 1,
    "strategy.exit_offset_bars": 12,
    "strategy.hold_hours": 12,
    "strategy.delayed_entry_offset_bars": 2,
    "strategy.delayed_exit_offset_bars": 13,
    "strategy.max_concurrent_positions": 1,
    "strategy.allocation_fraction": 1.0,
    "strategy.allow_leverage": False,
    "strategy.allow_short": False,
    "strategy.use_stops": False,
    "strategy.sparse_trade_label_threshold": 20,
    "costs.scenarios.low.fee_per_side": 0.0005,
    "costs.scenarios.low.slippage_per_side": 0.0001,
    "costs.scenarios.primary.fee_per_side": 0.001,
    "costs.scenarios.primary.slippage_per_side": 0.0002,
    "costs.scenarios.high.fee_per_side": 0.001,
    "costs.scenarios.high.slippage_per_side": 0.001,
    "performance.sharpe_periods_per_year": 365,
    "performance.risk_free_rate": 0.0,
    "reproducibility.seed": 20260901,
    "reproducibility.bit_generator": "PCG64",
    "reproducibility.atol": 1e-12,
    "reproducibility.rtol": 1e-12,
}


# Scalar conversion


def _convert_scalar(value: Any, spec: Scalar, where: str) -> Any:
    if value is None:
        if spec.allow_none:
            return None
        raise ConfigError(f"{where}: value must not be null")

    if spec.kind == "bool":
        if type(value) is not bool:
            raise ConfigError(f"{where}: expected a boolean, got {type(value).__name__}")
        converted: Any = value
    elif spec.kind == "int":
        # bool is a subclass of int; reject it explicitly.
        if type(value) is not int:
            raise ConfigError(f"{where}: expected an integer, got {type(value).__name__}")
        converted = value
    elif spec.kind == "float":
        if type(value) is not float:
            raise ConfigError(
                f"{where}: expected a float written with a decimal point, "
                f"got {type(value).__name__}"
            )
        converted = value
    elif spec.kind == "str":
        if type(value) is not str:
            raise ConfigError(f"{where}: expected a string, got {type(value).__name__}")
        converted = value
    elif spec.kind == "timestamp":
        converted = _parse_timestamp(value, where)
    elif spec.kind == "date":
        converted = _parse_date(value, where)
    elif spec.kind == "month":
        converted = _parse_month(value, where)
    elif spec.kind == "relpath":
        converted = _parse_relpath(value, where)
    else:  # pragma: no cover - guards a schema authoring mistake
        raise ConfigError(f"{where}: unknown schema kind {spec.kind!r}")

    if spec.choices is not None and converted not in spec.choices:
        raise ConfigError(f"{where}: {converted!r} is not one of {list(spec.choices)}")

    for bound, symbol, ok in (
        (spec.gt, ">", lambda a, b: a > b),
        (spec.ge, ">=", lambda a, b: a >= b),
        (spec.lt, "<", lambda a, b: a < b),
        (spec.le, "<=", lambda a, b: a <= b),
    ):
        if bound is not None and not ok(converted, bound):
            raise ConfigError(f"{where}: {converted!r} must be {symbol} {bound}")

    return converted


def _parse_timestamp(value: Any, where: str) -> datetime:
    if type(value) is not str:
        raise ConfigError(
            f"{where}: expected a quoted UTC timestamp string 'YYYY-MM-DDTHH:MM:SSZ', "
            f"got {type(value).__name__}"
        )
    if not _TIMESTAMP_PATTERN.match(value):
        raise ConfigError(
            f"{where}: {value!r} is not a timezone-aware UTC timestamp of the form "
            "'YYYY-MM-DDTHH:MM:SSZ'"
        )
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if parsed.minute or parsed.second:
        raise ConfigError(f"{where}: {value!r} is not aligned to a whole UTC hour")
    return parsed


def _parse_date(value: Any, where: str) -> date:
    if type(value) is not str or not _DATE_PATTERN.match(value):
        raise ConfigError(f"{where}: expected a quoted 'YYYY-MM-DD' date, got {value!r}")
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_month(value: Any, where: str) -> tuple[int, int]:
    if type(value) is not str or not _MONTH_PATTERN.match(value):
        raise ConfigError(f"{where}: expected a quoted 'YYYY-MM' month, got {value!r}")
    year, month = (int(part) for part in value.split("-"))
    if not 1 <= month <= 12:
        raise ConfigError(f"{where}: {value!r} has an out-of-range month")
    return year, month


def _parse_relpath(value: Any, where: str) -> str:
    if type(value) is not str:
        raise ConfigError(f"{where}: expected a relative path string, got {type(value).__name__}")
    if not value:
        raise ConfigError(f"{where}: relative path must not be empty")
    if "\\" in value:
        raise ConfigError(f"{where}: {value!r} must use forward slashes")
    candidate = Path(value)
    if candidate.is_absolute() or value.startswith("~"):
        raise ConfigError(f"{where}: {value!r} must be relative to the project root")
    if ".." in candidate.parts:
        raise ConfigError(f"{where}: {value!r} must not escape the project root")
    return value


# Structural validation


def _validate_node(value: Any, spec: Any, where: str) -> Any:
    if isinstance(spec, dict):
        return _validate_mapping(value, spec, where)
    if isinstance(spec, Seq):
        return _validate_sequence(value, spec, where)
    if isinstance(spec, Scalar):
        return _convert_scalar(value, spec, where)
    raise ConfigError(f"{where}: unusable schema node")  # pragma: no cover


def _validate_mapping(value: Any, spec: dict[str, Any], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where or 'config'}: expected a mapping, got {type(value).__name__}")
    missing = sorted(set(spec) - set(value))
    if missing:
        raise ConfigError(f"{where or 'config'}: missing required key(s) {missing}")
    unknown = sorted(set(value) - set(spec))
    if unknown:
        raise ConfigError(f"{where or 'config'}: unknown key(s) {unknown}")
    out: dict[str, Any] = {}
    for key, child_spec in spec.items():
        child_where = f"{where}.{key}" if where else key
        out[key] = _validate_node(value[key], child_spec, child_where)
    return out


def _validate_sequence(value: Any, spec: Seq, where: str) -> Any:
    if value is None:
        if spec.allow_none:
            return None
        raise ConfigError(f"{where}: value must not be null")
    if not isinstance(value, list):
        raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
    if spec.length is not None and len(value) != spec.length:
        raise ConfigError(f"{where}: expected exactly {spec.length} entries, got {len(value)}")
    if spec.min_length is not None and len(value) < spec.min_length:
        raise ConfigError(f"{where}: expected at least {spec.min_length} entries, got {len(value)}")
    items = [_validate_node(item, spec.item, f"{where}[{index}]") for index, item in enumerate(value)]
    if spec.unique:
        seen: list[Any] = []
        for item in items:
            key = item if not isinstance(item, dict) else tuple(sorted(item.items()))
            if key in seen:
                raise ConfigError(f"{where}: duplicate entry {item!r}")
            seen.append(key)
    return items


# Cross-field checks


def _dotted(data: Mapping[str, Any], dotted_key: str) -> Any:
    node: Any = data
    for part in dotted_key.split("."):
        node = node[part]
    return node


def _check_locked_primaries(data: Mapping[str, Any]) -> None:
    for dotted_key, expected in LOCKED_PRIMARIES.items():
        observed = _dotted(data, dotted_key)
        if isinstance(observed, datetime):
            observed_repr = observed.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            observed_repr = observed
        if type(observed_repr) is not type(expected) or observed_repr != expected:
            raise ConfigError(
                f"{dotted_key}: {observed_repr!r} conflicts with specification "
                f"v{EXPECTED_SPEC_VERSION}, which locks {expected!r}"
            )


def _check_cross_fields(data: Mapping[str, Any]) -> None:
    if data["meta"]["spec_version"] != EXPECTED_SPEC_VERSION:
        raise ConfigError(
            f"meta.spec_version: {data['meta']['spec_version']!r} is not the active "
            f"specification version {EXPECTED_SPEC_VERSION!r}"
        )

    splits = data["splits"]
    for name in ("warmup", "development", "held_out"):
        start = splits[name]["start_inclusive"]
        end = splits[name]["end_exclusive"]
        if start >= end:
            raise ConfigError(f"splits.{name}: start {start} must precede end {end}")

    if splits["warmup"]["end_exclusive"] != splits["development"]["start_inclusive"]:
        raise ConfigError(
            "splits: warm-up must end exactly where the development split begins"
        )
    if splits["development"]["end_exclusive"] != splits["held_out"]["start_inclusive"]:
        raise ConfigError(
            "splits: development must end exactly where the held-out split begins"
        )
    if splits["development"]["start_inclusive"] >= splits["held_out"]["start_inclusive"]:
        raise ConfigError("splits: development must precede held_out")
    if splits["held_out"]["start_inclusive"] < splits["development"]["end_exclusive"]:
        raise ConfigError("splits: development and held_out must not overlap")

    coverage = data["coverage"]
    if coverage["raw_start_inclusive"] >= coverage["raw_end_exclusive"]:
        raise ConfigError("coverage: raw start must precede raw end")
    if coverage["raw_start_inclusive"] != splits["warmup"]["start_inclusive"]:
        raise ConfigError("coverage.raw_start_inclusive must equal splits.warmup.start_inclusive")
    if coverage["raw_end_exclusive"] != splits["held_out"]["end_exclusive"]:
        raise ConfigError("coverage.raw_end_exclusive must equal splits.held_out.end_exclusive")
    if coverage["raw_start_inclusive"] >= splits["development"]["start_inclusive"]:
        raise ConfigError(
            "coverage.raw_start_inclusive must precede the development start so rolling "
            "features have warm-up history"
        )

    timestamps = data["timestamps"]
    if not (
        coverage["raw_start_inclusive"]
        < timestamps["unit_change_at"]
        < coverage["raw_end_exclusive"]
    ):
        raise ConfigError("timestamps.unit_change_at must fall inside the acquisition coverage")

    activity = data["activity"]
    expected_history = activity["window_hours"] + activity["reference_window"]
    if activity["required_history_hours"] != expected_history:
        raise ConfigError(
            f"activity.required_history_hours must equal window_hours + reference_window "
            f"({expected_history}), got {activity['required_history_hours']}"
        )

    warmup_hours = int(
        (
            splits["development"]["start_inclusive"] - coverage["raw_start_inclusive"]
        ).total_seconds()
        // HOUR_SECONDS
    )
    shock_requirement = data["shock"]["window_hours"] + 1
    for requirement, label in (
        (activity["required_history_hours"], "activity history"),
        (shock_requirement, "shock window plus the current return"),
        (data["controls"]["rv_window_hours"], "RV24 history"),
    ):
        if warmup_hours < requirement:
            raise ConfigError(
                f"coverage: warm-up supplies {warmup_hours} hours before the development "
                f"start, fewer than the {requirement} required for {label}"
            )

    shock = data["shock"]
    if shock["positive_threshold"] != -shock["negative_threshold"]:
        raise ConfigError(
            "shock: the positive threshold must be the symmetric counterpart of the negative one"
        )

    robustness = data["robustness"]
    if robustness["positive_comparison_threshold"] != shock["positive_threshold"]:
        raise ConfigError(
            "robustness.positive_comparison_threshold must equal shock.positive_threshold"
        )
    if shock["negative_threshold"] not in robustness["thresholds"]:
        raise ConfigError(
            "robustness.thresholds must contain the primary negative threshold"
        )
    if robustness["diagnostic_cooldown_hours"] and data["events"]["cooldown_hours"] in robustness[
        "diagnostic_cooldown_hours"
    ]:
        raise ConfigError(
            "robustness.diagnostic_cooldown_hours must list alternatives to the primary cooldown"
        )
    if list(robustness["activity_measures"]) != [
        activity["primary_measure"],
        activity["secondary_measure"],
    ]:
        raise ConfigError(
            "robustness.activity_measures must be the primary then secondary activity measure"
        )
    if sorted(robustness["cost_scenarios"]) != sorted(data["costs"]["scenarios"]):
        raise ConfigError("robustness.cost_scenarios must match the declared cost scenarios")

    outcomes = data["outcomes"]
    expected_horizons = sorted({outcomes["primary_horizon_hours"], *outcomes["secondary_horizon_hours"]})
    if sorted(robustness["horizons"]) != expected_horizons:
        raise ConfigError(
            f"robustness.horizons must be the primary and secondary horizons {expected_horizons}"
        )
    if outcomes["primary_horizon_hours"] not in robustness["horizons"]:
        raise ConfigError("robustness.horizons must contain the primary outcome horizon")
    if outcomes["cumulative_path_hours"] < max(robustness["horizons"]):
        raise ConfigError(
            "outcomes.cumulative_path_hours must cover the longest reported horizon"
        )

    calendar = data["calendar"]
    block_names = [block["name"] for block in calendar["time_blocks"]]
    if len(set(block_names)) != len(block_names):
        raise ConfigError("calendar.time_blocks: block names must be unique")
    if calendar["time_block_reference"] not in block_names:
        raise ConfigError(
            f"calendar.time_block_reference {calendar['time_block_reference']!r} is not a "
            f"declared block {block_names}"
        )
    covered: list[int] = []
    for block in sorted(calendar["time_blocks"], key=lambda item: item["start_hour"]):
        if block["end_hour"] < block["start_hour"]:
            raise ConfigError(f"calendar.time_blocks: block {block['name']!r} ends before it starts")
        covered.extend(range(block["start_hour"], block["end_hour"] + 1))
    if covered != list(range(24)):
        raise ConfigError(
            "calendar.time_blocks must partition the 24 UTC hours exactly once"
        )
    if calendar["monday_iso_weekday"] in calendar["weekend_iso_weekdays"]:
        raise ConfigError("calendar: Monday must not be listed as a weekend day")

    strategy = data["strategy"]
    if strategy["titled_primary"] not in strategy["variants"]:
        raise ConfigError("strategy.titled_primary must be one of the declared variants")
    if strategy["principal_benchmark"] not in strategy["variants"]:
        raise ConfigError("strategy.principal_benchmark must be one of the declared variants")
    if strategy["principal_benchmark"] not in strategy["benchmarks"]:
        raise ConfigError("strategy.principal_benchmark must appear in the benchmark list")
    if strategy["hold_hours"] != outcomes["primary_horizon_hours"]:
        raise ConfigError(
            "strategy.hold_hours must equal the primary outcome horizon"
        )
    if strategy["exit_offset_bars"] != strategy["entry_offset_bars"] + strategy["hold_hours"] - 1:
        raise ConfigError(
            "strategy: exit offset must be entry offset plus hold hours minus one "
            f"(expected {strategy['entry_offset_bars'] + strategy['hold_hours'] - 1})"
        )
    if strategy["delayed_entry_offset_bars"] != strategy["entry_offset_bars"] + 1:
        raise ConfigError("strategy.delayed_entry_offset_bars must be one bar after primary entry")
    if strategy["delayed_exit_offset_bars"] != strategy["exit_offset_bars"] + 1:
        raise ConfigError("strategy.delayed_exit_offset_bars must be one bar after primary exit")

    costs = data["costs"]
    if costs["active_scenario"] not in costs["scenarios"]:
        raise ConfigError(
            f"costs.active_scenario {costs['active_scenario']!r} is not a declared scenario"
        )
    ordered = ("low", "primary", "high")
    for field in ("fee_per_side", "slippage_per_side"):
        values = [costs["scenarios"][name][field] for name in ordered]
        if values != sorted(values):
            raise ConfigError(
                f"costs: {field} must be non-decreasing across low, primary, high scenarios"
            )
    round_trips = [
        2 * (costs["scenarios"][name]["fee_per_side"] + costs["scenarios"][name]["slippage_per_side"])
        for name in ordered
    ]
    if round_trips != sorted(round_trips):
        raise ConfigError("costs: round-trip cost must be non-decreasing from low to high")

    inference = data["inference"]
    if inference["holm"]["family_size"] != len(inference["holm"]["members"]):
        raise ConfigError(
            "inference.holm.family_size must equal the number of declared family members"
        )
    if abs(inference["confidence_level"] - (1.0 - inference["alpha"])) > 1e-12:
        raise ConfigError(
            "inference.confidence_level must equal 1 - alpha"
        )
    if inference["h5"]["in_holm_family"]:
        raise ConfigError(
            "inference.h5.in_holm_family must be false; H5 sits outside the H1-H4 family"
        )

    bootstrap = data["bootstrap"]
    if bootstrap["target_valid_replicates"] > bootstrap["max_attempts"]:
        raise ConfigError(
            "bootstrap.target_valid_replicates must not exceed max_attempts"
        )

    reproducibility = data["reproducibility"]
    if reproducibility["atol"] != reproducibility["rtol"]:
        raise ConfigError(
            "reproducibility: the default absolute and relative tolerances are locked equal"
        )

    source = data["source"]
    if source["allow_rest_fallback"]:
        raise ConfigError(
            "source.allow_rest_fallback must be false; mixing REST data is forbidden"
        )
    if source["schema_verified"] and source["expected_column_names"] is None:
        raise ConfigError(
            "source.expected_column_names must be populated once schema_verified is true"
        )
    if not source["schema_verified"] and source["expected_column_names"] is not None:
        raise ConfigError(
            "source.expected_column_names must stay null until the source schema is verified"
        )
    if source["expected_column_names"] is not None and len(
        source["expected_column_names"]
    ) != source["expected_column_count"]:
        raise ConfigError(
            "source.expected_column_names must have expected_column_count entries"
        )

    months = _derive_archive_months(
        coverage["raw_start_inclusive"], coverage["raw_end_exclusive"]
    )
    if months[0] != source["first_month"]:
        raise ConfigError(
            f"source.first_month {source['first_month']} does not match the coverage start "
            f"month {months[0]}"
        )
    if months[-1] != source["last_month"]:
        raise ConfigError(
            f"source.last_month {source['last_month']} does not match the coverage end "
            f"month {months[-1]}"
        )

    # Run last so a more specific conflict above is reported first.
    _check_locked_primaries(data)


def _derive_archive_months(
    start_inclusive: datetime, end_exclusive: datetime
) -> tuple[tuple[int, int], ...]:
    """Monthly archive periods covering ``[start, end)``."""
    months: list[tuple[int, int]] = []
    year, month = start_inclusive.year, start_inclusive.month
    # The exclusive end is a month boundary, so the last archive month is the
    # month immediately before it.
    last_year, last_month = end_exclusive.year, end_exclusive.month - 1
    if last_month == 0:
        last_year, last_month = last_year - 1, 12
    while (year, month) <= (last_year, last_month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return tuple(months)


# Public loading interface


def validate_config(raw: Mapping[str, Any]) -> ConfigNode:
    """Validate a raw configuration mapping and return a read-only node.

    Reads no file, so tests can validate an edited copy directly.
    """
    validated = _validate_mapping(raw, SCHEMA, "")
    _check_cross_fields(validated)
    return _freeze(validated)


def read_raw_config(path: Path | None = None) -> dict[str, Any]:
    """Read the configuration YAML without validating it."""
    target = paths.require_file(path if path is not None else paths.config_path())
    with target.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{target}: expected a YAML mapping at the top level")
    return raw


def spec_document_version(path: Path | None = None) -> str:
    """Version declared in the header of the project spec."""
    target = paths.require_file(path if path is not None else paths.spec_path())
    match = _SPEC_VERSION_PATTERN.search(target.read_text(encoding="utf-8"))
    if match is None:
        raise ConfigError(f"{target}: no '**Specification version:**' header found")
    return match.group(1)


def load_config(path: Path | None = None, *, check_spec_document: bool = True) -> ConfigNode:
    """Load, validate, and return the central configuration.

    The path is found from this module's location, not the working directory.
    """
    config = validate_config(read_raw_config(path))
    if check_spec_document:
        declared = spec_document_version()
        if declared != config.meta.spec_version:
            raise ConfigError(
                f"meta.spec_version {config.meta.spec_version!r} disagrees with "
                f"{config.meta.spec_path} header version {declared!r}"
            )
    return config


# Configuration-derived helpers


def low_activity_threshold(config: ConfigNode) -> Fraction:
    """Low-activity percentile cut-off as an exact fraction."""
    node = config.activity.low_activity_threshold
    return Fraction(node.numerator, node.denominator)


def is_low_activity(count_at_or_below: int, reference_count: int, config: ConfigNode) -> bool:
    """Apply ``LowActivity = 1(P < 1/3)`` with exact fractions.

    ``count_at_or_below`` is the number of reference values at or below the
    current value. ``reference_count`` must equal the configured reference
    window (2,160); a shorter window is not allowed, so any other value is a
    caller error.

    The comparison is strict, so a percentile on the cut-off is not low
    activity.
    """
    expected = config.activity.reference_window
    if type(reference_count) is not int:
        raise ConfigError("reference_count must be an integer")
    if reference_count != expected:
        raise ConfigError(
            f"reference_count must equal the configured reference window {expected}, "
            f"got {reference_count}; a shorter or longer reference window is not allowed"
        )
    if type(count_at_or_below) is not int:
        raise ConfigError("count_at_or_below must be an integer")
    if not 0 <= count_at_or_below <= reference_count:
        raise ConfigError("count_at_or_below must lie within [0, reference_count]")
    return Fraction(count_at_or_below, reference_count) < low_activity_threshold(config)


def split_bounds(config: ConfigNode, name: str) -> tuple[datetime, datetime]:
    """Half-open ``[start, end)`` bounds of a named split."""
    try:
        split = config.splits[name]
    except KeyError:
        raise ConfigError(f"unknown split {name!r}") from None
    return split.start_inclusive, split.end_exclusive


def in_split(config: ConfigNode, moment: datetime, name: str) -> bool:
    """Whether ``moment`` lies in the half-open interval of ``name``."""
    if moment.tzinfo is None:
        raise ConfigError("split membership requires a timezone-aware timestamp")
    start, end = split_bounds(config, name)
    return start <= moment < end


def split_of(config: ConfigNode, moment: datetime) -> str | None:
    """Name of the split containing ``moment``, or ``None`` when outside all."""
    for name in ("warmup", "development", "held_out"):
        if in_split(config, moment, name):
            return name
    return None


def source_timestamp_unit(config: ConfigNode, moment: datetime) -> str:
    """Archive timestamp unit in use at ``moment``."""
    if moment.tzinfo is None:
        raise ConfigError("timestamp unit lookup requires a timezone-aware timestamp")
    node = config.timestamps
    if moment < node.unit_change_at:
        return node.unit_before_change
    return node.unit_from_change


def archive_months(config: ConfigNode) -> tuple[tuple[int, int], ...]:
    """Monthly archive periods implied by the acquisition coverage."""
    return _derive_archive_months(
        config.coverage.raw_start_inclusive, config.coverage.raw_end_exclusive
    )


def warmup_hours(config: ConfigNode) -> int:
    """Whole hours of history available before the development split starts."""
    delta = config.splits.development.start_inclusive - config.coverage.raw_start_inclusive
    return int(delta.total_seconds() // HOUR_SECONDS)


def cost_scenario(config: ConfigNode, name: str | None = None) -> ConfigNode:
    """Fee and slippage rates for a named cost scenario."""
    scenario = name if name is not None else config.costs.active_scenario
    try:
        return config.costs.scenarios[scenario]
    except KeyError:
        raise ConfigError(f"unknown cost scenario {scenario!r}") from None
