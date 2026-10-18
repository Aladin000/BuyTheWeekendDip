"""Configuration values and types agree with version 1.5 of the project spec.

Expected values are copied from the project spec, not imported from
``src.config``, so these tests check both the configuration file and the
loader's locked-value guard independently.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from fractions import Fraction

import pytest

from src.config import (
    EXPECTED_SPEC_VERSION,
    ConfigError,
    cost_scenario,
    is_low_activity,
    low_activity_threshold,
    spec_document_version,
    validate_config,
)


def _utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# (dotted key, expected value, expected type).
LOCKED_VALUES: tuple[tuple[str, object, type], ...] = (
    # Market and source
    ("market.exchange", "binance", str),
    ("market.market_type", "spot", str),
    ("market.symbol", "BTCUSDT", str),
    ("market.interval", "1h", str),
    ("market.timezone", "UTC", str),
    ("source.base_url", "https://data.binance.vision", str),
    ("source.checksum_suffix", ".CHECKSUM", str),
    ("source.first_month", (2019, 10), tuple),
    ("source.last_month", (2026, 7), tuple),
    ("source.local_digest_algorithm", "sha256", str),
    ("source.allow_rest_fallback", False, bool),
    ("source.expected_column_count", 12, int),
    (
        "source.expected_column_names",
        (
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume", "ignore",
        ),
        tuple,
    ),
    ("source.schema_verified", True, bool),
    ("source.max_header_rows", 1, int),
    # Timestamps
    ("timestamps.unit_change_at", _utc("2025-01-01T00:00:00Z"), datetime),
    ("timestamps.unit_before_change", "milliseconds", str),
    ("timestamps.unit_from_change", "microseconds", str),
    ("timestamps.assign_calendar_from", "close_time", str),
    ("data_quality.nonstandard_close_action", "exclude_as_gap", str),
    ("data_quality.largest_return_audit_count", 20, int),
    ("data_quality.largest_open_gap_audit_count", 20, int),
    ("data_quality.review_zero_base_volume", True, bool),
    ("data_quality.review_zero_quote_volume", True, bool),
    ("data_quality.review_zero_trade_count", True, bool),
    # Coverage and splits
    ("coverage.raw_start_inclusive", _utc("2019-10-01T00:00:00Z"), datetime),
    ("coverage.raw_end_exclusive", _utc("2026-08-01T00:00:00Z"), datetime),
    ("splits.warmup.start_inclusive", _utc("2019-10-01T00:00:00Z"), datetime),
    ("splits.warmup.end_exclusive", _utc("2020-01-01T00:00:00Z"), datetime),
    ("splits.development.start_inclusive", _utc("2020-01-01T00:00:00Z"), datetime),
    ("splits.development.end_exclusive", _utc("2024-01-01T00:00:00Z"), datetime),
    ("splits.held_out.start_inclusive", _utc("2024-01-01T00:00:00Z"), datetime),
    ("splits.held_out.end_exclusive", _utc("2026-08-01T00:00:00Z"), datetime),
    ("splits.membership_timestamp", "close_time", str),
    # Calendar
    ("calendar.weekend_iso_weekdays", (6, 7), tuple),
    ("calendar.monday_iso_weekday", 1, int),
    ("calendar.time_block_reference", "00-05", str),
    ("calendar.time_block_source", "close_time", str),
    # Returns and shock
    ("returns.type", "log", str),
    ("returns.require_contiguous_previous_hour", True, bool),
    ("shock.window_hours", 720, int),
    ("shock.exclude_current_return", True, bool),
    ("shock.ddof", 1, int),
    ("shock.negative_threshold", -2.5, float),
    ("shock.positive_threshold", 2.5, float),
    ("shock.strict_inequality", True, bool),
    # Events
    ("events.cooldown_hours", 24, int),
    ("events.cooldown_boundary_eligible", True, bool),
    ("events.retain", "first_candidate", str),
    ("events.sign_specific", True, bool),
    ("events.elapsed_time_basis", "close_time", str),
    ("events.continuous_through_warmup_and_splits", True, bool),
    ("events.acceptance_ignores_future_information", True, bool),
    # Activity
    ("activity.primary_measure", "quote_volume", str),
    ("activity.secondary_measure", "trade_count", str),
    ("activity.window_hours", 24, int),
    ("activity.exclude_shock_hour", True, bool),
    ("activity.reference_window", 2160, int),
    ("activity.reference_same_segment_consecutive", True, bool),
    ("activity.tie_rule", "less_than_or_equal", str),
    ("activity.required_history_hours", 2184, int),
    ("activity.low_activity_threshold.numerator", 1, int),
    ("activity.low_activity_threshold.denominator", 3, int),
    # Controls and outcomes
    ("controls.rv_window_hours", 24, int),
    ("controls.rv_excludes_shock_return", True, bool),
    ("controls.shock_magnitude_definition", "negative_z", str),
    ("outcomes.primary_horizon_hours", 12, int),
    ("outcomes.secondary_horizon_hours", (6, 24), tuple),
    ("outcomes.require_contiguous_path", True, bool),
    ("outcomes.cumulative_path_hours", 24, int),
    # Inference
    ("inference.alpha", 0.05, float),
    ("inference.covariance", "HC3", str),
    ("inference.h1_h4_side", "one_sided_greater", str),
    ("inference.confidence_level", 0.95, float),
    ("inference.confidence_side", "two_sided", str),
    ("inference.t_reference", "finite_sample_residual_df", str),
    ("inference.holm.method", "holm", str),
    ("inference.holm.family_size", 4, int),
    (
        "inference.holm.members",
        ("H1_model0", "H2_model1", "H3_model2", "H4_model2"),
        tuple,
    ),
    ("inference.holm.unestimable_placeholder_p", 1.0, float),
    ("inference.h5.side", "two_sided", str),
    ("inference.h5.in_holm_family", False, bool),
    ("inference.model3_min_events_per_cell", 10, int),
    ("inference.model3_min_n_per_coefficient", 10, int),
    ("inference.cooks_distance_rule", "4/N", str),
    # Bootstrap
    ("bootstrap.method", "calendar_month_block_pairs", str),
    ("bootstrap.block_unit", "utc_month", str),
    ("bootstrap.target_valid_replicates", 10000, int),
    ("bootstrap.max_attempts", 20000, int),
    ("bootstrap.with_replacement", True, bool),
    # Strategy and execution
    ("strategy.initial_capital", 1.0, float),
    (
        "strategy.variants",
        ("all_shock", "weekend", "low_activity", "combined"),
        tuple,
    ),
    ("strategy.titled_primary", "weekend", str),
    ("strategy.principal_benchmark", "all_shock", str),
    ("strategy.entry_offset_bars", 1, int),
    ("strategy.exit_offset_bars", 12, int),
    ("strategy.hold_hours", 12, int),
    ("strategy.delayed_entry_offset_bars", 2, int),
    ("strategy.delayed_exit_offset_bars", 13, int),
    ("strategy.max_concurrent_positions", 1, int),
    ("strategy.allocation_fraction", 1.0, float),
    ("strategy.allow_leverage", False, bool),
    ("strategy.allow_short", False, bool),
    ("strategy.use_stops", False, bool),
    ("strategy.sparse_trade_label_threshold", 20, int),
    (
        "strategy.benchmarks",
        ("cash", "btc_buy_and_hold", "all_shock", "weekday_shock"),
        tuple,
    ),
    # Costs
    ("costs.scenarios.low.fee_per_side", 0.0005, float),
    ("costs.scenarios.low.slippage_per_side", 0.0001, float),
    ("costs.scenarios.primary.fee_per_side", 0.0010, float),
    ("costs.scenarios.primary.slippage_per_side", 0.0002, float),
    ("costs.scenarios.high.fee_per_side", 0.0010, float),
    ("costs.scenarios.high.slippage_per_side", 0.0010, float),
    ("costs.active_scenario", "primary", str),
    # Performance
    ("performance.sharpe_periods_per_year", 365, int),
    ("performance.risk_free_rate", 0.0, float),
    ("performance.annualisation_basis", "exact_calendar_days", str),
    ("performance.daily_mark", "last_utc_hourly_mark", str),
    # Robustness
    ("robustness.thresholds", (-2.0, -2.5, -3.0), tuple),
    ("robustness.horizons", (6, 12, 24), tuple),
    ("robustness.activity_measures", ("quote_volume", "trade_count"), tuple),
    ("robustness.execution", ("next_open", "delayed_one_hour"), tuple),
    ("robustness.cost_scenarios", ("low", "primary", "high"), tuple),
    ("robustness.inference", ("hc3", "month_block_bootstrap"), tuple),
    ("robustness.positive_comparison_threshold", 2.5, float),
    ("robustness.diagnostic_cooldown_hours", (12, 48), tuple),
    (
        "robustness.diagnostics",
        (
            "calendar_year_stability",
            "leave_one_event_out",
            "leave_one_year_out",
            "weekend_contained_vs_monday_crossing",
        ),
        tuple,
    ),
    ("robustness.one_family_at_a_time", True, bool),
    # Reproducibility
    ("reproducibility.seed", 20260901, int),
    ("reproducibility.bit_generator", "PCG64", str),
    ("reproducibility.record_numpy_version", True, bool),
    ("reproducibility.atol", 1e-12, float),
    ("reproducibility.rtol", 1e-12, float),
    ("reproducibility.deterministic_sorting", True, bool),
    # Paths
    ("paths.data_raw", "data/raw", str),
    ("paths.data_interim", "data/interim", str),
    ("paths.data_processed", "data/processed", str),
    ("paths.outputs_figures", "outputs/figures", str),
    ("paths.outputs_tables", "outputs/tables", str),
    ("paths.outputs_runs", "outputs/runs", str),
    # Meta
    ("meta.spec_version", "1.5", str),
    ("meta.spec_frozen_date", date(2026, 9, 4), date),
    ("meta.spec_path", "docs/01_PROJECT_SPEC.md", str),
    ("meta.config_version", "1.1.0", str),
)


def _value(config, dotted: str):
    node = config
    for part in dotted.split("."):
        node = node[part]
    return node


@pytest.mark.parametrize(("dotted", "expected", "expected_type"), LOCKED_VALUES)
def test_s1_03_locked_value_and_type(config, dotted: str, expected, expected_type) -> None:
    observed = _value(config, dotted)
    assert type(observed) is expected_type, f"{dotted}: unexpected type"
    assert observed == expected, f"{dotted}: unexpected value"


def test_s1_03_time_blocks_match_specification(config) -> None:
    """Four fixed six-hour UTC blocks."""
    observed = [
        (block.name, block.start_hour, block.end_hour) for block in config.calendar.time_blocks
    ]
    assert observed == [
        ("00-05", 0, 5),
        ("06-11", 6, 11),
        ("12-17", 12, 17),
        ("18-23", 18, 23),
    ]


def test_s1_03_cost_scenario_helper_returns_active_rates(config) -> None:
    active = cost_scenario(config)
    assert (active.fee_per_side, active.slippage_per_side) == (0.0010, 0.0002)
    high = cost_scenario(config, "high")
    assert (high.fee_per_side, high.slippage_per_side) == (0.0010, 0.0010)


def test_s1_04_config_declares_the_active_specification_version(config) -> None:
    assert config.meta.spec_version == EXPECTED_SPEC_VERSION == "1.5"


def test_s1_04_specification_document_header_agrees(config) -> None:
    """The configuration version must match the project spec header."""
    assert spec_document_version() == config.meta.spec_version


@pytest.mark.parametrize(
    ("key", "replacement"),
    (
        ("nonstandard_close_action", "reject"),
        ("largest_return_audit_count", 19),
        ("largest_open_gap_audit_count", 21),
        ("review_zero_base_volume", False),
        ("review_zero_quote_volume", False),
        ("review_zero_trade_count", False),
    ),
)
def test_s1_04_source_quality_amendment_values_cannot_drift(
    raw_config, key: str, replacement
) -> None:
    raw_config["data_quality"][key] = replacement
    with pytest.raises(ConfigError, match="conflicts with specification|not one of"):
        validate_config(raw_config)


def test_s1_05_low_activity_threshold_is_exactly_one_third(config) -> None:
    assert low_activity_threshold(config) == Fraction(1, 3)


def test_s1_05_low_activity_boundary_is_strict(config) -> None:
    """P = 1/3 is not low activity; the next value below it is.

    With the 2,160-value reference window the cut-off falls on 720 counted
    values, so this is the real tie case.
    """
    reference = config.activity.reference_window
    assert reference == 2160
    assert is_low_activity(719, reference, config) is True
    assert is_low_activity(720, reference, config) is False
    assert is_low_activity(721, reference, config) is False
    assert is_low_activity(0, reference, config) is True
    assert is_low_activity(reference, reference, config) is False


@pytest.mark.parametrize("reference_count", (2159, 2161, 1, 720, 4320))
def test_s1_05_low_activity_rejects_any_other_denominator(config, reference_count) -> None:
    """The denominator is always 2,160; any other value is a caller error."""
    with pytest.raises(ConfigError, match="must equal the configured reference window 2160"):
        is_low_activity(100, reference_count, config)


def test_s1_05_low_activity_rejects_out_of_range_and_non_integer_counts(config) -> None:
    reference = config.activity.reference_window
    with pytest.raises(ConfigError, match=r"must lie within \[0, reference_count\]"):
        is_low_activity(reference + 1, reference, config)
    with pytest.raises(ConfigError, match=r"must lie within \[0, reference_count\]"):
        is_low_activity(-1, reference, config)
    with pytest.raises(ConfigError, match="count_at_or_below must be an integer"):
        is_low_activity(720.0, reference, config)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="reference_count must be an integer"):
        is_low_activity(720, float(reference), config)  # type: ignore[arg-type]
