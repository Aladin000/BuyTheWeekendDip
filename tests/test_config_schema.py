"""Configuration key completeness.

The expected keys are written out here rather than taken from
``src.config.SCHEMA``, so a key added to or removed from the schema is caught.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

EXPECTED_KEYS: dict[str, tuple[str, ...]] = {
    "meta": ("project", "spec_version", "spec_frozen_date", "spec_path", "config_version"),
    "market": ("exchange", "market_type", "symbol", "interval", "timezone"),
    "source": (
        "base_url",
        "path_pattern",
        "checksum_suffix",
        "first_month",
        "last_month",
        "local_digest_algorithm",
        "allow_rest_fallback",
        "expected_column_count",
        "expected_column_names",
        "schema_verified",
        "max_header_rows",
    ),
    "timestamps": (
        "unit_change_at",
        "unit_before_change",
        "unit_from_change",
        "close_time_rule",
        "assign_calendar_from",
    ),
    "data_quality": (
        "nonstandard_close_action",
        "largest_return_audit_count",
        "largest_open_gap_audit_count",
        "review_zero_base_volume",
        "review_zero_quote_volume",
        "review_zero_trade_count",
    ),
    "coverage": ("raw_start_inclusive", "raw_end_exclusive"),
    "splits": ("warmup", "development", "held_out", "membership_timestamp", "containment"),
    "splits.warmup": ("start_inclusive", "end_exclusive"),
    "splits.development": ("start_inclusive", "end_exclusive"),
    "splits.held_out": ("start_inclusive", "end_exclusive"),
    "calendar": (
        "weekend_iso_weekdays",
        "monday_iso_weekday",
        "time_blocks",
        "time_block_reference",
        "time_block_source",
    ),
    "returns": ("type", "require_contiguous_previous_hour"),
    "shock": (
        "window_hours",
        "exclude_current_return",
        "ddof",
        "negative_threshold",
        "positive_threshold",
        "strict_inequality",
    ),
    "events": (
        "cooldown_hours",
        "cooldown_boundary_eligible",
        "retain",
        "sign_specific",
        "elapsed_time_basis",
        "continuous_through_warmup_and_splits",
        "acceptance_ignores_future_information",
    ),
    "activity": (
        "primary_measure",
        "secondary_measure",
        "window_hours",
        "exclude_shock_hour",
        "reference_window",
        "reference_same_segment_consecutive",
        "tie_rule",
        "required_history_hours",
        "low_activity_threshold",
    ),
    "activity.low_activity_threshold": ("numerator", "denominator"),
    "controls": ("rv_window_hours", "rv_excludes_shock_return", "shock_magnitude_definition"),
    "outcomes": (
        "primary_horizon_hours",
        "secondary_horizon_hours",
        "require_contiguous_path",
        "cumulative_path_hours",
    ),
    "inference": (
        "alpha",
        "covariance",
        "h1_h4_side",
        "confidence_level",
        "confidence_side",
        "t_reference",
        "holm",
        "h5",
        "model3_min_events_per_cell",
        "model3_min_n_per_coefficient",
        "cooks_distance_rule",
    ),
    "inference.holm": ("method", "family_size", "members", "unestimable_placeholder_p"),
    "inference.h5": ("side", "in_holm_family"),
    "bootstrap": (
        "method",
        "block_unit",
        "target_valid_replicates",
        "max_attempts",
        "with_replacement",
    ),
    "strategy": (
        "initial_capital",
        "variants",
        "titled_primary",
        "principal_benchmark",
        "entry_offset_bars",
        "exit_offset_bars",
        "hold_hours",
        "delayed_entry_offset_bars",
        "delayed_exit_offset_bars",
        "max_concurrent_positions",
        "allocation_fraction",
        "allow_leverage",
        "allow_short",
        "use_stops",
        "sparse_trade_label_threshold",
        "benchmarks",
    ),
    "costs": ("scenarios", "active_scenario"),
    "costs.scenarios": ("low", "primary", "high"),
    "costs.scenarios.low": ("fee_per_side", "slippage_per_side"),
    "costs.scenarios.primary": ("fee_per_side", "slippage_per_side"),
    "costs.scenarios.high": ("fee_per_side", "slippage_per_side"),
    "performance": (
        "sharpe_periods_per_year",
        "risk_free_rate",
        "annualisation_basis",
        "daily_mark",
    ),
    "robustness": (
        "thresholds",
        "horizons",
        "activity_measures",
        "execution",
        "cost_scenarios",
        "inference",
        "positive_comparison_threshold",
        "diagnostic_cooldown_hours",
        "diagnostics",
        "one_family_at_a_time",
    ),
    "reproducibility": (
        "seed",
        "bit_generator",
        "record_numpy_version",
        "atol",
        "rtol",
        "deterministic_sorting",
    ),
    "paths": (
        "data_raw",
        "data_interim",
        "data_processed",
        "outputs_figures",
        "outputs_tables",
        "outputs_runs",
    ),
}

TOP_LEVEL_SECTIONS = (
    "meta",
    "market",
    "source",
    "timestamps",
    "data_quality",
    "coverage",
    "splits",
    "calendar",
    "returns",
    "shock",
    "events",
    "activity",
    "controls",
    "outcomes",
    "inference",
    "bootstrap",
    "strategy",
    "costs",
    "performance",
    "robustness",
    "reproducibility",
    "paths",
)


def _node(config: Mapping[str, Any], dotted: str) -> Mapping[str, Any]:
    node: Any = config
    for part in dotted.split("."):
        node = node[part]
    return node


def test_s1_01_top_level_sections_present(config) -> None:
    assert sorted(config.keys()) == sorted(TOP_LEVEL_SECTIONS)


@pytest.mark.parametrize("dotted", sorted(EXPECTED_KEYS))
def test_s1_01_required_keys_present(config, dotted: str) -> None:
    observed = sorted(_node(config, dotted).keys())
    assert observed == sorted(EXPECTED_KEYS[dotted]), f"key mismatch under {dotted}"


def test_s1_02_no_unknown_keys(config) -> None:
    """No mapping contains keys beyond the expected ones."""
    for dotted, expected in EXPECTED_KEYS.items():
        observed = set(_node(config, dotted).keys())
        assert not observed - set(expected), f"unknown key(s) under {dotted}"
