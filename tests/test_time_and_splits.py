"""UTC parsing, half-open splits, coverage, and warm-up.

No market data is read. Every assertion is about the configured calendar
boundaries and the arithmetic they imply.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.config import (
    ConfigError,
    archive_months,
    in_split,
    source_timestamp_unit,
    split_bounds,
    split_of,
    warmup_hours,
)

BOUNDARY_KEYS = (
    ("coverage", "raw_start_inclusive"),
    ("coverage", "raw_end_exclusive"),
)
SPLIT_NAMES = ("warmup", "development", "held_out")


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# Timestamp parsing


@pytest.mark.parametrize(("section", "key"), BOUNDARY_KEYS)
def test_s1_12_coverage_boundaries_are_utc_aware(config, section: str, key: str) -> None:
    moment = config[section][key]
    assert isinstance(moment, datetime)
    assert moment.tzinfo is not None
    assert moment.utcoffset() == timedelta(0)


@pytest.mark.parametrize("name", SPLIT_NAMES)
def test_s1_12_split_boundaries_are_utc_aware(config, name: str) -> None:
    for moment in split_bounds(config, name):
        assert moment.tzinfo is not None
        assert moment.utcoffset() == timedelta(0)
        assert (moment.minute, moment.second, moment.microsecond) == (0, 0, 0)


def test_s1_12_boundaries_have_the_expected_instants(config) -> None:
    assert split_bounds(config, "warmup") == (_utc(2019, 10, 1), _utc(2020, 1, 1))
    assert split_bounds(config, "development") == (_utc(2020, 1, 1), _utc(2024, 1, 1))
    assert split_bounds(config, "held_out") == (_utc(2024, 1, 1), _utc(2026, 8, 1))


# Half-open membership


@pytest.mark.parametrize("name", SPLIT_NAMES)
def test_s1_13_start_is_included_and_end_is_excluded(config, name: str) -> None:
    start, end = split_bounds(config, name)
    assert in_split(config, start, name) is True
    assert in_split(config, end, name) is False
    assert in_split(config, end - timedelta(hours=1), name) is True
    assert in_split(config, start - timedelta(hours=1), name) is False


@pytest.mark.parametrize(
    ("moment", "expected"),
    (
        (_utc(2019, 9, 30, 23), None),
        (_utc(2019, 10, 1, 0), "warmup"),
        (_utc(2019, 12, 31, 23), "warmup"),
        (_utc(2020, 1, 1, 0), "development"),
        (_utc(2023, 12, 31, 23), "development"),
        (_utc(2024, 1, 1, 0), "held_out"),
        (_utc(2026, 7, 31, 23), "held_out"),
        (_utc(2026, 8, 1, 0), None),
    ),
)
def test_s1_13_split_assignment_at_the_boundaries(config, moment, expected) -> None:
    assert split_of(config, moment) == expected


def test_s1_13_naive_timestamp_is_rejected(config) -> None:
    with pytest.raises(ConfigError, match="timezone-aware"):
        split_of(config, datetime(2022, 1, 1))


def test_s1_13_unknown_split_is_rejected(config) -> None:
    with pytest.raises(ConfigError, match="unknown split"):
        split_bounds(config, "validation")


# Development and held-out do not overlap


def test_s1_14_development_and_held_out_are_disjoint(config) -> None:
    dev_start, dev_end = split_bounds(config, "development")
    out_start, out_end = split_bounds(config, "held_out")
    assert dev_start < dev_end <= out_start < out_end
    assert max(dev_start, out_start) >= min(dev_end, out_end)


def test_s1_14_development_and_held_out_are_contiguous(config) -> None:
    _, dev_end = split_bounds(config, "development")
    out_start, _ = split_bounds(config, "held_out")
    assert dev_end == out_start == _utc(2024, 1, 1)


def test_s1_14_warmup_precedes_and_abuts_development(config) -> None:
    _, warm_end = split_bounds(config, "warmup")
    dev_start, _ = split_bounds(config, "development")
    assert warm_end == dev_start


def test_s1_14_every_split_hour_belongs_to_exactly_one_split(config) -> None:
    """Sampled boundary-adjacent hours never fall in two splits at once."""
    probes = [
        _utc(2019, 10, 1),
        _utc(2019, 12, 31, 23),
        _utc(2020, 1, 1),
        _utc(2022, 6, 15, 12),
        _utc(2023, 12, 31, 23),
        _utc(2024, 1, 1),
        _utc(2026, 7, 31, 23),
    ]
    for moment in probes:
        matches = [name for name in SPLIT_NAMES if in_split(config, moment, name)]
        assert len(matches) == 1, f"{moment} matched {matches}"


# Warm-up coverage


def test_s1_15_warmup_supplies_the_expected_number_of_hours(config) -> None:
    """2019-10-01 to 2020-01-01 is 92 days, so 2,208 whole hours."""
    assert warmup_hours(config) == 92 * 24 == 2208


@pytest.mark.parametrize(
    ("requirement_key", "expected"),
    (
        ("activity", 2184),
        ("shock", 721),
        ("rv", 24),
    ),
)
def test_s1_15_warmup_covers_every_rolling_requirement(
    config, requirement_key: str, expected: int
) -> None:
    requirements = {
        "activity": config.activity.required_history_hours,
        "shock": config.shock.window_hours + 1,
        "rv": config.controls.rv_window_hours,
    }
    assert requirements[requirement_key] == expected
    assert warmup_hours(config) >= expected


def test_s1_15_activity_history_requirement_is_window_plus_reference(config) -> None:
    """2,184 = 24 trailing hours + 2,160 reference values."""
    assert (
        config.activity.required_history_hours
        == config.activity.window_hours + config.activity.reference_window
    )


# Derived archive months


def test_s1_16_archive_month_list_matches_the_locked_coverage(config) -> None:
    months = archive_months(config)
    assert len(months) == 82
    assert months[0] == (2019, 10)
    assert months[-1] == (2026, 7)
    assert months[0] == config.source.first_month
    assert months[-1] == config.source.last_month


def test_s1_16_archive_months_are_consecutive_and_unique(config) -> None:
    months = archive_months(config)
    assert len(set(months)) == len(months)
    for earlier, later in zip(months, months[1:]):
        year, month = earlier
        expected = (year + 1, 1) if month == 12 else (year, month + 1)
        assert later == expected


def test_s1_16_archive_path_pattern_formats_a_month(config) -> None:
    year, month = config.source.first_month
    rendered = config.source.path_pattern.format(year=year, month=month)
    assert rendered == "data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2019-10.zip"


# Source timestamp unit across the 2025 boundary


@pytest.mark.parametrize(
    ("moment", "expected"),
    (
        (_utc(2019, 10, 1, 0), "milliseconds"),
        (_utc(2024, 12, 31, 23), "milliseconds"),
        (_utc(2025, 1, 1, 0), "microseconds"),
        (_utc(2026, 7, 31, 23), "microseconds"),
    ),
)
def test_s1_17_timestamp_unit_switches_at_2025(config, moment, expected) -> None:
    assert source_timestamp_unit(config, moment) == expected


def test_s1_17_unit_change_boundary_is_the_configured_instant(config) -> None:
    assert config.timestamps.unit_change_at == _utc(2025, 1, 1)


def test_s1_17_timestamp_unit_requires_awareness(config) -> None:
    with pytest.raises(ConfigError, match="timezone-aware"):
        source_timestamp_unit(config, datetime(2025, 1, 1))
