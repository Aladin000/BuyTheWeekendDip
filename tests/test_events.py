"""Shock candidates and the chronological cooldown."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.events import EventError, construct_sign_specific_events

UTC = timezone.utc
START = datetime(2020, 1, 1, tzinfo=UTC)


def decisions(zscores, hours):
    times = [START + timedelta(hours=hour) for hour in hours]
    raw = [int(moment.timestamp() * 1_000) for moment in times]
    return construct_sign_specific_events(
        zscores,
        times,
        raw,
        negative_threshold=-2.5,
        positive_threshold=2.5,
        cooldown_hours=24,
    )


def test_g2_strict_candidate_thresholds_exclude_equality() -> None:
    result = decisions([-2.5000001, -2.5, 2.5, 2.5000001], [0, 1, 2, 3])
    assert [item.negative_candidate for item in result] == [True, False, False, False]
    assert [item.positive_candidate for item in result] == [False, False, False, True]


def test_g2_cooldown_accepts_elapsed_hours_zero_twenty_four_and_forty_nine() -> None:
    result = decisions([-3.0] * 4, [0, 12, 24, 49])
    assert [item.negative_event for item in result] == [True, False, True, True]
    assert result[1].event_rejection_reason == "same_sign_cooldown"
    assert result[0].event_id.startswith("negative-")


def test_g2_positive_and_negative_cooldowns_are_sign_specific() -> None:
    result = decisions([-3.0, 3.0, -3.0, 3.0], [0, 1, 2, 25])
    assert [item.negative_event for item in result] == [True, False, False, False]
    assert [item.positive_event for item in result] == [False, True, False, True]
    assert result[2].event_rejection_reason == "same_sign_cooldown"


def test_g2_future_candidate_changes_cannot_change_earlier_event_identity() -> None:
    first = decisions([-3.0, -3.0, None, None], [0, 12, 24, 36])
    changed_future = decisions([-3.0, -3.0, None, 4.0], [0, 12, 24, 36])
    assert first[:3] == changed_future[:3]


def test_g2_cooldown_state_does_not_reset_at_analysis_boundaries() -> None:
    warmup_end = datetime(2020, 1, 1, tzinfo=UTC)
    development_end = datetime(2024, 1, 1, tzinfo=UTC)
    times = [
        warmup_end - timedelta(hours=1),
        warmup_end,
        development_end - timedelta(hours=1),
        development_end,
    ]
    raw = [int(moment.timestamp() * 1_000) for moment in times]
    result = construct_sign_specific_events(
        [-3.0, -3.0, -3.0, -3.0],
        times,
        raw,
        negative_threshold=-2.5,
        positive_threshold=2.5,
        cooldown_hours=24,
    )
    assert [item.negative_event for item in result] == [True, False, True, False]


def test_g2_event_constructor_fails_on_disorder_or_nonfinite_score() -> None:
    with pytest.raises(EventError, match="strictly chronological"):
        decisions([-3.0, -3.0], [1, 0])
    with pytest.raises(EventError, match="non-finite"):
        decisions([-3.0, float("nan")], [0, 1])
