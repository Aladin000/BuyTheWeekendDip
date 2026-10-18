"""Shock candidates and cooldown for the event stage, using past data only.

Only each completed candle's timestamp and shock z-score are used. Future bars,
outcomes, covariates and strategy state are not seen, so they cannot change
which candidates exist or which pass the cooldown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence


class EventError(RuntimeError):
    """Raised when event construction inputs are invalid."""


@dataclass(frozen=True)
class EventDecision:
    """Candidate and accepted-event flags for one chronological candle."""

    negative_candidate: bool
    positive_candidate: bool
    negative_event: bool
    positive_event: bool
    cooldown_accepted: bool
    event_id: str
    event_rejection_reason: str


@dataclass(frozen=True)
class AnySignEventDecision:
    """Decision for the secondary H5 cooldown shared by both signs."""

    candidate_sign: str
    accepted: bool
    event_id: str
    rejection_reason: str


def construct_sign_specific_events(
    zscores: Sequence[float | None],
    close_times: Sequence[datetime],
    close_time_raw: Sequence[int],
    *,
    negative_threshold: float,
    positive_threshold: float,
    cooldown_hours: int,
) -> tuple[EventDecision, ...]:
    """Apply strict thresholds and a continuous first-candidate cooldown.

    Negative and positive shocks have separate cooldown clocks. A candidate
    ``cooldown_hours`` or more after the last accepted event of the same sign
    is accepted. Event IDs are built from the sign and the raw close timestamp.
    """
    size = len(zscores)
    if len(close_times) != size or len(close_time_raw) != size:
        raise EventError("event inputs must have identical lengths")
    if not negative_threshold < positive_threshold:
        raise EventError("negative threshold must be below positive threshold")
    if type(cooldown_hours) is not int or cooldown_hours <= 0:
        raise EventError("cooldown_hours must be a positive integer")

    cooldown = timedelta(hours=cooldown_hours)
    last_accepted: dict[str, datetime | None] = {"negative": None, "positive": None}
    previous_time: datetime | None = None
    decisions: list[EventDecision] = []

    for index, (zscore, close_time, raw_time) in enumerate(
        zip(zscores, close_times, close_time_raw, strict=True)
    ):
        if close_time.tzinfo is None or close_time.tzinfo.utcoffset(close_time) is None:
            raise EventError(f"close time at index {index} must be timezone aware")
        if previous_time is not None and close_time <= previous_time:
            raise EventError("event inputs must be strictly chronological")
        previous_time = close_time
        if type(raw_time) is not int or raw_time < 0:
            raise EventError(f"raw close time at index {index} must be a non-negative integer")
        if zscore is not None and not math.isfinite(zscore):
            raise EventError(f"z-score at index {index} is non-finite")

        negative = zscore is not None and zscore < negative_threshold
        positive = zscore is not None and zscore > positive_threshold
        sign = "negative" if negative else "positive" if positive else None
        accepted = False
        rejection = ""
        event_id = ""
        if sign is not None:
            prior = last_accepted[sign]
            if prior is None or close_time - prior >= cooldown:
                accepted = True
                last_accepted[sign] = close_time
                event_id = f"{sign}-{raw_time}"
            else:
                rejection = "same_sign_cooldown"

        decisions.append(
            EventDecision(
                negative_candidate=negative,
                positive_candidate=positive,
                negative_event=negative and accepted,
                positive_event=positive and accepted,
                cooldown_accepted=accepted,
                event_id=event_id,
                event_rejection_reason=rejection,
            )
        )
    return tuple(decisions)


def construct_any_sign_events(
    zscores: Sequence[float | None],
    close_times: Sequence[datetime],
    close_time_raw: Sequence[int],
    *,
    negative_threshold: float,
    positive_threshold: float,
    cooldown_hours: int,
) -> tuple[AnySignEventDecision, ...]:
    """Apply one first-candidate cooldown shared by positive and negative shocks.

    Used only for the secondary H5 check; the main event sets come from
    :func:`construct_sign_specific_events`.
    """
    size = len(zscores)
    if len(close_times) != size or len(close_time_raw) != size:
        raise EventError("any-sign event inputs must have identical lengths")
    if not negative_threshold < positive_threshold:
        raise EventError("negative threshold must be below positive threshold")
    if type(cooldown_hours) is not int or cooldown_hours <= 0:
        raise EventError("cooldown_hours must be a positive integer")
    cooldown = timedelta(hours=cooldown_hours)
    last_accepted: datetime | None = None
    previous_time: datetime | None = None
    output: list[AnySignEventDecision] = []
    for index, (zscore, close_time, raw_time) in enumerate(
        zip(zscores, close_times, close_time_raw, strict=True)
    ):
        if close_time.tzinfo is None or close_time.tzinfo.utcoffset(close_time) is None:
            raise EventError(f"close time at index {index} must be timezone aware")
        if previous_time is not None and close_time <= previous_time:
            raise EventError("event inputs must be strictly chronological")
        previous_time = close_time
        if type(raw_time) is not int or raw_time < 0:
            raise EventError(
                f"raw close time at index {index} must be a non-negative integer"
            )
        if zscore is not None and not math.isfinite(zscore):
            raise EventError(f"z-score at index {index} is non-finite")
        sign = (
            "negative"
            if zscore is not None and zscore < negative_threshold
            else "positive"
            if zscore is not None and zscore > positive_threshold
            else ""
        )
        accepted = False
        event_id = ""
        rejection = ""
        if sign:
            if last_accepted is None or close_time - last_accepted >= cooldown:
                accepted = True
                last_accepted = close_time
                event_id = f"{sign}-{raw_time}"
            else:
                rejection = "any_sign_cooldown"
        output.append(AnySignEventDecision(sign, accepted, event_id, rejection))
    return tuple(output)
