"""In-memory replay of accepted stages.

The notebook replays every stage; these tests keep the comparison logic and
the fast stages under the ordinary test suite.
"""

from __future__ import annotations

import random
from decimal import Decimal
from pathlib import Path

import pytest

from src import paths, replay
from src.features import lagged_percentile_numerators


def test_naive_percentile_matches_sorted_window_across_gaps_and_segments() -> None:
    generator = random.Random(20260901)
    values: list[Decimal | None] = []
    segments: list[int] = []
    segment = 1
    for index in range(1_500):
        if index and generator.random() < 0.01:
            segment += 1
        segments.append(segment)
        values.append(
            None if generator.random() < 0.02 else Decimal(generator.randint(0, 40))
        )
    expected, _ = lagged_percentile_numerators(values, segments, window=37)
    assert replay.naive_percentile_numerators(values, segments, window=37) == expected
    assert any(value is not None for value in expected)


def test_compare_detects_a_single_byte_difference(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "resolve", lambda relative: tmp_path / relative)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    same = replay._compare("stage", Path("run"), {"table.csv": "a,b\n1,2\n"}, 0.0)
    different = replay._compare("stage", Path("run"), {"table.csv": "a,b\n1,3\n"}, 0.0)
    assert same.matches
    assert not different.matches


def test_compare_refuses_a_missing_accepted_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "resolve", lambda relative: tmp_path / relative)
    with pytest.raises(replay.ReplayError, match="missing"):
        replay._compare("stage", Path("run"), {"absent.csv": ""}, 0.0)


def test_empty_replay_is_not_a_match() -> None:
    assert not replay.StageReplay("stage", "run", 0.0, ()).matches


@pytest.mark.parametrize(
    "stage", [replay.replay_features, replay.replay_count_audit]
)
def test_fast_stages_reproduce_accepted_artifacts(config, stage) -> None:
    result = stage(config)
    assert result.comparisons
    assert result.matches, [item.name for item in result.comparisons if not item.matches]
