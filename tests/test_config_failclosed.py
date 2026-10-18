"""Invalid, missing, and conflicting configuration values stop with an error.

Each case comes from a mutation fixture in ``tests/fixtures``. The valid
configuration must load, and every mutation must raise ``ConfigError`` with a
message that names the problem.
"""

from __future__ import annotations

import pytest

from src.config import ConfigError, validate_config
from tests.conftest import MUTATION_CASES


def test_valid_configuration_passes_validation(raw_config) -> None:
    """Without this, a mutation test could pass for the wrong reason."""
    assert validate_config(raw_config) is not None


@pytest.mark.parametrize("case", MUTATION_CASES, ids=[case.name for case in MUTATION_CASES])
def test_s1_06_mutation_fails_closed(raw_config, case) -> None:
    mutated = case.apply(raw_config)
    with pytest.raises(ConfigError) as caught:
        validate_config(mutated)
    message = str(caught.value)
    assert case.expect_message_contains in message, (
        f"{case.name}: expected {case.expect_message_contains!r} in {message!r}"
    )


def test_s1_06_fixture_coverage_spans_every_failure_class() -> None:
    """The fixtures must keep covering every kind of failure."""
    names = {case.name for case in MUTATION_CASES}
    required = {
        "missing_required_key",
        "missing_top_level_section",
        "unknown_leaf_key",
        "unknown_top_level_key",
        "wrong_type_string_for_int",
        "wrong_type_bool_for_int",
        "wrong_type_int_for_float",
        "spec_version_mismatch",
        "timestamp_naive",
        "timestamp_offset_form",
        "timestamp_not_hour_aligned",
        "splits_overlap",
        "splits_gap",
        "short_warmup",
        "seed_changed",
        "tolerance_mismatch",
        "bootstrap_target_above_attempts",
        "cost_scenarios_non_monotone",
        "absolute_path",
    }
    assert required <= names, f"missing fixtures: {sorted(required - names)}"


def test_s1_07_top_level_must_be_a_mapping() -> None:
    with pytest.raises(ConfigError, match="expected a mapping"):
        validate_config([])  # type: ignore[arg-type]


def test_s1_09_empty_configuration_reports_every_missing_section() -> None:
    with pytest.raises(ConfigError, match="missing required key"):
        validate_config({})
