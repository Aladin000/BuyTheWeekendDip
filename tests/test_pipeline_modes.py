"""Execution-mode guards.

Every planned mode is registered, unbuilt stages stop with an error, and the
saved held-out modes stay closed in the completed submission. Acquisition itself is tested in ``tests/test_acquisition.py``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src import outcomes, pipeline
from src.artifacts import EXECUTION_MODES
from src.outcomes import OutcomeError
from src.pipeline import (
    AuthorisationError,
    ModeNotImplementedError,
    build_parser,
    dispatch,
    main,
    require_held_out_authorisation,
)

HANDOFF_MODES = (
    "acquire",
    "validate",
    "features",
    "count-only",
    "outcomes",
    "development",
    "backtest",
    "robustness",
    "freeze",
    "held-out",
    "held-out-robustness",
    "report",
)


def test_every_handoff_mode_is_registered() -> None:
    assert set(HANDOFF_MODES) <= set(EXECUTION_MODES)
    assert "config-check" in EXECUTION_MODES


def test_parser_rejects_an_unregistered_mode() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["optimise"])


def test_config_check_mode_succeeds(capsys) -> None:
    assert dispatch("config-check") == 0
    printed = capsys.readouterr().out
    assert "specification version   1.5" in printed
    assert "seed / bit generator    20260901 / PCG64" in printed


def test_unknown_dispatch_fails_closed() -> None:
    with pytest.raises(ModeNotImplementedError, match="unknown mode"):
        dispatch("optimise")


def test_outcomes_mode_dispatches_to_stage6(monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr(
        outcomes, "run_development_outcomes", lambda config: calls.append(config) or "result"
    )
    monkeypatch.setattr(outcomes, "summarise", lambda result: [f"summary {result}"])
    assert dispatch("outcomes") == 0
    assert len(calls) == 1
    assert "summary result" in capsys.readouterr().out


def test_outcomes_mode_refuses_accepted_inputs_from_a_dirty_tree(monkeypatch, config) -> None:
    real_build = outcomes.build_manifest

    def dirty_manifest(*args, **kwargs):
        draft = real_build(*args, **kwargs)
        return replace(draft, code_identity={**draft.code_identity, "git_dirty": True})

    monkeypatch.setattr(outcomes, "build_manifest", dirty_manifest)
    with pytest.raises(OutcomeError, match="clean committed repository"):
        outcomes.run_development_outcomes(config)
    assert main(["outcomes"]) == 2


def test_held_out_mode_refuses_without_authorisation(monkeypatch) -> None:
    """The guard refuses saved execution when authorisation is closed."""
    monkeypatch.setattr(pipeline, "held_out_authorised", lambda: False)
    with pytest.raises(AuthorisationError, match="Held-out evaluation refused"):
        require_held_out_authorisation()
    with pytest.raises(AuthorisationError):
        dispatch("held-out")


def test_held_out_robustness_mode_refuses_without_authorisation(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "held_out_robustness_authorised", lambda: False)
    with pytest.raises(AuthorisationError, match="robustness completion refused"):
        pipeline.require_held_out_robustness_authorisation()
    with pytest.raises(AuthorisationError):
        dispatch("held-out-robustness")


def test_main_returns_a_failure_code_for_a_blocked_mode(monkeypatch, capsys) -> None:
    monkeypatch.setattr(pipeline, "held_out_authorised", lambda: False)
    assert main(["held-out"]) == 2
    assert "Held-out evaluation refused" in capsys.readouterr().err


def test_main_returns_success_for_config_check() -> None:
    assert main(["config-check"]) == 0
