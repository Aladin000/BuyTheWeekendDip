"""Offline checks for the robustness stage."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import paths, pipeline, robustness
from src.artifacts import build_manifest, read_manifest, sha256_file
from src.inference import EventRecord
from src.robustness import (
    DevelopmentRobustnessResult,
    HeldOutRobustnessResult,
    Hour,
    RobustnessBundle,
    RobustnessError,
    _outcome,
    rebuild_events,
    run_development_robustness,
    variant_config,
    verify_inventory,
    with_horizon,
    with_trade_activity,
    validate_result_coverage,
    validate_primary_replay,
)

UTC = timezone.utc
START = datetime(2020, 1, 1, tzinfo=UTC)


def hour(
    index: int, *, z: float | None = None, segment: int = 0, period: str = "development"
) -> Hour:
    open_time = START + timedelta(hours=index)
    return Hour(
        open_time,
        open_time + timedelta(hours=1, milliseconds=-1),
        int((open_time.timestamp() + 3600) * 1000 - 1),
        segment,
        period,
        100.0 + index if period == "development" else None,
        100.5 + index if period == "development" else None,
        z is not None,
        z,
        False if period == "development" else None,
        "00-05" if period == "development" else "",
        2020 if period == "development" else None,
        "2020-01" if period == "development" else "",
        0.1 if period == "development" else None,
        0.2 if period == "development" else None,
        0.8 if period == "development" else None,
        True if period == "development" else None,
        0.7 if period == "development" else None,
        0.3 if period == "development" else None,
        True if period == "development" else None,
        period == "development",
    )


def event(index: int = 0) -> EventRecord:
    row = hour(index, z=-3)
    return EventRecord(
        f"negative-{row.raw}",
        "negative",
        row.open_time,
        row.close_time,
        float(row.close),
        row.segment,
        2020,
        "2020-01",
        False,
        None,
        "00-05",
        -0.05,
        -3,
        3,
        3,
        0.2,
        0.8,
        True,
        0.1,
        True,
        True,
        True,
        0.01,
        True,
        True,
        True,
    )


def heldout_hour(index: int, *, z: float | None = None) -> Hour:
    row = hour(index, z=z, period="held_out")
    return replace(
        row,
        raw=int((row.open_time.timestamp() + 3600) * 1000 - 1),
        open=100.0 + index,
        close=100.5 + index,
        eligible=z is not None,
        weekend=False,
        block="00-05",
        year=2024,
        month="2024-01",
        rv24=0.1,
        quote_p=0.2,
        quote_score=0.8,
        quote_low=True,
        trade_p=0.7,
        trade_score=0.3,
        trade_low=True,
        full=True,
    )


def test_outcomes_use_exact_horizon_and_reject_a_gap() -> None:
    rows = tuple(hour(i) for i in range(30))
    assert _outcome(rows, 0, 6) == pytest.approx(__import__("math").log(106.5 / 100.5))
    broken = list(rows)
    broken[4] = replace(broken[4], segment=1)
    assert _outcome(broken, 0, 6) is None


def test_rebuild_changes_threshold_and_preserves_cooldown() -> None:
    rows = [hour(i, z=-2.7 if i == 0 else -3.2 if i == 12 else None) for i in range(50)]
    assert len([e for e in rebuild_events(rows, -2.5, 24) if e.sign == "negative"]) == 1
    assert len([e for e in rebuild_events(rows, -3.0, 12) if e.sign == "negative"]) == 1


def test_warmup_candidate_suppresses_development_candidate() -> None:
    rows = [
        hour(-1, z=-3, period="warmup"),
        *[hour(i, z=-3 if i == 0 else None) for i in range(30)],
    ]
    assert not [e for e in rebuild_events(rows, -2.5, 24) if e.sign == "negative"]


def test_heldout_rebuild_carries_development_cooldown_across_split() -> None:
    rows = [hour(0, z=-3)] + [
        heldout_hour(i, z=-3 if i in {12, 24} else None) for i in range(1, 55)
    ]
    events = rebuild_events(rows, -2.5, 24, analysis_period="held_out")
    negative = [item for item in events if item.sign == "negative"]
    assert [item.signal_open_time for item in negative] == [START + timedelta(hours=24)]
    assert negative[0].outcome_12h == pytest.approx(
        __import__("math").log(136.5 / 124.5)
    )


def test_horizon_and_trade_activity_change_only_declared_fields() -> None:
    rows = tuple(hour(i) for i in range(30))
    primary = event()
    changed = with_horizon((primary,), rows, 6)[0]
    assert (
        changed.event_id == primary.event_id
        and changed.outcome_12h == pytest.approx(__import__("math").log(106.5 / 100.5))
    )
    trade = with_trade_activity((primary,), rows)[0]
    assert (
        trade.activity_percentile,
        trade.low_activity_score,
        trade.low_activity,
    ) == (0.7, 0.3, True)
    assert (
        trade.event_id == primary.event_id and trade.outcome_12h == primary.outcome_12h
    )


def test_variant_config_changes_only_execution_fields(config) -> None:
    changed = variant_config(config, 2, 13, 12)
    assert (
        changed.strategy.entry_offset_bars,
        changed.strategy.exit_offset_bars,
        changed.strategy.hold_hours,
    ) == (2, 13, 12)
    assert changed.shock.negative_threshold == config.shock.negative_threshold


def test_strategy_helpers_forward_the_selected_split(config, monkeypatch) -> None:
    selected: list[str] = []

    def plans(*args, analysis_period="development", **kwargs):
        selected.append(analysis_period)
        return (), {name: () for name in config.strategy.variants}

    monkeypatch.setattr(robustness, "build_trade_plans", plans)
    robustness.strategy_rows(
        "threshold", "-2.5", (), (), config, analysis_period="held_out"
    )
    robustness._weekend_return((), (), config, analysis_period="held_out")
    assert selected == ["held_out", "held_out"]


def test_bundle_preserves_held_out_figure_label_after_boundary_rows(
    config, monkeypatch
) -> None:
    primary = (event(),)
    captured: dict[str, str] = {}
    monkeypatch.setattr(robustness, "backtest_hours", lambda *a, **k: ())
    monkeypatch.setattr(robustness, "rebuild_events", lambda *a, **k: primary)
    monkeypatch.setattr(robustness, "inference_rows", lambda *a, **k: [])
    monkeypatch.setattr(robustness, "strategy_rows", lambda *a, **k: [])
    monkeypatch.setattr(robustness, "with_horizon", lambda *a, **k: primary)
    monkeypatch.setattr(robustness, "with_trade_activity", lambda *a, **k: primary)
    monkeypatch.setattr(
        robustness, "_core_values", lambda *a, **k: {"model0:intercept": 0.0}
    )
    monkeypatch.setattr(robustness, "_weekend_return", lambda *a, **k: (0.0, 0))

    def figure(rows, *, analysis_period, label):
        captured.update(analysis_period=analysis_period, label=label)
        return robustness.HELD_OUT_FIGURE_NAME, b"png"

    monkeypatch.setattr(robustness, "_stability_figure", figure)
    bundle = robustness.build_robustness_bundle(
        primary,
        (),
        config,
        analysis_period="held_out",
        label="Held-out",
        evidence_rows=(),
    )
    assert bundle.figure == (robustness.HELD_OUT_FIGURE_NAME, b"png")
    assert captured == {"analysis_period": "held_out", "label": "Held-out"}


def test_coverage_validator_fails_when_a_frozen_family_is_missing(config) -> None:
    with pytest.raises(RobustnessError, match="coverage is incomplete"):
        validate_result_coverage(
            (robustness._record("threshold", "-2.5", "inference"),), config
        )


@pytest.mark.parametrize("refresh_sidecar", (False, True))
def test_primary_heldout_manifest_identity_detects_corruption(monkeypatch, tmp_path, refresh_sidecar) -> None:
    relative = robustness.ACCEPTED_PRIMARY_HELD_OUT_MANIFEST
    content = paths.resolve(relative).read_bytes()
    sidecar = Path(str(relative) + ".sha256")
    checksum = paths.resolve(sidecar).read_bytes()
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    (tmp_path / sidecar).write_bytes(checksum)
    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    assert robustness.accepted_primary_held_out_manifest() == relative
    target.write_bytes(content + b"\n")
    if refresh_sidecar:
        (tmp_path / sidecar).write_text(f"{sha256_file(target)}  {relative.name}\n")
    with pytest.raises(RuntimeError, match="checksum|identity"):
        robustness.accepted_primary_held_out_manifest()


def test_any_completed_robustness_manifest_blocks_another_saved_run(config, temp_acquisition_root) -> None:
    robustness.ensure_no_completed_held_out_robustness_run(config)
    target = temp_acquisition_root / config.paths.outputs_runs / "old" / "held-out-robustness.manifest.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RobustnessError, match="already exists"):
        robustness.ensure_no_completed_held_out_robustness_run(config)


def test_post_access_freeze_check_allows_only_decision_log_to_advance(
    temp_project_root, monkeypatch
) -> None:
    run = temp_project_root / "outputs/runs/freeze"
    run.mkdir(parents=True)
    decision = temp_project_root / "docs/DECISION_LOG.md"
    decision.write_text("new post-access entry\n", encoding="utf-8")
    frozen = temp_project_root / "frozen-input.txt"
    frozen.write_text("unchanged\n", encoding="utf-8")
    record = run / "held_out_freeze_record.json"
    record.write_text(
        '{"status":"COMPLETE","held_out_outcomes_accessed":false,'
        '"decision_log_sha256":"'
        + "1" * 64
        + '"}\n',
        encoding="utf-8",
    )
    manifest = {
        "stage": "10. Held-out freeze",
        "mode": "freeze",
        "output_checksum": sha256_file(record),
        "input_checksums": {
            "docs/DECISION_LOG.md": "1" * 64,
            "frozen-input.txt": sha256_file(frozen),
        },
    }
    monkeypatch.setattr(robustness, "verify_manifest_checksum", lambda *a: None)
    monkeypatch.setattr(robustness, "read_manifest", lambda *a: manifest)
    robustness._verify_original_freeze_after_access(
        Path("outputs/runs/freeze/held-out-freeze.manifest.json")
    )
    frozen.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RobustnessError, match="frozen input changed"):
        robustness._verify_original_freeze_after_access(
            Path("outputs/runs/freeze/held-out-freeze.manifest.json")
        )


def test_primary_replay_must_match_accepted_reference(config) -> None:
    rows = [
        robustness._record(
            "cost", "primary", "strategy", strategy="weekend", estimate=0.1
        ),
        robustness._record(
            "inference",
            "hc3",
            "inference",
            model="model0",
            term="intercept",
            estimate=0.01,
        ),
        robustness._record(
            "threshold", "-2.5", "strategy", strategy="weekend", estimate=0.1
        ),
        robustness._record(
            "threshold",
            "-2.5",
            "inference",
            model="model0",
            term="intercept",
            estimate=0.01,
        ),
    ]
    validate_primary_replay(rows, config)
    rows[-1]["estimate"] = 0.02
    with pytest.raises(RobustnessError, match="differs from accepted evidence"):
        validate_primary_replay(rows, config)


def test_import_has_no_data_or_network_io(project_root) -> None:
    script = r"""import sys
opened=[]
def audit(event,args):
    if event=="open" and "BuyTheWeekendDip" in str(args[0]) and "outputs/runs" in str(args[0]): opened.append(str(args[0]))
sys.addaudithook(audit)
import src.robustness
assert opened == [], opened"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=project_root, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_pipeline_dispatch_is_thin(monkeypatch, capsys) -> None:
    result = DevelopmentRobustnessResult(
        "run",
        Path("outputs/runs/run/development-robustness.manifest.json"),
        Path("outputs/runs/run/development-robustness.manifest.json.sha256"),
        Path("outputs/runs/run/run.log.jsonl"),
        Path("outputs/runs/run/development_robustness_inventory.csv"),
        10,
    )
    monkeypatch.setattr(pipeline, "load_config", lambda: object())
    monkeypatch.setattr(robustness, "run_development_robustness", lambda config: result)
    assert pipeline.dispatch("robustness") == 0
    assert "held-out outcomes       not accessed" in capsys.readouterr().out


def test_held_out_robustness_dispatch_is_thin(monkeypatch, capsys) -> None:
    result = HeldOutRobustnessResult(
        "run",
        Path("outputs/runs/run/held-out-robustness.manifest.json"),
        Path("outputs/runs/run/held-out-robustness.manifest.json.sha256"),
        Path("outputs/runs/run/run.log.jsonl"),
        Path("outputs/runs/run/held_out_robustness_inventory.csv"),
        10,
    )
    monkeypatch.setattr(pipeline, "load_config", lambda: object())
    monkeypatch.setattr(
        pipeline, "require_held_out_robustness_authorisation", lambda: None
    )
    monkeypatch.setattr(robustness, "run_held_out_robustness", lambda config: result)
    assert pipeline.dispatch("held-out-robustness") == 0
    assert "development outcomes    not analysed" in capsys.readouterr().out


def test_run_writes_and_reverifies_evidence(
    config, temp_acquisition_root, monkeypatch
) -> None:
    rows = tuple(
        robustness._record(family, "x", "reference")
        for family in (
            "threshold",
            "horizon",
            "activity",
            "execution",
            "cost",
            "inference",
            "positive_shock",
            "cooldown",
            "calendar_year",
            "leave_one_event_out",
            "leave_one_year_out",
            "weekend_boundary",
        )
    )
    bundle = RobustnessBundle(
        rows, ("development_calendar_year_stability.png", b"\x89PNG\r\n\x1a\nsynthetic")
    )
    monkeypatch.setattr(
        robustness,
        "verify_inputs",
        lambda *a, **k: (Path("features.csv"), 0, Path("events.csv"), {}),
    )
    monkeypatch.setattr(robustness, "read_hours", lambda *a: ())
    monkeypatch.setattr(robustness, "read_event_records", lambda *a: ())
    monkeypatch.setattr(robustness, "build_robustness_bundle", lambda *a: bundle)
    monkeypatch.setattr(robustness, "validate_result_coverage", lambda *a: None)
    monkeypatch.setattr(robustness, "validate_primary_replay", lambda *a: None)
    result = run_development_robustness(
        config, backtest_manifest=Path("synthetic.json")
    )
    manifest = read_manifest(result.manifest_relative_path)
    assert manifest["exclusions"] == {"held_out_rows_analysed": 0}
    assert len(verify_inventory(result.manifest_relative_path, config)) == 2
    target = (
        temp_acquisition_root
        / result.manifest_relative_path.parent
        / "development_robustness_results.csv"
    )
    target.write_text("tampered\n")
    with pytest.raises(RobustnessError, match="checksum failed"):
        verify_inventory(result.manifest_relative_path, config)


def test_held_out_robustness_run_writes_two_checksum_bound_artifacts(
    config, temp_acquisition_root, monkeypatch
) -> None:
    rows = tuple(
        robustness._record(family, "x", "reference")
        for family in (
            "threshold",
            "horizon",
            "activity",
            "execution",
            "cost",
            "inference",
            "positive_shock",
            "cooldown",
            "calendar_year",
            "leave_one_event_out",
            "leave_one_year_out",
            "weekend_boundary",
        )
    )
    bundle = RobustnessBundle(
        rows,
        (
            robustness.HELD_OUT_FIGURE_NAME,
            b"\x89PNG\r\n\x1a\nsynthetic",
        ),
    )
    monkeypatch.setattr(
        pipeline, "require_held_out_robustness_authorisation", lambda: None
    )
    draft = build_manifest(robustness.HELD_OUT_STAGE, robustness.HELD_OUT_MODE, config)
    clean = replace(
        draft,
        code_identity={
            **draft.code_identity,
            "git_commit": "0" * 40,
            "git_dirty": False,
        },
    )
    monkeypatch.setattr(robustness, "build_manifest", lambda *a, **k: clean)
    monkeypatch.setattr(
        robustness,
        "verify_held_out_robustness_inputs",
        lambda *a, **k: (Path("features.csv"), 0, Path("events.csv"), {}),
    )
    monkeypatch.setattr(robustness, "read_hours", lambda *a, **k: ())
    monkeypatch.setattr(robustness, "read_event_records", lambda *a: ())
    monkeypatch.setattr(robustness, "held_out_evidence_rows", lambda *a: [])
    monkeypatch.setattr(robustness, "build_robustness_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(robustness, "validate_result_coverage", lambda *a, **k: None)
    monkeypatch.setattr(robustness, "validate_primary_replay", lambda *a: None)
    result = robustness.run_held_out_robustness(
        config, primary_manifest=Path("primary.manifest.json")
    )
    manifest = read_manifest(result.manifest_relative_path)
    assert manifest["exclusions"] == {"development_outcome_rows_analysed": 0}
    assert len(
        robustness.verify_held_out_robustness_inventory(
            result.manifest_relative_path, config
        )
    ) == 2


def test_default_run_refuses_dirty_tree_before_log(config, monkeypatch) -> None:
    draft = build_manifest(robustness.STAGE, robustness.MODE, config)
    dirty = replace(draft, code_identity={**draft.code_identity, "git_dirty": True})
    monkeypatch.setattr(robustness, "build_manifest", lambda *a, **k: dirty)
    monkeypatch.setattr(
        robustness, "create_run_log", lambda *a, **k: pytest.fail("opened log")
    )
    with pytest.raises(RobustnessError, match="clean committed"):
        run_development_robustness(config)
