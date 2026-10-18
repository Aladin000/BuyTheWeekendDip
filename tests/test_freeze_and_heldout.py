"""Offline guards for the freeze stage and the single held-out run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import freeze, heldout, paths, pipeline
from src.artifacts import sha256_file
from src.freeze import FreezeError
from src.heldout import HeldOutError


def test_freeze_input_digest_is_order_independent_and_name_bound() -> None:
    first = freeze._input_set_digest({"b": "2" * 64, "a": "1" * 64})
    second = freeze._input_set_digest({"a": "1" * 64, "b": "2" * 64})
    changed_name = freeze._input_set_digest({"c": "1" * 64, "b": "2" * 64})
    assert first == second
    assert first != changed_name


def test_source_identity_allows_only_the_evidence_commit_to_change() -> None:
    frozen = {
        name: ({"src/freeze.py": "1" * 64} if name == "file_digests" else "1" * 64)
        for name in freeze.FROZEN_SOURCE_IDENTITY_FIELDS
    }
    frozen.update({"git_commit": "a" * 40, "git_dirty": False})
    evidence_commit = {**frozen, "git_commit": "b" * 40}
    assert freeze.source_identity_matches(frozen, evidence_commit) is True
    changed_tests = {**evidence_commit, "test_digest": "2" * 64}
    assert freeze.source_identity_matches(frozen, changed_tests) is False


def test_core_check_failure_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        freeze.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="failed", stderr=""
        ),
    )
    with pytest.raises(FreezeError, match="pytest failed"):
        freeze.run_core_checks()


def test_core_check_record_uses_portable_commands(monkeypatch) -> None:
    monkeypatch.setattr(
        freeze.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="passed", stderr=""
        ),
    )
    records = freeze.run_core_checks()
    assert [record["name"] for record in records] == [
        "pytest",
        "pip_check",
        "git_diff_check",
    ]
    assert all(not str(record["command"][0]).startswith("/") for record in records)


@pytest.mark.parametrize("refresh_sidecar", (False, True))
def test_accepted_freeze_identity_detects_corruption(monkeypatch, tmp_path, refresh_sidecar) -> None:
    relative = heldout.ACCEPTED_FREEZE_MANIFEST
    content = paths.resolve(relative).read_bytes()
    sidecar = Path(str(relative) + ".sha256")
    checksum = paths.resolve(sidecar).read_bytes()
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    (tmp_path / sidecar).write_bytes(checksum)
    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    assert heldout.accepted_freeze_manifest() == relative
    target.write_bytes(content + b"\n")
    if refresh_sidecar:
        (tmp_path / sidecar).write_text(f"{sha256_file(target)}  {relative.name}\n")
    with pytest.raises(RuntimeError, match="checksum|identity"):
        heldout.accepted_freeze_manifest()


def test_submission_authorisation_is_closed_even_if_old_notes_say_otherwise(temp_project_root) -> None:
    notes = temp_project_root / "docs/RESULTS_LEDGER.md"
    notes.write_text("## 6. Primary held-out run\n\n**Status:** AUTHORISED\n")
    assert pipeline.held_out_authorised() is False
    with pytest.raises(pipeline.AuthorisationError, match="complete"):
        pipeline.require_held_out_authorisation()


def test_submission_robustness_authorisation_is_closed_even_if_old_notes_say_otherwise(temp_project_root) -> None:
    notes = temp_project_root / "docs/RESULTS_LEDGER.md"
    notes.write_text("### Held-out robustness completion\n\n**Status:** AUTHORISED\n")
    assert pipeline.held_out_robustness_authorised() is False
    with pytest.raises(pipeline.AuthorisationError, match="complete"):
        pipeline.require_held_out_robustness_authorisation()


def test_completed_held_out_run_blocks_an_accidental_second_run(
    config, temp_project_root
) -> None:
    manifest = (
        temp_project_root
        / config.paths.outputs_runs
        / "prior-run"
        / "held-out-evaluation.manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(HeldOutError, match="already exists"):
        heldout.ensure_no_completed_held_out_run(config)


def test_held_out_checks_authorisation_before_building_or_reading(monkeypatch) -> None:
    touched = False

    def forbidden(*args, **kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("held-out input was touched")

    monkeypatch.setattr(
        pipeline,
        "require_held_out_authorisation",
        lambda: (_ for _ in ()).throw(pipeline.AuthorisationError("blocked")),
    )
    monkeypatch.setattr(heldout, "read_outcome_rows", forbidden)
    with pytest.raises(pipeline.AuthorisationError, match="blocked"):
        heldout.run_held_out()
    assert touched is False


def test_held_out_verifies_freeze_before_opening_attempt_log(
    config, monkeypatch
) -> None:
    class Draft:
        run_id = "synthetic-held-out"
        code_identity = {"git_dirty": False}

    log_opened = False

    def verify_before_log(*args, **kwargs):
        assert log_opened is False
        raise HeldOutError("synthetic preflight stop")

    def open_log(*args, **kwargs):
        nonlocal log_opened
        log_opened = True
        raise AssertionError("log opened before preflight")

    monkeypatch.setattr(pipeline, "require_held_out_authorisation", lambda: None)
    monkeypatch.setattr(heldout, "ensure_no_completed_held_out_run", lambda cfg: None)
    monkeypatch.setattr(heldout, "build_manifest", lambda *args: Draft())
    monkeypatch.setattr(
        heldout,
        "accepted_freeze_manifest",
        lambda: Path("outputs/runs/freeze/held-out-freeze.manifest.json"),
    )
    monkeypatch.setattr(heldout, "verify_held_out_inputs", verify_before_log)
    monkeypatch.setattr(heldout, "create_run_log", open_log)
    with pytest.raises(HeldOutError, match="synthetic preflight stop"):
        heldout.run_held_out(config)
    assert log_opened is False


def test_freeze_pipeline_dispatch_is_thin(monkeypatch, capsys) -> None:
    result = freeze.FreezeResult(
        "run",
        Path("outputs/runs/run/held-out-freeze.manifest.json"),
        Path("outputs/runs/run/held-out-freeze.manifest.json.sha256"),
        Path("outputs/runs/run/held_out_freeze_record.json"),
        12,
    )
    monkeypatch.setattr(freeze, "run_freeze", lambda config: result)
    assert pipeline.dispatch("freeze") == 0
    assert "frozen input files      12" in capsys.readouterr().out


def test_imports_open_no_market_data(project_root) -> None:
    script = r"""
import sys
opened = []
def audit(event, args):
    if event == "open":
        path = str(args[0])
        if "BuyTheWeekendDip" in path and ("/data/" in path or "/outputs/runs/" in path):
            opened.append(path)
sys.addaudithook(audit)
import src.freeze
import src.heldout
assert opened == [], opened
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_completed_freeze_refuses_before_reading_or_writing(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("a closed freeze must not read files or create a run")

    monkeypatch.setattr(freeze, "build_manifest", forbidden)
    monkeypatch.setattr(freeze, "load_config", forbidden)
    monkeypatch.setattr(paths, "resolve", forbidden)
    with pytest.raises(FreezeError, match="complete"):
        freeze.run_freeze()
