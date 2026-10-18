"""Accepted run references and rerun protection use saved evidence directly."""

from pathlib import Path

import pytest

from src import freeze, heldout, paths, pipeline, replay, robustness
from src.artifacts import read_manifest, sha256_file


def test_accepted_freeze_is_selected_without_a_ledger() -> None:
    assert not paths.resolve("docs/RESULTS_LEDGER.md").exists()
    assert heldout.accepted_freeze_manifest() == heldout.ACCEPTED_FREEZE_MANIFEST


def test_primary_reference_matches_the_notebook_replay() -> None:
    assert robustness.accepted_primary_held_out_manifest() == replay.ACCEPTED_HELD_OUT_MANIFEST


def test_primary_manifest_binds_the_accepted_freeze() -> None:
    primary = read_manifest(robustness.accepted_primary_held_out_manifest())
    selected = heldout.accepted_freeze_manifest()
    references = {name: digest for name, digest in primary["input_checksums"].items()
                  if name.endswith("/held-out-freeze.manifest.json")}
    assert references == {selected.as_posix(): sha256_file(paths.resolve(selected))}


def test_completed_submission_keeps_both_authorisations_closed() -> None:
    assert pipeline.held_out_authorised() is False
    assert pipeline.held_out_robustness_authorised() is False


def test_completed_runs_still_refuse_a_second_held_out_look(config) -> None:
    with pytest.raises(heldout.HeldOutError, match="already exists"):
        heldout.ensure_no_completed_held_out_run(config)
    with pytest.raises(robustness.RobustnessError, match="rerun refused"):
        robustness.ensure_no_completed_held_out_robustness_run(config)


def test_every_accepted_manifest_exists_and_is_self_consistent() -> None:
    manifests = [*freeze.ACCEPTED_MANIFESTS.values(), heldout.accepted_freeze_manifest(),
                 replay.ACCEPTED_HELD_OUT_MANIFEST, replay.ACCEPTED_HELD_OUT_ROBUSTNESS_MANIFEST]
    for relative in manifests:
        payload = read_manifest(relative)
        assert payload["run_id"] == relative.parent.name
        assert paths.resolve(Path(str(relative) + ".sha256")).is_file()


def test_historical_freeze_verifies_without_retired_documents(monkeypatch) -> None:
    resolve = paths.resolve

    def no_retired_notes(*parts):
        if Path(*parts).as_posix() in {"docs/DECISION_LOG.md", "docs/RESULTS_LEDGER.md"}:
            raise AssertionError("retired documentation was accessed")
        return resolve(*parts)

    monkeypatch.setattr(paths, "resolve", no_retired_notes)
    record = freeze.verify_freeze(heldout.ACCEPTED_FREEZE_MANIFEST, require_current_identity=False)
    assert record["status"] == "COMPLETE"


def test_historical_freeze_rejects_rewritten_metadata(monkeypatch, tmp_path) -> None:
    import hashlib
    import json

    relative = heldout.ACCEPTED_FREEZE_MANIFEST
    manifest = read_manifest(relative)
    record_path = relative.parent / freeze.FREEZE_RECORD_NAME
    record = json.loads(paths.resolve(record_path).read_text())
    record["decision_log_sha256"] = "0" * 64
    record_text = json.dumps(record, indent=2) + "\n"
    manifest["input_checksums"]["docs/DECISION_LOG.md"] = "0" * 64
    manifest["output_checksum"] = hashlib.sha256(record_text.encode()).hexdigest()
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(manifest_text)
    (tmp_path / record_path).write_text(record_text)
    Path(str(target) + ".sha256").write_text(
        f"{hashlib.sha256(manifest_text.encode()).hexdigest()}  {relative.name}\n"
    )
    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    with pytest.raises(freeze.FreezeError, match="identity changed"):
        freeze.verify_freeze(relative, require_current_identity=False)
