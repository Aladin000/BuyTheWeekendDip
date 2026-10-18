"""A valid-looking edit to accepted evidence must not reach the report."""

import csv
import pytest

from src import paths, pipeline, report_tables
from src.heldout import HeldOutError
from src.inference import InferenceError


@pytest.mark.parametrize(
    "relative, field, expected_error",
    [
        (
            report_tables.HELD_OUT_RUN / "held_out_strategy_metrics.csv",
            "net_cumulative_return",
            HeldOutError,
        ),
        (
            report_tables.ACCEPTED_INFERENCE_MANIFEST.parent
            / "development_bootstrap_results.csv",
            "percentile_95_high",
            InferenceError,
        ),
    ],
)
def test_report_rejects_changed_result_before_overwriting_generated_file(
    tmp_path, monkeypatch, relative, field, expected_error
):
    original_resolve = paths.resolve
    original = original_resolve(relative)
    with original.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0][field] = str(float(rows[0][field]) + 0.25)
    tampered = tmp_path / relative.name
    with tampered.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    generated = tmp_path / "results.tex"
    generated.write_text("previous verified report", encoding="utf-8")
    target = original_resolve(report_tables.OUTPUT_RELATIVE_PATH)

    def redirect(*parts):
        resolved = original_resolve(*parts)
        if resolved == original:
            return tampered
        if resolved == target:
            return generated
        return resolved

    monkeypatch.setattr(paths, "resolve", redirect)
    with pytest.raises(expected_error, match="checksum"):
        report_tables.write()
    assert generated.read_text(encoding="utf-8") == "previous verified report"


def test_pipeline_report_regenerates_verified_tables_without_new_run(
    tmp_path, monkeypatch, capsys
):
    original_resolve = paths.resolve
    target = original_resolve(report_tables.OUTPUT_RELATIVE_PATH)
    generated = tmp_path / "results.tex"
    accepted = target.read_text(encoding="utf-8")
    runs = set(original_resolve("outputs/runs").iterdir())

    def redirect(*parts):
        resolved = original_resolve(*parts)
        return generated if resolved == target else resolved

    monkeypatch.setattr(paths, "resolve", redirect)
    assert pipeline.main(["report"]) == 0
    assert capsys.readouterr().out.strip() == report_tables.OUTPUT_RELATIVE_PATH.as_posix()
    assert generated.read_text(encoding="utf-8") == accepted
    assert set(original_resolve("outputs/runs").iterdir()) == runs
