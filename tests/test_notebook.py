import json
import re
from pathlib import Path

import nbformat

from src import paths


NOTEBOOK = Path("notebook.ipynb")


def _notebook() -> dict:
    return json.loads(paths.resolve(NOTEBOOK).read_text(encoding="utf-8"))


def _source(notebook: dict, cell_type: str | None = None) -> str:
    cells = notebook["cells"]
    if cell_type is not None:
        cells = [cell for cell in cells if cell["cell_type"] == cell_type]
    return "\n".join("".join(cell["source"]) for cell in cells)


def test_notebook_is_valid_and_every_code_cell_compiles() -> None:
    notebook = nbformat.read(paths.resolve(NOTEBOOK), as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        compile(cell.source, f"notebook-cell-{index}", "exec")


def _code_cells(notebook: dict) -> list[dict]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _output_text(notebook: dict) -> str:
    parts: list[str] = []
    for cell in _code_cells(notebook):
        for output in cell.get("outputs", []):
            parts.append("".join(output.get("text", "")))
            for mime in ("text/plain", "text/markdown", "text/html"):
                parts.append("".join(output.get("data", {}).get(mime, "")))
    return "\n".join(parts)


def test_notebook_is_portable() -> None:
    notebook = _notebook()
    source = _source(notebook)
    outputs = _output_text(notebook)
    for text in (source, outputs):
        assert "/Users/" not in text
        assert "file://" not in text
    assert "Path.cwd()" in source


def test_saved_notebook_was_executed_once_in_order_without_errors() -> None:
    cells = _code_cells(_notebook())
    assert [cell["execution_count"] for cell in cells] == list(range(1, len(cells) + 1))
    assert all(cell["outputs"] for cell in cells)
    for cell in cells:
        assert all(output["output_type"] != "error" for output in cell["outputs"])


def test_saved_outputs_record_a_complete_byte_identical_replay() -> None:
    outputs = _output_text(_notebook())
    match = re.search(r"(\d+) of (\d+) rebuilt files are byte-identical", outputs)
    assert match is not None
    assert match.group(1) == match.group(2)
    assert int(match.group(2)) > 0
    assert "identical ranks" in outputs


STAGE_HEADINGS = ("## 1. Data acquisition", "## 2. Cleaning", "## 3. Strategy", "## 4. Backtest")


def _sections(notebook: dict) -> dict[str, str]:
    """Code under each level-two heading, keyed by the heading line."""
    sections: dict[str, list[str]] = {}
    current = None
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and source.startswith("## "):
            current = source.splitlines()[0]
            sections[current] = []
        elif cell["cell_type"] == "code" and current is not None:
            sections[current].append(source)
    return {heading: "\n".join(code) for heading, code in sections.items()}


def test_notebook_shows_the_required_stages_in_order_with_dates() -> None:
    notebook = _notebook()
    headings = list(_sections(notebook))
    positions = [headings.index(heading) for heading in STAGE_HEADINGS]
    assert positions == sorted(positions)
    markdown = _source(notebook, "markdown")
    for period in ("October 2019–July 2026", "2020–2023", "2024–July 2026"):
        assert period in markdown
    assert "historical held-out (pseudo-out-of-sample)" in markdown


def test_each_stage_is_rebuilt_in_its_own_section() -> None:
    sections = _sections(_notebook())
    expected = {
        "## 2. Cleaning": ("replay_validation", "replay_features", "replay_count_audit"),
        "## 3. Strategy": (
            "replay_development_outcomes",
            "replay_development_inference",
            "replay_held_out)",
        ),
        "## 4. Backtest": ("replay_development_backtest",),
        "## 5. Robustness": ("replay_development_robustness", "replay_held_out_robustness"),
    }
    for heading, calls in expected.items():
        for call in calls:
            assert f"check_replay(replay.{call}" in sections[heading], (heading, call)


def test_notebook_verifies_and_replays_without_creating_runs() -> None:
    code = _source(_notebook(), "code")
    for verifier in (
        "verify_acquisition_inputs",
        "verify_validation_input",
        "verify_feature_input",
        "verify_count_input",
        "verify_inference_inventory",
        "verify_backtest_inventory",
        "verify_held_out_inventory",
        "verify_held_out_robustness_inventory",
        "STAGE_REPLAYS",
        "benchmark_feature_steps",
        "benchmark_bootstrap",
    ):
        assert verifier in code
    for forbidden in (
        "run_acquisition(",
        "run_validation(",
        "run_features(",
        "run_count_audit(",
        "run_development_outcomes(",
        "run_development_inference(",
        "run_development_backtest(",
        "run_development_robustness(",
        "run_freeze(",
        "run_held_out(",
        "run_held_out_robustness(",
        "src.pipeline",
        "requests.",
        "urllib.request",
    ):
        assert forbidden not in code


def test_notebook_labels_p_values_and_install_target_correctly() -> None:
    notebook = _notebook()
    assert "Adjusted p" not in _source(notebook)
    assert "not in Holm family" in _source(notebook, "code")
    assert "requirements-notebook.txt" in _source(notebook, "markdown")


def test_notebook_runtime_is_directly_pinned() -> None:
    requirements = paths.resolve("requirements-notebook.txt").read_text(encoding="utf-8")
    assert "-r requirements-lock.txt" in requirements
    for pin in ("ipykernel==6.29.5", "nbclient==0.10.2", "nbformat==5.10.4"):
        assert pin in requirements
