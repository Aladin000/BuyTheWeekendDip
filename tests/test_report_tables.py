"""The report's generated numbers must match the accepted runs and the brief's layout."""

from __future__ import annotations

import csv
import re

import pytest

from src import paths, replay, report_tables, robustness

REPORT = paths.resolve("report/report.tex")
BRIEF_SECTIONS = (
    "Hypothesis Pitch",
    "Data Pipeline",
    "Strategy and Backtest",
    "Methodology Defense",
    "Pitfall Checklist",
    "Efficiency Touchpoint",
    "Risk Touchpoint",
    "Reflection",
)
PITFALLS = (
    "Look-ahead bias",
    "Survivorship bias",
    "Overfitting/data-snooping",
    "Transaction costs and slippage",
    "Out-of-sample validation",
)


@pytest.fixture(scope="module")
def generated() -> str:
    return report_tables.render()


def test_saved_generated_file_is_up_to_date(generated: str) -> None:
    saved = paths.resolve(report_tables.OUTPUT_RELATIVE_PATH).read_text(encoding="utf-8")
    assert saved == generated, "run `python -m src.report_tables` and rebuild the PDF"


def test_every_macro_used_in_the_report_is_defined(generated: str) -> None:
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", generated))
    defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", REPORT.read_text(encoding="utf-8")))
    used = set(re.findall(r"\\((?:Dev|Ho)[A-Za-z]+)", REPORT.read_text(encoding="utf-8")))
    assert used <= defined, sorted(used - defined)


def test_key_results_match_saved_strategy_outputs(generated: str) -> None:
    expected = {
        "DevAllNet": "-57.47%",
        "DevWeekendNet": "1.88%",
        "HoAllNet": "-52.57%",
        "HoWeekendNet": "-11.74%",
        "HoAllSharpe": "-1.328",
        "HoAllDrawdown": "-57.10%",
        "HoBuyHold": "48.37%",
    }
    for name, value in expected.items():
        latex = value.replace("-", "\\textminus{}", 1).replace("%", "\\%")
        assert f"\\newcommand{{\\{name}}}{{{latex}}}" in generated, name

    sources = {
        "development": robustness.ACCEPTED_BACKTEST_MANIFEST.parent / "development_strategy_metrics.csv",
        "held_out": replay.ACCEPTED_HELD_OUT_MANIFEST.parent / "held_out_strategy_metrics.csv",
        "benchmark": replay.ACCEPTED_HELD_OUT_MANIFEST.parent / "held_out_benchmarks.csv",
    }
    rows = {}
    for name, relative in sources.items():
        with paths.resolve(relative).open(newline="") as handle:
            rows[name] = [r for r in csv.DictReader(handle) if r["cost_scenario"] == "primary"]
    checks = {
        "DevAllNet": ("development", "all_shock", "net_cumulative_return", ".2%"),
        "DevWeekendNet": ("development", "weekend", "net_cumulative_return", ".2%"),
        "HoAllNet": ("held_out", "all_shock", "net_cumulative_return", ".2%"),
        "HoWeekendNet": ("held_out", "weekend", "net_cumulative_return", ".2%"),
        "HoAllSharpe": ("held_out", "all_shock", "sharpe_ratio", ".3f"),
        "HoAllDrawdown": ("held_out", "all_shock", "maximum_drawdown", ".2%"),
        "HoBuyHold": ("benchmark", "btc_buy_and_hold", "net_cumulative_return", ".2%"),
    }
    for macro, (source, account, field, fmt) in checks.items():
        name_field = "benchmark_name" if source == "benchmark" else "strategy_name"
        row = next(r for r in rows[source] if r[name_field] == account)
        assert format(float(row[field]), fmt) == expected[macro], macro


def test_report_uses_the_brief_headings_in_order() -> None:
    text = REPORT.read_text(encoding="utf-8")
    sections = re.findall(r"^\\section\{([^}]*)\}", text, flags=re.MULTILINE)
    assert tuple(sections) == BRIEF_SECTIONS
    pitfall_block = text.split("\\section{Pitfall Checklist}")[1].split("\\section{")[0]
    subsections = re.findall(r"^\\subsection\{([^}]*)\}", pitfall_block, flags=re.MULTILINE)
    assert tuple(subsections) == PITFALLS


def test_report_starts_with_name_date_and_title() -> None:
    text = REPORT.read_text(encoding="utf-8")
    assert "[Submission date]" not in text
    assert re.search(r"\\newcommand\{\\SubmissionDate\}\{[^\[\]{}]+\}", text)
    body = text.split("\\begin{document}")[1]
    header = body.split("\\section{")[0]
    for item in ("Buy the Weekend Dip?", "\\StudentName", "\\SubmissionDate"):
        assert item in header, item
    assert header.index("\\StudentName") < header.index("\\SubmissionDate") < header.index("Buy the Weekend Dip?")


def test_report_includes_required_uncertainty_and_horizons(generated: str) -> None:
    report = REPORT.read_text(encoding="utf-8")
    for item in ("\\InferenceDetailRows", "\\GroupComparisonRows"):
        assert item in report
        assert f"\\newcommand{{{item}}}{{%" in generated
    for item in ("\\DevSixMean", "\\HoSixMean", "\\DevTwentyFourMean", "\\HoTwentyFourMean"):
        assert item in report
        assert f"\\newcommand{{{item}}}" in generated
    for item in ("Raw $p$", "HC3 SE", "Block 95\\% CI", "Draws A/V/E/R"):
        assert item in report


def test_every_citation_has_a_reference_and_every_reference_is_cited() -> None:
    text = REPORT.read_text(encoding="utf-8")
    cited = {key.strip() for group in re.findall(r"\\cite\{([^}]*)\}", text) for key in group.split(",")}
    listed = set(re.findall(r"\\bibitem\{([^}]*)\}", text))
    assert cited == listed


def test_report_avoids_forbidden_claims() -> None:
    text = REPORT.read_text(encoding="utf-8").lower()
    for phrase in ("proves", "causes overreaction", "live out-of-sample", "the strategy is profitable"):
        assert phrase not in text, phrase
