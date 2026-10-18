"""Write the report's numbers and tables as LaTeX from the accepted runs.

Every value in report/generated/results.tex is read from a saved output of
an accepted run; nothing is recalculated from market data and nothing is
typed by hand. Run with ``python -m src.report_tables``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src import backtest, count_audit, features, heldout, inference, outcomes, paths, robustness
from src.backtest import ACCEPTED_INFERENCE_MANIFEST
from src.config import load_config
from src.count_audit import ACCEPTED_FEATURE_MANIFEST
from src.freeze import ACCEPTED_ROBUSTNESS_MANIFEST, ACCEPTED_VALIDATION_MANIFEST
from src.inference import ACCEPTED_OUTCOME_MANIFEST
from src.outcomes import ACCEPTED_COUNT_MANIFEST
from src.robustness import ACCEPTED_BACKTEST_MANIFEST

HELD_OUT_RUN = Path("outputs/runs/20260910T133543.269946Z-held-out-dad36cad-632a5e")
HELD_OUT_ROBUSTNESS_RUN = Path(
    "outputs/runs/20260910T144404.535764Z-held-out-robustness-de2ac148-51637c"
)
OUTPUT_RELATIVE_PATH = Path("report/generated/results.tex")

RUNS = {
    "validation": ACCEPTED_VALIDATION_MANIFEST.parent,
    "features": ACCEPTED_FEATURE_MANIFEST.parent,
    "counts": ACCEPTED_COUNT_MANIFEST.parent,
    "outcomes": ACCEPTED_OUTCOME_MANIFEST.parent,
    "development_models": ACCEPTED_INFERENCE_MANIFEST.parent,
    "development_backtest": ACCEPTED_BACKTEST_MANIFEST.parent,
    "development_robustness": ACCEPTED_ROBUSTNESS_MANIFEST.parent,
    "held_out": HELD_OUT_RUN,
    "held_out_robustness": HELD_OUT_ROBUSTNESS_RUN,
}

SPLITS = ("development", "held_out")
SPLIT_PREFIX = {"development": "Dev", "held_out": "Ho"}
STRATEGIES = ("all_shock", "weekend", "low_activity", "combined")
STRATEGY_LABEL = {
    "all_shock": "All shock",
    "weekend": "Weekend",
    "low_activity": "Low activity",
    "combined": "Combined",
}
STRATEGY_MACRO = {
    "all_shock": "All",
    "weekend": "Weekend",
    "low_activity": "Low",
    "combined": "Combined",
}


def _rows(relative: Path) -> list[dict[str, str]]:
    with paths.resolve(relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_file(split: str, development_run: str, name: str) -> Path:
    if split == "development":
        return RUNS[development_run] / f"development_{name}"
    return RUNS["held_out"] / f"held_out_{name}"


def _robustness_file(split: str) -> Path:
    if split == "development":
        return RUNS["development_robustness"] / "development_robustness_results.csv"
    return RUNS["held_out_robustness"] / "held_out_robustness_results.csv"


def _one(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    found = [row for row in rows if all(row[key] == value for key, value in match.items())]
    if len(found) != 1:
        raise ValueError(f"expected one row for {match}, found {len(found)}")
    return found[0]


def _minus(text: str) -> str:
    return "\\textminus{}" + text[1:] if text.startswith("-") else text


def pct(value: float | str, digits: int = 2) -> str:
    """A decimal fraction written as a percentage, e.g. -0.5747 -> -57.47%."""
    number = float(value) * 100
    if round(number, digits) == 0:
        number = 0.0
    return _minus(f"{number:.{digits}f}") + "\\%"


def dec(value: float | str, digits: int = 3) -> str:
    number = float(value)
    if round(number, digits) == 0:
        number = 0.0
    return _minus(f"{number:.{digits}f}")


def whole(value: float | str) -> str:
    return f"{int(float(value)):,}".replace(",", "{,}")


def interval(low: float | str, high: float | str) -> str:
    return f"[{pct(low)}, {pct(high)}]"


def _macro(name: str, value: str) -> str:
    if not name.isalpha():
        raise ValueError(f"LaTeX macro names must be letters only: {name}")
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _quality(macros: dict[str, str]) -> list[str]:
    quality = json.loads(
        paths.resolve(RUNS["validation"] / "data_quality_summary.json").read_text(
            encoding="utf-8"
        )
    )
    counts = _rows(RUNS["counts"] / "feasibility_counts.csv")
    for name, key in (
        ("SourceFiles", "source_files"),
        ("ParsedRows", "parsed_data_rows"),
        ("CleanRows", "accepted_complete_rows"),
        ("ExcludedRows", "nonstandard_close_rows_excluded"),
        ("DuplicateRows", "duplicate_rows"),
        ("InvalidRows", "invalid_rows"),
        ("MissingHours", "missing_hours"),
        ("Segments", "contiguous_segments"),
        ("AnomalyCandidates", "anomaly_candidates"),
    ):
        macros[name] = whole(quality[key])

    def split_count(split: str, measure: str) -> str:
        return _one(
            counts, scope_type="split", scope=split, sign="", weekend_group="",
            activity_group="", measure=measure,
        )["count"]

    lines = []
    for label, measure in (
        ("Observed clean hours", "analysis_hours"),
        ("Hours with a valid shock score", "shock_eligible_hours"),
        ("Hours with every feature", "full_feature_eligible_hours"),
    ):
        values = [split_count(split, measure) for split in SPLITS]
        lines.append(f"{label} & {whole(values[0])} & {whole(values[1])} \\\\")
    for split in SPLITS:
        prefix = SPLIT_PREFIX[split]
        macros[f"{prefix}Hours"] = whole(split_count(split, "analysis_hours"))
        macros[f"{prefix}ShockHours"] = whole(split_count(split, "shock_eligible_hours"))
        macros[f"{prefix}FeatureHours"] = whole(
            split_count(split, "full_feature_eligible_hours")
        )
    return lines


def _events(macros: dict[str, str]) -> list[str]:
    counts = _rows(RUNS["counts"] / "feasibility_counts.csv")

    def count(split: str, **match: str) -> str:
        base = {
            "scope_type": "split", "scope": split, "sign": "negative",
            "weekend_group": "", "activity_group": "", "horizon_hours": "",
            "model_or_strategy": "",
        }
        base.update(match)
        return _one(counts, **base)["count"]

    spec = (
        ("Negative candidates ($z<-2.5$)", {"measure": "candidates"}, "Candidates"),
        ("Accepted after the 24-hour cooldown", {"measure": "accepted_events"}, "Events"),
        ("\\quad weekend", {"measure": "accepted_events", "weekend_group": "weekend"}, "WeekendEvents"),
        ("\\quad weekday", {"measure": "accepted_events", "weekend_group": "weekday"}, "WeekdayEvents"),
        ("\\quad low activity", {"measure": "accepted_events", "activity_group": "low"}, "LowEvents"),
        ("\\quad activity rank unavailable", {"measure": "accepted_events", "activity_group": "unavailable"}, "UnavailableEvents"),
        ("\\quad weekend and low activity", {"measure": "accepted_events", "weekend_group": "weekend", "activity_group": "low"}, "CombinedEvents"),
        ("Complete 12-hour outcome (H1, H2)", {"measure": "structurally_eligible_events", "horizon_hours": "12", "model_or_strategy": "H1_H2"}, "SampleHOneN"),
        ("Complete outcome and features (H3, H4)", {"measure": "structurally_eligible_events", "horizon_hours": "12", "model_or_strategy": "H3_H4"}, "SampleHThreeN"),
        ("Accepted positive events ($z>2.5$)", {"measure": "accepted_events", "sign": "positive"}, "PositiveEvents"),
    )
    lines = []
    for label, match, macro in spec:
        values = [count(split, **match) for split in SPLITS]
        for split, value in zip(SPLITS, values):
            macros[f"{SPLIT_PREFIX[split]}{macro}"] = whole(value)
        lines.append(f"{label} & {whole(values[0])} & {whole(values[1])} \\\\")
    return lines


def _metrics(split: str) -> list[dict[str, str]]:
    return _rows(_split_file(split, "development_backtest", "strategy_metrics.csv"))


def _benchmarks(split: str) -> list[dict[str, str]]:
    return _rows(_split_file(split, "development_backtest", "benchmarks.csv"))


def _key_results(macros: dict[str, str]) -> list[str]:
    lines = []
    for strategy in STRATEGIES:
        cells = [STRATEGY_LABEL[strategy]]
        for split in SPLITS:
            row = _one(_metrics(split), strategy_name=strategy, cost_scenario="primary")
            prefix = SPLIT_PREFIX[split] + STRATEGY_MACRO[strategy]
            values = {
                "Trades": whole(row["trade_count"]),
                "Gross": pct(row["gross_cumulative_return"]),
                "Net": pct(row["net_cumulative_return"]),
                "Sharpe": dec(row["sharpe_ratio"]),
                "Drawdown": pct(row["maximum_drawdown"]),
                "HitRate": pct(row["net_hit_rate"], 1),
                "Invested": pct(row["time_invested_fraction"], 1),
                "TopFiveShare": pct(row["top5_trade_share_absolute_pnl"], 1),
            }
            for name, value in values.items():
                macros[prefix + name] = value
            cells += [values["Trades"], values["Net"], values["Sharpe"], values["Drawdown"]]
        lines.append(" & ".join(cells) + " \\\\")
    cells = ["BTC buy and hold"]
    for split in SPLITS:
        row = _one(_benchmarks(split), benchmark_name="btc_buy_and_hold", cost_scenario="primary")
        macros[f"{SPLIT_PREFIX[split]}BuyHold"] = pct(row["net_cumulative_return"])
        cells += ["1", pct(row["net_cumulative_return"]), "n/r", "n/r"]
    lines.append("\\midrule")
    lines.append(" & ".join(cells) + " \\\\")
    return lines


def _costs(macros: dict[str, str]) -> list[str]:
    lines = []
    for split in SPLITS:
        label = "Development" if split == "development" else "Held out"
        for index, strategy in enumerate(STRATEGIES):
            rows = {
                scenario: _one(_metrics(split), strategy_name=strategy, cost_scenario=scenario)
                for scenario in ("low", "primary", "high")
            }
            primary = rows["primary"]
            if primary["break_even_status"] == "no_positive_break_even_cost":
                break_even = "none"
            else:
                break_even = dec(float(primary["break_even_cost_per_side"]) * 10_000, 1) + " bp"
            macros[f"{SPLIT_PREFIX[split]}{STRATEGY_MACRO[strategy]}BreakEven"] = break_even
            for scenario in ("low", "high"):
                macros[
                    f"{SPLIT_PREFIX[split]}{STRATEGY_MACRO[strategy]}{scenario.title()}Cost"
                ] = pct(rows[scenario]["net_cumulative_return"])
            first = label if index == 0 else ""
            lines.append(
                " & ".join(
                    [
                        first,
                        STRATEGY_LABEL[strategy],
                        pct(primary["gross_cumulative_return"]),
                        pct(rows["low"]["net_cumulative_return"]),
                        pct(primary["net_cumulative_return"]),
                        pct(rows["high"]["net_cumulative_return"]),
                        break_even,
                    ]
                )
                + " \\\\"
            )
        if split == "development":
            lines.append("\\midrule")
    return lines


def _groups(macros: dict[str, str]) -> list[str]:
    spec = (
        ("All negative events", "negative_events", "all", "All"),
        ("\\quad weekend", "weekend", "weekend", "Weekend"),
        ("\\quad weekday", "weekend", "weekday", "Weekday"),
        ("\\quad low activity", "activity", "low_activity", "Low"),
        ("\\quad other activity", "activity", "other_activity", "Other"),
        ("\\quad weekend and low activity", "combined", "weekend_low_activity", "Combined"),
        ("Positive events", "positive_events", "all", "Positive"),
    )
    tables = {
        split: _rows(_split_file(split, "development_models", "descriptives.csv"))
        for split in SPLITS
    }
    lines = []
    for label, comparison, group, macro in spec:
        cells = [label]
        for split in SPLITS:
            rows = tables[split]
            values = {}
            for statistic in ("mean", "median", "hit_rate"):
                values[statistic] = _one(
                    rows, section="event_descriptives", comparison=comparison, group=group,
                    variable="post_log_return_12h", statistic=statistic,
                )
            n = values["mean"]["n"]
            prefix = f"{SPLIT_PREFIX[split]}{macro}"
            macros[prefix + "Mean"] = pct(values["mean"]["value"])
            macros[prefix + "Median"] = pct(values["median"]["value"])
            macros[prefix + "Hit"] = pct(values["hit_rate"]["value"], 1)
            macros[prefix + "N"] = whole(n)
            cells += [
                whole(n), pct(values["mean"]["value"]), pct(values["median"]["value"]),
                pct(values["hit_rate"]["value"], 1),
            ]
        lines.append(" & ".join(cells) + " \\\\")
    cells = ["Non-candidate hours (context only)"]
    for split in SPLITS:
        mean = _one(tables[split], section="market_drift_reference", statistic="mean",
                    variable="post_log_return_12h")
        median = _one(tables[split], section="market_drift_reference", statistic="median",
                      variable="post_log_return_12h")
        macros[f"{SPLIT_PREFIX[split]}DriftMean"] = pct(mean["value"], 3)
        macros[f"{SPLIT_PREFIX[split]}DriftMedian"] = pct(median["value"], 3)
        cells += [whole(mean["n"]), pct(mean["value"], 3), pct(median["value"], 3), ""]
    lines.append("\\midrule")
    lines.append(" & ".join(cells) + " \\\\")
    return lines


HYPOTHESES = (
    ("H1", "Mean return", "model0", "intercept", "H1_model0", "HOne"),
    ("H2", "Weekend", "model1", "weekend", "H2_model1", "HTwo"),
    ("H3", "Activity score", "model2", "low_activity_score", "H3_model2", "HThree"),
    ("H4", "Weekend, controlled", "model2", "weekend", "H4_model2", "HFour"),
)


def _model_rows(split: str) -> list[dict[str, str]]:
    return _rows(_split_file(split, "development_models", "model_results.csv"))


def _bootstrap_rows(split: str) -> list[dict[str, str]]:
    return _rows(_split_file(split, "development_models", "bootstrap_results.csv"))


def _hypotheses(macros: dict[str, str]) -> list[str]:
    lines = []
    for code, label, model, term, hypothesis, macro in HYPOTHESES:
        cells = [f"{code} & {label}"]
        for split in SPLITS:
            row = _one(_model_rows(split), model=model, term=term, hypothesis=hypothesis)
            boot = _one(
                _bootstrap_rows(split), analysis="model_inference", model=model, term=term
            )
            supported = (
                float(row["estimate"]) > 0 and float(row["holm_adjusted_p_value"]) < 0.05
            )
            prefix = f"{SPLIT_PREFIX[split]}{macro}"
            macros[prefix + "N"] = whole(row["n"])
            macros[prefix + "Est"] = pct(row["estimate"])
            macros[prefix + "CI"] = interval(row["ci_95_low"], row["ci_95_high"])
            macros[prefix + "P"] = dec(row["raw_p_value"])
            macros[prefix + "Holm"] = dec(row["holm_adjusted_p_value"])
            macros[prefix + "Boot"] = interval(
                boot["percentile_95_low"], boot["percentile_95_high"]
            )
            cells.append(
                " & ".join(
                    [
                        whole(row["n"]),
                        pct(row["estimate"]),
                        interval(row["ci_95_low"], row["ci_95_high"]),
                        dec(row["holm_adjusted_p_value"]),
                    ]
                )
            )
            if split == "held_out":
                cells.append("Yes" if supported else "No")
        lines.append(" & ".join(cells) + " \\\\")
    return lines


def _inference_details(macros: dict[str, str]) -> list[str]:
    """Report the remaining locked inference fields beside the main table."""
    lines = []
    for split in SPLITS:
        for code, _, model, term, hypothesis, _ in HYPOTHESES:
            row = _one(_model_rows(split), model=model, term=term, hypothesis=hypothesis)
            boot = _one(_bootstrap_rows(split), analysis="model_inference", model=model, term=term)
            label = "Development" if split == "development" else "Held out"
            lines.append(" & ".join([
                label, code, pct(row["hc3_standard_error"]), dec(row["raw_p_value"]),
                dec(row["holm_adjusted_p_value"]),
                interval(boot["percentile_95_low"], boot["percentile_95_high"]),
                "/".join(boot[key] for key in ("attempted", "valid", "empty", "rank_deficient")),
            ]) + " \\\\")
        if split == "development":
            lines.append("\\midrule")
    return lines


def _group_comparisons(macros: dict[str, str]) -> list[str]:
    lines = []
    labels = {
        "weekend_vs_weekday": "Weekend minus weekday",
        "low_vs_other_activity": "Low minus other activity",
        "combined_vs_remaining": "Combined minus remaining",
    }
    for split in SPLITS:
        rows = _rows(_split_file(split, "development_models", "group_comparisons.csv"))
        boots = _bootstrap_rows(split)
        for comparison, label in labels.items():
            row = _one(rows, comparison=comparison)
            boot = _one(boots, analysis=comparison, model="group_indicator", term="group_indicator")
            lines.append(" & ".join([
                "Development" if split == "development" else "Held out", label,
                whole(row["n_1"]), whole(row["n_0"]), pct(row["difference_1_minus_0"]),
                interval(row["ci_95_low"], row["ci_95_high"]),
                interval(boot["percentile_95_low"], boot["percentile_95_high"]),
            ]) + " \\\\")
        if split == "development":
            lines.append("\\midrule")
    return lines


def _horizon_results(macros: dict[str, str]) -> None:
    for split in SPLITS:
        rows = _rows(_robustness_file(split))
        for horizon, label in (("6", "Six"), ("24", "TwentyFour")):
            row = _one(rows, family="horizon", variant=horizon, analysis="inference",
                       model="model0", term="intercept", metric="coefficient")
            prefix = f"{SPLIT_PREFIX[split]}{label}"
            macros[f"{prefix}N"] = whole(row["sample_n"])
            macros[f"{prefix}Mean"] = pct(row["estimate"])
            macros[f"{prefix}CI"] = interval(row["range_low"], row["range_high"])


def _secondary(macros: dict[str, str]) -> None:
    for split in SPLITS:
        rows = _model_rows(split)
        prefix = SPLIT_PREFIX[split]
        for macro, model, term in (
            ("HFive", "h5_sign", "positive"),
            ("HFiveAny", "h5_any_sign", "positive"),
            ("Interaction", "model3", "weekend_x_low_activity_score"),
            ("WeekendCC", "model1_cc", "weekend"),
        ):
            row = _one(rows, analysis=_analysis_for(model), model=model, term=term)
            macros[f"{prefix}{macro}N"] = whole(row["n"])
            macros[f"{prefix}{macro}Est"] = pct(row["estimate"])
            macros[f"{prefix}{macro}CI"] = interval(row["ci_95_low"], row["ci_95_high"])
            macros[f"{prefix}{macro}P"] = dec(row["raw_p_value"])
            if macro in {"HFive", "Interaction"}:
                boot = _one(_bootstrap_rows(split), analysis="model_inference",
                            model=model, term=term)
                macros[f"{prefix}{macro}Boot"] = interval(
                    boot["percentile_95_low"], boot["percentile_95_high"]
                )
        positive = _one(rows, model="positive_model0", term="intercept")
        macros[f"{prefix}PositiveMeanCI"] = interval(positive["ci_95_low"], positive["ci_95_high"])
        bootstraps = _bootstrap_rows(split)
        hypothesis_boots = [
            _one(bootstraps, analysis="model_inference", model=model, term=term)
            for _, _, model, term, _, _ in HYPOTHESES
        ]
        includes_zero = all(
            float(row["percentile_95_low"]) <= 0 <= float(row["percentile_95_high"])
            for row in hypothesis_boots
        )
        macros[f"{prefix}BootAllIncludeZero"] = "all" if includes_zero else "not all"


def _analysis_for(model: str) -> str:
    return "positive_shock_comparison" if model.startswith("h5") else "primary"


def _robustness(macros: dict[str, str]) -> list[str]:
    tables = {split: _rows(_robustness_file(split)) for split in SPLITS}

    def weekend(split: str, family: str, variant: str) -> tuple[str, str]:
        row = _one(
            tables[split], family=family, variant=variant, strategy="weekend",
            metric="primary_net_cumulative_return",
        )
        return pct(row["estimate"]), whole(row["trade_count"])

    spec = (
        ("Primary rule", "threshold", "-2.5"),
        ("Shock threshold $-2.0$", "threshold", "-2.0"),
        ("Shock threshold $-3.0$", "threshold", "-3.0"),
        ("Hold 6 hours", "horizon", "6"),
        ("Hold 24 hours", "horizon", "24"),
        ("Enter one hour later", "execution", "delayed_one_hour"),
        ("Cooldown 12 hours", "cooldown", "12"),
        ("Cooldown 48 hours", "cooldown", "48"),
        ("Signals inside the weekend", "weekend_boundary", "weekend_contained"),
        ("Signals whose hold reaches Monday", "weekend_boundary", "monday_crossing"),
    )
    lines = []
    for label, family, variant in spec:
        cells = [label]
        for split in SPLITS:
            value, trades = weekend(split, family, variant)
            cells += [value, trades]
        lines.append(" & ".join(cells) + " \\\\")
    for label, family, macro in (
        ("Leave one event out (range)", "leave_one_event_out", "LeaveEvent"),
        ("Leave one year out (range)", "leave_one_year_out", "LeaveYear"),
    ):
        cells = [label]
        for split in SPLITS:
            row = _one(tables[split], family=family, metric="weekend_net_return")
            macros[f"{SPLIT_PREFIX[split]}{macro}Low"] = pct(row["range_low"])
            macros[f"{SPLIT_PREFIX[split]}{macro}High"] = pct(row["range_high"])
            cells += [f"{pct(row['range_low'])} to {pct(row['range_high'])}", ""]
        lines.append(" & ".join(cells) + " \\\\")
    for split in SPLITS:
        _, trades = weekend(split, "threshold", "-3.0")
        macros[f"{SPLIT_PREFIX[split]}StrictThresholdTrades"] = trades

    for split in SPLITS:
        prefix = SPLIT_PREFIX[split]
        rows = tables[split]
        h1 = [
            row for row in rows
            if row["model"] == "model0" and row["term"] == "intercept"
            and row["metric"] == "coefficient" and row["family"] in {"threshold", "horizon", "cooldown"}
        ]
        macros[f"{prefix}HOneVariantsIncludeZero"] = (
            "every" if all(float(r["range_low"]) <= 0 <= float(r["range_high"]) for r in h1) else "not every"
        )
        primary = {
            ("threshold", "-2.5"), ("horizon", "12"), ("execution", "next_open"),
            ("cooldown", "24"),
        }
        alternatives = [
            row for row in rows
            if row["family"] in {"threshold", "horizon", "execution", "cooldown"}
            and (row["family"], row["variant"]) not in primary
            and row["strategy"] == "weekend" and row["metric"] == "primary_net_cumulative_return"
        ]
        positive = [row for row in alternatives if float(row["estimate"]) > 0]
        macros[f"{prefix}WeekendVariants"] = whole(len(alternatives))
        macros[f"{prefix}WeekendPositiveVariants"] = whole(len(positive))
        years = [
            row for row in rows
            if row["family"] == "calendar_year" and row["strategy"] == "weekend"
            and row["metric"] == "primary_net_cumulative_return"
        ]
        macros[f"{prefix}WeekendYearsPositive"] = whole(
            sum(float(row["estimate"]) > 0 for row in years)
        )
        macros[f"{prefix}WeekendYears"] = whole(len(years))
    return lines


def _verify_accepted_inputs() -> None:
    """Refuse changed evidence before reading any reported value."""
    config = load_config()
    features.verify_validation_input(config, ACCEPTED_VALIDATION_MANIFEST)
    count_audit.verify_feature_input(
        config, ACCEPTED_FEATURE_MANIFEST, require_frozen_acceptance=True
    )
    outcomes.verify_count_input(
        config, ACCEPTED_COUNT_MANIFEST, require_frozen_acceptance=True
    )
    outcomes.verify_development_inventory(ACCEPTED_OUTCOME_MANIFEST)
    inference.verify_inference_inventory(ACCEPTED_INFERENCE_MANIFEST)
    backtest.verify_backtest_inventory(ACCEPTED_BACKTEST_MANIFEST)
    robustness.verify_inventory(ACCEPTED_ROBUSTNESS_MANIFEST, config)
    heldout.verify_held_out_inventory(HELD_OUT_RUN / "held-out-evaluation.manifest.json")
    robustness.verify_held_out_robustness_inventory(
        HELD_OUT_ROBUSTNESS_RUN / "held-out-robustness.manifest.json", config
    )


def render() -> str:
    _verify_accepted_inputs()
    macros: dict[str, str] = {}
    tables = {
        "QualityRows": _quality(macros),
        "EventRows": _events(macros),
        "KeyResultRows": _key_results(macros),
        "CostRows": _costs(macros),
        "GroupRows": _groups(macros),
        "HypothesisRows": _hypotheses(macros),
        "InferenceDetailRows": _inference_details(macros),
        "GroupComparisonRows": _group_comparisons(macros),
        "RobustnessRows": _robustness(macros),
    }
    _horizon_results(macros)
    _secondary(macros)
    for value in macros.values():
        if "nan" in value.lower() or "inf" in value.lower():
            raise ValueError(f"non-finite value reached the report: {value}")
    lines = [
        "% Generated by `python -m src.report_tables` from the accepted runs.",
        "% Do not edit by hand.",
    ]
    lines += [_macro(name, macros[name]) for name in sorted(macros)]
    for name, rows in tables.items():
        lines.append(f"\\newcommand{{\\{name}}}{{%")
        lines += rows
        lines.append("}")
    return "\n".join(lines) + "\n"


def write() -> Path:
    target = paths.resolve(OUTPUT_RELATIVE_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(write())
