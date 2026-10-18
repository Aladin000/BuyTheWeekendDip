"""Pre-specified robustness checks for development and held-out evidence."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src import paths
from src.acquisition import verify_manifest_checksum, write_manifest_checksum
from src.artifacts import (
    build_manifest,
    matches_active_protocol,
    create_exclusive,
    create_exclusive_bytes,
    create_run_log,
    read_manifest,
    run_relative_dir,
    sha256_bytes,
    sha256_file,
)
from src.backtest import (
    ACCEPTED_INFERENCE_MANIFEST,
    BacktestHour,
    _scenario_cost,
    build_trade_plans,
    simulate_trades,
    verify_backtest_inventory,
)
from src.config import ConfigNode, load_config
from src.count_audit import ACCEPTED_FEATURE_MANIFEST, verify_feature_input
from src.events import construct_sign_specific_events
from src.features import FEATURE_COLUMNS
from src.inference import (
    ACCEPTED_OUTCOME_MANIFEST,
    EventRecord,
    build_model_data,
    fit_hc3,
    read_event_records,
)

STAGE = "9. Development robustness"
MODE = "robustness"
HELD_OUT_STAGE = "11. Held-out robustness completion"
HELD_OUT_MODE = "held-out-robustness"
ACCEPTED_BACKTEST_MANIFEST = Path(
    "outputs/runs/20260908T165150.697815Z-backtest-509b876c-aa730c/development-backtest.manifest.json"
)
ACCEPTED_BACKTEST_SHA256 = (
    "c51cc4538b8139e8de1e3ae82ed36b7a77aa6fb4f0d4bbac96a2bcbd526536fb"
)
RESULT_COLUMNS = (
    "family",
    "variant",
    "analysis",
    "strategy",
    "model",
    "term",
    "metric",
    "estimate",
    "range_low",
    "range_high",
    "sample_n",
    "trade_count",
    "status",
    "note",
)
INVENTORY_COLUMNS = (
    "artifact_path",
    "sha256",
    "record_count",
    "artifact_type",
    "column_schema_json",
)
HELD_OUT_RESULT_NAME = "held_out_robustness_results.csv"
HELD_OUT_FIGURE_NAME = "held_out_calendar_year_stability.png"
HELD_OUT_INVENTORY_NAME = "held_out_robustness_inventory.csv"
HELD_OUT_MANIFEST_SLUG = "held-out-robustness"
ACCEPTED_PRIMARY_HELD_OUT_MANIFEST = Path(
    "outputs/runs/20260910T133543.269946Z-held-out-dad36cad-632a5e/"
    "held-out-evaluation.manifest.json"
)
ACCEPTED_PRIMARY_HELD_OUT_SHA256 = "cc5b89e69fd61c70eb558c7af80c162bec9807d14b0a7aa223e7171ca151f681"


class RobustnessError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hour:
    open_time: datetime
    close_time: datetime
    raw: int
    segment: int
    period: str
    open: float | None
    close: float | None
    eligible: bool
    z: float | None
    weekend: bool | None
    block: str
    year: int | None
    month: str
    rv24: float | None
    quote_p: float | None
    quote_score: float | None
    quote_low: bool | None
    trade_p: float | None
    trade_score: float | None
    trade_low: bool | None
    full: bool


@dataclass(frozen=True)
class RobustnessBundle:
    rows: tuple[dict[str, object], ...]
    figure: tuple[str, bytes]


@dataclass(frozen=True)
class DevelopmentRobustnessResult:
    run_id: str
    manifest_relative_path: Path
    manifest_checksum_relative_path: Path
    log_relative_path: Path
    inventory_relative_path: Path
    result_rows: int


@dataclass(frozen=True)
class HeldOutRobustnessResult:
    run_id: str
    manifest_relative_path: Path
    manifest_checksum_relative_path: Path
    log_relative_path: Path
    inventory_relative_path: Path
    result_rows: int


def _utc(text: str) -> datetime:
    if not text.endswith("Z"):
        raise RobustnessError("timestamp is not UTC")
    try:
        return datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        raise RobustnessError("invalid timestamp") from None


def _float(text: str, optional: bool = False) -> float | None:
    if optional and text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        raise RobustnessError("invalid numeric field") from None
    if not math.isfinite(value):
        raise RobustnessError("non-finite numeric field")
    return value


def _bool(text: str, optional: bool = False) -> bool | None:
    if optional and text == "":
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    raise RobustnessError("invalid boolean field")


def read_hours(
    relative: Path, expected: int, *, analysis_period: str = "development"
) -> tuple[Hour, ...]:
    if analysis_period not in {"development", "held_out"}:
        raise RobustnessError("robustness split must be development or held_out")
    rows: list[Hour] = []
    with paths.require_file(paths.resolve(relative)).open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle, strict=True)
        if tuple(reader.fieldnames or ()) != FEATURE_COLUMNS:
            raise RobustnessError("invalid feature schema")
        for source in reader:
            period = source["analysis_period"]
            if period not in {"warmup", "development", "held_out"}:
                raise RobustnessError("invalid split label")
            analytical = period != "held_out" or analysis_period == "held_out"
            selected = period == analysis_period
            rows.append(
                Hour(
                    _utc(source["open_time"]),
                    _utc(source["close_time"]),
                    int(source["close_time_raw"]) if analytical else 0,
                    int(source["contiguous_segment_id"]),
                    period,
                    _float(source["open"]) if selected else None,
                    _float(source["close"]) if selected else None,
                    bool(_bool(source["shock_eligible_hour"])) if analytical else False,
                    _float(source["shock_zscore"], True) if analytical else None,
                    _bool(source["weekend"]) if selected else None,
                    source["utc_time_block"] if selected else "",
                    int(source["calendar_year"]) if selected else None,
                    source["calendar_month"] if selected else "",
                    _float(source["rv24"], True) if selected else None,
                    (
                        _float(source["activity_percentile_90d"], True)
                        if selected
                        else None
                    ),
                    _float(source["low_activity_score"], True) if selected else None,
                    _bool(source["low_activity"], True) if selected else None,
                    (
                        _float(source["trade_activity_percentile_90d"], True)
                        if selected
                        else None
                    ),
                    (
                        _float(source["trade_low_activity_score"], True)
                        if selected
                        else None
                    ),
                    _bool(source["trade_low_activity"], True) if selected else None,
                    (
                        bool(_bool(source["full_feature_eligible_hour"]))
                        if selected
                        else False
                    ),
                )
            )
    if len(rows) != expected or any(
        b.open_time <= a.open_time for a, b in zip(rows, rows[1:])
    ):
        raise RobustnessError("feature rows do not reconcile")
    return tuple(rows)


def _outcome(
    rows: Sequence[Hour],
    i: int,
    horizon: int,
    *,
    analysis_period: str = "development",
) -> float | None:
    origin = rows[i]
    for step in range(1, horizon + 1):
        if i + step >= len(rows):
            return None
        future = rows[i + step]
        if (
            future.open_time != origin.open_time + timedelta(hours=step)
            or future.segment != origin.segment
            or future.period != analysis_period
        ):
            return None
    if origin.close is None or rows[i + horizon].close is None:
        raise RobustnessError("complete horizon lacks price")
    return math.log(rows[i + horizon].close / origin.close)


def rebuild_events(
    rows: Sequence[Hour],
    threshold: float,
    cooldown: int,
    positive_threshold: float = 2.5,
    *,
    analysis_period: str = "development",
) -> tuple[EventRecord, ...]:
    decisions = construct_sign_specific_events(
        [row.z if row.eligible else None for row in rows],
        [row.close_time for row in rows],
        [row.raw for row in rows],
        negative_threshold=threshold,
        positive_threshold=positive_threshold,
        cooldown_hours=cooldown,
    )
    output = []
    for i, (row, decision) in enumerate(zip(rows, decisions, strict=True)):
        if row.period != analysis_period or not (
            decision.negative_event or decision.positive_event
        ):
            continue
        sign = "negative" if decision.negative_event else "positive"
        y = _outcome(rows, i, 12, analysis_period=analysis_period)
        if (
            row.z is None
            or row.close is None
            or row.weekend is None
            or row.year is None
        ):
            raise RobustnessError("event lacks fields")
        crossing = (
            None
            if not row.weekend or y is None
            else any(rows[i + s].close_time.isoweekday() == 1 for s in range(1, 13))
        )
        output.append(
            EventRecord(
                f"{sign}-{row.raw}",
                sign,
                row.open_time,
                row.close_time,
                row.close,
                row.segment,
                row.year,
                row.month,
                row.weekend,
                crossing,
                row.block,
                0.0,
                row.z,
                -row.z if sign == "negative" else None,
                abs(row.z),
                row.quote_p,
                row.quote_score,
                row.quote_low,
                row.rv24,
                row.full,
                y is not None,
                _outcome(rows, i, 24, analysis_period=analysis_period) is not None,
                y,
                None if y is None else y > 0,
                sign == "negative" and y is not None,
                sign == "negative" and y is not None and row.full,
            )
        )
    return tuple(output)


def backtest_hours(
    rows: Sequence[Hour], *, analysis_period: str = "development"
) -> tuple[BacktestHour, ...]:
    return tuple(
        BacktestHour(
            x.open_time, x.close_time, x.raw, x.segment, float(x.open), float(x.close)
        )
        for x in rows
        if x.period == analysis_period
    )


def _freeze(value):
    if isinstance(value, dict):
        return ConfigNode({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def variant_config(config: ConfigNode, entry: int, exit_: int, hold: int) -> ConfigNode:
    data = config.to_dict()
    data["strategy"].update(
        entry_offset_bars=entry, exit_offset_bars=exit_, hold_hours=hold
    )
    return _freeze(data)


def with_horizon(
    events: Sequence[EventRecord],
    rows: Sequence[Hour],
    horizon: int,
    *,
    analysis_period: str = "development",
) -> tuple[EventRecord, ...]:
    positions = {x.close_time: i for i, x in enumerate(rows)}
    out = []
    for event in events:
        y = _outcome(
            rows,
            positions[event.signal_close_time],
            horizon,
            analysis_period=analysis_period,
        )
        out.append(
            replace(
                event,
                complete_12h=y is not None,
                outcome_12h=y,
                post_positive_12h=None if y is None else y > 0,
                eligible_h1_h2=event.sign == "negative" and y is not None,
                eligible_h3_h4=event.sign == "negative"
                and y is not None
                and event.full_feature_eligible,
            )
        )
    return tuple(out)


def with_trade_activity(
    events: Sequence[EventRecord],
    rows: Sequence[Hour],
    *,
    analysis_period: str = "development",
) -> tuple[EventRecord, ...]:
    lookup = {x.close_time: x for x in rows if x.period == analysis_period}
    output = []
    for event in events:
        row = lookup[event.signal_close_time]
        complete = (
            row.trade_p is not None
            and row.trade_score is not None
            and row.trade_low is not None
            and row.rv24 is not None
        )
        output.append(
            replace(
                event,
                activity_percentile=row.trade_p,
                low_activity_score=row.trade_score,
                low_activity=row.trade_low,
                full_feature_eligible=complete,
                eligible_h3_h4=event.sign == "negative"
                and event.complete_12h
                and complete,
            )
        )
    return tuple(output)


def _record(
    family,
    variant,
    analysis,
    *,
    strategy="",
    model="",
    term="",
    metric="coefficient",
    estimate=None,
    low=None,
    high=None,
    n=None,
    trades=None,
    status="complete",
    note="",
):
    return dict(
        family=family,
        variant=str(variant),
        analysis=analysis,
        strategy=strategy,
        model=model,
        term=term,
        metric=metric,
        estimate=estimate,
        range_low=low,
        range_high=high,
        sample_n=n,
        trade_count=trades,
        status=status,
        note=note,
    )


def inference_rows(
    family: str, variant: str, events: Sequence[EventRecord], config: ConfigNode
):
    negative = tuple(e for e in events if e.sign == "negative")
    specs = (
        ("model0", "intercept", [e for e in negative if e.eligible_h1_h2]),
        ("model1", "weekend", [e for e in negative if e.eligible_h1_h2]),
        ("model2", "weekend", [e for e in negative if e.eligible_h3_h4]),
        ("model2", "low_activity_score", [e for e in negative if e.eligible_h3_h4]),
    )
    output = []
    cache = {}
    for model, term, sample in specs:
        fit = cache.setdefault(
            model,
            fit_hc3(
                build_model_data(model, model, sample),
                config.inference.confidence_level,
            ),
        )
        idx = fit.term_index(term)
        output.append(
            _record(
                family,
                variant,
                "inference",
                model=model,
                term=term,
                estimate=fit.coefficient(term),
                low=None if fit.ci_low is None else float(fit.ci_low[idx]),
                high=None if fit.ci_high is None else float(fit.ci_high[idx]),
                n=len(sample),
                status=fit.status,
                note=fit.reason,
            )
        )
    return output


def strategy_rows(
    family: str,
    variant: str,
    events: Sequence[EventRecord],
    hours: Sequence[BacktestHour],
    config: ConfigNode,
    *,
    analysis_period: str = "development",
):
    _, plans = build_trade_plans(
        events, hours, config, analysis_period=analysis_period
    )
    fee, slip = _scenario_cost(config, "primary")
    output = []
    for strategy in config.strategy.variants:
        net = simulate_trades(
            plans[strategy], "primary", fee, slip, config.strategy.initial_capital
        )
        terminal = net[-1].capital_after if net else config.strategy.initial_capital
        output.append(
            _record(
                family,
                variant,
                "strategy",
                strategy=strategy,
                metric="primary_net_cumulative_return",
                estimate=terminal / config.strategy.initial_capital - 1,
                n=sum(e.sign == "negative" for e in events),
                trades=len(net),
                status=(
                    "descriptive_highly_unstable"
                    if len(net) < config.strategy.sparse_trade_label_threshold
                    else "complete"
                ),
            )
        )
    return output


def _core_values(
    events: Sequence[EventRecord], config: ConfigNode
) -> dict[str, float | None]:
    values = {}
    for row in inference_rows("internal", "internal", events, config):
        values[f"{row['model']}:{row['term']}"] = row["estimate"]
    return values


def _weekend_return(
    events: Sequence[EventRecord],
    hours: Sequence[BacktestHour],
    config: ConfigNode,
    *,
    analysis_period: str = "development",
) -> tuple[float, int]:
    _, plans = build_trade_plans(
        events, hours, config, analysis_period=analysis_period
    )
    fee, slip = _scenario_cost(config, "primary")
    trades = simulate_trades(
        plans["weekend"], "primary", fee, slip, config.strategy.initial_capital
    )
    terminal = trades[-1].capital_after if trades else config.strategy.initial_capital
    return terminal / config.strategy.initial_capital - 1, len(trades)


def _read_csv(relative: Path) -> list[dict[str, str]]:
    with paths.require_file(paths.resolve(relative)).open(
        encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle, strict=True))


def _existing_evidence_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stage8 = ACCEPTED_BACKTEST_MANIFEST.parent
    for source in _read_csv(stage8 / "development_strategy_metrics.csv"):
        if source["strategy_name"] not in {
            "all_shock",
            "weekend",
            "low_activity",
            "combined",
        }:
            continue
        rows.append(
            _record(
                "cost",
                source["cost_scenario"],
                "strategy",
                strategy=source["strategy_name"],
                metric="net_cumulative_return",
                estimate=float(source["net_cumulative_return"]),
                n=int(source["trade_count"]),
                trades=int(source["trade_count"]),
                status="verified_stage8_evidence",
                note=(stage8 / "development_strategy_metrics.csv").as_posix(),
            )
        )
    stage7 = ACCEPTED_INFERENCE_MANIFEST.parent
    for source in _read_csv(stage7 / "development_model_results.csv"):
        if source["hypothesis"] not in {
            "H1_model0",
            "H2_model1",
            "H3_model2",
            "H4_model2",
        }:
            continue
        rows.append(
            _record(
                "inference",
                "hc3",
                "inference",
                model=source["model"],
                term=source["term"],
                metric="coefficient",
                estimate=float(source["estimate"]),
                low=float(source["ci_95_low"]),
                high=float(source["ci_95_high"]),
                n=int(source["n"]),
                status="verified_stage7_evidence",
                note=(stage7 / "development_model_results.csv").as_posix(),
            )
        )
    for source in _read_csv(stage7 / "development_bootstrap_results.csv"):
        if (source["model"], source["term"]) not in {
            ("model0", "intercept"),
            ("model1", "weekend"),
            ("model2", "weekend"),
            ("model2", "low_activity_score"),
        }:
            continue
        rows.append(
            _record(
                "inference",
                "month_block_bootstrap",
                "inference",
                model=source["model"],
                term=source["term"],
                metric="percentile_interval",
                low=float(source["percentile_95_low"]),
                high=float(source["percentile_95_high"]),
                n=int(source["sample_n"]),
                status="verified_stage7_evidence",
                note=(stage7 / "development_bootstrap_results.csv").as_posix(),
            )
        )
    for source in _read_csv(stage7 / "development_model_results.csv"):
        if (
            source["hypothesis"] not in {"H5", "H5_diagnostic"}
            or source["term"] != "positive"
        ):
            continue
        rows.append(
            _record(
                "positive_shock",
                "2.5",
                "inference",
                model=source["model"],
                term="positive",
                metric="coefficient",
                estimate=float(source["estimate"]),
                low=float(source["ci_95_low"]),
                high=float(source["ci_95_high"]),
                n=int(source["n"]),
                status="verified_stage7_evidence",
                note=(stage7 / "development_model_results.csv").as_posix(),
            )
        )
    return rows


def held_out_evidence_rows(run_dir: Path) -> list[dict[str, object]]:
    """Turn the accepted primary held-out outputs into robustness reference rows."""
    rows: list[dict[str, object]] = []
    metrics = run_dir / "held_out_strategy_metrics.csv"
    for source in _read_csv(metrics):
        if source["strategy_name"] not in {
            "all_shock",
            "weekend",
            "low_activity",
            "combined",
        }:
            continue
        rows.append(
            _record(
                "cost",
                source["cost_scenario"],
                "strategy",
                strategy=source["strategy_name"],
                metric="net_cumulative_return",
                estimate=float(source["net_cumulative_return"]),
                n=int(source["trade_count"]),
                trades=int(source["trade_count"]),
                status="verified_held_out_primary_evidence",
                note=metrics.as_posix(),
            )
        )
    models = run_dir / "held_out_model_results.csv"
    for source in _read_csv(models):
        if source["hypothesis"] in {
            "H1_model0",
            "H2_model1",
            "H3_model2",
            "H4_model2",
        }:
            rows.append(
                _record(
                    "inference",
                    "hc3",
                    "inference",
                    model=source["model"],
                    term=source["term"],
                    metric="coefficient",
                    estimate=float(source["estimate"]),
                    low=float(source["ci_95_low"]),
                    high=float(source["ci_95_high"]),
                    n=int(source["n"]),
                    status="verified_held_out_primary_evidence",
                    note=models.as_posix(),
                )
            )
        if source["hypothesis"] in {"H5", "H5_diagnostic"} and source[
            "term"
        ] == "positive":
            rows.append(
                _record(
                    "positive_shock",
                    "2.5",
                    "inference",
                    model=source["model"],
                    term="positive",
                    metric="coefficient",
                    estimate=float(source["estimate"]),
                    low=float(source["ci_95_low"]),
                    high=float(source["ci_95_high"]),
                    n=int(source["n"]),
                    status="verified_held_out_primary_evidence",
                    note=models.as_posix(),
                )
            )
    bootstraps = run_dir / "held_out_bootstrap_results.csv"
    for source in _read_csv(bootstraps):
        if (source["model"], source["term"]) not in {
            ("model0", "intercept"),
            ("model1", "weekend"),
            ("model2", "weekend"),
            ("model2", "low_activity_score"),
        }:
            continue
        rows.append(
            _record(
                "inference",
                "month_block_bootstrap",
                "inference",
                model=source["model"],
                term=source["term"],
                metric="percentile_interval",
                low=float(source["percentile_95_low"]),
                high=float(source["percentile_95_high"]),
                n=int(source["sample_n"]),
                status="verified_held_out_primary_evidence",
                note=bootstraps.as_posix(),
            )
        )
    return rows


def _stability_figure(
    year_rows: Sequence[dict[str, object]],
    *,
    analysis_period: str = "development",
    label: str = "Development",
) -> tuple[str, bytes]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    years = sorted({int(row["variant"]) for row in year_rows})
    means = [
        next(
            float(row["estimate"])
            for row in year_rows
            if int(row["variant"]) == year and row["model"] == "model0"
        )
        for year in years
    ]
    returns = [
        next(
            float(row["estimate"])
            for row in year_rows
            if int(row["variant"]) == year and row["strategy"] == "weekend"
        )
        for year in years
    ]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].bar(years, means, color="#335c99")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(ylabel="12h log return", title=f"{label} calendar-year stability")
    axes[1].bar(years, returns, color="#c43c39")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Signal year (UTC)", ylabel="Weekend net return")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
        metadata={"Software": f"matplotlib {matplotlib.__version__}"},
    )
    plt.close(figure)
    return f"{analysis_period}_calendar_year_stability.png", buffer.getvalue()


def build_robustness_bundle(
    primary: Sequence[EventRecord],
    rows: Sequence[Hour],
    config: ConfigNode,
    *,
    analysis_period: str = "development",
    label: str = "Development",
    evidence_rows: Sequence[dict[str, object]] | None = None,
) -> RobustnessBundle:
    if analysis_period not in {"development", "held_out"}:
        raise RobustnessError("robustness split must be development or held_out")
    market = backtest_hours(rows, analysis_period=analysis_period)
    results: list[dict[str, object]] = []
    reconstructed = rebuild_events(
        rows,
        config.shock.negative_threshold,
        config.events.cooldown_hours,
        config.shock.positive_threshold,
        analysis_period=analysis_period,
    )
    if {e.event_id for e in reconstructed} != {e.event_id for e in primary}:
        raise RobustnessError(
            "primary event reconstruction does not match accepted Stage 6"
        )

    for threshold in config.robustness.thresholds:
        events = rebuild_events(
            rows,
            threshold,
            config.events.cooldown_hours,
            config.robustness.positive_comparison_threshold,
            analysis_period=analysis_period,
        )
        results.extend(inference_rows("threshold", str(threshold), events, config))
        results.extend(
            strategy_rows(
                "threshold",
                str(threshold),
                events,
                market,
                config,
                analysis_period=analysis_period,
            )
        )
    for horizon in config.robustness.horizons:
        events = with_horizon(
            primary, rows, horizon, analysis_period=analysis_period
        )
        cfg = variant_config(config, 1, horizon, horizon)
        results.extend(inference_rows("horizon", str(horizon), events, config))
        results.extend(
            strategy_rows(
                "horizon",
                str(horizon),
                events,
                market,
                cfg,
                analysis_period=analysis_period,
            )
        )
    for measure, events in (
        ("quote_volume", primary),
        (
            "trade_count",
            with_trade_activity(primary, rows, analysis_period=analysis_period),
        ),
    ):
        results.extend(inference_rows("activity", measure, events, config))
        results.extend(
            strategy_rows(
                "activity",
                measure,
                events,
                market,
                config,
                analysis_period=analysis_period,
            )
        )
    for name, cfg in (
        ("next_open", variant_config(config, 1, 12, 12)),
        ("delayed_one_hour", variant_config(config, 2, 13, 12)),
    ):
        results.extend(
            strategy_rows(
                "execution",
                name,
                primary,
                market,
                cfg,
                analysis_period=analysis_period,
            )
        )
    for cooldown in (
        config.events.cooldown_hours,
        *config.robustness.diagnostic_cooldown_hours,
    ):
        events = rebuild_events(
            rows,
            config.shock.negative_threshold,
            cooldown,
            config.robustness.positive_comparison_threshold,
            analysis_period=analysis_period,
        )
        results.extend(inference_rows("cooldown", str(cooldown), events, config))
        results.extend(
            strategy_rows(
                "cooldown",
                str(cooldown),
                events,
                market,
                config,
                analysis_period=analysis_period,
            )
        )
    results.extend(
        _existing_evidence_rows() if evidence_rows is None else evidence_rows
    )

    year_rows: list[dict[str, object]] = []
    for year in sorted({e.calendar_year for e in primary}):
        subset = tuple(e for e in primary if e.calendar_year == year)
        year_rows.extend(inference_rows("calendar_year", str(year), subset, config))
        weekend, trades = _weekend_return(
            subset, market, config, analysis_period=analysis_period
        )
        year_rows.append(
            _record(
                "calendar_year",
                year,
                "strategy",
                strategy="weekend",
                metric="primary_net_cumulative_return",
                estimate=weekend,
                n=sum(e.sign == "negative" for e in subset),
                trades=trades,
                status="descriptive_year_slice",
            )
        )
    results.extend(year_rows)

    negative_ids = [e.event_id for e in primary if e.sign == "negative"]
    base_values = _core_values(primary, config)
    base_weekend, _ = _weekend_return(
        primary, market, config, analysis_period=analysis_period
    )
    for family, keys in (
        ("leave_one_event_out", negative_ids),
        (
            "leave_one_year_out",
            sorted({str(e.calendar_year) for e in primary if e.sign == "negative"}),
        ),
    ):
        collections: dict[str, list[float]] = {
            key: [] for key in (*base_values, "weekend_net_return")
        }
        for key in keys:
            subset = tuple(
                e
                for e in primary
                if (
                    e.event_id != key
                    if family == "leave_one_event_out"
                    else str(e.calendar_year) != key
                )
            )
            values = _core_values(subset, config)
            weekend, _ = _weekend_return(
                subset, market, config, analysis_period=analysis_period
            )
            for metric, value in {**values, "weekend_net_return": weekend}.items():
                if value is not None:
                    collections[metric].append(float(value))
        for metric, values in collections.items():
            primary_value = (
                base_weekend if metric == "weekend_net_return" else base_values[metric]
            )
            model, _, term = metric.partition(":")
            results.append(
                _record(
                    family,
                    "range",
                    "strategy" if metric == "weekend_net_return" else "inference",
                    strategy="weekend" if metric == "weekend_net_return" else "",
                    model="" if metric == "weekend_net_return" else model,
                    term=term,
                    metric=metric,
                    estimate=primary_value,
                    low=min(values),
                    high=max(values),
                    n=len(keys),
                    status="complete_fixed_primary_ids",
                )
            )

    weekend = [
        e for e in primary if e.sign == "negative" and e.eligible_h1_h2 and e.weekend
    ]
    for boundary_label, crossing in (
        ("weekend_contained", False),
        ("monday_crossing", True),
    ):
        subset = tuple(e for e in weekend if e.crosses_into_monday_12h is crossing)
        values = [float(e.outcome_12h) for e in subset if e.outcome_12h is not None]
        results.append(
            _record(
                "weekend_boundary",
                boundary_label,
                "event",
                metric="mean_post_log_return_12h",
                estimate=float(np.mean(values)) if values else None,
                n=len(values),
                status="descriptive",
            )
        )
        net, trades = _weekend_return(
            subset, market, config, analysis_period=analysis_period
        )
        results.append(
            _record(
                "weekend_boundary",
                boundary_label,
                "strategy",
                strategy="weekend",
                metric="primary_net_cumulative_return",
                estimate=net,
                n=len(subset),
                trades=trades,
                status="descriptive",
            )
        )
    return RobustnessBundle(
        tuple(results),
        _stability_figure(
            year_rows, analysis_period=analysis_period, label=label
        ),
    )


def _format(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RobustnessError("cannot serialise non-finite result")
        return format(value, ".17g")
    return str(value)


def render_results(rows: Sequence[Mapping[str, object]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RESULT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(RESULT_COLUMNS):
            raise RobustnessError("result row violates schema")
        writer.writerow({key: _format(row[key]) for key in RESULT_COLUMNS})
    return buffer.getvalue()


def validate_result_coverage(
    rows: Sequence[Mapping[str, object]],
    config: ConfigNode,
    *,
    analysis_period: str = "development",
) -> None:
    variants = {}
    for row in rows:
        variants.setdefault(str(row["family"]), set()).add(str(row["variant"]))
    split = getattr(config.splits, analysis_period)
    start = split.start_inclusive
    end = split.end_exclusive
    expected_years = {str(year) for year in range(start.year, end.year + 1)}
    if end.month == 1 and end.day == 1 and end.hour == 0:
        expected_years.discard(str(end.year))
    expected = {
        "threshold": {str(value) for value in config.robustness.thresholds},
        "horizon": {str(value) for value in config.robustness.horizons},
        "activity": set(config.robustness.activity_measures),
        "execution": set(config.robustness.execution),
        "cost": set(config.robustness.cost_scenarios),
        "inference": set(config.robustness.inference),
        "positive_shock": {str(config.robustness.positive_comparison_threshold)},
        "cooldown": {
            str(config.events.cooldown_hours),
            *(str(value) for value in config.robustness.diagnostic_cooldown_hours),
        },
        "calendar_year": expected_years,
        "leave_one_event_out": {"range"},
        "leave_one_year_out": {"range"},
        "weekend_boundary": {"weekend_contained", "monday_crossing"},
    }
    if variants != expected:
        raise RobustnessError("robustness family or variant coverage is incomplete")
    keys = [
        (
            row["family"],
            row["variant"],
            row["analysis"],
            row["strategy"],
            row["model"],
            row["term"],
            row["metric"],
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RobustnessError("robustness results contain duplicate identities")


def validate_primary_replay(
    rows: Sequence[Mapping[str, object]], config: ConfigNode
) -> None:
    canonical_strategy = {
        str(row["strategy"]): float(row["estimate"])
        for row in rows
        if row["family"] == "cost" and row["variant"] == "primary"
    }
    canonical_inference = {
        (str(row["model"]), str(row["term"])): float(row["estimate"])
        for row in rows
        if row["family"] == "inference" and row["variant"] == "hc3"
    }
    primary_variants = {
        "threshold": str(config.shock.negative_threshold),
        "horizon": str(config.outcomes.primary_horizon_hours),
        "activity": "quote_volume",
        "execution": "next_open",
        "cooldown": str(config.events.cooldown_hours),
    }
    for family, variant in primary_variants.items():
        for row in rows:
            if row["family"] != family or row["variant"] != variant:
                continue
            reference = (
                canonical_strategy.get(str(row["strategy"]))
                if row["analysis"] == "strategy"
                else canonical_inference.get((str(row["model"]), str(row["term"])))
            )
            estimate = row["estimate"]
            if (
                reference is None
                or estimate is None
                or not math.isclose(
                    float(estimate),
                    reference,
                    rel_tol=config.reproducibility.rtol,
                    abs_tol=config.reproducibility.atol,
                )
            ):
                raise RobustnessError(
                    f"primary {family} replay differs from accepted evidence"
                )


def _render_inventory(
    entries: Sequence[tuple[Path, str, int, str, Sequence[str]]]
) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=INVENTORY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for path, digest, count, kind, schema in entries:
        writer.writerow(
            dict(
                artifact_path=path.as_posix(),
                sha256=digest,
                record_count=count,
                artifact_type=kind,
                column_schema_json=json.dumps(list(schema), separators=(",", ":")),
            )
        )
    return buffer.getvalue()


def verify_inputs(
    config: ConfigNode, backtest_manifest: Path, *, frozen: bool
) -> tuple[Path, int, Path, dict[str, str]]:
    if frozen and backtest_manifest != ACCEPTED_BACKTEST_MANIFEST:
        raise RobustnessError("default Stage 9 requires accepted Stage 8")
    verified = verify_backtest_inventory(backtest_manifest)
    digest = sha256_file(paths.resolve(backtest_manifest))
    if frozen and digest != ACCEPTED_BACKTEST_SHA256:
        raise RobustnessError("accepted Stage 8 identity changed")
    manifest = read_manifest(backtest_manifest)
    if (
        manifest["code_identity"]["git_dirty"] is not False
        or manifest["exclusions"].get("held_out_rows_analysed") != 0
    ):
        raise RobustnessError("Stage 8 evidence is not clean and isolated")
    if not matches_active_protocol(manifest, config):
        raise RobustnessError(
            "Stage 8 evidence does not match active specification/config"
        )
    feature, count, feature_checksums = verify_feature_input(
        config, ACCEPTED_FEATURE_MANIFEST, require_frozen_acceptance=frozen
    )
    event = ACCEPTED_OUTCOME_MANIFEST.parent / "development_event_outcomes.csv"
    # The accepted backtest run already records this event file's checksum.
    expected = manifest["input_checksums"].get(event.as_posix())
    if expected is None or sha256_file(paths.resolve(event)) != expected:
        raise RobustnessError("accepted event input changed")
    checksums = dict(feature_checksums)
    checksums.update(verified)
    for path in (
        backtest_manifest,
        Path(str(backtest_manifest) + ".sha256"),
        backtest_manifest.parent / "development_backtest_inventory.csv",
        event,
    ):
        checksums[path.as_posix()] = sha256_file(paths.resolve(path))
    return feature, count, event, dict(sorted(checksums.items()))


def verify_inventory(
    manifest_relative: Path, config: ConfigNode | None = None
) -> dict[str, str]:
    verify_manifest_checksum(manifest_relative)
    manifest = read_manifest(manifest_relative)
    if (
        manifest["stage"] != STAGE
        or manifest["mode"] != MODE
        or manifest["column_schema"] != list(INVENTORY_COLUMNS)
    ):
        raise RobustnessError("invalid Stage 9 manifest")
    inventory = manifest_relative.parent / "development_robustness_inventory.csv"
    if sha256_file(paths.resolve(inventory)) != manifest["output_checksum"]:
        raise RobustnessError("inventory checksum failed")
    rows = _read_csv(inventory)
    if len(rows) != 2 or manifest["row_count"] != 2:
        raise RobustnessError("inventory count failed")
    expected = {
        "development_robustness_results.csv",
        "development_calendar_year_stability.png",
    }
    if {Path(row["artifact_path"]).name for row in rows} != expected:
        raise RobustnessError("inventory artifacts failed")
    verified = {}
    for row in rows:
        relative = Path(row["artifact_path"])
        if relative.parent != manifest_relative.parent:
            raise RobustnessError("artifact escapes run directory")
        digest = sha256_file(paths.resolve(relative))
        if digest != row["sha256"]:
            raise RobustnessError("artifact checksum failed")
        if relative.suffix == ".csv":
            records = _read_csv(relative)
            if tuple(records[0]) != RESULT_COLUMNS or len(records) != int(
                row["record_count"]
            ):
                raise RobustnessError("results schema/count failed")
            validate_result_coverage(records, config or load_config())
        elif not paths.resolve(relative).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise RobustnessError("stability figure is not PNG")
        verified[relative.as_posix()] = digest
    return dict(sorted(verified.items()))


def accepted_primary_held_out_manifest() -> Path:
    """Locate the accepted primary evaluation without a prose dependency."""
    relative = ACCEPTED_PRIMARY_HELD_OUT_MANIFEST
    verify_manifest_checksum(relative)
    if sha256_file(paths.resolve(relative)) != ACCEPTED_PRIMARY_HELD_OUT_SHA256:
        raise RobustnessError("accepted primary held-out manifest identity changed")
    manifest = read_manifest(relative)
    if manifest["stage"] != "11. Held-out evaluation" or manifest["mode"] != "held-out":
        raise RobustnessError("accepted primary held-out manifest has the wrong stage")
    return relative


def ensure_no_completed_held_out_robustness_run(config: ConfigNode) -> None:
    """Every completed run blocks another saved held-out execution."""
    root = paths.resolve(config.paths.outputs_runs)
    if any(root.glob(f"*/{HELD_OUT_MANIFEST_SLUG}.manifest.json")):
        raise RobustnessError(
            "a completed held-out robustness run already exists; rerun refused"
        )


def verify_held_out_robustness_inputs(
    config: ConfigNode, primary_manifest: Path
) -> tuple[Path, int, Path, dict[str, str]]:
    """Verify the accepted primary result and its original pre-access freeze."""
    from src.heldout import verify_held_out_inventory

    primary_artifacts = verify_held_out_inventory(primary_manifest)
    payload = read_manifest(primary_manifest)
    if (
        payload["code_identity"]["git_dirty"] is not False
        or payload["exclusions"].get("development_outcome_rows_analysed") != 0
    ):
        raise RobustnessError("primary held-out evidence is not clean and isolated")
    if not matches_active_protocol(payload, config):
        raise RobustnessError(
            "primary held-out evidence does not match active specification/config"
        )
    freeze_paths = [
        Path(name)
        for name in payload["input_checksums"]
        if name.endswith("/held-out-freeze.manifest.json")
    ]
    if len(freeze_paths) != 1:
        raise RobustnessError("primary held-out evidence has no unique freeze")
    _verify_original_freeze_after_access(freeze_paths[0])
    feature, count, feature_checksums = verify_feature_input(
        config, ACCEPTED_FEATURE_MANIFEST, require_frozen_acceptance=True
    )
    for name, digest in feature_checksums.items():
        if payload["input_checksums"].get(name) != digest:
            raise RobustnessError(
                f"primary held-out evidence does not bind required input {name}"
            )
    event = primary_manifest.parent / "held_out_event_outcomes.csv"
    required = {
        event,
        primary_manifest.parent / "held_out_model_results.csv",
        primary_manifest.parent / "held_out_bootstrap_results.csv",
        primary_manifest.parent / "held_out_strategy_metrics.csv",
    }
    if any(path.as_posix() not in primary_artifacts for path in required):
        raise RobustnessError("primary held-out evidence lacks a robustness input")
    checksums = {**feature_checksums, **primary_artifacts}
    for path in (
        primary_manifest,
        Path(f"{primary_manifest}.sha256"),
        primary_manifest.parent / "held_out_inventory.csv",
        *freeze_paths,
        Path(f"{freeze_paths[0]}.sha256"),
    ):
        checksums[path.as_posix()] = sha256_file(paths.resolve(path))
    return feature, count, event, dict(sorted(checksums.items()))


def _verify_original_freeze_after_access(manifest_relative: Path) -> None:
    """Check the original freeze, allowing the decision log to have changed since."""
    verify_manifest_checksum(manifest_relative)
    manifest = read_manifest(manifest_relative)
    if manifest["stage"] != "10. Held-out freeze" or manifest["mode"] != "freeze":
        raise RobustnessError("primary run does not reference Stage 10 freeze evidence")
    record_path = manifest_relative.parent / "held_out_freeze_record.json"
    if sha256_file(paths.resolve(record_path)) != manifest["output_checksum"]:
        raise RobustnessError("original freeze record checksum failed")
    try:
        record = json.loads(paths.resolve(record_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RobustnessError(f"cannot read original freeze record: {error}") from None
    if (
        record.get("status") != "COMPLETE"
        or record.get("held_out_outcomes_accessed") is not False
        or record.get("decision_log_sha256")
        != manifest["input_checksums"].get("docs/DECISION_LOG.md")
    ):
        raise RobustnessError("original freeze isolation record is invalid")
    for name, digest in manifest["input_checksums"].items():
        if name == "docs/DECISION_LOG.md":
            continue
        if sha256_file(paths.resolve(name)) != digest:
            raise RobustnessError(f"frozen input changed after primary run: {name}")


def verify_held_out_robustness_inventory(
    manifest_relative: Path, config: ConfigNode | None = None
) -> dict[str, str]:
    verify_manifest_checksum(manifest_relative)
    manifest = read_manifest(manifest_relative)
    if (
        manifest["stage"] != HELD_OUT_STAGE
        or manifest["mode"] != HELD_OUT_MODE
        or manifest["column_schema"] != list(INVENTORY_COLUMNS)
    ):
        raise RobustnessError("invalid held-out robustness manifest")
    inventory = manifest_relative.parent / HELD_OUT_INVENTORY_NAME
    if sha256_file(paths.resolve(inventory)) != manifest["output_checksum"]:
        raise RobustnessError("held-out robustness inventory checksum failed")
    rows = _read_csv(inventory)
    expected = {HELD_OUT_RESULT_NAME, HELD_OUT_FIGURE_NAME}
    if (
        not rows
        or tuple(rows[0]) != INVENTORY_COLUMNS
        or len(rows) != 2
        or manifest["row_count"] != 2
        or {Path(row["artifact_path"]).name for row in rows} != expected
    ):
        raise RobustnessError("held-out robustness inventory is incomplete")
    verified: dict[str, str] = {}
    for row in rows:
        relative = Path(row["artifact_path"])
        if relative.parent != manifest_relative.parent:
            raise RobustnessError("held-out robustness artifact escapes run directory")
        digest = sha256_file(paths.resolve(relative))
        if digest != row["sha256"]:
            raise RobustnessError("held-out robustness artifact checksum failed")
        if relative.name == HELD_OUT_RESULT_NAME:
            records = _read_csv(relative)
            if (
                row["artifact_type"] != "csv"
                or json.loads(row["column_schema_json"]) != list(RESULT_COLUMNS)
                or not records
                or tuple(records[0]) != RESULT_COLUMNS
                or len(records) != int(row["record_count"])
            ):
                raise RobustnessError("held-out robustness result schema/count failed")
            validate_result_coverage(
                records, config or load_config(), analysis_period="held_out"
            )
        elif (
            row["artifact_type"] != "png"
            or json.loads(row["column_schema_json"]) != ["binary_png"]
            or int(row["record_count"]) != 1
            or not paths.resolve(relative).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        ):
            raise RobustnessError("held-out stability figure is invalid")
        verified[relative.as_posix()] = digest
    return dict(sorted(verified.items()))


def run_held_out_robustness(
    config: ConfigNode | None = None, *, primary_manifest: Path | None = None
) -> HeldOutRobustnessResult:
    from src.pipeline import require_held_out_robustness_authorisation

    require_held_out_robustness_authorisation()
    cfg = config or load_config()
    ensure_no_completed_held_out_robustness_run(cfg)
    selected = primary_manifest or accepted_primary_held_out_manifest()
    draft = build_manifest(HELD_OUT_STAGE, HELD_OUT_MODE, cfg)
    if draft.code_identity["git_dirty"] is not False:
        raise RobustnessError(
            "held-out robustness completion requires a clean committed repository"
        )
    verified_inputs = verify_held_out_robustness_inputs(cfg, selected)
    log = create_run_log(draft.run_id, config=cfg)
    log.record(
        "held_out_robustness_started",
        reason="complete_pre_specified_output_omitted_from_primary_run",
        primary_manifest=selected.as_posix(),
        held_out_primary_results_already_accessed=True,
    )
    try:
        feature, count, event_path, checksums = verified_inputs
        source = read_hours(feature, count, analysis_period="held_out")
        events = read_event_records(event_path, cfg)
        evidence = held_out_evidence_rows(selected.parent)
        bundle = build_robustness_bundle(
            events,
            source,
            cfg,
            analysis_period="held_out",
            label="Held-out",
            evidence_rows=evidence,
        )
        validate_result_coverage(bundle.rows, cfg, analysis_period="held_out")
        validate_primary_replay(bundle.rows, cfg)
        run_dir = run_relative_dir(draft.run_id, cfg)
        result_path = run_dir / HELD_OUT_RESULT_NAME
        content = render_results(bundle.rows)
        create_exclusive(result_path, content)
        result_digest = sha256_bytes(content.encode("utf-8"))
        figure_path = run_dir / HELD_OUT_FIGURE_NAME
        if bundle.figure[0] != HELD_OUT_FIGURE_NAME:
            raise RobustnessError("held-out robustness figure has the wrong name")
        create_exclusive_bytes(figure_path, bundle.figure[1])
        figure_digest = sha256_bytes(bundle.figure[1])
        inventory_path = run_dir / HELD_OUT_INVENTORY_NAME
        inventory = _render_inventory(
            (
                (result_path, result_digest, len(bundle.rows), "csv", RESULT_COLUMNS),
                (figure_path, figure_digest, 1, "png", ("binary_png",)),
            )
        )
        create_exclusive(inventory_path, inventory)
        inventory_digest = sha256_bytes(inventory.encode("utf-8"))
        finished = replace(
            draft,
            input_checksums=checksums,
            exclusions={"development_outcome_rows_analysed": 0},
            eligibility_changes={
                "robustness_result_records": len(bundle.rows),
                "robustness_families": 12,
                "output_artifacts": 2,
            },
        ).completed(inventory_digest, 2, list(INVENTORY_COLUMNS))
        manifest = finished.write_for_run(HELD_OUT_MANIFEST_SLUG, cfg)
        manifest_relative = manifest.relative_to(paths.project_root().resolve())
        sidecar, digest = write_manifest_checksum(manifest_relative)
        verify_held_out_robustness_inventory(manifest_relative, cfg)
        log.record("manifest_checksum_written", path=sidecar.as_posix(), sha256=digest)
        log.record(
            "held_out_robustness_completed",
            result_rows=len(bundle.rows),
            held_out_primary_results_already_accessed=True,
        )
    except Exception as error:
        try:
            log.record(
                "held_out_robustness_failed",
                reason=type(error).__name__,
                detail=str(error)[:500],
                held_out_primary_results_already_accessed=True,
            )
        except Exception:  # pragma: no cover
            pass
        raise
    return HeldOutRobustnessResult(
        draft.run_id,
        manifest_relative,
        sidecar,
        log.relative_path,
        inventory_path,
        len(bundle.rows),
    )


def run_development_robustness(
    config: ConfigNode | None = None, *, backtest_manifest: Path | None = None
):
    cfg = config or load_config()
    selected = backtest_manifest or ACCEPTED_BACKTEST_MANIFEST
    frozen = selected == ACCEPTED_BACKTEST_MANIFEST
    draft = build_manifest(STAGE, MODE, cfg)
    if frozen and draft.code_identity["git_dirty"] is not False:
        raise RobustnessError(
            "development robustness requires a clean committed repository"
        )
    log = create_run_log(draft.run_id, config=cfg)
    log.record("development_robustness_started", held_out_outcomes_accessed=False)
    try:
        feature, count, event_path, checksums = verify_inputs(
            cfg, selected, frozen=frozen
        )
        source = read_hours(feature, count)
        events = read_event_records(event_path, cfg)
        bundle = build_robustness_bundle(events, source, cfg)
        validate_result_coverage(bundle.rows, cfg)
        validate_primary_replay(bundle.rows, cfg)
        run_dir = run_relative_dir(draft.run_id, cfg)
        result_path = run_dir / "development_robustness_results.csv"
        content = render_results(bundle.rows)
        create_exclusive(result_path, content)
        result_digest = sha256_bytes(content.encode())
        figure_path = run_dir / bundle.figure[0]
        create_exclusive_bytes(figure_path, bundle.figure[1])
        figure_digest = sha256_bytes(bundle.figure[1])
        inventory_path = run_dir / "development_robustness_inventory.csv"
        inventory = _render_inventory(
            (
                (result_path, result_digest, len(bundle.rows), "csv", RESULT_COLUMNS),
                (figure_path, figure_digest, 1, "png", ("binary_png",)),
            )
        )
        create_exclusive(inventory_path, inventory)
        inventory_digest = sha256_bytes(inventory.encode())
        finished = replace(
            draft,
            input_checksums=checksums,
            exclusions={"held_out_rows_analysed": 0},
            eligibility_changes={
                "robustness_result_records": len(bundle.rows),
                "robustness_families": 12,
                "output_artifacts": 2,
            },
        ).completed(inventory_digest, 2, list(INVENTORY_COLUMNS))
        manifest = finished.write_for_run("development-robustness", cfg)
        manifest_relative = manifest.relative_to(paths.project_root().resolve())
        sidecar, digest = write_manifest_checksum(manifest_relative)
        verify_inventory(manifest_relative, cfg)
        log.record("manifest_checksum_written", path=sidecar.as_posix(), sha256=digest)
        log.record(
            "development_robustness_completed",
            result_rows=len(bundle.rows),
            held_out_outcomes_accessed=False,
        )
    except Exception as error:
        try:
            log.record(
                "development_robustness_failed",
                reason=type(error).__name__,
                detail=str(error)[:500],
                held_out_outcomes_accessed=False,
            )
        except Exception:
            pass
        raise
    return DevelopmentRobustnessResult(
        draft.run_id,
        manifest_relative,
        sidecar,
        log.relative_path,
        inventory_path,
        len(bundle.rows),
    )


def summarise(result: DevelopmentRobustnessResult) -> list[str]:
    return [
        f"run id                  {result.run_id}",
        f"robustness result rows   {result.result_rows}",
        f"inventory               {result.inventory_relative_path}",
        f"manifest                {result.manifest_relative_path}",
        f"manifest checksum       {result.manifest_checksum_relative_path}",
        "held-out outcomes       not accessed",
    ]


def summarise_held_out(result: HeldOutRobustnessResult) -> list[str]:
    return [
        f"run id                  {result.run_id}",
        f"robustness result rows   {result.result_rows}",
        f"inventory               {result.inventory_relative_path}",
        f"manifest                {result.manifest_relative_path}",
        f"manifest checksum       {result.manifest_checksum_relative_path}",
        "development outcomes    not analysed",
    ]
