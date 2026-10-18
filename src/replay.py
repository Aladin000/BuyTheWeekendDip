"""Rebuild each saved pipeline stage in memory and compare it with the saved files.

Validation starts from the raw archives; every later stage starts from the
checked output of the stage before it. Each rebuilt file is hashed and
compared with the saved one. Nothing is written to disk, so a replay cannot
create a new run or change a saved result.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Callable, Mapping, Sequence

from src import backtest, inference, paths
from src.artifacts import sha256_bytes, sha256_file
from src.backtest import (
    ACCEPTED_INFERENCE_MANIFEST,
    BENCHMARK_COLUMNS,
    DAILY_COLUMNS,
    EQUITY_COLUMNS,
    METRIC_COLUMNS,
    TRADE_AUDIT_COLUMNS,
    TRADE_RESULT_COLUMNS,
    build_backtest_bundle,
    read_analysis_hours,
    read_development_hours,
    verify_backtest_inputs,
)
from src.config import ConfigNode
from src.count_audit import (
    ACCEPTED_FEATURE_MANIFEST,
    build_count_records,
    read_count_rows,
    render_count_records,
    verify_feature_input,
)
from src.features import (
    compute_feature_set,
    compute_log_returns,
    lagged_decimal_sums,
    lagged_percentile_numerators,
    lagged_rolling_mean_std,
    lagged_rv,
    read_clean_rows,
    render_feature_rows,
    verify_validation_input,
)
from src.freeze import (
    ACCEPTED_ACQUISITION_MANIFEST,
    ACCEPTED_ROBUSTNESS_MANIFEST,
    ACCEPTED_VALIDATION_MANIFEST,
)
from src.inference import (
    ACCEPTED_OUTCOME_MANIFEST,
    DESCRIPTIVE_COLUMNS,
    DIAGNOSTIC_COLUMNS,
    GROUP_COMPARISON_COLUMNS,
    MODEL_RESULT_COLUMNS,
    PATH_SUMMARY_COLUMNS,
    DriftRecord,
    PathRecord,
    build_any_sign_event_records,
    build_inference_bundle,
    event_records_from_outcomes,
    read_any_sign_candidates,
    read_drift_records,
    read_event_records,
    read_path_records,
    reconcile_stage6_records,
    render_bootstraps,
    verify_inference_inputs,
)
from src.outcomes import (
    ACCEPTED_COUNT_MANIFEST,
    build_outcomes,
    read_outcome_rows,
    reconcile_with_gate3,
    render_drift_reference,
    render_event_outcomes,
    render_paths,
    verify_count_input,
    verify_development_inputs,
)
from src.robustness import (
    ACCEPTED_BACKTEST_MANIFEST,
    build_robustness_bundle,
    held_out_evidence_rows,
    read_hours,
    render_results,
    validate_primary_replay,
    validate_result_coverage,
    verify_held_out_robustness_inputs,
)
from src.robustness import verify_inputs as verify_robustness_inputs
from src.validation import (
    REVIEW_RELATIVE_PATH,
    assign_segments_and_gaps,
    build_anomaly_candidates,
    load_review_ledger,
    month_targets,
    parse_month,
    render_anomaly_candidates,
    render_anomaly_review,
    render_clean_rows,
    render_exclusions,
    render_gaps,
    render_reconciliation,
    render_schema_observations,
    validate_review_ledger,
    verify_acquisition_inputs,
)

ACCEPTED_HELD_OUT_MANIFEST = Path(
    "outputs/runs/20260910T133543.269946Z-held-out-dad36cad-632a5e/"
    "held-out-evaluation.manifest.json"
)
ACCEPTED_HELD_OUT_ROBUSTNESS_MANIFEST = Path(
    "outputs/runs/20260910T144404.535764Z-held-out-robustness-de2ac148-51637c/"
    "held-out-robustness.manifest.json"
)


class ReplayError(RuntimeError):
    """Raised when a rebuilt stage cannot be compared with its saved files."""


@dataclass(frozen=True)
class ArtifactComparison:
    name: str
    accepted_sha256: str
    replayed_sha256: str

    @property
    def matches(self) -> bool:
        return self.accepted_sha256 == self.replayed_sha256


@dataclass(frozen=True)
class StageReplay:
    stage: str
    accepted_run: str
    seconds: float
    comparisons: tuple[ArtifactComparison, ...]

    @property
    def matches(self) -> bool:
        return bool(self.comparisons) and all(item.matches for item in self.comparisons)


def _compare(
    stage: str,
    run_dir: Path,
    outputs: Mapping[str, str | bytes],
    seconds: float,
) -> StageReplay:
    comparisons = []
    for name, content in outputs.items():
        accepted = paths.resolve(run_dir / name)
        if not accepted.is_file():
            raise ReplayError(f"{stage}: accepted artifact {name} is missing")
        payload = content.encode("utf-8") if isinstance(content, str) else content
        comparisons.append(
            ArtifactComparison(name, sha256_file(accepted), sha256_bytes(payload))
        )
    return StageReplay(stage, run_dir.name, seconds, tuple(comparisons))


def _timed(function: Callable[[], Mapping[str, str | bytes]]) -> tuple[Mapping[str, str | bytes], float]:
    start = time.perf_counter()
    outputs = function()
    return outputs, time.perf_counter() - start


def replay_validation(config: ConfigNode) -> StageReplay:
    """Rebuild the cleaning outputs from the raw monthly archives."""

    def build() -> dict[str, str]:
        verify_acquisition_inputs(config, ACCEPTED_ACQUISITION_MANIFEST)
        rows, audits, exclusions = [], [], []
        headers: set[tuple[str, ...]] = set()
        for target in month_targets(config):
            month_rows, audit, month_exclusions = parse_month(config, target)
            rows.extend(month_rows)
            audits.append(audit)
            exclusions.extend(month_exclusions)
            if audit.header_present:
                headers.add(audit.header_tokens)
        if len(headers) > 1:
            raise ReplayError("source archives contain inconsistent header token sets")
        annotated, gaps = assign_segments_and_gaps(
            rows,
            config.coverage.raw_start_inclusive,
            config.coverage.raw_end_exclusive,
        )
        missing_by_period: dict[str, int] = {}
        for gap in gaps:
            missing_by_period[gap.source_period_expected] = (
                missing_by_period.get(gap.source_period_expected, 0) + 1
            )
        audits = [
            replace(audit, missing_hours=missing_by_period.get(audit.target.period, 0))
            for audit in audits
        ]
        if any(
            audit.accepted_complete_rows + audit.missing_hours != audit.expected_hours
            for audit in audits
        ):
            raise ReplayError("expected-hour reconciliation failed")
        candidates = build_anomaly_candidates(annotated, config)
        candidate_content = render_anomaly_candidates(candidates)
        recorded_digest, decisions = load_review_ledger(REVIEW_RELATIVE_PATH)
        validate_review_ledger(candidates, candidate_content, recorded_digest, decisions)
        outputs = {
            "schema_observations.csv": render_schema_observations(audits),
            "file_reconciliation.csv": render_reconciliation(audits),
            "gaps.csv": render_gaps(gaps),
            "excluded_rows.csv": render_exclusions(exclusions),
            "anomaly_candidates.csv": candidate_content,
            "anomaly_review.csv": render_anomaly_review(candidates, decisions),
            "clean_hourly.csv": render_clean_rows(annotated, candidates, decisions),
        }
        summary = {
            "status": "complete",
            "source_files": len(audits),
            "parsed_data_rows": sum(audit.parsed_data_rows for audit in audits),
            "accepted_complete_rows": len(annotated),
            "incomplete_rows_excluded": sum(
                audit.incomplete_rows_excluded for audit in audits
            ),
            "nonstandard_close_rows_excluded": len(exclusions),
            "duplicate_rows": sum(audit.duplicate_rows for audit in audits),
            "invalid_rows": sum(audit.invalid_rows for audit in audits),
            "missing_hours": len(gaps),
            "contiguous_segments": max(row.contiguous_segment_id for row in annotated),
            "anomaly_candidates": len(candidates),
            "schema_verified_in_config": config.source.schema_verified,
            "manual_reviews_complete": True,
            "artifact_sha256": dict(
                sorted(
                    (name, sha256_bytes(content.encode("utf-8")))
                    for name, content in outputs.items()
                )
            ),
        }
        outputs["data_quality_summary.json"] = (
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return outputs

    outputs, seconds = _timed(build)
    return _compare(
        "Cleaning",
        ACCEPTED_VALIDATION_MANIFEST.parent,
        outputs,
        seconds,
    )


def replay_features(config: ConfigNode) -> StageReplay:
    """Rebuild the feature table from the saved clean rows."""

    def build() -> dict[str, str]:
        clean_relative, expected_rows, _ = verify_validation_input(
            config, ACCEPTED_VALIDATION_MANIFEST
        )
        rows = read_clean_rows(clean_relative, expected_rows)
        feature_set = compute_feature_set(rows, config)
        return {"pre_outcome_features.csv": render_feature_rows(rows, feature_set, config)}

    outputs, seconds = _timed(build)
    return _compare(
        "Features", ACCEPTED_FEATURE_MANIFEST.parent, outputs, seconds
    )


def replay_count_audit(config: ConfigNode) -> StageReplay:
    def build() -> dict[str, str]:
        feature_relative, expected_rows, _ = verify_feature_input(
            config, ACCEPTED_FEATURE_MANIFEST, require_frozen_acceptance=True
        )
        rows = read_count_rows(
            feature_relative, expected_rows, cooldown_hours=config.events.cooldown_hours
        )
        return {"feasibility_counts.csv": render_count_records(build_count_records(rows, config))}

    outputs, seconds = _timed(build)
    return _compare("Count check", ACCEPTED_COUNT_MANIFEST.parent, outputs, seconds)


def replay_development_outcomes(config: ConfigNode) -> StageReplay:
    def build() -> dict[str, str]:
        feature_relative, expected_rows, gate3, _ = verify_development_inputs(
            config,
            ACCEPTED_FEATURE_MANIFEST,
            ACCEPTED_COUNT_MANIFEST,
            require_frozen_acceptance=True,
        )
        rows = read_outcome_rows(feature_relative, expected_rows, config)
        bundle = build_outcomes(rows, config, analysis_period="development")
        reconcile_with_gate3(bundle, gate3)
        return {
            "development_event_outcomes.csv": render_event_outcomes(bundle.events),
            "development_event_paths_24h.csv": render_paths(bundle.paths),
            "development_drift_reference.csv": render_drift_reference(bundle.drift_reference),
        }

    outputs, seconds = _timed(build)
    return _compare(
        "Development outcomes", ACCEPTED_OUTCOME_MANIFEST.parent, outputs, seconds
    )


def _inference_outputs(prefix: str, bundle) -> dict[str, str | bytes]:
    outputs: dict[str, str | bytes] = {
        f"{prefix}_descriptives.csv": inference.render_records(
            bundle.descriptives, DESCRIPTIVE_COLUMNS
        ),
        f"{prefix}_group_comparisons.csv": inference.render_records(
            bundle.group_comparisons, GROUP_COMPARISON_COLUMNS
        ),
        f"{prefix}_average_path.csv": inference.render_records(
            bundle.path_summary, PATH_SUMMARY_COLUMNS
        ),
        f"{prefix}_model_results.csv": inference.render_records(
            bundle.model_results, MODEL_RESULT_COLUMNS
        ),
        f"{prefix}_bootstrap_results.csv": render_bootstraps(bundle.bootstrap_results),
        f"{prefix}_diagnostics.csv": inference.render_records(
            bundle.diagnostics, DIAGNOSTIC_COLUMNS
        ),
    }
    outputs.update(dict(bundle.figures))
    return outputs


def _backtest_outputs(prefix: str, bundle) -> dict[str, str | bytes]:
    outputs: dict[str, str | bytes] = {
        f"{prefix}_trade_audit.csv": backtest.render_records(
            bundle.audit_rows, TRADE_AUDIT_COLUMNS
        ),
        f"{prefix}_trade_results.csv": backtest.render_records(
            bundle.trade_rows, TRADE_RESULT_COLUMNS
        ),
        f"{prefix}_strategy_metrics.csv": backtest.render_records(
            bundle.metric_rows, METRIC_COLUMNS
        ),
        f"{prefix}_equity_hourly.csv": backtest.render_records(
            bundle.equity_rows, EQUITY_COLUMNS
        ),
        f"{prefix}_equity_daily.csv": backtest.render_records(
            bundle.daily_rows, DAILY_COLUMNS
        ),
        f"{prefix}_benchmarks.csv": backtest.render_records(
            bundle.benchmark_rows, BENCHMARK_COLUMNS
        ),
    }
    outputs.update(dict(bundle.figures))
    return outputs


def replay_development_inference(config: ConfigNode) -> StageReplay:
    def build() -> dict[str, str | bytes]:
        artifact_paths, feature_relative, expected_rows, _ = verify_inference_inputs(
            config,
            ACCEPTED_OUTCOME_MANIFEST,
            ACCEPTED_FEATURE_MANIFEST,
            require_frozen_acceptance=True,
        )
        events = read_event_records(artifact_paths["development_event_outcomes.csv"], config)
        path_records = read_path_records(artifact_paths["development_event_paths_24h.csv"])
        drift = read_drift_records(artifact_paths["development_drift_reference.csv"])
        reconcile_stage6_records(events, path_records, drift, config)
        feature_rows = read_outcome_rows(feature_relative, expected_rows, config)
        candidates = read_any_sign_candidates(feature_relative)
        any_sign = build_any_sign_event_records(feature_rows, candidates, config)
        bundle = build_inference_bundle(
            events, path_records, drift, any_sign, feature_rows, config
        )
        return _inference_outputs("development", bundle)

    outputs, seconds = _timed(build)
    return _compare(
        "Development models", ACCEPTED_INFERENCE_MANIFEST.parent, outputs, seconds
    )


def replay_development_backtest(config: ConfigNode) -> StageReplay:
    def build() -> dict[str, str | bytes]:
        event_relative, feature_relative, feature_rows, _ = verify_backtest_inputs(
            config, ACCEPTED_INFERENCE_MANIFEST, require_frozen_acceptance=True
        )
        events = read_event_records(event_relative, config)
        hours = read_development_hours(feature_relative, feature_rows, config)
        return _backtest_outputs("development", build_backtest_bundle(events, hours, config))

    outputs, seconds = _timed(build)
    return _compare(
        "Development backtest", ACCEPTED_BACKTEST_MANIFEST.parent, outputs, seconds
    )


def replay_development_robustness(config: ConfigNode) -> StageReplay:
    def build() -> dict[str, str | bytes]:
        feature, count, event_path, _ = verify_robustness_inputs(
            config, ACCEPTED_BACKTEST_MANIFEST, frozen=True
        )
        source = read_hours(feature, count)
        events = read_event_records(event_path, config)
        bundle = build_robustness_bundle(events, source, config)
        validate_result_coverage(bundle.rows, config)
        validate_primary_replay(bundle.rows, config)
        return {
            "development_robustness_results.csv": render_results(bundle.rows),
            bundle.figure[0]: bundle.figure[1],
        }

    outputs, seconds = _timed(build)
    return _compare(
        "Development robustness", ACCEPTED_ROBUSTNESS_MANIFEST.parent, outputs, seconds
    )


def replay_held_out(config: ConfigNode) -> StageReplay:
    """Rebuild the held-out outcomes, models, and backtest, which were saved as one run."""

    def build() -> dict[str, str | bytes]:
        feature_path, feature_count, _ = verify_feature_input(
            config, ACCEPTED_FEATURE_MANIFEST, require_frozen_acceptance=True
        )
        _, expectations, _ = verify_count_input(
            config,
            ACCEPTED_COUNT_MANIFEST,
            require_frozen_acceptance=True,
            analysis_period="held_out",
        )
        rows = read_outcome_rows(feature_path, feature_count, config, analysis_period="held_out")
        outcome_bundle = build_outcomes(rows, config, analysis_period="held_out")
        reconcile_with_gate3(outcome_bundle, expectations)
        events = event_records_from_outcomes(outcome_bundle.events)
        path_records = tuple(
            PathRecord(row.event_id, row.signal_close_time, row.step, row.value)
            for row in outcome_bundle.paths
        )
        drift = tuple(
            DriftRecord(
                f"reference-{row.row.close_time_raw}", row.row.calendar_month, row.value_12h
            )
            for row in outcome_bundle.drift_reference
        )
        candidates = read_any_sign_candidates(feature_path, analysis_period="held_out")
        any_sign = build_any_sign_event_records(
            rows, candidates, config, analysis_period="held_out"
        )
        inference_bundle = build_inference_bundle(
            events,
            path_records,
            drift,
            any_sign,
            rows,
            config,
            analysis_period="held_out",
            label="Held-out",
        )
        hours = read_analysis_hours(
            feature_path, feature_count, config, analysis_period="held_out"
        )
        backtest_bundle = build_backtest_bundle(
            events, hours, config, analysis_period="held_out", label="Held-out"
        )
        outputs: dict[str, str | bytes] = {
            "held_out_event_outcomes.csv": render_event_outcomes(outcome_bundle.events),
            "held_out_event_paths_24h.csv": render_paths(outcome_bundle.paths),
            "held_out_drift_reference.csv": render_drift_reference(
                outcome_bundle.drift_reference
            ),
        }
        outputs.update(_inference_outputs("held_out", inference_bundle))
        outputs.update(_backtest_outputs("held_out", backtest_bundle))
        return outputs

    outputs, seconds = _timed(build)
    return _compare(
        "Held-out evaluation", ACCEPTED_HELD_OUT_MANIFEST.parent, outputs, seconds
    )


def replay_held_out_robustness(config: ConfigNode) -> StageReplay:
    def build() -> dict[str, str | bytes]:
        feature, count, event_path, _ = verify_held_out_robustness_inputs(
            config, ACCEPTED_HELD_OUT_MANIFEST
        )
        source = read_hours(feature, count, analysis_period="held_out")
        events = read_event_records(event_path, config)
        bundle = build_robustness_bundle(
            events,
            source,
            config,
            analysis_period="held_out",
            label="Held-out",
            evidence_rows=held_out_evidence_rows(ACCEPTED_HELD_OUT_MANIFEST.parent),
        )
        validate_result_coverage(bundle.rows, config, analysis_period="held_out")
        validate_primary_replay(bundle.rows, config)
        return {
            "held_out_robustness_results.csv": render_results(bundle.rows),
            bundle.figure[0]: bundle.figure[1],
        }

    outputs, seconds = _timed(build)
    return _compare(
        "Held-out robustness",
        ACCEPTED_HELD_OUT_ROBUSTNESS_MANIFEST.parent,
        outputs,
        seconds,
    )


STAGE_REPLAYS: tuple[Callable[[ConfigNode], StageReplay], ...] = (
    replay_validation,
    replay_features,
    replay_count_audit,
    replay_development_outcomes,
    replay_development_inference,
    replay_development_backtest,
    replay_development_robustness,
    replay_held_out,
    replay_held_out_robustness,
)


def naive_percentile_numerators(
    values: Sequence[Decimal | None], segment_ids: Sequence[int], *, window: int
) -> tuple[int | None, ...]:
    """Rank each value by recounting the previous ``window`` values in its segment."""
    numerators: list[int | None] = [None] * len(values)
    start = 0
    for index, value in enumerate(values):
        if index == 0 or segment_ids[index] != segment_ids[index - 1]:
            start = index
        if value is not None and index - start >= window:
            numerators[index] = sum(item <= value for item in values[index - window : index])
        if value is None:
            start = index + 1
    return tuple(numerators)


@dataclass(frozen=True)
class FeatureBenchmark:
    rows: int
    percentile_rows: int
    reference_window: int
    timings: tuple[dict[str, object], ...]


def benchmark_feature_steps(config: ConfigNode) -> FeatureBenchmark:
    """Time each rolling feature on the saved clean rows.

    The naive recount is only a speed and correctness reference; it must give
    exactly the same ranks as the sorted window.
    """
    clean_relative, expected_rows, _ = verify_validation_input(
        config, ACCEPTED_VALIDATION_MANIFEST
    )
    rows = read_clean_rows(clean_relative, expected_rows)
    segments = [row.segment_id for row in rows]
    window = config.activity.reference_window
    timings: list[dict[str, object]] = []

    def timed(step: str, complexity: str, function):
        start = time.perf_counter()
        value = function()
        timings.append(
            {
                "step": step,
                "implemented_complexity": complexity,
                "seconds": time.perf_counter() - start,
            }
        )
        return value

    log_returns, _ = timed(
        "log returns", "O(n)", lambda: compute_log_returns([row.close for row in rows], segments)
    )
    timed(
        "720-hour lagged mean and standard deviation",
        "O(n)",
        lambda: lagged_rolling_mean_std(
            log_returns, segments, window=config.shock.window_hours, ddof=config.shock.ddof
        ),
    )
    activity = timed(
        "24-hour lagged quote-volume sum",
        "O(n)",
        lambda: lagged_decimal_sums(
            [row.quote_volume for row in rows], segments, window=config.activity.window_hours
        ),
    )
    implemented = timed(
        "2,160-value activity percentile (sorted window)",
        "O(n log w) comparisons plus O(n w) list shifts",
        lambda: lagged_percentile_numerators(activity, segments, window=window),
    )[0]
    timed(
        "24-hour lagged RV",
        "O(n)",
        lambda: lagged_rv(log_returns, segments, window=config.controls.rv_window_hours),
    )

    naive_result = timed(
        "2,160-value activity percentile (naive recount reference)",
        "O(n w) comparisons",
        lambda: naive_percentile_numerators(activity, segments, window=window),
    )
    if implemented != naive_result:
        raise ReplayError("naive and implemented activity percentiles disagree")
    return FeatureBenchmark(
        rows=len(rows),
        percentile_rows=sum(value is not None for value in implemented),
        reference_window=window,
        timings=tuple(timings),
    )


MODEL_BOOTSTRAP_TERMS: tuple[tuple[str, str], ...] = (
    ("model0", "intercept"),
    ("model1", "weekend"),
    ("model2", "low_activity_score"),
    ("model2", "weekend"),
    ("model3", "weekend_x_low_activity_score"),
    ("positive_model0", "intercept"),
    ("positive_controlled", "weekend"),
    ("positive_controlled", "low_activity_score"),
    ("h5_sign", "positive"),
    ("h5_any_sign", "positive"),
)


@dataclass(frozen=True)
class BootstrapBenchmark:
    summaries: int
    refits: int
    seconds: float
    largest_sample: int
    largest_terms: int


def benchmark_bootstrap(config: ConfigNode) -> BootstrapBenchmark:
    """Time the ten development model bootstraps.

    Each bootstrap refits its regression on 10,000 resampled sets of months.
    The rebuilt percentile intervals must equal the saved ones exactly.
    """
    artifact_paths, feature_relative, expected_rows, _ = verify_inference_inputs(
        config,
        ACCEPTED_OUTCOME_MANIFEST,
        ACCEPTED_FEATURE_MANIFEST,
        require_frozen_acceptance=True,
    )
    events = read_event_records(artifact_paths["development_event_outcomes.csv"], config)
    feature_rows = read_outcome_rows(feature_relative, expected_rows, config)
    candidates = read_any_sign_candidates(feature_relative)
    any_sign = build_any_sign_event_records(feature_rows, candidates, config)
    models = inference.build_primary_models(events)
    models["h5_any_sign"] = inference.build_model_data(
        "h5_any_sign",
        "pooled_any_sign_cooldown_complete_case",
        inference._eligible_model_rows(any_sign, "h5_any_sign"),
    )
    start = time.perf_counter()
    results = [
        inference.month_block_bootstrap(
            models[model], term, config, analysis="model_inference"
        )
        for model, term in MODEL_BOOTSTRAP_TERMS
    ]
    seconds = time.perf_counter() - start
    saved = paths.resolve(
        ACCEPTED_INFERENCE_MANIFEST.parent / "development_bootstrap_results.csv"
    ).read_text(encoding="utf-8").splitlines()
    rebuilt = render_bootstraps(results).splitlines()
    if rebuilt[0] != saved[0] or not set(rebuilt[1:]) <= set(saved[1:]):
        raise ReplayError("rebuilt bootstrap intervals differ from the saved run")
    used = [models[model] for model, _ in MODEL_BOOTSTRAP_TERMS]
    return BootstrapBenchmark(
        summaries=len(results),
        refits=sum(result.attempted for result in results),
        seconds=seconds,
        largest_sample=max(len(data.rows) for data in used),
        largest_terms=max(len(data.terms) for data in used),
    )
