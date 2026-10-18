"""Run the held-out outcomes, inference, and backtest once, behind guards."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from src import backtest, inference, paths
from src.acquisition import verify_manifest_checksum, write_manifest_checksum
from src.artifacts import (
    build_manifest,
    create_exclusive,
    create_exclusive_bytes,
    create_run_log,
    read_manifest,
    run_relative_dir,
    sha256_bytes,
    sha256_file,
)
from src.backtest import (
    BENCHMARK_COLUMNS,
    DAILY_COLUMNS,
    EQUITY_COLUMNS,
    METRIC_COLUMNS,
    TRADE_AUDIT_COLUMNS,
    TRADE_RESULT_COLUMNS,
    build_backtest_bundle,
    read_analysis_hours,
)
from src.config import ConfigNode, load_config
from src.count_audit import ACCEPTED_FEATURE_MANIFEST, verify_feature_input
from src.freeze import ACCEPTED_FREEZE_MANIFEST, ACCEPTED_FREEZE_SHA256, verify_freeze
from src.inference import (
    BOOTSTRAP_COLUMNS,
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
    render_bootstraps,
)
from src.outcomes import (
    ACCEPTED_COUNT_MANIFEST,
    DRIFT_OUTPUT_COLUMNS,
    EVENT_OUTPUT_COLUMNS,
    Gate3Expectations,
    PATH_OUTPUT_COLUMNS,
    build_outcomes,
    read_outcome_rows,
    reconcile_with_gate3,
    render_drift_reference,
    render_event_outcomes,
    render_paths,
    verify_count_input,
)

STAGE = "11. Held-out evaluation"
MODE = "held-out"
ANALYSIS_PERIOD = "held_out"
MANIFEST_SLUG = "held-out-evaluation"
INVENTORY_NAME = "held_out_inventory.csv"
INVENTORY_COLUMNS = (
    "artifact_path",
    "sha256",
    "record_count",
    "artifact_type",
    "column_schema_json",
)

TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "held_out_event_outcomes.csv": EVENT_OUTPUT_COLUMNS,
    "held_out_event_paths_24h.csv": PATH_OUTPUT_COLUMNS,
    "held_out_drift_reference.csv": DRIFT_OUTPUT_COLUMNS,
    "held_out_descriptives.csv": DESCRIPTIVE_COLUMNS,
    "held_out_group_comparisons.csv": GROUP_COMPARISON_COLUMNS,
    "held_out_average_path.csv": PATH_SUMMARY_COLUMNS,
    "held_out_model_results.csv": MODEL_RESULT_COLUMNS,
    "held_out_bootstrap_results.csv": BOOTSTRAP_COLUMNS,
    "held_out_diagnostics.csv": DIAGNOSTIC_COLUMNS,
    "held_out_trade_audit.csv": TRADE_AUDIT_COLUMNS,
    "held_out_trade_results.csv": TRADE_RESULT_COLUMNS,
    "held_out_strategy_metrics.csv": METRIC_COLUMNS,
    "held_out_equity_hourly.csv": EQUITY_COLUMNS,
    "held_out_equity_daily.csv": DAILY_COLUMNS,
    "held_out_benchmarks.csv": BENCHMARK_COLUMNS,
}
FIGURE_NAMES = {
    "held_out_price_events.png",
    "held_out_average_path.png",
    "held_out_weekend_distribution.png",
    "held_out_activity_distribution.png",
    "held_out_equity_curves.png",
    "held_out_drawdowns.png",
}


class HeldOutError(RuntimeError):
    """Raised when the single held-out run cannot go ahead."""


@dataclass(frozen=True)
class HeldOutResult:
    run_id: str
    manifest_relative_path: Path
    manifest_checksum_relative_path: Path
    inventory_relative_path: Path
    event_count: int
    trade_counts: Mapping[str, int]


def accepted_freeze_manifest() -> Path:
    """Locate the accepted freeze by its recorded file identity."""
    relative = ACCEPTED_FREEZE_MANIFEST
    verify_manifest_checksum(relative)
    if sha256_file(paths.resolve(relative)) != ACCEPTED_FREEZE_SHA256:
        raise HeldOutError("accepted freeze manifest identity changed")
    manifest = read_manifest(relative)
    if manifest["stage"] != "10. Held-out freeze" or manifest["mode"] != "freeze":
        raise HeldOutError("accepted freeze manifest has the wrong stage")
    return relative


def ensure_no_completed_held_out_run(config: ConfigNode) -> None:
    """Stop with an error if a held-out run has already completed."""
    root = paths.resolve(config.paths.outputs_runs)
    completed = sorted(root.glob("*/held-out-evaluation.manifest.json"))
    if completed:
        raise HeldOutError(
            "a completed held-out evaluation already exists; an accidental rerun is refused"
        )


def verify_held_out_inputs(
    config: ConfigNode, freeze_manifest: Path
) -> tuple[Path, int, Path, Gate3Expectations, dict[str, str]]:
    """Check the freeze and the locked inputs before any price is read."""
    verify_freeze(freeze_manifest, require_current_identity=True)
    feature, rows, feature_checksums = verify_feature_input(
        config, ACCEPTED_FEATURE_MANIFEST, require_frozen_acceptance=True
    )
    count_path, expectations, count_checksums = verify_count_input(
        config,
        ACCEPTED_COUNT_MANIFEST,
        require_frozen_acceptance=True,
        analysis_period=ANALYSIS_PERIOD,
    )
    freeze_payload = read_manifest(freeze_manifest)
    required = {**feature_checksums, **count_checksums}
    for name, digest in required.items():
        if freeze_payload["input_checksums"].get(name) != digest:
            raise HeldOutError(f"freeze does not bind required held-out input {name}")
    sidecar = freeze_manifest.parent / f"{freeze_manifest.name}.sha256"
    checksums = {
        **required,
        freeze_manifest.as_posix(): sha256_file(paths.resolve(freeze_manifest)),
        sidecar.as_posix(): sha256_file(paths.resolve(sidecar)),
    }
    return feature, rows, count_path, expectations, dict(sorted(checksums.items()))


def _render_inventory(
    artifacts: Sequence[tuple[Path, str, int, str, Sequence[str]]],
) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=INVENTORY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for path, digest, count, kind, schema in artifacts:
        writer.writerow(
            {
                "artifact_path": path.as_posix(),
                "sha256": digest,
                "record_count": count,
                "artifact_type": kind,
                "column_schema_json": json.dumps(list(schema), separators=(",", ":")),
            }
        )
    return buffer.getvalue()


def verify_held_out_inventory(manifest_relative: Path) -> dict[str, str]:
    from src.acquisition import verify_manifest_checksum

    verify_manifest_checksum(manifest_relative)
    manifest = read_manifest(manifest_relative)
    if (
        manifest["stage"] != STAGE
        or manifest["mode"] != MODE
        or manifest["column_schema"] != list(INVENTORY_COLUMNS)
    ):
        raise HeldOutError("invalid held-out manifest")
    inventory = manifest_relative.parent / INVENTORY_NAME
    if sha256_file(paths.resolve(inventory)) != manifest["output_checksum"]:
        raise HeldOutError("held-out inventory checksum failed")
    try:
        with paths.resolve(inventory).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != INVENTORY_COLUMNS:
                raise HeldOutError("held-out inventory schema is invalid")
            records = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise HeldOutError(f"cannot read held-out inventory: {error}") from None
    expected = set(TABLE_SCHEMAS) | FIGURE_NAMES
    if (
        len(records) != len(expected)
        or manifest["row_count"] != len(expected)
        or {Path(row["artifact_path"]).name for row in records} != expected
    ):
        raise HeldOutError(
            "held-out inventory does not name every required artifact once"
        )
    verified: dict[str, str] = {}
    for row in records:
        relative = Path(row["artifact_path"])
        if relative.parent != manifest_relative.parent:
            raise HeldOutError("held-out artifact escapes its run directory")
        digest = sha256_file(paths.resolve(relative))
        if digest != row["sha256"]:
            raise HeldOutError(f"held-out artifact checksum failed for {relative.name}")
        try:
            count = int(row["record_count"])
            schema = json.loads(row["column_schema_json"])
        except (ValueError, TypeError, json.JSONDecodeError):
            raise HeldOutError("held-out inventory metadata is malformed") from None
        if relative.name in TABLE_SCHEMAS:
            if row["artifact_type"] != "csv" or schema != list(
                TABLE_SCHEMAS[relative.name]
            ):
                raise HeldOutError(
                    f"held-out table metadata failed for {relative.name}"
                )
            with paths.resolve(relative).open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                reader = csv.reader(handle, strict=True)
                header = next(reader, None)
                actual = sum(1 for _ in reader)
            if header != schema or actual != count:
                raise HeldOutError(f"held-out table count failed for {relative.name}")
        elif (
            row["artifact_type"] != "png"
            or schema != ["binary_png"]
            or count != 1
            or not paths.resolve(relative).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        ):
            raise HeldOutError(f"held-out figure metadata failed for {relative.name}")
        verified[relative.as_posix()] = digest
    return dict(sorted(verified.items()))


def run_held_out(config: ConfigNode | None = None) -> HeldOutResult:
    from src.pipeline import require_held_out_authorisation

    require_held_out_authorisation()
    cfg = config or load_config()
    ensure_no_completed_held_out_run(cfg)
    draft = build_manifest(STAGE, MODE, cfg)
    if draft.code_identity["git_dirty"] is not False:
        raise HeldOutError("held-out evaluation requires a clean committed repository")
    freeze_manifest = accepted_freeze_manifest()
    verified_inputs = verify_held_out_inputs(cfg, freeze_manifest)
    log = create_run_log(draft.run_id, config=cfg)
    log.record(
        "held_out_evaluation_started",
        reason="primary_pre_authorised_historical_evaluation",
        freeze_manifest=freeze_manifest.as_posix(),
    )
    try:
        feature_path, feature_count, _, expectations, input_checksums = verified_inputs
        feature_rows = read_outcome_rows(
            feature_path,
            feature_count,
            cfg,
            analysis_period=ANALYSIS_PERIOD,
        )
        outcome_bundle = build_outcomes(
            feature_rows, cfg, analysis_period=ANALYSIS_PERIOD
        )
        reconcile_with_gate3(outcome_bundle, expectations)
        events = event_records_from_outcomes(outcome_bundle.events)
        path_records = tuple(
            PathRecord(row.event_id, row.signal_close_time, row.step, row.value)
            for row in outcome_bundle.paths
        )
        drift = tuple(
            DriftRecord(
                f"reference-{row.row.close_time_raw}",
                row.row.calendar_month,
                row.value_12h,
            )
            for row in outcome_bundle.drift_reference
        )
        candidates = read_any_sign_candidates(
            feature_path, analysis_period=ANALYSIS_PERIOD
        )
        any_sign_events = build_any_sign_event_records(
            feature_rows,
            candidates,
            cfg,
            analysis_period=ANALYSIS_PERIOD,
        )
        inference_bundle = build_inference_bundle(
            events,
            path_records,
            drift,
            any_sign_events,
            feature_rows,
            cfg,
            analysis_period=ANALYSIS_PERIOD,
            label="Held-out",
        )
        hours = read_analysis_hours(
            feature_path,
            feature_count,
            cfg,
            analysis_period=ANALYSIS_PERIOD,
        )
        backtest_bundle = build_backtest_bundle(
            events,
            hours,
            cfg,
            analysis_period=ANALYSIS_PERIOD,
            label="Held-out",
        )

        tables: tuple[tuple[str, str, int], ...] = (
            (
                "held_out_event_outcomes.csv",
                render_event_outcomes(outcome_bundle.events),
                len(outcome_bundle.events),
            ),
            (
                "held_out_event_paths_24h.csv",
                render_paths(outcome_bundle.paths),
                len(outcome_bundle.paths),
            ),
            (
                "held_out_drift_reference.csv",
                render_drift_reference(outcome_bundle.drift_reference),
                len(outcome_bundle.drift_reference),
            ),
            (
                "held_out_descriptives.csv",
                inference.render_records(
                    inference_bundle.descriptives, DESCRIPTIVE_COLUMNS
                ),
                len(inference_bundle.descriptives),
            ),
            (
                "held_out_group_comparisons.csv",
                inference.render_records(
                    inference_bundle.group_comparisons, GROUP_COMPARISON_COLUMNS
                ),
                len(inference_bundle.group_comparisons),
            ),
            (
                "held_out_average_path.csv",
                inference.render_records(
                    inference_bundle.path_summary, PATH_SUMMARY_COLUMNS
                ),
                len(inference_bundle.path_summary),
            ),
            (
                "held_out_model_results.csv",
                inference.render_records(
                    inference_bundle.model_results, MODEL_RESULT_COLUMNS
                ),
                len(inference_bundle.model_results),
            ),
            (
                "held_out_bootstrap_results.csv",
                render_bootstraps(inference_bundle.bootstrap_results),
                len(inference_bundle.bootstrap_results),
            ),
            (
                "held_out_diagnostics.csv",
                inference.render_records(
                    inference_bundle.diagnostics, DIAGNOSTIC_COLUMNS
                ),
                len(inference_bundle.diagnostics),
            ),
            (
                "held_out_trade_audit.csv",
                backtest.render_records(
                    backtest_bundle.audit_rows, TRADE_AUDIT_COLUMNS
                ),
                len(backtest_bundle.audit_rows),
            ),
            (
                "held_out_trade_results.csv",
                backtest.render_records(
                    backtest_bundle.trade_rows, TRADE_RESULT_COLUMNS
                ),
                len(backtest_bundle.trade_rows),
            ),
            (
                "held_out_strategy_metrics.csv",
                backtest.render_records(backtest_bundle.metric_rows, METRIC_COLUMNS),
                len(backtest_bundle.metric_rows),
            ),
            (
                "held_out_equity_hourly.csv",
                backtest.render_records(backtest_bundle.equity_rows, EQUITY_COLUMNS),
                len(backtest_bundle.equity_rows),
            ),
            (
                "held_out_equity_daily.csv",
                backtest.render_records(backtest_bundle.daily_rows, DAILY_COLUMNS),
                len(backtest_bundle.daily_rows),
            ),
            (
                "held_out_benchmarks.csv",
                backtest.render_records(
                    backtest_bundle.benchmark_rows, BENCHMARK_COLUMNS
                ),
                len(backtest_bundle.benchmark_rows),
            ),
        )
        run_dir = run_relative_dir(draft.run_id, cfg)
        inventory_rows: list[tuple[Path, str, int, str, Sequence[str]]] = []
        for name, content, count in tables:
            relative = run_dir / name
            create_exclusive(relative, content)
            digest = sha256_bytes(content.encode("utf-8"))
            inventory_rows.append((relative, digest, count, "csv", TABLE_SCHEMAS[name]))
            log.record(
                "held_out_table_written",
                path=relative.as_posix(),
                rows=count,
                sha256=digest,
            )
        for name, payload in (*inference_bundle.figures, *backtest_bundle.figures):
            relative = run_dir / name
            create_exclusive_bytes(relative, payload)
            digest = sha256_bytes(payload)
            inventory_rows.append((relative, digest, 1, "png", ("binary_png",)))
            log.record(
                "held_out_figure_written", path=relative.as_posix(), sha256=digest
            )
        inventory_relative = run_dir / INVENTORY_NAME
        inventory_content = _render_inventory(inventory_rows)
        create_exclusive(inventory_relative, inventory_content)
        inventory_digest = sha256_bytes(inventory_content.encode("utf-8"))
        finished = replace(
            draft,
            input_checksums=input_checksums,
            exclusions={"development_outcome_rows_analysed": 0},
            eligibility_changes={
                "held_out_event_rows": len(outcome_bundle.events),
                "held_out_path_rows": len(outcome_bundle.paths),
                "held_out_drift_rows": len(outcome_bundle.drift_reference),
                "h5_any_sign_event_rows": len(any_sign_events),
                "output_artifacts": len(inventory_rows),
            },
        ).completed(inventory_digest, len(inventory_rows), list(INVENTORY_COLUMNS))
        manifest_path = finished.write_for_run(MANIFEST_SLUG, cfg)
        manifest_relative = manifest_path.relative_to(paths.project_root().resolve())
        sidecar, _ = write_manifest_checksum(manifest_relative)
        verify_held_out_inventory(manifest_relative)
        log.record(
            "held_out_evaluation_completed",
            manifest=manifest_relative.as_posix(),
            event_count=len(outcome_bundle.events),
        )
        return HeldOutResult(
            draft.run_id,
            manifest_relative,
            sidecar,
            inventory_relative,
            len(outcome_bundle.events),
            backtest_bundle.executed_trade_counts,
        )
    except Exception as error:
        try:
            log.record(
                "held_out_evaluation_failed",
                reason=type(error).__name__,
                detail=str(error)[:500],
            )
        except Exception:  # pragma: no cover
            pass
        raise


def summarise(result: HeldOutResult) -> tuple[str, ...]:
    return (
        f"run id                  {result.run_id}",
        f"manifest                {result.manifest_relative_path.as_posix()}",
        f"held-out events         {result.event_count}",
        f"primary trades          {dict(result.trade_counts)}",
    )
