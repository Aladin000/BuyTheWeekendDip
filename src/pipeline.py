"""Execution-mode entry point.

Each mode runs one stage: ``config-check`` prints the active identities,
``acquire`` downloads and verifies the monthly archives, ``validate`` runs the
data checks, ``features`` builds past-only fields, ``count-only`` counts events
without outcomes, ``outcomes`` builds development outcomes, ``development``
runs inference, ``backtest`` runs portfolio accounting, and ``robustness`` runs
the development robustness checks (2020-2023 only). The submitted historical evaluation is
complete, so new freeze records and saved held-out execution stay disabled; the notebook replays
its calculations in memory without creating a new run.

This module holds no analysis; it only orders stages and checks.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src import (
    acquisition,
    artifacts,
    backtest,
    count_audit,
    features,
    freeze,
    heldout,
    inference,
    outcomes,
    paths,
    robustness,
    validation,
)
from src.acquisition import AcquisitionError
from src.artifacts import ArtifactError
from src.backtest import BacktestError
from src.config import load_config
from src.count_audit import CountAuditError
from src.events import EventError
from src.features import FeatureError
from src.freeze import FreezeError
from src.heldout import HeldOutError
from src.inference import InferenceError
from src.outcomes import OutcomeError
from src.robustness import RobustnessError
from src.validation import ValidationError

# Stage that owns each mode.
_MODE_STAGE: dict[str, str] = {
    "config-check": "1. Environment and configuration",
    "acquire": "2. Acquisition",
    "validate": "3. Validation and cleaning",
    "features": "4. Pre-outcome features",
    "count-only": "5. Count-only audit",
    "outcomes": "6. Development outcomes",
    "development": "7. Development inference",
    "backtest": "8. Development backtest",
    "robustness": "9. Development robustness",
    "freeze": "10. Held-out freeze",
    "held-out": "11. Held-out evaluation",
    "held-out-robustness": "11. Held-out robustness completion",
    "report": "12. Notebook and report",
}


class ModeNotImplementedError(RuntimeError):
    """Raised when a mode belongs to a stage that has not been built yet."""


class AuthorisationError(RuntimeError):
    """Raised when the held-out mode is invoked without authorisation."""


def held_out_authorised() -> bool:
    """Saved held-out evaluations are closed in the completed submission."""
    return False


def require_held_out_authorisation() -> None:
    if not held_out_authorised():
        raise AuthorisationError(
            "Held-out evaluation refused: the historical evaluation is complete. "
            "Use the notebook for deterministic in-memory replay."
        )


def held_out_robustness_authorised() -> bool:
    """Saved held-out robustness completion is closed in this submission."""
    return False


def require_held_out_robustness_authorisation() -> None:
    if not held_out_robustness_authorised():
        raise AuthorisationError(
            "Held-out robustness completion refused: the historical evaluation is complete. "
            "Use the notebook for deterministic in-memory replay."
        )


def run_config_check() -> int:
    """Load the configuration and print its identities.

    Read-only: there is no run to record, so no manifest or run log is written.
    """
    config = load_config()
    spec = artifacts.spec_identity(config)
    conf = artifacts.config_identity(config)
    code = artifacts.code_identity()
    environment = artifacts.environment_identity()

    print(f"project                 {config.meta.project}")
    print(f"specification version   {spec.version}")
    print(f"specification sha256    {spec.sha256}")
    print(f"configuration version   {conf.version}")
    print(f"configuration sha256    {conf.sha256}")
    print(f"code source digest      {code.source_digest}")
    print(f"  code (src)            {code.code_digest}")
    print(f"  tests and fixtures    {code.test_digest}")
    print(f"  configuration         {code.config_digest}")
    print(f"  pinned dependencies   {code.dependency_digest}")
    print(f"identity files covered  {len(code.file_digests)}")
    print(f"git commit              {code.git_commit or 'unavailable'}")
    print(f"git dirty               {code.git_dirty}")
    print(f"python                  {environment['python_version']}")
    print(f"numpy                   {environment['numpy_version']}")
    print(
        f"seed / bit generator    {config.reproducibility.seed} / "
        f"{config.reproducibility.bit_generator}"
    )
    print(f"run id (config-check)   {artifacts.new_run_id('config-check', config)}")
    print(f"held-out authorised     {held_out_authorised()}")
    return 0


def run_acquire() -> int:
    """Download and verify the official monthly archives.

    Writes the raw snapshot plus this run's inventory, log and manifest. No
    data rows are read.
    """
    result = acquisition.run_acquisition(load_config())
    for line in acquisition.summarise(result):
        print(line)
    return 0


def run_validate() -> int:
    """Validate the accepted raw snapshot without constructing outcomes."""
    result = validation.run_validation(load_config())
    for line in validation.summarise(result):
        print(line)
    return 0


def run_features() -> int:
    """Build the feature-stage fields without forward outcomes."""
    result = features.run_features(load_config())
    for line in features.summarise(result):
        print(line)
    return 0


def run_count_only() -> int:
    """Count events without using future values."""
    result = count_audit.run_count_audit(load_config())
    for line in count_audit.summarise(result):
        print(line)
    return 0


def run_outcomes() -> int:
    """Build development outcomes from the accepted feature and count runs."""
    result = outcomes.run_development_outcomes(load_config())
    for line in outcomes.summarise(result):
        print(line)
    return 0


def run_development() -> int:
    """Run inference on accepted 2020-2023 outcomes only."""
    result = inference.run_development_inference(load_config())
    for line in inference.summarise(result):
        print(line)
    return 0


def run_backtest() -> int:
    """Run the development-only strategy backtest."""
    result = backtest.run_development_backtest(load_config())
    for line in backtest.summarise(result):
        print(line)
    return 0


def run_robustness() -> int:
    """Run every pre-specified development robustness check."""
    result = robustness.run_development_robustness(load_config())
    for line in robustness.summarise(result):
        print(line)
    return 0


def run_freeze() -> int:
    """Verify and record every identity before held-out access."""
    result = freeze.run_freeze(load_config())
    for line in freeze.summarise(result):
        print(line)
    return 0


def run_held_out() -> int:
    """Run the one authorised held-out evaluation."""
    result = heldout.run_held_out(load_config())
    for line in heldout.summarise(result):
        print(line)
    return 0


def run_held_out_robustness() -> int:
    """Run the pre-specified held-out robustness checks once."""
    result = robustness.run_held_out_robustness(load_config())
    for line in robustness.summarise_held_out(result):
        print(line)
    return 0


def run_report() -> int:
    """Regenerate verified report tables without creating an analytical run."""
    from src import report_tables

    report_tables.write()
    print(report_tables.OUTPUT_RELATIVE_PATH.as_posix())
    return 0


def dispatch(mode: str) -> int:
    """Run the handler for a mode; raise an error for unbuilt stages."""
    if mode not in artifacts.EXECUTION_MODES:
        raise ModeNotImplementedError(f"unknown mode {mode!r}")
    if mode == "config-check":
        return run_config_check()
    if mode == "acquire":
        return run_acquire()
    if mode == "validate":
        return run_validate()
    if mode == "features":
        return run_features()
    if mode == "count-only":
        return run_count_only()
    if mode == "outcomes":
        return run_outcomes()
    if mode == "development":
        return run_development()
    if mode == "backtest":
        return run_backtest()
    if mode == "robustness":
        return run_robustness()
    if mode == "freeze":
        return run_freeze()
    if mode == "held-out":
        require_held_out_authorisation()
        return run_held_out()
    if mode == "held-out-robustness":
        require_held_out_robustness_authorisation()
        return run_held_out_robustness()
    if mode == "report":
        return run_report()
    raise ModeNotImplementedError(
        f"Mode {mode!r} belongs to stage '{_MODE_STAGE[mode]}', which is not implemented."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Buy the Weekend Dip? pipeline execution modes.",
    )
    parser.add_argument("mode", choices=list(artifacts.EXECUTION_MODES))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return dispatch(args.mode)
    except (
        ModeNotImplementedError,
        AuthorisationError,
        AcquisitionError,
        ArtifactError,
        BacktestError,
        CountAuditError,
        EventError,
        FeatureError,
        FreezeError,
        HeldOutError,
        InferenceError,
        OutcomeError,
        RobustnessError,
        ValidationError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
