"""Verify historical freeze evidence without reopening held-out evaluation.

The completed submission cannot create another freeze. Historical document
checksums remain metadata; the accepted manifest binds their original values.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src import paths
from src.acquisition import verify_manifest_checksum
from src.artifacts import (
    build_manifest,
    read_manifest,
    sha256_bytes,
    sha256_file,
)
from src.backtest import ACCEPTED_INFERENCE_MANIFEST
from src.config import ConfigNode, load_config
from src.count_audit import ACCEPTED_FEATURE_MANIFEST
from src.inference import ACCEPTED_OUTCOME_MANIFEST
from src.outcomes import ACCEPTED_COUNT_MANIFEST
from src.robustness import ACCEPTED_BACKTEST_MANIFEST

STAGE = "10. Held-out freeze"
MODE = "freeze"
MANIFEST_SLUG = "held-out-freeze"
FREEZE_RECORD_NAME = "held_out_freeze_record.json"

ACCEPTED_FREEZE_MANIFEST = Path(
    "outputs/runs/20260909T151727.978912Z-freeze-dad36cad-31d06a/"
    "held-out-freeze.manifest.json"
)
ACCEPTED_FREEZE_SHA256 = "4e8e36b99b5f04e238d867592cbd1f82defd65a99896c149bb8d63dd7e7812f9"

ACCEPTED_ACQUISITION_MANIFEST = Path(
    "outputs/runs/20260904T112326.201050Z-acquire-61466e5a-95fc17/"
    "acquisition.manifest.json"
)
ACCEPTED_VALIDATION_MANIFEST = Path(
    "outputs/runs/20260904T150910.352311Z-validate-2114d1d3-82e4cb/"
    "validation.manifest.json"
)
ACCEPTED_ROBUSTNESS_MANIFEST = Path(
    "outputs/runs/20260909T104454.475358Z-robustness-e052ad2d-dc18b3/"
    "development-robustness.manifest.json"
)

ACCEPTED_MANIFESTS: dict[str, Path] = {
    "acquisition": ACCEPTED_ACQUISITION_MANIFEST,
    "validation": ACCEPTED_VALIDATION_MANIFEST,
    "features": ACCEPTED_FEATURE_MANIFEST,
    "count_only": ACCEPTED_COUNT_MANIFEST,
    "development_outcomes": ACCEPTED_OUTCOME_MANIFEST,
    "development_inference": ACCEPTED_INFERENCE_MANIFEST,
    "development_backtest": ACCEPTED_BACKTEST_MANIFEST,
    "development_robustness": ACCEPTED_ROBUSTNESS_MANIFEST,
}

FREEZE_RECORD_FIELDS = (
    "status",
    "specification",
    "configuration",
    "code_identity",
    "environment",
    "seed",
    "accepted_manifests",
    "input_count",
    "input_set_sha256",
    "core_checks",
    "decision_log_sha256",
    "held_out_outcomes_accessed",
)

FROZEN_SOURCE_IDENTITY_FIELDS = (
    "source_digest",
    "code_digest",
    "test_digest",
    "config_digest",
    "dependency_digest",
    "file_digests",
)


class FreezeError(RuntimeError):
    """Raised when the held-out freeze cannot be shown to be complete."""


@dataclass(frozen=True)
class FreezeResult:
    run_id: str
    manifest_relative_path: Path
    manifest_checksum_relative_path: Path
    record_relative_path: Path
    input_count: int


def _input_set_digest(checksums: Mapping[str, str]) -> str:
    payload = "\n".join(
        f"{name}:{digest}" for name, digest in sorted(checksums.items())
    )
    return sha256_bytes(payload.encode("utf-8"))


def source_identity_matches(
    frozen: Mapping[str, object], current: Mapping[str, object]
) -> bool:
    """Compare the files that affect results, ignoring commits that only add outputs."""
    return all(
        frozen.get(name) == current.get(name) for name in FROZEN_SOURCE_IDENTITY_FIELDS
    )


def run_core_checks() -> tuple[dict[str, object], ...]:
    """Run the test suite and environment checks without reading market data."""
    commands: Sequence[tuple[str, list[str], list[str]]] = (
        (
            "pytest",
            [sys.executable, "-m", "pytest"],
            [Path(sys.executable).name, "-m", "pytest"],
        ),
        (
            "pip_check",
            [sys.executable, "-m", "pip", "check"],
            [Path(sys.executable).name, "-m", "pip", "check"],
        ),
        ("git_diff_check", ["git", "diff", "--check"], ["git", "diff", "--check"]),
    )
    records: list[dict[str, object]] = []
    for name, command, recorded_command in commands:
        completed = subprocess.run(
            command,
            cwd=paths.project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        combined = completed.stdout + completed.stderr
        records.append(
            {
                "name": name,
                "command": recorded_command,
                "exit_code": completed.returncode,
                "output_sha256": sha256_bytes(combined.encode("utf-8")),
            }
        )
        if completed.returncode != 0:
            raise FreezeError(
                f"core check {name} failed with exit {completed.returncode}"
            )
    return tuple(records)


def run_freeze(config: ConfigNode | None = None) -> FreezeResult:
    """Refuse new freeze records in the completed historical submission."""
    raise FreezeError(
        "the historical evaluation is complete; creating another freeze is disabled. "
        "Run notebook.ipynb to reproduce the accepted analysis in memory."
    )


def verify_freeze(
    manifest_relative: Path, *, require_current_identity: bool
) -> dict[str, object]:
    verify_manifest_checksum(manifest_relative)
    manifest = read_manifest(manifest_relative)
    historical_note = "docs/DECISION_LOG.md"
    if historical_note in manifest["input_checksums"] and (
        manifest_relative != ACCEPTED_FREEZE_MANIFEST
        or sha256_file(paths.resolve(manifest_relative)) != ACCEPTED_FREEZE_SHA256
    ):
        raise FreezeError("historical freeze manifest identity changed")
    if manifest["stage"] != STAGE or manifest["mode"] != MODE:
        raise FreezeError("manifest is not Stage 10 freeze evidence")
    # Historical freeze records include additional metadata fields, so each
    # record is checked against the schema its own manifest declares.
    schema = manifest["column_schema"]
    if (
        not isinstance(schema, list)
        or not set(FREEZE_RECORD_FIELDS) <= set(schema)
        or manifest["row_count"] != 1
    ):
        raise FreezeError("freeze manifest schema or count is invalid")
    record_relative = manifest_relative.parent / FREEZE_RECORD_NAME
    if sha256_file(paths.resolve(record_relative)) != manifest["output_checksum"]:
        raise FreezeError("freeze record checksum does not match its manifest")
    try:
        record = json.loads(paths.resolve(record_relative).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"cannot read freeze record: {error}") from None
    if not isinstance(record, dict) or list(record) != schema:
        raise FreezeError("freeze record fields are invalid")
    if record["status"] != "COMPLETE" or record["held_out_outcomes_accessed"] is not False:
        raise FreezeError("freeze record does not preserve analytical isolation")
    if record["code_identity"] != manifest["code_identity"]:
        raise FreezeError("freeze record code identity differs from its manifest")
    if (
        record["environment"] != manifest["environment"]
        or record["seed"] != manifest["seed"]
    ):
        raise FreezeError("freeze record environment or seed differs from its manifest")
    specification = record["specification"]
    configuration = record["configuration"]
    if (
        not isinstance(specification, dict)
        or not isinstance(configuration, dict)
        or set(specification) != {"relative_path", "version", "sha256"}
        or set(configuration) != {"relative_path", "version", "sha256"}
        or specification.get("version") != manifest["spec_version"]
        or specification.get("sha256") != manifest["spec_sha256"]
        or configuration.get("version") != manifest["config_version"]
        or configuration.get("sha256") != manifest["config_sha256"]
    ):
        raise FreezeError(
            "freeze record specification/configuration differs from manifest"
        )
    accepted = record["accepted_manifests"]
    if not isinstance(accepted, dict) or set(accepted) != set(ACCEPTED_MANIFESTS):
        raise FreezeError("freeze record does not name every accepted stage")
    for name, expected_path in ACCEPTED_MANIFESTS.items():
        item = accepted[name]
        if not isinstance(item, dict) or item.get("path") != expected_path.as_posix():
            raise FreezeError(f"freeze record has the wrong {name} manifest")
        if manifest["input_checksums"].get(item["path"]) != item.get("sha256"):
            raise FreezeError(f"freeze record does not bind the {name} manifest")
    core_checks = record["core_checks"]
    if (
        not isinstance(core_checks, list)
        or any(not isinstance(item, dict) for item in core_checks)
        or [item.get("name") for item in core_checks]
        != ["pytest", "pip_check", "git_diff_check"]
        or any(item.get("exit_code") != 0 for item in core_checks)
        or any(
            not isinstance(item.get("output_sha256"), str)
            or len(item["output_sha256"]) != 64
            for item in core_checks
        )
    ):
        raise FreezeError("freeze core-check evidence is incomplete or unsuccessful")
    if record["input_count"] != len(manifest["input_checksums"]):
        raise FreezeError("freeze input count is inconsistent")
    if record["input_set_sha256"] != _input_set_digest(manifest["input_checksums"]):
        raise FreezeError("freeze aggregate input checksum is inconsistent")
    if record["decision_log_sha256"] != manifest["input_checksums"].get(
        "docs/DECISION_LOG.md"
    ):
        raise FreezeError("freeze decision-log checksum is inconsistent")
    ledger_digest = record.get("ledger_pre_authorisation_sha256")
    if "ledger_pre_authorisation_sha256" in record and (
        not isinstance(ledger_digest, str)
        or len(ledger_digest) != 64
        or any(character not in "0123456789abcdef" for character in ledger_digest)
    ):
        raise FreezeError("freeze pre-authorisation ledger checksum is invalid")
    for name, digest in manifest["input_checksums"].items():
        # The exact accepted manifest above binds the retired note's hash.
        # It is historical metadata, not an input to notebook reproduction.
        if name == historical_note:
            continue
        if sha256_file(paths.resolve(name)) != digest:
            raise FreezeError(f"frozen input changed: {name}")
    if require_current_identity:
        current = build_manifest(STAGE, MODE, load_config())
        if not source_identity_matches(
            manifest["code_identity"], current.code_identity
        ):
            raise FreezeError(
                "current code/test/dependency identity differs from freeze"
            )
        if (
            current.spec_sha256 != manifest["spec_sha256"]
            or current.config_sha256 != manifest["config_sha256"]
        ):
            raise FreezeError(
                "current specification or configuration differs from freeze"
            )
        if current.code_identity["git_dirty"] is not False:
            raise FreezeError("held-out execution requires a clean repository")
        if (
            current.environment != manifest["environment"]
            or current.seed != manifest["seed"]
        ):
            raise FreezeError("current environment or seed differs from freeze")
    return record


def summarise(result: FreezeResult) -> tuple[str, ...]:
    return (
        f"run id                  {result.run_id}",
        f"manifest                {result.manifest_relative_path.as_posix()}",
        f"freeze record           {result.record_relative_path.as_posix()}",
        f"frozen input files      {result.input_count}",
        "held-out outcomes       not accessed",
    )
