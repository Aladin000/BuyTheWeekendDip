"""Tests for manifests, run IDs, code identity, write-once files, and run logs."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone

import pytest

from src import artifacts, paths
from tests.conftest import build_temp_project_root
from src.artifacts import (
    EXECUTION_MODES,
    IDENTITY_GROUPS,
    IDENTITY_PACKAGES,
    REQUIRED_CODE_IDENTITY_FIELDS,
    REQUIRED_ENVIRONMENT_FIELDS,
    REQUIRED_IDENTITY_FILES,
    REQUIRED_MANIFEST_FIELDS,
    RUN_ID_PATTERN,
    ArtifactError,
    Manifest,
    build_manifest,
    classify_identity_path,
    code_identity,
    compose_run_id,
    config_identity,
    create_exclusive,
    create_exclusive_bytes,
    create_run_log,
    environment_identity,
    identity_suffix,
    reconcile_identity_digests,
    manifest_relative_path,
    new_attempt_token,
    new_run_id,
    read_manifest,
    run_id_instant,
    run_identity_digest,
    run_log_relative_path,
    run_relative_dir,
    sha256_bytes,
    sha256_file,
    spec_identity,
    validate_manifest_complete,
    validate_manifest_mapping,
    validate_run_id,
)

EXPECTED_MANIFEST_FIELDS = (
    "run_id",
    "created_at_utc",
    "stage",
    "mode",
    "spec_version",
    "spec_sha256",
    "config_version",
    "config_sha256",
    "input_checksums",
    "output_checksum",
    "code_identity",
    "row_count",
    "column_schema",
    "exclusions",
    "eligibility_changes",
    "environment",
    "seed",
)

FIXED_MOMENT = datetime(2026, 9, 3, 22, 15, 0, tzinfo=timezone.utc)
STAGE = "1. Environment and configuration"


def literal_file_digests(salt: str) -> dict[str, str]:
    """A complete identity file set with made-up digests.

    Every group and required file is present, so the set is self-consistent
    but matches no real commit.
    """
    names = (
        "src/artifacts.py",
        "src/config.py",
        "src/paths.py",
        "src/pipeline.py",
        "src/rng.py",
        "tests/conftest.py",
        "tests/test_artifacts.py",
        "tests/fixtures/short_warmup.yaml",
        "config/project.yaml",
        "pyproject.toml",
        "requirements.txt",
        "requirements-lock.txt",
    )
    return {
        name: sha256_bytes(f"{salt}:{name}".encode("utf-8")) for name in sorted(names)
    }


def literal_code_identity(salt: str) -> dict:
    """Git state plus digests reconciled from ``literal_file_digests``."""
    file_digests = literal_file_digests(salt)
    return {
        "git_commit": "0" * 40,
        "git_dirty": False,
        **reconcile_identity_digests(file_digests),
        "file_digests": file_digests,
    }


# Self-consistent but unrelated to the checked-out project, like a manifest
# from an earlier commit.
SYNTHETIC_CODE_IDENTITY = literal_code_identity("synthetic")
SYNTHETIC_FILE_DIGESTS = SYNTHETIC_CODE_IDENTITY["file_digests"]
SPEC_SHA = "a" * 64
CONFIG_SHA = SYNTHETIC_FILE_DIGESTS["config/project.yaml"]
SOURCE_DIGEST = SYNTHETIC_CODE_IDENTITY["source_digest"]
SYNTHETIC_IDENTITY = identity_suffix(SPEC_SHA, CONFIG_SHA, SOURCE_DIGEST)
VALID_RUN_ID = f"20260903T221500.000000Z-config-check-{SYNTHETIC_IDENTITY}-abc123"

SYNTHETIC_ENVIRONMENT = {
    "python_version": "3.12.13",
    "python_implementation": "CPython",
    "platform": "macOS-15.0-arm64-arm-64bit",
    "numpy_version": "2.3.5",
    "packages": {"numpy": "2.3.5", "PyYAML": "6.0.3", "pytest": "9.1.1"},
    "executable": "python3.12",
}


def mutated_code_identity(**mutation) -> dict:
    """The synthetic code identity with one member replaced."""
    return {**SYNTHETIC_CODE_IDENTITY, **mutation}


def synthetic_manifest(**overrides) -> Manifest:
    """A complete, self-consistent manifest built from literals.

    It does not depend on the real spec, config, or code digests, so it also
    works inside a temporary project root.
    """
    payload = {
        "run_id": VALID_RUN_ID,
        "created_at_utc": "2026-09-03T22:15:00Z",
        "stage": STAGE,
        "mode": "config-check",
        "spec_version": "1.4",
        "spec_sha256": SPEC_SHA,
        "config_version": "1.0.0",
        "config_sha256": CONFIG_SHA,
        "input_checksums": {},
        "output_checksum": "9" * 64,
        "code_identity": mutated_code_identity(),
        "row_count": 1000,
        "column_schema": ["open_time", "close", "volume"],
        "exclusions": {},
        "eligibility_changes": {},
        "environment": dict(SYNTHETIC_ENVIRONMENT),
        "seed": 20260901,
    }
    payload.update(overrides)
    return Manifest(**payload)  # type: ignore[arg-type]


def synthetic_draft(**overrides) -> Manifest:
    """The same manifest before its artifact exists."""
    return synthetic_manifest(
        output_checksum=None, row_count=None, column_schema=[], **overrides
    )


# Manifest schema


def test_s1_22_required_manifest_fields_are_exactly_the_specified_set() -> None:
    assert REQUIRED_MANIFEST_FIELDS == EXPECTED_MANIFEST_FIELDS


def test_s1_22_manifest_dataclass_matches_the_required_fields(config) -> None:
    manifest = build_manifest(STAGE, "config-check", config, moment=FIXED_MOMENT)
    assert tuple(manifest.to_dict()) == EXPECTED_MANIFEST_FIELDS


def test_s1_22_manifest_records_specification_and_config_identity(config) -> None:
    manifest = build_manifest(STAGE, "config-check", config, moment=FIXED_MOMENT)
    assert manifest.spec_version == "1.5"
    assert manifest.spec_sha256 == spec_identity(config).sha256
    assert manifest.config_version == config.meta.config_version
    assert manifest.config_sha256 == config_identity(config).sha256
    assert manifest.seed == config.reproducibility.seed
    assert manifest.environment["numpy_version"]
    assert tuple(manifest.code_identity) == REQUIRED_CODE_IDENTITY_FIELDS


def test_s1_22_deferred_fields_are_present_but_empty(config) -> None:
    """The keys are always present, even before the output exists."""
    manifest = build_manifest(STAGE, "config-check", config, moment=FIXED_MOMENT)
    assert manifest.output_checksum is None
    assert manifest.row_count is None
    assert manifest.column_schema == []
    assert manifest.exclusions == {}
    assert manifest.eligibility_changes == {}
    assert manifest.is_complete() is False


def test_s1_22_a_draft_becomes_complete_when_the_artifact_exists(config) -> None:
    draft = build_manifest(STAGE, "config-check", config, moment=FIXED_MOMENT)
    finished = draft.completed("7" * 64, 52_608, ["open_time", "close"])
    assert finished.is_complete() is True
    assert finished.run_id == draft.run_id
    assert finished.output_checksum == "7" * 64
    assert finished.row_count == 52_608
    assert finished.column_schema == ["open_time", "close"]
    assert draft.is_complete() is False, "the draft must not be mutated"


# Checksums


def test_s1_23_sha256_bytes_matches_hashlib() -> None:
    payload = b"buy the weekend dip"
    assert sha256_bytes(payload) == hashlib.sha256(payload).hexdigest()


def test_s1_23_sha256_file_matches_hashlib(tmp_path) -> None:
    target = tmp_path / "fixture.txt"
    payload = b"a" * (3 * (1 << 20) + 17)  # spans several read chunks
    target.write_bytes(payload)
    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_s1_23_sha256_file_fails_closed_on_a_missing_file(tmp_path) -> None:
    with pytest.raises(paths.ProjectLayoutError, match="Required file missing"):
        sha256_file(tmp_path / "absent.txt")


def test_s1_23_config_and_spec_digests_are_stable(config) -> None:
    assert config_identity(config).sha256 == sha256_file(paths.config_path())
    assert spec_identity(config).sha256 == sha256_file(paths.spec_path())


# Run IDs


def test_s1_24_run_id_matches_the_documented_format(config) -> None:
    run_id = new_run_id("config-check", config, moment=FIXED_MOMENT)
    assert re.match(RUN_ID_PATTERN, run_id), run_id
    assert run_id.startswith("20260903T221500.000000Z-config-check-")
    assert validate_run_id(run_id) is not None


def test_s1_24_run_id_carries_the_identity_digest(config) -> None:
    run_id = new_run_id("features", config, moment=FIXED_MOMENT)
    identity = run_identity_digest(config)[:8]
    assert run_id.split("-")[-2] == identity


def test_s1_24_two_attempts_in_the_same_microsecond_are_distinct(config) -> None:
    """Same mode, config, code, and instant still give different IDs.

    Every attempt needs its own ID, so uniqueness cannot rely on the clock.
    """
    first = new_run_id("development", config, moment=FIXED_MOMENT)
    second = new_run_id("development", config, moment=FIXED_MOMENT)
    assert first != second
    assert first.rsplit("-", 1)[0] == second.rsplit("-", 1)[0]


def test_s1_24_many_attempts_at_one_instant_stay_unique(config) -> None:
    ids = {new_run_id("development", config, moment=FIXED_MOMENT) for _ in range(200)}
    assert len(ids) == 200


def test_s1_24_wall_clock_attempts_stay_unique(config) -> None:
    """Back-to-back real calls, which land in the same second."""
    ids = {new_run_id("config-check", config) for _ in range(50)}
    assert len(ids) == 50


def test_s1_24_attempt_token_is_six_lowercase_hex_characters() -> None:
    for _ in range(20):
        assert re.fullmatch(r"[0-9a-f]{6}", new_attempt_token())


def test_s1_24_attempt_tokens_differ_across_draws() -> None:
    """Tokens are random, so IDs stay unique across processes."""
    tokens = {new_attempt_token() for _ in range(500)}
    assert len(tokens) > 400, "uuid4-derived tokens should rarely repeat"


def test_s1_24_compose_run_id_reuses_a_measured_identity(config) -> None:
    """build_manifest uses this to measure the identity only once."""
    identity = run_identity_digest(config)[:8]
    composed = compose_run_id("acquire", FIXED_MOMENT, identity, "0f0f0f")
    assert composed == f"20260903T221500.000000Z-acquire-{identity}-0f0f0f"
    assert validate_run_id(composed).group("identity") == identity
    with pytest.raises(ArtifactError, match="run identity"):
        compose_run_id("acquire", FIXED_MOMENT, "xyz", "0f0f0f")


def test_s1_24_manifest_identity_is_measured_once(config) -> None:
    """The run ID and the manifest body describe the same measurement."""
    manifest = build_manifest(STAGE, "config-check", config, moment=FIXED_MOMENT)
    expected = identity_suffix(
        manifest.spec_sha256,
        manifest.config_sha256,
        manifest.code_identity["source_digest"],
    )
    assert validate_run_id(manifest.run_id).group("identity") == expected


def test_s1_24_explicit_attempt_token_is_used(config) -> None:
    run_id = new_run_id("report", config, moment=FIXED_MOMENT, attempt_token="0f0f0f")
    assert run_id.endswith("-0f0f0f")


def test_s1_24_malformed_attempt_token_fails_closed(config) -> None:
    with pytest.raises(ArtifactError, match="lowercase hex characters"):
        new_run_id("report", config, moment=FIXED_MOMENT, attempt_token="ZZZZZZ")


def test_s1_24_microsecond_precision_is_preserved(config) -> None:
    moment = FIXED_MOMENT.replace(microsecond=123456)
    run_id = new_run_id("validate", config, moment=moment, attempt_token="000000")
    assert run_id.startswith("20260903T221500.123456Z-validate-")


@pytest.mark.parametrize("mode", EXECUTION_MODES)
def test_s1_24_every_execution_mode_yields_a_valid_run_id(config, mode: str) -> None:
    run_id = new_run_id(mode, config, moment=FIXED_MOMENT)
    assert validate_run_id(run_id).group("mode") == mode


def test_s1_24_unknown_mode_fails_closed(config) -> None:
    with pytest.raises(ArtifactError, match="unknown execution mode"):
        new_run_id("optimise", config, moment=FIXED_MOMENT)


def test_s1_24_naive_run_timestamp_fails_closed(config) -> None:
    with pytest.raises(ArtifactError, match="timezone aware"):
        new_run_id("config-check", config, moment=datetime(2026, 9, 3, 22, 15))


@pytest.mark.parametrize(
    "run_id",
    (
        "20260903T221500Z-config-check-0123abcd-abc123",  # no microseconds
        "20260903T221500.000000Z-config-check-0123abcd",  # no attempt token
        "20260903T221500.000000Z-config-check-0123ABCD-abc123",  # uppercase digest
        "20260903T221500.000000Z-optimise-0123abcd-abc123",  # unknown mode
        "20260932T221500.000000Z-config-check-0123abcd-abc123",  # impossible date
        "not-a-run-id",
        "",
    ),
)
def test_s1_24_malformed_run_ids_fail_closed(run_id: str) -> None:
    with pytest.raises(ArtifactError):
        validate_run_id(run_id)


# Identity coverage


def test_s1_25_code_identity_records_git_state_and_group_digests() -> None:
    identity = code_identity()
    if (paths.project_root() / ".git").exists():
        assert re.fullmatch(r"[0-9a-f]{40}", identity.git_commit or "")
        assert isinstance(identity.git_dirty, bool)
    else:
        assert identity.git_commit is None
        assert identity.git_dirty is None
    for digest in (
        identity.source_digest,
        identity.code_digest,
        identity.test_digest,
        identity.config_digest,
        identity.dependency_digest,
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert tuple(identity.to_dict()) == REQUIRED_CODE_IDENTITY_FIELDS


def test_s1_25_snapshot_without_git_keeps_file_and_group_digests(tmp_path, monkeypatch):
    original_root = paths.project_root()
    snapshot = build_temp_project_root(tmp_path)
    for name in REQUIRED_IDENTITY_FILES:
        destination = snapshot / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original_root / name, destination)
    for name in ("src", "tests"):
        shutil.copytree(
            original_root / name,
            snapshot / name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    monkeypatch.setattr(paths, "project_root", lambda: snapshot)
    identity = code_identity()
    assert identity.git_commit is None
    assert identity.git_dirty is None
    assert set(REQUIRED_IDENTITY_FILES) <= set(identity.file_digests)
    assert (
        reconcile_identity_digests(identity.file_digests)["source_digest"]
        == identity.source_digest
    )


def test_s1_25_identity_is_deterministic() -> None:
    assert code_identity().to_dict() == code_identity().to_dict()


def test_s1_25_identity_covers_source_configuration_and_dependency_metadata() -> None:
    covered = code_identity().file_digests
    for name in REQUIRED_IDENTITY_FILES:
        assert name in covered, f"{name} is not represented in the recorded identity"
    for name in (
        "src/config.py",
        "src/artifacts.py",
        "src/paths.py",
        "src/pipeline.py",
        "src/rng.py",
        "pyproject.toml",
        "requirements.txt",
        "requirements-lock.txt",
        "config/project.yaml",
    ):
        assert name in covered, f"{name} is not represented in the recorded identity"


def test_s1_25_identity_covers_every_test_module_and_yaml_fixture(project_root) -> None:
    """Mutation fixtures change test outcomes, so they are part of the identity."""
    covered = set(code_identity().file_digests)
    fixtures = {
        path.relative_to(project_root).as_posix()
        for path in (project_root / "tests" / "fixtures").glob("*.yaml")
    }
    assert fixtures, "expected mutation fixtures to exist"
    assert fixtures <= covered
    assert "tests/fixtures/short_warmup.yaml" in covered
    assert "tests/fixtures/_format.yaml" in covered

    modules = {
        path.relative_to(project_root).as_posix()
        for path in (project_root / "tests").rglob("*.py")
    }
    assert modules <= covered


def test_s1_25_identity_excludes_caches_and_virtual_environments() -> None:
    covered = code_identity().file_digests
    for name in covered:
        parts = set(name.split("/"))
        assert not parts.intersection({".venv", "__pycache__", ".pytest_cache", ".git"}), name


def test_s1_25_group_digests_partition_the_discovered_files() -> None:
    identity = code_identity()
    covered = set(identity.file_digests)
    assert {name for name in covered if name.startswith("src/")}
    assert {name for name in covered if name.startswith("tests/")}
    assert set(IDENTITY_GROUPS) == {"code", "test", "config", "dependency"}
    assert len({
        identity.code_digest,
        identity.test_digest,
        identity.config_digest,
        identity.dependency_digest,
    }) == 4


def test_s1_25_environment_identity_lists_pinned_packages() -> None:
    environment = environment_identity()
    assert environment["python_version"].startswith("3.12")
    assert set(environment["packages"]) == {
        "numpy",
        "PyYAML",
        "pytest",
        "scipy",
        "statsmodels",
        "matplotlib",
    }
    assert all(version != "not installed" for version in environment["packages"].values())


def test_s1_25_missing_dependency_metadata_fails_closed(temp_project_root) -> None:
    """A snapshot without pinned dependencies cannot claim a code identity."""
    (temp_project_root / "src").mkdir()
    (temp_project_root / "src" / "config.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="missing.*requirements"):
        code_identity()


# Manifest validation


def test_s1_25_synthetic_manifest_is_valid() -> None:
    assert validate_manifest_mapping(synthetic_manifest().to_dict())


def test_s1_25_manifest_missing_field_fails_closed() -> None:
    payload = synthetic_manifest().to_dict()
    del payload["code_identity"]
    with pytest.raises(ArtifactError, match=r"missing required field\(s\) \['code_identity'\]"):
        validate_manifest_mapping(payload)


def test_s1_25_manifest_unknown_field_fails_closed() -> None:
    payload = synthetic_manifest().to_dict()
    payload["notes"] = "hand written"
    with pytest.raises(ArtifactError, match=r"unknown field\(s\) \['notes'\]"):
        validate_manifest_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        # Presence and basic types
        ("run_id", "", "non-empty string"),
        ("stage", 1, "non-empty string"),
        ("input_checksums", [], "must be a mapping"),
        ("column_schema", {}, "must be a list"),
        # Run identity
        ("run_id", "not-a-run-id", "malformed run_id"),
        (
            "run_id",
            "20260903T221500.000000Z-optimise-0123abcd-abc123",
            "unknown execution mode",
        ),
        ("mode", "optimise", "not a known execution mode"),
        ("mode", "features", "disagrees with the recorded mode"),
        # Creation timestamp
        ("created_at_utc", "2026-09-03 22:15:00", "must be 'YYYY-MM-DDTHH:MM:SSZ'"),
        ("created_at_utc", "2026-09-03T22:15:00+00:00", "must be 'YYYY-MM-DDTHH:MM:SSZ'"),
        ("created_at_utc", "2026-13-03T22:15:00Z", "unparseable"),
        # Checksums
        ("spec_sha256", "abc", "64 lowercase hex"),
        ("config_sha256", "A" * 64, "64 lowercase hex"),
        ("output_checksum", "deadbeef", "64 lowercase hex"),
        ("input_checksums", {"raw": "short"}, "64 lowercase hex"),
        ("input_checksums", {"raw": None}, "64 lowercase hex"),
        # Input checksum names
        ("input_checksums", {"": "f" * 64}, "input checksum name must be a non-empty string"),
        ("input_checksums", {1: "f" * 64}, "input checksum name must be a non-empty string"),
        # Counts
        ("row_count", "12", "must be an integer or null"),
        ("row_count", True, "must be an integer or null"),
        ("row_count", -1, "must not be negative"),
        ("seed", "20260901", "non-negative integer"),
        ("seed", -1, "non-negative integer"),
        ("exclusions", {"gap": -3}, "must not be negative"),
        ("exclusions", {"gap": 1.5}, "must be an integer"),
        ("eligibility_changes", {"model": -1}, "must not be negative"),
        ("eligibility_changes", {"model": True}, "must be an integer"),
        # Column schema
        ("column_schema", ["open", 2], "non-empty strings"),
        ("column_schema", ["open", ""], "non-empty strings"),
        ("column_schema", ["open", "open"], "must not repeat a column name"),
    ),
)
def test_s1_25_manifest_semantic_validation_fails_closed(field, value, message) -> None:
    payload = synthetic_manifest().to_dict()
    payload[field] = value
    with pytest.raises(ArtifactError, match=message):
        validate_manifest_mapping(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        # Group digests
        ({"source_digest": "nope"}, "64 lowercase hex"),
        ({"dependency_digest": "1" * 63}, "64 lowercase hex"),
        ({"config_digest": None}, "64 lowercase hex"),
        # Unknown members
        ({"reviewer": "approved"}, r"unknown field\(s\) \['reviewer'\]"),
        # Git state
        ({"git_commit": "abc"}, "null or 40 lowercase hex"),
        ({"git_commit": "A" * 40}, "null or 40 lowercase hex"),
        ({"git_commit": 12345}, "null or 40 lowercase hex"),
        ({"git_commit": "0" * 41}, "null or 40 lowercase hex"),
        ({"git_dirty": "false"}, "null or a boolean"),
        ({"git_dirty": 0}, "null or a boolean"),
        ({"git_dirty": 1}, "null or a boolean"),
        ({"git_commit": None}, "'git_dirty' without a 'git_commit'"),
        # File digests
        ({"file_digests": {}}, "non-empty mapping"),
        ({"file_digests": []}, "non-empty mapping"),
        ({"file_digests": {"src/config.py": "short"}}, "64 lowercase hex"),
        ({"file_digests": {"src/config.py": None}}, "64 lowercase hex"),
        ({"file_digests": {"/etc/passwd": "2" * 64}}, "relative POSIX path"),
        ({"file_digests": {"~/secrets": "2" * 64}}, "relative POSIX path"),
        ({"file_digests": {"src\\config.py": "2" * 64}}, "relative POSIX path"),
        ({"file_digests": {"C:/src/config.py": "2" * 64}}, "relative POSIX path"),
        ({"file_digests": {"../outside/config.py": "2" * 64}}, "traversal segments"),
        ({"file_digests": {"src//config.py": "2" * 64}}, "traversal segments"),
        ({"file_digests": {"src/./config.py": "2" * 64}}, "traversal segments"),
        ({"file_digests": {"": "2" * 64}}, "non-empty string"),
        ({"file_digests": {7: "2" * 64}}, "non-empty string"),
    ),
)
def test_s1_25_manifest_code_identity_validation_fails_closed(mutation, message) -> None:
    payload = synthetic_manifest().to_dict()
    payload["code_identity"] = {**payload["code_identity"], **mutation}
    with pytest.raises(ArtifactError, match=message):
        validate_manifest_mapping(payload)


def test_s1_25_manifest_code_identity_missing_group_fails_closed() -> None:
    payload = synthetic_manifest().to_dict()
    code = dict(payload["code_identity"])
    del code["test_digest"]
    payload["code_identity"] = code
    with pytest.raises(ArtifactError, match=r"missing field\(s\) \['test_digest'\]"):
        validate_manifest_mapping(payload)


def test_s1_25_manifest_accepts_a_snapshot_without_git_metadata() -> None:
    """A submitted snapshot outside a repository still carries an identity."""
    payload = synthetic_manifest().to_dict()
    payload["code_identity"] = mutated_code_identity(git_commit=None, git_dirty=None)
    assert validate_manifest_mapping(payload)


# Identity reconciliation


@pytest.mark.parametrize(
    ("relative_path", "group"),
    (
        ("src/config.py", "code"),
        ("src/artifacts.py", "code"),
        ("src/nested/deeper/module.py", "code"),
        ("tests/test_config_values.py", "test"),
        ("tests/conftest.py", "test"),
        ("tests/fixtures/short_warmup.yaml", "test"),
        ("tests/fixtures/_format.yaml", "test"),
        ("config/project.yaml", "config"),
        ("pyproject.toml", "dependency"),
        ("requirements.txt", "dependency"),
        ("requirements-lock.txt", "dependency"),
    ),
)
def test_s1_30_paths_classify_into_exactly_one_group(relative_path, group) -> None:
    assert classify_identity_path(relative_path) == group


@pytest.mark.parametrize(
    "relative_path",
    (
        "docs/01_PROJECT_SPEC.md",
        "notebook.ipynb",
        "README.md",
        "src/config.pyc",
        "config/project.yml",
        "requirements-dev.txt",
        "outputs/runs/manifest.json",
    ),
)
def test_s1_30_an_unclassified_path_fails_closed(relative_path) -> None:
    with pytest.raises(ArtifactError, match="belongs to no identity group"):
        classify_identity_path(relative_path)


def test_s1_30_a_multiply_classified_path_fails_closed(monkeypatch) -> None:
    """The check still works if two group patterns ever overlap."""
    monkeypatch.setitem(
        artifacts.IDENTITY_GROUPS, "dependency", ("pyproject.toml", "src/**/*.py")
    )
    with pytest.raises(ArtifactError, match="more than one"):
        classify_identity_path("src/config.py")


def test_s1_30_classification_agrees_with_discovery_on_the_real_project() -> None:
    """The classifier and the discovery globs split files the same way."""
    covered = code_identity().file_digests
    grouped: dict[str, set[str]] = {group: set() for group in artifacts.IDENTITY_GROUPS}
    for name in covered:
        grouped[classify_identity_path(name)].add(name)

    assert grouped["code"] == {name for name in covered if name.startswith("src/")}
    assert grouped["test"] == {name for name in covered if name.startswith("tests/")}
    assert grouped["config"] == {name for name in covered if name.startswith("config/")}
    assert grouped["dependency"] == set(
        artifacts.IDENTITY_GROUPS["dependency"]
    ), "the dependency group is an exact file list"


def test_s1_30_recorded_group_digests_reconcile_with_the_real_identity() -> None:
    identity = code_identity()
    reconciled = reconcile_identity_digests(identity.file_digests)
    assert reconciled["code_digest"] == identity.code_digest
    assert reconciled["test_digest"] == identity.test_digest
    assert reconciled["config_digest"] == identity.config_digest
    assert reconciled["dependency_digest"] == identity.dependency_digest
    assert reconciled["source_digest"] == identity.source_digest


def test_s1_30_a_changed_per_file_digest_fails_closed() -> None:
    """A per-file digest cannot change behind an unchanged group digest."""
    tampered = {**SYNTHETIC_FILE_DIGESTS, "src/config.py": "0" * 64}
    payload = synthetic_manifest(
        code_identity=mutated_code_identity(file_digests=tampered)
    ).to_dict()
    with pytest.raises(ArtifactError, match="'code_digest'.*does not match"):
        validate_manifest_mapping(payload)


@pytest.mark.parametrize(
    ("changed_path", "expected_group"),
    (
        ("src/paths.py", "code_digest"),
        ("tests/conftest.py", "test_digest"),
        ("tests/fixtures/short_warmup.yaml", "test_digest"),
        ("config/project.yaml", "config_digest"),
        ("requirements-lock.txt", "dependency_digest"),
    ),
)
def test_s1_30_a_changed_file_is_attributed_to_its_group(changed_path, expected_group) -> None:
    tampered = {**SYNTHETIC_FILE_DIGESTS, changed_path: "0" * 64}
    payload = synthetic_manifest(
        code_identity=mutated_code_identity(file_digests=tampered)
    ).to_dict()
    with pytest.raises(ArtifactError, match=f"{expected_group}.*does not match"):
        validate_manifest_mapping(payload)


@pytest.mark.parametrize(
    "group_field",
    ("code_digest", "test_digest", "config_digest", "dependency_digest"),
)
def test_s1_30_a_changed_group_digest_fails_closed(group_field) -> None:
    """A group digest cannot change behind an unchanged source digest."""
    payload = synthetic_manifest(
        code_identity=mutated_code_identity(**{group_field: "3" * 64})
    ).to_dict()
    with pytest.raises(ArtifactError, match=f"{group_field}.*does not match"):
        validate_manifest_mapping(payload)


def test_s1_30_a_changed_source_digest_fails_closed() -> None:
    payload = synthetic_manifest(
        code_identity=mutated_code_identity(source_digest="4" * 64)
    ).to_dict()
    with pytest.raises(ArtifactError, match="'source_digest'.*does not match"):
        validate_manifest_mapping(payload)


@pytest.mark.parametrize("dropped", REQUIRED_IDENTITY_FILES)
def test_s1_30_a_missing_required_identity_file_fails_closed(dropped) -> None:
    remaining = {
        name: digest for name, digest in SYNTHETIC_FILE_DIGESTS.items() if name != dropped
    }
    payload = synthetic_manifest(
        code_identity=mutated_code_identity(file_digests=remaining)
    ).to_dict()
    with pytest.raises(ArtifactError, match=f"missing.*{re.escape(dropped)}"):
        validate_manifest_mapping(payload)


@pytest.mark.parametrize("group", ("src", "tests"))
def test_s1_30_an_empty_identity_group_fails_closed(group) -> None:
    """Every group must describe part of the project.

    Only the code and test groups can be emptied without also dropping a
    required file, which is caught earlier and more specifically.
    """
    remaining = {
        name: digest
        for name, digest in SYNTHETIC_FILE_DIGESTS.items()
        if not name.startswith(f"{group}/")
    }
    payload = synthetic_manifest(
        code_identity=mutated_code_identity(file_digests=remaining)
    ).to_dict()
    with pytest.raises(ArtifactError, match="record no file"):
        validate_manifest_mapping(payload)


def test_s1_30_an_unclassified_recorded_file_fails_closed() -> None:
    extended = {**SYNTHETIC_FILE_DIGESTS, "docs/01_PROJECT_SPEC.md": "5" * 64}
    payload = synthetic_manifest(
        code_identity=mutated_code_identity(file_digests=extended)
    ).to_dict()
    with pytest.raises(ArtifactError, match="belongs to no identity group"):
        validate_manifest_mapping(payload)


def test_s1_30_configuration_identity_must_agree_with_the_recorded_digest() -> None:
    """The top-level config checksum is the project.yaml digest, not a copy."""
    payload = synthetic_manifest(config_sha256="6" * 64).to_dict()
    with pytest.raises(ArtifactError, match="'config_sha256'.*does not match the recorded digest"):
        validate_manifest_mapping(payload)


def test_s1_30_a_real_manifest_links_configuration_identity(config) -> None:
    manifest = build_manifest(STAGE, "config-check", config, moment=FIXED_MOMENT)
    assert manifest.config_sha256 == manifest.code_identity["file_digests"]["config/project.yaml"]
    assert manifest.config_sha256 == config_identity(config).sha256


@pytest.mark.parametrize(
    "packages",
    (
        {"numpy": "2.3.4", "PyYAML": "6.0.3", "pytest": "9.1.1"},
        {"numpy": "1.26.0", "PyYAML": "6.0.3", "pytest": "9.1.1"},
    ),
)
def test_s1_30_two_numpy_versions_fail_closed(packages) -> None:
    """A manifest must not record two different NumPy versions."""
    environment = {**SYNTHETIC_ENVIRONMENT, "packages": packages}
    payload = synthetic_manifest(environment=environment).to_dict()
    with pytest.raises(ArtifactError, match="two different NumPy versions"):
        validate_manifest_mapping(payload)


def test_s1_30_a_real_environment_records_one_numpy_version() -> None:
    environment = environment_identity()
    assert environment["numpy_version"] == environment["packages"]["numpy"]


# Environment validation


def test_s1_25_environment_identity_produces_exactly_the_required_fields() -> None:
    assert tuple(environment_identity()) == REQUIRED_ENVIRONMENT_FIELDS
    assert set(environment_identity()["packages"]) == set(IDENTITY_PACKAGES)


def test_s1_25_a_real_environment_identity_validates() -> None:
    payload = synthetic_manifest(environment=environment_identity()).to_dict()
    assert validate_manifest_mapping(payload)


@pytest.mark.parametrize(
    ("environment", "message"),
    (
        ({}, "must not be empty"),
        (
            {name: value for name, value in SYNTHETIC_ENVIRONMENT.items() if name != "platform"},
            r"environment is missing field\(s\) \['platform'\]",
        ),
        (
            {**SYNTHETIC_ENVIRONMENT, "reviewer": "approved"},
            r"environment has unknown field\(s\) \['reviewer'\]",
        ),
        ({**SYNTHETIC_ENVIRONMENT, "platform": ""}, "must be a non-empty string"),
        ({**SYNTHETIC_ENVIRONMENT, "executable": None}, "must be a non-empty string"),
        ({**SYNTHETIC_ENVIRONMENT, "python_implementation": 3}, "must be a non-empty string"),
        ({**SYNTHETIC_ENVIRONMENT, "python_version": "three"}, "must look like a version"),
        ({**SYNTHETIC_ENVIRONMENT, "numpy_version": "latest"}, "must look like a version"),
        ({**SYNTHETIC_ENVIRONMENT, "packages": {}}, "must be a non-empty mapping"),
        ({**SYNTHETIC_ENVIRONMENT, "packages": []}, "must be a non-empty mapping"),
        (
            {**SYNTHETIC_ENVIRONMENT, "packages": {"pytest": "9.1.1"}},
            r"must record version\(s\) for \['numpy'\]",
        ),
        (
            {**SYNTHETIC_ENVIRONMENT, "packages": {"numpy": None}},
            "version for 'numpy' must be a non-empty string",
        ),
        (
            {**SYNTHETIC_ENVIRONMENT, "packages": {"numpy": "2.3.5", "": "1.0"}},
            "package name must be a non-empty string",
        ),
    ),
)
def test_s1_25_manifest_environment_validation_fails_closed(environment, message) -> None:
    payload = synthetic_manifest(environment=environment).to_dict()
    with pytest.raises(ArtifactError, match=message):
        validate_manifest_mapping(payload)


# Run ID consistency


def test_s1_25_forged_identity_suffix_fails_closed() -> None:
    """A run ID cannot claim an identity its own digests do not produce."""
    forged = VALID_RUN_ID.replace(SYNTHETIC_IDENTITY, "deadbeef")
    assert forged != VALID_RUN_ID
    payload = synthetic_manifest(run_id=forged).to_dict()
    with pytest.raises(ArtifactError, match="does not match the identity"):
        validate_manifest_mapping(payload)


def test_s1_25_swapped_specification_digest_fails_closed() -> None:
    """Editing a recorded digest no longer matches the run ID.

    ``spec_sha256`` is the only identity part not also in ``file_digests``, so
    it reaches this check directly. Edits to ``config_sha256`` or
    ``source_digest`` are caught earlier by the reconciliation checks.
    """
    payload = synthetic_manifest().to_dict()
    payload["spec_sha256"] = "7" * 64
    with pytest.raises(ArtifactError, match="does not match the identity"):
        validate_manifest_mapping(payload)


@pytest.mark.parametrize(
    "created_at_utc",
    (
        "2026-09-03T21:15:00Z",  # different hour
        "2026-09-03T22:15:01Z",  # one second later
        "2026-09-03T22:14:59Z",  # one second earlier
        "2026-09-04T22:15:00Z",  # different day
        "2026-09-03T22:16:00Z",  # different minute
    ),
)
def test_s1_25_creation_timestamp_must_match_the_run_id_second(created_at_utc) -> None:
    payload = synthetic_manifest(created_at_utc=created_at_utc).to_dict()
    with pytest.raises(ArtifactError, match="describe different UTC seconds"):
        validate_manifest_mapping(payload)


def test_s1_25_run_id_may_keep_microseconds_within_the_same_second() -> None:
    """Second-resolution ``created_at_utc`` and a microsecond run ID agree."""
    run_id = f"20260903T221500.987654Z-config-check-{SYNTHETIC_IDENTITY}-abc123"
    payload = synthetic_manifest(run_id=run_id).to_dict()
    assert validate_manifest_mapping(payload)
    assert run_id_instant(run_id).microsecond == 987654


def test_s1_25_a_historical_manifest_validates_against_its_own_digests() -> None:
    """A manifest from an older commit still validates against itself.

    None of these digests belong to the checked-out project.
    """
    historical = literal_code_identity("an-earlier-commit")
    spec = "1" * 64
    conf = historical["file_digests"]["config/project.yaml"]
    source = historical["source_digest"]

    current = code_identity()
    assert source != current.source_digest
    assert conf != config_identity().sha256
    assert historical["file_digests"] != current.file_digests

    identity = identity_suffix(spec, conf, source)
    payload = synthetic_manifest(
        run_id=f"20200101T000000.000000Z-acquire-{identity}-0f0f0f",
        mode="acquire",
        created_at_utc="2020-01-01T00:00:00Z",
        spec_sha256=spec,
        config_sha256=conf,
        code_identity=historical,
    ).to_dict()
    assert validate_manifest_mapping(payload)
    assert validate_manifest_complete(payload)


def test_s1_25_build_manifest_rejects_a_naive_timestamp(config) -> None:
    """Also when the caller passes its own run ID."""
    with pytest.raises(ArtifactError, match="timezone aware"):
        build_manifest(STAGE, "config-check", config, moment=datetime(2026, 9, 3, 22, 15))
    with pytest.raises(ArtifactError, match="timezone aware"):
        build_manifest(
            STAGE,
            "config-check",
            config,
            run_id=new_run_id("config-check", config, moment=FIXED_MOMENT),
            moment=datetime(2026, 9, 3, 22, 15),
        )


def test_s1_25_build_manifest_rejects_a_non_datetime_moment(config) -> None:
    with pytest.raises(ArtifactError, match="must be a datetime"):
        build_manifest(STAGE, "config-check", config, moment="2026-09-03T22:15:00Z")


def test_s1_25_build_manifest_converts_a_non_utc_timestamp(config) -> None:
    from datetime import timedelta

    tokyo = timezone(timedelta(hours=9))
    manifest = build_manifest(
        STAGE, "config-check", config, moment=FIXED_MOMENT.astimezone(tokyo)
    )
    assert manifest.created_at_utc == "2026-09-03T22:15:00Z"


def test_s1_25_build_manifest_rejects_an_inconsistent_run_id(config) -> None:
    with pytest.raises(ArtifactError, match="disagrees with the recorded mode"):
        build_manifest(
            STAGE,
            "features",
            config,
            run_id=new_run_id("config-check", config, moment=FIXED_MOMENT),
            moment=FIXED_MOMENT,
        )


# Write-once files


def test_s1_25_first_write_succeeds_then_second_write_fails(temp_project_root) -> None:
    manifest = synthetic_manifest()
    target = manifest.write("outputs/runs/manifest.json")
    assert target.is_file()
    assert target.is_relative_to(temp_project_root)

    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        manifest.write("outputs/runs/manifest.json")


def test_s1_25_failed_overwrite_leaves_the_original_untouched(temp_project_root) -> None:
    original = synthetic_manifest(stage="1. Environment and configuration")
    target = original.write("outputs/runs/manifest.json")
    before = target.read_text(encoding="utf-8")

    replacement = synthetic_manifest(stage="tampered", row_count=999)
    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        replacement.write("outputs/runs/manifest.json")

    assert target.read_text(encoding="utf-8") == before
    assert json.loads(before)["stage"] == "1. Environment and configuration"
    assert json.loads(before)["row_count"] == 1000


def test_s1_25_a_file_appearing_after_resolution_is_not_overwritten(
    temp_project_root, monkeypatch
) -> None:
    """The protection comes from the exclusive create, not an earlier check.

    ``paths.resolve`` is patched to create the target first, as another
    process winning the race would. A check-then-write would destroy that
    file; an ``x``-mode create cannot.
    """
    real_resolve = paths.resolve

    def resolve_then_lose_the_race(*parts):
        target = real_resolve(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("written by another process\n", encoding="utf-8")
        return target

    monkeypatch.setattr(paths, "resolve", resolve_then_lose_the_race)
    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        synthetic_manifest().write("outputs/runs/manifest.json")

    monkeypatch.undo()
    assert (temp_project_root / "outputs" / "runs" / "manifest.json").read_text(
        encoding="utf-8"
    ) == "written by another process\n"


def test_s1_25_create_exclusive_refuses_an_existing_file(temp_project_root) -> None:
    relative = "outputs/runs/evidence.json"
    assert create_exclusive(relative, "first\n").is_file()
    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        create_exclusive(relative, "second\n")
    assert (temp_project_root / relative).read_text(encoding="utf-8") == "first\n"


def test_s1_25_create_exclusive_requires_serialised_content(temp_project_root) -> None:
    """Content is serialised by the caller, before any file exists."""
    with pytest.raises(ArtifactError, match="serialised to a string"):
        create_exclusive("outputs/runs/evidence.json", {"not": "serialised"})  # type: ignore[arg-type]
    assert not (temp_project_root / "outputs" / "runs" / "evidence.json").exists()


def test_s1_25_create_exclusive_bytes_preserves_the_payload(temp_project_root) -> None:
    """A source archive is stored byte for byte, with no text translation."""
    payload = b"PK\x03\x04binary\r\n\x00\xffbytes"
    target = create_exclusive_bytes("data/raw/archive.zip", payload)
    assert target.read_bytes() == payload
    assert target.is_relative_to(temp_project_root)


def test_s1_25_create_exclusive_bytes_refuses_an_existing_file(temp_project_root) -> None:
    create_exclusive_bytes("data/raw/archive.zip", b"first")
    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        create_exclusive_bytes("data/raw/archive.zip", b"second")
    assert (temp_project_root / "data" / "raw" / "archive.zip").read_bytes() == b"first"


def test_s1_25_create_exclusive_bytes_rejects_text_and_traversal(temp_project_root) -> None:
    with pytest.raises(ArtifactError, match="must be bytes"):
        create_exclusive_bytes("data/raw/archive.zip", "not bytes")  # type: ignore[arg-type]
    with pytest.raises(paths.ProjectLayoutError, match="parent traversal"):
        create_exclusive_bytes("data/raw/../../outside.zip", b"payload")
    assert not (temp_project_root / "data" / "raw" / "archive.zip").exists()
    assert not list(temp_project_root.parent.glob("outside*"))


@pytest.mark.parametrize(
    "relative",
    (
        "../outside/manifest.json",
        "outputs/../../outside/manifest.json",
        "outputs/runs/../../../manifest.json",
    ),
)
def test_s1_25_manifest_write_rejects_parent_traversal(temp_project_root, relative) -> None:
    with pytest.raises(paths.ProjectLayoutError, match="parent traversal"):
        synthetic_manifest().write(relative)
    assert not list(temp_project_root.parent.glob("outside*"))


@pytest.mark.parametrize("absolute", ("/tmp/manifest.json", "~/manifest.json"))
def test_s1_25_manifest_write_rejects_an_absolute_target(temp_project_root, absolute) -> None:
    with pytest.raises(paths.ProjectLayoutError, match="absolute"):
        synthetic_manifest().write(absolute)


def test_s1_25_manifest_round_trips_through_json(temp_project_root) -> None:
    manifest = synthetic_manifest()
    manifest.write("outputs/runs/manifest.json")
    restored = read_manifest("outputs/runs/manifest.json")
    assert restored == manifest.to_dict()
    assert restored["created_at_utc"] == "2026-09-03T22:15:00Z"


def test_s1_25_manifest_write_creates_missing_parents(temp_project_root) -> None:
    target = synthetic_manifest().write("outputs/runs/nested/deeper/manifest.json")
    assert target.is_file()
    assert validate_manifest_mapping(json.loads(target.read_text(encoding="utf-8")))


# Only complete manifests are saved


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"output_checksum": None}, "output_checksum"),
        ({"row_count": None}, "row_count"),
        ({"column_schema": []}, "column_schema"),
    ),
)
def test_s1_25_an_incomplete_manifest_cannot_be_persisted(
    temp_project_root, overrides, expected
) -> None:
    """A draft is valid in memory but cannot be written to disk."""
    draft = synthetic_manifest(**overrides)
    assert validate_manifest_mapping(draft.to_dict())
    assert draft.is_complete() is False

    with pytest.raises(ArtifactError, match=expected):
        validate_manifest_complete(draft.to_dict())
    with pytest.raises(ArtifactError, match="incomplete manifest"):
        draft.write("outputs/runs/manifest.json")
    assert not (temp_project_root / "outputs" / "runs" / "manifest.json").exists()


def test_s1_25_a_fully_placeholder_draft_names_every_missing_field(temp_project_root) -> None:
    draft = synthetic_draft()
    with pytest.raises(ArtifactError) as error:
        draft.write("outputs/runs/manifest.json")
    message = str(error.value)
    for name in ("output_checksum", "row_count", "column_schema"):
        assert name in message
    assert not (temp_project_root / "outputs" / "runs").exists()


def test_s1_25_a_completed_draft_can_be_persisted(temp_project_root) -> None:
    finished = synthetic_draft().completed("4" * 64, 0, ["open_time"])
    target = finished.write("outputs/runs/manifest.json")
    assert read_manifest("outputs/runs/manifest.json")["row_count"] == 0
    assert target.is_file()


def test_s1_25_read_manifest_rejects_incomplete_persisted_evidence(temp_project_root) -> None:
    """A hand-written incomplete manifest is rejected on read."""
    content = json.dumps(synthetic_draft().to_dict(), indent=2, sort_keys=True) + "\n"
    create_exclusive("outputs/runs/manifest.json", content)
    with pytest.raises(ArtifactError, match="incomplete manifest"):
        read_manifest("outputs/runs/manifest.json")


def test_s1_25_read_manifest_rejects_a_tampered_persisted_manifest(temp_project_root) -> None:
    payload = synthetic_manifest().to_dict()
    payload["row_count"] = 999_999
    payload["code_identity"] = {**payload["code_identity"], "source_digest": "5" * 64}
    create_exclusive(
        "outputs/runs/manifest.json", json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ArtifactError, match="'source_digest'.*does not match"):
        read_manifest("outputs/runs/manifest.json")


def test_s1_25_build_manifest_can_produce_persistable_evidence(
    config, monkeypatch, tmp_path
) -> None:
    """Built from the real project, then written into an isolated root.

    The real identities are measured first; the isolated root is switched in
    only for the write.
    """
    manifest = build_manifest(
        STAGE,
        "config-check",
        config,
        moment=FIXED_MOMENT,
        output_checksum="6" * 64,
        row_count=1,
        column_schema=["configuration_version"],
    )
    assert manifest.is_complete() is True
    assert manifest.code_identity["source_digest"] == code_identity().source_digest

    root = build_temp_project_root(tmp_path)
    monkeypatch.setattr(paths, "project_root", lambda: root)
    target = manifest.write_for_run("environment", config)
    restored = read_manifest(target.relative_to(root))
    assert restored["stage"] == STAGE
    assert restored["run_id"] == manifest.run_id


def test_s1_25_write_for_run_uses_the_canonical_location(temp_project_root, config) -> None:
    manifest = synthetic_manifest()
    target = manifest.write_for_run("environment", config)
    expected = temp_project_root / "outputs" / "runs" / VALID_RUN_ID / "environment.manifest.json"
    assert target == expected.resolve()


def test_s1_25_run_locations_reject_a_malformed_run_id(config) -> None:
    with pytest.raises(ArtifactError, match="malformed run_id"):
        run_relative_dir("not-a-run-id", config)
    with pytest.raises(ArtifactError, match="malformed run_id"):
        manifest_relative_path("not-a-run-id", "environment", config)


def test_s1_25_manifest_slug_must_be_a_safe_name(config) -> None:
    with pytest.raises(ArtifactError, match="stage slug"):
        manifest_relative_path(VALID_RUN_ID, "../escape", config)


def test_s1_25_manifest_is_immutable(config) -> None:
    manifest = build_manifest(STAGE, "config-check", config, moment=FIXED_MOMENT)
    with pytest.raises(Exception):
        manifest.run_id = "tampered"  # type: ignore[misc]
    assert isinstance(manifest, Manifest)


# Run logs


def test_s1_29_run_log_is_created_beside_the_run_manifest(temp_project_root, config) -> None:
    log = create_run_log(VALID_RUN_ID, config=config, moment=FIXED_MOMENT)
    assert log.path.is_file()
    assert log.path.parent == (temp_project_root / "outputs" / "runs" / VALID_RUN_ID).resolve()
    assert log.relative_path == run_log_relative_path(VALID_RUN_ID, "run", config)


def test_s1_29_run_log_records_structured_entries(temp_project_root, config) -> None:
    log = create_run_log(VALID_RUN_ID, config=config, moment=FIXED_MOMENT)
    log.record("gate_evaluated", moment=FIXED_MOMENT, gate="G1", status="PASS")
    entries = log.entries()
    assert [entry["event"] for entry in entries] == ["run_log_opened", "gate_evaluated"]
    assert entries[0] == {
        "run_id": VALID_RUN_ID,
        "recorded_at_utc": "2026-09-03T22:15:00Z",
        "event": "run_log_opened",
    }
    assert entries[1]["gate"] == "G1"
    assert entries[1]["status"] == "PASS"


def test_s1_29_run_log_refuses_to_reopen_an_existing_log(temp_project_root, config) -> None:
    create_run_log(VALID_RUN_ID, config=config, moment=FIXED_MOMENT)
    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        create_run_log(VALID_RUN_ID, config=config, moment=FIXED_MOMENT)


def test_s1_29_run_log_creation_is_atomic(temp_project_root, config, monkeypatch) -> None:
    """A log appearing after resolution is not truncated."""
    real_resolve = paths.resolve

    def resolve_then_lose_the_race(*parts):
        target = real_resolve(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("earlier evidence\n", encoding="utf-8")
        return target

    monkeypatch.setattr(paths, "resolve", resolve_then_lose_the_race)
    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        create_run_log(VALID_RUN_ID, config=config, moment=FIXED_MOMENT)

    monkeypatch.undo()
    log_path = temp_project_root / run_log_relative_path(VALID_RUN_ID, "run", config)
    assert log_path.read_text(encoding="utf-8") == "earlier evidence\n"


def test_s1_29_a_refused_log_creation_writes_nothing(temp_project_root, config) -> None:
    """A rejected opening entry leaves no empty file behind."""
    with pytest.raises(ArtifactError, match="timezone aware"):
        create_run_log(VALID_RUN_ID, config=config, moment=datetime(2026, 9, 3, 22, 15))
    assert not (temp_project_root / run_log_relative_path(VALID_RUN_ID, "run", config)).exists()


def test_s1_29_a_new_log_is_never_empty(temp_project_root, config) -> None:
    log = create_run_log(VALID_RUN_ID, config=config, moment=FIXED_MOMENT)
    assert log.path.read_text(encoding="utf-8").endswith("\n")
    assert len(log.entries()) == 1


def test_s1_29_run_log_preserves_earlier_entries_after_a_refused_reopen(
    temp_project_root, config
) -> None:
    log = create_run_log(VALID_RUN_ID, config=config, moment=FIXED_MOMENT)
    log.record("first_event", moment=FIXED_MOMENT)
    before = log.path.read_text(encoding="utf-8")
    with pytest.raises(ArtifactError):
        create_run_log(VALID_RUN_ID, config=config, moment=FIXED_MOMENT)
    assert log.path.read_text(encoding="utf-8") == before


def test_s1_29_run_log_rejects_naive_timestamps_and_reserved_fields(
    temp_project_root, config
) -> None:
    log = create_run_log(VALID_RUN_ID, config=config, moment=FIXED_MOMENT)
    with pytest.raises(ArtifactError, match="timezone aware"):
        log.record("bad_timestamp", moment=datetime(2026, 9, 3, 22, 15))
    with pytest.raises(ArtifactError, match="reserved key"):
        log.record("shadowing", moment=FIXED_MOMENT, run_id="other")


def test_s1_29_run_log_rejects_unsafe_names(temp_project_root, config) -> None:
    with pytest.raises(ArtifactError, match="log name"):
        create_run_log(VALID_RUN_ID, "../escape", config=config, moment=FIXED_MOMENT)
    log = create_run_log(VALID_RUN_ID, config=config, moment=FIXED_MOMENT)
    with pytest.raises(ArtifactError, match="log event"):
        log.record("Bad Event", moment=FIXED_MOMENT)


def test_s1_29_two_runs_get_separate_logs(temp_project_root, config) -> None:
    other = "20260903T221501.000000Z-development-0123abcd-def456"
    first = create_run_log(VALID_RUN_ID, config=config, moment=FIXED_MOMENT)
    second = create_run_log(other, config=config, moment=FIXED_MOMENT)
    assert first.path != second.path
    assert first.path.parent != second.path.parent


# Historical protocol compatibility after editorial specification changes.


def _accepted_protocol_manifest(project_root):
    return json.loads((project_root / "outputs/runs/20260906T003831.725064Z-features-92e5ff02-878985/features.manifest.json").read_text())


def test_current_protocol_manifest_matches(config) -> None:
    payload = build_manifest(STAGE, "config-check", config).to_dict()
    assert artifacts.matches_active_protocol(payload, config)
    payload["spec_sha256"] = "0" * 64
    assert not artifacts.matches_active_protocol(payload, config)


def test_historical_protocol_survives_editorial_revision(monkeypatch, tmp_path, project_root, config) -> None:
    payload = _accepted_protocol_manifest(project_root)
    original_config = paths.config_path().read_bytes()
    root = build_temp_project_root(tmp_path)
    (root / paths.CONFIG_RELATIVE_PATH).write_bytes(original_config)
    monkeypatch.setattr(paths, "project_root", lambda: root)

    before = spec_identity(config)
    assert artifacts.matches_active_protocol(payload, config)
    document = paths.spec_path()
    document.write_text(document.read_text() + "Revised explanatory prose\n", encoding="utf-8")
    assert spec_identity(config).sha256 != before.sha256
    assert artifacts.matches_active_protocol(payload, config)
    assert payload["spec_sha256"] != spec_identity(config).sha256

    # Any further unrecognised edit still changes configuration integrity.
    (root / paths.CONFIG_RELATIVE_PATH).write_bytes(original_config + b"\n# changed\n")
    assert not artifacts.matches_active_protocol(payload, config)


@pytest.mark.parametrize("field,value", [
    ("spec_sha256", "f" * 64),
    ("spec_version", "1.6"),
    ("config_sha256", "f" * 64),
    ("config_version", "1.2.0"),
])
def test_historical_compatibility_rejects_unrecognised_identity(project_root, config, field, value) -> None:
    payload = _accepted_protocol_manifest(project_root)
    assert artifacts.matches_active_protocol(payload, config)
    payload[field] = value
    assert not artifacts.matches_active_protocol(payload, config)


def test_configuration_comment_cleanup_preserves_all_locked_values() -> None:
    from src.config import read_raw_config

    payload = json.dumps(read_raw_config(), sort_keys=True, separators=(",", ":"))
    # Fingerprint of all parsed values before the comment cleanup.
    assert hashlib.sha256(payload.encode()).hexdigest() == "e5bc7e2e08116541a5f5b451091fd209bf4b08801d82544d7dba550ed02cb8f6"


def test_historical_compatibility_rejects_parameter_edits(monkeypatch, tmp_path, project_root, config) -> None:
    payload = _accepted_protocol_manifest(project_root)
    content = paths.config_path().read_text()
    root = build_temp_project_root(tmp_path)
    assert "negative_threshold: -2.5" in content
    (root / paths.CONFIG_RELATIVE_PATH).write_text(
        content.replace("negative_threshold: -2.5", "negative_threshold: -2.0", 1)
    )
    monkeypatch.setattr(paths, "project_root", lambda: root)
    assert not artifacts.matches_active_protocol(payload, config)
