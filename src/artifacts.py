"""Run identity, checksums, manifests, and run-scoped logs.

Every saved output records a run ID, creation time, spec and config versions
and checksums, input checksums, code identity, row count, column schema and
output checksum, plus the rows excluded and eligibility changes for its stage.

Rules enforced here:

- run files are opened in ``x`` mode, so an existing file is never replaced;
- content is serialised and validated before the file is created, so a
  failure leaves no empty or partial file;
- every write target must resolve inside the project root;
- every manifest timestamp must be timezone aware;
- a manifest must agree with itself: the run-ID identity part is recomputed
  from its recorded digests, and the run-ID time must match its creation
  time to the second.

Nothing here does I/O at import time.

:func:`new_attempt_token` uses ``uuid4`` because run IDs must be unique across
processes, which a seeded stream cannot give. It never reaches any result;
analysis randomness comes from :mod:`src.rng`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

from src import paths
from src.config import ConfigNode, load_config
from src.rng import numpy_version

_CHUNK_BYTES = 1 << 20
_RUN_IDENTITY_CHARS = 8
_ATTEMPT_TOKEN_CHARS = 6

# Pipeline modes, including the configuration self-check.
EXECUTION_MODES: tuple[str, ...] = (
    "config-check",
    "acquire",
    "validate",
    "features",
    "count-only",
    "outcomes",
    "development",
    "backtest",
    "robustness",
    "freeze",
    "held-out",
    "held-out-robustness",
    "report",
)

# Identity groups. Every file that can change a result, a check, or the
# installed environment belongs to exactly one group.
IDENTITY_GROUPS: dict[str, tuple[str, ...]] = {
    "code": ("src/**/*.py",),
    "test": ("tests/**/*.py", "tests/**/*.yaml"),
    "config": ("config/**/*.yaml",),
    "dependency": ("pyproject.toml", "requirements.txt", "requirements-lock.txt"),
}

# Files without which the recorded identity would be incomplete.
REQUIRED_IDENTITY_FILES: tuple[str, ...] = (
    "config/project.yaml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
)

# Generated caches and installed environments are never part of identity.
_EXCLUDED_PATH_PARTS = frozenset(
    {".venv", "venv", "__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints"}
)

REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
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

REQUIRED_CODE_IDENTITY_FIELDS: tuple[str, ...] = (
    "git_commit",
    "git_dirty",
    "source_digest",
    "code_digest",
    "test_digest",
    "config_digest",
    "dependency_digest",
    "file_digests",
)

# The fields produced by environment_identity().
REQUIRED_ENVIRONMENT_FIELDS: tuple[str, ...] = (
    "python_version",
    "python_implementation",
    "platform",
    "numpy_version",
    "packages",
    "executable",
)

# Package versions recorded in every manifest. NumPy is required because it
# drives every random procedure.
IDENTITY_PACKAGES: tuple[str, ...] = (
    "numpy",
    "PyYAML",
    "pytest",
    "scipy",
    "statsmodels",
    "matplotlib",
)
REQUIRED_PACKAGE_VERSIONS: tuple[str, ...] = ("numpy",)

# Run ID: UTC time to the microsecond, mode, eight hex characters of the
# spec/config/code identity digest, and a six-hex attempt token that keeps two
# attempts in the same microsecond apart.
RUN_ID_PATTERN = r"^\d{8}T\d{6}\.\d{6}Z-(?P<mode>[a-z][a-z-]*[a-z])-(?P<identity>[0-9a-f]{8})-(?P<attempt>[0-9a-f]{6})$"
_RUN_ID_RE = re.compile(RUN_ID_PATTERN)
_CREATED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
_VERSION_RE = re.compile(r"^\d+\.\d+")

CREATED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
RUN_ID_INSTANT_FORMAT = "%Y%m%dT%H%M%S.%fZ"


class ArtifactError(RuntimeError):
    """Raised when identity, run, manifest, or log information is invalid."""


# Checksums


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with paths.require_file(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


# Write targets


def _create_exclusive(relative_path: str | Path, payload: bytes) -> Path:
    """Create a new project-relative file from bytes, never replacing one.

    The ``x``-mode open checks and creates in one atomic step, so an existing
    file is never replaced, even one that appears after the path was resolved.
    Content is serialised and validated before this is called, so those
    failures happen while no file exists.

    A disk failure mid-write can still leave a partial file; that run must be
    repeated under a new run ID.
    """
    target = paths.resolve(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        raise ArtifactError(
            f"Refusing to overwrite existing run-scoped evidence at "
            f"{target.relative_to(paths.project_root().resolve())}; a corrected run needs a "
            "new run ID"
        ) from None
    return target


def create_exclusive(relative_path: str | Path, content: str) -> Path:
    """Create a new text file, never replacing an existing one.

    The text is written as UTF-8 with the caller's newlines unchanged.
    """
    if not isinstance(content, str):
        raise ArtifactError("evidence content must be serialised to a string before writing")
    return _create_exclusive(relative_path, content.encode("utf-8"))


def create_exclusive_bytes(relative_path: str | Path, payload: bytes) -> Path:
    """Create a new binary file, never replacing an existing one.

    Used for downloaded archives, which must be kept byte for byte. Same
    guarantees as :func:`create_exclusive`.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise ArtifactError("evidence content must be bytes before writing")
    return _create_exclusive(relative_path, bytes(payload))


# Identities


@dataclass(frozen=True)
class FileIdentity:
    """Version string and digest of the project spec or config file."""

    relative_path: str
    version: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "relative_path": self.relative_path,
            "version": self.version,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CodeIdentity:
    """Identity of the code, tests, configuration, and pinned dependencies.

    ``git_commit`` and ``git_dirty`` are recorded when the project is a Git
    checkout. The digests are always present, so a snapshot without Git data
    still has a complete identity.
    """

    git_commit: str | None
    git_dirty: bool | None
    source_digest: str
    code_digest: str
    test_digest: str
    config_digest: str
    dependency_digest: str
    file_digests: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "source_digest": self.source_digest,
            "code_digest": self.code_digest,
            "test_digest": self.test_digest,
            "config_digest": self.config_digest,
            "dependency_digest": self.dependency_digest,
            "file_digests": dict(self.file_digests),
        }


def _is_identity_candidate(path: Path) -> bool:
    return path.is_file() and not _EXCLUDED_PATH_PARTS.intersection(path.parts)


def _discover(patterns: Iterable[str]) -> dict[str, str]:
    """Digest every file matching ``patterns``, keyed by relative POSIX path."""
    root = paths.project_root().resolve()
    found: dict[str, str] = {}
    for pattern in patterns:
        for match in root.glob(pattern):
            if not _is_identity_candidate(match):
                continue
            found[match.relative_to(root).as_posix()] = sha256_file(match)
    return dict(sorted(found.items()))


def _group_digest(file_digests: Mapping[str, str]) -> str:
    combined = "\n".join(f"{name}:{digest}" for name, digest in sorted(file_digests.items()))
    return sha256_bytes(combined.encode("utf-8"))


def _combined_source_digest(group_digests: Mapping[str, str]) -> str:
    """Fold the group digests into one source digest, in sorted group order."""
    combined = "|".join(f"{name}:{group_digests[name]}" for name in sorted(group_digests))
    return sha256_bytes(combined.encode("utf-8"))


@lru_cache(maxsize=None)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Translate an identity glob into a regex with ``Path.glob`` semantics.

    Supports only what the identity groups use: ``**`` for zero or more
    folders, ``*`` within one path segment, and literal text. Anything else
    raises an error, so this cannot silently disagree with :func:`_discover`.
    """
    compiled: list[str] = []
    segments = pattern.split("/")
    for index, segment in enumerate(segments):
        if segment == "**":
            compiled.append(r"(?:[^/]+/)*")
            continue
        if "**" in segment or any(character in segment for character in "?[]"):
            raise ArtifactError(f"unsupported identity glob pattern {pattern!r}")
        compiled.append(
            "".join(r"[^/]*" if character == "*" else re.escape(character) for character in segment)
        )
        if index != len(segments) - 1:
            compiled.append("/")
    return re.compile("^" + "".join(compiled) + "$")


def classify_identity_path(relative_path: str) -> str:
    """Name the identity group a recorded path belongs to.

    Uses the same patterns as discovery and reads no file, so it works on a
    manifest from an earlier commit.
    """
    matched = [
        group
        for group, patterns in IDENTITY_GROUPS.items()
        if any(_glob_regex(pattern).match(relative_path) for pattern in patterns)
    ]
    if not matched:
        raise ArtifactError(
            f"recorded identity path {relative_path!r} belongs to no identity group; "
            f"expected a path matching one of {sorted(IDENTITY_GROUPS)}"
        )
    if len(matched) > 1:
        raise ArtifactError(
            f"recorded identity path {relative_path!r} is classified into more than one "
            f"identity group {sorted(matched)}; groups must partition the recorded files"
        )
    return matched[0]


def reconcile_identity_digests(file_digests: Mapping[str, str]) -> dict[str, str]:
    """Group and source digests implied by a set of per-file digests.

    :func:`code_identity` records what this returns, and manifest validation
    recomputes it from the manifest's ``file_digests``. A changed file digest
    therefore cannot hide behind an unchanged group or source digest.
    """
    missing = [name for name in REQUIRED_IDENTITY_FILES if name not in file_digests]
    if missing:
        raise ArtifactError(
            f"incomplete identity: missing {missing}. These files determine validation "
            "and the reproducible environment."
        )

    grouped: dict[str, dict[str, str]] = {group: {} for group in IDENTITY_GROUPS}
    for name, digest in file_digests.items():
        grouped[classify_identity_path(name)][name] = digest

    empty = sorted(group for group, members in grouped.items() if not members)
    if empty:
        raise ArtifactError(
            f"incomplete identity: group(s) {empty} record no file, so the recorded "
            "identity cannot describe the whole project"
        )

    group_digests = {group: _group_digest(members) for group, members in grouped.items()}
    reconciled = {f"{group}_digest": digest for group, digest in group_digests.items()}
    reconciled["source_digest"] = _combined_source_digest(group_digests)
    return reconciled


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=paths.project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def code_identity() -> CodeIdentity:
    """Git state plus digests of code, tests, config, and pinned dependencies.

    Group and source digests come from :func:`reconcile_identity_digests`, the
    same function manifest validation uses.
    """
    file_digests = _discover(
        pattern for patterns in IDENTITY_GROUPS.values() for pattern in patterns
    )
    reconciled = reconcile_identity_digests(file_digests)

    commit = _git("rev-parse", "HEAD")
    dirty: bool | None = None
    if commit is not None:
        status = _git("status", "--porcelain")
        dirty = None if status is None else bool(status)

    return CodeIdentity(
        git_commit=commit,
        git_dirty=dirty,
        source_digest=reconciled["source_digest"],
        code_digest=reconciled["code_digest"],
        test_digest=reconciled["test_digest"],
        config_digest=reconciled["config_digest"],
        dependency_digest=reconciled["dependency_digest"],
        file_digests=dict(sorted(file_digests.items())),
    )


def spec_identity(config: ConfigNode | None = None) -> FileIdentity:
    cfg = config if config is not None else load_config()
    return FileIdentity(
        relative_path=cfg.meta.spec_path,
        version=cfg.meta.spec_version,
        sha256=sha256_file(paths.resolve(cfg.meta.spec_path)),
    )


def config_identity(config: ConfigNode | None = None) -> FileIdentity:
    cfg = config if config is not None else load_config()
    return FileIdentity(
        relative_path=str(paths.CONFIG_RELATIVE_PATH),
        version=cfg.meta.config_version,
        sha256=sha256_file(paths.config_path()),
    )



# Historical evidence retains its original specification and configuration hashes.
# Only these exact configuration files are recognised for that protocol: the
# original and its verified comment-only cleanup. Current identities always
# hash the actual files; saved manifests are never rewritten.
_HISTORICAL_PROTOCOL_IDENTITY = (
    "1.5",
    "a27ba6787e53b961a0a8fe12c2e0b68daadb3f509fa24993737421c8474b25e7",
    "1.1.0",
    "7b9af13823e7b2464545bc8225d0f0beb31e6992bd7822a7625aa3ded4903aa1",
)
_COMPATIBLE_HISTORICAL_CONFIGURATIONS = frozenset({
    ("1.1.0", "7b9af13823e7b2464545bc8225d0f0beb31e6992bd7822a7625aa3ded4903aa1"),
    ("1.1.0", "3982afa55b509d81de445d7885c12580b1366c1a33570bc47e98cebc7d813902"),
})


def matches_active_protocol(manifest: Mapping[str, Any], config: ConfigNode) -> bool:
    """Accept the current identity or the exact recognised historical protocol.

    The historical exception allows the verified configuration comment cleanup
    and editorial specification changes. Unrecognised configuration bytes or
    versions fail. File identities always report their actual current hashes.
    """
    spec, conf = spec_identity(config), config_identity(config)
    recorded = tuple(manifest.get(key) for key in (
        "spec_version", "spec_sha256", "config_version", "config_sha256"
    ))
    current = (spec.version, spec.sha256, conf.version, conf.sha256)
    if recorded == current:
        return True
    return (
        recorded == _HISTORICAL_PROTOCOL_IDENTITY
        and recorded[0] == spec.version
        and (conf.version, conf.sha256) in _COMPATIBLE_HISTORICAL_CONFIGURATIONS
    )


def environment_identity() -> dict[str, Any]:
    """Interpreter and package versions.

    The keys are :data:`REQUIRED_ENVIRONMENT_FIELDS`, which manifest
    validation checks.
    """
    packages: dict[str, str] = {}
    for name in IDENTITY_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover - defensive
            packages[name] = "not installed"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": numpy_version(),
        "packages": packages,
        "executable": Path(sys.executable).name,
    }


# Run identity


def identity_digest(spec_sha256: str, config_sha256: str, source_digest: str) -> str:
    """Combine the three digests into a run identity.

    Manifest validation also calls this, so a saved manifest can be rechecked
    from its own digests without the current checkout.
    """
    for value, label in (
        (spec_sha256, "spec_sha256"),
        (config_sha256, "config_sha256"),
        (source_digest, "source_digest"),
    ):
        _require_checksum(value, f"identity component {label!r}")
    combined = "|".join((spec_sha256, config_sha256, source_digest))
    return sha256_bytes(combined.encode("utf-8"))


def identity_suffix(spec_sha256: str, config_sha256: str, source_digest: str) -> str:
    """The eight-character run-ID identity component for these digests."""
    return identity_digest(spec_sha256, config_sha256, source_digest)[:_RUN_IDENTITY_CHARS]


def run_identity_digest(config: ConfigNode | None = None) -> str:
    """Combined digest of the project spec, configuration, and code."""
    cfg = config if config is not None else load_config()
    return identity_digest(
        spec_identity(cfg).sha256,
        config_identity(cfg).sha256,
        code_identity().source_digest,
    )


def new_attempt_token() -> str:
    """Six lowercase hex characters that tell run attempts apart.

    Uses ``uuid4`` because run IDs must be unique across processes. It never
    affects any result; analysis randomness comes from :mod:`src.rng`.
    """
    return uuid.uuid4().hex[:_ATTEMPT_TOKEN_CHARS]


def _require_aware(moment: datetime | None, label: str) -> datetime:
    stamp = moment if moment is not None else datetime.now(timezone.utc)
    if not isinstance(stamp, datetime):
        raise ArtifactError(f"{label} must be a datetime")
    if stamp.tzinfo is None or stamp.tzinfo.utcoffset(stamp) is None:
        raise ArtifactError(f"{label} must be timezone aware")
    return stamp.astimezone(timezone.utc)


def _require_mode(mode: str) -> str:
    if mode not in EXECUTION_MODES:
        raise ArtifactError(
            f"unknown execution mode {mode!r}; expected one of {list(EXECUTION_MODES)}"
        )
    return mode


def _require_attempt_token(attempt_token: str | None) -> str:
    token = attempt_token if attempt_token is not None else new_attempt_token()
    if not isinstance(token, str) or not re.fullmatch(
        rf"[0-9a-f]{{{_ATTEMPT_TOKEN_CHARS}}}", token
    ):
        raise ArtifactError(
            f"attempt token {token!r} must be {_ATTEMPT_TOKEN_CHARS} lowercase hex characters"
        )
    return token


def compose_run_id(mode: str, stamp: datetime, identity: str, attempt_token: str) -> str:
    """Format a run ID from an already-computed identity suffix.

    :func:`build_manifest` uses this so the run ID and the manifest body come
    from the same identity measurement and cannot disagree.
    """
    _require_mode(mode)
    _require_attempt_token(attempt_token)
    aware = _require_aware(stamp, "run timestamp")
    if not re.fullmatch(rf"[0-9a-f]{{{_RUN_IDENTITY_CHARS}}}", str(identity)):
        raise ArtifactError(
            f"run identity {identity!r} must be {_RUN_IDENTITY_CHARS} lowercase hex characters"
        )
    return f"{aware:%Y%m%dT%H%M%S}.{aware.microsecond:06d}Z-{mode}-{identity}-{attempt_token}"


def new_run_id(
    mode: str,
    config: ConfigNode | None = None,
    *,
    moment: datetime | None = None,
    attempt_token: str | None = None,
) -> str:
    """Build a new run ID.

    Format: ``YYYYMMDDTHHMMSS.ffffffZ-<mode>-<identity8>-<attempt6>``.

    The microsecond time and attempt token keep two attempts apart even when
    they start in the same second. The identity part ties the run to the spec,
    configuration and code that produced it.
    """
    _require_mode(mode)
    token = _require_attempt_token(attempt_token)
    stamp = _require_aware(moment, "run timestamp")
    identity = run_identity_digest(config)[:_RUN_IDENTITY_CHARS]
    return compose_run_id(mode, stamp, identity, token)


def validate_run_id(run_id: str) -> re.Match[str]:
    """Raise an error unless ``run_id`` has the expected format and a known mode."""
    if not isinstance(run_id, str):
        raise ArtifactError("run_id must be a string")
    match = _RUN_ID_RE.match(run_id)
    if match is None:
        raise ArtifactError(
            f"malformed run_id {run_id!r}; expected "
            "YYYYMMDDTHHMMSS.ffffffZ-<mode>-<identity8>-<attempt6>"
        )
    if match.group("mode") not in EXECUTION_MODES:
        raise ArtifactError(
            f"run_id {run_id!r} names unknown execution mode {match.group('mode')!r}"
        )
    run_id_instant(run_id)
    return match


def run_id_instant(run_id: str) -> datetime:
    """The UTC instant encoded in a run ID, to microsecond resolution."""
    try:
        stamp = datetime.strptime(run_id.split("-", 1)[0], RUN_ID_INSTANT_FORMAT)
    except ValueError as error:
        raise ArtifactError(f"run_id {run_id!r} has an unparseable instant: {error}") from None
    return stamp.replace(tzinfo=timezone.utc)


# Run-scoped locations


def run_relative_dir(run_id: str, config: ConfigNode | None = None) -> Path:
    """Project-relative folder holding one run's files."""
    validate_run_id(run_id)
    cfg = config if config is not None else load_config()
    return Path(cfg.paths.outputs_runs) / run_id


def manifest_relative_path(
    run_id: str, stage_slug: str, config: ConfigNode | None = None
) -> Path:
    """Project-relative manifest path for one stage of one run."""
    if not _SLUG_RE.match(stage_slug):
        raise ArtifactError(
            f"stage slug {stage_slug!r} must be lowercase alphanumeric with - or _ separators"
        )
    return run_relative_dir(run_id, config) / f"{stage_slug}.manifest.json"


def run_log_relative_path(
    run_id: str, name: str = "run", config: ConfigNode | None = None
) -> Path:
    """Project-relative run-log path."""
    if not _SLUG_RE.match(name):
        raise ArtifactError(
            f"log name {name!r} must be lowercase alphanumeric with - or _ separators"
        )
    return run_relative_dir(run_id, config) / f"{name}.log.jsonl"


# Manifests


@dataclass(frozen=True)
class Manifest:
    """Provenance record for one saved pipeline output.

    While a stage is running, the draft may have ``output_checksum=None``,
    ``row_count=None`` and an empty ``column_schema``, because the output does
    not exist yet; :func:`validate_manifest_mapping` accepts that.
    :meth:`write` and :func:`read_manifest` use
    :func:`validate_manifest_complete`, which rejects those placeholders.
    """

    run_id: str
    created_at_utc: str
    stage: str
    mode: str
    spec_version: str
    spec_sha256: str
    config_version: str
    config_sha256: str
    input_checksums: dict[str, str] = field(default_factory=dict)
    output_checksum: str | None = None
    code_identity: dict[str, Any] = field(default_factory=dict)
    row_count: int | None = None
    column_schema: list[str] = field(default_factory=list)
    exclusions: dict[str, int] = field(default_factory=dict)
    eligibility_changes: dict[str, int] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in REQUIRED_MANIFEST_FIELDS}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def is_complete(self) -> bool:
        """Whether this manifest describes an output that has been written."""
        return (
            self.output_checksum is not None
            and self.row_count is not None
            and bool(self.column_schema)
        )

    def completed(
        self, output_checksum: str, row_count: int, column_schema: list[str]
    ) -> Manifest:
        """A copy of this draft describing the written output."""
        return replace(
            self,
            output_checksum=output_checksum,
            row_count=row_count,
            column_schema=list(column_schema),
        )

    def write(self, relative_path: str | Path) -> Path:
        """Write the manifest to a project-relative path.

        It is validated before the file is created, and an existing file is
        never replaced.
        """
        payload = validate_manifest_complete(self.to_dict())
        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        return create_exclusive(relative_path, content)

    def write_for_run(self, stage_slug: str, config: ConfigNode | None = None) -> Path:
        """Write to this run's standard manifest path."""
        return self.write(manifest_relative_path(self.run_id, stage_slug, config))


def build_manifest(
    stage: str,
    mode: str,
    config: ConfigNode | None = None,
    *,
    run_id: str | None = None,
    moment: datetime | None = None,
    attempt_token: str | None = None,
    input_checksums: Mapping[str, str] | None = None,
    output_checksum: str | None = None,
    row_count: int | None = None,
    column_schema: list[str] | None = None,
    exclusions: Mapping[str, int] | None = None,
    eligibility_changes: Mapping[str, int] | None = None,
) -> Manifest:
    """Build a draft manifest with every provenance field filled in.

    ``moment`` must be timezone aware, even when ``run_id`` is supplied. The
    spec, config and code identities are measured once and used for both the
    run ID and the body, so they always match.

    ``output_checksum``, ``row_count`` and ``column_schema`` stay as
    placeholders unless supplied; fill them with :meth:`Manifest.completed`
    before writing.
    """
    cfg = config if config is not None else load_config()
    _require_mode(mode)
    stamp = _require_aware(moment, "manifest creation timestamp")
    spec = spec_identity(cfg)
    conf = config_identity(cfg)
    code = code_identity()
    manifest = Manifest(
        run_id=(
            run_id
            if run_id is not None
            else compose_run_id(
                mode,
                stamp,
                identity_suffix(spec.sha256, conf.sha256, code.source_digest),
                _require_attempt_token(attempt_token),
            )
        ),
        created_at_utc=stamp.strftime(CREATED_AT_FORMAT),
        stage=stage,
        mode=mode,
        spec_version=spec.version,
        spec_sha256=spec.sha256,
        config_version=conf.version,
        config_sha256=conf.sha256,
        input_checksums=dict(input_checksums or {}),
        output_checksum=output_checksum,
        code_identity=code.to_dict(),
        row_count=row_count,
        column_schema=list(column_schema or []),
        exclusions=dict(exclusions or {}),
        eligibility_changes=dict(eligibility_changes or {}),
        environment=environment_identity(),
        seed=cfg.reproducibility.seed,
    )
    validate_manifest_mapping(manifest.to_dict())
    return manifest


def _require_count_mapping(payload: Mapping[str, Any], name: str) -> None:
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ArtifactError(f"manifest field {name!r} has a non-string key {key!r}")
        if type(value) is not int:
            raise ArtifactError(
                f"manifest field {name!r} entry {key!r} must be an integer, "
                f"got {type(value).__name__}"
            )
        if value < 0:
            raise ArtifactError(f"manifest field {name!r} entry {key!r} must not be negative")


def _require_checksum(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise ArtifactError(f"{label} must be 64 lowercase hex characters, got {value!r}")


def _require_name(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{label} must be a non-empty string, got {value!r}")


def _require_relative_posix_path(value: Any, label: str) -> None:
    """A recorded path must be relative, POSIX, and free of traversal."""
    _require_name(value, label)
    if value.startswith(("/", "~", "\\")) or "\\" in value or ":" in value:
        raise ArtifactError(f"{label} must be a relative POSIX path, got {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ArtifactError(f"{label} must not contain empty or traversal segments, got {value!r}")


def _validate_code_identity(code: Mapping[str, Any]) -> None:
    code_missing = sorted(set(REQUIRED_CODE_IDENTITY_FIELDS) - set(code))
    if code_missing:
        raise ArtifactError(f"manifest code_identity is missing field(s) {code_missing}")
    code_unknown = sorted(set(code) - set(REQUIRED_CODE_IDENTITY_FIELDS))
    if code_unknown:
        raise ArtifactError(f"manifest code_identity has unknown field(s) {code_unknown}")

    for name in (
        "source_digest",
        "code_digest",
        "test_digest",
        "config_digest",
        "dependency_digest",
    ):
        _require_checksum(code[name], f"manifest code_identity {name!r}")

    commit = code["git_commit"]
    if commit is not None and (
        not isinstance(commit, str) or not _GIT_COMMIT_RE.match(commit)
    ):
        raise ArtifactError(
            "manifest code_identity 'git_commit' must be null or 40 lowercase hex "
            f"characters, got {commit!r}"
        )
    dirty = code["git_dirty"]
    if dirty is not None and type(dirty) is not bool:
        raise ArtifactError(
            f"manifest code_identity 'git_dirty' must be null or a boolean, got {dirty!r}"
        )
    if commit is None and dirty is not None:
        raise ArtifactError(
            "manifest code_identity records 'git_dirty' without a 'git_commit'"
        )

    digests = code["file_digests"]
    if not isinstance(digests, Mapping) or not digests:
        raise ArtifactError("manifest code_identity 'file_digests' must be a non-empty mapping")
    for name, digest in digests.items():
        _require_relative_posix_path(name, "manifest code_identity file path")
        _require_checksum(digest, f"manifest code_identity digest for {name!r}")

    # The recorded group and source digests must follow from the recorded file
    # digests. Only manifest values are used, never the current checkout.
    reconciled = reconcile_identity_digests(digests)
    for name, expected in reconciled.items():
        if code[name] != expected:
            raise ArtifactError(
                f"manifest code_identity {name!r} {code[name]!r} does not match {expected!r} "
                "recomputed from the manifest's own file_digests"
            )


def _validate_environment(environment: Mapping[str, Any]) -> None:
    if not environment:
        raise ArtifactError("manifest field 'environment' must not be empty")
    missing = sorted(set(REQUIRED_ENVIRONMENT_FIELDS) - set(environment))
    if missing:
        raise ArtifactError(f"manifest environment is missing field(s) {missing}")
    unknown = sorted(set(environment) - set(REQUIRED_ENVIRONMENT_FIELDS))
    if unknown:
        raise ArtifactError(f"manifest environment has unknown field(s) {unknown}")

    for name in ("python_version", "python_implementation", "platform", "executable"):
        _require_name(environment[name], f"manifest environment {name!r}")
    for name in ("python_version", "numpy_version"):
        value = environment[name]
        _require_name(value, f"manifest environment {name!r}")
        if not _VERSION_RE.match(value):
            raise ArtifactError(
                f"manifest environment {name!r} must look like a version, got {value!r}"
            )

    packages = environment["packages"]
    if not isinstance(packages, Mapping) or not packages:
        raise ArtifactError("manifest environment 'packages' must be a non-empty mapping")
    for name, version in packages.items():
        _require_name(name, "manifest environment package name")
        _require_name(version, f"manifest environment version for {name!r}")
    package_missing = sorted(set(REQUIRED_PACKAGE_VERSIONS) - set(packages))
    if package_missing:
        raise ArtifactError(
            f"manifest environment 'packages' must record version(s) for {package_missing}"
        )
    if environment["numpy_version"] != packages["numpy"]:
        raise ArtifactError(
            f"manifest environment records two different NumPy versions: "
            f"numpy_version {environment['numpy_version']!r} and "
            f"packages['numpy'] {packages['numpy']!r}"
        )


def validate_manifest_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Raise an error unless a manifest draft is well formed and consistent.

    Checks fields, types, ID and checksum formats, counts, and that the run
    ID, recorded digests and creation time agree. It does not read the output
    file; later stages check checksums and row counts against files.

    ``output_checksum``, ``row_count`` and ``column_schema`` may still be
    placeholders; use :func:`validate_manifest_complete` for a manifest being
    written or read back.
    """
    if not isinstance(payload, Mapping):
        raise ArtifactError("manifest must be a mapping")
    missing = sorted(set(REQUIRED_MANIFEST_FIELDS) - set(payload))
    if missing:
        raise ArtifactError(f"manifest is missing required field(s) {missing}")
    unknown = sorted(set(payload) - set(REQUIRED_MANIFEST_FIELDS))
    if unknown:
        raise ArtifactError(f"manifest has unknown field(s) {unknown}")

    for name in (
        "run_id",
        "created_at_utc",
        "stage",
        "mode",
        "spec_version",
        "spec_sha256",
        "config_version",
        "config_sha256",
    ):
        if not isinstance(payload[name], str) or not payload[name]:
            raise ArtifactError(f"manifest field {name!r} must be a non-empty string")
    for name in (
        "input_checksums",
        "code_identity",
        "exclusions",
        "eligibility_changes",
        "environment",
    ):
        if not isinstance(payload[name], Mapping):
            raise ArtifactError(f"manifest field {name!r} must be a mapping")

    run_match = validate_run_id(payload["run_id"])
    if payload["mode"] not in EXECUTION_MODES:
        raise ArtifactError(
            f"manifest field 'mode' {payload['mode']!r} is not a known execution mode"
        )
    if run_match.group("mode") != payload["mode"]:
        raise ArtifactError("manifest run_id mode disagrees with the recorded mode")

    if not _CREATED_AT_RE.match(payload["created_at_utc"]):
        raise ArtifactError(
            "manifest field 'created_at_utc' must be 'YYYY-MM-DDTHH:MM:SSZ', "
            f"got {payload['created_at_utc']!r}"
        )
    try:
        created = datetime.strptime(payload["created_at_utc"], CREATED_AT_FORMAT).replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ArtifactError(f"manifest field 'created_at_utc' is unparseable: {error}") from None

    _require_checksum(payload["spec_sha256"], "manifest field 'spec_sha256'")
    _require_checksum(payload["config_sha256"], "manifest field 'config_sha256'")
    for key, value in payload["input_checksums"].items():
        _require_name(key, "manifest input checksum name")
        _require_checksum(value, f"manifest input checksum for {key!r}")
    if payload["output_checksum"] is not None:
        _require_checksum(payload["output_checksum"], "manifest field 'output_checksum'")

    if not isinstance(payload["column_schema"], list):
        raise ArtifactError("manifest field 'column_schema' must be a list")
    for column in payload["column_schema"]:
        if not isinstance(column, str) or not column:
            raise ArtifactError(
                f"manifest field 'column_schema' must hold non-empty strings, got {column!r}"
            )
    if len(set(payload["column_schema"])) != len(payload["column_schema"]):
        raise ArtifactError("manifest field 'column_schema' must not repeat a column name")

    if payload["row_count"] is not None:
        if type(payload["row_count"]) is not int:
            raise ArtifactError("manifest field 'row_count' must be an integer or null")
        if payload["row_count"] < 0:
            raise ArtifactError("manifest field 'row_count' must not be negative")

    _require_count_mapping(payload["exclusions"], "exclusions")
    _require_count_mapping(payload["eligibility_changes"], "eligibility_changes")

    if type(payload["seed"]) is not int or payload["seed"] < 0:
        raise ArtifactError("manifest field 'seed' must be a non-negative integer")

    code = payload["code_identity"]
    _validate_code_identity(code)
    _validate_environment(payload["environment"])

    # The checks below use only the manifest's own fields, so a manifest from
    # an earlier commit still verifies.
    recorded_config_digest = code["file_digests"].get(str(paths.CONFIG_RELATIVE_PATH.as_posix()))
    if payload["config_sha256"] != recorded_config_digest:
        raise ArtifactError(
            f"manifest field 'config_sha256' {payload['config_sha256']!r} does not match the "
            f"recorded digest {recorded_config_digest!r} for "
            f"{paths.CONFIG_RELATIVE_PATH.as_posix()} in code_identity.file_digests"
        )

    expected_suffix = identity_suffix(
        payload["spec_sha256"], payload["config_sha256"], code["source_digest"]
    )
    if run_match.group("identity") != expected_suffix:
        raise ArtifactError(
            "manifest run_id identity component "
            f"{run_match.group('identity')!r} does not match the identity "
            f"{expected_suffix!r} recomputed from the manifest's own spec_sha256, "
            "config_sha256, and code_identity.source_digest"
        )

    run_instant = run_id_instant(payload["run_id"])
    if run_instant.replace(microsecond=0) != created:
        raise ArtifactError(
            f"manifest run_id instant {run_instant.isoformat()} and created_at_utc "
            f"{payload['created_at_utc']} describe different UTC seconds"
        )

    return dict(payload)


def validate_manifest_complete(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Raise an error unless a manifest is complete enough to save.

    A saved manifest must record an output checksum, a row count and a column
    schema, so draft placeholders are rejected here.
    """
    validated = validate_manifest_mapping(payload)
    placeholders = []
    if validated["output_checksum"] is None:
        placeholders.append("output_checksum")
    if validated["row_count"] is None:
        placeholders.append("row_count")
    if not validated["column_schema"]:
        placeholders.append("column_schema")
    if placeholders:
        raise ArtifactError(
            f"refusing to treat an incomplete manifest as finished evidence: "
            f"{placeholders} still hold draft placeholders. A persisted stage manifest "
            "must record its output checksum, row count, and column schema."
        )
    return validated


def read_manifest(relative_path: str | Path) -> dict[str, Any]:
    """Read a saved manifest; it must be complete."""
    target = paths.require_file(paths.resolve(relative_path))
    payload = json.loads(target.read_text(encoding="utf-8"))
    return validate_manifest_complete(payload)


# Run-scoped logging


def serialise_log_entry(
    run_id: str,
    event: str,
    moment: datetime | None = None,
    fields: Mapping[str, Any] | None = None,
) -> str:
    """Validate and serialise one log entry before any file is touched."""
    supplied = dict(fields or {})
    if not _SLUG_RE.match(event):
        raise ArtifactError(
            f"log event {event!r} must be lowercase alphanumeric with - or _ separators"
        )
    overlap = sorted({"run_id", "recorded_at_utc", "event"}.intersection(supplied))
    if overlap:
        raise ArtifactError(f"log fields may not override reserved key(s) {overlap}")
    stamp = _require_aware(moment, "log timestamp")
    entry: dict[str, Any] = {
        "run_id": run_id,
        "recorded_at_utc": stamp.strftime(CREATED_AT_FORMAT),
        "event": event,
        **supplied,
    }
    return json.dumps(entry, sort_keys=True) + "\n"


@dataclass(frozen=True)
class RunLog:
    """Append-only JSON-lines log for one run.

    Records structured events beside a run's manifest. Nothing is set up at
    import, no file exists until :func:`create_run_log` is called, and an
    existing log is never reopened or overwritten.
    """

    run_id: str
    relative_path: Path

    @property
    def path(self) -> Path:
        return paths.resolve(self.relative_path)

    def record(self, event: str, moment: datetime | None = None, **fields: Any) -> dict[str, Any]:
        """Append one event and return the entry that was written."""
        line = serialise_log_entry(self.run_id, event, moment, fields)
        target = paths.require_file(self.path)
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
        return json.loads(line)

    def entries(self) -> list[dict[str, Any]]:
        """Every recorded entry, in order."""
        target = paths.require_file(self.path)
        return [
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def create_run_log(
    run_id: str,
    name: str = "run",
    config: ConfigNode | None = None,
    *,
    moment: datetime | None = None,
) -> RunLog:
    """Create a run log; raise an error if this run already has one.

    The first entry is written as the file is created, so the log is never
    empty and an existing log is never reopened or truncated.
    """
    relative = run_log_relative_path(run_id, name, config)
    opening = serialise_log_entry(run_id, "run_log_opened", moment)
    create_exclusive(relative, opening)
    return RunLog(run_id=run_id, relative_path=relative)
