"""Download and verify the official Binance monthly archives.

Each archive is downloaded into memory and fully checked before it is written,
so a partial or unverified download never reaches data/raw.

Only the URL, the checksum, and the ZIP container are checked here. Parsing
and cleaning happen in validation.py; no data row is read.
"""

from __future__ import annotations

import csv
import io
import re
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import urlsplit

from src import paths
from src.artifacts import (
    CREATED_AT_FORMAT,
    RunLog,
    build_manifest,
    create_exclusive,
    create_exclusive_bytes,
    create_run_log,
    run_relative_dir,
    sha256_bytes,
    sha256_file,
)
from src.config import ConfigNode, archive_months, load_config, source_timestamp_unit

STAGE = "2. Acquisition"
MODE = "acquire"
MANIFEST_SLUG = "acquisition"
INVENTORY_NAME = "raw_inventory.csv"
RAW_README_NAME = "README.md"
MANIFEST_CHECKSUM_SUFFIX = ".sha256"

# The archive is public, so requests need no credentials.
USER_AGENT = "BuyTheWeekendDip-capstone/1.0 (+python-urllib)"

ALLOWED_HOST = "data.binance.vision"
DEFAULT_ATTEMPTS = 3
DEFAULT_TIMEOUT_SECONDS = 30.0
# Short waits between attempts; a persistent failure should be reported, not waited out.
RETRY_BACKOFF_SECONDS: tuple[float, ...] = (2.0, 4.0)
# Only these statuses are retried. Anything else, including 403 and 404, stops
# the run: a missing official file must be reported, not worked around.
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

STATUS_DOWNLOADED = "downloaded"
STATUS_REVERIFIED = "reverified"

_SHA256_LINE_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_MEMBER_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_ALLOWED_COMPRESSION = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
_ENCRYPTED_FLAG = 0x1

# The source fields the project spec asks to record, plus the checksums.
# Per-run ZIP details go in the run log instead.
INVENTORY_COLUMNS: tuple[str, ...] = (
    "period",
    "symbol",
    "market_type",
    "interval",
    "source_timezone",
    "zip_url",
    "checksum_url",
    "zip_filename",
    "checksum_filename",
    "zip_bytes",
    "checksum_bytes",
    "official_zip_sha256",
    "local_zip_sha256",
    "checksum_file_sha256",
    "expected_column_count",
    "expected_timestamp_unit",
    "accessed_at_utc",
    "verified_at_utc",
    "status",
)


class AcquisitionError(RuntimeError):
    """Raised when a source, download, or integrity requirement fails."""


# Targets


@dataclass(frozen=True)
class MonthTarget:
    """Everything one monthly archive pair is expected to be."""

    period: str
    year: int
    month: int
    zip_filename: str
    checksum_filename: str
    zip_url: str
    checksum_url: str
    zip_relative_path: Path
    checksum_relative_path: Path
    expected_csv_member: str
    expected_timestamp_unit: str


@dataclass(frozen=True)
class ZipFacts:
    """Container facts about a verified archive; no CSV content is read."""

    member_count: int
    member_name: str
    member_bytes: int
    compress_type: int


@dataclass(frozen=True)
class PairVerification:
    """Digests and container facts of one fully verified archive pair."""

    official_sha256: str
    local_sha256: str
    checksum_file_sha256: str
    zip_facts: ZipFacts


@dataclass(frozen=True)
class MonthRecord:
    """One inventory record: what was acquired and how it was verified."""

    target: MonthTarget
    zip_bytes: int
    checksum_bytes: int
    official_sha256: str
    local_sha256: str
    checksum_file_sha256: str
    accessed_at_utc: str
    verified_at_utc: str
    status: str


@dataclass(frozen=True)
class AcquisitionResult:
    """Where one acquisition run left its evidence."""

    run_id: str
    inventory_relative_path: Path
    manifest_relative_path: Path
    manifest_checksum_relative_path: Path
    log_relative_path: Path
    records: tuple[MonthRecord, ...]
    downloaded: int
    reverified: int


def archive_url(config: ConfigNode, relative_path: str) -> str:
    """Build one archive URL and check it is HTTPS under the locked base URL."""
    base = config.source.base_url.rstrip("/")
    url = f"{base}/{relative_path.lstrip('/')}"
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise AcquisitionError(f"source URL must use https, got {url!r}")
    if parts.netloc != ALLOWED_HOST:
        raise AcquisitionError(
            f"source URL host {parts.netloc!r} is not the locked source host "
            f"{ALLOWED_HOST!r}; mixing sources is not permitted"
        )
    if not url.startswith(f"{base}/"):
        raise AcquisitionError(f"source URL {url!r} is not under the locked base {base!r}")
    return url


def month_targets(config: ConfigNode) -> tuple[MonthTarget, ...]:
    """List the monthly archive pairs for the configured coverage, in order."""
    data_raw = config.paths.data_raw
    pattern = config.source.path_pattern
    suffix = config.source.checksum_suffix
    targets: list[MonthTarget] = []
    for year, month in archive_months(config):
        relative = pattern.format(year=year, month=month)
        zip_filename = relative.rsplit("/", 1)[-1]
        if not zip_filename.endswith(".zip"):
            raise AcquisitionError(
                f"source.path_pattern must name a .zip archive, got {zip_filename!r}"
            )
        checksum_filename = f"{zip_filename}{suffix}"
        first_instant = datetime(year, month, 1, tzinfo=timezone.utc)
        targets.append(
            MonthTarget(
                period=f"{year:04d}-{month:02d}",
                year=year,
                month=month,
                zip_filename=zip_filename,
                checksum_filename=checksum_filename,
                zip_url=archive_url(config, relative),
                checksum_url=archive_url(config, f"{relative}{suffix}"),
                zip_relative_path=Path(data_raw) / zip_filename,
                checksum_relative_path=Path(data_raw) / checksum_filename,
                expected_csv_member=f"{zip_filename[: -len('.zip')]}.csv",
                expected_timestamp_unit=source_timestamp_unit(config, first_instant),
            )
        )
    return tuple(targets)


# Transport


def fetch_bytes(
    url: str,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Download one file into memory, retrying only transient failures.

    A redirect to another host stops with an error, because that would mix sources.
    """
    if attempts < 1:
        raise AcquisitionError("attempts must be at least 1")
    requested = urlsplit(url)
    if requested.scheme != "https":
        raise AcquisitionError(f"refusing to fetch a non-https URL {url!r}")
    if requested.netloc != ALLOWED_HOST:
        raise AcquisitionError(
            f"refusing to fetch from host {requested.netloc!r}; only {ALLOWED_HOST!r} "
            "may serve raw data"
        )

    last_reason = ""
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                final_url = response.geturl()
                final = urlsplit(final_url)
                if final.scheme != "https" or final.netloc != ALLOWED_HOST:
                    raise AcquisitionError(
                        f"request for {url!r} ended at {final_url!r}; only HTTPS on the "
                        "locked official host may serve raw data"
                    )
                payload = response.read()
                declared = response.headers.get("Content-Length")
            if declared is None:
                return payload
            try:
                declared_length = int(declared)
            except (TypeError, ValueError):
                last_reason = f"invalid Content-Length {declared!r}"
            else:
                if declared_length < 0:
                    last_reason = f"invalid Content-Length {declared!r}"
                elif declared_length != len(payload):
                    last_reason = (
                        f"incomplete body: Content-Length {declared} but "
                        f"{len(payload)} bytes read"
                    )
                else:
                    return payload
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE_STATUS:
                raise AcquisitionError(
                    f"official source returned HTTP {error.code} for {url!r}. A missing or "
                    "unavailable official archive or checksum stops the run and must be "
                    "documented; no fallback source is permitted"
                ) from None
            last_reason = f"HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_reason = f"{type(error).__name__}: {error}"

        if attempt < attempts:
            sleep(RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)])

    raise AcquisitionError(
        f"failed to download {url!r} after {attempts} attempt(s); last failure was "
        f"{last_reason}"
    )


# Verification


def parse_checksum_file(payload: bytes, expected_filename: str) -> str:
    """Read the official digest from a companion ``.CHECKSUM`` file.

    Only the standard ``sha256sum`` line is accepted: a 64-character lowercase
    digest, whitespace, an optional ``*`` marker, and the archive filename.
    Any other layout is an error rather than something to guess at.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AcquisitionError(
            f"checksum file for {expected_filename!r} is not UTF-8 text: {error}"
        ) from None

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise AcquisitionError(
            f"checksum file for {expected_filename!r} must hold exactly one non-empty line, "
            f"found {len(lines)}"
        )
    fields = lines[0].split()
    if len(fields) != 2:
        raise AcquisitionError(
            f"checksum file for {expected_filename!r} must hold a digest and a filename, "
            f"got {lines[0]!r}"
        )

    digest, recorded_name = fields
    if not _SHA256_LINE_RE.match(digest):
        raise AcquisitionError(
            f"checksum file for {expected_filename!r} must record 64 lowercase hex "
            f"characters, got {digest!r}"
        )
    if recorded_name.startswith("*"):
        recorded_name = recorded_name[1:]
    if recorded_name != expected_filename:
        raise AcquisitionError(
            f"checksum file names {recorded_name!r} but this archive is "
            f"{expected_filename!r}; the pair does not belong together"
        )
    return digest


def parse_manifest_checksum_sidecar(payload: bytes, expected_filename: str) -> str:
    """Read the digest from a manifest's ``.sha256`` sidecar.

    The sidecar must be one ``sha256sum`` line: 64 lowercase hex characters,
    two spaces, the manifest filename, and one trailing newline.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AcquisitionError(
            f"manifest checksum sidecar for {expected_filename!r} is not UTF-8 text: {error}"
        ) from None
    expected = re.compile(rf"^([0-9a-f]{{64}})  {re.escape(expected_filename)}\n\Z")
    match = expected.fullmatch(text)
    if match is None:
        raise AcquisitionError(
            f"manifest checksum sidecar must be '{{digest}}  {expected_filename}\\n', "
            f"got {text!r}"
        )
    return match.group(1)


def manifest_checksum_relative_path(manifest_relative: str | Path) -> Path:
    """Sidecar path beside a completed acquisition manifest."""
    relative = Path(manifest_relative)
    return relative.parent / f"{relative.name}{MANIFEST_CHECKSUM_SUFFIX}"


def write_manifest_checksum(manifest_relative: str | Path) -> tuple[Path, str]:
    """Hash the saved manifest and write its sidecar, refusing to overwrite."""
    relative = Path(manifest_relative)
    digest = sha256_file(paths.resolve(relative))
    sidecar = manifest_checksum_relative_path(relative)
    create_exclusive(sidecar, f"{digest}  {relative.name}\n")
    return sidecar, digest


def verify_manifest_checksum(
    manifest_relative: str | Path,
    sidecar_relative: str | Path | None = None,
) -> str:
    """Stop with an error unless the sidecar names this manifest and matches its bytes."""
    relative = Path(manifest_relative)
    sidecar = (
        Path(sidecar_relative)
        if sidecar_relative is not None
        else manifest_checksum_relative_path(relative)
    )
    recorded = parse_manifest_checksum_sidecar(
        paths.require_file(paths.resolve(sidecar)).read_bytes(),
        relative.name,
    )
    actual = sha256_file(paths.resolve(relative))
    if recorded != actual:
        raise AcquisitionError(
            f"manifest checksum sidecar records {recorded} but "
            f"{relative.as_posix()} digests to {actual}"
        )
    return actual


def inspect_zip_container(payload: bytes, target: MonthTarget) -> ZipFacts:
    """Check the ZIP container without reading any data row.

    Checks the CRC, that there is one safely named CSV member, and that it is
    not encrypted or oddly compressed.
    """
    buffer = io.BytesIO(payload)
    if not zipfile.is_zipfile(buffer):
        raise AcquisitionError(
            f"downloaded archive for {target.period} is not a valid ZIP container"
        )
    buffer.seek(0)
    try:
        with zipfile.ZipFile(buffer) as archive:
            members = archive.infolist()
            if len(members) != 1:
                raise AcquisitionError(
                    f"archive for {target.period} must hold exactly one member, "
                    f"found {len(members)}: {sorted(info.filename for info in members)}"
                )
            info = members[0]
            if not _SAFE_MEMBER_RE.match(info.filename):
                raise AcquisitionError(
                    f"archive for {target.period} holds an unsafe member path "
                    f"{info.filename!r}; a member may not be absolute, traverse, or name a "
                    "directory"
                )
            if info.filename != target.expected_csv_member:
                raise AcquisitionError(
                    f"archive for {target.period} holds member {info.filename!r} but "
                    f"{target.expected_csv_member!r} was expected"
                )
            if info.flag_bits & _ENCRYPTED_FLAG:
                raise AcquisitionError(
                    f"archive member {info.filename!r} is encrypted; the official archive is not"
                )
            if info.compress_type not in _ALLOWED_COMPRESSION:
                raise AcquisitionError(
                    f"archive member {info.filename!r} uses unexpected compression method "
                    f"{info.compress_type}"
                )
            if info.file_size <= 0:
                raise AcquisitionError(f"archive member {info.filename!r} is empty")
            damaged = archive.testzip()
            if damaged is not None:
                raise AcquisitionError(
                    f"archive member {damaged!r} for {target.period} fails its CRC check"
                )
            return ZipFacts(
                member_count=len(members),
                member_name=info.filename,
                member_bytes=info.file_size,
                compress_type=info.compress_type,
            )
    except zipfile.BadZipFile as error:
        raise AcquisitionError(
            f"downloaded archive for {target.period} is not a valid ZIP container: {error}"
        ) from None


def verify_pair(
    zip_bytes: bytes, checksum_bytes: bytes, target: MonthTarget
) -> PairVerification:
    """Verify one archive pair before anything is written."""
    official = parse_checksum_file(checksum_bytes, target.zip_filename)
    local = sha256_bytes(zip_bytes)
    if local != official:
        raise AcquisitionError(
            f"SHA-256 mismatch for {target.zip_filename}: official checksum {official} but "
            f"the received bytes digest to {local}"
        )
    facts = inspect_zip_container(zip_bytes, target)
    return PairVerification(
        official_sha256=official,
        local_sha256=local,
        checksum_file_sha256=sha256_bytes(checksum_bytes),
        zip_facts=facts,
    )


# Storage and resumption


def read_stored_pair(target: MonthTarget) -> tuple[bytes, bytes] | None:
    """Read a stored pair, or stop with an error if only one file is present.

    ``None`` means neither file exists, so the month may be downloaded.
    """
    zip_path = paths.resolve(target.zip_relative_path)
    checksum_path = paths.resolve(target.checksum_relative_path)
    has_zip = zip_path.is_file()
    has_checksum = checksum_path.is_file()
    if not has_zip and not has_checksum:
        return None
    if has_zip != has_checksum:
        present, missing = (
            (target.zip_filename, target.checksum_filename)
            if has_zip
            else (target.checksum_filename, target.zip_filename)
        )
        raise AcquisitionError(
            f"raw snapshot holds {present!r} for {target.period} but not {missing!r}. An "
            "incomplete pair is never completed silently: inspect it, remove the stray file, "
            "and rerun so both files are downloaded and verified together"
        )
    return zip_path.read_bytes(), checksum_path.read_bytes()


def store_pair(zip_bytes: bytes, checksum_bytes: bytes, target: MonthTarget) -> None:
    """Write a verified pair, never replacing an accepted raw file."""
    create_exclusive_bytes(target.zip_relative_path, zip_bytes)
    create_exclusive_bytes(target.checksum_relative_path, checksum_bytes)


def _utc_stamp(moment: datetime | None) -> str:
    stamp = moment if moment is not None else datetime.now(timezone.utc)
    if stamp.tzinfo is None or stamp.tzinfo.utcoffset(stamp) is None:
        raise AcquisitionError("acquisition timestamps must be timezone aware")
    return stamp.astimezone(timezone.utc).strftime(CREATED_AT_FORMAT)


def acquire_month(
    config: ConfigNode,
    target: MonthTarget,
    log: RunLog,
    *,
    fetch: Callable[[str], bytes] = fetch_bytes,
    moment: datetime | None = None,
) -> MonthRecord:
    """Download or re-verify one month and record the result.

    A pair already on disk is re-verified, not downloaded again or rewritten.
    """
    stored = read_stored_pair(target)
    accessed_at = ""
    if stored is None:
        zip_bytes = fetch(target.zip_url)
        checksum_bytes = fetch(target.checksum_url)
        accessed_at = _utc_stamp(moment)
        status = STATUS_DOWNLOADED
    else:
        zip_bytes, checksum_bytes = stored
        status = STATUS_REVERIFIED

    verification = verify_pair(zip_bytes, checksum_bytes, target)
    if stored is None:
        store_pair(zip_bytes, checksum_bytes, target)

    # Digest the files as they now sit on disk, so a bad write is caught too.
    stored_zip_digest = sha256_file(paths.resolve(target.zip_relative_path))
    stored_checksum_digest = sha256_file(paths.resolve(target.checksum_relative_path))
    if stored_zip_digest != verification.official_sha256:
        raise AcquisitionError(
            f"stored archive {target.zip_filename} digests to {stored_zip_digest} but the "
            f"official checksum is {verification.official_sha256}"
        )
    if stored_checksum_digest != verification.checksum_file_sha256:
        raise AcquisitionError(
            f"stored checksum file {target.checksum_filename} does not match the verified "
            "bytes"
        )

    facts = verification.zip_facts
    log.record(
        "month_downloaded" if status == STATUS_DOWNLOADED else "month_reverified",
        moment=moment,
        period=target.period,
        zip_url=target.zip_url,
        zip_bytes=len(zip_bytes),
        checksum_bytes=len(checksum_bytes),
    )
    log.record(
        "month_verified",
        moment=moment,
        period=target.period,
        official_zip_sha256=verification.official_sha256,
        local_zip_sha256=stored_zip_digest,
        checksum_file_sha256=stored_checksum_digest,
        zip_member_count=facts.member_count,
        csv_member_name=facts.member_name,
        csv_member_bytes=facts.member_bytes,
        expected_timestamp_unit=target.expected_timestamp_unit,
    )
    return MonthRecord(
        target=target,
        zip_bytes=len(zip_bytes),
        checksum_bytes=len(checksum_bytes),
        official_sha256=verification.official_sha256,
        local_sha256=stored_zip_digest,
        checksum_file_sha256=stored_checksum_digest,
        accessed_at_utc=accessed_at,
        verified_at_utc=_utc_stamp(moment),
        status=status,
    )


# Inventory and raw-data README


def inventory_columns() -> tuple[str, ...]:
    """Inventory columns, also used as the manifest column schema."""
    return INVENTORY_COLUMNS


def _inventory_row(config: ConfigNode, record: MonthRecord) -> dict[str, object]:
    target = record.target
    return {
        "period": target.period,
        "symbol": config.market.symbol,
        "market_type": config.market.market_type,
        "interval": config.market.interval,
        "source_timezone": config.market.timezone,
        "zip_url": target.zip_url,
        "checksum_url": target.checksum_url,
        "zip_filename": target.zip_filename,
        "checksum_filename": target.checksum_filename,
        "zip_bytes": record.zip_bytes,
        "checksum_bytes": record.checksum_bytes,
        "official_zip_sha256": record.official_sha256,
        "local_zip_sha256": record.local_sha256,
        "checksum_file_sha256": record.checksum_file_sha256,
        "expected_column_count": config.source.expected_column_count,
        "expected_timestamp_unit": target.expected_timestamp_unit,
        "accessed_at_utc": record.accessed_at_utc,
        "verified_at_utc": record.verified_at_utc,
        "status": record.status,
    }


def render_inventory(config: ConfigNode, records: Sequence[MonthRecord]) -> str:
    """Write the inventory as CSV text, in date order."""
    periods = [record.target.period for record in records]
    if len(set(periods)) != len(periods):
        raise AcquisitionError("inventory would repeat a period")
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(INVENTORY_COLUMNS),
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for record in sorted(records, key=lambda item: item.target.period):
        writer.writerow(_inventory_row(config, record))
    return buffer.getvalue()


def render_raw_readme(config: ConfigNode) -> str:
    """Describe the raw snapshot from the configuration alone.

    It has no timestamp or run ID, so every run produces the same bytes.
    """
    source = config.source
    pattern = source.path_pattern.replace("{year:04d}", "YYYY").replace("{month:02d}", "MM")
    months = archive_months(config)
    # Column names are left out so this file does not change when the
    # validation stage adds them to the configuration.
    schema_note = f"{source.expected_column_count} columns, names verified in Stage 3"
    change_at = config.timestamps.unit_change_at.strftime(CREATED_AT_FORMAT)
    lines = [
        "# Raw data snapshot",
        "",
        "Official Binance Public Data monthly spot-kline archives, preserved exactly as",
        "downloaded. These files are immutable: nothing in this project rewrites, cleans,",
        "or repairs them (PROJECT_SPEC.md section 7.4 line 232).",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Source | {source.base_url} |",
        f"| Path pattern | `{pattern}` |",
        f"| Exchange and market | {config.market.exchange}, {config.market.market_type} |",
        f"| Symbol and interval | {config.market.symbol}, {config.market.interval} |",
        f"| Source timezone | {config.market.timezone} |",
        f"| Coverage | {source.first_month[0]:04d}-{source.first_month[1]:02d} through "
        f"{source.last_month[0]:04d}-{source.last_month[1]:02d} "
        f"({len(months)} monthly archives) |",
        f"| Expected schema | {schema_note} |",
        f"| Timestamp unit | {config.timestamps.unit_before_change} before {change_at}, "
        f"{config.timestamps.unit_from_change} from it |",
        f"| Local digest | {source.local_digest_algorithm} |",
        "| REST fallback | not permitted |",
        "",
        "## Layout",
        "",
        "Each month is one pair, named by symbol, interval, and period:",
        "",
        "~~~text",
        f"{config.market.symbol}-{config.market.interval}-YYYY-MM.zip",
        f"{config.market.symbol}-{config.market.interval}-YYYY-MM.zip{source.checksum_suffix}",
        "~~~",
        "",
        "## Verification",
        "",
        "A pair is accepted only when the companion checksum file names this archive, the",
        "official SHA-256 matches the bytes received, the stored file digests to the same",
        "value, and the ZIP container holds exactly one non-empty, unencrypted, safely",
        "named CSV member that passes its CRC check. Verification happens in memory, so an",
        "unverified download never becomes a file here.",
        "",
        "## Evidence",
        "",
        "Each acquisition run writes its own inventory, log, and manifest under",
        f"`{config.paths.outputs_runs}/<run_id>/`. This file is generated by",
        "`python -m src.pipeline acquire` and re-verified byte for byte on every later run.",
        "",
    ]
    return "\n".join(lines)


def raw_readme_relative_path(config: ConfigNode) -> Path:
    return Path(config.paths.data_raw) / RAW_README_NAME


def ensure_raw_readme(config: ConfigNode) -> tuple[str, str]:
    """Write the raw-data README once, then require it to stay identical.

    The preserved snapshot may use the historical spec filename. All other
    differences stop with an error instead of being overwritten.
    """
    relative = raw_readme_relative_path(config)
    content = render_raw_readme(config)
    payload = content.encode("utf-8")
    target = paths.resolve(relative)
    if target.is_file():
        stored = target.read_bytes()
        # Accept only the exact historical reference; the rest must still match
        # the description generated from this configuration byte for byte.
        historical_payload = payload.replace(
            b"or repairs them (PROJECT_SPEC.md section 7.4 line 232).",
            b"or repairs them (01_PROJECT_SPEC.md section 7.4 line 232).",
            1,
        )
        if stored not in (payload, historical_payload):
            raise AcquisitionError(
                f"{relative.as_posix()} differs from the description this configuration "
                "generates. Raw files are not overwritten: review the difference and "
                "resolve it by hand"
            )
        return sha256_bytes(stored), "reverified"
    create_exclusive(relative, content)
    return sha256_bytes(payload), "written"


# Stage entry point


def _input_checksums(records: Sequence[MonthRecord]) -> dict[str, str]:
    """Checksums of every raw source file, keyed by project-relative path."""
    checksums: dict[str, str] = {}
    for record in records:
        checksums[record.target.zip_relative_path.as_posix()] = record.local_sha256
        checksums[record.target.checksum_relative_path.as_posix()] = record.checksum_file_sha256
    return dict(sorted(checksums.items()))


def _eligibility_changes(records: Sequence[MonthRecord], expected: int) -> dict[str, int]:
    downloaded = sum(1 for record in records if record.status == STATUS_DOWNLOADED)
    return {
        "months_expected": expected,
        "months_downloaded": downloaded,
        "months_reverified": len(records) - downloaded,
        "months_accepted": len(records),
    }


def _record_failure(log: RunLog, error: Exception, moment: datetime | None) -> None:
    """Log why a run stopped, without hiding the original error."""
    try:
        log.record(
            "acquisition_failed",
            moment=moment,
            reason=type(error).__name__,
            detail=str(error)[:500],
        )
    except Exception:  # pragma: no cover - logging must not replace the cause
        pass


def run_acquisition(
    config: ConfigNode | None = None,
    *,
    moment: datetime | None = None,
    fetch: Callable[[str], bytes] = fetch_bytes,
) -> AcquisitionResult:
    """Download or re-verify every monthly archive and save the run's records.

    Writes only to ``data/raw/`` and this run's directory and never replaces a
    stored file. The manifest is written only after every month is verified,
    so an interrupted run cannot look finished.
    """
    cfg = config if config is not None else load_config()
    if cfg.source.allow_rest_fallback:
        raise AcquisitionError(
            "source.allow_rest_fallback is true, but acquisition implements no fallback "
            "source; an unavailable official archive stops the run"
        )

    targets = month_targets(cfg)
    draft = build_manifest(STAGE, MODE, cfg, moment=moment)
    log = create_run_log(draft.run_id, config=cfg, moment=moment)
    log.record(
        "acquisition_started",
        moment=moment,
        months_expected=len(targets),
        base_url=cfg.source.base_url,
        symbol=cfg.market.symbol,
        market_type=cfg.market.market_type,
        interval=cfg.market.interval,
        first_period=targets[0].period,
        last_period=targets[-1].period,
    )

    try:
        records = [
            acquire_month(cfg, target, log, fetch=fetch, moment=moment) for target in targets
        ]
        if len(records) != len(targets):  # pragma: no cover - defensive
            raise AcquisitionError(
                f"acquired {len(records)} month(s) but {len(targets)} were expected"
            )

        readme_digest, readme_status = ensure_raw_readme(cfg)
        log.record(
            f"raw_readme_{readme_status}",
            moment=moment,
            path=raw_readme_relative_path(cfg).as_posix(),
            sha256=readme_digest,
        )

        content = render_inventory(cfg, records)
        inventory_relative = run_relative_dir(draft.run_id, cfg) / INVENTORY_NAME
        create_exclusive(inventory_relative, content)
        output_checksum = sha256_bytes(content.encode("utf-8"))
        log.record(
            "inventory_written",
            moment=moment,
            path=inventory_relative.as_posix(),
            rows=len(records),
            sha256=output_checksum,
        )

        finished = replace(
            draft,
            input_checksums=_input_checksums(records),
            eligibility_changes=_eligibility_changes(records, len(targets)),
        ).completed(output_checksum, len(records), list(INVENTORY_COLUMNS))
        manifest_relative = finished.write_for_run(MANIFEST_SLUG, cfg).relative_to(
            paths.project_root().resolve()
        )
        sidecar_relative, manifest_digest = write_manifest_checksum(manifest_relative)
        log.record(
            "manifest_checksum_written",
            moment=moment,
            path=sidecar_relative.as_posix(),
            sha256=manifest_digest,
            manifest=manifest_relative.as_posix(),
        )
        log.record(
            "acquisition_completed",
            moment=moment,
            manifest=manifest_relative.as_posix(),
            **_eligibility_changes(records, len(targets)),
        )
    except Exception as error:
        _record_failure(log, error, moment)
        raise

    downloaded = sum(1 for record in records if record.status == STATUS_DOWNLOADED)
    return AcquisitionResult(
        run_id=draft.run_id,
        inventory_relative_path=inventory_relative,
        manifest_relative_path=manifest_relative,
        manifest_checksum_relative_path=sidecar_relative,
        log_relative_path=log.relative_path,
        records=tuple(records),
        downloaded=downloaded,
        reverified=len(records) - downloaded,
    )


def summarise(result: AcquisitionResult) -> list[str]:
    """Summary lines for one completed acquisition run."""
    return [
        f"run id                  {result.run_id}",
        f"months accepted         {len(result.records)}",
        f"  downloaded            {result.downloaded}",
        f"  re-verified           {result.reverified}",
        f"raw files preserved     {2 * len(result.records)}",
        f"inventory               {result.inventory_relative_path.as_posix()}",
        f"manifest                {result.manifest_relative_path.as_posix()}",
        f"manifest checksum       {result.manifest_checksum_relative_path.as_posix()}",
        f"run log                 {result.log_relative_path.as_posix()}",
    ]
