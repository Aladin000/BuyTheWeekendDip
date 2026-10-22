"""Tests for downloading, verifying, and storing the monthly archives.

Everything runs offline: no test opens a socket. The archives are built in
memory and hold only a comment line, not market data. They test the container
and checksum logic, are never parsed, and never enter the real raw snapshot.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import acquisition, paths, pipeline
from src.acquisition import (
    INVENTORY_COLUMNS,
    RETRYABLE_STATUS,
    STATUS_DOWNLOADED,
    STATUS_REVERIFIED,
    AcquisitionError,
    MonthTarget,
    archive_url,
    ensure_raw_readme,
    fetch_bytes,
    inspect_zip_container,
    inventory_columns,
    month_targets,
    parse_checksum_file,
    parse_manifest_checksum_sidecar,
    raw_readme_relative_path,
    read_stored_pair,
    render_inventory,
    render_raw_readme,
    run_acquisition,
    store_pair,
    verify_manifest_checksum,
    verify_pair,
    write_manifest_checksum,
)
from src.artifacts import (
    ArtifactError,
    create_exclusive,
    create_exclusive_bytes,
    create_run_log,
    new_run_id,
    read_manifest,
    sha256_bytes,
)
from src.config import ConfigNode

FIXED_MOMENT = datetime(2026, 9, 4, 11, 30, 0, tzinfo=timezone.utc)
# A constant member timestamp, so a synthetic archive is byte-reproducible.
FIXED_ZIP_DATE = (1980, 1, 1, 0, 0, 0)
BASE_URL = "https://data.binance.vision"
FIRST_ZIP_URL = f"{BASE_URL}/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2019-10.zip"
LAST_ZIP_URL = f"{BASE_URL}/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-07.zip"


# Synthetic sources


def build_zip(
    *members: tuple[str, bytes], compress_type: int = zipfile.ZIP_DEFLATED
) -> bytes:
    """A ZIP container built in memory, deterministic byte for byte."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compress_type) as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_DATE)
            info.compress_type = compress_type
            archive.writestr(info, payload)
    return buffer.getvalue()


def build_checksum(digest: str, filename: str, *, marker: str = "") -> bytes:
    """A checksum file in standard ``sha256sum`` form."""
    return f"{digest}  {marker}{filename}\n".encode("utf-8")


def synthetic_pair(target: MonthTarget) -> tuple[bytes, bytes]:
    payload = f"# synthetic container for {target.period}\n".encode("utf-8")
    archive = build_zip((target.expected_csv_member, payload))
    return archive, build_checksum(hashlib.sha256(archive).hexdigest(), target.zip_filename)


class FakeSource:
    """An offline stand-in for the official archive, recording every request."""

    def __init__(self, targets, *, replace: dict[str, bytes] | None = None) -> None:
        self.payloads: dict[str, bytes] = {}
        for target in targets:
            archive, checksum = synthetic_pair(target)
            self.payloads[target.zip_url] = archive
            self.payloads[target.checksum_url] = checksum
        self.payloads.update(replace or {})
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if url not in self.payloads:  # pragma: no cover - guards a test mistake
            raise AssertionError(f"unexpected request for {url!r}")
        return self.payloads[url]


class FakeResponse:
    """The subset of an HTTP response ``fetch_bytes`` uses."""

    def __init__(self, payload: bytes, *, url: str, content_length: int | None = -1) -> None:
        self._payload = payload
        self._url = url
        declared = len(payload) if content_length == -1 else content_length
        self.headers: dict[str, str] = (
            {} if declared is None else {"Content-Length": str(declared)}
        )

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def read(self) -> bytes:
        return self._payload

    def geturl(self) -> str:
        return self._url


def fake_opener(*outcomes):
    """A urlopen replacement returning or raising each outcome in turn."""
    calls: list[str] = []

    def opener(request, timeout=None):
        calls.append(request.full_url)
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    opener.calls = calls  # type: ignore[attr-defined]
    return opener


def first_target(config) -> MonthTarget:
    return month_targets(config)[0]


def with_source(config, **overrides) -> ConfigNode:
    """The real configuration with one ``source`` value replaced.

    Frozen two levels deep, which is as far as these tests read.
    """
    data = config.to_dict()
    data["source"] = {**data["source"], **overrides}
    return ConfigNode(
        {
            key: ConfigNode(value) if isinstance(value, dict) else value
            for key, value in data.items()
        }
    )


def fresh_log(config):
    """A run log for tests that exercise one month outside a full run."""
    run_id = new_run_id("acquire", config, moment=FIXED_MOMENT)
    return create_run_log(run_id, config=config, moment=FIXED_MOMENT)


def complete_run(config, *, source: FakeSource | None = None, moment=FIXED_MOMENT):
    """Run a full acquisition against a synthetic source."""
    fetch = source if source is not None else FakeSource(month_targets(config))
    return run_acquisition(config, moment=moment, fetch=fetch), fetch


# Targets from the locked config


def test_s2_01_targets_cover_the_locked_82_months(config) -> None:
    targets = month_targets(config)
    assert len(targets) == 82
    assert targets[0].period == "2019-10"
    assert targets[-1].period == "2026-07"


def test_s2_01_first_and_last_urls_are_exact(config) -> None:
    targets = month_targets(config)
    assert targets[0].zip_url == FIRST_ZIP_URL
    assert targets[0].checksum_url == f"{FIRST_ZIP_URL}.CHECKSUM"
    assert targets[-1].zip_url == LAST_ZIP_URL
    assert targets[-1].checksum_url == f"{LAST_ZIP_URL}.CHECKSUM"


def test_s2_01_every_url_is_under_the_locked_base(config) -> None:
    for target in month_targets(config):
        for url in (target.zip_url, target.checksum_url):
            assert url.startswith(f"{config.source.base_url}/")
            assert "data.binance.vision" in url


def test_s2_01_filenames_and_raw_paths_follow_the_source(config) -> None:
    target = month_targets(config)[3]
    assert target.zip_filename == "BTCUSDT-1h-2020-01.zip"
    assert target.checksum_filename == "BTCUSDT-1h-2020-01.zip.CHECKSUM"
    assert target.expected_csv_member == "BTCUSDT-1h-2020-01.csv"
    assert target.zip_relative_path == Path("data/raw/BTCUSDT-1h-2020-01.zip")
    assert target.checksum_relative_path == Path(
        "data/raw/BTCUSDT-1h-2020-01.zip.CHECKSUM"
    )


def test_s2_01_periods_are_unique_and_chronological(config) -> None:
    periods = [target.period for target in month_targets(config)]
    assert len(set(periods)) == len(periods)
    assert periods == sorted(periods)


@pytest.mark.parametrize(
    ("period", "unit"),
    (
        ("2019-10", "milliseconds"),
        ("2024-12", "milliseconds"),
        ("2025-01", "microseconds"),
        ("2026-07", "microseconds"),
    ),
)
def test_s2_01_expected_timestamp_unit_switches_at_2025(config, period, unit) -> None:
    by_period = {target.period: target for target in month_targets(config)}
    assert by_period[period].expected_timestamp_unit == unit


@pytest.mark.parametrize(
    ("base", "expected"),
    (
        ("http://data.binance.vision", "https"),
        ("https://api.binance.com", "not the locked source host"),
    ),
)
def test_s2_01_archive_url_refuses_another_source(config, base, expected) -> None:
    """No REST endpoint, no mirror, no second host."""
    with pytest.raises(AcquisitionError, match=expected):
        archive_url(
            with_source(config, base_url=base),
            "data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2019-10.zip",
        )


def test_s2_01_run_refuses_a_configured_rest_fallback(config, temp_acquisition_root) -> None:
    with pytest.raises(AcquisitionError, match="implements no fallback"):
        run_acquisition(
            with_source(config, allow_rest_fallback=True),
            moment=FIXED_MOMENT,
            fetch=lambda url: b"",
        )


# Official checksum files


def test_s2_02_canonical_checksum_line_is_accepted() -> None:
    digest = "a" * 64
    assert parse_checksum_file(build_checksum(digest, "x.zip"), "x.zip") == digest


@pytest.mark.parametrize("marker", ("", "*"))
def test_s2_02_binary_marker_is_accepted(marker) -> None:
    digest = "b" * 64
    payload = build_checksum(digest, "x.zip", marker=marker)
    assert parse_checksum_file(payload, "x.zip") == digest


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        (b"", "exactly one non-empty line"),
        (b"   \n\n", "exactly one non-empty line"),
        (b"a" * 64 + b"  x.zip\n" + b"c" * 64 + b"  y.zip\n", "exactly one non-empty line"),
        (b"a" * 64 + b"\n", "a digest and a filename"),
        (b"x.zip\n", "a digest and a filename"),
        (b"a" * 64 + b"  x.zip  extra\n", "a digest and a filename"),
        (b"A" * 64 + b"  x.zip\n", "64 lowercase hex"),
        (b"a" * 63 + b"  x.zip\n", "64 lowercase hex"),
        (b"z" * 64 + b"  x.zip\n", "64 lowercase hex"),
        (b"a" * 64 + b"  other.zip\n", "does not belong together"),
        (b"a" * 64 + b"  klines/x.zip\n", "does not belong together"),
        (b"\xff\xfe" + b"a" * 64, "not UTF-8"),
    ),
)
def test_s2_02_malformed_checksum_files_fail_closed(payload, expected) -> None:
    with pytest.raises(AcquisitionError, match=expected):
        parse_checksum_file(payload, "x.zip")


# Byte integrity


def test_s2_03_a_verified_pair_reports_both_digests(config) -> None:
    target = first_target(config)
    archive, checksum = synthetic_pair(target)
    verification = verify_pair(archive, checksum, target)
    assert verification.official_sha256 == hashlib.sha256(archive).hexdigest()
    assert verification.local_sha256 == verification.official_sha256
    assert verification.checksum_file_sha256 == sha256_bytes(checksum)
    assert verification.zip_facts.member_name == target.expected_csv_member


def test_s2_03_digest_mismatch_fails_closed(config, temp_acquisition_root) -> None:
    target = first_target(config)
    archive, _ = synthetic_pair(target)
    other = build_checksum("d" * 64, target.zip_filename)
    with pytest.raises(AcquisitionError, match="SHA-256 mismatch"):
        verify_pair(archive, other, target)
    assert not list((temp_acquisition_root / "data" / "raw").glob("*.zip"))


def test_s2_03_a_tampered_stored_archive_fails_closed(config, temp_acquisition_root) -> None:
    """A stored ZIP that no longer matches its checksum stops with an error."""
    target = first_target(config)
    archive, checksum = synthetic_pair(target)
    store_pair(archive, checksum, target)
    (temp_acquisition_root / target.zip_relative_path).write_bytes(archive + b"\x00")
    log = fresh_log(config)
    with pytest.raises(AcquisitionError, match="SHA-256 mismatch"):
        acquisition.acquire_month(
            config, target, log, fetch=FakeSource([target]), moment=FIXED_MOMENT
        )


def test_s2_03_a_bad_stored_write_is_detected(
    config, temp_acquisition_root, monkeypatch
) -> None:
    """The stored file is digested again, so a corrupted write cannot pass."""
    target = first_target(config)
    source = FakeSource([target])
    log = fresh_log(config)
    monkeypatch.setattr(acquisition, "sha256_file", lambda path: "e" * 64)
    with pytest.raises(AcquisitionError, match="stored archive"):
        acquisition.acquire_month(config, target, log, fetch=source, moment=FIXED_MOMENT)


# ZIP container checks


def test_s2_04_a_valid_container_reports_its_member(config) -> None:
    target = first_target(config)
    archive, _ = synthetic_pair(target)
    facts = inspect_zip_container(archive, target)
    assert facts.member_count == 1
    assert facts.member_name == target.expected_csv_member
    assert facts.member_bytes > 0


def _crc_corrupted_archive(target: MonthTarget) -> bytes:
    """A structurally valid archive whose stored member data was altered."""
    payload = b"# synthetic container needing a long enough body to edit\n"
    archive = build_zip((target.expected_csv_member, payload), compress_type=zipfile.ZIP_STORED)
    offset = archive.index(payload)
    mutated = bytearray(archive)
    mutated[offset] = mutated[offset] ^ 0xFF
    return bytes(mutated)


def _encrypted_flag_archive(target: MonthTarget) -> bytes:
    """A container whose general-purpose flag claims encryption.

    ``zipfile`` clears the flag when writing, so the bit is set by hand in
    the local header and in the central directory, where ``ZipFile`` reads it.
    """
    data = bytearray(build_zip((target.expected_csv_member, b"# synthetic\n")))
    data[6] |= 0x1
    central = data.rindex(b"PK\x01\x02")
    data[central + 8] |= 0x1
    return bytes(data)


def _damaged_container(case: str, target: MonthTarget) -> bytes:
    """Build the bad archive for one rejection case."""
    member = target.expected_csv_member
    body = b"# synthetic\n"
    builders = {
        "not_a_zip": lambda: b"this is not an archive",
        "truncated": lambda: build_zip((member, body))[:-12],
        "crc_failure": lambda: _crc_corrupted_archive(target),
        "no_members": build_zip,
        "two_members": lambda: build_zip((member, body), ("extra.csv", body)),
        "wrong_member": lambda: build_zip(("other.csv", body)),
        "empty_member": lambda: build_zip((member, b"")),
        "encrypted_member": lambda: _encrypted_flag_archive(target),
        "traversing_member": lambda: build_zip((f"../{member}", body)),
        "absolute_member": lambda: build_zip((f"/{member}", body)),
        "nested_member": lambda: build_zip((f"sub/{member}", body)),
    }
    return builders[case]()


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("not_a_zip", "not a valid ZIP"),
        ("truncated", "not a valid ZIP"),
        ("crc_failure", "fails its CRC check"),
        ("no_members", "exactly one member"),
        ("two_members", "exactly one member"),
        ("wrong_member", "was expected"),
        ("empty_member", "is empty"),
        ("encrypted_member", "is encrypted"),
        ("traversing_member", "unsafe member path"),
        ("absolute_member", "unsafe member path"),
        ("nested_member", "unsafe member path"),
    ),
)
def test_s2_04_unexpected_containers_fail_closed(config, case, expected) -> None:
    target = first_target(config)
    with pytest.raises(AcquisitionError, match=expected):
        inspect_zip_container(_damaged_container(case, target), target)


# Downloads and retries


def test_s2_05_a_successful_fetch_returns_the_body(monkeypatch) -> None:
    opener = fake_opener(FakeResponse(b"payload", url=FIRST_ZIP_URL))
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    assert fetch_bytes(FIRST_ZIP_URL) == b"payload"
    assert opener.calls == [FIRST_ZIP_URL]


@pytest.mark.parametrize(
    "transient",
    (
        urllib.error.URLError("connection reset"),
        urllib.error.HTTPError(FIRST_ZIP_URL, 503, "Service Unavailable", None, None),
        TimeoutError("timed out"),
    ),
)
def test_s2_05_a_transient_failure_is_retried(monkeypatch, transient) -> None:
    waits: list[float] = []
    opener = fake_opener(transient, FakeResponse(b"payload", url=FIRST_ZIP_URL))
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    assert fetch_bytes(FIRST_ZIP_URL, sleep=waits.append) == b"payload"
    assert len(opener.calls) == 2
    assert waits == [2.0]


@pytest.mark.parametrize("status", (403, 404))
def test_s2_05_a_missing_archive_is_not_retried(monkeypatch, status) -> None:
    """A missing official file stops the run; it is never worked around."""
    assert status not in RETRYABLE_STATUS
    error = urllib.error.HTTPError(FIRST_ZIP_URL, status, "gone", None, None)
    opener = fake_opener(error)
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    with pytest.raises(AcquisitionError, match="stops the run"):
        fetch_bytes(FIRST_ZIP_URL, sleep=lambda seconds: None)
    assert len(opener.calls) == 1


def test_s2_05_the_attempt_budget_is_bounded(monkeypatch) -> None:
    waits: list[float] = []
    opener = fake_opener(urllib.error.URLError("down"))
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    with pytest.raises(AcquisitionError, match="after 3 attempt"):
        fetch_bytes(FIRST_ZIP_URL, sleep=waits.append)
    assert len(opener.calls) == 3
    assert waits == [2.0, 4.0]


def test_s2_05_a_short_body_is_treated_as_transient(monkeypatch) -> None:
    opener = fake_opener(
        FakeResponse(b"short", url=FIRST_ZIP_URL, content_length=4096),
        FakeResponse(b"complete", url=FIRST_ZIP_URL),
    )
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    assert fetch_bytes(FIRST_ZIP_URL, sleep=lambda seconds: None) == b"complete"
    assert len(opener.calls) == 2


def test_s2_05_a_redirect_off_the_locked_host_fails_closed(monkeypatch) -> None:
    opener = fake_opener(FakeResponse(b"payload", url="https://example.invalid/archive.zip"))
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    with pytest.raises(AcquisitionError, match="only HTTPS on the locked official host"):
        fetch_bytes(FIRST_ZIP_URL)


def test_s2_05_a_same_host_http_redirect_fails_closed(monkeypatch) -> None:
    opener = fake_opener(
        FakeResponse(b"payload", url="http://data.binance.vision/archive.zip")
    )
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    with pytest.raises(AcquisitionError, match="only HTTPS on the locked official host"):
        fetch_bytes(FIRST_ZIP_URL)


@pytest.mark.parametrize("declared", ("not-an-integer", -2))
def test_s2_05_an_invalid_content_length_fails_within_the_attempt_budget(
    monkeypatch, declared
) -> None:
    opener = fake_opener(
        FakeResponse(b"payload", url=FIRST_ZIP_URL, content_length=declared)
    )
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    with pytest.raises(AcquisitionError, match="invalid Content-Length"):
        fetch_bytes(FIRST_ZIP_URL, sleep=lambda seconds: None)
    assert len(opener.calls) == 3


def test_s2_05_a_non_https_url_is_refused() -> None:
    with pytest.raises(AcquisitionError, match="non-https"):
        fetch_bytes("http://data.binance.vision/archive.zip")


def test_s2_05_a_request_to_another_host_is_refused() -> None:
    with pytest.raises(AcquisitionError, match="may serve raw data"):
        fetch_bytes("https://api.binance.com/api/v3/klines")


# Write-once storage


def test_s2_06_store_pair_preserves_both_files(config, temp_acquisition_root) -> None:
    target = first_target(config)
    archive, checksum = synthetic_pair(target)
    store_pair(archive, checksum, target)
    assert (temp_acquisition_root / target.zip_relative_path).read_bytes() == archive
    assert (temp_acquisition_root / target.checksum_relative_path).read_bytes() == checksum


def test_s2_06_an_accepted_archive_is_never_replaced(config, temp_acquisition_root) -> None:
    target = first_target(config)
    archive, checksum = synthetic_pair(target)
    store_pair(archive, checksum, target)
    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        store_pair(b"different bytes", checksum, target)
    assert (temp_acquisition_root / target.zip_relative_path).read_bytes() == archive


# Resuming a run


def test_s2_07_an_existing_pair_is_reverified_without_a_download(
    config, temp_acquisition_root
) -> None:
    target = first_target(config)
    archive, checksum = synthetic_pair(target)
    store_pair(archive, checksum, target)
    log = fresh_log(config)
    source = FakeSource([target])
    record = acquisition.acquire_month(
        config, target, log, fetch=source, moment=FIXED_MOMENT
    )
    assert record.status == STATUS_REVERIFIED
    assert record.accessed_at_utc == ""
    assert source.calls == []


@pytest.mark.parametrize("keep", ("zip", "checksum"))
def test_s2_07_a_half_present_month_fails_closed(config, temp_acquisition_root, keep) -> None:
    target = first_target(config)
    archive, checksum = synthetic_pair(target)
    store_pair(archive, checksum, target)
    doomed = target.checksum_relative_path if keep == "zip" else target.zip_relative_path
    (temp_acquisition_root / doomed).unlink()
    with pytest.raises(AcquisitionError, match="but not"):
        read_stored_pair(target)


def test_s2_07_a_resumed_run_completes_the_snapshot(config, temp_acquisition_root) -> None:
    targets = month_targets(config)
    for target in targets[:5]:
        store_pair(*synthetic_pair(target), target)

    result, source = complete_run(config)
    assert len(result.records) == 82
    assert result.reverified == 5
    assert result.downloaded == 77
    # Two files per month, requested only for the 77 missing months.
    assert len(source.calls) == 154


# Inventory file


def test_s2_08_columns_are_exactly_the_documented_set(config, temp_acquisition_root) -> None:
    assert inventory_columns() == INVENTORY_COLUMNS
    assert len(INVENTORY_COLUMNS) == len(set(INVENTORY_COLUMNS))
    result, _ = complete_run(config)
    content = (temp_acquisition_root / result.inventory_relative_path).read_text(
        encoding="utf-8"
    )
    assert content.splitlines()[0] == ",".join(INVENTORY_COLUMNS)


def test_s2_08_rows_are_chronological_and_complete(config, temp_acquisition_root) -> None:
    result, _ = complete_run(config)
    lines = (temp_acquisition_root / result.inventory_relative_path).read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 83
    periods = [line.split(",")[0] for line in lines[1:]]
    assert periods == sorted(periods)
    assert periods[0] == "2019-10"
    assert periods[-1] == "2026-07"
    for line in lines[1:]:
        fields = line.split(",")
        row = dict(zip(INVENTORY_COLUMNS, fields))
        assert row["symbol"] == "BTCUSDT"
        assert row["interval"] == "1h"
        assert row["source_timezone"] == "UTC"
        assert row["official_zip_sha256"] == row["local_zip_sha256"]
        assert row["expected_column_count"] == "12"
        assert row["status"] == STATUS_DOWNLOADED


def test_s2_08_rendering_is_byte_identical_and_order_independent(
    config, temp_acquisition_root
) -> None:
    result, _ = complete_run(config)
    records = list(result.records)
    first = render_inventory(config, records)
    assert render_inventory(config, records) == first
    assert render_inventory(config, list(reversed(records))) == first


# Run manifest and log


def test_s2_09_a_completed_run_writes_reconciling_evidence(
    config, temp_acquisition_root
) -> None:
    result, _ = complete_run(config)

    manifest = read_manifest(result.manifest_relative_path)
    assert manifest["stage"] == "2. Acquisition"
    assert manifest["mode"] == "acquire"
    assert manifest["run_id"] == result.run_id
    assert manifest["row_count"] == 82
    assert manifest["column_schema"] == list(INVENTORY_COLUMNS)
    assert manifest["eligibility_changes"] == {
        "months_expected": 82,
        "months_downloaded": 82,
        "months_reverified": 0,
        "months_accepted": 82,
    }
    assert manifest["exclusions"] == {}


def test_s2_09_the_output_checksum_is_the_inventory_digest(
    config, temp_acquisition_root
) -> None:
    result, _ = complete_run(config)
    inventory = (temp_acquisition_root / result.inventory_relative_path).read_bytes()
    manifest = read_manifest(result.manifest_relative_path)
    assert manifest["output_checksum"] == sha256_bytes(inventory)


def test_s2_09_input_checksums_cover_every_preserved_file(
    config, temp_acquisition_root
) -> None:
    result, _ = complete_run(config)
    manifest = read_manifest(result.manifest_relative_path)
    checksums = manifest["input_checksums"]
    assert len(checksums) == 164
    for relative, digest in checksums.items():
        stored = temp_acquisition_root / relative
        assert relative.startswith("data/raw/")
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == digest


def test_s2_09_the_run_log_records_every_month(config, temp_acquisition_root) -> None:
    result, _ = complete_run(config)
    entries = [
        json.loads(line)
        for line in (temp_acquisition_root / result.log_relative_path)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    events = [entry["event"] for entry in entries]
    assert events[0] == "run_log_opened"
    assert events[1] == "acquisition_started"
    assert events[-1] == "acquisition_completed"
    assert events.count("month_downloaded") == 82
    assert events.count("month_verified") == 82
    verified = [entry for entry in entries if entry["event"] == "month_verified"]
    assert verified[0]["csv_member_name"] == "BTCUSDT-1h-2019-10.csv"
    assert verified[0]["expected_timestamp_unit"] == "milliseconds"
    assert verified[-1]["expected_timestamp_unit"] == "microseconds"


def test_s2_09_a_second_run_writes_separate_evidence(config, temp_acquisition_root) -> None:
    first, _ = complete_run(config)
    second, source = complete_run(config)
    assert second.run_id != first.run_id
    assert source.calls == []
    assert second.reverified == 82
    assert read_manifest(second.manifest_relative_path)["eligibility_changes"] == {
        "months_expected": 82,
        "months_downloaded": 0,
        "months_reverified": 82,
        "months_accepted": 82,
    }
    assert (temp_acquisition_root / first.manifest_relative_path).is_file()


# A failed run writes no manifest


def test_s2_10_a_failing_month_writes_no_manifest(config, temp_acquisition_root) -> None:
    targets = month_targets(config)
    source = FakeSource(targets, replace={targets[2].zip_url: b"corrupted archive"})
    with pytest.raises(AcquisitionError, match="SHA-256 mismatch"):
        run_acquisition(config, moment=FIXED_MOMENT, fetch=source)

    runs = temp_acquisition_root / "outputs" / "runs"
    assert not list(runs.glob("*/*.manifest.json"))
    assert not list(runs.glob("*/*.manifest.json.sha256"))
    assert not list(runs.glob("*/raw_inventory.csv"))
    # The two accepted months stay; the failing month was never stored.
    stored = sorted(path.name for path in (temp_acquisition_root / "data" / "raw").iterdir())
    assert stored == [
        "BTCUSDT-1h-2019-10.zip",
        "BTCUSDT-1h-2019-10.zip.CHECKSUM",
        "BTCUSDT-1h-2019-11.zip",
        "BTCUSDT-1h-2019-11.zip.CHECKSUM",
    ]


def test_s2_10_the_failure_is_recorded_in_the_run_log(config, temp_acquisition_root) -> None:
    targets = month_targets(config)
    source = FakeSource(targets, replace={targets[0].zip_url: b"corrupted archive"})
    with pytest.raises(AcquisitionError):
        run_acquisition(config, moment=FIXED_MOMENT, fetch=source)

    logs = list((temp_acquisition_root / "outputs" / "runs").glob("*/run.log.jsonl"))
    assert len(logs) == 1
    entries = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["event"] == "acquisition_failed"
    assert entries[-1]["reason"] == "AcquisitionError"
    assert "SHA-256 mismatch" in entries[-1]["detail"]


# Raw-data README


def test_s2_11_the_readme_is_written_once_and_then_reverified(
    config, temp_acquisition_root
) -> None:
    digest, status = ensure_raw_readme(config)
    assert status == "written"
    target = temp_acquisition_root / raw_readme_relative_path(config)
    content = target.read_text(encoding="utf-8")
    assert content == render_raw_readme(config)
    assert "82 monthly archives" in content
    assert "data.binance.vision" in content
    assert "(PROJECT_SPEC.md section 7.4 line 232)" in content
    assert "01_PROJECT_SPEC.md" not in content

    again_digest, again_status = ensure_raw_readme(config)
    assert (again_digest, again_status) == (digest, "reverified")
    assert target.read_text(encoding="utf-8") == content


def test_s2_11_the_preserved_snapshot_readme_is_reverified(
    config, temp_acquisition_root
) -> None:
    preserved = (Path(__file__).resolve().parents[1] / "data/raw/README.md").read_bytes()
    target = temp_acquisition_root / raw_readme_relative_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(preserved)

    digest, status = ensure_raw_readme(config)

    assert status == "reverified"
    assert digest == sha256_bytes(preserved)
    assert digest != sha256_bytes(render_raw_readme(config).encode("utf-8"))
    assert target.read_bytes() == preserved


def test_s2_11_a_modified_historical_readme_fails_closed(
    config, temp_acquisition_root
) -> None:
    preserved = (Path(__file__).resolve().parents[1] / "data/raw/README.md").read_bytes()
    modified = preserved.replace(b"82 monthly archives", b"81 monthly archives")
    assert modified != preserved
    target = temp_acquisition_root / raw_readme_relative_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(modified)

    with pytest.raises(AcquisitionError, match="differs from the description"):
        ensure_raw_readme(config)
    assert target.read_bytes() == modified


def test_s2_11_a_divergent_readme_fails_closed(config, temp_acquisition_root) -> None:
    target = temp_acquisition_root / raw_readme_relative_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hand-edited description\n", encoding="utf-8")
    with pytest.raises(AcquisitionError, match="differs from the description"):
        ensure_raw_readme(config)
    assert target.read_text(encoding="utf-8") == "hand-edited description\n"


# Pipeline integration


def test_s2_12_acquire_is_dispatched(config, temp_acquisition_root, monkeypatch, capsys) -> None:
    """The real stage runs through the pipeline; only the transport is faked."""
    real = acquisition.run_acquisition
    monkeypatch.setattr(pipeline, "load_config", lambda: config)
    monkeypatch.setattr(
        acquisition,
        "run_acquisition",
        lambda cfg: real(cfg, moment=FIXED_MOMENT, fetch=FakeSource(month_targets(cfg))),
    )
    assert pipeline.dispatch("acquire") == 0
    printed = capsys.readouterr().out
    assert "months accepted         82" in printed
    assert "raw files preserved     164" in printed
    assert "acquisition.manifest.json" in printed
    assert "acquisition.manifest.json.sha256" in printed


def test_s2_12_an_acquisition_failure_exits_two(monkeypatch, capsys) -> None:
    def refuse(*args, **kwargs):
        raise AcquisitionError("official source returned HTTP 500")

    monkeypatch.setattr(acquisition, "run_acquisition", refuse)
    assert pipeline.main(["acquire"]) == 2
    assert "official source returned HTTP 500" in capsys.readouterr().err


def test_s2_12_a_refused_overwrite_exits_two(monkeypatch, capsys) -> None:
    def refuse(*args, **kwargs):
        raise ArtifactError("Refusing to overwrite existing run-scoped evidence")

    monkeypatch.setattr(acquisition, "run_acquisition", refuse)
    assert pipeline.main(["acquire"]) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_s2_12_held_out_still_refuses_without_authorisation(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "held_out_authorised", lambda: False)
    assert pipeline.main(["held-out"]) == 2


# Import side effects


_AUDIT_PROGRAM = """
import sys

opened = []
network = []


def hook(event, args):
    if event == "open":
        opened.append(str(args[0]))
    elif event.startswith(("socket.", "urllib.")):
        network.append(event)


sys.addaudithook(hook)

import src.acquisition

offenders = sorted(
    {path for path in opened if path.endswith(("project.yaml", "01_PROJECT_SPEC.md", "PROJECT_SPEC.md"))}
    | set(network)
)
print("OFFENDERS " + repr(offenders) if offenders else "CLEAN")
"""


def test_s2_13_import_opens_no_file_and_no_socket(project_root: Path) -> None:
    """Importing the module must neither read the project nor reach the network."""
    completed = subprocess.run(
        [sys.executable, "-c", _AUDIT_PROGRAM],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(project_root)},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "CLEAN", completed.stdout


def test_s2_13_no_run_writes_outside_the_project_root(config, temp_acquisition_root) -> None:
    """Every write is project-relative and contained."""
    result, _ = complete_run(config)
    for relative in (
        result.inventory_relative_path,
        result.manifest_relative_path,
        result.manifest_checksum_relative_path,
        result.log_relative_path,
    ):
        resolved = paths.resolve(relative)
        assert resolved.is_relative_to(temp_acquisition_root)
        assert resolved.is_file()


# Manifest SHA-256 sidecar


def test_s2_14_a_completed_run_writes_an_exact_manifest_sidecar(
    config, temp_acquisition_root
) -> None:
    result, _ = complete_run(config)
    sidecar = temp_acquisition_root / result.manifest_checksum_relative_path
    manifest = temp_acquisition_root / result.manifest_relative_path
    payload = sidecar.read_bytes()
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert sidecar.name == "acquisition.manifest.json.sha256"
    assert sidecar.parent == manifest.parent
    assert payload == f"{digest}  acquisition.manifest.json\n".encode("utf-8")
    assert parse_manifest_checksum_sidecar(payload, "acquisition.manifest.json") == digest
    assert verify_manifest_checksum(result.manifest_relative_path) == digest


def test_s2_14_the_run_log_records_the_manifest_checksum_before_completion(
    config, temp_acquisition_root
) -> None:
    result, _ = complete_run(config)
    entries = [
        json.loads(line)
        for line in (temp_acquisition_root / result.log_relative_path)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    events = [entry["event"] for entry in entries]
    assert events[-2] == "manifest_checksum_written"
    assert events[-1] == "acquisition_completed"
    recorded = entries[-2]
    digest = hashlib.sha256(
        (temp_acquisition_root / result.manifest_relative_path).read_bytes()
    ).hexdigest()
    assert recorded["sha256"] == digest
    assert recorded["path"] == result.manifest_checksum_relative_path.as_posix()
    assert recorded["manifest"] == result.manifest_relative_path.as_posix()


def test_s2_14_an_existing_sidecar_is_not_overwritten(config, temp_acquisition_root) -> None:
    result, _ = complete_run(config)
    sidecar = temp_acquisition_root / result.manifest_checksum_relative_path
    before = sidecar.read_bytes()
    raw_before = {
        path.relative_to(temp_acquisition_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((temp_acquisition_root / "data" / "raw").iterdir())
        if path.is_file()
    }
    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        write_manifest_checksum(result.manifest_relative_path)
    assert sidecar.read_bytes() == before
    raw_after = {
        path.relative_to(temp_acquisition_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((temp_acquisition_root / "data" / "raw").iterdir())
        if path.is_file()
    }
    assert raw_after == raw_before


@pytest.mark.parametrize(
    ("sidecar_payload", "expected"),
    (
        (b"", "must be"),
        (b"a" * 64 + b" acquisition.manifest.json\n", "must be"),
        (b"a" * 64 + b"  *acquisition.manifest.json\n", "must be"),
        (b"A" * 64 + b"  acquisition.manifest.json\n", "must be"),
        (b"a" * 64 + b"  other.manifest.json\n", "must be"),
        (b"a" * 64 + b"  acquisition.manifest.json\nextra\n", "must be"),
        (b"\xff\xfe" + b"a" * 64, "not UTF-8"),
    ),
)
def test_s2_14_a_malformed_sidecar_fails_closed(
    temp_acquisition_root, sidecar_payload, expected
) -> None:
    create_exclusive("outputs/runs/probe/acquisition.manifest.json", "{}\n")
    create_exclusive_bytes(
        "outputs/runs/probe/acquisition.manifest.json.sha256", sidecar_payload
    )
    with pytest.raises(AcquisitionError, match=expected):
        parse_manifest_checksum_sidecar(sidecar_payload, "acquisition.manifest.json")
    with pytest.raises(AcquisitionError, match=expected):
        verify_manifest_checksum(Path("outputs/runs/probe/acquisition.manifest.json"))


def test_s2_14_a_mismatched_sidecar_digest_fails_closed(temp_acquisition_root) -> None:
    create_exclusive("outputs/runs/probe/acquisition.manifest.json", "{}\n")
    create_exclusive(
        "outputs/runs/probe/acquisition.manifest.json.sha256",
        f"{'b' * 64}  acquisition.manifest.json\n",
    )
    with pytest.raises(AcquisitionError, match="digests to"):
        verify_manifest_checksum(Path("outputs/runs/probe/acquisition.manifest.json"))
