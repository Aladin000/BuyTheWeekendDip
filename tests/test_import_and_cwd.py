"""Clean imports and working-directory independence.

A fresh interpreter must import every core module without side effects, and
configuration loading must not depend on where the process was started.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from src import paths
from src.config import ConfigError, load_config

STAGE1_MODULES = (
    "src",
    "src.paths",
    "src.config",
    "src.rng",
    "src.artifacts",
    "src.pipeline",
)


# Imports


@pytest.mark.parametrize("name", STAGE1_MODULES)
def test_s1_26_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_s1_26_package_imports_in_a_clean_interpreter(project_root: Path) -> None:
    """A subprocess with no inherited state imports every core module."""
    program = "import " + ", ".join(STAGE1_MODULES) + "; print('ok')"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(project_root)},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


_AUDIT_PROGRAM = """
import sys

opened = []
sys.addaudithook(lambda event, args: opened.append(str(args[0])) if event == "open" else None)

import {modules}

offenders = [p for p in opened if p.endswith(("project.yaml", "01_PROJECT_SPEC.md", "PROJECT_SPEC.md"))]
print("OFFENDERS " + repr(offenders) if offenders else "CLEAN")
"""


def test_s1_26_import_does_not_read_project_files(project_root: Path) -> None:
    """Importing must not read the configuration or the project spec."""
    completed = subprocess.run(
        [sys.executable, "-c", _AUDIT_PROGRAM.format(modules=", ".join(STAGE1_MODULES))],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(project_root)},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "CLEAN", completed.stdout


# Working-directory independence


def test_s1_27_config_loads_from_an_unrelated_directory(monkeypatch, tmp_path, config) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_config() == config


def test_s1_27_config_loads_from_the_filesystem_root(monkeypatch, config) -> None:
    monkeypatch.chdir("/")
    assert load_config() == config


def test_s1_27_project_root_is_stable_across_directories(monkeypatch, tmp_path) -> None:
    expected = paths.project_root()
    monkeypatch.chdir(tmp_path)
    assert paths.project_root() == expected
    monkeypatch.chdir("/")
    assert paths.project_root() == expected


def test_s1_27_no_configured_path_is_absolute(config) -> None:
    for key, value in config.paths.items():
        assert not Path(value).is_absolute(), f"paths.{key} is absolute"
    assert not Path(config.meta.spec_path).is_absolute()


@pytest.mark.parametrize(
    "absolute",
    ("/etc/passwd", "/tmp/data.csv", "~/data.csv", "~root/data.csv"),
)
def test_s1_27_resolve_rejects_an_absolute_path(absolute: str) -> None:
    with pytest.raises(paths.ProjectLayoutError, match="absolute"):
        paths.resolve(absolute)


@pytest.mark.parametrize(
    "escaping",
    (
        "../outside",
        "../outside/file.csv",
        "../../outside",
        "outputs/../../outside",
        "data/raw/../../../outside/file.csv",
        "outputs/runs/../../..",
        "..",
    ),
)
def test_s1_27_resolve_rejects_parent_traversal(escaping: str) -> None:
    """Regression: ``resolve("../outside")`` used to escape the project root."""
    with pytest.raises(paths.ProjectLayoutError, match="parent traversal"):
        paths.resolve(escaping)


@pytest.mark.parametrize(
    "relative",
    (
        "config/project.yaml",
        "docs/PROJECT_SPEC.md",
        "data/raw",
        "outputs/runs",
        "outputs/runs/nested/deeper/manifest.json",
        "tests/fixtures/short_warmup.yaml",
    ),
)
def test_s1_27_resolve_accepts_valid_nested_paths(relative: str) -> None:
    root = paths.project_root().resolve()
    resolved = paths.resolve(relative)
    assert resolved.is_absolute()
    assert resolved.is_relative_to(root)
    assert resolved.relative_to(root).as_posix() == relative


def test_s1_27_resolve_accepts_multipart_input() -> None:
    assert paths.resolve("outputs", "runs", "manifest.json") == paths.resolve(
        "outputs/runs/manifest.json"
    )


def test_s1_27_resolve_is_independent_of_the_working_directory(monkeypatch, tmp_path) -> None:
    expected = paths.resolve("config/project.yaml")
    monkeypatch.chdir(tmp_path)
    assert paths.resolve("config/project.yaml") == expected
    monkeypatch.chdir("/")
    assert paths.resolve("config/project.yaml") == expected


def test_s1_27_resolve_requires_at_least_one_part() -> None:
    with pytest.raises(paths.ProjectLayoutError, match="at least one relative path part"):
        paths.resolve()


def test_s1_27_configured_paths_resolve_under_the_project_root(config) -> None:
    root = paths.project_root()
    for value in config.paths.values():
        resolved = paths.resolve(value)
        assert resolved.is_relative_to(root)
        assert resolved.is_dir(), f"{value} does not exist"


def test_s1_27_configuration_object_is_immutable(config) -> None:
    with pytest.raises(AttributeError, match="immutable"):
        config.shock = None  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        del config.shock  # type: ignore[misc]
    with pytest.raises(TypeError):
        config["shock"]["window_hours"] = 1  # type: ignore[index]
    # to_dict() hands back a copy, so editing it cannot reach the live object.
    snapshot = config.to_dict()
    snapshot["shock"]["window_hours"] = 1
    assert config.shock.window_hours == 720


def test_s1_27_missing_configuration_file_fails_closed(tmp_path) -> None:
    with pytest.raises(paths.ProjectLayoutError, match="Required file missing"):
        load_config(tmp_path / "absent.yaml")


def test_s1_27_non_mapping_configuration_file_fails_closed(tmp_path) -> None:
    target = tmp_path / "list.yaml"
    target.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="expected a YAML mapping"):
        load_config(target)


def test_single_spec_resolves_historical_logical_path(config) -> None:
    """Both current and historical names resolve to the single document."""
    from src.artifacts import sha256_file, spec_identity

    expected = paths.project_root() / paths.SPEC_RELATIVE_PATH
    assert paths.spec_path() == expected
    assert paths.resolve("docs/01_PROJECT_SPEC.md") == expected
    assert paths.resolve("docs", "01_PROJECT_SPEC.md") == expected
    assert paths.resolve(config.meta.spec_path) == expected
    assert spec_identity(config).relative_path == "docs/01_PROJECT_SPEC.md"
    assert spec_identity(config).sha256 == sha256_file(expected)
    assert not (paths.project_root() / "docs/01_PROJECT_SPEC.md").exists()


def test_spec_rename_does_not_alias_other_paths() -> None:
    root = paths.project_root()
    for relative in ("other/01_PROJECT_SPEC.md", "docs/nested/01_PROJECT_SPEC.md"):
        assert paths.resolve(relative) == root / relative


def test_legacy_spec_alias_still_rejects_symlink_escape(monkeypatch, tmp_path) -> None:
    root = tmp_path / "project"
    (root / "docs").mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (root / paths.SPEC_RELATIVE_PATH).symlink_to(outside)
    monkeypatch.setattr(paths, "project_root", lambda: root)
    with pytest.raises(paths.ProjectLayoutError, match="escapes"):
        paths.resolve("docs/01_PROJECT_SPEC.md")



def test_spec_identity_tracks_the_current_document(temp_project_root, config) -> None:
    from src.artifacts import sha256_file, spec_identity

    document = temp_project_root / paths.SPEC_RELATIVE_PATH
    before = spec_identity(config)
    document.write_text(document.read_text() + "Editorial revision\n", encoding="utf-8")
    after = spec_identity(config)
    assert after.sha256 == sha256_file(document)
    assert after.sha256 != before.sha256
    assert paths.resolve(config.meta.spec_path) == document


def test_missing_specification_fails_closed(temp_project_root, config) -> None:
    from src.artifacts import spec_identity

    (temp_project_root / paths.SPEC_RELATIVE_PATH).unlink()
    with pytest.raises(paths.ProjectLayoutError, match="Required file missing"):
        spec_identity(config)
