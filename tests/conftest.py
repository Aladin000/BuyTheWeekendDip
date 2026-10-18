"""Shared test fixtures."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from src import paths
from src.config import ConfigNode, load_config, read_raw_config

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def config() -> ConfigNode:
    """The validated central configuration."""
    return load_config()


@pytest.fixture
def raw_config() -> dict[str, Any]:
    """A fresh, unvalidated copy of the configuration mapping."""
    return read_raw_config()


@pytest.fixture(scope="session")
def project_root() -> Path:
    return paths.project_root()


def build_temp_project_root(tmp_path: Path) -> Path:
    """Create a temporary directory that passes project-root discovery.

    Both marker files exist, so discovery works without patching.
    """
    root = (tmp_path / "project").resolve()
    (root / paths.CONFIG_RELATIVE_PATH.parent).mkdir(parents=True, exist_ok=True)
    (root / paths.SPEC_RELATIVE_PATH.parent).mkdir(parents=True, exist_ok=True)
    (root / paths.CONFIG_RELATIVE_PATH).write_text("placeholder: true\n", encoding="utf-8")
    (root / paths.SPEC_RELATIVE_PATH).write_text(
        "**Specification version:** 1.5\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def temp_project_root(monkeypatch, tmp_path) -> Path:
    """A temporary project root for tests that write run outputs.

    Writes are only allowed inside the project root, so such tests need their
    own root. A test that must first read the real project should call
    :func:`build_temp_project_root` itself and patch afterwards.
    """
    root = build_temp_project_root(tmp_path)
    monkeypatch.setattr(paths, "project_root", lambda: root)
    return root


def build_temp_identity_root(tmp_path: Path) -> Path:
    """A temporary project root complete enough for `code_identity()`.

    :func:`build_temp_project_root` stays bare because some tests need an
    incomplete snapshot to fail. A stage that builds a real manifest needs one
    file in every identity group plus every required identity file.
    """
    root = build_temp_project_root(tmp_path)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "src" / "placeholder.py").write_text("PLACEHOLDER = True\n", encoding="utf-8")
    (root / "tests" / "test_placeholder.py").write_text(
        "def test_placeholder() -> None:\n    assert True\n", encoding="utf-8"
    )
    (root / "pyproject.toml").write_text("[project]\nname = 'placeholder'\n", encoding="utf-8")
    (root / "requirements.txt").write_text("numpy==2.3.5\n", encoding="utf-8")
    (root / "requirements-lock.txt").write_text("numpy==2.3.5\n", encoding="utf-8")
    return root


@pytest.fixture
def temp_acquisition_root(monkeypatch, tmp_path) -> Path:
    """A temporary project root that a full acquisition run can use."""
    root = build_temp_identity_root(tmp_path)
    monkeypatch.setattr(paths, "project_root", lambda: root)
    return root


# Invalid-configuration mutation fixtures


@dataclass(frozen=True)
class MutationCase:
    name: str
    description: str
    expect_message_contains: str
    operations: tuple[dict[str, Any], ...]

    def apply(self, raw: dict[str, Any]) -> dict[str, Any]:
        mutated = copy.deepcopy(raw)
        for operation in self.operations:
            _apply_operation(mutated, operation)
        return mutated


def _apply_operation(data: dict[str, Any], operation: dict[str, Any]) -> None:
    op = operation["op"]
    parts = str(operation["target"]).split(".")
    node: Any = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"mutation target {operation['target']!r} has no parent {part!r}")
        node = node[part]
    leaf = parts[-1]
    if op == "delete":
        del node[leaf]
    elif op == "set":
        node[leaf] = operation["value"]
    else:
        raise ValueError(f"unsupported mutation op {op!r}")


def load_mutation_cases() -> list[MutationCase]:
    """Every mutation fixture, excluding files whose name starts with '_'."""
    cases: list[MutationCase] = []
    for path in sorted(FIXTURES_DIR.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        cases.append(
            MutationCase(
                name=path.stem,
                description=str(payload["description"]).strip(),
                expect_message_contains=payload["expect_message_contains"],
                operations=tuple(payload["operations"]),
            )
        )
    if not cases:  # pragma: no cover - guards an empty fixture directory
        raise RuntimeError(f"no mutation fixtures found in {FIXTURES_DIR}")
    return cases


MUTATION_CASES = load_mutation_cases()
