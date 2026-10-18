"""Project-root discovery and relative path resolution.

Every path is relative to the project root. The root is found by walking up
from this file, so it does not depend on the working directory or on an
environment variable.
"""

from __future__ import annotations

from pathlib import Path

CONFIG_RELATIVE_PATH = Path("config") / "project.yaml"
SPEC_RELATIVE_PATH = Path("docs") / "PROJECT_SPEC.md"
# Accepted manifests and the frozen configuration retain this logical name.
# Resolve the historical name to the single current specification.
_LEGACY_SPEC_RELATIVE_PATH = Path("docs") / "01_PROJECT_SPEC.md"

# Requiring both files avoids matching an unrelated parent folder that just
# happens to contain a config folder.
_ROOT_SENTINELS = (CONFIG_RELATIVE_PATH, SPEC_RELATIVE_PATH)


class ProjectLayoutError(RuntimeError):
    """Raised when the project root or a required project path is missing."""


def project_root() -> Path:
    """Return the absolute project root directory."""
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if all((candidate / sentinel).is_file() for sentinel in _ROOT_SENTINELS):
            return candidate
    sentinels = ", ".join(str(sentinel) for sentinel in _ROOT_SENTINELS)
    raise ProjectLayoutError(
        f"Project root not found above {start}: expected a directory containing {sentinels}"
    )


def resolve(*relative_parts: str | Path) -> Path:
    """Resolve a root-relative path to an absolute path inside the project.

    Absolute paths, ``~`` paths and ``..`` parts are rejected, and the result
    must lie inside the project root, which also catches escape through a
    symlink. The historical specification name resolves to PROJECT_SPEC.md.
    """
    if not relative_parts:
        raise ProjectLayoutError("resolve() requires at least one relative path part")
    relative = Path(*[Path(part) for part in relative_parts])
    if relative.is_absolute() or str(relative).startswith("~"):
        raise ProjectLayoutError(f"Expected a project-relative path, got absolute {relative}")
    if ".." in relative.parts:
        raise ProjectLayoutError(
            f"Expected a path without parent traversal, got {relative}"
        )
    if relative == _LEGACY_SPEC_RELATIVE_PATH:
        relative = SPEC_RELATIVE_PATH
    root = project_root().resolve()
    candidate = (root / relative).resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise ProjectLayoutError(
            f"Resolved path {candidate} escapes the project root {root}"
        )
    return candidate


def config_path() -> Path:
    """Absolute path of the central configuration file."""
    return project_root() / CONFIG_RELATIVE_PATH


def spec_path() -> Path:
    """Absolute path of the current project specification."""
    return resolve(SPEC_RELATIVE_PATH)


def require_file(path: Path) -> Path:
    """Return ``path`` when it is an existing file, otherwise raise an error."""
    if not path.is_file():
        raise ProjectLayoutError(f"Required file missing: {path}")
    return path
