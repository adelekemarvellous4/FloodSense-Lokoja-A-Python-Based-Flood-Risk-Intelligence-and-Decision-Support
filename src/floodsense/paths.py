"""Path helpers for the FloodSense project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def get_project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = get_project_root() / resolved
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_project_directories(config: dict | None = None) -> None:
    """Ensure configured input, interim, processed, and output directories exist."""
    default_dirs = [
        "data/raw/boundary",
        "data/raw/dem",
        "data/raw/rainfall",
        "data/raw/sentinel1",
        "data/raw/landcover",
        "data/raw/buildings",
        "data/raw/roads",
        "data/raw/population",
        "data/interim/clipped",
        "data/interim/reprojected",
        "data/interim/cleaned",
        "data/interim/masks",
        "data/processed/rasters",
        "data/processed/vectors",
        "data/processed/tables",
        "data/processed/dashboard_layers",
        "outputs/maps",
        "outputs/tables",
        "outputs/validation",
        "outputs/dashboard",
    ]
    configured_dirs = list((config or {}).get("paths", {}).values())
    for folder in [*default_dirs, *configured_dirs]:
        ensure_dir(folder)


def _matches_extension(path: Path, extensions: Iterable[str] | None) -> bool:
    if extensions is None:
        return True
    normalized = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    return path.suffix.lower() in normalized


def list_input_files(folder: str | Path, extensions: Iterable[str] | None = None) -> list[Path]:
    """List files in a folder, optionally filtered by extension."""
    root = Path(folder)
    if not root.is_absolute():
        root = get_project_root() / root
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and _matches_extension(path, extensions))


def first_existing_file(folder: str | Path, extensions: Iterable[str] | None = None) -> Path | None:
    """Return the first matching file in a folder, or None."""
    files = list_input_files(folder, extensions)
    return files[0] if files else None
