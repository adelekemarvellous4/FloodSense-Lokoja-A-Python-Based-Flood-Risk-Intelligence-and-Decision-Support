"""Configuration loading and validation for FloodSense Lokoja."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = (
    "project.name",
    "project.study_area",
    "project.crs",
    "paths.raw_boundary",
    "paths.processed_rasters",
    "paths.processed_vectors",
    "paths.processed_tables",
    "weights",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_config_value(config: dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Return a nested config value using dot notation."""
    value: Any = config
    for key in key_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def validate_config(config: dict[str, Any]) -> None:
    """Validate required configuration keys."""
    missing = [key for key in REQUIRED_KEYS if get_config_value(config, key) is None]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Config is missing required key(s): {missing_text}")


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load and validate the project YAML config."""
    path = Path(config_path)
    if not path.is_absolute():
        path = _project_root() / path
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Create config/config.yaml from the project template."
        )
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    validate_config(config)
    return config
