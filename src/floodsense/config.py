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
    "susceptibility_weights",
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
    """Validate required configuration keys are present."""
    missing = [key for key in REQUIRED_KEYS if get_config_value(config, key) is None]
    if missing:
        raise ValueError(f"Config missing required key(s): {', '.join(missing)}")


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load and validate the project YAML config."""
    path = Path(config_path)
    if not path.is_absolute():
        path = _project_root() / path
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}\n"
            "Ensure config/config.yaml exists in the project root."
        )
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    validate_config(config)
    return config


def get_weights(config: dict[str, Any]) -> dict[str, float]:
    """Return susceptibility weights as a plain dict of floats."""
    return {k: float(v) for k, v in config["susceptibility_weights"].items()}


def get_crs(config: dict[str, Any]) -> str:
    """Return the projected CRS string."""
    return config["project"]["crs"]


def get_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path key from config relative to the project root."""
    raw = config["paths"].get(key)
    if raw is None:
        raise KeyError(f"Path key '{key}' not found in config.")
    p = Path(raw)
    return p if p.is_absolute() else _project_root() / p
