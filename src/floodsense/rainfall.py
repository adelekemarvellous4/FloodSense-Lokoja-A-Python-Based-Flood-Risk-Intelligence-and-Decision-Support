"""Rainfall and scenario modelling functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from floodsense.paths import first_existing_file, get_project_root
from floodsense.raster_utils import save_raster


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else get_project_root() / path


def read_rainfall_input(path: str | Path) -> pd.DataFrame | Path | None:
    """Read rainfall CSV/table or return raster path."""
    path = Path(path)
    if not path.exists():
        print(f"[rainfall] Rainfall input missing: {path}")
        return None
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    return path


def create_scenario_modifiers(config: dict) -> dict[str, float]:
    """Create low, moderate, and severe rainfall scenario modifiers."""
    scenarios = config.get("rainfall_scenarios", {})
    return {
        "low": float(scenarios.get("low_modifier", 0.85)),
        "moderate": float(scenarios.get("moderate_modifier", 1.0)),
        "severe": float(scenarios.get("severe_modifier", 1.2)),
    }


def apply_scenarios_to_susceptibility(
    susceptibility_path: str | Path, output_dir: str | Path, config: dict
) -> list[Path]:
    """Apply rainfall scenario modifiers to a susceptibility score raster."""
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with rasterio.open(susceptibility_path) as src:
        score = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None:
        score[score == nodata] = np.nan
    rows = []
    for name, modifier in create_scenario_modifiers(config).items():
        scenario = np.clip(score * modifier, 0, 1).astype("float32")
        out_path = output_dir / f"flood_scenario_{name}.tif"
        save_raster(scenario, profile, out_path, nodata=np.nan)
        written.append(out_path)
        rows.append({"scenario": name, "modifier": modifier, "mean_score": float(np.nanmean(scenario))})
    pd.DataFrame(rows).to_csv(output_dir / "rainfall_scenario_summary.csv", index=False)
    # TODO: Add 10-year, 50-year, and 100-year return-period scenarios when IDF data is available.
    return written


def run_rainfall_scenarios(config: dict) -> None:
    """Run rainfall scenario workflow."""
    susceptibility = _resolve(config["paths"]["processed_rasters"]) / "susceptibility_score.tif"
    if not susceptibility.exists():
        print("[rainfall] Missing susceptibility_score.tif. Run susceptibility workflow first.")
        return
    rainfall_input = first_existing_file(config["paths"]["raw_rainfall"], [".tif", ".tiff", ".csv"])
    if rainfall_input is None:
        print("[rainfall] Rainfall data missing. Running modifier-only scenarios from susceptibility score.")
    else:
        read_rainfall_input(rainfall_input)
    outputs = apply_scenarios_to_susceptibility(
        susceptibility, _resolve(config["paths"]["processed_rasters"]), config
    )
    print(f"[rainfall] Wrote {len(outputs)} scenario raster(s).")
