"""Flood susceptibility modelling functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from floodsense.mapping import plot_raster_preview
from floodsense.paths import get_project_root
from floodsense.raster_utils import classify_array, normalize_array, save_raster


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else get_project_root() / path


def normalize_flood_factor(array: np.ndarray, factor_name: str, nodata: float | int | None = None) -> np.ndarray:
    """Normalize a flood factor so higher values mean higher flood risk."""
    if factor_name == "landcover":
        landcover = array.astype("float32", copy=True)
        if nodata is not None:
            landcover[landcover == nodata] = np.nan
        risk = np.full(landcover.shape, np.nan, dtype="float32")
        risk_map = {
            10: 0.25,  # tree cover
            20: 0.35,  # shrubland
            30: 0.40,  # grassland
            40: 0.55,  # cropland
            50: 0.85,  # built-up
            60: 0.45,  # bare/sparse vegetation
            70: 0.20,  # snow/ice
            80: 0.90,  # permanent water
            90: 0.75,  # herbaceous wetland
            95: 0.70,  # mangroves
            100: 0.60,  # moss/lichen
        }
        for code, value in risk_map.items():
            risk[landcover == code] = value
        if np.isfinite(risk).any():
            return risk
        print("[susceptibility] Land-cover codes not recognized; using generic normalization.")
    inverse_factors = {"elevation", "slope", "distance_to_river", "distance_to_stream"}
    inverse = factor_name in inverse_factors
    return normalize_array(array, inverse=inverse, nodata=nodata)


def calculate_weighted_susceptibility(
    factor_arrays: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    """Calculate weighted flood susceptibility from normalized factors."""
    if not factor_arrays:
        raise ValueError("No factor arrays supplied for susceptibility calculation.")
    score = None
    used_weight = 0.0
    for name, array in factor_arrays.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            print(f"[susceptibility] No weight for {name}; skipping.")
            continue
        clean = np.nan_to_num(array.astype("float32"), nan=0.0)
        score = clean * weight if score is None else score + clean * weight
        used_weight += weight
    if score is None or used_weight == 0:
        raise ValueError("No weighted factors were available.")
    return np.clip(score / used_weight, 0, 1).astype("float32")


def classify_susceptibility(score_array: np.ndarray) -> np.ndarray:
    """Classify 0-1 susceptibility scores into five classes."""
    return classify_array(score_array, bins=[0.2, 0.4, 0.6, 0.8]).astype("uint8")


def calculate_class_area(class_raster_path: str | Path, output_csv: str | Path) -> Path:
    """Calculate area by susceptibility class."""
    with rasterio.open(class_raster_path) as src:
        data = src.read(1)
        pixel_area_km2 = abs(src.transform.a * src.transform.e) / 1_000_000
    rows = []
    labels = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
    for value, label in labels.items():
        cells = int(np.sum(data == value))
        rows.append({"class_value": value, "class_name": label, "area_km2": cells * pixel_area_km2})
    output_csv = _resolve(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def _factor_paths(config: dict) -> dict[str, Path]:
    rasters = _resolve(config["paths"]["processed_rasters"])
    interim = _resolve(config["paths"]["interim_reprojected"])
    candidates = {
        "elevation": [rasters / "dem_filled.tif", interim / "dem.tif"],
        "slope": [rasters / "slope.tif"],
        "flow_accumulation": [rasters / "flow_accumulation.tif"],
        "distance_to_river": [rasters / "distance_to_stream.tif", rasters / "distance_to_river.tif"],
        "rainfall": [interim / "rainfall.tif"],
        "landcover": [interim / "landcover.tif"],
    }
    found = {}
    for name, paths in candidates.items():
        for path in paths:
            if path.exists():
                found[name] = path
                break
        if name not in found:
            print(f"[susceptibility] Missing factor raster: {name}.")
    return found


def run_susceptibility_workflow(config: dict) -> None:
    """Build susceptibility score and classified risk rasters."""
    paths = _factor_paths(config)
    if not paths:
        print("[susceptibility] No factor rasters found. Run preprocessing and hydrology first.")
        return
    reference_path = next(iter(paths.values()))
    factor_arrays = {}
    profile = None
    shape = None
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        shape = (ref.height, ref.width)
    for name, path in paths.items():
        with rasterio.open(path) as src:
            if (src.height, src.width) != shape:
                print(f"[susceptibility] {name} grid does not match reference; skipping for now.")
                continue
            array = src.read(1)
            factor_arrays[name] = normalize_flood_factor(array, name, src.nodata)
    if not factor_arrays:
        print("[susceptibility] No aligned factor rasters available.")
        return
    score = calculate_weighted_susceptibility(factor_arrays, config.get("weights", {}))
    classes = classify_susceptibility(score)
    rasters = _resolve(config["paths"]["processed_rasters"])
    tables = _resolve(config["paths"]["processed_tables"])
    maps = _resolve(config["paths"]["outputs_maps"])
    score_path = save_raster(score.astype("float32"), profile, rasters / "susceptibility_score.tif", nodata=np.nan)
    class_path = save_raster(classes.astype("uint8"), profile, rasters / "susceptibility_class.tif", nodata=0)
    calculate_class_area(class_path, tables / "susceptibility_class_area.csv")
    plot_raster_preview(class_path, maps / "susceptibility_preview.png", "Flood Susceptibility")
    print(f"[susceptibility] Wrote {score_path} and {class_path}")
