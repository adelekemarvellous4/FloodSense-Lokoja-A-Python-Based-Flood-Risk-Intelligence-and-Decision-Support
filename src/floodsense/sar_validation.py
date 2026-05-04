"""Sentinel-1 SAR flood extraction and validation functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from floodsense.paths import first_existing_file, get_project_root, list_input_files
from floodsense.raster_utils import save_raster


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else get_project_root() / path


def _find_sar_pair(config: dict) -> tuple[Path | None, Path | None]:
    sentinel_dir = config["paths"]["raw_sentinel1"]
    files = list_input_files(sentinel_dir, [".tif", ".tiff"])
    if not files:
        files = list_input_files(config["paths"]["interim_reprojected"], [".tif", ".tiff"])
        files = [path for path in files if "sentinel" in path.name.lower() or "sar" in path.name.lower()]
    pre = next((p for p in files if "pre" in p.name.lower()), None)
    flood = next(
        (p for p in files if any(token in p.name.lower() for token in ["flood", "post", "during"])),
        None,
    )
    if pre is None:
        pre = first_existing_file(Path(sentinel_dir) / "pre_flood", [".tif", ".tiff"])
    if flood is None:
        flood = first_existing_file(Path(sentinel_dir) / "during_flood", [".tif", ".tiff"])
    return pre, flood


def derive_flood_extent_from_sar(
    pre_flood_path: str | Path,
    flood_path: str | Path,
    output_path: str | Path,
    method: str = "ratio",
    ratio_threshold: float = 0.7,
) -> Path:
    """Derive a simple SAR observed flood extent using ratio or difference change detection."""
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(pre_flood_path) as pre_src, rasterio.open(flood_path) as flood_src:
        pre = pre_src.read(1).astype("float32")
        flood = flood_src.read(1).astype("float32")
        profile = pre_src.profile.copy()
        if pre.shape != flood.shape:
            raise ValueError("Pre-flood and flood SAR rasters must be aligned before validation.")
        pre_nodata = pre_src.nodata
        flood_nodata = flood_src.nodata
    invalid = np.zeros(pre.shape, dtype=bool)
    if pre_nodata is not None:
        invalid |= pre == pre_nodata
    if flood_nodata is not None:
        invalid |= flood == flood_nodata
    if method == "ratio":
        ratio = np.divide(flood, pre, out=np.ones_like(flood), where=pre != 0)
        observed = (ratio < ratio_threshold).astype("uint8")
    else:
        diff = flood - pre
        threshold = np.nanpercentile(diff[~invalid], 10) if np.any(~invalid) else 0
        observed = (diff < threshold).astype("uint8")
    observed[invalid] = 0
    save_raster(observed, profile, output_path, nodata=0)
    return output_path


def validate_modelled_flood(
    modelled_class_path: str | Path, observed_flood_path: str | Path, output_csv: str | Path
) -> Path:
    """Compare modelled high-risk classes against observed flood extent."""
    with rasterio.open(modelled_class_path) as model_src, rasterio.open(observed_flood_path) as obs_src:
        model = model_src.read(1)
        observed = obs_src.read(1)
        if model.shape != observed.shape:
            raise ValueError("Modelled and observed rasters must be aligned before validation.")
        predicted = np.isin(model, [4, 5]).astype("uint8")
        actual = (observed == 1).astype("uint8")
        valid = model != 0
    y_true = actual[valid].ravel()
    y_pred = predicted[valid].ravel()
    if y_true.size == 0:
        raise ValueError("No valid pixels available for SAR validation.")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "overall_accuracy": accuracy_score(y_true, y_pred),
    }
    output_csv = _resolve(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_csv, index=False)
    return output_csv


def run_sar_validation(config: dict) -> None:
    """Run Sentinel-1 validation workflow."""
    pre, flood = _find_sar_pair(config)
    if pre is None or flood is None:
        print("[sar] Sentinel-1 data missing. Place pre-flood and flood/post-flood rasters in:")
        print("[sar] data/raw/sentinel1/pre_flood/ and data/raw/sentinel1/during_flood/")
        return
    rasters = _resolve(config["paths"]["processed_rasters"])
    modelled = rasters / "susceptibility_class.tif"
    if not modelled.exists():
        print("[sar] Missing susceptibility_class.tif. Run susceptibility workflow first.")
        return
    observed = rasters / "observed_flood_extent.tif"
    method = config.get("sar_validation", {}).get("method", "ratio")
    ratio_threshold = float(config.get("sar_validation", {}).get("ratio_threshold", 0.7))
    derive_flood_extent_from_sar(pre, flood, observed, method=method, ratio_threshold=ratio_threshold)
    metrics_csv = _resolve(config["paths"]["outputs_validation"]) / "validation_metrics.csv"
    validate_modelled_flood(modelled, observed, metrics_csv)

    with rasterio.open(modelled) as model_src, rasterio.open(observed) as obs_src:
        model = np.isin(model_src.read(1), [4, 5]).astype("uint8")
        observed_data = (obs_src.read(1) == 1).astype("uint8")
        comparison = model + observed_data * 2
        profile = model_src.profile.copy()
    save_raster(comparison.astype("uint8"), profile, rasters / "model_vs_observed_comparison.tif", nodata=0)
    print(f"[sar] Validation metrics written: {metrics_csv}")
