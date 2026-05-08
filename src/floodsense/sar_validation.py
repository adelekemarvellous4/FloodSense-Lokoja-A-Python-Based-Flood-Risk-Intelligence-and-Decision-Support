"""Sentinel-1 SAR flood extraction and validation for FloodSense Lokoja."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score
)

from floodsense.mapping import plot_raster_preview
from floodsense.paths import (
    ensure_dir, first_existing_file, get_project_root, list_input_files
)
from floodsense.raster_utils import save_raster


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else get_project_root() / path


# ── SAR pair discovery ────────────────────────────────────────────────────────

def _find_sar_pair(config: dict) -> tuple[Path | None, Path | None]:
    """Locate pre-flood and flood-date SAR GeoTIFF files."""
    sentinel_dir = Path(config["paths"]["raw_sentinel1"])

    # Check dedicated sub-folders first (preferred structure)
    pre   = first_existing_file(sentinel_dir / "pre_flood",    [".tif", ".tiff"])
    flood = first_existing_file(sentinel_dir / "during_flood", [".tif", ".tiff"])

    # Fallback: search root sentinel1 folder by filename keyword
    if pre is None or flood is None:
        all_files = list_input_files(sentinel_dir, [".tif", ".tiff"])
        if pre is None:
            pre = next(
                (p for p in all_files if "pre" in p.name.lower()), None
            )
        if flood is None:
            flood = next(
                (p for p in all_files
                 if any(t in p.name.lower() for t in ["flood", "during", "post"])),
                None,
            )

    return pre, flood


# ── SAR preprocessing ─────────────────────────────────────────────────────────

def linear_to_db(array: np.ndarray, nodata: float | None = None) -> np.ndarray:
    """
    Convert SAR backscatter from linear power scale to decibels.

    GEE exports Sentinel-1 GRD in linear scale by default.
    dB = 10 × log10(linear)

    Working in dB is important because:
    - It compresses the very wide dynamic range of SAR backscatter
    - Open water typically shows very low dB values (< -20 dB)
    - Flooded areas are darker (lower dB) than dry land
    - Thresholds are more intuitive in dB space
    """
    db = np.full(array.shape, np.nan, dtype="float32")
    valid = np.isfinite(array) & (array > 0)
    if nodata is not None:
        valid &= array != nodata
    db[valid] = 10.0 * np.log10(array[valid])
    return db


def mask_permanent_water(
    flood_array: np.ndarray,
    landcover_path: Path | None,
    water_class: int = 80,
    reference_profile: dict | None = None,
) -> np.ndarray:
    """
    Remove permanent water bodies from the SAR-derived flood extent.

    The land-cover raster is reprojected/resampled to the SAR flood raster grid
    before masking. This avoids comparing rasters with different CRS, transform,
    resolution, shape, or bounds.
    """
    if landcover_path is None or not landcover_path.exists():
        print("  [sar] No land cover found — permanent water not masked")
        return flood_array

    if reference_profile is None:
        raise ValueError("reference_profile is required to align land cover to the SAR grid.")

    dst_shape = flood_array.shape
    dst_transform = reference_profile["transform"]
    dst_crs = reference_profile["crs"]

    with rasterio.open(landcover_path) as lc_src:
        lc = lc_src.read(1)

        needs_alignment = (
            lc.shape != dst_shape
            or lc_src.transform != dst_transform
            or lc_src.crs != dst_crs
        )

        if needs_alignment:
            print("  [sar] Aligning permanent water mask to SAR grid...")
            aligned = np.zeros(dst_shape, dtype="float32")
            reproject(
                source=lc.astype("float32"),
                destination=aligned,
                src_transform=lc_src.transform,
                src_crs=lc_src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
            lc = np.round(aligned).astype("int16")

    permanent_water = lc == water_class
    overlap = (flood_array == 1) & permanent_water

    masked = flood_array.copy()
    masked[permanent_water] = 0

    print(f"  [sar] Total permanent-water pixels in mask: {int(permanent_water.sum()):,}")
    print(f"  [sar] Flood pixels overlapping permanent water: {int(overlap.sum()):,}")
    print(f"  [sar] Final flood pixels after water removal: {int(masked.sum()):,}")

    return masked

# ── Flood extent derivation ───────────────────────────────────────────────────
def derive_flood_extent_from_sar(
    pre_flood_path: str | Path,
    flood_path: str | Path,
    output_path: str | Path,
    landcover_path: Path | None = None,
    method: str = "combined_db",
    ratio_threshold: float = 0.7,
    flood_vh_threshold: float = -16.5,
    db_difference_threshold: float = -1.0,
) -> Path:
    """
    Derive observed flood extent from pre-flood and flood-date Sentinel-1 VH images.

    IMPORTANT:
    Google Earth Engine COPERNICUS/S1_GRD exports are already in dB scale.
    Therefore, this workflow uses direct dB logic:

      difference_db = flood_vh_db - pre_flood_vh_db

    Recommended method for this project:

      combined_db:
        observed_flood =
          (flood_vh_db < flood_vh_threshold)
          AND
          ((flood_vh_db - pre_vh_db) < db_difference_threshold)

    This means a pixel is treated as flooded only if:
      1. it is dark during the flood period, and
      2. it became darker than the pre-flood reference.
    """
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(pre_flood_path) as pre_src:
        pre = pre_src.read(1).astype("float32")
        profile = pre_src.profile.copy()
        pre_nd = pre_src.nodata
        ref_shape = pre.shape
        ref_transform = pre_src.transform
        ref_crs = pre_src.crs

    with rasterio.open(flood_path) as fl_src:
        flood = fl_src.read(1).astype("float32")
        fl_nd = fl_src.nodata

        needs_alignment = (
            flood.shape != ref_shape
            or fl_src.transform != ref_transform
            or fl_src.crs != ref_crs
        )

        if needs_alignment:
            print("  [sar] Aligning flood scene to pre-flood SAR grid...")
            aligned = np.full(ref_shape, np.nan, dtype="float32")
            reproject(
                source=flood,
                destination=aligned,
                src_transform=fl_src.transform,
                src_crs=fl_src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
            flood = aligned

    invalid = np.zeros(pre.shape, dtype=bool)

    if pre_nd is not None:
        invalid |= pre == pre_nd
    if fl_nd is not None:
        invalid |= flood == fl_nd

    invalid |= ~np.isfinite(pre) | ~np.isfinite(flood)

    # GEE Sentinel-1 GRD is already dB, so use direct dB difference.
    diff = flood - pre

    valid_diff = diff[~invalid]
    if valid_diff.size > 0:
        print(f"  [sar] Mean dB change: {float(np.mean(valid_diff)):.3f} dB")
        print(f"  [sar] Std dB change : {float(np.std(valid_diff)):.3f} dB")
        print(f"  [sar] Min/Max diff  : {float(np.min(valid_diff)):.3f} / {float(np.max(valid_diff)):.3f} dB")

    method = method.lower().strip()

    if method == "combined_db":
        dark_condition = flood < flood_vh_threshold
        change_condition = diff < db_difference_threshold

        dark_condition[invalid] = False
        change_condition[invalid] = False

        observed = (dark_condition & change_condition).astype("uint8")

        print("  [sar] Combined dB method")
        print(f"  [sar] Flood VH threshold      : {flood_vh_threshold:.2f} dB")
        print(f"  [sar] Difference threshold    : {db_difference_threshold:.2f} dB")
        print(f"  [sar] Pixels flood VH < threshold: {int(dark_condition.sum()):,}")
        print(f"  [sar] Pixels diff < threshold    : {int(change_condition.sum()):,}")
        print(f"  [sar] Pixels satisfying both     : {int(observed.sum()):,}")

    elif method == "difference":
        observed = (diff < db_difference_threshold).astype("uint8")
        observed[invalid] = 0

        print("  [sar] Difference dB method")
        print(f"  [sar] Difference threshold: {db_difference_threshold:.2f} dB")
        print(f"  [sar] Flooded pixels before water mask: {int(observed.sum()):,}")

    elif method == "ratio":
        # Ratio is only valid in linear scale. For your GEE dB exports,
        # convert dB to linear first, then compute ratio.
        pre_linear = np.power(10.0, pre / 10.0)
        flood_linear = np.power(10.0, flood / 10.0)

        ratio = np.where(pre_linear > 0, flood_linear / pre_linear, 1.0)
        ratio[invalid] = 1.0

        observed = (ratio < ratio_threshold).astype("uint8")
        observed[invalid] = 0

        print("  [sar] Ratio method using dB-to-linear conversion")
        print(f"  [sar] Ratio threshold: {ratio_threshold:.2f}")
        print(f"  [sar] Flooded pixels before water mask: {int(observed.sum()):,}")

    else:
        raise ValueError(
            f"Unsupported SAR method: {method}. "
            "Use 'combined_db', 'difference', or 'ratio'."
        )

    observed = mask_permanent_water(
        observed,
        landcover_path,
        water_class=80,
        reference_profile=profile,
    )

    final_px = int(observed.sum())
    pixel_km2 = abs(profile["transform"].a * profile["transform"].e) / 1_000_000
    flood_km2 = final_px * pixel_km2

    print(f"  [sar] Final flooded pixels: {final_px:,}")
    print(f"  [sar] Final flood area    : {flood_km2:.4f} km²")

    profile.update(dtype="uint8", count=1, nodata=0)

    if output_path.exists():
        output_path.unlink()

    save_raster(observed, profile, output_path, nodata=0)
    print(f"  [sar] Observed flood extent saved: {output_path.name}")

    return output_path

# ── Grid alignment for validation ────────────────────────────────────────────

def _align_to_model(
    observed_path: Path, model_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align observed SAR flood extent to the susceptibility model grid.
    Returns (model_binary, observed_binary) arrays of identical shape.
    """
    with rasterio.open(model_path) as msrc:
        model   = msrc.read(1)
        mprofile = msrc.profile.copy()
        mshape   = (msrc.height, msrc.width)

    with rasterio.open(observed_path) as osrc:
        obs = osrc.read(1).astype("float32")
        if obs.shape != mshape:
            aligned = np.zeros(mshape, dtype="float32")
            reproject(
                source=obs,
                destination=aligned,
                src_transform=osrc.transform,
                src_crs=osrc.crs,
                dst_transform=mprofile["transform"],
                dst_crs=mprofile["crs"],
                resampling=Resampling.nearest,
            )
            obs = aligned

    model_binary    = np.isin(model, [4, 5]).astype("uint8")
    observed_binary = (np.round(obs) == 1).astype("uint8")
    valid_mask      = model != 0
    return model_binary, observed_binary, valid_mask


# ── Validation metrics ────────────────────────────────────────────────────────

def validate_modelled_flood(
    modelled_class_path: str | Path,
    observed_flood_path: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """
    Compare modelled High/Very High susceptibility zones against
    the SAR-observed flood extent.

    WHAT EACH METRIC MEANS (plain language):

    Precision — of all pixels the MODEL said would flood,
                what fraction actually flooded according to SAR?
                High precision = few false alarms.

    Recall    — of all pixels SAR shows as flooded,
                what fraction did the MODEL correctly predict?
                High recall = few missed floods.

    F1 Score  — the balance between precision and recall.
                The single most useful accuracy number for flood maps.
                F1 > 0.5 is acceptable, > 0.65 is good.

    Overall   — raw percentage of pixels correctly classified.
    Accuracy    Can be misleadingly high if one class dominates.

    Kappa     — agreement adjusted for chance.
                > 0.4 = moderate, > 0.6 = substantial.

    INTERPRETING RESULTS FOR LOKOJA:
      The 2022 flood was one of Nigeria's worst — the SAR extent
      will be large. If F1 is moderate (0.4–0.6) the model is
      performing well given it uses no hydraulic routing.
      F1 > 0.6 is excellent for a GIS-only susceptibility model.
    """
    model_binary, observed_binary, valid = _align_to_model(
        _resolve(observed_flood_path),
        _resolve(modelled_class_path),
    )

    y_true = observed_binary[valid].ravel()
    y_pred = model_binary[valid].ravel()

    if y_true.size == 0:
        raise ValueError("No valid pixels for validation.")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    # Cohen's Kappa
    n      = len(y_true)
    po     = (tp + tn) / n
    pe     = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    kappa  = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    metrics = {
        "true_positive":    int(tp),
        "true_negative":    int(tn),
        "false_positive":   int(fp),
        "false_negative":   int(fn),
        "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "overall_accuracy": round(accuracy_score(y_true, y_pred), 4),
        "kappa":            round(kappa, 4),
        "flood_event":      "2022 Niger-Benue flood",
        "sar_polarization": "VH",
        "model_classes":    "4 (High) + 5 (Very High)",
    }

    output_csv = _resolve(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(output_csv, index=False)
    return df


# ── Comparison map ────────────────────────────────────────────────────────────

def make_comparison_raster(
    modelled_class_path: Path,
    observed_flood_path: Path,
    output_path: Path,
) -> Path:
    """
    Create a 4-class comparison raster showing:
      1 = True Negative  (model: not flood, SAR: not flood)  — grey
      2 = False Positive (model: flood,     SAR: not flood)  — yellow
      3 = False Negative (model: not flood, SAR: flood)      — red
      4 = True Positive  (model: flood,     SAR: flood)      — blue
    """
    model_b, obs_b, valid = _align_to_model(
        _resolve(observed_flood_path),
        _resolve(modelled_class_path),
    )

    comparison = np.zeros(model_b.shape, dtype="uint8")
    comparison[valid & (model_b == 0) & (obs_b == 0)] = 1   # TN
    comparison[valid & (model_b == 1) & (obs_b == 0)] = 2   # FP
    comparison[valid & (model_b == 0) & (obs_b == 1)] = 3   # FN
    comparison[valid & (model_b == 1) & (obs_b == 1)] = 4   # TP

    with rasterio.open(_resolve(modelled_class_path)) as src:
        profile = src.profile.copy()
    profile.update(dtype="uint8", count=1, nodata=0)

    output_path = _resolve(output_path)
    if output_path.exists():
        output_path.unlink()
    save_raster(comparison, profile, output_path, nodata=0)
    return output_path


# ── Summary printer ───────────────────────────────────────────────────────────

def print_validation_summary(metrics_df: pd.DataFrame) -> None:
    m = metrics_df.iloc[0]
    f1   = float(m["f1_score"])
    prec = float(m["precision"])
    rec  = float(m["recall"])
    acc  = float(m["overall_accuracy"])
    kap  = float(m["kappa"])
    tp   = int(m["true_positive"])
    tn   = int(m["true_negative"])
    fp   = int(m["false_positive"])
    fn   = int(m["false_negative"])

    print("\n" + "─" * 55)
    print("  SAR validation results — 2022 Lokoja flood")
    print("─" * 55)
    print(f"  {'Confusion matrix':}")
    print(f"                      SAR:flood  SAR:dry")
    print(f"  Model:flood (4, 5)   {tp:>8,}  {fp:>8,}   (TP / FP)")
    print(f"  Model:dry  (1-3)    {fn:>8,}  {tn:>8,}   (FN / TN)")
    print()
    print(f"  {'Metric':<22} {'Value':>8}  {'Interpretation'}")
    print(f"  {'─'*52}")
    print(f"  {'Precision':<22} {prec:>8.4f}  {_interpret(prec, 'precision')}")
    print(f"  {'Recall':<22} {rec:>8.4f}  {_interpret(rec, 'recall')}")
    print(f"  {'F1 Score':<22} {f1:>8.4f}  {_interpret(f1, 'f1')}")
    print(f"  {'Overall Accuracy':<22} {acc:>8.4f}")
    print(f"  {'Kappa':<22} {kap:>8.4f}  {_interpret(kap, 'kappa')}")
    print()

    if f1 >= 0.65:
        verdict = "Excellent — model performs very well for a GIS susceptibility model"
    elif f1 >= 0.50:
        verdict = "Good — acceptable for terrain-based susceptibility without hydraulics"
    elif f1 >= 0.35:
        verdict = "Moderate — consider reviewing weight configuration in config.yaml"
    else:
        verdict = "Low — check SAR threshold and susceptibility class boundaries"

    print(f"  Verdict: {verdict}")
    print(f"\n  Note: F1 > 0.5 is the benchmark for GIS-only flood susceptibility")
    print(f"  models validated against SAR without hydraulic routing.")


def _interpret(val: float, metric: str) -> str:
    if metric == "precision":
        return "few false alarms" if val >= 0.6 else "some over-prediction"
    if metric == "recall":
        return "few missed floods" if val >= 0.6 else "some under-prediction"
    if metric == "f1":
        if val >= 0.65: return "excellent"
        if val >= 0.50: return "good"
        if val >= 0.35: return "moderate"
        return "low"
    if metric == "kappa":
        if val >= 0.60: return "substantial agreement"
        if val >= 0.40: return "moderate agreement"
        return "fair agreement"
    return ""


# ── Main workflow ─────────────────────────────────────────────────────────────

def run_sar_validation(config: dict) -> None:
    """
    Full Sentinel-1 SAR validation workflow.

    Stages:
      1. Locate SAR pre-flood and flood-date GeoTIFFs
      2. Derive observed flood extent via dB difference change detection
      3. Mask permanent water bodies using land cover
      4. Align SAR extent to susceptibility model grid
      5. Compare model High/Very High zones vs SAR flood extent
      6. Compute precision, recall, F1, accuracy, kappa
      7. Save metrics CSV, comparison raster, preview maps
    """
    print("\n[sar] Starting Sentinel-1 SAR validation")

    pre, flood = _find_sar_pair(config)
    if pre is None or flood is None:
        print("[sar] SAR files not found.")
        print("  Place scenes in:")
        print("    data/raw/sentinel1/pre_flood/    <- September 2022")
        print("    data/raw/sentinel1/during_flood/ <- October 2022")
        return

    print(f"  Pre-flood scene : {pre.name}")
    print(f"  Flood scene     : {flood.name}")

    paths   = config["paths"]
    rasters = ensure_dir(_resolve(paths["processed_rasters"]))
    val_dir = ensure_dir(_resolve(paths["outputs_validation"]))
    maps    = ensure_dir(_resolve(paths["outputs_maps"]))

    modelled = rasters / "susceptibility_class.tif"
    if not modelled.exists():
        print("[sar] susceptibility_class.tif not found — run script 03 first.")
        return

    lc_path = _resolve(paths["interim_reprojected"]) / "landcover.tif"
    if not lc_path.exists():
        lc_path = None
        print("  [sar] Land cover not available — permanent water masking skipped")

    sar_cfg = config.get("sar_validation", {})
    method = sar_cfg.get("method", "combined_db")
    ratio_threshold = float(sar_cfg.get("ratio_threshold", 0.7))
    flood_vh_threshold = float(sar_cfg.get("flood_vh_threshold", -16.5))
    db_difference_threshold = float(sar_cfg.get("db_difference_threshold", -1.0))

    # All outputs go directly to processed/rasters — no intermediate copy
    observed_path   = rasters / "observed_flood_extent.tif"
    comparison_path = rasters / "model_vs_observed_comparison.tif"
    metrics_csv     = val_dir  / "validation_metrics.csv"

    # Clean stale outputs
    for p in [observed_path, comparison_path, metrics_csv]:
        if p.exists():
            p.unlink()

    # Stage 1
    print(f"\n[sar] Stage 1/4 — Deriving flood extent ({method} method)")
    derive_flood_extent_from_sar(
        pre, flood, observed_path,
        landcover_path=lc_path,
        method=method,
        ratio_threshold=ratio_threshold,
    )

    if not observed_path.exists():
        print("[sar] ERROR: flood extent file was not created.")
        print(f"  Expected: {observed_path}")
        return

    # Stage 2
    print("\n[sar] Stage 2/4 — Validating against susceptibility model")
    metrics_df = validate_modelled_flood(modelled, observed_path, metrics_csv)
    print(f"  Metrics saved: {metrics_csv.name}")

    # Stage 3
    print("\n[sar] Stage 3/4 — Building comparison map")
    make_comparison_raster(modelled, observed_path, comparison_path)
    print(f"  Comparison raster saved: {comparison_path.name}")

    # Stage 4
    print("\n[sar] Stage 4/4 — Saving preview maps")
    plot_raster_preview(
        observed_path,
        maps / "observed_flood_preview.png",
        "Lokoja — SAR observed flood extent (Oct 2022)",
    )
    plot_raster_preview(
        comparison_path,
        maps / "model_vs_sar_comparison.png",
        "Lokoja — Model vs SAR (1=TN  2=FP  3=FN  4=TP)",
    )

    print_validation_summary(metrics_df)

    print("\n[sar] Complete")
    print(f"  Metrics    : {metrics_csv}")
    print(f"  Comparison : {comparison_path}")
    print("\n  Next step: python scripts/05_run_exposure_analysis.py")
