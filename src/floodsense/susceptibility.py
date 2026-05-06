"""Flood susceptibility modelling for FloodSense Lokoja."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from floodsense.mapping import plot_raster_preview, save_bar_chart
from floodsense.paths import ensure_dir, get_project_root
from floodsense.raster_utils import save_raster


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else get_project_root() / path


# ── Land cover risk reclassification ─────────────────────────────────────────

# ESA WorldCover class codes → flood risk score (0–1)
# Higher score = greater flood risk contribution
LANDCOVER_RISK = {
    10:  0.10,   # Tree cover      — absorbs water, low runoff
    20:  0.25,   # Shrubland       — moderate infiltration
    30:  0.40,   # Grassland       — moderate runoff
    40:  0.55,   # Cropland        — seasonal water, moderate-high risk
    50:  0.85,   # Built-up        — impervious surface, high runoff
    60:  0.45,   # Bare/sparse     — moderate runoff
    70:  0.10,   # Snow/ice        — not relevant for Nigeria
    80:  1.00,   # Permanent water — already flooded
    90:  0.90,   # Herbaceous wetland — very high
    95:  0.85,   # Mangroves       — high
    100: 0.20,   # Moss/lichen     — not relevant for Nigeria
}


# ── Factor normalization ──────────────────────────────────────────────────────

def normalize_flood_factor(
    array: np.ndarray, factor_name: str, nodata: float | int | None = None
) -> np.ndarray:
    """
    Normalize a factor array to 0–1 where 1 = highest flood risk.

    Inversion logic (higher raw value = LOWER risk → invert to 0–1):
      elevation        : high ground = low risk → invert
      slope            : steep = fast drainage = low risk → invert
      distance_to_river: far from river = low risk → invert
      distance_to_stream: far from stream = low risk → invert

    Forward (higher raw value = HIGHER risk):
      flow_accumulation : high convergence = high risk
      rainfall          : heavy rain = high risk
      hand              : low HAND = high risk → invert

    Special handling:
      landcover         : reclassified using LANDCOVER_RISK lookup table
    """
    arr = array.astype("float32", copy=True)

    # Replace nodata with NaN
    if nodata is not None:
        arr[arr == nodata] = np.nan

    # Land cover — reclassify by ESA WorldCover code
    if factor_name == "landcover":
        risk = np.full(arr.shape, np.nan, dtype="float32")
        for code, score in LANDCOVER_RISK.items():
            risk[np.round(array).astype(int) == code] = score
        if np.isfinite(risk).any():
            return risk
        print("[susceptibility] Land cover codes not recognised — using generic normalization")

    # Flow accumulation — log-transform before normalizing (skewed distribution)
    if factor_name == "flow_accumulation":
        arr = np.log10(np.where(np.isfinite(arr) & (arr > 0), arr, np.nan) + 1)

    # HAND — low value = high risk, so invert
    if factor_name == "hand":
        factor_name = "distance_to_river"   # triggers inversion below

    inverse_factors = {"elevation", "slope", "distance_to_river", "distance_to_stream"}
    inverse = factor_name in inverse_factors

    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(arr.shape, np.nan, dtype="float32")

    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if np.isclose(mx, mn):
        return np.zeros(arr.shape, dtype="float32")

    norm = ((arr - mn) / (mx - mn)).astype("float32")
    if inverse:
        norm = 1.0 - norm
    norm[~finite] = np.nan
    return norm


# ── Grid alignment ────────────────────────────────────────────────────────────

def align_to_reference(
    src_array: np.ndarray,
    src_profile: dict,
    ref_profile: dict,
) -> np.ndarray:
    """
    Resample src_array to match the reference grid exactly.
    Used when a factor raster has a different resolution than the DEM
    (e.g. land cover at 10m, population at 100m vs DEM at 30m).
    """
    destination = np.full(
        (ref_profile["height"], ref_profile["width"]),
        np.nan,
        dtype="float32",
    )
    reproject(
        source=src_array.astype("float32"),
        destination=destination,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        resampling=Resampling.bilinear,
    )
    return destination


# ── Weighted overlay ──────────────────────────────────────────────────────────

def calculate_weighted_susceptibility(
    factor_arrays: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """
    Combine normalized factor arrays into a single susceptibility score.

    Each factor is multiplied by its weight. Weights are renormalized
    to the factors actually available — so if rainfall is missing, its
    weight is redistributed proportionally among the other factors.

    Returns a float32 array with values 0–1 (NaN where all factors are NaN).
    """
    if not factor_arrays:
        raise ValueError("No factor arrays provided.")

    # Build mask: pixel is valid if at least one factor has a value there
    ref_shape = next(iter(factor_arrays.values())).shape
    score      = np.zeros(ref_shape, dtype="float32")
    weight_sum = 0.0

    for name, array in factor_arrays.items():
        w = float(weights.get(name, 0.0))
        if w <= 0:
            print(f"  [susceptibility] No weight configured for '{name}' — skipping")
            continue
        clean = np.nan_to_num(array.astype("float32"), nan=0.0)
        score      += clean * w
        weight_sum += w

    if weight_sum == 0:
        raise ValueError("All factor weights are zero.")

    # Renormalize by actual weight used
    score = np.clip(score / weight_sum, 0.0, 1.0).astype("float32")

    # Mask pixels where all inputs were NaN
    all_nan = np.all(
        [~np.isfinite(a) for a in factor_arrays.values()], axis=0
    )
    score[all_nan] = np.nan
    return score


# ── Classification ────────────────────────────────────────────────────────────

def classify_susceptibility(
    score_array: np.ndarray, method: str = "quantile"
) -> np.ndarray:
    """
    Classify 0–1 susceptibility scores into 5 risk classes.

      1 — Very Low
      2 — Low
      3 — Moderate
      4 — High
      5 — Very High

    method='quantile' (default): each class covers ~20% of valid pixels.
      Recommended — produces a balanced, informative map regardless of
      how the score distribution is shaped.

    method='equal': fixed breaks at 0.2, 0.4, 0.6, 0.8.
      Useful when you want class boundaries to have absolute meaning.
    """
    valid = score_array[np.isfinite(score_array)]
    if len(valid) == 0:
        return np.zeros(score_array.shape, dtype="uint8")

    if method == "quantile":
        breaks = np.percentile(valid, [20, 40, 60, 80])
    else:
        breaks = np.array([0.2, 0.4, 0.6, 0.8])

    classified = np.zeros(score_array.shape, dtype="uint8")
    classified[np.isfinite(score_array) & (score_array <= breaks[0])]                               = 1
    classified[np.isfinite(score_array) & (score_array > breaks[0]) & (score_array <= breaks[1])]  = 2
    classified[np.isfinite(score_array) & (score_array > breaks[1]) & (score_array <= breaks[2])]  = 3
    classified[np.isfinite(score_array) & (score_array > breaks[2]) & (score_array <= breaks[3])]  = 4
    classified[np.isfinite(score_array) & (score_array > breaks[3])]                               = 5
    return classified


# ── Area statistics ───────────────────────────────────────────────────────────

def calculate_class_area(
    class_raster_path: str | Path, output_csv: str | Path
) -> pd.DataFrame:
    """Calculate area in km² per susceptibility class and save to CSV."""
    with rasterio.open(class_raster_path) as src:
        data = src.read(1)
        res  = src.res       # (pixel_width, pixel_height) in CRS units
        pixel_area_km2 = abs(res[0] * res[1]) / 1_000_000

    labels = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
    rows = []
    for val, label in labels.items():
        cells = int((data == val).sum())
        rows.append({
            "class_value": val,
            "class_name":  label,
            "pixel_count": cells,
            "area_km2":    round(cells * pixel_area_km2, 2),
            "pct_of_total": 0.0,
        })

    df = pd.DataFrame(rows)
    total_area = df["area_km2"].sum()
    if total_area > 0:
        df["pct_of_total"] = (df["area_km2"] / total_area * 100).round(1)

    output_csv = _resolve(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


# ── Factor path resolution ────────────────────────────────────────────────────

def _factor_paths(config: dict) -> dict[str, Path]:
    """Locate all available factor rasters from processed and interim folders."""
    rasters = _resolve(config["paths"]["processed_rasters"])
    interim = _resolve(config["paths"]["interim_reprojected"])

    candidates = {
        "elevation":         [rasters / "dem_filled.tif",          interim / "dem.tif"],
        "slope":             [rasters / "slope.tif"],
        "flow_accumulation": [rasters / "flow_accumulation.tif"],
        "distance_to_river": [rasters / "distance_to_stream.tif",  rasters / "distance_to_river.tif"],
        "hand":              [rasters / "hand.tif"],
        "rainfall":          [interim / "rainfall.tif",             rasters / "rainfall_mean.tif"],
        "landcover":         [interim / "landcover.tif"],
    }

    found, missing = {}, []
    for name, paths in candidates.items():
        for p in paths:
            if p.exists():
                found[name] = p
                break
        else:
            missing.append(name)

    if missing:
        print(f"  [susceptibility] Factors not found (will be skipped): {missing}")
    return found


# ── Summary printer ───────────────────────────────────────────────────────────

def print_susceptibility_summary(
    factor_arrays: dict,
    weights: dict,
    df_area: pd.DataFrame,
    out_dir: Path,
) -> None:
    print("\n" + "─" * 60)
    print("  Susceptibility model — inputs")
    print("─" * 60)
    print(f"  {'Factor':<22} {'Weight':>7}  {'Used':>5}  {'Min':>6}  {'Max':>6}")
    print(f"  {'─'*55}")
    for name, arr in factor_arrays.items():
        w     = weights.get(name, 0.0)
        valid = arr[np.isfinite(arr)]
        mn    = f"{valid.min():.3f}" if len(valid) else "n/a"
        mx    = f"{valid.max():.3f}" if len(valid) else "n/a"
        print(f"  {name:<22} {w:>7.2f}  {'yes':>5}  {mn:>6}  {mx:>6}")

    print("\n" + "─" * 60)
    print("  Susceptibility class areas")
    print("─" * 60)
    print(f"  {'Class':<12} {'Label':<12} {'Area km²':>10}  {'%':>6}")
    print(f"  {'─'*44}")
    for _, row in df_area.iterrows():
        print(
            f"  {row['class_value']:<12} {row['class_name']:<12} "
            f"{row['area_km2']:>10.1f}  {row['pct_of_total']:>5.1f}%"
        )

    # Sanity check — high+very high combined
    high_area = df_area.loc[df_area["class_value"].isin([4, 5]), "area_km2"].sum()
    high_pct  = df_area.loc[df_area["class_value"].isin([4, 5]), "pct_of_total"].sum()
    print(f"\n  High + Very High combined: {high_area:.1f} km²  ({high_pct:.1f}%)")
    print("  (Expected: ~15–35% for a flood-prone confluence city like Lokoja)")

    score_path = out_dir / "susceptibility_score.tif"
    if score_path.exists():
        with rasterio.open(score_path) as src:
            s = src.read(1, masked=True)
            print(f"\n  Score stats: min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}")


# ── Main workflow ─────────────────────────────────────────────────────────────

def run_susceptibility_workflow(config: dict) -> None:
    """
    Build flood susceptibility score and classified risk rasters.

    Stages:
      1. Locate available factor rasters
      2. Load and normalize each factor to 0–1 (higher = more flood risk)
      3. Resample factors to match the DEM reference grid
      4. Apply weighted overlay → raw susceptibility score (0–1)
      5. Classify into 5 risk levels using quantile breaks
      6. Calculate area statistics per class
      7. Save outputs and preview maps
    """
    print("\n[susceptibility] Starting flood susceptibility modelling")

    factor_paths = _factor_paths(config)
    if not factor_paths:
        print("[susceptibility] No factor rasters found.")
        print("  Run scripts 01 and 02 first.")
        return

    # Use filled DEM as the reference grid
    reference_path = factor_paths.get("elevation") or next(iter(factor_paths.values()))
    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile.copy()
        ref_shape   = (ref.height, ref.width)

    print(f"\n  Reference grid : {reference_path.name}")
    print(f"  Grid size      : {ref_shape[1]} × {ref_shape[0]} px")
    print(f"  CRS            : {ref_profile['crs']}")
    print(f"\n  Loading and normalizing {len(factor_paths)} factor(s)...")

    weights       = config.get("susceptibility_weights", {})
    factor_arrays = {}

    for name, path in factor_paths.items():
        with rasterio.open(path) as src:
            src_profile = src.profile.copy()
            array       = src.read(1).astype("float32")
            nodata      = src.nodata

        # Resample to reference grid if needed
        if (src_profile["height"], src_profile["width"]) != ref_shape:
            print(f"  Resampling '{name}' from {src_profile['width']}×{src_profile['height']} → {ref_shape[1]}×{ref_shape[0]}")
            array = align_to_reference(array, src_profile, ref_profile)

        normed = normalize_flood_factor(array, name, nodata)
        factor_arrays[name] = normed
        valid  = normed[np.isfinite(normed)]
        print(f"  ✓  {name:<22} weight={weights.get(name, 0.0):.2f}  "
              f"valid={len(valid):,}  range=[{valid.min():.3f}, {valid.max():.3f}]")

    # Weighted overlay
    print("\n  Running weighted overlay...")
    score = calculate_weighted_susceptibility(factor_arrays, weights)

    # Classify
    print("  Classifying into 5 risk levels (quantile breaks)...")
    classified = classify_susceptibility(score, method="quantile")

    # Output paths
    rasters_dir = ensure_dir(_resolve(config["paths"]["processed_rasters"]))
    tables_dir  = ensure_dir(_resolve(config["paths"]["processed_tables"]))
    maps_dir    = ensure_dir(_resolve(config["paths"]["outputs_maps"]))
    vectors_dir = ensure_dir(_resolve(config["paths"]["processed_vectors"]))

    # Save score raster
    score_nodata = -9999.0
    score_saved  = np.where(np.isfinite(score), score, score_nodata)
    score_profile = ref_profile.copy()
    score_profile.update(dtype="float32", count=1, nodata=score_nodata, compress="deflate")
    score_path = rasters_dir / "susceptibility_score.tif"
    if score_path.exists():
        score_path.unlink()
    save_raster(score_saved.astype("float32"), score_profile, score_path, nodata=score_nodata)
    print(f"  Saved: {score_path.name}")

    # Save classified raster
    class_profile = ref_profile.copy()
    class_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    class_path = rasters_dir / "susceptibility_class.tif"
    if class_path.exists():
        class_path.unlink()
    save_raster(classified.astype("uint8"), class_profile, class_path, nodata=0)
    print(f"  Saved: {class_path.name}")

    # Area statistics
    df_area   = calculate_class_area(class_path, tables_dir / "susceptibility_class_area.csv")
    print(f"  Saved: susceptibility_class_area.csv")

    # Preview maps
    plot_raster_preview(score_path, maps_dir / "susceptibility_score_preview.png",
                        "Lokoja — Flood susceptibility score (0–1)")
    plot_raster_preview(class_path, maps_dir / "susceptibility_class_preview.png",
                        "Lokoja — Flood susceptibility class (1=Very Low, 5=Very High)")
    save_bar_chart(df_area, "class_name", "area_km2",
                   maps_dir / "susceptibility_area_chart.png",
                   "Area by flood susceptibility class (km²)")

    # Summary
    print_susceptibility_summary(factor_arrays, weights, df_area, rasters_dir)

    print("\n[susceptibility] ✓ Complete")
    print("  Outputs:")
    print(f"    {rasters_dir / 'susceptibility_score.tif'}")
    print(f"    {rasters_dir / 'susceptibility_class.tif'}")
    print(f"    {tables_dir  / 'susceptibility_class_area.csv'}")
    print(f"    {maps_dir    / 'susceptibility_class_preview.png'}")
    print("\n  Next step: python scripts/04_run_sar_validation.py")
    print("  (Or skip to exposure: python scripts/05_run_exposure_analysis.py)")