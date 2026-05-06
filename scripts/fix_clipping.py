"""
Fix raster clipping — re-clips DEM and land cover properly to boundary.

Run this if inspect_inputs.py or visual inspection shows that the
rasters have large nodata areas or were only bbox-cropped.

What this does differently from script 01:
  - Uses filled=True in rasterio.mask so pixels OUTSIDE the boundary
    polygon are set to nodata (not just cropped to the bounding box)
  - Sets a proper nodata value in the output profile
  - Verifies the result and prints nodata statistics

Usage:
    python scripts/fix_clipping.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from floodsense.config import load_config, get_crs
from floodsense.paths import ensure_dir, first_existing_file, get_project_root
from floodsense.raster_utils import (
    check_clip_quality,
    clip_raster_to_boundary,
    reproject_raster,
)
from floodsense.vector_utils import (
    clean_geometries,
    dissolve_boundary,
    read_vector,
    reproject_vector,
)

ROOT = get_project_root()
RASTER_EXTS = [".tif", ".tiff"]


def section(title: str) -> None:
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"{'═' * 55}")


def load_boundary(config: dict):
    """Load, clean and dissolve the boundary in geographic CRS."""
    paths  = config["paths"]
    src    = first_existing_file(paths["raw_boundary"], [".gpkg", ".geojson", ".shp"])
    if src is None:
        raise FileNotFoundError("No boundary file in data/raw/boundary/")
    gdf = read_vector(src)
    gdf = clean_geometries(gdf)
    gdf = dissolve_boundary(gdf)
    print(f"  Boundary loaded: {src.name}  ({len(gdf)} feature, CRS={gdf.crs})")
    return gdf


def reclip_and_reproject(
    raw_path: Path,
    boundary_geo,
    clip_out: Path,
    reproj_out: Path,
    target_crs: str,
    label: str,
) -> None:
    section(f"Fixing: {label}")

    print(f"  Source : {raw_path.name}")
    with rasterio.open(raw_path) as src:
        print(f"  CRS    : {src.crs}")
        print(f"  Size   : {src.width} × {src.height} px")
        nodata = src.nodata
        data   = src.read(1)
        if nodata is not None:
            nd_before = int((data == nodata).sum())
            print(f"  NoData before clip: {nd_before:,} ({nd_before/data.size*100:.1f}%)")
        else:
            print(f"  NoData before clip: no nodata value set in source")

    # Step 1: clip to boundary polygon (polygon mask, not just bbox)
    print(f"\n  Step 1: Clipping to boundary polygon...")
    ensure_dir(clip_out.parent)
    clip_raster_to_boundary(raw_path, boundary_geo, clip_out)
    check_clip_quality(clip_out, f"{label} (clipped)")

    # Step 2: reproject to target CRS
    print(f"\n  Step 2: Reprojecting to {target_crs}...")
    ensure_dir(reproj_out.parent)
    reproject_raster(clip_out, reproj_out, target_crs)
    check_clip_quality(reproj_out, f"{label} (reprojected)")

    print(f"\n  ✓ Done → {reproj_out.name}")


def main() -> None:
    print("=" * 55)
    print("  FloodSense — Fix raster clipping")
    print("=" * 55)

    config     = load_config()
    paths      = config["paths"]
    target_crs = get_crs(config)

    boundary = load_boundary(config)

    # DEM
    dem_raw = first_existing_file(paths["raw_dem"], RASTER_EXTS)
    if dem_raw:
        reclip_and_reproject(
            dem_raw,
            boundary,
            ROOT / paths["interim_clipped"]    / "dem_clipped.tif",
            ROOT / paths["interim_reprojected"] / "dem.tif",
            target_crs,
            "DEM",
        )
    else:
        print("\n  ⚠ DEM not found in data/raw/dem/ — skipping")

    # Land cover — check both folder names
    lc_raw = first_existing_file(paths["raw_landcover"], RASTER_EXTS)
    if lc_raw is None:
        lc_raw = first_existing_file("data/raw/land_cover", RASTER_EXTS)
    if lc_raw:
        reclip_and_reproject(
            lc_raw,
            boundary,
            ROOT / paths["interim_clipped"]    / "landcover_clipped.tif",
            ROOT / paths["interim_reprojected"] / "landcover.tif",
            target_crs,
            "Land cover",
        )
    else:
        print("\n  ⚠ Land cover not found — skipping")

    # Population
    pop_raw = first_existing_file(paths["raw_population"], RASTER_EXTS)
    if pop_raw:
        reclip_and_reproject(
            pop_raw,
            boundary,
            ROOT / paths["interim_clipped"]    / "population_clipped.tif",
            ROOT / paths["interim_reprojected"] / "population.tif",
            target_crs,
            "Population",
        )
    else:
        print("\n  ⚠ Population not found in data/raw/population/ — skipping")

    # Final check
    section("Final clip quality check")
    for name in ["dem.tif", "landcover.tif", "population.tif"]:
        p = ROOT / paths["interim_reprojected"] / name
        check_clip_quality(p, name)

    print(f"\n  All fixed files are in: data/interim/reprojected/")
    print(f"  Next step: python scripts/02_run_hydrology.py")


if __name__ == "__main__":
    main()
