"""
Inspect downloaded input datasets before running the pipeline.
Prints CRS, resolution, extent, and basic stats for each file.
Run this first to confirm all inputs are ready.

Usage:
    python scripts/inspect_inputs.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio
import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"{'═' * 55}")

def ok(msg: str) -> None:
    print(f"  ✓  {msg}")

def warn(msg: str) -> None:
    print(f"  ⚠  {msg}")

def info(label: str, value) -> None:
    print(f"     {label:<22}: {value}")

TARGET_CRS = "EPSG:32632"


# ── Vector inspection ─────────────────────────────────────────────────────────

def inspect_vector(path: Path, label: str) -> dict:
    section(f"Vector — {label}")
    if not path.exists():
        warn(f"NOT FOUND: {path}")
        return {}

    gdf = gpd.read_file(path)
    crs = str(gdf.crs)
    is_projected = gdf.crs is not None and gdf.crs.is_projected
    bounds = gdf.total_bounds        # [minx, miny, maxx, maxy]
    n_features = len(gdf)
    invalid = (~gdf.geometry.is_valid).sum()
    empty_geoms = gdf.geometry.is_empty.sum()

    ok(f"File found: {path.name}")
    info("Features", n_features)
    info("CRS", crs)
    info("Is projected", is_projected)
    info("Columns", list(gdf.columns))
    info("Bounds (minx)", f"{bounds[0]:.4f}")
    info("Bounds (miny)", f"{bounds[1]:.4f}")
    info("Bounds (maxx)", f"{bounds[2]:.4f}")
    info("Bounds (maxy)", f"{bounds[3]:.4f}")
    info("Invalid geometries", invalid)
    info("Empty geometries", empty_geoms)

    # Area in km² (use projected copy if not already)
    proj = gdf.to_crs(TARGET_CRS)
    area_km2 = proj.geometry.area.sum() / 1_000_000
    info("Area (km²)", f"{area_km2:.2f}")

    # Sanity checks
    if invalid > 0:
        warn(f"{invalid} invalid geometry(ies) — will be fixed in preprocessing")
    if not is_projected:
        warn("Not in a projected CRS — preprocessing will reproject to EPSG:32632")
    else:
        if TARGET_CRS not in crs:
            warn(f"CRS is {crs} — preprocessing will reproject to {TARGET_CRS}")
        else:
            ok("CRS matches target EPSG:32632")

    if 6.0 < bounds[0] < 7.5 and 7.0 < bounds[1] < 8.5:
        ok("Extent looks correct for Lokoja area")
    else:
        warn("Extent is outside expected Lokoja range — check this file")

    return {"path": path, "crs": crs, "features": n_features, "area_km2": area_km2}


# ── Raster inspection ─────────────────────────────────────────────────────────

def inspect_raster(path: Path, label: str, expected_unit: str = "") -> dict:
    section(f"Raster — {label}")
    if not path.exists():
        warn(f"NOT FOUND: {path}")
        return {}

    with rasterio.open(path) as src:
        crs       = str(src.crs)
        res_x     = abs(src.transform.a)
        res_y     = abs(src.transform.e)
        width     = src.width
        height    = src.height
        bands     = src.count
        dtype     = src.dtypes[0]
        nodata    = src.nodata
        bounds    = src.bounds
        data      = src.read(1, masked=True)

    is_projected = src.crs is not None and src.crs.is_projected

    ok(f"File found: {path.name}")
    info("CRS", crs)
    info("Is projected", is_projected)
    info("Resolution (x)", f"{res_x:.2f}")
    info("Resolution (y)", f"{res_y:.2f}")
    info("Dimensions", f"{width} × {height} px")
    info("Bands", bands)
    info("Dtype", dtype)
    info("NoData value", nodata)
    info("Bounds west",  f"{bounds.left:.4f}")
    info("Bounds south", f"{bounds.bottom:.4f}")
    info("Bounds east",  f"{bounds.right:.4f}")
    info("Bounds north", f"{bounds.top:.4f}")

    valid = data.compressed()
    if len(valid) > 0:
        info("Value min", f"{valid.min():.4f}")
        info("Value max", f"{valid.max():.4f}")
        info("Value mean", f"{valid.mean():.4f}")
        info("NoData pixels", f"{data.mask.sum():,}")
        info("Valid pixels",  f"{len(valid):,}")
    else:
        warn("All pixels are NoData — check this file")

    if expected_unit:
        info("Expected unit", expected_unit)

    # Checks
    if not is_projected:
        warn(f"Not projected — preprocessing will reproject to {TARGET_CRS}")
    else:
        if TARGET_CRS not in crs:
            warn(f"Projected but not EPSG:32632 — will be reprojected")
        else:
            ok("CRS matches target EPSG:32632")

    if abs(res_x - 30) < 5:
        ok(f"Resolution ~30m — good for DEM/analysis")
    elif abs(res_x - 100) < 20:
        ok(f"Resolution ~100m — typical for population raster")
    elif abs(res_x - 10) < 3:
        ok(f"Resolution ~10m — good for land cover")
    else:
        info("Resolution note", f"{res_x:.1f}m — will be resampled if needed")

    return {
        "path": path, "crs": crs, "res_x": res_x,
        "width": width, "height": height, "nodata": nodata,
        "min": float(valid.min()) if len(valid) else None,
        "max": float(valid.max()) if len(valid) else None,
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    section("Summary — readiness for pipeline")

    checks = {
        "Boundary (.gpkg)":    results.get("boundary"),
        "DEM (.tif)":          results.get("dem"),
        "Population (.tif)":   results.get("population"),
        "Land cover (.tif)":   results.get("landcover"),
    }

    all_found = True
    print(f"\n  {'Dataset':<25} {'Found':>6}  {'CRS ok':>7}  {'Notes'}")
    print(f"  {'─'*60}")

    for name, r in checks.items():
        if not r:
            print(f"  {name:<25} {'✗':>6}  {'—':>7}  File missing")
            all_found = False
            continue
        crs_ok = TARGET_CRS in r.get("crs", "")
        print(
            f"  {name:<25} {'✓':>6}  "
            f"{'✓' if crs_ok else '⚠':>7}  "
            f"{'Ready' if crs_ok else 'Will reproject in script 01'}"
        )

    print()
    if all_found:
        ok("All four core datasets found — ready to run script 01_prepare_data.py")
        print("""
  Recommended next command:
    python scripts/01_prepare_data.py

  Still needed before full pipeline:
    data/raw/roads/         — osmnx auto-downloads in script 01
    data/raw/buildings/     — osmnx auto-downloads in script 01
    data/raw/sentinel1/     — GEE export needed for stage 04
        """)
    else:
        warn("Some datasets are missing — see notes above before proceeding")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print("  FloodSense Lokoja — Input dataset inspection")
    print("=" * 55)

    root = Path(__file__).resolve().parents[1]

    # Locate files — first match in each raw folder
    def first_file(folder: str, exts: list[str]) -> Path:
        d = root / folder
        for ext in exts:
            matches = sorted(d.glob(f"*{ext}"))
            if matches:
                return matches[0]
        return root / folder / "NOT_FOUND"

    boundary_path  = first_file("data/raw/boundary",   [".gpkg", ".geojson", ".shp"])
    dem_path       = first_file("data/raw/dem",        [".tif", ".tiff"])
    population_path = first_file("data/raw/population", [".tif", ".tiff"])
    landcover_path  = first_file("data/raw/landcover",  [".tif", ".tiff"])

    results = {}
    results["boundary"]  = inspect_vector(boundary_path,   "Lokoja LGA boundary")
    results["dem"]       = inspect_raster(dem_path,        "DEM (elevation)", "metres above sea level")
    results["population"] = inspect_raster(population_path,"Population (WorldPop)", "people per pixel")
    results["landcover"] = inspect_raster(landcover_path,  "Land cover (ESA WorldCover)", "class code 10–100")

    print_summary(results)


if __name__ == "__main__":
    main()
