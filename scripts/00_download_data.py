"""
Script 00 — Boundary download and project setup
================================================
First script to run. Sets up all project directories
and downloads the Lokoja LGA boundary from your chosen source.

The boundary file is required before any other script runs.

BOUNDARY DOWNLOAD OPTIONS
--------------------------
Option A — geoBoundaries (easiest, no login needed):
  curl -L "https://www.geoboundaries.org/api/current/gbOpen/NGA/ADM2/" \
       -o data/raw/boundary/nga_adm2.geojson

Option B — GADM (most detailed):
  1. Go to https://gadm.org/download_country.html
  2. Select Nigeria → Shapefile
  3. Extract gadm41_NGA_2.* into data/raw/boundary/

Option C — HDX (humanitarian):
  https://data.humdata.org/dataset/cod-ab-nga
  Download nga_admbnda_adm2_osgof.zip → extract into data/raw/boundary/

After placing the file, run this script to extract and save Lokoja LGA.

Usage:
  python scripts/00_download_data.py

Output:
  data/interim/cleaned/lokoja_boundary.gpkg     (projected, for analysis)
  data/interim/cleaned/lokoja_boundary.geojson  (geographic, for web/GEE)
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from floodsense.config import load_config, get_crs
from floodsense.paths import ensure_project_directories, list_input_files
from floodsense.vector_utils import clean_geometries, dissolve_boundary, reproject_vector

TARGET_LGA   = "Lokoja"
TARGET_STATE = "Kogi"


def find_boundary_file(raw_dir: Path) -> Path | None:
    """Find any shapefile or GeoJSON in the boundary raw folder."""
    candidates = list_input_files(raw_dir, [".shp", ".geojson", ".json", ".gpkg"])
    return candidates[0] if candidates else None


def extract_lokoja(source_path: Path) -> gpd.GeoDataFrame:
    """Search all text columns for Lokoja and return the matching feature(s)."""
    gdf = gpd.read_file(source_path)
    print(f"  Loaded {len(gdf)} features from {source_path.name}")
    print(f"  Columns: {list(gdf.columns)}")

    for col in gdf.select_dtypes(include="object").columns:
        matches = gdf[gdf[col].str.contains(TARGET_LGA, case=False, na=False)]
        if not matches.empty:
            print(f"  Found '{TARGET_LGA}' in column '{col}' — {len(matches)} feature(s)")
            return matches.copy()

    raise ValueError(
        f"Could not find '{TARGET_LGA}' in any column of {source_path.name}.\n"
        f"Check the file contains Kogi State LGA boundaries."
    )


def main() -> None:
    print("=" * 55)
    print("  FloodSense — Stage 00: Boundary download and setup")
    print("=" * 55)

    config = load_config()
    ensure_project_directories(config)
    print("  Project directories confirmed.")

    crs   = get_crs(config)
    raw   = Path(config["paths"]["raw_boundary"])
    out   = Path(config["paths"]["interim_cleaned"])

    # Locate boundary file
    src = find_boundary_file(raw)
    if src is None:
        print(f"""
  No boundary file found in {raw}/

  Download one using the instructions at the top of this script.
  Then re-run: python scripts/00_download_data.py
        """)
        return

    # Extract, clean, reproject
    lokoja = extract_lokoja(src)
    lokoja = clean_geometries(lokoja)
    lokoja = dissolve_boundary(lokoja)
    lokoja_proj = reproject_vector(lokoja, crs)
    lokoja_geo  = reproject_vector(lokoja, "EPSG:4326")

    # Area sanity check
    area_km2 = lokoja_proj.geometry.area.sum() / 1_000_000
    print(f"  Area: {area_km2:.1f} km²  (expected ~260–400 km² for Lokoja LGA)")

    # Save
    gpkg_out    = out / "lokoja_boundary.gpkg"
    geojson_out = out / "lokoja_boundary.geojson"
    lokoja_proj.to_file(gpkg_out)
    lokoja_geo.to_file(geojson_out, driver="GeoJSON")

    print(f"\n  Saved (projected) : {gpkg_out}")
    print(f"  Saved (geographic): {geojson_out}")
    print("\n  Next step: python scripts/01_prepare_data.py")


if __name__ == "__main__":
    main()
