from pathlib import Path

import geopandas as gpd


ROOT = Path(__file__).resolve().parents[1]
LAYER_DIR = ROOT / "data" / "processed" / "dashboard_layers"
TARGET_CRS = "EPSG:32632"

# Streamlit Cloud / Folium rule of thumb:
# keep each interactive GeoJSON comfortably below a few MB. Full-resolution
# flood rasters converted to polygons are too heavy for browser rendering.
WEB_LIMITS = {
    "boundary": {"tolerance_m": 60, "max_features": None},
    "flood_zones": {"tolerance_m": 250, "max_features": 2500},
    "observed_flood": {"tolerance_m": 250, "max_features": 2500},
    "exposed_roads": {"tolerance_m": 90, "max_features": 4000},
    "exposed_buildings": {"tolerance_m": 35, "max_features": 3000},
}


def read_layer(filename: str) -> gpd.GeoDataFrame | None:
    path = LAYER_DIR / filename

    if not path.exists():
        print(f"[skip] Missing: {filename}")
        return None

    print(f"[read] {filename}")
    gdf = gpd.read_file(path)

    if gdf.empty:
        print(f"[skip] Empty: {filename}")
        return None

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.make_valid()

    if gdf.crs is None:
        print(f"[warn] {filename} has no CRS. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    return gdf


def save_geojson(gdf: gpd.GeoDataFrame, filename: str) -> None:
    output = LAYER_DIR / filename

    if output.exists():
        output.unlink()

    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(output, driver="GeoJSON")

    size_mb = output.stat().st_size / 1024 / 1024
    print(f"[saved] {filename} | {len(gdf):,} feature(s) | {size_mb:.2f} MB")


def simplify_layer(
    input_file: str,
    output_file: str,
    tolerance_m: float,
    max_features: int | None = None,
) -> None:
    gdf = read_layer(input_file)

    if gdf is None:
        return

    print(f"[process] Simplifying {input_file}")
    print(f"[info] Original features: {len(gdf):,}")

    gdf = gdf.to_crs(TARGET_CRS)

    if max_features is not None and len(gdf) > max_features:
        step = max(1, len(gdf) // max_features)
        print(f"[process] Thinning layer: keeping every {step}th feature")
        gdf = gdf.iloc[::step].copy()

    gdf["geometry"] = gdf.geometry.simplify(
        tolerance=tolerance_m,
        preserve_topology=True,
    )

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    keep_cols = [c for c in ["value", "class", "class_name", "length_km", "name", "geometry"] if c in gdf.columns]
    if "geometry" not in keep_cols:
        keep_cols.append("geometry")

    gdf = gdf[keep_cols].copy()

    save_geojson(gdf, output_file)


def simplify_building_points(
    input_file: str,
    output_file: str,
    max_features: int = 3000,
) -> None:
    """Create a very light exposed-building point layer for web maps.

    Building polygons are expensive in Folium. Centroid points are much lighter
    and still communicate the distribution of exposed structures.
    """
    gdf = read_layer(input_file)

    if gdf is None:
        return

    print(f"[process] Creating thinned building centroid layer from {input_file}")
    print(f"[info] Original features: {len(gdf):,}")

    gdf = gdf.to_crs(TARGET_CRS)

    if len(gdf) > max_features:
        step = max(1, len(gdf) // max_features)
        print(f"[process] Thinning buildings: keeping every {step}th feature")
        gdf = gdf.iloc[::step].copy()

    gdf["geometry"] = gdf.geometry.centroid
    gdf["layer"] = "exposed_buildings_sample"
    gdf = gdf[["layer", "geometry"]].copy()

    save_geojson(gdf, output_file)


def main() -> None:
    print("\nOptimizing dashboard layers...\n")

    simplify_layer(
        "lokoja_boundary.geojson",
        "lokoja_boundary_simplified.geojson",
        tolerance_m=WEB_LIMITS["boundary"]["tolerance_m"],
    )

    simplify_layer(
        "flood_susceptibility_zones.geojson",
        "flood_susceptibility_zones_simplified.geojson",
        tolerance_m=WEB_LIMITS["flood_zones"]["tolerance_m"],
        max_features=WEB_LIMITS["flood_zones"]["max_features"],
    )

    simplify_layer(
        "observed_flood_extent.geojson",
        "observed_flood_extent_simplified.geojson",
        tolerance_m=WEB_LIMITS["observed_flood"]["tolerance_m"],
        max_features=WEB_LIMITS["observed_flood"]["max_features"],
    )

    simplify_layer(
        "exposed_roads.geojson",
        "exposed_roads_simplified.geojson",
        tolerance_m=WEB_LIMITS["exposed_roads"]["tolerance_m"],
        max_features=WEB_LIMITS["exposed_roads"]["max_features"],
    )

    simplify_building_points(
        "exposed_buildings.geojson",
        "exposed_buildings_simplified.geojson",
        max_features=WEB_LIMITS["exposed_buildings"]["max_features"],
    )

    print("\nDone. Restart Streamlit after this finishes.\n")


if __name__ == "__main__":
    main()
