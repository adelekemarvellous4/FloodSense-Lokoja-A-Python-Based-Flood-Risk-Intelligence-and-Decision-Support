"""Streamlit WebGIS dashboard for FloodSense Lokoja."""

from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st

try:
    from streamlit_folium import st_folium
except Exception:  # pragma: no cover - optional dashboard dependency
    st_folium = None


ROOT = Path(__file__).resolve().parents[1]
LAYER_DIR = ROOT / "data" / "processed" / "dashboard_layers"
TABLE_DIRS = [
    ROOT / "outputs" / "dashboard",
    ROOT / "data" / "processed" / "tables",
    ROOT / "outputs" / "tables",
    ROOT / "outputs" / "validation",
]


def load_geojson(name: str) -> gpd.GeoDataFrame | None:
    """Load a dashboard GeoJSON layer if it exists."""
    path = LAYER_DIR / name
    if not path.exists():
        return None
    try:
        return gpd.read_file(path)
    except Exception as exc:
        st.warning(f"Could not load {name}: {exc}")
        return None


def load_table(name: str) -> pd.DataFrame | None:
    """Load a dashboard table from the first folder where it exists."""
    for folder in TABLE_DIRS:
        path = folder / name
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as exc:
                st.warning(f"Could not load {name}: {exc}")
                return None
    return None


def metric_value(df: pd.DataFrame | None, column: str) -> float | int:
    """Return a dashboard metric from a table column."""
    if df is None or df.empty or column not in df.columns:
        return 0
    value = pd.to_numeric(df[column], errors="coerce").sum()
    return 0 if pd.isna(value) else value


def add_layer(map_obj: folium.Map, gdf: gpd.GeoDataFrame | None, name: str) -> None:
    """Add a GeoDataFrame to a Folium map."""
    if gdf is None or gdf.empty:
        return
    folium.GeoJson(gdf.to_crs("EPSG:4326"), name=name, tooltip=name).add_to(map_obj)


def build_map(layers: dict[str, gpd.GeoDataFrame | None]) -> folium.Map:
    """Build an interactive Folium map."""
    boundary = layers.get("Boundary")
    center = [7.80, 6.74]
    if boundary is not None and not boundary.empty:
        centroid = boundary.to_crs("EPSG:4326").geometry.union_all().centroid
        center = [centroid.y, centroid.x]
    fmap = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")
    for name, gdf in layers.items():
        add_layer(fmap, gdf, name)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def table_section(title: str, df: pd.DataFrame | None, filename: str) -> None:
    """Render a table and CSV download button."""
    st.subheader(title)
    if df is None:
        st.info(f"{filename} is not available yet.")
        return
    st.dataframe(df, use_container_width=True)
    st.download_button(
        f"Download {filename}",
        df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


st.set_page_config(page_title="FloodSense Lokoja", layout="wide")
st.title("FloodSense Lokoja")
st.caption("Flood Risk Intelligence and Decision Support System.")

exposure = load_table("exposure_summary.csv")
validation = load_table("validation_metrics.csv")
priority = load_table("priority_ranking.csv")

metric_cols = st.columns(4)
metric_cols[0].metric("Exposed Population", f"{metric_value(exposure, 'exposed_population'):,.0f}")
metric_cols[1].metric("Exposed Buildings", f"{metric_value(exposure, 'exposed_buildings'):,.0f}")
metric_cols[2].metric("Affected Road Length", f"{metric_value(exposure, 'affected_road_length_km'):,.2f} km")
critical_count = 0
if priority is not None and "priority_class" in priority.columns:
    critical_count = int((priority["priority_class"] == "Critical").sum())
metric_cols[3].metric("Critical Priority Areas", critical_count)

layers = {
    "Boundary": load_geojson("lokoja_boundary.geojson"),
    "Flood Susceptibility Zones": load_geojson("flood_susceptibility_zones.geojson"),
    "Observed Flood Extent": load_geojson("observed_flood_extent.geojson"),
    "Exposed Buildings": load_geojson("exposed_buildings.geojson"),
    "Exposed Roads": load_geojson("exposed_roads.geojson"),
    "Priority Zones": load_geojson("priority_zones.geojson"),
}

st.subheader("Interactive WebGIS Map")
available_layers = [name for name, gdf in layers.items() if gdf is not None and not gdf.empty]
if not available_layers:
    st.info("No dashboard GeoJSON layers found yet. Run the analysis and export stages first.")
else:
    fmap = build_map(layers)
    if st_folium is None:
        st.warning("Install streamlit-folium for embedded map rendering: pip install streamlit-folium")
        st.components.v1.html(fmap._repr_html_(), height=650)
    else:
        st_folium(fmap, height=650, use_container_width=True)

table_section("Exposure Summary", exposure, "exposure_summary.csv")
table_section("Validation Metrics", validation, "validation_metrics.csv")
table_section("Priority Ranking", priority, "priority_ranking.csv")
