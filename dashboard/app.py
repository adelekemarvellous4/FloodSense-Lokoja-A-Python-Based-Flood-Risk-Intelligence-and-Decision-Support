from pathlib import Path
import json

import folium
import pandas as pd
import streamlit as st

try:
    from streamlit_folium import st_folium

    HAS_ST_FOLIUM = True
except Exception:
    HAS_ST_FOLIUM = False


ROOT = Path(__file__).resolve().parents[1]
LAYER_DIR = ROOT / "data" / "processed" / "dashboard_layers"
MAPS_DIR = ROOT / "outputs" / "maps"
GITHUB_URL = (
    "https://github.com/adelekemarvellous4/"
    "floodsense-lokoja-a-python-based-flood-risk-intelligence-and-decision-support"
)

TABLE_DIRS = [
    ROOT / "outputs" / "dashboard",
    ROOT / "data" / "processed" / "tables",
    ROOT / "outputs" / "tables",
    ROOT / "outputs" / "validation",
]


INTERACTIVE_LAYERS = {
    "lokoja_boundary_simplified.geojson": {
        "label": "Study area boundary",
        "color": "#2c3e50",
        "weight": 3,
        "fill": False,
        "default": True,
    },
    "flood_susceptibility_zones_simplified.geojson": {
        "label": "Simplified flood zones",
        "color": "#e74c3c",
        "weight": 1,
        "fill": True,
        "fillOpacity": 0.35,
        "default": False,
    },
    "observed_flood_extent_simplified.geojson": {
        "label": "Simplified observed flood extent",
        "color": "#2980b9",
        "weight": 1,
        "fill": True,
        "fillOpacity": 0.45,
        "default": False,
    },
    "exposed_buildings_simplified.geojson": {
        "label": "Exposed buildings sample",
        "color": "#e67e22",
        "weight": 1,
        "fill": True,
        "fillOpacity": 0.75,
        "default": False,
    },
    "exposed_roads_simplified.geojson": {
        "label": "Simplified exposed roads",
        "color": "#8e44ad",
        "weight": 2,
        "fill": False,
        "default": False,
    },
}


STATIC_MAPS = {
    "Flood susceptibility map": [
        "susceptibility_class_preview.png",
        "susceptibility_preview.png",
        "susceptibility_class_report_preview.png",
    ],
    "SAR validation map": [
        "model_vs_sar_comparison.png",
        "model_vs_observed_comparison_preview.png",
        "model_vs_observed_comparison_report_preview.png",
    ],
    "Observed flood extent": [
        "observed_flood_preview.png",
        "observed_flood_extent_preview.png",
    ],
    "Exposure map": [
        "exposure_flood_zones_preview.png",
        "exposure_preview.png",
        "high_risk_flood_zones_report_preview.png",
    ],
}


@st.cache_data(show_spinner=False)
def load_geojson(filename: str) -> dict | None:
    path = LAYER_DIR / filename

    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not data.get("features"):
            return None

        return data
    except Exception as exc:
        st.warning(f"Could not load {filename}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_table(filename: str) -> pd.DataFrame | None:
    for folder in TABLE_DIRS:
        path = folder / filename
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as exc:
                st.warning(f"Could not read {filename}: {exc}")
                return None
    return None


def table_locations(filename: str) -> list[Path]:
    return [folder / filename for folder in TABLE_DIRS]


def show_missing_table_help(filename: str) -> None:
    st.info(f"{filename} is not available yet.")
    with st.expander("Paths checked"):
        for path in table_locations(filename):
            st.code(str(path))


def safe_metric(df: pd.DataFrame | None, column: str, fmt: str = "{:,.0f}") -> str:
    if df is None or df.empty or column not in df.columns:
        return "-"

    value = pd.to_numeric(df[column], errors="coerce").sum()

    if pd.isna(value):
        return "-"

    return fmt.format(value)


def add_geojson_layer(fmap: folium.Map, filename: str, style: dict) -> None:
    geojson_data = load_geojson(filename)

    if geojson_data is None:
        return

    folium.GeoJson(
        geojson_data,
        name=style["label"],
        style_function=lambda feature, s=style: {
            "color": s.get("color", "#3498db"),
            "weight": s.get("weight", 1),
            "fillColor": s.get("color", "#3498db"),
            "fillOpacity": s.get("fillOpacity", 0.0) if s.get("fill") else 0,
            "opacity": 0.85,
        },
        tooltip=style["label"],
    ).add_to(fmap)


def build_map(selected_layers: list[str]) -> folium.Map:
    fmap = folium.Map(
        location=[7.80, 6.74],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Satellite",
    ).add_to(fmap)

    for filename in selected_layers:
        add_geojson_layer(fmap, filename, INTERACTIVE_LAYERS[filename])

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def find_static_map(candidates: list[str]) -> Path | None:
    for filename in candidates:
        path = MAPS_DIR / filename
        if path.exists():
            return path
    return None


st.set_page_config(
    page_title="FloodSense Lokoja",
    page_icon="Flood",
    layout="wide",
    initial_sidebar_state="expanded",
)


with st.sidebar:
    st.title("FloodSense")
    st.caption("Flood Risk Intelligence and Decision Support System")

    st.markdown("**Study area:** Lokoja LGA, Kogi State")
    st.markdown("**Validation event:** October 2022 flood")

    st.divider()
    st.subheader("About")
    st.markdown(
        "Python-based flood risk intelligence and decision support system for "
        "Lokoja, Kogi State, Nigeria."
    )
    st.markdown(
        "Combines DEM hydrology, flood susceptibility mapping, Sentinel-1 SAR "
        "validation, exposure analysis, and intervention priority ranking."
    )
    st.markdown(f"[GitHub Repository]({GITHUB_URL})")

    st.divider()
    st.subheader("Interactive map layers")

    selected_layers = []

    for filename, style in INTERACTIVE_LAYERS.items():
        path = LAYER_DIR / filename
        disabled = not path.exists()

        checked = st.checkbox(
            style["label"],
            value=style.get("default", False) and not disabled,
            disabled=disabled,
            key=f"layer_{filename}",
        )

        if checked:
            selected_layers.append(filename)

        if disabled:
            st.caption(f"Missing: {filename}")

    st.divider()
    st.caption("Use simplified GeoJSON layers for fast map rendering.")


exposure = load_table("exposure_summary.csv")
validation = load_table("validation_metrics.csv")
priority = load_table("priority_ranking.csv")


st.title("FloodSense Lokoja")
st.markdown(
    "Flood Risk Intelligence and Decision Support System - "
    "Lokoja LGA, Kogi State, Nigeria"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Exposed population", safe_metric(exposure, "exposed_population"))
c2.metric("Exposed buildings", safe_metric(exposure, "exposed_buildings"))
c3.metric("Affected roads", safe_metric(exposure, "affected_road_length_km", "{:,.1f} km"))

if priority is not None and not priority.empty and "priority_class" in priority.columns:
    priority_class = str(priority["priority_class"].iloc[0])
else:
    priority_class = "-"

c4.metric("Priority class", priority_class)

st.divider()


tab_map, tab_static, tab_exposure, tab_validation, tab_priority = st.tabs(
    [
        "Interactive Folium map",
        "Static analysis maps",
        "Exposure table",
        "SAR validation",
        "Priority index",
    ]
)


with tab_map:
    st.subheader("Interactive flood risk map")
    st.caption(
        "This map uses simplified dashboard layers only. "
        "Large full-resolution GeoJSON layers should not be loaded directly in Folium."
    )

    if not selected_layers:
        st.info("Select at least one available layer from the sidebar.")
    else:
        with st.spinner(f"Loading {len(selected_layers)} simplified layer(s)..."):
            fmap = build_map(selected_layers)

        if HAS_ST_FOLIUM:
            st_folium(fmap, height=620, width="stretch")
        else:
            st.warning("Install streamlit-folium for better map rendering.")
            st.components.v1.html(fmap._repr_html_(), height=620)


with tab_static:
    st.subheader("Static analysis maps")
    st.caption(
        "These maps show full analysis outputs as images. "
        "This is better for heavy raster-derived flood layers."
    )

    shown = 0

    for title, candidates in STATIC_MAPS.items():
        path = find_static_map(candidates)
        if path is not None:
            st.markdown(f"### {title}")
            st.image(str(path), width="stretch")
            shown += 1

    if shown == 0:
        st.info(
            "No static maps found in outputs/maps/. "
            "Run the analysis scripts that generate preview maps."
        )


with tab_exposure:
    st.subheader("Exposure summary")

    if exposure is None:
        show_missing_table_help("exposure_summary.csv")
    else:
        st.dataframe(exposure, width="stretch")
        st.download_button(
            "Download exposure summary",
            exposure.to_csv(index=False).encode("utf-8"),
            "exposure_summary.csv",
            "text/csv",
        )


with tab_validation:
    st.subheader("Sentinel-1 SAR validation")

    if validation is None:
        show_missing_table_help("validation_metrics.csv")
    else:
        st.dataframe(validation, width="stretch")

        if not validation.empty:
            row = validation.iloc[0]
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Precision", f"{float(row.get('precision', 0)):.4f}")
            v2.metric("Recall", f"{float(row.get('recall', 0)):.4f}")
            v3.metric("F1 score", f"{float(row.get('f1_score', 0)):.4f}")
            v4.metric("Accuracy", f"{float(row.get('overall_accuracy', 0)):.4f}")

        st.download_button(
            "Download validation metrics",
            validation.to_csv(index=False).encode("utf-8"),
            "validation_metrics.csv",
            "text/csv",
        )


with tab_priority:
    st.subheader("Flood Intervention Priority Index")

    if priority is None:
        show_missing_table_help("priority_ranking.csv")
    else:
        st.dataframe(priority, width="stretch")
        st.download_button(
            "Download priority ranking",
            priority.to_csv(index=False).encode("utf-8"),
            "priority_ranking.csv",
            "text/csv",
        )


st.divider()
st.caption("FloodSense Lokoja - Python-based Flood Risk Intelligence and Decision Support System")
