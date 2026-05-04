"""Static map and figure generation utilities."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from rasterio.plot import show


def _missing(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        print(f"[mapping] Missing file, skipping preview: {path}")
        return True
    return False


def plot_raster_preview(raster_path: str | Path, output_png: str | Path, title: str | None = None) -> None:
    """Save a simple raster preview image."""
    if _missing(raster_path):
        return
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as src:
        fig, ax = plt.subplots(figsize=(8, 7))
        show(src, ax=ax)
        ax.set_title(title or Path(raster_path).stem)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_png, dpi=180)
        plt.close(fig)


def plot_vector_preview(vector_path: str | Path, output_png: str | Path, title: str | None = None) -> None:
    """Save a simple vector preview image."""
    if _missing(vector_path):
        return
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(vector_path)
    fig, ax = plt.subplots(figsize=(8, 7))
    if not gdf.empty:
        gdf.plot(ax=ax)
    ax.set_title(title or Path(vector_path).stem)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def plot_overlay_preview(
    base_vector: str | Path, overlay_vector: str | Path, output_png: str | Path, title: str | None = None
) -> None:
    """Save a simple two-layer vector overlay preview."""
    if _missing(base_vector) or _missing(overlay_vector):
        return
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    base = gpd.read_file(base_vector)
    overlay = gpd.read_file(overlay_vector)
    if overlay.crs != base.crs and base.crs is not None:
        overlay = overlay.to_crs(base.crs)
    fig, ax = plt.subplots(figsize=(8, 7))
    if not base.empty:
        base.boundary.plot(ax=ax, linewidth=1)
    if not overlay.empty:
        overlay.plot(ax=ax, alpha=0.5)
    ax.set_title(title or "Overlay preview")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def save_bar_chart(
    df: pd.DataFrame, x_col: str, y_col: str, output_png: str | Path, title: str | None = None
) -> None:
    """Save a simple bar chart."""
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print("[mapping] Missing chart columns or empty table, skipping chart.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title or f"{y_col} by {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
