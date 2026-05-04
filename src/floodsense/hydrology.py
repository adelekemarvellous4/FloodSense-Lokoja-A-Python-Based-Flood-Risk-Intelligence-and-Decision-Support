"""DEM terrain and hydrological analysis functions."""

from __future__ import annotations

from pathlib import Path

from floodsense.mapping import plot_raster_preview
from floodsense.paths import first_existing_file, get_project_root


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else get_project_root() / path


def check_whitebox_available() -> bool:
    """Return True if WhiteboxTools can be imported and initialized."""
    try:
        from whitebox.whitebox_tools import WhiteboxTools

        WhiteboxTools()
        return True
    except Exception as exc:
        print("[hydrology] WhiteboxTools is unavailable.")
        print("[hydrology] Install it with: pip install whitebox")
        print(f"[hydrology] Details: {exc}")
        return False


def _wbt():
    from whitebox.whitebox_tools import WhiteboxTools

    tool = WhiteboxTools()
    tool.verbose = False
    return tool


def fill_depressions(dem_path: str | Path, output_path: str | Path) -> Path:
    """Fill DEM depressions."""
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().fill_depressions(str(_resolve(dem_path)), str(output_path))
    return output_path


def calculate_slope(dem_path: str | Path, output_path: str | Path) -> Path:
    """Calculate slope from DEM."""
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().slope(str(_resolve(dem_path)), str(output_path))
    return output_path


def calculate_flow_direction(dem_path: str | Path, output_path: str | Path) -> Path:
    """Calculate D8 flow direction."""
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().d8_pointer(str(_resolve(dem_path)), str(output_path))
    return output_path


def calculate_flow_accumulation(flow_direction_path: str | Path, output_path: str | Path) -> Path:
    """Calculate D8 flow accumulation from a flow direction raster."""
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().d8_flow_accumulation(
        str(_resolve(flow_direction_path)), str(output_path), out_type="cells", pntr=True
    )
    return output_path


def extract_streams(flow_accumulation_path: str | Path, output_path: str | Path, threshold: int) -> Path:
    """Extract streams from flow accumulation."""
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().extract_streams(str(_resolve(flow_accumulation_path)), str(output_path), threshold)
    return output_path


def distance_to_streams(streams_path: str | Path, reference_raster_path: str | Path, output_path: str | Path) -> Path:
    """Create Euclidean distance-to-stream raster."""
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().euclidean_distance(str(_resolve(streams_path)), str(output_path))
    return output_path


def run_hydrology_workflow(config: dict) -> None:
    """Run DEM hydrological analysis."""
    if not check_whitebox_available():
        return
    paths = config["paths"]
    dem = first_existing_file(paths["interim_reprojected"], [".tif", ".tiff"])
    candidate = _resolve(paths["interim_reprojected"]) / "dem.tif"
    if candidate.exists():
        dem = candidate
    if dem is None:
        print("[hydrology] Missing DEM. Run preprocessing after placing DEM in data/raw/dem/.")
        return

    out_dir = _resolve(paths["processed_rasters"])
    map_dir = _resolve(paths["outputs_maps"])
    filled = out_dir / "dem_filled.tif"
    slope = out_dir / "slope.tif"
    flow_dir = out_dir / "flow_direction.tif"
    flow_acc = out_dir / "flow_accumulation.tif"
    streams = out_dir / "streams.tif"
    distance = out_dir / "distance_to_stream.tif"

    print("[hydrology] Filling depressions.")
    fill_depressions(dem, filled)
    print("[hydrology] Calculating slope.")
    calculate_slope(filled, slope)
    print("[hydrology] Calculating flow direction and accumulation.")
    calculate_flow_direction(filled, flow_dir)
    calculate_flow_accumulation(flow_dir, flow_acc)
    print("[hydrology] Extracting stream raster.")
    extract_streams(flow_acc, streams, int(config.get("hydrology", {}).get("stream_threshold", 1000)))
    print("[hydrology] Calculating distance to streams.")
    distance_to_streams(streams, dem, distance)
    for raster in [filled, slope, flow_acc, streams, distance]:
        plot_raster_preview(raster, map_dir / f"{raster.stem}_preview.png", raster.stem.replace("_", " ").title())
