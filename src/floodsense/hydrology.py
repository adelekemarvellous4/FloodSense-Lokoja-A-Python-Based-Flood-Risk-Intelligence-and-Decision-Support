"""DEM terrain and hydrological analysis for FloodSense Lokoja."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from floodsense.mapping import plot_raster_preview
from floodsense.paths import ensure_dir, first_existing_file, get_project_root
from floodsense.raster_utils import save_raster


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
        print("[hydrology] WhiteboxTools unavailable — install with: pip install whitebox")
        print(f"[hydrology] Details: {exc}")
        return False


def _wbt():
    from whitebox.whitebox_tools import WhiteboxTools
    tool = WhiteboxTools()
    tool.verbose = False
    return tool


# ── Core hydrological functions ───────────────────────────────────────────────

def fill_depressions(dem_path: str | Path, output_path: str | Path) -> Path:
    """
    Fill artificial pits (sinks) in the DEM using Wang & Liu (2006).

    WHY: Real DEMs contain small depressions where water pools incorrectly.
    Filling them ensures water flows continuously downhill — a prerequisite
    for correct flow direction and accumulation.
    """
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().fill_depressions_wang_and_liu(
        str(_resolve(dem_path)), str(output_path), fix_flats=True
    )
    print(f"  [hydrology] Depressions filled → {output_path.name}")
    return output_path


def calculate_slope(dem_path: str | Path, output_path: str | Path) -> Path:
    """
    Calculate slope in degrees from the filled DEM.

    Flood relevance: flat areas (0–3°) accumulate water and drain slowly.
    Steep areas shed water rapidly and are generally less flood-prone
    (except flash flood corridors on steep terrain above floodplains).
    """
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().slope(str(_resolve(dem_path)), str(output_path), units="degrees")
    print(f"  [hydrology] Slope → {output_path.name}")
    return output_path


def calculate_flow_direction(dem_path: str | Path, output_path: str | Path) -> Path:
    """
    Calculate D8 flow direction from the filled DEM.

    Each cell is assigned one of 8 directions pointing toward the steepest
    downslope neighbour. This is the foundation for flow accumulation.
    """
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().d8_pointer(str(_resolve(dem_path)), str(output_path))
    print(f"  [hydrology] Flow direction (D8) → {output_path.name}")
    return output_path


def calculate_flow_accumulation(
    flow_direction_path: str | Path, output_path: str | Path
) -> Path:
    """
    Calculate D8 flow accumulation — how many upstream cells drain into each cell.

    High accumulation = large catchment area draining into that point.
    The Niger-Benue confluence will have extremely high accumulation.
    Used for stream extraction and as a flood susceptibility factor.
    """
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().d8_flow_accumulation(
        str(_resolve(flow_direction_path)),
        str(output_path),
        out_type="cells",
        pntr=True,
        log=False,
        clip=False,
    )
    print(f"  [hydrology] Flow accumulation → {output_path.name}")
    return output_path


def calculate_flow_accumulation_log(
    flow_accumulation_path: str | Path, output_path: str | Path
) -> Path:
    """
    Save a log10-transformed flow accumulation raster for visualization.

    Raw flow accumulation spans many orders of magnitude — the Niger
    confluence will be millions while hillslopes are 1–10. Log10 compresses
    this range so spatial patterns are visible in a preview map.
    Used only for visualization, not for susceptibility modelling.
    """
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(_resolve(flow_accumulation_path)) as src:
        data = src.read(1, masked=True).astype("float32")
        profile = src.profile.copy()
    log_data = np.log10(data + 1)
    profile.update(dtype="float32", nodata=-9999)
    save_raster(log_data.filled(-9999), profile, output_path, nodata=-9999)
    print(f"  [hydrology] Flow accumulation (log10) → {output_path.name}")
    return output_path


def extract_streams(
    flow_accumulation_path: str | Path, output_path: str | Path, threshold: int
) -> Path:
    """
    Extract stream network from flow accumulation using a threshold.

    A cell is classified as a stream if its upstream contributing area
    (in cells) exceeds the threshold. At 30m resolution:
      threshold=1000 → streams draining > ~900,000 m² (0.9 km²)
      threshold=500  → denser network including smaller channels
      threshold=5000 → only major channels

    Adjust in config.yaml → hydrology.stream_threshold.
    """
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().extract_streams(
        str(_resolve(flow_accumulation_path)),
        str(output_path),
        threshold=threshold,
        zero_background=True,
    )
    print(f"  [hydrology] Streams extracted (threshold={threshold}) → {output_path.name}")
    return output_path


def distance_to_streams(
    streams_path: str | Path,
    reference_raster_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Calculate Euclidean distance (metres) from each cell to the nearest stream.

    This is one of the strongest single predictors of flood risk:
    - cells within 100m of a stream are in the active floodplain
    - cells 100–500m away are at moderate risk
    - cells > 1km from streams are generally low-risk terrain
    The Niger-Benue confluence shoreline will show the lowest values.
    """
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().euclidean_distance(str(_resolve(streams_path)), str(output_path))
    print(f"  [hydrology] Distance to streams → {output_path.name}")
    return output_path


def calculate_hand(
    dem_path: str | Path, streams_path: str | Path, output_path: str | Path
) -> Path:
    """
    Calculate Height Above Nearest Drainage (HAND).

    HAND measures how many metres each cell sits above the nearest stream
    in its own drainage network — not just straight-line distance.

    A cell 2m above the nearest stream will flood in minor events.
    A cell 20m above the stream only floods in extreme events.
    HAND is one of the most powerful flood inundation predictors available
    from DEM analysis alone. Especially valuable at Lokoja where the
    Niger and Benue run through low-gradient floodplains.
    """
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().elevation_above_stream(
        str(_resolve(dem_path)), str(_resolve(streams_path)), str(output_path)
    )
    print(f"  [hydrology] HAND → {output_path.name}")
    return output_path


def calculate_twi(
    flow_accumulation_path: str | Path, slope_path: str | Path, output_path: str | Path
) -> Path:
    """
    Calculate Topographic Wetness Index (TWI).

    TWI = ln(specific contributing area / tan(slope))
    High TWI = large catchment + flat terrain = likely wet / flood-prone
    Low TWI  = small catchment + steep terrain = likely dry

    Useful as a secondary susceptibility factor alongside HAND.
    """
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _wbt().wetness_index(
        str(_resolve(flow_accumulation_path)),
        str(_resolve(slope_path)),
        str(output_path),
    )
    print(f"  [hydrology] TWI → {output_path.name}")
    return output_path


# ── Output summary ────────────────────────────────────────────────────────────

def _raster_stats(path: Path) -> str:
    """Return a compact min/max/mean string for a raster."""
    if not path.exists():
        return "NOT CREATED"
    try:
        with rasterio.open(path) as src:
            data = src.read(1, masked=True)
            if data.count() == 0:
                return "all nodata"
            return (
                f"min={data.min():.2f}  max={data.max():.2f}  "
                f"mean={data.mean():.2f}  ({src.width}×{src.height} px)"
            )
    except Exception as exc:
        return f"error reading: {exc}"


def print_hydrology_summary(out_dir: Path) -> None:
    outputs = {
        "dem_filled.tif":           "Filled DEM",
        "slope.tif":                "Slope (degrees)",
        "flow_direction.tif":       "Flow direction (D8)",
        "flow_accumulation.tif":    "Flow accumulation (cells)",
        "flow_accumulation_log.tif":"Flow accumulation (log10, viz only)",
        "streams.tif":              "Stream network",
        "distance_to_stream.tif":   "Distance to stream (m)",
        "hand.tif":                 "HAND (m above nearest drainage)",
        "twi.tif":                  "TWI",
    }
    print("\n" + "─" * 60)
    print("  Hydrology outputs")
    print("─" * 60)
    for fname, label in outputs.items():
        path = out_dir / fname
        status = "✓" if path.exists() else "✗"
        stats = _raster_stats(path) if path.exists() else "missing"
        print(f"  {status}  {label:<38} {stats}")

    sanity = out_dir / "slope.tif"
    if sanity.exists():
        with rasterio.open(sanity) as src:
            data = src.read(1, masked=True)
            flat_pct = float((data < 3).sum() / data.count() * 100)
        print(f"\n  Slope sanity: {flat_pct:.1f}% of valid pixels have slope < 3°")
        print("  (Expected ~30–50% for Lokoja's mixed terrain — low near river, higher on ridges)")

    hand = out_dir / "hand.tif"
    if hand.exists():
        with rasterio.open(hand) as src:
            data = src.read(1, masked=True)
            low_pct = float((data < 5).sum() / data.count() * 100)
        print(f"  HAND sanity : {low_pct:.1f}% of pixels within 5m of nearest drainage")
        print("  (These are the primary floodplain cells — should match river corridors)")


# ── Main workflow ─────────────────────────────────────────────────────────────

def run_hydrology_workflow(config: dict) -> None:
    """
    Run the full DEM and hydrological analysis workflow.

    Stages:
      1. Fill DEM depressions
      2. Calculate slope
      3. Calculate flow direction (D8)
      4. Calculate flow accumulation
      5. Extract stream network
      6. Calculate distance to streams
      7. Calculate HAND
      8. Calculate TWI
      9. Save log-transformed flow accumulation (visualization)
     10. Save preview maps for all outputs
    """
    if not check_whitebox_available():
        print("[hydrology] Install WhiteboxTools: pip install whitebox")
        return

    paths = config["paths"]
    threshold = int(config.get("hydrology", {}).get("stream_threshold", 1000))

    # Locate reprojected DEM
    dem = _resolve(paths["interim_reprojected"]) / "dem.tif"
    if not dem.exists():
        # Fallback: any tif in reprojected folder
        dem = first_existing_file(paths["interim_reprojected"], [".tif", ".tiff"])
    if dem is None or not dem.exists():
        print("[hydrology] DEM not found in data/interim/reprojected/")
        print("[hydrology] Run script 01_prepare_data.py first.")
        return

    out_dir  = ensure_dir(_resolve(paths["processed_rasters"]))
    map_dir  = ensure_dir(_resolve(paths["outputs_maps"]))

    # Output paths
    filled   = out_dir / "dem_filled.tif"
    slope    = out_dir / "slope.tif"
    flow_dir = out_dir / "flow_direction.tif"
    flow_acc = out_dir / "flow_accumulation.tif"
    flow_log = out_dir / "flow_accumulation_log.tif"
    streams  = out_dir / "streams.tif"
    dist     = out_dir / "distance_to_stream.tif"
    hand     = out_dir / "hand.tif"
    twi      = out_dir / "twi.tif"

    print("\n[hydrology] Starting hydrological analysis")
    print(f"  DEM input  : {dem}")
    print(f"  Output dir : {out_dir}")
    print(f"  Stream threshold: {threshold} cells\n")

    print("[hydrology] Stage 1/8 — Fill depressions")
    fill_depressions(dem, filled)

    print("[hydrology] Stage 2/8 — Slope")
    calculate_slope(filled, slope)

    print("[hydrology] Stage 3/8 — Flow direction (D8)")
    calculate_flow_direction(filled, flow_dir)

    print("[hydrology] Stage 4/8 — Flow accumulation")
    calculate_flow_accumulation(flow_dir, flow_acc)
    calculate_flow_accumulation_log(flow_acc, flow_log)

    print("[hydrology] Stage 5/8 — Extract streams")
    extract_streams(flow_acc, streams, threshold)

    print("[hydrology] Stage 6/8 — Distance to streams")
    distance_to_streams(streams, dem, dist)

    print("[hydrology] Stage 7/8 — HAND (Height Above Nearest Drainage)")
    calculate_hand(filled, streams, hand)

    print("[hydrology] Stage 8/8 — TWI (Topographic Wetness Index)")
    calculate_twi(flow_acc, slope, twi)

    # Save preview maps
    print("\n[hydrology] Saving preview maps...")
    preview_map = {
        slope:    ("Lokoja — Slope (degrees)",              "YlOrRd"),
        flow_log: ("Lokoja — Flow accumulation (log₁₀)",    "Blues"),
        dist:     ("Lokoja — Distance to streams (m)",      "RdYlGn"),
        hand:     ("Lokoja — HAND (m above nearest drainage)", "RdYlGn_r"),
        twi:      ("Lokoja — TWI",                           "Blues"),
    }
    for raster_path, (title, _cmap) in preview_map.items():
        if raster_path.exists():
            plot_raster_preview(
                raster_path,
                map_dir / f"{raster_path.stem}_preview.png",
                title,
            )

    print_hydrology_summary(out_dir)

    print("\n[hydrology] ✓ Complete")
    print("  Next step: python scripts/03_build_susceptibility_model.py")
