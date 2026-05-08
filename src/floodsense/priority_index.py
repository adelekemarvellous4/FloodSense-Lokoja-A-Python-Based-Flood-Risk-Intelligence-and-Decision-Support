"""Flood intervention priority index functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from floodsense.paths import get_project_root


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else get_project_root() / path


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series to 0-1."""
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    min_value = values.min()
    max_value = values.max()
    if np.isclose(min_value, max_value):
        return pd.Series(1.0, index=series.index)
    return (values - min_value) / (max_value - min_value)


def calculate_priority_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Calculate intervention priority score using available indicators."""
    weights = weights or {
        "exposed_population": 0.4,
        "exposed_buildings": 0.25,
        "affected_road_length_km": 0.2,
        "hazard_score": 0.15,
    }
    result = df.copy()
    score = pd.Series(0.0, index=result.index)
    used_weight = 0.0
    for column, weight in weights.items():
        if column not in result.columns:
            print(f"[priority] Missing indicator: {column}. Skipping.")
            continue
        normalized = normalize_series(result[column]).fillna(0)
        result[f"{column}_normalized"] = normalized
        score += normalized * weight
        used_weight += weight
    if used_weight == 0:
        raise ValueError("No priority indicators are available.")
    result["priority_score"] = score / used_weight
    return result


def classify_priority(score: float) -> str:
    """Classify a priority score."""
    if pd.isna(score):
        return "Low"
    if score < 0.25:
        return "Low"
    if score < 0.5:
        return "Moderate"
    if score < 0.75:
        return "High"
    return "Critical"


def run_priority_index(config: dict) -> None:
    """Build the Flood Intervention Priority Index table."""
    tables = _resolve(config["paths"]["processed_tables"])
    source = tables / "exposure_summary.csv"
    if not source.exists():
        print("[priority] Missing exposure_summary.csv. Run exposure analysis first.")
        return
    df = pd.read_csv(source)
    if df.empty:
        print("[priority] Exposure summary is empty. Skipping priority index.")
        return
    ranking = calculate_priority_score(df)
    ranking["priority_class"] = ranking["priority_score"].apply(classify_priority)
    ranking = ranking.sort_values("priority_score", ascending=False)
    out_processed = tables / "priority_ranking.csv"
    out_outputs = _resolve(config["paths"].get("outputs_tables", "outputs/tables")) / "priority_ranking.csv"
    out_outputs.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out_processed, index=False)
    ranking.to_csv(out_outputs, index=False)
    print(f"[priority] Wrote priority ranking: {out_processed}")
    print_priority_summary(ranking)
    print("\n[priority] Complete")
    print("  Next step: python scripts/07_export_dashboard_layers.py")


def print_priority_summary(ranking: pd.DataFrame) -> None:
    """Print a clean priority ranking summary table."""
    print("\n" + "─" * 60)
    print("  Flood Intervention Priority Index — Lokoja LGA")
    print("─" * 60)
    score = float(ranking["priority_score"].iloc[0]) if "priority_score" in ranking.columns else 0.0
    pclass = ranking["priority_class"].iloc[0] if "priority_class" in ranking.columns else "Unknown"

    print(f"  Priority class  : {pclass}")
    print(f"  Priority score  : {score:.4f}  (0=lowest, 1=highest)")
    print()

    indicators = [
        ("exposed_population",      "Exposed population",    "{:,.0f} people"),
        ("exposed_buildings",        "Exposed buildings",     "{:,.0f}"),
        ("affected_road_length_km",  "Affected road length",  "{:.1f} km"),
        ("hazard_score",             "Hazard score",          "{:.4f}"),
    ]

    for col, label, fmt in indicators:
        if col in ranking.columns:
            val = ranking[col].iloc[0]
            try:
                print(f"  {label:<28}: {fmt.format(float(val))}")
            except (ValueError, TypeError):
                print(f"  {label:<28}: {val}")

    print()
    print(f"  Recommendation:")
    if pclass == "Critical":
        print("  → Immediate intervention required.")
        print("  → Priority for flood barriers, early warning, and evacuation planning.")
    elif pclass == "High":
        print("  → High-priority flood risk management measures needed.")
        print("  → Consider drainage improvement and community awareness.")
    elif pclass == "Moderate":
        print("  → Moderate risk — monitor and plan for future mitigation.")
    else:
        print("  → Lower priority relative to other zones.")
        print("  → Maintain standard flood preparedness measures.")
    print("─" * 60)