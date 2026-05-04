"""Orchestrate the full FloodSense Lokoja analysis pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from floodsense.config import load_config
from floodsense.export import export_dashboard_layers
from floodsense.exposure import run_exposure_workflow
from floodsense.hydrology import run_hydrology_workflow
from floodsense.paths import ensure_project_directories
from floodsense.preprocessing import run_preprocessing_workflow
from floodsense.priority_index import run_priority_index
from floodsense.rainfall import run_rainfall_scenarios
from floodsense.sar_validation import run_sar_validation
from floodsense.susceptibility import run_susceptibility_workflow


STAGE_ORDER = [
    "prepare_data",
    "hydrology",
    "susceptibility",
    "rainfall",
    "sar_validation",
    "exposure",
    "priority",
    "export_dashboard",
]


def _stage_functions(config: dict) -> dict[str, Callable[[], None]]:
    return {
        "prepare_data": lambda: run_preprocessing_workflow(config),
        "hydrology": lambda: run_hydrology_workflow(config),
        "susceptibility": lambda: run_susceptibility_workflow(config),
        "rainfall": lambda: run_rainfall_scenarios(config),
        "sar_validation": lambda: run_sar_validation(config),
        "exposure": lambda: run_exposure_workflow(config),
        "priority": lambda: run_priority_index(config),
        "export_dashboard": lambda: export_dashboard_layers(config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FloodSense Lokoja pipeline stages.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=[*STAGE_ORDER, "all"],
        help="Pipeline stage to run.",
    )
    args = parser.parse_args()
    config = load_config()
    ensure_project_directories(config)
    stages = STAGE_ORDER if args.stage == "all" else [args.stage]
    functions = _stage_functions(config)
    for stage in stages:
        print(f"\n=== Running stage: {stage} ===")
        try:
            functions[stage]()
        except FileNotFoundError as exc:
            print(f"[pipeline] Required input missing for {stage}: {exc}")
            if stage == "prepare_data":
                break
        except Exception as exc:
            print(f"[pipeline] Stage failed: {stage}")
            raise exc


if __name__ == "__main__":
    main()
