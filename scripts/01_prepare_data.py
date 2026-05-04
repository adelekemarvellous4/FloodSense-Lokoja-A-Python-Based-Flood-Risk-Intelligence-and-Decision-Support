"""Prepare raw FloodSense Lokoja datasets for analysis."""

from floodsense.config import load_config
from floodsense.paths import ensure_project_directories
from floodsense.preprocessing import run_preprocessing_workflow


def main() -> None:
    config = load_config()
    ensure_project_directories(config)
    run_preprocessing_workflow(config)

if __name__ == "__main__":
    main()
