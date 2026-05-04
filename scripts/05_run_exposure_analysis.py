"""Run flood exposure analysis for buildings, roads, and population."""

from floodsense.config import load_config
from floodsense.exposure import run_exposure_workflow


def main() -> None:
    run_exposure_workflow(load_config())

if __name__ == "__main__":
    main()
