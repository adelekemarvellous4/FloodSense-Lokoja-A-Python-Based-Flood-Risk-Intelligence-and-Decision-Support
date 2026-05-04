"""Run DEM terrain and hydrological analysis."""

from floodsense.config import load_config
from floodsense.hydrology import run_hydrology_workflow


def main() -> None:
    run_hydrology_workflow(load_config())

if __name__ == "__main__":
    main()
