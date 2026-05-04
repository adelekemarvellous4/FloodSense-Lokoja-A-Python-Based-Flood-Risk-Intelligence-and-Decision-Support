"""Build Flood Intervention Priority Index layers and tables."""

from floodsense.config import load_config
from floodsense.priority_index import run_priority_index


def main() -> None:
    run_priority_index(load_config())

if __name__ == "__main__":
    main()
