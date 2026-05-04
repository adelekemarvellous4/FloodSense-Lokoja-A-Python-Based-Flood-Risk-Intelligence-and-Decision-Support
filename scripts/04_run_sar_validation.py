"""Validate modelled flood zones using Sentinel-1 SAR flood evidence."""

from floodsense.config import load_config
from floodsense.sar_validation import run_sar_validation


def main() -> None:
    run_sar_validation(load_config())

if __name__ == "__main__":
    main()
