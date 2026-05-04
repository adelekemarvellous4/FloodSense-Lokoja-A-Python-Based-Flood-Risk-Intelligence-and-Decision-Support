"""Export dashboard-ready GeoJSON and CSV outputs."""

from floodsense.config import load_config
from floodsense.export import export_dashboard_layers


def main() -> None:
    export_dashboard_layers(load_config())

if __name__ == "__main__":
    main()
