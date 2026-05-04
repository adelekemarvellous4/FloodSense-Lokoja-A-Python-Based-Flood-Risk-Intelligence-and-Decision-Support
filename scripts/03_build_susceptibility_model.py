"""Build flood susceptibility score and classified risk maps."""

from floodsense.config import load_config
from floodsense.susceptibility import run_susceptibility_workflow


def main() -> None:
    run_susceptibility_workflow(load_config())

if __name__ == "__main__":
    main()
