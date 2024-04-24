"""Paths for package"""

import os
from pathlib import Path

# Paths
PATH_REPO = Path(os.path.dirname(os.path.realpath(__file__))).parents[1]
PATH_CODE = PATH_REPO / "crnsynth"
PATH_RESULTS = PATH_REPO / "results"

# Default directories to create for results
DEFAULT_DIRS = ["synthetic_data", "generators", "reports", "configs"]


# path to all data files available for synthesis.
PATH_DATA = {
    "adult": PATH_REPO / "data/adult.csv",
}


def create_output_dir(path_out: Path, dirs: list[str] = None) -> None:
    """Create output directories for results."""
    if dirs is None:
        dirs = DEFAULT_DIRS

    for dir_name in dirs:
        path_dir = path_out / dir_name

        # only make directory if it does not exist
        if not path_dir.exists():
            path_dir.mkdir(parents=True, exist_ok=True)
