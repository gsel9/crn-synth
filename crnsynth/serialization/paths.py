"""Paths for package"""
import os
from pathlib import Path

# Paths
PATH_REPO = Path(os.path.dirname(os.path.realpath(__file__))).parents[1]
PATH_CODE = PATH_REPO / "crnsynth"
PATH_RESULTS = PATH_REPO / "results"
PATH_SYNTH_CONFIGS = PATH_CODE / "configs"

# Default directories to create for results
DEFAULT_DIRS = ["synthetic_data", "generators", "configs", "reports"]


# path to all data files available for synthesis.
PATH_DATA = {
    "adult": PATH_REPO / "data/adult.csv",
}
