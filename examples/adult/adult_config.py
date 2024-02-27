import os
from pathlib import Path

# paths
PATH_REPO = Path(os.path.dirname(os.path.realpath(__file__))).parents[1]
PATH_DATA = PATH_REPO / "data"
PATH_ADULT = PATH_DATA / "adult.csv"


# column ranges used for generalization
AGE_BOUNDS = (17, 90)
HOURS_PER_WEEK_BOUNDS = (1, 99)
