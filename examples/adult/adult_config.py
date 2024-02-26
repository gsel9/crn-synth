import os
from pathlib import Path

PATH_REPO = Path(os.path.dirname(os.path.realpath(__file__))).parents[1]
PATH_DATA = PATH_REPO / "data"
PATH_ADULT = PATH_DATA / "adult.csv"
