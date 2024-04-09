import os
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from crnsynth.generators.base_generator import BaseGenerator
from crnsynth.processing import postprocessing, preprocessing
from crnsynth.processing.generalization import (
    BaseGeneralizationMech,
    NumericGeneralizationMech,
)

# paths
PATH_REPO = Path(os.path.dirname(os.path.realpath(__file__))).parents[1]
PATH_DATA = PATH_REPO / "data"
PATH_ADULT = PATH_DATA / "adult.csv"
PATH_RESULTS = PATH_REPO / "results"

# column ranges used for generalization
AGE_BOUNDS = (17, 90)
HOURS_PER_WEEK_BOUNDS = (1, 99)

# column types
ORDINAL_COLUMNS = ["age", "hours-per-week"]
NOMINAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]
TARGET_COLUMN = "income"

# generalization mechanisms
GEN_MECHS = [
    NumericGeneralizationMech(column="age", epsilon=0.05, bins=5, bounds=AGE_BOUNDS),
    NumericGeneralizationMech(
        column="hours-per-week", epsilon=0.05, bins=5, bounds=HOURS_PER_WEEK_BOUNDS
    ),
]


def preprocess_real_data(data_real: pd.DataFrame) -> pd.DataFrame:
    """Process data before fitting the generator."""
    # reduce columns
    columns_subset = ORDINAL_COLUMNS + NOMINAL_COLUMNS
    data_real = data_real[columns_subset]
    return data_real


def postprocess_synthetic_data(data_synth: pd.DataFrame) -> pd.DataFrame:
    """Post-processing synthetic data after generation."""
    # no post-processing needed for this dataset
    return data_synth
