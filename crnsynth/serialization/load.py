"""Functions for loading data from disk"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd


def load_generator(path: Path):
    """Load a generator from disk"""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    """Load data from disk"""
    return pd.read_csv(path)
