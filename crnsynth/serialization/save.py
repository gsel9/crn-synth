"""Functions for saving data to disk"""
import pickle
from pathlib import Path

import pandas as pd


def save_generator(generator, path: Path) -> None:
    """Save a generator to disk"""
    with open(path, "wb") as f:
        pickle.dump(generator, f)


def save_csv(data: pd.DataFrame, path: Path) -> None:
    """Save data to disk"""
    data.to_csv(path, index=False)
