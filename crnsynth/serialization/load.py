"""Functions for loading data from disk"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd


def load_csv(path, index_col=None, kwargs=None, verbose=1):
    """Load csv file from disk"""

    if kwargs is None:
        kwargs = {}
    data = pd.read_csv(path, index_col=index_col, **kwargs)

    # sanity check
    assert data is not None

    if verbose > 0:
        print("Loaded:", path)

    return data


def load_csv_generator(path_to_dir, filenames, yield_filename=True, index_col=0):
    """Load csv files from disk as a generator"""
    for filename in filenames:
        data = load_csv(path_to_dir / filename, index_col=index_col)

        if yield_filename:
            yield filename, data

        else:
            yield data


def load_json(path, verbose=1):
    """Load json file from disk"""
    with open(path, "r") as infile:
        loaded_file = json.load(infile)

    # sanity check
    assert loaded_file is not None

    if verbose > 0:
        print("Loaded:", path)

    return loaded_file


def load_generator(path: Path):
    """Load a generator from disk"""
    with open(path, "rb") as f:
        return pickle.load(f)
