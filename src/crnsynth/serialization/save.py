"""Functions for saving data to disk"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from synthcity.plugins import Plugin
from synthcity.utils.serialization import save_to_file


def save_generator(generator, path: Path) -> None:
    """Save a generator to disk"""
    if isinstance(generator, Plugin):
        save_to_file(path, generator)

    with open(path, "wb") as f:
        pickle.dump(generator, f)


def save_csv(data: pd.DataFrame, path: Path) -> None:
    """Save data to disk"""
    data.to_csv(path, index=False)


def save_json(path_to_file, data, verbose=1):
    with open(path_to_file, "w", encoding="utf-8") as outfile:
        json.dump(
            data, outfile, ensure_ascii=False, indent=4, default=make_json_serializable
        )

    if verbose > 0:
        print("Saved to disk:", path_to_file)


def make_json_serializable(value):
    """Convert values to be JSON serializable."""

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        return float(value)

    if isinstance(value, (str)):
        return str(value)

    if hasattr(value, "__dict__"):
        return str(value)

    raise ValueError(f"Cannot serialize input type: {type(value)}")


def object_to_dict(obj, exclude_keys=None):
    """
    Convert all parameters and values of a class to a dictionary that is JSON serializable.
    """
    if exclude_keys is None:
        exclude_keys = []

    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [object_to_dict(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return {
            k: object_to_dict(v)
            for k, v in obj.__dict__.items()
            if not (k.startswith("_") or k in exclude_keys)
        }
    else:
        return str(obj)
