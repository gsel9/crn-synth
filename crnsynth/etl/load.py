import json

import pandas as pd


def load_csv(path_to_file, index_col=None, kwargs={}, verbose=1):
    data = pd.read_csv(path_to_file, index_col=index_col, **kwargs)

    # sanity check
    assert data is not None

    if verbose > 0:
        print("Loaded:", path_to_file)

    return data


def load_json(path_to_file, verbose=1):
    with open(path_to_file, "r") as infile:
        loaded_file = json.load(infile)

    # sanity check
    assert loaded_file is not None

    if verbose > 0:
        print("Loaded:", path_to_file)

    return loaded_file


def load_csv_generator(path_to_dir, filenames, yield_filename=True, index_col=0):
    for filename in filenames:
        data = load_csv(path_to_dir / filename, index_col=index_col)

        if yield_filename:
            yield filename, data

        else:
            yield data


def load_exp_config(path_to_dir, filename, location="configs"):
    """Load a specific experiment configuration file.

    Args:
        path_to_dir: Path to the results directory.
        filename: Name of the config file.

    Returns:

    """

    with open(path_to_dir / location / filename) as infile:
        exp_config = json.load(infile)

    return exp_config
