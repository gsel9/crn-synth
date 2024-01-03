"""Utility functions"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from synthcity.utils.serialization import load, load_from_file, save, save_to_file
from synthesis.synthesizers.privbayes import PrivBayes

from crnsynth.configs import config


def sample_subset(items, size, seed=None, return_residual=True):
    """Sample a subset from a collection of items."""

    if seed is not None:
        rnd = np.random.RandomState(seed)
        sample = rnd.choice(items, size=size, replace=False)

    else:
        sample = np.random.choice(items, size=size, replace=False)

    if return_residual:
        return sample, list(set(items) - set(sample))

    return sample


def update_config_from_kwargs(
    run_config, base_kwargs=None, data_kwargs=None, generator_kwargs=None
):
    """Modify the config key: value pairs in place based on optional keyword
    arguments.

    Args:
        run_config:
        base_kwargs:
        data_kwargs:
        generator_kwargs:

    """

    if base_kwargs is not None:
        _update_config(run_config, "base", base_kwargs)

    if data_kwargs is not None:
        _update_config(run_config, "data", data_kwargs)

    if generator_kwargs is not None:
        _update_config(run_config, "generator", generator_kwargs)


# TODO: handle nested dict updates by recursive func call
def _update_config(config, mode_key, kwargs):
    if not kwargs:
        return

    for value_key, value in kwargs.items():
        # insert data directly if unknown field
        if value_key not in config[mode_key]:
            _insert_new_field(config, mode_key, value_key, value)
            continue

        # nested values
        if isinstance(value, dict):
            for subject_key, subject_value in value.items():
                config[mode_key][value_key][subject_key] = subject_value
        else:
            config[mode_key][value_key] = value


def _insert_new_field(config, mode_key, value_key, value):
    config[mode_key].update({value_key: value})


def infmax(values):
    """Get the max value while ignoring <np.inf> values."""
    return np.max(values[np.invert(np.isinf(values))])


def gen_random_seed(n_seeds, min_val=0, max_val=100, seed=42, sorted=True):
    rnd = np.random.default_rng(seed)
    seeds = rnd.choice(np.arange(min_val, max_val), n_seeds, replace=False)

    if sorted:
        seeds.sort()

    return seeds


def check_synthetic_shape(data_synth, data_real):
    if np.shape(data_synth) != np.shape(data_real):
        warnings.warn(
            f"Data real has shape {np.shape(data_synth)} while data synth shape {np.shape(data_real)}"
        )


def make_output_dir(dataset_name):
    """Ensure output directory for results exists.
    Output path is a results folder broken down by input dataset dataset"""
    path_output = config.PATH_RESULTS / dataset_name

    if not os.path.exists(path_output):
        # create results / dataset_name
        os.makedirs(path_output)

    if not os.path.exists(path_output / "synthetic_data"):
        # create results / dataset_name / synthetic_data
        os.makedirs(path_output / "synthetic_data")

    if not os.path.exists(path_output / "generators"):
        # create results / dataset_name / generators
        os.makedirs(path_output / "generators")

    if not os.path.exists(path_output / "figures"):
        # create results / dataset_name / figures
        os.makedirs(path_output / "figures")

    if not os.path.exists(path_output / "reports"):
        # create results / dataset_name / reports
        os.makedirs(path_output / "reports")

    if not os.path.exists(path_output / "configs"):
        os.makedirs(path_output / "configs")


def get_path_output(dataset_name, output_type=None, verbose=0):
    """Get path to output broken down by dataset."""
    # ensure that output directory exists
    make_output_dir(dataset_name)
    out = config.PATH_RESULTS / dataset_name

    # optional: add output type
    if output_type is not None:
        output_types = ["synthetic_data", "generators", "figures", "reports", "configs"]
        assert output_type in output_types, f"output_type must be one of {output_types}"
        out = out / output_type

    if verbose > 0:
        print(f"Output saved to: {out}")
    return out


def get_filename(dataset_name, generator_name, fname_suffix):
    """Ensure consistent filenaming"""
    fname = f"{dataset_name}_{generator_name}"
    return fname + f"_{fname_suffix}"


def parse_filename_arg(filename, arg_idx, split="_", strip=None, arg_dtype=None):
    """Parse filename to get value at specified index and convert to required data type"""
    # split filename into parts
    fname_split = filename.split(split)

    if arg_idx >= len(fname_split):
        raise ValueError(f"Filename {filename} does not have {arg_idx} arguments")

    fname_arg = fname_split[arg_idx]

    # strip characters from argument
    if strip is not None:
        fname_arg = fname_arg.strip(strip)

    # convert argument to required data type
    if arg_dtype is not None:
        fname_arg = arg_dtype(fname_arg)
    return fname_arg


def parse_filename_args(filename, arg_type, extension=None):
    """Parse filename to get dataset name, generator name, epsilon value, or suffix"""
    # remove filename extension (e.g. .csv)
    fname = filename.strip(extension)

    # split filename into parts
    fname_split = fname.split("_")
    if arg_type == "dataset_name":
        return fname_split[0]
    elif arg_type == "generator_name":
        return fname_split[1]
    elif arg_type == "epsilon":
        return float(fname_split[2].strip("eps"))
    elif arg_type == "suffix":
        try:
            return fname_split[3]
        except IndexError:
            raise ValueError(f"Filename {filename} does not have a suffix")
    else:
        raise ValueError(
            f"Invalid arg_type: {arg_type}, must be one of: dataset_name, generator_name, epsilon, suffix"
        )


def save_output(
    filename,
    fname_suffix,
    df_synth=None,
    generator=None,
    config=None,
    score_report=None,
):
    """Save synthetic dataset and trained generator"""
    # ensure output directory for results exists
    # output path is a results folder broken down by input dataset dataset
    make_output_dir(filename)
    path_output = get_path_output(filename)

    fname = f"{filename}_{fname_suffix}"
    print(f"Filename: {fname}")

    # save synthetic data
    if df_synth is not None:
        df_synth.to_csv(path_output / f"synthetic_data/{fname}.csv", index=False)

    if generator is not None:
        # fix for recursion error in privbayes when trying to save using synthcity serializatoin
        if generator.name() == "privbayes-dk":
            generator.model.save(path_output / f"generators/{fname}.pkl")
        else:
            save_to_file(path_output / f"generators/{fname}.pkl", generator)

    if config is not None:
        with open(path_output / f"configs/{fname}.json", "w") as outfile:
            json.dump(config, outfile, indent=4, default=make_json_serializeable)

    if score_report is not None:
        score_report.to_csv(path_output / f"reports/{fname}.csv")


def make_json_serializeable(value):
    # Force output values to be generic type

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        return float(value)

    if isinstance(value, (str)):
        return str(value)

    raise ValueError(f"Cannot serialize input type: {type(value)}")


def load_output(
    filename,
    generator_name,
    epsilon,
    fname_suffix=None,
    load_generator=False,
    index_col=None,
):
    """Load synthetic dataset and generator"""
    path_output = get_path_output(filename)
    # fname = filename_suffix(filename, generator_name, epsilon, fname_suffix)
    fname = f"{filename}_{generator_name}"
    if fname_suffix is not None:
        fname += f"_{fname_suffix}"
    print(fname)

    df_synth = pd.read_csv(
        path_output / f"synthetic_data/{fname}.csv", index_col=index_col
    )

    if not load_generator:
        return df_synth

    # fix for recursion error in privbayes when trying to save using synthcity serializatoin
    if generator_name == "privbayes-dk":
        generator = PrivBayes.load(path_output / f"generators/{fname}.pkl")
    else:
        generator = load_from_file(path_output / f"generators/{fname}.pkl")

    return df_synth, generator
