import os

from crnsynth.etl.load import load_exp_config


# TODO: filter out files in a dir based on target str to include in output
def filenames_from_dir(path_to_dir, target_generators=None):
    to_keep = []

    filenames = os.listdir(path_to_dir)
    for filename in filenames:
        append = False

        name, _ = filename.split(".")
        _, gen_name, run_id = name.split("_")

        if target_generators is not None:
            append = any([gen_name == name for name in target_generators])

        if append:
            to_keep.append(filename)

    return to_keep


def filter_on_feature_value(data, feature_name, feature_vaule):
    """Filter a dataframe by the value of a specific feature"""
    return data[data[feature_name] == feature_vaule]


def run_id_from_filename(filename):
    # expect file naming convetion to be run-id_gen-name_exp-id
    return filename.split("_")[-1]


def config_query(exp_config, queries):
    """Extract information from a config file.

    Args:
        exp_config: A specific config file.
        queries: A list of keys to query the config file.

    Returns:
        Result from query.
    """

    result = exp_config.copy()
    for key in queries:
        result = result[key]

    return result
