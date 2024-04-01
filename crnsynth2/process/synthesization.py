"""Functions to run synthesis pipeline on real data."""
from typing import Dict, List, Union

import pandas as pd

from crnsynth2.generators.base_generator import BaseGenerator
from crnsynth2.process import check, postprocessing, preprocessing, utils
from crnsynth2.process.generalization import BaseGeneralizationMech


def generate_synth_data(
    data_real: pd.DataFrame,
    generator: BaseGenerator,
    n_records: Union[None, int],
    verbose: int = 1,
) -> tuple[pd.DataFrame, BaseGenerator]:
    """Generate synthetic data using a generator."""
    if n_records is None:
        n_records = data_real.shape[0]

    # fit generator
    if verbose:
        print(f"Fitting generator {generator} on input data")
    generator.fit(data_real)

    # generate synthetic data
    if verbose:
        print(f"Generator fitted. Generating {n_records} records")
    data_synth = generator.generate(n_records)

    # return synthetic data and fitted generator
    return data_synth, generator


def run_synth_pipeline(
    data_real: pd.DataFrame,
    generator: BaseGenerator,
    preprocess_func: Union[None, callable] = None,
    postprocess_func: Union[None, callable] = None,
    generalizers: Union[List[BaseGeneralizationMech], None] = None,
    holdout_size=0.2,
    target_column: Union[None, str] = None,
    n_records: Union[None, int] = None,
    output_keys: List[str] = ["train", "holdout", "synth", "generator"],
    random_state: Union[None, int] = None,
    verbose: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Generic synthesis pipeline from raw to synth data.

    Flexibility to specify custom preprocessing, generalization, and postprocessing functions.
    Can be used as an example for how to run a synthesis pipeline. Customize to own needs.
    """
    # check if random state is consistent for all classes
    check.check_consistent_random_state(
        classes=[generator, *generalizers], random_state=random_state
    )

    # split into training and testing data to allow for evaluation with unseen data
    data_train, data_holdout = preprocessing.split_train_holdout(
        data_real,
        holdout_size=holdout_size,
        target_column=target_column,
        random_state=random_state,
    )

    # process real data
    if preprocess_func is not None:
        data_train = preprocess_func(data_train)

    # generalize data by binning numeric columns or grouping nominal columns
    # note: assign to new dataframe data_train_input to be able to output training data prior to generalization later
    if generalizers is not None:
        data_train_input, generalizers = preprocessing.generalize_data(
            data_train, generalizers
        )
    else:
        data_train_input = data_train.copy()

    # use number of records of real data prior to splitting
    n_records = n_records if n_records is not None else data_real.shape[0]

    # generate synthetic data
    data_synth, generator = generate_synth_data(
        data_train_input, generator, n_records=n_records, verbose=verbose
    )

    # reverse generalization of synthetic data
    if generalizers is not None:
        data_synth, generalizers = postprocessing.reverse_generalization(
            data_synth, generalizers
        )

    # postprocess synthetic data
    if postprocess_func is not None:
        data_synth = postprocess_func(data_synth)

    # return output in a dictionary
    output = {
        "train": data_train,
        "input": data_train_input,
        "holdout": data_holdout,
        "synth": data_synth,
        "generator": generator,
        "generalizers": generalizers,
    }
    # reduce output dictionary to only include specified keys
    utils.reduce_dict(output, output_keys)
    return output
