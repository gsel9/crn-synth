"""Functions to run synthesis pipeline on real data."""
from typing import Union

import pandas as pd

from crnsynth.generators.base_generator import BaseGenerator


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
