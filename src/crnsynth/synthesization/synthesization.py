"""Functions to run synthesis pipeline on real data."""

from pathlib import Path
from typing import Union, Tuple, Optional

import pandas as pd

from crnsynth.generators.base import BaseGenerator
from crnsynth.serialization.paths import create_output_dir


# TODO: make decorators to integrate synthcity-specific objects 
# into workflow 
#@synthcity_integration
def generate_synth_data(
    data_real: pd.DataFrame,
    generator: BaseGenerator,
    n_samples: Optional[int] = None,
    verbose: bool = True 
) -> Tuple[pd.DataFrame, BaseGenerator]:
    """ Generate synthetic data using a generator.

    Args:
        data_real (pd.DataFrame): _description_
        generator (BaseGenerator): _description_
        n_samples (Optional[int], optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, BaseGenerator]: _description_
    """
    if n_samples is None:
        n_samples = data_real.shape[0]

    if verbose:
        print(f"Fitting generator {generator} on input data")
    
    # fit generator
    generator.fit(data_real)

    if verbose:
        print(f"Generating {n_samples} data samples")
        
    # generate synthetic data
    data_synth = generator.generate(n_samples)

    # return synthetic data and fitted generator
    return data_synth, generator
