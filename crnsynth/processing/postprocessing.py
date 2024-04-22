"""Post-processing functions for synthetic data after synthesis"""
from typing import List

import pandas as pd

from crnsynth.processing.generalization import BaseGeneralizationMech


def reverse_generalization(
    data_synth: pd.DataFrame, generalizers: List[BaseGeneralizationMech]
) -> pd.DataFrame:
    """Reverse generalization of synthetic data using a list of generalization mechanisms."""
    data_synth_rev = data_synth.copy()
    for gen_mech in generalizers:
        data_synth_rev = gen_mech.inverse_transform(data_synth_rev)
    return data_synth_rev, generalizers
