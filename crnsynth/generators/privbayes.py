from typing import Any, List

import pandas as pd

# rename the import to avoid name conflict
from synthesis.synthesizers.privbayes import PrivBayes as PrivBayesDK

from crnsynth.generators.base_generator import BaseGenerator


class PrivBayes(PrivBayesDK, BaseGenerator):
    """PrivBayes implementation from synthesis.

    PrivBayes implementation of synthetic-data-generation library (DK). Other implementations can be found in synthcity
    or DataSynthesizer. However, these only concern the base version of the original PrivBayes paper.

    This version also implemented the following improvements mentioned in the extended paper:
    - R score function instead of Mutual Information - which has a lower sensitivity and thus requires less noise to compute.
    - Candidate attribute-parent pairs (AP-pairs) are determined based on the theta-usefulness criterion instead of setting a fixed max degree K.
    """

    def __init__(self, epsilon, **kwargs: Any) -> None:
        super().__init__(epsilon=epsilon)

    def fit(self, data_real) -> None:
        """Fit the model to the real data."""
        super().fit(data_real)

    def generate(self, n_records: int) -> pd.DataFrame:
        """Generate records based on the trained model."""
        return super().sample(n_records)
