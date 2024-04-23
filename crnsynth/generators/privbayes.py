from pathlib import Path
from typing import Any, List, Union

import pandas as pd

# rename the import to avoid name conflict
from synthesis.synthesizers.privbayes import PrivBayes as PrivBayesDK

from crnsynth.generators.base import BaseGenerator


class PrivBayes(BaseGenerator):
    """PrivBayes implementation from synthesis.

    PrivBayes implementation of synthetic-data-generation library (DK). Other implementations can be found in synthcity
    or DataSynthesizer. However, these only concern the base version of the original PrivBayes paper.

    This version also implemented the following improvements mentioned in the extended paper:
    - R score function instead of Mutual Information - which has a lower sensitivity and thus requires less noise to compute.
    - Candidate attribute-parent pairs (AP-pairs) are determined based on the theta-usefulness criterion instead of setting a fixed max degree K.
    """

    def __init__(self, epsilon, verbose=1, **kwargs: Any) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.model = PrivBayesDK(epsilon=epsilon, verbose=verbose)

    def fit(self, data_real) -> None:
        """Fit the model to the real data."""
        self.model.fit(data_real)

    def generate(self, n_records: int) -> pd.DataFrame:
        """Generate records based on the trained model."""
        return self.model.sample(n_records)

    def save(self, path: Union[str, Path]) -> None:
        """Save the model to a file.

        PrivBayes has its own saving method, so we use that to avoid errors.
        """
        self.model.save(path)

    def load(path: Union[str, Path]) -> Any:
        """Load the model from a file.

        PrivBayes has its own loading method, so we use that to avoid errors.
        """
        return PrivBayesDK.load(path)
