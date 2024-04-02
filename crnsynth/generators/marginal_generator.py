from typing import Any, List

import pandas as pd
from synthesis.synthesizers.marginal import MarginalSynthesizer

from crnsynth.generators.base_generator import BaseGenerator


class MarginalGenerator(MarginalSynthesizer, BaseGenerator):
    """Marginal Generator based on the MarginalSynthesizer from synthesis

    Generate records based on marginal distribution of columns independently.
    """

    def __init__(self, epsilon: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def fit(self, data_real: pd.DataFrame) -> None:
        """Fit the model to the real data."""
        super().fit(data_real)

    def generate(self, n_records: int) -> pd.DataFrame:
        """Generate records based on the marginal distribution."""
        return super().sample(n_records)
