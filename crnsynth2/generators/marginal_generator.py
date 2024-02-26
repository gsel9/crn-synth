from typing import Any, List

import pandas as pd
from synthesis.synthesizers.marginal import MarginalSynthesizer

from crnsynth2.generators.base_generator import BaseGenerator


class MarginalGenerator(BaseGenerator):
    """Marginal Generator based on the MarginalSynthesizer from synthesis

    Generate records based on marginal distribution of columns independently.
    """

    def __init__(self, epsilon: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.model = MarginalSynthesizer(epsilon=epsilon)

    def fit(self, data_real: pd.DataFrame, **kwargs: Any) -> None:
        """Fit the model to the real data."""
        self.model.fit(data_real)

    def generate(self, n_records: int) -> pd.DataFrame:
        """Generate records based on the marginal distribution."""
        return self.model.sample(n_records)
