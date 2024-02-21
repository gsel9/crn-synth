from typing import Any, List

import pandas as pd
from synthesis.synthesizers.privbayes import PrivBayes

from crnsynth2.generators.base import BaseGenerator


class PrivBayesDK(BaseGenerator):
    """PrivBayes implementation from synthesis."""

    def __init__(self, epsilon, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.model = PrivBayes(epsilon=epsilon)

    def fit(self, data_real, **kwargs: Any) -> None:
        """Fit the model to the real data."""
        self.model.fit(data_real)

    def generate(self, n_records: int) -> pd.DataFrame:
        """Generate records based on the marginal distribution."""
        return self.model.sample(n_records)
