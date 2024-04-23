from crnsynth.generators.base import BaseGenerator


class DummyGenerator(BaseGenerator):
    """Makes a copy of the original dataset. Useful as a baseline for privacy metrics.

    CAUTION: don't use this for generating synthetic data based on sensitive data!"""

    def __init__(self, **kwargs):
        super().__init__()
        self.model = None

    def fit(self, data_real) -> None:
        """Fit the model to the real data."""
        self.model = data_real.copy()

    def generate(self, n_records: int):
        """Copy the real data"""
        if n_records != len(self.model):
            raise ValueError(
                "n_records should be equal to the number of records in the real data."
            )
        return self.model.copy()
