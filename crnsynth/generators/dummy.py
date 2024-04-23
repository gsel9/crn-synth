from crnsynth.generators.base import BaseGenerator


class DummyGenerator(BaseGenerator):
    """Samples records from the original dataset. Useful as a baseline for privacy metrics.

    When setting sample_with_replace to False and n_records equal to the original dataset,
    the generator will just create a copy of the fitted data.

    CAUTION: do not release the output of this generator when based on sensitive data!
    """

    def __init__(self, sample_with_replace=False, random_state=None, **kwargs):
        """Initialize the generator.

        Args:
            sample_with_replace (bool): if true record from data_real can be sampled more than once. Default is False.
            random_state (int): random state for reproducibility. Default is None.
        """
        super().__init__()
        self.sample_with_replace = sample_with_replace
        self.random_state = random_state
        self.model = None

    def fit(self, data_real) -> None:
        """Fit the model to the real data."""
        self.model = data_real.copy()

    def generate(self, n_records: int):
        """Copy the real data"""
        if (self.sample_with_replace is False) and (n_records > len(self.model)):
            raise ValueError(
                "Cannot sample more records than the original dataset. "
                "Reduce n_records or set sample_with_replace=True."
            )

        # copy original dataframe when sample_with_replace is False and n_records is equal to the original dataset
        if (self.sample_with_replace is False) and (n_records == len(self.model)):
            data_synth = self.model.copy()

        # sample records from the original dataset with or without replacement
        else:
            data_synth = (
                self.model.sample(
                    n_records,
                    replace=self.sample_with_replace,
                    random_state=self.random_state,
                )
                .copy()
                .reset_index(drop=True)
            )
        return data_synth
