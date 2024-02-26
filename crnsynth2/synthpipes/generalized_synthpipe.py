from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split

from crnsynth2.generators.base_generator import BaseGenerator
from crnsynth2.synthpipes.base_synthpipe import BaseSynthPipe


class GeneralizedSynthPipe(BaseSynthPipe):
    """Pipeline for synthesizing data with generalization. This pipeline includes steps for generalizing data before synthesis and reversing generalization after synthesis."""

    def __init__(
        self,
        generator: BaseGenerator,
        random_state: Union[int, None] = None,
        verbose: int = 1,
        generalize: bool = False,
        test_size: float = 0.2,
        target_column: Union[str, None] = None,
    ) -> None:
        """Initialize synthesis pipeline.

        Args:
            generator: Generator to use for synthesizing data
            random_state: Random state
            verbose: Verbosity level
            generalize: Whether to generalize data before synthesis
            test_size: Proportion of data to use for testing
            target_column: Name of column to use as target variable for stratifying train and test
        """
        super().__init__(
            generator=generator,
            random_state=random_state,
            verbose=verbose,
        )
        self.generalize = generalize
        self.test_size = test_size
        self.target_column = target_column

    def run(self, data_real, n_records: Union[int, None] = None):
        """Run all steps in synthesis pipeline. User can run these steps one by one themselves as well."""
        # process real data
        data_real = self.process_data(data_real)

        # split into training and testing data to allow for evaluation with unseen data
        data_train, _ = self._split_data(data_real)

        # generalize data by binning numeric columns or grouping nominal columns
        if self.generalize:
            data_train = self._generalize_data(data_train)

        # fit generator on processed training data
        self.fit(data_train)

        # generate records using fitted generator
        n_records = n_records if n_records is not None else data_real.shape[0]
        data_synth = self.generate(n_records)

        # reverse generalization of synthetic data
        if self.generalize:
            data_synth = self._reverse_generalization(data_synth)

        # postprocess synthetic data
        data_synth = self.postprocess_synthetic_data(data_synth)
        return data_synth

    def _split_data(self, data_real: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # split into training and testing data to allow for evaluation with unseen
        if self.test_size > 0:
            # stratify based on target column if specified
            stratify = (
                None if self.target_column is None else data_real[self.target_column]
            )

            data_train, data_test = train_test_split(
                data_real,
                test_size=self.test_size,
                stratify=stratify,
                random_state=self.random_state,
            )
        return data_train, data_test

    def _generalize_data(self, data_real: pd.DataFrame) -> pd.DataFrame:
        """Generalize data by binning numeric columns or grouping nominal columns"""
        raise NotImplementedError("Should implement ._generalize_data()")

    def _reverse_generalization(self, data_synth: pd.DataFrame) -> pd.DataFrame:
        """Reverse generalization of synthetic data"""
        raise NotImplementedError("Should implement ._reverse_generalization()")
