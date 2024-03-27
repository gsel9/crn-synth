from typing import Dict, List, Union

import pandas as pd

from crnsynth2.generators.base_generator import BaseGenerator
from crnsynth2.process.generalize_mech import BaseGeneralizationMech
from crnsynth2.synthpipes.base_synthpipe import BaseSynthPipe


class GeneralizedSynthPipe(BaseSynthPipe):
    """Pipeline for synthesizing data with generalization. This pipeline includes steps for generalizing data before synthesis and reversing generalization after synthesis."""

    def __init__(
        self,
        generator: Union[BaseGenerator, None] = None,
        random_state: Union[int, None] = None,
        verbose: int = 1,
        generalizers: Union[List[BaseGeneralizationMech], None] = None,
        holdout_size: float = 0.2,
        target_column: Union[str, None] = None,
    ) -> None:
        """Initialize synthesis pipeline.

        Args:
            generator: Generator to use for synthesizing data
            random_state: Random state
            verbose: Verbosity level
            generalizers: List of generalization mechanisms to use
            holdout_size: Proportion of data to use for holdout
            target_column: Name of column to use as target variable for stratifying train and test
        """
        super().__init__(
            generator=generator,
            random_state=random_state,
            holdout_size=holdout_size,
            verbose=verbose,
        )
        self.generalizers = generalizers
        self.target_column = target_column

    def run(
        self, data_real, n_records: Union[int, None] = None
    ) -> Dict[str, pd.DataFrame]:
        """Run all steps in synthesis pipeline. User can run these steps one by one themselves as well."""
        # process real data
        data_real = self.process_data(data_real)

        # split into training and testing data to allow for evaluation with unseen data
        data_train, data_holdout = self._split_data(data_real)

        # generalize data by binning numeric columns or grouping nominal columns
        if self.generalizers:
            data_train_input = self._generalize_data(data_train)
            if self.verbose:
                self._compute_total_epsilon()
        else:
            data_train_input = data_train

        # fit generator on processed training data
        self.fit(data_train_input)

        # generate records using fitted generator
        n_records = n_records if n_records is not None else data_real.shape[0]
        data_synth = self.generate(n_records)

        # reverse generalization of synthetic data
        if self.generalizers:
            data_synth = self._reverse_generalization(data_synth)

        # postprocess synthetic data
        data_synth = self.postprocess_synthetic_data(data_synth)

        # return output datasets in a dictionary
        data_out = {
            "train": data_train,
            "holdout": data_holdout,
            "synth": data_synth,
        }
        return data_out

    def _generalize_data(self, data_real: pd.DataFrame) -> pd.DataFrame:
        """Generalize data by binning numeric columns or grouping nominal columns"""
        for generalizer in self.generalizers:
            data_real = generalizer.fit_transform(data_real)
        return data_real

    def _reverse_generalization(self, data_synth: pd.DataFrame) -> pd.DataFrame:
        """Reverse generalization of synthetic data"""
        for generalizer in self.generalizers:
            data_synth = generalizer.inverse_transform(data_synth)
        return data_synth

    def _check_params(self) -> None:
        """Check that all parameters are set"""
        if self.generalizers is None:
            raise ValueError("No generalizers have been set.")

        # consistent random state for all classes
        if self.random_state is not None:
            for generalizer in self.generalizers:
                generalizer.random_state = self.random_state
            self.generator.random_state = self.random_state

        # consistent verbosity level for all classes
        if self.verbose:
            for generalizer in self.generalizers:
                generalizer.verbose = self.verbose
            self.generator.verbose = self.verbose

    def _compute_total_epsilon(self):
        """Compute total epsilon value for synthesis"""
        epsilon_generalizers = 0
        for generalizer in self.generalizers:
            if generalizer.epsilon is not None:
                epsilon_generalizers += generalizer.epsilon

        epsilon_generator = (
            self.generator.epsilon if hasattr(self.generator, "epsilon") else 0
        )
        print(f"\nEpsilon generalizers: {epsilon_generalizers}")
        print(f"Epsilon generator: {epsilon_generator}")
        print(f"Total epsilon: {epsilon_generalizers + epsilon_generator}\n")
