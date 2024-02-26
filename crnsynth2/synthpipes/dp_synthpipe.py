from typing import Union

import pandas as pd

from crnsynth2.generators.base_generator import BaseGenerator
from crnsynth2.synthpipes.generalized_synthpipe import GeneralizedSynthPipe


class DPPipeline(GeneralizedSynthPipe):
    """Synthesis pipeline with generalization and option to compute differentially private params"""

    def __init__(
        self,
        epsilon_params: float,
        generator: BaseGenerator,
        random_state: Union[int, None] = None,
        verbose: int = 1,
        generalize: bool = False,
        test_size: float = 0.2,
        target_column: Union[str, None] = None,
    ):
        """Initialize synthesis pipeline.

        Args:
            epsilon_params: Epsilon value for differential private params (note: generator has its own epsilon value for synthesis - add up for total epsilon value)
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
            generalize=generalize,
            test_size=test_size,
            target_column=target_column,
        )
        self.epsilon_params = epsilon_params

    def run(self, data_real, n_records: Union[int, None] = None):
        """Run all steps in synthesis pipeline. User can run these steps one by one themselves as well."""
        # process real data
        data_real = self.process_data(data_real)

        # split into training and testing data to allow for evaluation with unseen data
        data_train, _ = self._split_data(data_real)

        # compute differentially private params
        self._compute_dp_params(data_train)

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

    def _compute_dp_params(self, data_train: pd.DataFrame) -> None:
        """Compute differentially private params"""
        raise NotImplementedError("Should implement ._compute_dp_params()")
