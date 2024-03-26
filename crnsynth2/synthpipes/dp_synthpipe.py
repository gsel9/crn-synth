from typing import Dict, Iterable, List, Union

import pandas as pd

from crnsynth2.generators.base_generator import BaseGenerator
from crnsynth2.process.dp_stats import DPParam, dp_mean, dp_std
from crnsynth2.synthpipes.generalized_synthpipe import GeneralizedSynthPipe


class DPPipeline(GeneralizedSynthPipe):
    """Synthesis pipeline with generalization and option to compute differentially private params"""

    def __init__(
        self,
        generator: Union[BaseGenerator, None] = None,
        dp_params: Union[List[DPParam], None] = None,
        random_state: Union[int, None] = None,
        verbose: int = 1,
        generalize: bool = False,
        holdout_size: float = 0.2,
        target_column: Union[str, None] = None,
    ):
        """Initialize synthesis pipeline.

        Args:
            generator: Generator to use for synthesizing data
            dp_params: List of differentially private parameters to compute
            random_state: Random state
            verbose: Verbosity level
            generalize: Whether to generalize data before synthesis
            holdout_size: Proportion of data to use for holdout
            target_column: Name of column to use as target variable for stratifying train and test
        """
        super().__init__(
            generator=generator,
            random_state=random_state,
            verbose=verbose,
            generalize=generalize,
            holdout_size=holdout_size,
            target_column=target_column,
        )
        self.dp_params = dp_params

    def run(
        self, data_real, n_records: Union[int, None] = None
    ) -> Dict[str, pd.DataFrame]:
        """Run all steps in synthesis pipeline. User can run these steps one by one themselves as well."""
        # process real data
        data_real = self.process_data(data_real)

        # split into training and testing data to allow for evaluation with unseen data
        data_train, data_holdout = self._split_data(data_real)

        # compute differentially private params
        self._compute_dp_params(data_train)

        if self.verbose:
            self._compute_total_epsilon()

        # generalize data by binning numeric columns or grouping nominal columns
        if self.generalize:
            data_train_input = self._generalize_data(data_train)
        else:
            data_train_input = data_train

        # fit generator on processed training data
        self.fit(data_train_input)

        # generate records using fitted generator
        n_records = n_records if n_records is not None else data_real.shape[0]
        data_synth = self.generate(n_records)

        # reverse generalization of synthetic data
        if self.generalize:
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

    def _compute_dp_params(self, data_train: pd.DataFrame) -> None:
        """Compute differentially private params"""
        # compute dp params
        if self.dp_params is None:
            raise ValueError("No differentially private parameters have been set.")

        for dp_param in self.dp_params:
            # ensure random state and verbosity level is equal to pipeline
            dp_param.random_state = self.random_state
            dp_param.verbose = self.verbose

            # compute dp param - value is stored in object
            dp_param.compute(data_train)

    def _get_dp_param(self, stat_name, column):
        """Get differentially private parameter"""
        if not hasattr(self.dp_params[0], "param_"):
            raise ValueError(
                "Differentially private parameters have not been computed yet, run .run() first."
            )

        return next(
            dp_param.param_
            for dp_param in self.dp_params
            if dp_param.stat_name == stat_name and dp_param.column == column
        )

    def _compute_total_epsilon(self):
        """Compute total epsilon value for synthesis"""
        epsilon_params = sum([dp_param.epsilon for dp_param in self.dp_params])
        epsilon_generator = (
            self.generator.epsilon if hasattr(self.generator, "epsilon") else 0
        )
        print(f"\nEpsilon params: {epsilon_params}")
        print(f"Epsilon generator: {epsilon_generator}")
        print(f"Total epsilon: {epsilon_params + epsilon_generator}\n")
