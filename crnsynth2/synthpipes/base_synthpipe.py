"""Synthesis pipeline"""
import warnings
from typing import Dict, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from crnsynth2.generators.base_generator import BaseGenerator

warnings.filterwarnings("default")


class BaseSynthPipe:
    """Base class for synthesis pipelines. Handles data processing and generation of synthetic data.
    Use AdvancedSynthPipe to inherit more functionality."""

    def __init__(
        self,
        generator: Union[BaseGenerator, None] = None,
        holdout_size: float = 0.2,
        target_column: Union[str, None] = None,
        random_state: Union[int, None] = None,
        verbose: int = 1,
    ) -> None:
        """Initialize synthesis pipeline.

        Args:
            generator: Generator to use for synthesizing data
            holdout_size: Proportion of data to use for evaluation and not seen by generator
            random_state: Random state
            verbose: Verbosity level
        """
        self.generator = generator
        self.holdout_size = holdout_size
        self.target_column = target_column
        self.random_state = random_state
        self.verbose = verbose

    def process_data(self, data_real: pd.DataFrame) -> pd.DataFrame:
        """Process real data"""
        raise NotImplementedError("Should implement .process_data()")

    def fit(self, data_real: pd.DataFrame) -> "BaseSynthPipe":
        """Fit generator on processed real data"""
        if self.generator is None:
            raise ValueError(
                "Generator is not set. Please set generator before fitting."
            )

        # ensure random state is equal to pipeline random state
        self.generator.random_state = self.random_state

        self.generator.fit(data_real)
        return self

    def generate(self, n_records: int) -> pd.DataFrame:
        """Generate records using fitted generator"""
        return self.generator.generate(n_records)

    def postprocess_synthetic_data(
        self, data_synth: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Postprocess synthetic data"""
        raise NotImplementedError("Should implement .postprocess_synthetic_data()")

    def run(
        self, data_real, n_records: Union[int, None] = None
    ) -> Dict[str, pd.DataFrame]:
        """Run all steps in synthesis pipeline. User can run these steps one by one themselves as well."""
        # process real data
        data_real = self.process_data(data_real)

        # split into training and testing data to allow for evaluation with unseen data
        data_train, data_holdout = self._split_data(data_real)

        # fit generator on processed real data
        self.fit(data_real)

        # generate records using fitted generator
        n_records = n_records if n_records is not None else data_real.shape[0]
        data_synth = self.generate(n_records)

        # postprocess synthetic data
        data_synth = self.postprocess_synthetic_data(data_synth)

        # return output datasets in a dictionary
        data_out = {
            "train": data_train,
            "holdout": data_holdout,
            "synth": data_synth,
        }
        return data_out

    def set_generator(self, generator: BaseGenerator):
        """Set generator for synthesis"""
        self.generator = generator
        return self

    def set_params(self, **params):
        """Set parameters"""
        for key, value in params.items():
            # check if attribute exists
            if hasattr(self, key):
                setattr(self, key, value)

    def get_params(self):
        return self.__dict__

    def _split_data(self, data_real: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # split into training and testing data to allow for evaluation with unseen
        if self.holdout_size > 0:
            # stratify based on target column if specified
            stratify = (
                None if self.target_column is None else data_real[self.target_column]
            )

            data_train, data_holdout = train_test_split(
                data_real,
                test_size=self.holdout_size,
                stratify=stratify,
                random_state=self.random_state,
            )
        else:
            data_train = data_real
            data_holdout = pd.DataFrame()
        return data_train, data_holdout

    def __repr__(self):
        msg = f"{self.__class__.__name__}({self.__dict__})"
        return msg

    def __copy__(self):
        return self.__class__(**self.__dict__)
