"""Synthesis pipeline"""
import warnings
from typing import Union

import pandas as pd

# from sklearn.model_selection import train_test_split
from synthcity.plugins.core.dataloader import (
    GenericDataLoader,
    SurvivalAnalysisDataLoader,
)

from crnsynth2.generators.base import BaseGenerator
from crnsynth.evaluation import check

warnings.filterwarnings("default")


class BaseSynthPipe:
    def __init__(
        self,
        generator: BaseGenerator,
        generalize: bool,
        random_state: Union[int, None] = None,
    ) -> None:
        self.generator = generator
        self.generalize = generalize
        self.random_state = random_state

    @property
    def name(self):
        """Return the name of the pipeline"""
        raise NotImplementedError("name() method not implemented")

    def process_data(self, data_real: pd.DataFrame) -> pd.DataFrame:
        """Process real data"""
        raise NotImplementedError("Should implement .process_data()")

    def fit(self, data_real: pd.DataFrame) -> None:
        """Fit generator on processed real data"""
        raise NotImplementedError("Should implement .fit()")

    def generate(self, n_records: Union[int, None] = None) -> pd.DataFrame:
        """Generate records using fitted generator"""
        raise NotImplementedError("Should implement .generate()")

    def postprocess_synthetic_data(self, data_synth: pd.DataFrame) -> pd.DataFrame:
        """Postprocess synthetic data"""
        raise NotImplementedError("Should implement .postprocess_synthetic_data()")

    def run(self, data_real, n_records: Union[int, None] = None):
        """Run all steps in synthesis pipeline. User can run these steps one by one themselves as well."""
        data_real = self.process_data(data_real)
        self.fit(data_real)
        data_synth = self.generate(n_records)
        data_synth = self.postprocess_synthetic_data(data_synth)
        return data_synth

    def set_generator(self, generator: BaseGenerator) -> None:
        """Set generator for synthesis"""
        self.generator = generator


class BaseSynthPipe2:
    """Base class for synthesis pipelines. Handles data processing and generation of synthetic data.

    User can define their own synthesis pipeline by subclassing this class and implementing the following methods:
    - process_data
    - _generalize_data
    - _reverse_generalization
    - postprocess_synthetic_data

    Other methods are executed are set-up to ensure that the steps in pipeline are executed in the correct order.
    But they can be overridden if needed as well.

    Args:
        generator: Generator to use for synthesizing data
        data_name: Name of dataset
        target_column: Name of column to use as target variable for stratifying train and test
        test_size: Proportion of data to use for testing
        output_train_format: Whether to output synthetic data in same format as training data
        generalize: Whether to generalize data before synthesis
        data_loader_name: Name of data loader to use for loading data into generator
        random_state: Random state
        warn: Whether to show warnings
        verbose: Verbosity level
    """

    def __init__(
        self,
        generator,
        data_name,
        target_column=None,
        test_size=0.2,
        output_train_format=False,
        generalize=None,
        data_loader_name=None,
        random_state=None,
        warn=True,
        verbose=2,
    ) -> None:
        self.generator = generator
        self.data_name = data_name
        self.target_column = target_column
        self.test_size = test_size
        self.output_train_format = output_train_format
        self.generalize = generalize
        self.data_loader_name = data_loader_name
        self.random_state = random_state
        self.warn = warn
        self.verbose = verbose

        # check args
        self._check_args()

        # learned attributes
        self.column_names_ = []
        self.n_records_ = 0

    def get_dataloader(self):
        """Use data loader to load data into generator"""

        name = self.data_loader_name

        if name == "generic":
            loader_fn = GenericDataLoader

        elif name == "survival":
            loader_fn = SurvivalAnalysisDataLoader

        else:
            raise ValueError(f"Invalid data loader {name}")

        if self.verbose > 0:
            print(f"Using data loader for {name}")

        return loader_fn

    def cast_to_dataloader(self, data, loader_fn=None, **kwargs):
        """Use data loader to load data into generator"""

        if loader_fn is None:
            loader_fn = self.get_dataloader()

        return loader_fn(data, **kwargs)

    def process_data(self, data):
        """Process real data"""

        raise NotImplementedError("Should implement .process_data()")

    def fit(self, data_real):
        """Fit generator on processed real data"""

        data_real = self._check_input_data(data_real)

        if self.generator is None:
            raise ValueError(
                """Generator not set during init, use \\
                             '.set_generator(generator)' method to \\
                              assign a generator."""
            )

        self.generator.fit(data_real)

    # NOTE: should output dataloader for consistency
    def generate(self, n_records=None):
        """Generate records using fitted generator"""
        # use same number of records as real data if not specified
        if n_records is None:
            n_records = self.n_records_

        # generate synthetic data
        data_synth = self.generator.generate(n_records)

        return data_synth

    def postprocess_synthetic_data(self, data_synth):
        """Postprocess synthetic data"""

        raise NotImplementedError("Should implement .postprocess_synthetic_data()")

    def run(self, data_real, n_records=None):
        """Run all steps in synthesis pipeline. User can run these steps one by one themselves as well."""
        data_real = self.process_data(data_real)
        self.fit(data_real)
        data_synth = self.generate(n_records)
        data_synth = self.postprocess_synthetic_data(data_synth)
        return data_synth

    def _generalize_data(self, data_real):
        """Generalize data by binning numeric columns or grouping nominal columns"""

        if self.generalize:
            raise NotImplementedError("Should implement ._generalize_data()")

    def _reverse_generalization(self, data_synth):
        """Reverse generalization by de-binning numeric columns or de-grouping nominal columns"""

        if self.generalize:
            raise NotImplementedError("Should implement ._reverse_generalization()")

    def _reorder_columns(self, data_synth):
        """Reorder columns to original order of real data"""
        # select column names based on output format
        column_names = self.column_names_

        # get column order
        column_order = [c for c in column_names if c in data_synth.columns]

        # find columns that are missing or extra in synthetic data
        columns_synth_missing = set(column_names) - set(data_synth.columns)
        columns_synth_extra = set(data_synth.columns) - set(column_names)

        if self.warn:
            if columns_synth_missing:
                warnings.warn(
                    f"Synthetic data does not contain all columns of real data, missing columns: {list(columns_synth_missing)}"
                )

        # add extra columns of synthetic data (e.g. after post-processing) to start of dataframe
        if (self.output_train_format is False) and columns_synth_extra:
            column_order = list(columns_synth_extra) + column_order
            if self.warn:
                warnings.warn(
                    f"Synthetic data contains extra columns, added columns to start of dataframe: {list(columns_synth_extra)}"
                )

        # reorder columns
        return data_synth[column_order]

    def _save_input_format(self, data_real):
        """Save info of input data to ensure synth data can have the same format"""
        if not self.column_names_:
            self.column_names_ = list(data_real.columns)

        if not self.n_records_:
            self.n_records_ = data_real.shape[0]

    def _check_args(self):
        """Check validity of arguments"""
        assert self.output_train_format in [
            True,
            False,
        ], "Output train format must be True or False"
        assert self.warn in [True, False], "Warn must be True or False"

    def _check_input_data(self, data_real):
        """Check  data for any potential privacy risks"""
        # check rare categories - turn off by setting warn to False

        if not isinstance(data_real, (GenericDataLoader, SurvivalAnalysisDataLoader)):
            raise ValueError("Input data should be a data loader!")

        if self.warn:
            for column in data_real.columns:
                check.check_rare_categories(
                    data_real, column, min_support=0.05, verbose=self.verbose
                )

        return data_real
