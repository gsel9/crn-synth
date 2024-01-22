"""Synthesis pipeline"""
import warnings

# from sklearn.model_selection import train_test_split
from synthcity.plugins.core.dataloader import (
    GenericDataLoader,
    SurvivalAnalysisDataLoader,
)

from crnsynth.evaluation import check

warnings.filterwarnings("default")


class BaseSynthPipe:
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

    def process_data(self, data_real, data_loader_kwargs=None):
        """Process real data"""

        # initialize data loader
        data_loader_kwargs = data_loader_kwargs or {}
        loader = self._init_data_loader(data_real, data_loader_kwargs)

        # save info of training data for creating synthetic data with same format
        if self.output_train_format:
            self._save_input_format(loader.train().dataframe())
        return loader

    def fit(self, data_real):
        """Fit generator on processed real data"""
        # only generalize training data to ensure that test data remains in same format as original
        if isinstance(data_real, GenericDataLoader) or isinstance(
            data_real, SurvivalAnalysisDataLoader
        ):
            data_real = data_real.train().dataframe().copy()

        if self.generator is None:
            raise ValueError(
                "Generator not set during init, use '.set_generator(generator)' method to assign a generator."
            )

        if self.generalize:
            data_real = self._generalize_data(data_real)

        self._check_input_data(data_real)

        self.generator.fit(data_real)

    def generate(self, n_records=None):
        """Generate records using fitted generator"""
        # use same number of records as real data if not specified
        if n_records is None:
            n_records = self.n_records_

        # generate synthetic data
        data_synth = self.generator.generate(n_records)

        # reverse generalization
        if self.generalize:
            data_synth.data = self._reverse_generalization(data_synth.dataframe())
        return data_synth

    def postprocess_synthetic_data(self, data_synth):
        """Postprocess synthetic data"""
        data_synth.data = self._reorder_columns(data_synth.dataframe())
        return data_synth

    def run(self, data_real, n_records=None):
        """Run all steps in synthesis pipeline. User can run these steps one by one themselves as well."""
        data_real = self.process_data(data_real)
        self.fit(data_real)
        data_synth = self.generate(n_records)
        data_synth = self.postprocess_synthetic_data(data_synth)
        return data_synth

    def set_generator(self, generator):
        """Set generator"""
        self.generator = generator

    def _generalize_data(self, data_real):
        """Generalize data by binning numeric columns or grouping nominal columns"""
        # user needs to implement their own generalization method
        if self.generalize:
            if type(self)._generalize_data == BaseSynthPipe._generalize_data:
                raise NotImplementedError(
                    "When 'generalize' is set to True, you must implement the '_generalize_data' method in your subclass."
                )
        return data_real

    def _reverse_generalization(self, data_synth):
        """Reverse generalization by de-binning numeric columns or de-grouping nominal columns"""
        # user needs to implement their own generalization method
        if self.generalize:
            if (
                type(self)._reverse_generalization
                == BaseSynthPipe._reverse_generalization
            ):
                raise NotImplementedError(
                    "When 'generalize' is set to True, you must implement the '_reverse_generalization' method in your subclass."
                )
        return data_synth

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
        assert (
            self.test_size >= 0 and self.test_size <= 1
        ), "Test size must be between 0 and 1"
        assert self.generalize in [True, False], "Generalize must be True or False"
        assert self.data_loader_name in [
            "generic",
            "survival",
        ], "Data loader must be 'generic' or 'survival'"
        assert self.warn in [True, False], "Warn must be True or False"

    def _check_input_data(self, data_real):
        """Check  data for any potential privacy risks"""
        # check rare categories - turn off by setting warn to False
        if self.warn:
            for column in data_real.columns:
                check.check_rare_categories(
                    data_real, column, min_support=0.05, verbose=self.verbose
                )

    def _init_data_loader(self, data, kwargs):
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

        return loader_fn(data, **kwargs)
