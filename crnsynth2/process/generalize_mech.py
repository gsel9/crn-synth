"""Generalization Mechanism"""
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.base import BaseEstimator, TransformerMixin

from crnsynth2.process.dp_stats import dp_mean, dp_std


class BaseGeneralizationMech(TransformerMixin, BaseEstimator):
    """Base class for generalization mechanisms."""

    def __init__(
        self, column, epsilon, bins, bounds, inverse, ignore_nan, random_state=None
    ):
        self.column = column
        self.epsilon = epsilon
        self.bins = bins
        self.bounds = bounds
        self.inverse = inverse
        self.ignore_nan = ignore_nan
        self.random_state = random_state

    def fit(self, data):
        """Fit the generalization mechanism to the data."""
        raise NotImplementedError("Fit method must be implemented.")

    def transform(self, data):
        """Transform the data using the generalization mechanism."""
        raise NotImplementedError("Transform method must be implemented.")

    def inverse_transform(self, data):
        """Inverse transform the data using the generalization mechanism."""
        raise NotImplementedError("Inverse transform method must be implemented.")

    def _check_params(self):
        """Check the parameters of the generalization mechanism."""
        if self.epsilon is None:
            raise ValueError("Epsilon must be provided.")

        if len(self.bounds) != 2:
            raise ValueError("Bounds must be a tuple of length 2.")

        if self.bounds[0] >= self.bounds[1]:
            raise ValueError("Lower bound must be less than upper bound.")

    def _check_data(self, data):
        """Check the data for the generalization mechanism."""
        if data is None:
            raise ValueError("Data must be provided.")

        if len(data) == 0:
            raise ValueError("Data must not be empty.")

        # store data type of column
        self.dtype_ = data[self.column].dtype


class NumericGeneralizationMech(BaseGeneralizationMech):
    """Generalization mechanism for numeric data"""

    def __init__(
        self,
        column,
        epsilon,
        bins,
        bounds,
        inverse="truncated_normal",
        ignore_nan=True,
        random_state=None,
    ):
        super().__init__(
            column=column,
            epsilon=epsilon,
            bins=bins,
            bounds=bounds,
            inverse=inverse,
            ignore_nan=ignore_nan,
            random_state=random_state,
        )

    def fit(self, data):
        """Fit the generalization mechanism to the data."""
        self._check_params()
        self._check_data(data)
        self.bin_edges_ = self._get_bin_edges(bins=self.bins, bounds=self.bounds)

        if self.inverse == "truncated_normal":
            # save dp mean and std for inverse transformation prior to generalization
            eps_param = self.epsilon / 2
            self.dp_mean_ = dp_mean(
                array=data[self.column],
                epsilon=eps_param,
                bounds=self.bounds,
                random_state=self.random_state,
            )
            self.dp_std_ = dp_std(
                array=data[self.column],
                epsilon=eps_param,
                bounds=self.bounds,
                random_state=self.random_state,
            )

        return self

    def transform(self, data):
        """Transform the data using the generalization mechanism."""
        data = data.copy()

        # clip values outside of bounds
        array = np.clip(data[self.column], self.bounds[0], self.bounds[1])

        # bin the data
        binned_data = np.digitize(array, self.bin_edges_[1:], right=False)
        data[self.column] = binned_data
        return data

    def inverse_transform(self, data):
        """Inverse transform the data using the generalization mechanism."""
        data = data.copy()

        # mask nan values or keep all data
        if self.ignore_nan:
            mask_nan = data[self.column].isna()
        else:
            mask_nan = np.zeros(data.shape[0], dtype=bool)

        array = data[self.column][~mask_nan]

        # get bounds of bins
        low_bound = self.bin_edges_[array]
        up_bound = self.bin_edges_[array + 1]

        # sample from uniform distribution
        if self.inverse == "uniform":
            if self.dtype_ == "float":
                sampled_values = np.random.uniform(low=low_bound, high=up_bound)
            elif self.dtype_ == "int":
                sampled_values = np.random.randint(low=low_bound, high=up_bound)
            else:
                raise ValueError("Data type must be either 'int' or 'float'.")

        # sample from truncated normal distribution
        elif self.inverse == "truncated_normal":
            low = (low_bound - self.dp_mean_) / self.dp_std_
            upp = (up_bound - self.dp_mean_) / self.dp_std_
            X = truncnorm(a=low, b=upp, loc=self.dp_mean_, scale=self.dp_std_)

            sampled_values = X.rvs(random_state=self.random_state)
            if self.dtype_ == "int":
                sampled_values = np.around(sampled_values, decimals=0).astype(int)

        # replace non-nan values with sampled values
        data.loc[~mask_nan, self.column] = sampled_values
        return data

    def _check_params(self):
        """Check the parameters of the generalization mechanism."""
        if self.inverse not in ["truncated_normal", "uniform"]:
            raise ValueError(
                "Inverse transformation must be either 'truncated_normal' or 'uniform'."
            )

        return super()._check_params()

    def _check_data(self, data):
        """Check the data for the generalization mechanism."""
        # check if data is numeric
        if not np.issubdtype(data[self.column].dtype, np.number):
            raise ValueError("Data must be numeric.")

        return super()._check_data(data)

    @staticmethod
    def _get_bin_edges(bins, bounds):
        """Get the bin edges based on bounds or check if provided bins are monotonically increasing."""
        # get equal-width bins when bins is integer
        if isinstance(bins, int):
            # take uniform bins
            bin_edges = np.linspace(bounds[0], bounds[1], bins)

            # round to 2 decimals
            bin_edges = np.round(bin_edges, 2)

        # check if provided bins are monotonically increasing
        else:
            bin_edges = np.asarray(bins)
            if np.any(bin_edges[:-1] > bin_edges[1:]):
                raise ValueError("`bins` must increase monotonically, when an array")
        return bin_edges


class CategoricalGeneralizationMech(BaseGeneralizationMech):
    """Generalization mechanism for categorical data"""

    def __init__(self, epsilon, inverse="uniform", random_state=None):
        self.epsilon = epsilon
        self.inverse = inverse
        self.random_state = random_state

    def fit(self, data, bounds):
        """Fit the generalization mechanism to the data."""
        self._check_params()

        if self.inverse == "uniform":
            pass

    def transform(self, data):
        pass

    def _check_data(self, data, bounds):
        """Check the data for the generalization mechanism."""
        # check if data is categorical
        if not data.dtype == "object":
            raise ValueError("Data must be categorical.")

        return super()._check_data(data, bounds)

    def _check_params(self):
        """Check the parameters of the generalization mechanism."""
        if self.inverse not in ["uniform"]:
            raise ValueError("Inverse transformation must be 'uniform'.")

        return super()._check_params()
