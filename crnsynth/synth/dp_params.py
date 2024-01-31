"""A module for computing differential privacy statistics on real data to be used as parameters for the synthetic data generation process."""
from typing import Iterable, Union

import pandas as pd
from diffprivlib.tools.quantiles import median
from diffprivlib.tools.utils import mean, std, var


def dp_mean(array, epsilon, bounds, random_state, verbose=1):
    """Wrapper of mean function from diffprivlib. Computes the DP mean of an array."""
    return mean(
        array=array,
        epsilon=epsilon,
        bounds=bounds,
        random_state=random_state,
        verbose=verbose,
    )


def dp_median(array, epsilon, bounds, random_state, verbose=1):
    """Wrapper of median function from diffprivlib. Computes the DP median of an array."""
    return median(
        array=array,
        epsilon=epsilon,
        bounds=bounds,
        random_state=random_state,
        verbose=verbose,
    )


def dp_max(array, epsilon, bounds, random_state, verbose=1):
    """Compute the DP max of an array. Just a placeholder function, as the sensitivty of the maximum in unbounded.
    Will thus not estimate based on array, but return the upperbound instead."""
    if verbose:
        print(
            "A maximum has unbounded sensitivity, thus cannot compute based on data. Will select max from bounds instead."
        )

    if bounds is None:
        raise ValueError("Bounds must be specified to compute DP max.")
    max_bound = max(bounds)
    return max_bound


def dp_min(array, epsilon, bounds, random_state, verbose=1):
    """Compute the DP min of an array. Just a placeholder function, as the sensitivty of the minimum in unbounded.
    Will thus not estimate based on array, but return the lowerbound instead."""
    if verbose:
        print(
            "A minimum has unbounded sensitivity, thus cannot compute based on data. Will select min from bounds instead."
        )

    if bounds is None:
        raise ValueError("Bounds must be specified to compute DP min.")
    min_bound = min(bounds)
    return min_bound


def dp_var(array, epsilon, bounds, random_state, verbose=1):
    """Wrapper of var function from diffprivlib. Computes the DP variance of an array."""
    return var(
        array=array,
        epsilon=epsilon,
        bounds=bounds,
        random_state=random_state,
        verbose=verbose,
    )


def dp_std(array, epsilon, bounds, random_state, verbose=1):
    """Wrapper of std function from diffprivlib. Computes the DP std of an array."""
    return std(
        array=array,
        epsilon=epsilon,
        bounds=bounds,
        random_state=random_state,
        verbose=verbose,
    )


DP_PARAM_FUNC = {
    "mean": dp_mean,
    "median": dp_median,
    "max": dp_max,
    "min": dp_min,
    "var": dp_var,
    "std": dp_std,
}


class DPParam:
    """Compute a differential privacy parameter from real data."""

    def __init__(
        self,
        stat_name: str,
        epsilon: float,
        column: str,
        bounds: Iterable,
        random_state: Union[int, None] = None,
        verbose: int = 1,
    ):
        """Initialize the DPParam object.

        Args:
            stat_name (str): The name of the statistic to compute.
            epsilon (float): The epsilon value to use for DP computation.
            column (str): The column to compute the statistic on.
            bounds (Iterable): The min and max bounds of the stat.
            random_state (int or None): The random state to use for DP computation.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        self.stat_name = stat_name
        self.epsilon = epsilon
        self.bounds = bounds
        self.column = column
        self.random_state = random_state
        self.verbose = verbose

    def compute(self, data_real: pd.DataFrame):
        """Compute a differential privacy parameter from real data.

        Args:
            data_real (pd.DataFrame): The real data to compute the parameter from.

        Returns:
            float: The computed parameter.
        """
        if self.verbose:
            print(f"Computing DP parameter {self.stat_name} on column {self.column}")

        # compute dp param
        dp_param = DP_PARAM_FUNC[self.stat_name](
            array=data_real[self.column],
            epsilon=self.epsilon,
            bounds=self.bounds,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # store param in object to re-use in case you don't want to re-compute
        self.param_ = dp_param
        return dp_param
