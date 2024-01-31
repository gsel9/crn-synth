"""A module for computing differential privacy statistics on real data to be used as parameters for the synthetic data generation process."""
from typing import Union

import pandas as pd
from diffprivlib.tools.quantiles import median
from diffprivlib.tools.utils import mean


class DPParams:
    """Compute differential privacy parameters from real data."""

    def __init__(self, stat_dict: dict, epsilon: Union[float, dict], verbose=1):
        """Initialize the DPParams object.

        Args:
            stat_dict (dict): A dictionary of DP statistics to compute. The keys are the column names, and the values are the statistics to compute.
            epsilon (float or dict: The epsilon value to use for DP computation.
                float: divide epsilon over all stats to be computed.
                dict: specify epsilon for each stat, same format as stat_dict.

        Example:
            input:
                stat_dict = {
                        'age': {
                            'mean': dp_mean,
                            'max': dp_max
                        }
                        'os': {
                            'median': dp_median
                        }
                }

            compute output:
                out = {
                    'age' : {
                        'mean' : 50
                        'max': 95
                    }
                    'os' : {
                        'median': 20
                    }
                }
        """
        self.stat_dict = stat_dict
        self.epsilon = epsilon
        self.verbose = verbose

    def compute(self, data_real):
        """Compute DP parameters from real data.

        Args:
            data_real (pd.DataFrame): The real data to compute DP parameters from.

        Returns:
            pd.DataFrame: A dataframe containing the DP parameters.
        """
        epsilon_dict = self._divide_epsilon()
        print(epsilon_dict)
        dp_params = {}
        for col, func_dict in self.stat_dict.items():
            pass

        return pd.DataFrame(dp_params)

    def _divide_epsilon(self):
        """Divide epsilon by the number of columns."""
        if isinstance(self.epsilon, float):
            epsilon_dict = {}

            epsilon_per_col = self.epsilon / len(self.stat_dict)

            for col, func_dict in self.stat_dict.items():
                epsilon_dict[col] = {}
                for func_name in func_dict.keys():
                    epsilon_dict[col][func_name] = epsilon_per_col / len(func_dict)
            return epsilon_dict
        else:
            epsilon_dict = self.epsilon

        if self.verbose:
            total_epsilon = self._count_total_epsilon(epsilon_dict)
            print(f"Total epsilon used for DP computation: {total_epsilon}")
            print(f"Epsilon split per stat: {epsilon_dict}")

        return epsilon_dict

    def _count_total_epsilon(self, epsilon_dict):
        """Count the total epsilon used for DP computation."""
        total_epsilon = 0
        for _, func_dict in epsilon_dict.items():
            for _, epsilon in func_dict.items():
                total_epsilon += epsilon
        return total_epsilon


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
    Will thus not estimate basedo on array, but return the upperbound instead."""
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
    Will thus not estimate basedo on array, but return the lowerbound instead."""
    if verbose:
        print(
            "A minimum has unbounded sensitivity, thus cannot compute based on data. Will select min from bounds instead."
        )

    if bounds is None:
        raise ValueError("Bounds must be specified to compute DP min.")
    min_bound = min(bounds)
    return min_bound
