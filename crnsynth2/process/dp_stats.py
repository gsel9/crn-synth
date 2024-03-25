"""Compute DP stats from real data."""
from diffprivlib.tools.quantiles import median
from diffprivlib.tools.utils import mean, std, var


def dp_mean(array, epsilon, bounds, random_state=None):
    """Wrapper of mean function from diffprivlib. Computes the DP mean of an array."""
    return mean(array=array, epsilon=epsilon, bounds=bounds, random_state=random_state)


def dp_median(array, epsilon, bounds, random_state=None):
    """Wrapper of median function from diffprivlib. Computes the DP median of an array."""
    return median(
        array=array, epsilon=epsilon, bounds=bounds, random_state=random_state
    )


def dp_var(array, epsilon, bounds, random_state):
    """Wrapper of var function from diffprivlib. Computes the DP variance of an array."""
    return var(array=array, epsilon=epsilon, bounds=bounds, random_state=random_state)


def dp_std(array, epsilon, bounds, random_state):
    """Wrapper of std function from diffprivlib. Computes the DP std of an array."""
    return std(array=array, epsilon=epsilon, bounds=bounds, random_state=random_state)


DP_PARAM_FUNC = {
    "mean": dp_mean,
    "median": dp_median,
    "var": dp_var,
    "std": dp_std,
}
