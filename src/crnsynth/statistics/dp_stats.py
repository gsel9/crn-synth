"""Compute differentially private statistics from data."""

from diffprivlib.tools.quantiles import median, quantile
from diffprivlib.tools.utils import mean, std, var


def dp_mean(array, epsilon, bounds, random_state=None):
    """Wrapper of mean function from diffprivlib. Computes the DP mean of an array."""
    return mean(array=array, epsilon=epsilon, bounds=bounds, random_state=random_state)


def dp_quantile(array, epsilon, bounds, quant, random_state=None):
    """Wrapper of quantile function from diffprivlib. Computes the DP quantile of an array."""
    return quantile(
        array=array,
        epsilon=epsilon,
        quant=quant,
        bounds=bounds,
        random_state=random_state,
    )


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
