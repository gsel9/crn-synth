"""Module for hypothesis testing of non-inferiority between two groups."""
import typing

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats


def ttest_non_inferiority(
    mean1,
    stddev1,
    nobs1,
    mean2,
    stddev2,
    nobs2,
    relative_margin,
    equal_variance=False,
    lower_is_better=True,
):
    """Perform a one-sided t-test with a non-inferiority threshold for two independent samples.

    Args:
        TODO
    Returns:
        TODO

    """

    delta = relative_margin * mean1

    if lower_is_better:
        threshold = mean1 + delta
    else:
        threshold = mean1 - delta

    tstat, pvalue = stats.ttest_ind_from_stats(
        mean1=threshold,
        std1=stddev1,
        nobs1=nobs1,
        mean2=mean2,
        std2=stddev2,
        nobs2=nobs2,
        equal_var=equal_variance,
    )
    if lower_is_better:
        return tstat, 1 - pvalue / 2.0

    return tstat, pvalue / 2.0


def cox_non_inferiority_test(
    data: pd.DataFrame,
    margin: int,
    duration_col: str,
    group_col: str,
    event_col: typing.Optional[str] = None,
    lower_is_better=True,
    alpha: float = 0.05,
    cox_kwargs: typing.Optional[typing.Dict] = {},
    cox_fit_kwargs: typing.Optional[typing.Dict] = {},
) -> pd.Series:
    """Hypothesis testing of whether one group is inferior to another group.

    Based on notes from "Two-Sample Non-Inferiority Tests for Survival Data using Cox Regression"
    by NCSS.

    The test hypotheses are:
        H0: Insufficient evidence that group A is superior to group B
        H1: Can reject H0

    By rejecting H0, non-inferiority can be thus concluded at the given
    significance level between two groups.

    Args:
        data: Input data with group indicator variable, time to even and possibly
            additional covariates.
        margin: Threshold to determine the test outcome. Should be > 1 if
            lower is better. Otherwise < 1.
        duration_col: Name of the column that contains the time to event.
        group_col: Name of the column indicating the groups.
        event_col (optional): Name of the column that contains the death
            observations. If left as None, assume all individuals are uncensored.
        lower_is_better: Determine the angle of the test for the two groups.
            E.g.: If the endpoint represents death or relapse, higher hazards are
            said to be worse and lower is better. Alternatively, if duration is the time
            to cure or remission, then higher hazards are better.
        alpha: Test significance level.

    Returns:
        Test results.
    """

    if lower_is_better and margin < 1:
        raise ValueError(
            f"Should have margin > 1 if lower_is_better=True. Got {margin}"
        )

    if not lower_is_better and margin > 1:
        raise ValueError(
            f"Should have margin < 1 if lower_is_better=False. Got {margin}"
        )

    cox = CoxPHFitter(**cox_kwargs)
    cox.fit(data, duration_col=duration_col, event_col=event_col, **cox_fit_kwargs)

    # alpha to critical (one-sided)
    z_crit = stats.norm.ppf(1 - alpha)

    # confidence interval for the hazard ratio
    upper = np.exp(cox.params_[group_col] + z_crit * cox.standard_errors_[group_col])
    lower = np.exp(cox.params_[group_col] - z_crit * cox.standard_errors_[group_col])

    # test if group A is non-inferior to group B
    passed_test = upper < margin if lower_is_better else lower > margin

    output = pd.Series(
        {
            "hazard_ratio": np.exp(cox.params_[group_col]),
            "lower": lower,
            "upper": upper,
            "test_passed": passed_test,
        }
    )
    return output
