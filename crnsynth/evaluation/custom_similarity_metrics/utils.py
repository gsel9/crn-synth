"""
"""
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.linear_model import LogisticRegression


def propensity_weights(X, y, clip_value=4, seed=42):
    model = LogisticRegression(random_state=seed, max_iter=500)
    model.fit(X, y)

    p_pred = model.predict_proba(X)
    p_score = (1 - y) * p_pred[:, 0] + y * p_pred[:, 1]

    if clip_value is not None:
        return np.clip(1 / p_score, 0, clip_value)

    return 1 / p_score


def fit_kaplanmeier(event_times, event_indicator, weights=None):
    """Kaplan-Meier estimate of median survival time.

    Args:'
        event_times: Time to event data.
        event_indicator: Outcome indicator.

    Returns:
        Median survival time estimate.
    """

    kmf = KaplanMeierFitter()
    kmf.fit(event_times, event_indicator, weights=weights)
    return kmf


def fit_cox(data, duration_col, cox_cols, weights_col=None, event_col=None):
    cox = CoxPHFitter()
    cox.fit(
        data[cox_cols],
        duration_col=duration_col,
        event_col=event_col,
        weights_col=weights_col,
    )

    return cox


def fit_flexible_parametric_model(
    data, duration_col, fit_cols, weights_col=None, event_col=None
):
    # fits a cubic spline for the baseline hazard
    fpm = CoxPHFitter(baseline_estimation_method="spline", n_baseline_knots=4)
    fpm.fit(
        data[fit_cols],
        duration_col=duration_col,
        weights_col=weights_col,
        event_col=event_col,
    )

    return fpm
