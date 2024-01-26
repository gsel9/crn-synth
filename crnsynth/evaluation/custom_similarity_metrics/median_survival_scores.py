"""Custom metric implementations.
TODO: Custom scores can be implemented from a single class taking
in an arbitrary score_fn and kwargs  to the score_fn. Should make an issue on this.
"""
from typing import Any, Dict

import numpy as np
from pydantic import validate_arguments
from scipy.integrate import trapezoid
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from synthcity.metrics.eval_statistical import StatisticalEvaluator
from synthcity.plugins.core.dataloader import DataLoader

from .utils import fit_flexible_parametric_model, fit_kaplanmeier


def median_survival_score(hybrid_data, real_data, duration_col, event_col):
    """Deviation between the median survival times in the original and
    synthetic data. Survival curves are estimated with the Kaplan-Meier method.

    Optimal score value is zero.
    """
    km_original = fit_kaplanmeier(real_data[duration_col], real_data[event_col])
    km_hybrid = fit_kaplanmeier(hybrid_data[duration_col], hybrid_data[event_col])

    S_original = km_original.median_survival_time_
    S_hybrid = km_hybrid.median_survival_time_

    # scale to unit range
    Tmax = max(km_original.timeline.max(), km_hybrid.timeline.max())
    return abs(S_hybrid - S_original) / Tmax


class MedianSurvivalScore(StatisticalEvaluator):
    """Cox beta score evaluator class."""

    DURATION_COL = None
    EVENT_COL = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "median_survival_score"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @classmethod
    def update_cls_params(cls, params):
        for name, value in params.items():
            setattr(cls, name, value)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt_aug: DataLoader, X_syn_aug: DataLoader) -> Dict:
        score = median_survival_score(
            X_syn_aug, X_gt_aug, self.DURATION_COL, self.EVENT_COL
        )
        return {"score": score}


def predicted_median_survival_score(
    synth_data,
    real_data,
    feature_cols,
    duration_col,
    event_col=None,
):
    fit_cols = list(feature_cols) + [event_col, duration_col]

    real_data_train = real_data.train().data
    real_data_test = real_data.test().data
    synth_data = synth_data.data

    fpm_real = fit_flexible_parametric_model(
        real_data_train, duration_col, fit_cols, event_col=event_col
    )
    fpm_synth = fit_flexible_parametric_model(
        synth_data, duration_col, fit_cols, event_col=event_col
    )

    Tmax = max(real_data.data[duration_col].max(), synth_data[duration_col].max())
    Tmin = min(real_data.data[duration_col].min(), synth_data[duration_col].min())
    Tmin = max(0, Tmin)

    times = np.linspace(Tmin, Tmax, 200)

    # predict median survival for each data point
    S_real = fpm_real.predict_survival_function(real_data_test[fit_cols], times=times)
    S_synth = fpm_synth.predict_survival_function(real_data_test[fit_cols], times=times)

    if np.invert(np.isfinite(S_real)).any():
        raise ValueError("predicted median: non-finite in S_real")
    if np.invert(np.isfinite(S_synth)).any():
        raise ValueError("predicted median: non-finte in S_synth")

    score = trapezoid(abs(S_synth.values - S_real.values)) / Tmax
    return score


class PredictedMedianSurvivalScore(StatisticalEvaluator):
    """Predicted median survival score."""

    FEATURE_COLS = None
    DURATION_COL = None
    EVENT_COL = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "predicted_median_survival_score"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @classmethod
    def update_cls_params(cls, params):
        """Update the clip value class method without
        instantiating the class."""
        for name, value in params.items():
            setattr(cls, name, value)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, data_real: DataLoader, data_synth: DataLoader) -> Dict:
        score = predicted_median_survival_score(
            synth_data=data_synth,
            real_data=data_real,
            feature_cols=self.FEATURE_COLS,
            duration_col=self.DURATION_COL,
            event_col=self.EVENT_COL,
        )
        return {"score": score}
