"""Custom metric implementations.
TODO: Custom scores can be implemented from a single class taking
in an arbitrary score_fn and kwargs  to the score_fn. Should make an issue on this.
"""
from typing import Any, Dict

import numpy as np
from pydantic import validate_arguments
from scipy.stats import pearsonr
from synthcity.metrics.eval_statistical import StatisticalEvaluator
from synthcity.plugins.core.dataloader import DataLoader

from .utils import fit_flexible_parametric_model, fit_kaplanmeier

# from crnsynth.process.util import infmax


def median_survival_score(hybrid_data, real_data, duration_col, event_col):
    """Deviation between the median survival times in the original and
    synthetic data. Survival curves are estimated with the Kaplan-Meier method.

    Optimal score value is zero.
    """
    km_original = fit_kaplanmeier(real_data[duration_col], real_data[event_col])
    km_hybrid = fit_kaplanmeier(hybrid_data[duration_col], hybrid_data[event_col])

    S_original = km_original.median_survival_time_
    S_hybrid = km_hybrid.median_survival_time_

    return abs(S_hybrid - S_original)


class MedianSurvivalScore(StatisticalEvaluator):
    """Cox beta score evaluator class."""

    DURATION_COL = None
    EVENT_COL = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "median_survival_augmented_score"

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
    hybrid_data_train,
    hybrid_data_test,
    real_data_train,
    real_data_test,
    feature_cols,
    duration_col,
    event_col=None,
):
    fit_cols = list(feature_cols) + [event_col, duration_col]

    fpm_original = fit_flexible_parametric_model(
        real_data_train, duration_col, fit_cols, event_col=event_col
    )
    fpm_hybrid = fit_flexible_parametric_model(
        hybrid_data_train, duration_col, fit_cols, event_col=event_col
    )
    # predict median survival for each data point
    t_original = fpm_original.predict_median(real_data_test[feature_cols]).values
    t_hybrid = fpm_hybrid.predict_median(hybrid_data_test[feature_cols]).values

    # handle censored data points
    Tmax = max(fpm_original.durations.max(), fpm_hybrid.durations.max())
    t_original[np.invert(np.isfinite(t_original))] = Tmax
    t_hybrid[np.invert(np.isfinite(t_hybrid))] = Tmax

    # baseline generators may produce constant output
    # here the correlation coefficient is undefined
    if np.var(t_hybrid) < 1e-12:
        return 0

    return pearsonr(t_original, t_hybrid).statistic


class PredictedMedianSurvivalScore(StatisticalEvaluator):
    """Predicted median survival score."""

    CLIP_VALUE = None
    FEATURE_COLS = None
    DURATION_COL = None
    TARGET_COL = None
    EVENT_COL = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "predicted_median_survival_augmented_score"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @classmethod
    def update_cls_params(cls, params):
        """Update the clip value class method without
        instantiating the class."""
        for name, value in params.items():
            setattr(cls, name, value)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt_aug: DataLoader, X_syn_aug: DataLoader) -> Dict:
        score = predicted_median_survival_score(
            hybrid_data_train=X_syn_aug.train().data,
            hybrid_data_test=X_syn_aug.test().data,
            real_data_train=X_gt_aug.train().data,
            real_data_test=X_gt_aug.test().data,
            feature_cols=self.FEATURE_COLS,
            duration_col=self.DURATION_COL,
            event_col=self.EVENT_COL,
        )
        # absolute pearson coefficient
        return {"score": abs(score)}
