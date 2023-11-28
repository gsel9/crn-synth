"""Custom metric implementations.
TODO: Custom scores can be implemented from a single class taking
in an arbitrary score_fn and kwargs  to the score_fn. Should make an issue on this.
"""
from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import validate_arguments
from sklearn.metrics import mean_squared_error
from synthcity.metrics.eval_statistical import StatisticalEvaluator
from synthcity.plugins.core.dataloader import DataLoader

from crnsynth.evaluation.custom_metrics.utils import (
    fit_cox,
    fit_kaplanmeier,
    propensity_weights,
)
from crnsynth.util import infmax


def median_survival_score(hybrid_data, real_data, duration_col, event_col):
    km_original = fit_kaplanmeier(real_data[duration_col], real_data[event_col])
    km_hybrid = fit_kaplanmeier(hybrid_data[duration_col], hybrid_data[event_col])

    S_original = km_original.median_survival_time_
    S_hybrid = km_hybrid.median_survival_time_

    return (S_original / S_original.max()) - (S_hybrid / S_original.max())


class MedianSurvivalScore(StatisticalEvaluator):
    """Cox beta score evaluator class."""

    def __init__(self, duration_col: str, event_col: str, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

        self.duration_col = duration_col
        self.event_col = event_col

    @staticmethod
    def name() -> str:
        return "median_survival_score"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt_aug: DataLoader, X_syn_aug: DataLoader) -> Dict:
        score = median_survival_score(
            X_syn_aug, X_gt_aug, self.duration_col, self.event_col
        )
        return {"score": score}


def predicted_median_survival_score(
    hybrid_data,
    real_data,
    feature_cols,
    target_col,
    duration_col,
    clip_value=4,
    event_col=None,
):
    hybrid_data["weights"] = propensity_weights(
        hybrid_data[feature_cols], hybrid_data[target_col], clip_value
    )
    real_data["weights"] = propensity_weights(
        real_data[feature_cols], real_data[target_col], clip_value
    )

    cox_cols = list(feature_cols) + [event_col, duration_col, "weights"]

    cox_original = fit_cox(
        real_data, duration_col, cox_cols, weights_col="weights", event_col=event_col
    )
    cox_hybrid = fit_cox(
        hybrid_data, duration_col, cox_cols, weights_col="weights", event_col=event_col
    )

    t_original = cox_original.predict_median(real_data[feature_cols]).values
    t_hybrid = cox_hybrid.predict_median(hybrid_data[feature_cols]).values

    # handle <inf> values by replacement to avoid alter sample count
    t_pred_max = infmax(t_original)
    t_original[np.isinf(t_original)] = t_pred_max
    t_hybrid[np.isinf(t_hybrid)] = t_pred_max

    return mean_squared_error(t_original / t_pred_max, t_hybrid / t_pred_max)


class PredictedMedianSurvivalScore(StatisticalEvaluator):
    """Predicted median survival score."""

    CLIP_VALUE = None

    def __init__(
        self,
        feature_cols: List,
        duration_col: str,
        target_col: str,
        event_col: str,
        **kwargs: Any
    ) -> None:
        super().__init__(default_metric="score", **kwargs)

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.duration_col = duration_col
        self.event_col = event_col

    @classmethod
    def update_clip_value(cls, new_clip_value):
        """Update the clip value class method without
        instantiating the class."""
        cls.CLIP_VALUE = new_clip_value

    @staticmethod
    def name() -> str:
        return "predicted_median_survival_augmented_score"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt_aug: DataLoader, X_syn_aug: DataLoader) -> Dict:
        score = predicted_median_survival_score(
            hybrid_data=X_syn_aug.data,
            real_data=X_gt_aug.data,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            duration_col=self.duration_col,
            clip_value=self.CLIP_VALUE,
            event_col=self.event_col,
        )
        return {"score": score}
