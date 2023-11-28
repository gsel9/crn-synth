"""Custom metric implementations.
TODO: Custom scores can be implemented from a single class taking
in an arbitrary score_fn and kwargs  to the score_fn. Should make an issue on this.
"""
from typing import Any, Dict, List, Tuple

import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from pydantic import validate_arguments
from scipy.integrate import trapezoid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from synthcity.metrics.eval_statistical import StatisticalEvaluator
from synthcity.plugins.core.dataloader import DataLoader

from crnsynth.evaluation.custom_metrics.utils import fit_kaplanmeier


def survival_curves_distance_score(hybrid_data, real_data, duration_col, event_col):
    kmf_real = fit_kaplanmeier(real_data[duration_col], real_data[event_col])
    kmf_hybrid = fit_kaplanmeier(hybrid_data[duration_col], hybrid_data[event_col])

    # pre-computing integrals allows for unequal number of time points in
    # real and hybrid data
    A_real = trapezoid(y=kmf_real.survival_function_.values.squeeze())
    A_hybrid = trapezoid(y=kmf_hybrid.survival_function_.values.squeeze())

    return abs(A_real - A_hybrid) / A_real


class SurvivalCurvesDistanceScore(StatisticalEvaluator):
    def __init__(self, duration_col: str, event_col: str, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

        self.duration_col = duration_col
        self.event_col = event_col

    @staticmethod
    def name() -> str:
        return "survival_curves_distance_augmented_score"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt_aug: DataLoader, X_syn_aug: DataLoader) -> Dict:
        score = survival_curves_distance_score(
            hybrid_data=X_syn_aug.data,
            real_data=X_gt_aug.data,
            duration_col=self.duration_col,
            event_col=self.event_col,
        )
        return {"score": score}
