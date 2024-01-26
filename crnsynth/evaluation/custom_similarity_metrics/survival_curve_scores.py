"""Custom metric implementations.
TODO: Custom scores can be implemented from a single class taking
in an arbitrary score_fn and kwargs  to the score_fn. Should make an issue on this.
"""
from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import validate_arguments
from scipy.integrate import trapezoid
from synthcity.metrics.eval_statistical import StatisticalEvaluator
from synthcity.plugins.core.dataloader import DataLoader

from .utils import fit_kaplanmeier


def survival_curves_deviation(hybrid_data, real_data, duration_col, event_col):
    kmf_real = fit_kaplanmeier(real_data[duration_col], real_data[event_col])
    kmf_hybrid = fit_kaplanmeier(hybrid_data[duration_col], hybrid_data[event_col])

    Tmax = max(kmf_real.timeline.max(), kmf_hybrid.timeline.max())
    Tmin = min(kmf_real.timeline.min(), kmf_hybrid.timeline.min())
    Tmin = max(0, Tmin)

    time_points = np.linspace(Tmin, Tmax, 200)

    S_hybrid = kmf_hybrid.survival_function_at_times(time_points)
    S_real = kmf_real.survival_function_at_times(time_points)

    return trapezoid(abs(S_hybrid.values - S_real.values)) / Tmax


class SurvivalCurvesDistanceScore(StatisticalEvaluator):
    DURATION_COL = None
    EVENT_COL = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "survival_curves_distance_score"

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
    def _evaluate(self, X_gt_aug: DataLoader, X_syn_aug: DataLoader) -> Dict:
        score = survival_curves_deviation(
            hybrid_data=X_syn_aug.data,
            real_data=X_gt_aug.data,
            duration_col=self.DURATION_COL,
            event_col=self.EVENT_COL,
        )
        return {"score": score}
