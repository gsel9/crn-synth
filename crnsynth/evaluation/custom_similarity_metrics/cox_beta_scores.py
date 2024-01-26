"""Custom metric implementations.
TODO: Custom scores can be implemented from a single class taking
in an arbitrary score_fn and kwargs  to the score_fn. Should make an issue on this.
"""
from typing import Any, Dict, List, Tuple

import numpy as np

# from lifelines import CoxPHFitter, KaplanMeierFitter
from pydantic import validate_arguments

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import mean_squared_error
from synthcity.metrics.eval_statistical import StatisticalEvaluator
from synthcity.plugins.core.dataloader import DataLoader

from .utils import fit_cox, propensity_weights


def cox_beta_score(
    hybrid_data,
    real_data,
    feature_cols,
    target_col,
    duration_col,
    clip_value=4,
    event_col=None,
):
    """Compute hybrid data quality score based on Cox coefficients
    similarity.

    Args:
        real_data: Pre-processed real data.
        hybrid_data: Pre-processed hybrid data.
        feature_cols: Covariate names.
        target_col: Test column name (eg treatment).
        duration_col: Time to event column for Cox model.
        event_col: Optional event indicator column for Cox model.

    Returns:
        Score value.
    """
    hybrid_data["weights"] = propensity_weights(
        hybrid_data[feature_cols].values, hybrid_data[target_col].values, clip_value
    )
    real_data["weights"] = propensity_weights(
        real_data[feature_cols].values, real_data[target_col].values, clip_value
    )

    cox_cols = [target_col, duration_col, "weights"]
    if event_col is not None:
        cox_cols.append(event_col)

    cox_original = fit_cox(
        real_data, duration_col, cox_cols, weights_col="weights", event_col=event_col
    )
    cox_hybrid = fit_cox(
        hybrid_data, duration_col, cox_cols, weights_col="weights", event_col=event_col
    )

    mean_original = cox_original.params_[target_col]
    mean_hybrid = cox_hybrid.params_[target_col]

    return mean_original - mean_hybrid


class CoxBetaScore(StatisticalEvaluator):
    """Cox beta score evaluator class."""

    CLIP_VALUE = None
    FEATURE_COLS = None
    DURATION_COL = None
    TARGET_COL = None
    EVENT_COL = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "cox_beta_augmented_score"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @classmethod
    def update_cls_params(cls, params):
        for name, value in params.items():
            setattr(cls, name, value)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt_aug: DataLoader, X_syn_aug: DataLoader) -> Dict:
        score = cox_beta_score(
            hybrid_data=X_syn_aug.data,
            real_data=X_gt_aug.data,
            feature_cols=self.FEATURE_COLS,
            target_col=self.TARGET_COL,
            duration_col=self.DURATION_COL,
            clip_value=self.CLIP_VALUE,
            event_col=self.EVENT_COL,
        )
        # minimize absolute difference
        return {"score": abs(score)}
