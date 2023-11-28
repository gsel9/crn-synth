"""Custom metric implementations.
TODO: Custom scores can be implemented from a single class taking
in an arbitrary score_fn and kwargs  to the score_fn. Should make an issue on this.
"""
from typing import Any, Dict, List, Tuple

from pydantic import validate_arguments
from sklearn.metrics import mean_squared_error
from synthcity.metrics.eval_statistical import StatisticalEvaluator
from synthcity.plugins.core.dataloader import DataLoader

from crnsynth.evaluation.custom_metrics.utils import fit_cox, propensity_weights


def predicted_cumulative_hazard_score(
    hybrid_data,
    real_data,
    feature_cols,
    target_col,
    duration_col,
    clip_value=4,
    event_col=None,
):
    """

    Ideally, score = 0. For errors, score > 0. The score is computed as
        int(abs(A_real - A_hybrid)) / int(A_real)

    Args:
        hybrid_data (_type_): _description_
        real_data (_type_): _description_
        feature_cols (_type_): _description_
        target_col (_type_): _description_
        duration_col (_type_): _description_
        clip_value (int, optional): _description_. Defaults to 4.
        event_col (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
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

    ch_original = cox_original.predict_cumulative_hazard(real_data[feature_cols]).values
    ch_hybrid = cox_hybrid.predict_cumulative_hazard(hybrid_data[feature_cols]).values

    # predicted CH output has shape (time points x individuals)
    ch_max = ch_original[-1, :].max()
    return mean_squared_error(
        ch_original[-1, :].squeeze() / ch_max, ch_hybrid[-1, :].squeeze() / ch_max
    )


class PredictedCumulativeHazardScore(StatisticalEvaluator):
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
        return "predicted_cumulative_hazard_augmented_score"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt_aug: DataLoader, X_syn_aug: DataLoader) -> Dict:
        score = predicted_cumulative_hazard_score(
            hybrid_data=X_syn_aug.data,
            real_data=X_gt_aug.data,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            duration_col=self.duration_col,
            clip_value=self.CLIP_VALUE,
            event_col=self.event_col,
        )
        return {"score": score}
