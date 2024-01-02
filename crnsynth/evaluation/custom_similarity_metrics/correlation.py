"""This metric describes how difficult it is for an attacker to correctly guess the sensitive information using an algorithm called Correct Attribution Probability (CAP)"""
from typing import Any, Dict, List

from pydantic import validate_arguments
from sdmetrics.column_pairs import CorrelationSimilarity
from synthcity.metrics.eval_statistical import StatisticalEvaluator
from synthcity.plugins.core.dataloader import DataLoader


class CorrelationSimilarityScore(StatisticalEvaluator):
    """TODO"""

    NUMERICAL_COLS = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "correlation_similarity_score"

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
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        score = CorrelationSimilarity.compute(
            real_data=X_gt.data[self.NUMERICAL_COLS],
            synthetic_data=X_syn.data[self.NUMERICAL_COLS],
            coefficient="Pearson",
        )
        return {"score": score}
