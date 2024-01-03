"""This metric describes how difficult it is for an attacker to correctly guess the sensitive information using an algorithm called Correct Attribution Probability (CAP)"""
import os
from typing import Any, Dict, List

import torch
from pydantic import validate_arguments
from sdmetrics.single_table import CategoricalCAP
from synthcity.metrics.eval_privacy import PrivacyEvaluator
from synthcity.plugins.core.dataloader import DataLoader

# from src.util import sample_subset
from crnsynth.process.util import sample_subset


class CategoricalCAPScore(PrivacyEvaluator):
    """TODO"""

    CATEGORICAL_COLS = None
    FRAC_SENSITIVE = None

    def __init__(self, seed=42, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)
        self.seed = seed

    @property
    def n_categorical(self):
        return int(len(self.CATEGORICAL_COLS))

    @staticmethod
    def name() -> str:
        return "cap_categorical_score"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @classmethod
    def update_cls_params(cls, params):
        """Update the clip value class method without
        instantiating the class."""
        for name, value in params.items():
            setattr(cls, name, value)

    # TODO: get sensitive_fields from X_gt.sensitive_columns
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        n_sensitive = int(self.n_categorical * self.FRAC_SENSITIVE)
        if n_sensitive == 0:
            return

        known_fields, sensitive_fields = sample_subset(
            self.CATEGORICAL_COLS,
            seed=self.seed,
            size=n_sensitive,
            return_residual=True,
        )

        score = CategoricalCAP.compute(
            real_data=X_gt.data[self.CATEGORICAL_COLS].astype(int),
            synthetic_data=X_syn.data[self.CATEGORICAL_COLS].astype(int),
            key_fields=list(known_fields),
            sensitive_fields=list(sensitive_fields),
        )
        return {"score": score}
