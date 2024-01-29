from typing import Any, Dict

import numpy as np
from pydantic import validate_arguments
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from synthcity.metrics.eval_performance import PerformanceEvaluator
from synthcity.plugins.core.dataloader import DataLoader


def train_synth_clf_real(
    model, X_test_gt, X_train_gt, y_train_gt, X_synth, y_test_synth
):
    # train on real and predict on real
    model.fit(X_train_gt, y_train_gt)
    y_real_pred = model.predict(X_test_gt)

    # train on synth and predict on real
    model.fit(X_synth, y_test_synth)
    y_synth_pred = model.predict(X_test_gt)

    return y_synth_pred, y_real_pred


class TreeClassication(PerformanceEvaluator):
    def __init__(self, seed=42, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)
        self.seed = seed

    @staticmethod
    def name() -> str:
        return "rf_classification_error"

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
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        X_train_gt, y_train_gt = X_gt.train().unpack()
        X_test_gt, y_test_gt = X_gt.test().unpack()
        X_synth, y_test_synth = X_syn.unpack()

        model = RandomForestClassifier(max_depth=3, n_estimators=50, random_state=42)

        y_synth_pred, y_real_pred = train_synth_clf_real(
            model, X_test_gt, X_train_gt, y_train_gt, X_synth, y_test_synth
        )

        score_real = roc_auc_score(y_test_gt, y_real_pred)
        score_synth = roc_auc_score(y_test_gt, y_synth_pred)

        return {"score": abs(score_real - score_synth)}


class LinearClassication(PerformanceEvaluator):
    def __init__(self, seed=42, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)
        self.seed = seed

    @staticmethod
    def name() -> str:
        return "linear_classification_error"

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
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        X_train_gt, y_train_gt = X_gt.train().unpack()
        X_test_gt, y_test_gt = X_gt.test().unpack()
        X_synth, y_test_synth = X_syn.unpack()

        model = LogisticRegression(max_iter=500, random_state=42)

        y_synth_pred, y_real_pred = train_synth_clf_real(
            model, X_test_gt, X_train_gt, y_train_gt, X_synth, y_test_synth
        )

        score_real = roc_auc_score(y_test_gt, y_real_pred)
        score_synth = roc_auc_score(y_test_gt, y_synth_pred)

        return {"score": abs(score_real - score_synth)}
