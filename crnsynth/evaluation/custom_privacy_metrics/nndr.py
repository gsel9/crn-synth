import platform
from typing import Any, Dict, List

import numpy as np
import torch
from pydantic import validate_arguments
from synthcity.metrics.eval_privacy import PrivacyEvaluator
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.utils.serialization import load_from_file, save_to_file

from crnsynth.evaluation.custom_privacy_metrics.utils import compute_distance_nn


class NearestNeighborDistanceRatio(PrivacyEvaluator):
    """
    Ratio of the distance between synthetic data's record to the closest and second closest real record.
    Allows comparing outliers and inliers in the population.

    Privacy risk: NNDR close to 0, where synthetic data points are close to real data points in sparse areas of the data space (outliers).
    Compare to holdout to determine an acceptable level. NNDR of synthetic data should be equal or higher than the NNDR of the
    holdout test set to the training data.
    """

    CATEGORICAL_COLS = None
    FRAC_SENSITIVE = None

    def __init__(self, seed=42, quantile=0.05, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)
        """
        Args:
            seed (int): Seed for random number generator.
            percentile (int): Percentile of distances to closest real record to take.
        """
        self.seed = seed
        self.quantile = quantile

    @property
    def n_categorical(self):
        return int(len(self.CATEGORICAL_COLS))

    @staticmethod
    def name() -> str:
        return "nearest_neighbor_distance_ratio"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @classmethod
    def update_cls_params(cls, params):
        for name, value in params.items():
            setattr(cls, name, value)

    # NOTE: needed to adapt cache_file name to include hash of test data
    # otherwise changing the test data won't have effect
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
        *args: Any,
        **kwargs: Any,
    ) -> Dict:
        X_train, X_test = X_gt.train(), X_gt.test()

        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{self.quantile}_{X_train.hash()}_{X_syn.hash()}_{X_test.hash()}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            return load_from_file(cache_file)

        results = self._evaluate(
            X_train=X_train, X_test=X_test, X_syn=X_syn, *args, **kwargs
        )
        save_to_file(cache_file, results)
        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self, X_train: DataLoader, X_test: DataLoader, X_syn: DataLoader
    ) -> Dict:
        distances_test, distances_synth = compute_distance_nn(
            df_train=X_train.data,
            df_test=X_test.data,
            df_synth=X_syn.data,
        )

        # get the ratio of closest real record by the distance to the second closest real record
        # and take the quantile of that ratio
        nndr_gt = np.quantile(
            distances_test[:, 0] / np.maximum(distances_test[:, 1], 1e-8),
            self.quantile,
        )
        nndr_synth = np.quantile(
            distances_synth[:, 0] / np.maximum(distances_synth[:, 1], 1e-8),
            self.quantile,
        )
        return {"gt": nndr_gt, "syn": nndr_synth}
