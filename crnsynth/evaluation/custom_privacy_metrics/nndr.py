import platform
from typing import Any, Dict, List

import numpy as np
import torch
from pydantic import validate_arguments
from synthcity.metrics.eval_privacy import PrivacyEvaluator
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.utils.serialization import load_from_file, save_to_file

from crnsynth.evaluation.custom_privacy_metrics.utils import EPS, compute_distance_nn


def compute_ratio_distances(
    df_train,
    df_test,
    df_synth,
    categorical_columns,
    n_neighbors=5,
    distance_metric="gower",
):
    distances_test, distances_synth = compute_distance_nn(
        df_train=df_train,
        df_test=df_test,
        df_synth=df_synth,
        categorical_columns=categorical_columns,
        n_neighbors=n_neighbors,
        normalize=False,
        distance_metric=distance_metric,
    )

    # get the ratio of closest real record by the distance to the n_neighbors (e.g. 1st to 5th) closest real record
    # note add smoothing factor:
    # - to numerator in case of 0 distance (exact match) which otherwise would result in NNDR of 0 regardless of second closest distance
    # - to denominator to avoid division by 0
    ratio_distances_test = np.maximum(distances_test[:, 0], EPS) / np.maximum(
        distances_test[:, n_neighbors - 1], EPS
    )

    ratio_distances_synth = np.maximum(distances_synth[:, 0], EPS) / np.maximum(
        distances_synth[:, n_neighbors - 1], EPS
    )
    return ratio_distances_test, ratio_distances_synth


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
    EPS = 1e-8

    def __init__(
        self,
        seed=42,
        quantile=0.5,
        distance_metric="gower",
        n_neighbors=5,
        **kwargs: Any,
    ) -> None:
        super().__init__(default_metric="score", **kwargs)
        """
        Args:
            seed (int): Seed for random number generator.
            quantile (int): Quantile to take distances to closest real record to take.
            distance_metric (str): Distance metric to use for computing nearest neighbors.
            n_neighbors (int): Number of nearest neighbors to use for computing NNDR.
        """
        self.seed = seed
        self.quantile = quantile
        self.metric = distance_metric
        self.n_neighbors = n_neighbors

    @property
    def n_categorical(self):
        return int(len(self.CATEGORICAL_COLS))

    @staticmethod
    def type() -> str:
        return "privacy"

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
        categorical_cols = [col for col in X_train.data.columns if "cat" in col]
        # compute distance ratio
        ratio_distances_test, ratio_distances_synth = compute_ratio_distances(
            df_train=X_train.data,
            df_test=X_test.data,
            df_synth=X_syn.data,
            categorical_columns=categorical_cols,  # self.CATEGORICAL_COLS,
            n_neighbors=self.n_neighbors,
            distance_metric=self.metric,
        )

        # take the quantile of that rati
        nndr_gt = np.quantile(ratio_distances_test, self.quantile)
        nndr_synth = np.quantile(ratio_distances_synth, self.quantile)
        return {"gt": nndr_gt, "syn": nndr_synth}
