"""Benchmark synthetic data generators"""
import os
from pathlib import Path

import pandas as pd
import torch  # import torch to avoid segmentation fault error in TSD
from synthcity.metrics.scores import ScoreEvaluator

# TODO: need to adapt dataloader to surv, generic etc
from synthcity.plugins.core.dataloader import GenericDataLoader, create_from_info

from crnsynth.configs import config
from crnsynth.evaluation.measure_evaluation import CustomMetrics
from crnsynth.evaluation.utils import remove_dir
from crnsynth.process import util

ALL_METRICS = {
    "sanity": [
        "data_mismatch",
        "common_rows_proportion",
        "nearest_syn_neighbor_distance",
        "close_values_probability",
        "distant_values_probability",
    ],
    "stats": [
        "chi_squared_test",
        "inv_kl_divergence",
        "wasserstein_dist",
        "jensenshannon_dist",
        "ks_test",
        "contingency_similarity_score",
        "correlation_similarity_score",
        # TODO: should be under performance metrics
        "predicted_median_survival_score",
        ###
        "survival_curves_distance_score",
        "cox_beta_augmented_score",
        "median_survival_score",
    ],
    "performance": [
        "linear_classification_error",
        "rf_classification_error" "linear_model",
        "xgb",
        "feat_rank_distance",
    ],
    "detection": [
        "detection_xgb",
        "detection_linear",
    ],
    "privacy": [
        # "delta-presence",
        # "k-anonymization",
        # "identifiability_score",
        "distance_closest_record",
        "nearest_neighbor_distance_ratio",
        "cap_categorical_score",
    ],
}


def score_report(
    data_real,
    data_fake,
    metrics,
    data_real_aug=None,
    data_synth_aug=None,
    target_column=None,
    sensitive_columns=None,
    reduce="mean",
    cache_dir="./tmp",
    random_state=42,
    debug=False,
):
    """Create score report for a single synthetic dataset when compared to real data."""

    metrics = {
        # "stats": ALL_METRICS["stats"],
        # "sanity": ALL_METRICS["sanity"],
        # "privacy": ALL_METRICS["privacy"],
        # "detection": ALL_METRICS["detection"],
        "performance": ALL_METRICS["performance"],
    }

    X_gt = GenericDataLoader(
        data_real, target_column=target_column, random_state=random_state
    )
    X_syn = create_from_info(data_fake, X_gt.info())
    X_syn.random_state = random_state

    X_gt_aug = GenericDataLoader(
        data_real_aug, target_column=target_column, random_state=random_state
    )

    X_syn_aug = create_from_info(data_synth_aug, X_gt_aug.info())
    X_syn_aug.random_state = random_state

    # import numpy as np
    # print((X_gt.train().encode()[0].data - X_syn.encode()[0].data).sum())
    # print((X_syn_aug.encode()[0].data - X_gt_aug.encode()[0].data).sum())

    eval = CustomMetrics.evaluate(
        X_gt=X_gt,
        X_syn=X_syn,
        X_gt_aug=X_gt_aug,
        X_syn_aug=X_syn_aug,
        metrics=metrics,
        workspace=Path(cache_dir),
    )
    # remove cache dir
    remove_dir(cache_dir)

    scores = eval[reduce].to_dict()
    if debug:
        print(scores)
        assert False

    errors = eval["errors"].to_dict()
    duration = eval["durations"].to_dict()
    direction = eval["direction"].to_dict()

    report = ScoreEvaluator()

    for key in scores:
        report.add(key, scores[key], errors[key], duration[key], direction[key])

    return report.to_dataframe()
