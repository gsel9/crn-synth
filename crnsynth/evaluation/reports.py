"""Benchmark synthetic data generators"""
import os
from pathlib import Path

import pandas as pd
import torch  # import torch to avoid segmentation fault error in TSD
from synthcity.metrics.scores import ScoreEvaluator

# TODO: need to adapt dataloader to surv, generic etc
from synthcity.plugins.core.dataloader import GenericDataLoader

from crnsynth import config, util
from crnsynth.evaluation.measure_evaluation import CustomMetrics
from crnsynth.evaluation.utils import remove_dir

ALL_METRICS = {
    "sanity": [
        "data_mismatch",
        "common_rows_proportion",
        "nearest_syn_neighbor_distance",
        "close_values_probability",
        "distant_values_probability",
    ],
    "stats": [
        # "chi_squared_test",
        # "feature_corr",
        # "inv_kl_divergence",
        # "ks_test",
        # "max_mean_discrepancy",
        # "prdc",
        # "alpha_precision",
        # "survival_km_distance",
        # -*- custom metrics -*-
        "wasserstein_dist",
        "jensenshannon_dist",
        "nrwe_cox_beta_augmented_score",
        "nrwe_median_survival_augmented_score",
        "nrwe_predicted_median_survival_augmented_score",
        "nrwe_survival_curves_distance_augmented_score",
        "nrwe_predicted_cindex_augmented_score",
        "nrwe_predicted_cumulative_hazard_augmented_score",
    ],
    "performance": ["linear_model", "mlp", "xgb", "feat_rank_distance"],
    "detection": [
        "detection_xgb",
        "detection_mlp",
        "detection_gmm",
        "detection_linear",
    ],
    "privacy": [
        "delta-presence",
        # "k-anonymization",
        "k-map",
        # "distinct l-diversity",
        "identifiability_score",
        # "DomiasMIA_BNAF",
        # "DomiasMIA_KDE",
        # "DomiasMIA_prior",
    ],
}


# TODO: enable custom selection of score stats
def score_report(
    data_real,
    data_fake,
    metrics,
    data_real_aug=None,
    data_synth_aug=None,
    reduce="mean",
    cache_dir="./tmp",
):
    """Create score report for a single synthetic dataset when compared to real data."""

    metrics = {
        "stats": ALL_METRICS["stats"],
        "sanity": ALL_METRICS["sanity"],
        "privacy": ALL_METRICS["privacy"],
    }

    if data_real_aug is not None:
        X_gt_aug = GenericDataLoader(data_real_aug)

    if data_synth_aug is not None:
        X_syn_aug = GenericDataLoader(data_synth_aug)

    eval = CustomMetrics.evaluate(
        X_gt=GenericDataLoader(data_real),
        X_syn=GenericDataLoader(data_fake),
        X_gt_aug=X_gt_aug,
        X_syn_aug=X_syn_aug,
        metrics=metrics,
        workspace=Path(cache_dir),
    )
    # remove cache dir
    remove_dir(cache_dir)

    scores = eval[reduce].to_dict()

    errors = eval["errors"].to_dict()
    duration = eval["durations"].to_dict()
    direction = eval["direction"].to_dict()

    report = ScoreEvaluator()

    for key in scores:
        report.add(key, scores[key], errors[key], duration[key], direction[key])

    return report.to_dataframe()


def create_score_reports(
    results_dir, dataset_name, metrics=None, generator_names=None, save=True
):
    """Create score reports for comparing synthetic datasets."""
    # get all metrics if not specified
    metrics = metrics if metrics is not None else ALL_METRICS

    # load real data
    print(dataset_name)
    data_real = pd.read_csv(config.PATH_DATA[dataset_name], index_col=0)

    # get paths to synthetic data either through specified generator_names or all csv's within folder
    path_results_synth = util.get_path_output(results_dir, output_type="synthetic_data")
    if generator_names is not None:
        paths_synth_data = [
            path_results_synth / f"{results_dir}_{name}.csv" for name in generator_names
        ]
    else:
        paths_synth_data = [
            os.path.join(path_results_synth, f)
            for f in os.listdir(path_results_synth)
            if f.endswith(".csv")
        ]
        generator_names = [
            os.path.basename(f).replace(f"{results_dir}_", "").replace(".csv", "")
            for f in paths_synth_data
        ]
        print(
            f"Found {len(generator_names)} synthetic datasets for {results_dir}: {generator_names}"
        )

    # load synthetic data one by one and compute metrics
    all_reports = []
    for i, path_s_df in enumerate(paths_synth_data):
        data_fake = pd.read_csv(path_s_df, index_col=0)
        report = score_report(data_real, data_fake, metrics)

        # post-processing
        report["measure"] = report.index
        report.index = pd.Index(
            [generator_names[i]] * report.shape[0], name="generator_name"
        )

        # add to list of reports
        all_reports.append(report)

        # memory release
        del data_fake

    df_reports = pd.concat(all_reports)

    # save files
    if save:
        path_out = util.get_path_output(
            results_dir, output_type="reports", verbose=True
        )
        path_file = path_out / f"{results_dir}_score_reports.csv"
        df_reports.to_csv(path_file)

    return df_reports


if __name__ == "__main__":
    # example of create score_reports for adult-minimal dataset
    # subset_metrics = {"sanity": ALL_METRICS["sanity"]}
    subset_metrics = None
    df_reports = create_score_reports(
        dataset_name="adult-minimal", metrics=subset_metrics
    )
