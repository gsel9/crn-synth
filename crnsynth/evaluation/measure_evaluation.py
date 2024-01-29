# stdlib
from pathlib import Path
from typing import Dict, List, Optional, Union

# third party
import pandas as pd
from pydantic import validate_arguments
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# synthcity absolute
from synthcity.metrics import Metrics
from synthcity.metrics.eval import standard_metrics
from synthcity.metrics.scores import ScoreEvaluator
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader


def _get_dataset_for_metric(metric, X_gt, X_syn, X_gt_aug, X_syn_aug):
    # including data from the hold-out set
    if metric.type() in ["privacy", "performance"]:
        return X_gt, X_syn

    if metric.type() == "stats":
        # need both outcomes of target variable
        if "augmented" in metric.name():
            return X_gt_aug, X_syn_aug

        # TODO: this is only for `predicted_median_survival_score`
        # should be under performance
        if "predicted" in metric.name():
            return X_gt, X_syn

        # compare against actual data used for synthetization
        return X_gt.train(), X_syn

    # compare against actual data used for synthetization
    if metric.type() in ["sanity", "detection"]:
        return X_gt.train(), X_syn

    raise ValueError(f"Unknown metric type: {metric.type()}")


def tmp_custom_transform(data_loader, encoders=None):
    if encoders is None:
        numerical_cols = ["age"]  # , "os_42"]
        categorical_cols = [
            "sex",
            "treatment",
            "treatyear",
            "os_42_status",
            "ecog",
            "squamous",
        ]

        transformers = [
            ("cat", OneHotEncoder(drop="if_binary"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]

        encoders = ColumnTransformer(transformers)

    encoders.fit(data_loader.data)  # fit(data_loader.train().data)
    data_enc = encoders.transform(data_loader.data)
    data_enc = pd.DataFrame(
        data_enc, columns=encoders.get_feature_names_out(data_loader.data.columns)
    )
    data_enc["os_42"] = data_loader.data["os_42"]

    data_loader_enc = GenericDataLoader(data_enc)
    data_loader_enc.train_size = data_loader.train_size
    data_loader_enc.random_state = data_loader.random_state
    data_loader_enc.target_column = "cat__os_42_status_1"  # data_loader.target_column

    return data_loader_enc, encoders


class CustomMetrics(Metrics):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        X_gt: Union[DataLoader, pd.DataFrame],
        X_syn: Union[DataLoader, pd.DataFrame],
        X_gt_aug: Optional[Union[DataLoader, pd.DataFrame]] = None,
        X_syn_aug: Optional[Union[DataLoader, pd.DataFrame]] = None,
        metrics: Optional[Dict] = None,
        task_type: str = "classification",
        random_state: int = 0,
        workspace: Path = Path("workspace"),
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Core evaluation logic for the metrics

        X_gt: Dataloader or DataFrame
            Reference real data
        X_syn: Dataloader or DataFrame
            Synthetic data
        metrics: dict
            the dictionary of metrics to evaluate
            Full dictionary of metrics is:
            {
                'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
                'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
                'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
                'detection': ['detection_xgb', 'detection_mlp', 'detection_gmm', 'detection_linear'],
                'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score']
            }
        reduction: str
            The way to aggregate metrics across folds. Can be: 'mean', "min", or "max".
        n_histogram_bins: int
            The number of bins used in histogram calculation of a given metric. Defaults to 10.
        task_type: str
            The type of problem. Relevant for evaluating the downstream models with the correct metrics. Valid tasks are:  "classification", "regression", "survival_analysis", "time_series", "time_series_survival".
        random_state: int
            random seed
        workspace: Path
            The folder for caching intermediary results.
        use_cache: bool
            If the a metric has been previously run and is cached, it will be reused for the experiments. Defaults to True.
        """
        workspace.mkdir(parents=True, exist_ok=True)

        supported_tasks = [
            "classification",
            "regression",
            "survival_analysis",
            "time_series",
            "time_series_survival",
        ]
        if task_type not in supported_tasks:
            raise ValueError(
                f"Invalid task type {task_type}. Supported: {supported_tasks}"
            )

        # column encoding
        # X_gt, encoders = X_gt.encode()
        # X_syn, _ = X_syn.encode(encoders=encoders)

        # X_gt_aug, _ = X_gt_aug.encode()
        # X_syn_aug, _ = X_syn_aug.encode(encoders=encoders)

        scores = ScoreEvaluator()

        if metrics is None:
            metrics = Metrics.list()

        for metric in standard_metrics:
            if metric.type() not in metrics:
                continue
            if metric.name() not in metrics[metric.type()]:
                continue

            # select dataset depending on the metric
            data_real, data_synth = _get_dataset_for_metric(
                metric, X_gt, X_syn, X_gt_aug, X_syn_aug
            )

            # column encoding
            data_real, encoders = tmp_custom_transform(data_real)
            data_synth, _ = tmp_custom_transform(data_synth, encoders)
            # print(data_real.data.columns)
            # assert afs
            # import numpy as np
            # D = (data_real.data - data_synth.data).sum()
            # print("RIGHT AFTER ENCODE:", np.sum(D))
            # assert asf

            # re-sampling to equal number of datapoints
            # eval_cnt = min(len(data_real), len(data_synth))
            # data_real = data_real.sample(eval_cnt)
            # data_synth = data_synth.sample(eval_cnt)

            scores.queue(
                metric(
                    reduction="mean",
                    n_histogram_bins=10,
                    task_type=task_type,
                    random_state=random_state,
                    workspace=workspace,
                    use_cache=use_cache,
                ),
                data_real,
                data_synth,
            )

        scores.compute()
        return scores.to_dataframe()

    @staticmethod
    def _type_checking(X_gt, X_syn):
        if X_gt.type() != X_syn.type():
            raise ValueError("Different dataloader types")
