# stdlib
from pathlib import Path
from typing import Dict, List, Optional, Union

# third party
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.metrics import Metrics
from synthcity.metrics.eval import standard_metrics
from synthcity.metrics.scores import ScoreEvaluator
from synthcity.plugins.core.dataloader import (
    DataLoader,
    GenericDataLoader,
    create_from_info,
)


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

        if not isinstance(X_gt, DataLoader):
            X_gt = GenericDataLoader(X_gt)
        if not isinstance(X_syn, DataLoader):
            X_syn = create_from_info(X_syn, X_gt.info())

        if metrics is None:
            metrics = Metrics.list()

        X_gt, encoders = X_gt.encode()
        X_syn, _ = X_syn.encode(encoders=encoders)

        if X_gt_aug:
            X_gt_aug, encoders = X_gt_aug.encode()  # encoders=encoders)
        if X_syn_aug:
            X_syn_aug, _ = X_syn_aug.encode(encoders=encoders)

        scores = ScoreEvaluator()

        eval_cnt = min(len(X_gt), len(X_syn))
        for metric in standard_metrics:
            if metric.type() not in metrics:
                continue
            if metric.name() not in metrics[metric.type()]:
                continue
            if "augmented" in metric.name():
                scores.queue(
                    metric(
                        reduction="mean",
                        n_histogram_bins=10,
                        task_type=task_type,
                        random_state=random_state,
                        workspace=workspace,
                        use_cache=use_cache,
                    ),
                    X_gt_aug.sample(eval_cnt),
                    X_syn_aug.sample(eval_cnt),
                )
            else:
                scores.queue(
                    metric(
                        reduction="mean",
                        n_histogram_bins=10,
                        task_type=task_type,
                        random_state=random_state,
                        workspace=workspace,
                        use_cache=use_cache,
                    ),
                    X_gt.sample(eval_cnt),
                    X_syn.sample(eval_cnt),
                )

        scores.compute()

        return scores.to_dataframe()

    @staticmethod
    def _type_checking(X_gt, X_syn):
        if X_gt.type() != X_syn.type():
            raise ValueError("Different dataloader types")
