"""Run a suite of metrics on a synthetic dataset and review its performance."""
import copy
import random
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from crnsynth.checks.params import set_class_param
from crnsynth.metrics.base_metric import BaseMetric
from crnsynth.processing.encoding import encode_data
from crnsynth.processing.utils import flatten_dict


class SyntheticDataReview:
    """Run a suite of metrics on a synthetic dataset and review its performance."""

    def __init__(
        self,
        metrics: Union[List[BaseMetric], Dict[str, BaseMetric]],
        metric_kwargs: Dict[str, Any] = None,
        encoder=None,
        n_jobs: int = 1,
        random_state: int = None,
        verbose: int = 1,
    ):
        """
        Args:
            metrics List[BaseMetric], Dict[str, BaseMetric]: List or Dict of metrics to run.
            metric_kwargs (Dict[str, Any], optional): Dictionary of keyword arguments for the metrics. Defaults to None.
            encoder (Any, optional): Use one type of encoding for all metrics. If None, use encoding method specified in
                                     each metric. Defaults to None.
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        self.metrics = metrics
        self.metric_kwargs = metric_kwargs
        self.encoder = encoder
        self.n_jobs = n_jobs
        self.verbose = verbose

        # results of the metrics after running .compute()
        self.scores_ = {}

    def compute(
        self,
        data_train: pd.DataFrame,
        data_synth: pd.DataFrame,
        data_holdout: Union[pd.DataFrame, None] = None,
    ) -> Dict:
        """Run all metrics on the synthetic data.

        Args:
            data_train (pd.DataFrame): Training data.
            data_synth (pd.DataFrame): Synthetic data.
            data_holdout (Union[pd.DataFrame, None], optional): Holdout data for evaluation. Defaults to None.

        Returns:
            Dict: Dictionary of metric results.
        """
        # use one encoding scheme for all metrics, only apply once to save computation time
        if self.encoder is not None:
            data_train, data_synth, data_holdout = self._encode(
                data_train, data_synth, data_holdout
            )

        # update parameters for all metrics
        self._set_metric_kwargs()

        if self.n_jobs == 1:
            self._compute_sequential(data_train, data_synth, data_holdout)
        else:
            self._compute_parallel(data_train, data_synth, data_holdout)

        return self.scores_

    def score_as_dataframe(self, name: str):
        """Convert scores to a DataFrame."""
        scores_flat = flatten_dict(self.scores_)
        df = pd.DataFrame(scores_flat, index=[name]).T
        return df

    def _encode(self, data_train, data_synth, data_holdout=None):
        """Use one encoding scheme for all metrics, only apply once to save computation time"""
        # fit and transform training data
        data_train, self.encoder = encode_data(
            data_train, encoder=self.encoder, refit=True, return_df=True
        )

        # transform synthetic data using the encoder fitted on the training data
        data_synth, _ = encode_data(
            data_synth, encoder=self.encoder, refit=False, return_df=True
        )

        # transform holdout data using the encoder fitted on the training data
        if data_holdout is not None:
            data_holdout, _ = encode_data(
                data_holdout, encoder=self.encoder, refit=False, return_df=True
            )

        # add encoder = None to metric_kwargs to avoid redundant encoding in each metric
        if self.metric_kwargs is None:
            self.metric_kwargs = {}
        self.metric_kwargs["encoder"] = None

        return data_train, data_synth, data_holdout

    def _compute_sequential(self, data_train, data_synth, data_holdout):
        """Compute metrics sequentially."""
        # compute list of metrics
        if isinstance(self.metrics, list):
            for metric in self.metrics:
                if self.verbose:
                    print(f"Running metric {metric.name}")
                self.scores_[metric.name] = metric.compute(
                    data_train, data_synth, data_holdout
                )
        # compute dictionary of metrics where metrics are separated by category
        elif isinstance(self.metrics, dict):
            for metric_category, metric_list in self.metrics.items():
                self.scores_[metric_category] = {}
                for metric in metric_list:
                    if self.verbose:
                        print(f"Running metric {metric.name}")
                    self.scores_[metric_category][metric.name] = metric.compute(
                        data_train, data_synth, data_holdout
                    )
        else:
            raise ValueError("metrics must be a list or dictionary of metrics.")

    def _compute_parallel(self, data_train, data_synth, data_holdout):
        """Compute metrics in parallel."""
        # compute list of metrics
        if isinstance(self.metrics, list):
            self.scores_ = {
                metric.name: result
                for metric, result in zip(
                    self.metrics,
                    Parallel(n_jobs=self.n_jobs)(
                        delayed(metric.compute)(data_train, data_synth, data_holdout)
                        for metric in self.metrics
                    ),
                )
            }
        # compute dictionary of metrics where metrics are separated by category
        elif isinstance(self.metrics, dict):
            # Flatten the dictionary into a list and keep track of categories
            metrics_list = [
                (metric_category, metric)
                for metric_category, metric_list in self.metrics.items()
                for metric in metric_list
            ]

            # Compute all metrics in parallel
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(metric.compute)(data_train, data_synth, data_holdout)
                for _, metric in metrics_list
            )

            # Reconstruct the dictionary structure in the results
            self.scores_ = {}
            for (metric_category, metric), result in zip(metrics_list, results):
                if metric_category not in self.scores_:
                    self.scores_[metric_category] = {}
                self.scores_[metric_category][metric.name] = result
        else:
            raise ValueError("metrics must be a list or dictionary of metrics.")

    def _set_metric_kwargs(self):
        """Set metric parameter value for all metrics."""
        # nothing to set
        if self.metric_kwargs is None:
            return

        # create a deep copy of the metrics to avoid modifying the reference to the original metric parameters
        self.metrics = copy.deepcopy(self.metrics)

        for param_name, param_value in self.metric_kwargs.items():
            # set parameter value in metric class has the parameter
            if isinstance(self.metrics, list):
                for metric in self.metrics:
                    # set parameter value in metric class if it has the parameter
                    set_class_param(
                        metric, param_name, param_value, check_has_param=True
                    )
            elif isinstance(self.metrics, dict):
                for metric_category, metric_list in self.metrics.items():
                    for metric in metric_list:
                        # set parameter value in metric class if it has the parameter
                        set_class_param(
                            metric, param_name, param_value, check_has_param=True
                        )
