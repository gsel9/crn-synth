import pandas as pd
import pytest

from crnsynth.benchmark.review import SyntheticDataReview
from crnsynth.metrics.privacy.dcr import DistanceClosestRecord
from crnsynth.metrics.privacy.nndr import NearestNeighborDistanceRatio


@pytest.fixture
def real_data():
    return pd.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10], "c": ["a", "b", "c", "d", "e"]}
    )


@pytest.fixture
def synthetic_data():
    # altered first three columns compared to real
    return pd.DataFrame(
        {"a": [2, 1, 5, 4, 5], "b": [7, 6, 6, 9, 10], "c": ["c", "a", "c", "d", "e"]}
    )


@pytest.fixture
def holdout_data():
    # altered first column only compared to real
    return pd.DataFrame(
        {"a": [6, 2, 3, 4, 5], "b": [5, 7, 8, 9, 10], "c": ["b", "b", "c", "d", "e"]}
    )


EXPECTED_DCR = {"holdout": 0.0, "synth": 0.03472222429182798}
EXPECTED_NNDR = {"holdout": 4e-08, "synth": 0.25}
EXPECTED_RESULTS_METRICS_LIST = {
    "DistanceClosestRecord": EXPECTED_DCR,
    "NearestNeighborDistanceRatio": EXPECTED_NNDR,
}
EXPECTED_RESULTS_METRICS_DICT = {"privacy": EXPECTED_RESULTS_METRICS_LIST}


def test_review_metrics_list_sequential(real_data, synthetic_data, holdout_data):
    """Test compute method with list of metrics and sequential computation"""
    # metrics as list
    metrics = [DistanceClosestRecord(), NearestNeighborDistanceRatio()]
    metric_kwargs = {
        "quantile": 0.5,
        "distance_metric": "gower",
        "n_neighbors": 2,
        "categorical_columns": ["c"],
    }

    # sequential computation
    n_jobs = 1

    # run benchmark
    benchmark = SyntheticDataReview(
        metrics=metrics, encoder="ordinal", n_jobs=n_jobs, metric_kwargs=metric_kwargs
    )
    results = benchmark.compute(real_data, synthetic_data, holdout_data)
    assert isinstance(results, dict), "Results should be a dictionary"
    assert "DistanceClosestRecord" in results, "DCR should be in results"
    assert "NearestNeighborDistanceRatio" in results, "NNDR should be in results"

    # check results
    assert isinstance(
        results["DistanceClosestRecord"], dict
    ), "DCR results should be a dictionary"
    assert isinstance(
        results["NearestNeighborDistanceRatio"], dict
    ), "NNDR results should be a dictionary"

    assert (
        results == EXPECTED_RESULTS_METRICS_LIST
    ), "Results should match expected results"


def test_review_metrics_dict_sequential(real_data, synthetic_data, holdout_data):
    """Test compute method with dictionary of metrics and sequential computation"""
    # metrics as dictionary
    metrics = {"privacy": [DistanceClosestRecord(), NearestNeighborDistanceRatio()]}
    metric_kwargs = {
        "quantile": 0.5,
        "distance_metric": "gower",
        "n_neighbors": 2,
        "categorical_columns": ["c"],
    }

    # sequential computation
    n_jobs = 1

    # run benchmark
    benchmark = SyntheticDataReview(
        metrics=metrics, encoder="ordinal", n_jobs=n_jobs, metric_kwargs=metric_kwargs
    )
    results = benchmark.compute(real_data, synthetic_data, holdout_data)
    assert isinstance(results, dict), "Results should be a dictionary"
    assert "privacy" in results, "Privacy category should be in results"
    assert isinstance(
        results["privacy"], dict
    ), "Privacy results should be a dictionary"
    assert (
        "DistanceClosestRecord" in results["privacy"]
    ), "DCR should be in privacy results"
    assert (
        "NearestNeighborDistanceRatio" in results["privacy"]
    ), "NNDR should be in privacy results"

    # check results
    assert (
        results == EXPECTED_RESULTS_METRICS_DICT
    ), "Results should match expected results"


def test_review_metrics_list_parallel(real_data, synthetic_data, holdout_data):
    """Test compute method with list of metrics and parallel computation"""
    # metrics as list
    metrics = [DistanceClosestRecord(), NearestNeighborDistanceRatio()]
    metric_kwargs = {
        "quantile": 0.5,
        "distance_metric": "gower",
        "n_neighbors": 2,
        "categorical_columns": ["c"],
    }

    # parallel computation
    n_jobs = -1

    # run benchmark
    benchmark = SyntheticDataReview(
        metrics=metrics, encoder="ordinal", n_jobs=n_jobs, metric_kwargs=metric_kwargs
    )
    results = benchmark.compute(real_data, synthetic_data, holdout_data)
    assert isinstance(results, dict), "Results should be a dictionary"
    assert "DistanceClosestRecord" in results, "DCR should be in results"
    assert "NearestNeighborDistanceRatio" in results, "NNDR should be in results"

    # check results
    assert isinstance(
        results["DistanceClosestRecord"], dict
    ), "DCR results should be a dictionary"
    assert isinstance(
        results["NearestNeighborDistanceRatio"], dict
    ), "NNDR results should be a dictionary"

    assert (
        results == EXPECTED_RESULTS_METRICS_LIST
    ), "Results should match expected results"


def test_review_metrics_dict_parallel(real_data, synthetic_data, holdout_data):
    """Test compute method with dictionary of metrics and parallel computation"""
    # metrics as dictionary
    metrics = {"privacy": [DistanceClosestRecord(), NearestNeighborDistanceRatio()]}
    metric_kwargs = {
        "quantile": 0.5,
        "distance_metric": "gower",
        "n_neighbors": 2,
        "categorical_columns": ["c"],
    }

    # parallel computation
    n_jobs = -1

    # run benchmark
    benchmark = SyntheticDataReview(
        metrics=metrics, encoder="ordinal", n_jobs=n_jobs, metric_kwargs=metric_kwargs
    )
    results = benchmark.compute(real_data, synthetic_data, holdout_data)
    assert isinstance(results, dict), "Results should be a dictionary"
    assert "privacy" in results, "Privacy category should be in results"
    assert isinstance(
        results["privacy"], dict
    ), "Privacy results should be a dictionary"
    assert (
        "DistanceClosestRecord" in results["privacy"]
    ), "DCR should be in privacy results"
    assert (
        "NearestNeighborDistanceRatio" in results["privacy"]
    ), "NNDR should be in privacy results"

    # check results
    assert (
        results == EXPECTED_RESULTS_METRICS_DICT
    ), "Results should match expected results"
