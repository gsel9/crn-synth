import pandas as pd
import pytest

from crnsynth.metrics.utility.correlation import (
    CorrelationSimilarityScore,
    FeatureCorrelation,
)


@pytest.fixture
def data_train():
    return pd.DataFrame(
        {
            "column1": ["a", "a", "b", "c"],
            "column2": ["d", "d", "e", "f"],
            "column3": ["g", "h", "i", "j"],
            "column4": [1, 2, 3, 4],
            "column5": [5, 6, 7, 8],
        }
    )


@pytest.fixture
def data_synth():
    return pd.DataFrame(
        {
            "column1": ["a", "a", "a", "c"],
            "column2": ["d", "d", "d", "f"],
            "column3": ["g", "h", "h", "j"],
            "column4": [1, 2, 3, 4],
            "column5": [5, 5, 7, 8],
        }
    )


def test_feature_correlation_type():
    metric = FeatureCorrelation()
    assert metric.type() == "utility"


def test_feature_correlation_direction():
    metric = FeatureCorrelation()
    assert metric.direction() == "maximize"


def test_feature_correlation_compute_with_valid_data(data_train, data_synth):
    metric = FeatureCorrelation(numerical_columns=["column4", "column5"])
    result = metric.compute(data_train, data_synth)
    assert "score" in result
    assert isinstance(result["score"], float)


def test_correlation_similarity_score_type():
    metric = CorrelationSimilarityScore()
    assert metric.type() == "utility"


def test_correlation_similarity_score_direction():
    metric = CorrelationSimilarityScore()
    assert metric.direction() == "maximize"


def test_correlation_similarity_score_compute_with_valid_data(data_train, data_synth):
    metric = CorrelationSimilarityScore(numerical_columns=["column4", "column5"])
    result = metric.compute(data_train, data_synth)
    assert "score" in result
    assert isinstance(result["score"], float)
