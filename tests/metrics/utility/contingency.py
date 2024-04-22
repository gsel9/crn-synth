import pandas as pd
import pytest

from crnsynth.metrics.utility.contingency import ContingencySimilarityScore


@pytest.fixture
def data_train():
    return pd.DataFrame(
        {
            "column1": ["a", "a", "b", "c"],
            "column2": ["d", "d", "e", "f"],
            "column3": ["g", "h", "i", "j"],
            "column4": [1, 2, 3, 4],
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
        }
    )


def test_contingency_similarity_score_type():
    metric = ContingencySimilarityScore()
    assert metric.type() == "utility"


def test_contingency_similarity_score_direction():
    metric = ContingencySimilarityScore()
    assert metric.direction() == "maximize"


def test_contingency_similarity_score_compute_with_valid_data(data_train, data_synth):
    metric = ContingencySimilarityScore(
        categorical_columns=["column1", "column2", "column3"]
    )
    result = metric.compute(data_train, data_synth)
    assert "score" in result
    assert isinstance(result["score"], float)


def test_contingency_similarity_score_compute_with_no_categorical_columns(
    data_train, data_synth
):
    metric = ContingencySimilarityScore()

    # categorical columns need to be defined
    with pytest.raises(AssertionError):
        metric.compute(data_train, data_synth)
