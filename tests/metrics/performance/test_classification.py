import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from crnsynth.metrics.performance.classification import ClassifierComparison


def classification_dataframe(add_categorical=False):
    """Create a classification test dataframe"""
    features, target = make_classification()
    df = pd.DataFrame(features)
    df.columns = [f"feature_{i}" for i in range(df.shape[1])]
    if add_categorical:
        df["cat_feature"] = np.random.choice(["A", "B", "C", "D"], df.shape[0])
    df["target"] = target
    return df


def test_compute_with_valid_data():
    data_train, data_synth, data_holdout = (
        classification_dataframe(),
        classification_dataframe(),
        classification_dataframe(),
    )

    classifier = ClassifierComparison(
        target_column="target", model=None, score_fn=accuracy_score
    )
    result = classifier.compute(data_train, data_synth, data_holdout)
    assert isinstance(result, dict)
    assert "real" in result
    assert "synth" in result


def test_compute_with_missing_holdout_data():
    data_train, data_synth = classification_dataframe(), classification_dataframe()
    classifier = ClassifierComparison(
        target_column="target", model=None, score_fn=accuracy_score
    )
    with pytest.raises(ValueError):
        classifier.compute(data_train, data_synth, None)


def test_compute_with_missing_target_column():
    data_train, data_synth, data_holdout = (
        classification_dataframe(),
        classification_dataframe(),
        classification_dataframe(),
    )
    classifier = ClassifierComparison(
        target_column="missing_target", model=None, score_fn=accuracy_score
    )
    with pytest.raises(ValueError):
        classifier.compute(data_train, data_synth, data_holdout)


def test_compute_with_default_model_and_score_fn():
    data_train, data_synth, data_holdout = (
        classification_dataframe(),
        classification_dataframe(),
        classification_dataframe(),
    )
    classifier = ClassifierComparison(target_column="target")
    result = classifier.compute(data_train, data_synth, data_holdout)
    assert isinstance(result, dict)
    assert "real" in result
    assert "synth" in result


def test_compute_categorical_data():
    data_train, data_synth, data_holdout = (
        classification_dataframe(add_categorical=True),
        classification_dataframe(add_categorical=True),
        classification_dataframe(add_categorical=True),
    )

    # transform categorical column
    encoder = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="if_binary"), ["cat_feature"])],
        remainder="passthrough",
    )

    # try different model and score function
    model = LogisticRegression(max_iter=500, random_state=42)
    score_fn = roc_auc_score

    # create classifier
    classifier = ClassifierComparison(
        target_column="target", model=model, score_fn=score_fn, encoder=encoder
    )

    # compute
    result = classifier.compute(data_train, data_synth, data_holdout)
    assert isinstance(result, dict)
    assert "real" in result
    assert "synth" in result
