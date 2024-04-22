"""Tests for the catcap metric."""

import numpy as np
import pandas as pd
import pytest

from crnsynth.metrics.privacy.catcap import CategoricalCAPScore


def test_catcap_initialization():
    """Test the initialization of the catcap metric."""
    catcap = CategoricalCAPScore()

    assert catcap.categorical_columns is None
    assert catcap.frac_sensitive == 0.5
    assert catcap.random_state is None

    # categorical_columns and frac_sensitive need to be set
    with pytest.raises(ValueError):
        catcap._check_params()


def test_catcap_compute():
    """Test the computation of the catcap metric."""

    # define datasets
    data_train = pd.DataFrame(
        {
            "a": ["a", "b", "c", "d", "e"],
            "b": ["a", "b", "c", "d", "e"],
            "c": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    data_synth = pd.DataFrame(
        {
            "a": ["a", "b", "b", "c", "e"],
            "b": ["a", "b", "a", "c", "b"],
            "c": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    data_holdout = None

    # set params
    categorical_columns = ["a", "b"]
    frac_sensitive = 0.5
    random_state = 42

    catcap = CategoricalCAPScore(
        categorical_columns=categorical_columns,
        frac_sensitive=frac_sensitive,
        random_state=random_state,
    )
    catcap_score = catcap.compute(data_train, data_synth, data_holdout)
    assert isinstance(catcap_score["score"], float)
