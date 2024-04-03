"""This metric describes how difficult it is for an attacker to correctly guess the sensitive information using an algorithm called Correct Attribution Probability (CAP)"""
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from sdmetrics.single_table import CategoricalCAP
from sklearn.preprocessing import OrdinalEncoder

from crnsynth.metrics.base_metric import BaseMetric
from crnsynth.processing import encoding


class CategoricalCAPScore(BaseMetric):
    """Categorical Correct Attribution Probability (CAP) score metric. This metric describes how difficult it is for an
    attacker to correctly guess the sensitive information based on a synthetic dataset and a fraction of the variables
    in the original dataset.
    """

    def __init__(
        self,
        categorical_columns=None,
        frac_sensitive=None,
        random_state=None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.categorical_columns = categorical_columns
        self.frac_sensitive = frac_sensitive
        self.random_state = random_state

    @staticmethod
    def name() -> str:
        return "cap_categorical_score"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def compute(
        self,
        data_train: pd.DataFrame,
        data_synth: pd.DataFrame,
        data_holdout: Union[pd.DataFrame, None] = None,
    ) -> Dict:
        self._check_params()

        # select sensitive and known columns
        n_sensitive = int(len(self.categorical_columns) * self.frac_sensitive)
        sensitive_columns = list(
            np.random.choice(self.categorical_columns, size=n_sensitive, replace=False)
        )
        known_columns = [
            col for col in self.categorical_columns if col not in sensitive_columns
        ]

        # encode categorical columns as integers and retain dataframe for CAP score
        ordinal_enc = OrdinalEncoder(handle_unknown="error")
        data_train_enc, ordinal_enc = encoding.encode_data(
            data_train[self.categorical_columns], encoder=ordinal_enc
        )
        data_synth_enc, _ = encoding.encode_data(
            data_synth[self.categorical_columns], encoder=ordinal_enc
        )

        # compute CAP score
        score = CategoricalCAP.compute(
            real_data=data_train_enc,
            synthetic_data=data_synth_enc,
            key_fields=known_columns,
            sensitive_fields=sensitive_columns,
        )
        return {"score": score}

    def _check_params(self):
        if self.categorical_columns is None:
            raise ValueError("categorical_columns is required.")
        if not 0 < self.frac_sensitive < 1:
            raise ValueError("frac_sensitive must be between 0 and 1.")

        # set random state
        np.random.RandomState(self.random_state)
