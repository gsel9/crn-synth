"""Integrating custom generators in synthcity as a Plugin"""
from typing import Any, List

import pandas as pd
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthesis.synthesizers.privbayes import PrivBayes


class PrivBayesDK(Plugin):
    """PrivBayes DK integration in synthcity."""

    def __init__(self, epsilon=1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.model = PrivBayes(epsilon=epsilon)

    @staticmethod
    def name() -> str:
        return "privbayes-dk"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "PrivBayesDK":
        self.model.fit(X.dataframe())
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any):
        return self._safe_generate(self.model.sample, count, syn_schema)
