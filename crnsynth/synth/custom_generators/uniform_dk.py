"""Integrating custom generators in synthcity as a Plugin"""
from typing import Any, List

import pandas as pd
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthesis.synthesizers.marginal import UniformSynthesizer


class UniformDK(Plugin):
    """Uniform DK integration in synthcity."""

    def __init__(self, epsilon=1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.model = UniformSynthesizer(epsilon=epsilon)

    @staticmethod
    def name() -> str:
        return "uniform-dk"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "UniformDK":
        self.model.fit(X.dataframe())
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any):
        return self._safe_generate(self.model.sample, count, syn_schema)
