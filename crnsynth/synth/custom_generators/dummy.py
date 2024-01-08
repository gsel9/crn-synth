"""Integrating custom generators in synthcity as a Plugin"""
from typing import Any, List

import pandas as pd
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthesis.synthesizers.marginal import MarginalSynthesizer


class DummySampler(Plugin):
    """Dummy sampler integration in synthcity.

    Makes a copy of the original dataset. Useful to evaluate as a baseline
    for privacy metrics.

    CAUTION: do not release output when based on sensitive data, due to privacy risk."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "dummy_sampler_custom"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "DummySampler":
        self.X = X.dataframe()
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any):
        def _sample(count: int) -> pd.DataFrame:
            if count != len(self.X):
                raise ValueError(
                    "DummySampler can only generate a copy of the original dataset, thus count must be equal to the original dataset size."
                )
            return self.X.copy()

        return self._safe_generate(_sample, count, syn_schema)
