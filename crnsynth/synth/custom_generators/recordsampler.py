"""Integrating custom generators in synthcity as a Plugin"""
from typing import Any, List

import pandas as pd
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthesis.synthesizers.marginal import MarginalSynthesizer


class RecordSampler(Plugin):
    """Record sampler integration in synthcity.

    Re-samples records from the original dataset. Useful to evaluate as a baseline
    for privacy metrics.

    CAUTION: do not release output when based on sensitive data, due to privacy risk."""

    def __init__(self, replace=True, **kwargs: Any) -> None:
        self.replace = True
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "recordsampler"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "RecordSampler":
        self.X = X.dataframe()
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any):
        def _sample(count: int) -> pd.DataFrame:
            if (self.replace is False) and (count > len(self.X)):
                raise ValueError(
                    "When replace is False, count must be less than or equal to the original dataset size."
                )
            return self.X.sample(count, replace=self.replace).reset_index(drop=True)

        return self._safe_generate(_sample, count, syn_schema)
