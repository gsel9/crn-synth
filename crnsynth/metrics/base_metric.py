"""Base metric class for CRN synthesis metrics."""
from typing import Union

import pandas as pd


class BaseMetric:
    """Base metric class for CRN synthesis metrics."""

    def __init__(self):
        """Initialize the metric."""

    @staticmethod
    def name(self) -> str:
        """Return the name of the metric."""
        raise NotImplementedError("name() method not implemented")

    @staticmethod
    def type() -> str:
        """Return the type of the metric."""
        raise NotImplementedError("type() method not implemented")

    @staticmethod
    def direction() -> str:
        """Return the direction of the metric."""
        raise NotImplementedError("direction() method not implemented")

    def compute(
        self,
        data_train: pd.DataFrame,
        data_synth: pd.DataFrame,
        data_holdout: Union[pd.DataFrame, None] = None,
    ) -> float:
        """Compute the metric."""
        raise NotImplementedError("compute() method not implemented")

    def __str__(self):
        return self.name()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"
