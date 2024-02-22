"""Base metric class for CRN synthesis metrics."""
import pandas as pd


class BaseMetric:
    """Base metric class for CRN synthesis metrics."""

    def __init__(self):
        """Initialize the metric."""

    @property
    def name(self) -> str:
        """Return the name of the metric."""
        raise NotImplementedError("name() method not implemented")

    def compute(self, data_real: pd.DataFrame, data_synth: pd.DataFrame) -> float:
        """Compute the metric."""
        raise NotImplementedError("compute() method not implemented")

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
