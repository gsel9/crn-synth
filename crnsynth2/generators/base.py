import pickle
from pathlib import Path
from typing import Any, Union

import pandas as pd


class BaseGenerator:
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        """Return the name of the generator."""
        raise NotImplementedError("name() method not implemented")

    def fit(self, data_real, **kwargs: Any) -> None:
        raise NotImplementedError("fit() method not implemented")

    def generate(self, n_records: int) -> pd.DataFrame:
        raise NotImplementedError("generate() method not implemented")

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)
