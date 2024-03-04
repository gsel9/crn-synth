import pickle
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd


class BaseGenerator:
    def __init__(self, random_state: Union[int, None] = None):
        self.random_state = random_state

        # set global random seed for both numpy and random
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def fit(self, data_real) -> None:
        raise NotImplementedError("fit() method not implemented")

    def generate(self, n_records: int) -> pd.DataFrame:
        raise NotImplementedError("generate() method not implemented")

    def set_params(self, **params):
        """Set parameters"""
        for key, value in params.items():
            # check if attribute exists
            if hasattr(self, key):
                setattr(self, key, value)

    def get_params(self):
        return self.__dict__

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"

    def __str__(self) -> str:
        return self.__repr__()

    def __copy__(self):
        return self.__class__(**self.__dict__)
