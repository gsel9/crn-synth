from typing import Any

import pandas as pd


class BaseGenerator:
    def __init__(self) -> None:
        pass

    def fit(self, data_real, **kwargs: Any) -> None:
        raise NotImplementedError("fit() method not implemented")

    def generate(self, n_records: int) -> pd.DataFrame:
        raise NotImplementedError("generate() method not implemented")
