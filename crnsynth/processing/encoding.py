"""Encoding data for synthesis or metrics"""
from typing import Iterable, List, Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_default_encoder(
    data: pd.DataFrame,
    categorical_columns: Union[Iterable, None] = None,
    numerical_columns: Union[Iterable, None] = None,
):
    """Default encoder for preparing data for synthesis or metrics"""

    # derive column types from data if not specified
    if categorical_columns is None:
        categorical_columns = data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    if numerical_columns is None:
        numerical_columns = data.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()

    encoders = ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="if_binary"), categorical_columns),
            ("num", StandardScaler(), numerical_columns),
        ]
    )

    return encoders


def encode_data(data: pd.DataFrame, encoders: ColumnTransformer):
    """Encode data using a ColumnTransformer"""
    # fit the encoder to the data
    encoders.fit(data)

    # transform the data
    data_enc = encoders.transform(data)

    # convert the transformed data to a DataFrame
    data_enc = pd.DataFrame(
        data_enc, columns=encoders.get_feature_names_out(data.columns)
    )
    return data_enc, encoders
