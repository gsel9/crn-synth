"""Encoding data for synthesis or metrics"""
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


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

    encoder = ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="if_binary"), categorical_columns),
            ("num", StandardScaler(), numerical_columns),
        ]
    )

    return encoder


def encode_data(data, encoder, refit: bool = False):
    """Encode data using a ColumnTransformer"""
    # make a copy of the data to prevent modifying the original
    data = data.copy()

    # fit the encoder to the data if refit is True or if it is not fitted yet
    if refit:
        encoder.fit(data)
    else:
        try:
            check_is_fitted(encoder)
        except NotFittedError:
            encoder.fit(data)

    # transform the data
    data_enc = encoder.transform(data)

    # convert the transformed data to a DataFrame
    data_enc = convert_encoded_data_to_dataframe(
        data_enc, encoder, column_names=data.columns
    )
    return data_enc, encoder


def convert_encoded_data_to_dataframe(data_enc, encoder, column_names=None):
    """Convert encoded data back to a dataframe"""
    # convert the encoded data to a DataFrame
    df_enc = pd.DataFrame(
        data_enc,
        columns=encoder.get_feature_names_out(column_names),
    )
    return df_enc
