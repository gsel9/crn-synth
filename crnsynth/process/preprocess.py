import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class EnsureConsistentType(BaseEstimator, TransformerMixin):
    """Ensure consistent type in dataset - used to avoid issues with mixed types in columns"""

    def __init__(self, dtype=str):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype(self.dtype)

    def get_feature_names_out(self, input_features=None):
        return input_features


class EncodeCategoricalColumns(BaseEstimator, TransformerMixin):
    """Performs standard processing operations on numeric and categorical columns"""

    def __init__(
        self,
        ordinal_cols=None,
        nominal_cols=None,
        ordinal_enc_kwargs={},
        nominal_enc_kwargs={},
    ):
        self.ordinal_cols = ordinal_cols
        self.nominal_cols = nominal_cols

        self.ordinal_enc_kwargs = ordinal_enc_kwargs
        self.nominal_enc_kwargs = nominal_enc_kwargs

    def fit(self, X, y=None):
        transformers = []

        if self.ordinal_cols is not None:
            # numerical encoding
            enc_ordinal = Pipeline(
                steps=[
                    ("dtype_encoding", EnsureConsistentType(dtype=str)),
                    ("ordinal_encoding", OrdinalEncoder(**self.ordinal_enc_kwargs)),
                ]
            )
            transformers.append(("ord", enc_ordinal, self.ordinal_cols))

        if self.nominal_cols is not None:
            # ohe encoding
            enc_nominal = Pipeline(
                steps=[
                    ("dtype_encoding", EnsureConsistentType(dtype=str)),
                    ("nominal_encoding", OneHotEncoder(**self.nominal_enc_kwargs)),
                ]
            )
            transformers.append(("nom", enc_nominal, self.nominal_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers)
        self.preprocessor.fit(X, y)

        return self

    def transform(self, X, y=None):
        return self.preprocessor.transform(X)

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.preprocessor.get_feature_names_out(input_features=input_features)


# TODO: add min-max scaling
def scale_numerical(data, columns, method="z-score"):
    if method == "z-score":
        encoder = StandardScaler()
        X = encoder.fit_transform(data[columns].values)

    else:
        raise ValueError(f"Invalid scaling method: {method}")

    return pd.DataFrame(X, columns=columns)


def encode_categorical(
    data, ordinal_cols, nominal_cols, ordinal_enc_kwargs, nominal_enc_kwargs
):
    encoder = EncodeCategoricalColumns(
        ordinal_cols=ordinal_cols,
        nominal_cols=nominal_cols,
        ordinal_enc_kwargs=ordinal_enc_kwargs,
        nominal_enc_kwargs=nominal_enc_kwargs,
    )
    X = encoder.fit_transform(data)

    return pd.DataFrame(X, columns=encoder.get_feature_names_out())


def basic_data_processing(
    data,
    numerical_cols=None,
    nominal_cols=None,
    ordinal_cols=None,
    ordinal_enc_kwargs={},
    nominal_enc_kwargs={},
):
    # encode categorical
    categorical_df = encode_categorical(
        data, ordinal_cols, nominal_cols, ordinal_enc_kwargs, nominal_enc_kwargs
    )

    # standardise numerical
    numerical_df = scale_numerical(data, numerical_cols)

    return pd.concat([numerical_df, categorical_df], axis=1)
