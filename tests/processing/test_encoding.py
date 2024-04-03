import pandas as pd
import pytest

from crnsynth.processing import encoding


# add fixture
@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "binary_cat_column": ["a", "b", "a", "a"],
            "cat_column": ["a", "b", "c", "b"],
            "binary_int_column": [0, 1, 0, 1],
            "float_column": [1.0, 2.0, 3.0, 2.0],
            "int_column": [1, 2, 3, 4],
        }
    )


def test_get_default_encoders(data):
    encoders = encoding.get_default_encoder(
        data, categorical_columns=None, numerical_columns=None
    )
    assert len(encoders.transformers) == 2

    # check if correct column categories are inferred
    assert encoders.transformers[0][2] == [
        "binary_cat_column",
        "cat_column",
    ], "Categorical columns are not inferred correctly"
    assert encoders.transformers[1][2] == [
        "binary_int_column",
        "float_column",
        "int_column",
    ], "Numerical columns are not inferred correctly"


def test_encode_data(data):
    encoders = encoding.get_default_encoder(
        data, categorical_columns=None, numerical_columns=None
    )
    data_enc, encoders = encoding.encode_data(data, encoders)

    # check if encoded data has correct number of columns
    assert data_enc.shape[1] == 7, "Encoded data has incorrect number of columns"

    # check if encoded data has correct column names
    assert data_enc.columns.tolist() == [
        "cat__binary_cat_column_b",
        "cat__cat_column_a",
        "cat__cat_column_b",
        "cat__cat_column_c",
        "num__binary_int_column",
        "num__float_column",
        "num__int_column",
    ], "Encoded data has incorrect column names"
