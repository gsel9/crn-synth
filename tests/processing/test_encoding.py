import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder

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


def test_default_encoder(data):
    encoder = encoding.DefaultEncoder(categorical_columns=None, numerical_columns=None)
    data_enc, encoder = encoding.encode_data(data, encoder, refit=False)

    assert len(encoder.transformers) == 2

    # check if correct column categories are inferred
    assert encoder.transformers[0][2] == [
        "binary_cat_column",
        "cat_column",
    ], "Categorical columns are not inferred correctly"
    assert encoder.transformers[1][2] == [
        "binary_int_column",
        "float_column",
        "int_column",
    ], "Numerical columns are not inferred correctly"


def test_encode_data(data):
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    data_enc, encoder = encoding.encode_data(data, encoder, refit=False)

    # check is fitted
    assert encoding.check_is_fitted(encoder) is None, "Encoder is not fitted"

    # alter some values in data
    data_new = data.copy()
    data_new.loc[0, "binary_cat_column"] = "c"
    data_new.loc[1, "cat_column"] = "d"

    # transform using fitted encoder
    data_enc_new, _ = encoding.encode_data(data_new, encoder)
    assert (
        data_enc_new.loc[0, "binary_cat_column"] == -1
    ), "Unknown value is not encoded"
    assert data_enc_new.loc[1, "cat_column"] == -1, "Unknown value is not encoded"

    # check if the encoded data is different
    assert not data_enc.equals(data_enc_new), "Encoded data is not different"
