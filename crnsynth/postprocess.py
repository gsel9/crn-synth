"""Postprocessing to apply to generated synthetic data"""
import numpy as np
import pandas as pd
import pandas_flavor as pf
from sklearn.preprocessing import LabelEncoder


@pf.register_dataframe_method
def ensure_column_bound(
    df, column_name, column_bound_name, missing_value=None, random_state=None
):
    """Ensures that values in column_name never exceeds column_bound_name."""
    np.random.seed(random_state)
    df = df.copy()

    # replace column_name with 0 when column_bound_name== 0
    mask_zero_ond = (df[column_name] > df[column_bound_name]) & (
        df[column_bound_name] == 0
    )
    df.loc[mask_zero_ond, column_name] = 0

    if missing_value:
        # replace column_name with missing_value when column_bound_name = missing_value
        mask_missing = (df[column_bound_name] == missing_value) | (
            df[column_bound_name].isna()
        )
        df.loc[mask_missing, column_name] = missing_value

    # replace column_name > column_bound_name with sample(0, column_bound_name)
    mask_invalid_values = df[column_name] > df[column_bound_name]
    df.loc[mask_invalid_values, column_name] = np.random.randint(
        0, df[mask_invalid_values][column_bound_name] + 1
    )
    return df


@pf.register_dataframe_method
def skew_lowerbound_to_upperbound(
    df, column_lowerbound, column_upperbound, min_ratio=0.8, random_state=None
):
    """Skews numeric values in the lowerbound are close to the upperbound to become equal."""
    np.random.seed(random_state)
    df = df.copy()

    # get ratio of lowerbound to upperbound
    ratio = df[column_lowerbound] / df[column_upperbound]

    # get indices of rows where ratio exceeds min_ratio
    idx_skew = ratio > min_ratio

    # convert lowerbound to upperbound
    df.loc[idx_skew, column_lowerbound] = df.loc[idx_skew, column_upperbound]
    return df


def encode_label_numeric(y_column, df_original, df_synth, df_test=None):
    """Encode label column as numeric."""
    # print(y_column.shape, df_original.shape, df_synth.shape)
    # check if the y column is not numeric
    if not pd.api.types.is_numeric_dtype(df_original[y_column]):
        le = LabelEncoder()
        df_original[y_column] = le.fit_transform(df_original[y_column])
        df_synth[y_column] = le.transform(df_synth[y_column])

        if df_test is not None:
            df_test[y_column] = le.transform(df_test[y_column])
    return df_original, df_synth, df_test
