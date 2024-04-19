"""Check whether information in data is consistent with domain knowledge."""


def check_bound(df, column_lowerbound, column_upperbound):
    """Check whether the values in the column with lowerbound never exceed the value
    of the column with the upperbound.

    For example: Progression Free Survival < Overall Survival"""
    return all(df[column_lowerbound] <= df[column_upperbound])


def check_no_invalid_bool_combination(df, col1, col2, invalid_combo):
    """Check if there aren't any invalid bool combination"""
    return not any((df[col1] == invalid_combo[0]) & (df[col2] == invalid_combo[1]))
