"""Checks to perform on data"""
from itertools import combinations

import numpy as np


def check_bound(df, column_lowerbound, column_upperbound):
    """Check whether the values in the column with lowerbound never exceed the value
    of the column with the upperbound.

    For example: Progression Free Survival < Overall Survival"""
    return all(df[column_lowerbound] <= df[column_upperbound])


def check_no_invalid_bool_combination(df, col1, col2, invalid_combo):
    """Check if there aren't any invalid bool combination"""
    return not any((df[col1] == invalid_combo[0]) & (df[col2] == invalid_combo[1]))


def check_rare_categories(df, column, min_support, verbose=1):
    """Check whether there are any rare categories in the column of the dataframe."""
    # normalize if floating number, use counts if int
    if isinstance(min_support, float):
        normalize = True
    else:
        normalize = False

    column_value_counts = df[column].value_counts(normalize=normalize, dropna=False)

    # if no rare categories under min support - no change required
    if not (column_value_counts < min_support).any():
        return None

    rare_categories = list(column_value_counts[column_value_counts < min_support].index)

    if verbose >= 1:
        msg = f"WARNING: Rare categories in column '{column}' that occur less than "
        msg += f"{min_support *100}%" if normalize else f"{min_support}x"
        msg += f" of the records: {rare_categories}"
        print(msg)
    return rare_categories


def check_rare_combinations(df, num_columns, min_support, warn=True, verbose=False):
    """Check whether there are any rare combinations of categories between columns of the dataframe."""
    # option to skip rare combination check
    if not warn:
        return

    assert num_columns >= 2, "num_columns must be greater than or equal to 2"

    # get all combinations of columns
    col_combos = []
    for i in range(2, num_columns + 1):
        col_combos.extend(list(combinations(df.columns, i)))

    # check for rare combinations
    rare_combinations = {}
    for combination in col_combos:
        # get counts of each combination
        counts = df.groupby(list(combination)).size()
        # get combination that occur less than min support of the records
        min_support = (
            min_support
            if isinstance(min_support, int)
            else int(min_support * df.shape[0])
        )
        # rare_combinations.extend(list(counts[counts < min_support].index))
        rare_combo = list(counts[counts < min_support].index)
        #  add to dict
        if rare_combo:
            rare_combinations[combination] = rare_combo

    if not rare_combinations:
        return None

    # print rare combinations
    if verbose:
        msg = "Rare combinations of categories between columns that occur less than "
        msg += f"{min_support}x of the records:"
        print(msg)
    return rare_combinations
