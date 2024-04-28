from collections import defaultdict

import numpy as np 
import pandas as pd 
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


def cardinality_report(data_real, data_fake, columns):
    """ Compare proportion of unique values per column. 

    Args:
        data_real (_type_): _description_
        data_fake (_type_): _description_
        columns (_type_): _description_

    Returns:
        _type_: _description_
    """
    report = defaultdict(list)

    for column in columns:

        unique_real = data_real[column].dropna().unique()
        unique_synth = data_fake[column].dropna().unique()

        report["column"].append(column)
        # number of unique values per column
        report["nunique_real"].append(len(unique_real))
        report["nunique_synth"].append(len(unique_synth))
        
        # if the set of synth values exceeds the set of real values 
        diff = np.setdiff1d(unique_synth, unique_real, assume_unique=True)
        report["has_invented_synth"] = diff.size > 0

    report = pd.DataFrame(report)    

    return report 


def nan_columns_report(data_real, data_fake):
    """ Compare proportion of missing values per column. 

    Args:
        data_real (_type_): _description_
        data_fake (_type_): _description_

    Returns:
        _type_: _description_
    """
    report = defaultdict(list)
    
    n_real = data_real.shape[0]
    n_fake = data_fake.shape[0]

    for column in data_real.columns:

        report["column"].append(column)
        report["real_nan"].append(data_real[column].isna().sum() / n_real)
        report["synth_nan"].append(data_fake[column].isna().sum() / n_fake)

    report = pd.DataFrame(report)    

    report["diff_real_synth"] = report["real_nan"] - report["synth_nan"]
    report["is_match_real_synth"] = (report["real_nan"] > 0) == (report["synth_nan"] > 0)

    return report 


def boundary_value_report(data_real, data_fake, dropna=False):
    """ Check if synthetic ata values exceed the value range in the 
    original data.

    Args:
        data_real (_type_): _description_
        data_fake (_type_): _description_
        dropna (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    report = defaultdict(list)

    for column in data_real.columns:

        col_real = data_real[column]
        col_fake = data_fake[column]

        if dropna:
            col_real = col_real.dropna() 
            col_fake = col_fake.dropna() 
            
        if is_string_dtype(col_real):
            _check_categorical_bounds(col_real, col_fake, report)
        
        elif is_numeric_dtype(col_real):
            _check_numerical_bounds(col_real, col_fake, report)
            
        else:
            # ignore this column
            continue

        report["column"].append(column)

    return pd.DataFrame(report)


def _check_categorical_bounds(data_real, data_fake, report):
    # boundaries in the real data 
    set_real = set(data_real)
    report["real_min"].append(list(set_real)[0])
    report["real_max"].append(list(set_real)[-1])

    # boundaries in the synth data 
    set_synth = set(data_fake)
    report["synth_min"].append(list(set_synth)[0])
    report["synth_max"].append(list(set_synth)[-1])

    # check if synth data is contained within the real data  
    test_result = set_synth.issubset(set_real)
    report["passed_check"].append(bool(test_result))


def _check_numerical_bounds(data_real, data_fake, report):
    # boundaries in the real data 
    report["real_min"].append(data_real.min())
    report["real_max"].append(data_real.max())

    # boundaries in the synth data 
    report["synth_min"].append(data_fake.min())
    report["synth_max"].append(data_fake.max())

    # check if synth data is contained within the real data  
    test_result = np.logical_and(report["synth_min"][-1] >=report["real_min"][-1],
                                 report["synth_max"][-1] <= report["real_max"][-1])

    report["passed_check"].append(bool(test_result))
