import numpy as np 

from datetime import datetime
from matplotlib.dates import date2num, num2date 


def subsample_dataframe(data, n_rows, seed=42):

    np.random.seed(seed)
    to_keep = np.random.choice(range(data.shape[0]), replace=False, size=n_rows)

    return data.iloc[to_keep]


def format_date_data(datum, date_format="%Y-%m-%d"):
    return datetime.strptime(datum, date_format)


def date_to_numeric(datums, date_format="%Y-%m-%d"):
    return date2num(datums.apply(lambda row: format_date_data(row, date_format))).astype(int)
    

def numeric_to_date(num, date_format="%Y-%m-%d"):
    return num2date(num).strftime(date_format)