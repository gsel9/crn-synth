"""Utility for custom privacy metrics."""
import gower
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder


def compute_distance_nn(
    df_train, df_test, df_synth, categorical_columns, metric="gower"
):
    """Compute distance to closest real record for each synthetic record.
    Normalize using holdout test data."""

    def gower_distance(x, y, categorical_columns):
        # set numeric columns to float - needed for gower distance
        num_cols = [c for c in x.columns if c not in categorical_columns]
        x[num_cols] = x[num_cols].astype("float")
        y[num_cols] = y[num_cols].astype("float")

        # boolean vector of indices of categorical columns
        cat_columns_bool_mask = np.array([c in categorical_columns for c in y.columns])

        # compute distance matrix
        dm = gower.gower_matrix(x, y, cat_features=cat_columns_bool_mask)

        dist_x = np.zeros((x.shape[0], 2))
        for i in range(x.shape[0]):
            # find top 2 distances from record in x to dataset y
            dist_x[i, :] = gower.gower_dist.smallest_indices(
                np.nan_to_num(dm[i], nan=1), 2
            )["values"]

        return dist_x

    def nn_distance(df_train, df_test, df_synth, metric):
        # fit nearest neighbors to training distance)
        knn = NearestNeighbors(
            n_neighbors=2, algorithm="brute", metric=metric, n_jobs=-1
        )
        knn.fit(df_train)

        # nearest-neighbor search for test and synthetic data
        dist_test, _ = knn.kneighbors(df_test)
        dist_synth, _ = knn.kneighbors(df_synth)
        return dist_test, dist_synth

    if metric == "gower":
        dist_test = gower_distance(df_test, df_train, categorical_columns)
        dist_synth = gower_distance(df_synth, df_train, categorical_columns)
    else:
        dist_test, dist_synth = nn_distance(df_train, df_test, df_synth, metric)

    dist_test = np.square(dist_test)
    dist_synth = np.square(dist_synth)
    return dist_test, dist_synth
