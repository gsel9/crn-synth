"""Utility for custom privacy metrics."""
import gower
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

SMOOTHING_FACTOR = 1e-8


def compute_distance_nn(
    data_train,
    data_synth,
    data_holdout,
    categorical_columns,
    normalize,
    n_neighbors,
    distance_metric="gower",
):
    """Compute distance to closest real record for each synthetic record.
    Normalize using holdout holdout data."""

    def gower_distance(x, y, categorical_columns, n_neighbors):
        # set numeric columns to float - needed for gower distance
        num_cols = [c for c in x.columns if c not in categorical_columns]
        x[num_cols] = x[num_cols].astype("float")
        y[num_cols] = y[num_cols].astype("float")

        # boolean vector of indices of categorical columns
        cat_columns_bool_mask = np.array([c in categorical_columns for c in y.columns])

        # compute distance matrix
        dm = gower.gower_matrix(x, y, cat_features=cat_columns_bool_mask)

        dist_x = np.zeros((x.shape[0], n_neighbors))
        for i in range(x.shape[0]):
            # find top 2 distances from record in x to dataset y
            dist_x[i, :] = gower.gower_dist.smallest_indices(
                np.nan_to_num(dm[i], nan=1), n_neighbors
            )["values"]

        return dist_x

    def nn_distance(data_train, data_synth, data_holdout, metric, n_neighbors):
        # fit nearest neighbors to training distance)
        knn = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm="brute", metric=metric, n_jobs=-1
        )
        knn.fit(data_train)

        # nearest-neighbor search for holdout and synthetic data
        dist_holdout, _ = knn.kneighbors(data_holdout)
        dist_synth, _ = knn.kneighbors(data_synth)
        return dist_holdout, dist_synth

    if distance_metric == "gower":
        dist_holdout = gower_distance(
            data_holdout, data_train, categorical_columns, n_neighbors=n_neighbors
        )
        dist_synth = gower_distance(
            data_synth, data_train, categorical_columns, n_neighbors=n_neighbors
        )
    else:
        dist_holdout, dist_synth = nn_distance(
            data_train,
            data_synth,
            data_holdout,
            distance_metric,
            n_neighbors=n_neighbors,
        )

    dist_holdout = np.square(dist_holdout)
    dist_synth = np.square(dist_synth)

    # normalize only needed for DCR
    if normalize:
        # normalize distances using 0.95 quantile of holdout data to avoid outliers
        # use smoothing factor to avoid division by zero
        bound = np.maximum(
            np.quantile(dist_holdout[~np.isnan(dist_holdout)], 0.95), SMOOTHING_FACTOR
        )
        dist_holdout = np.where(dist_holdout <= bound, dist_holdout / bound, 1)
        dist_synth = np.where(dist_synth <= bound, dist_synth / bound, 1)

    return dist_holdout, dist_synth
