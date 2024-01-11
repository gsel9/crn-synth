"""Utility for custom privacy metrics."""
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

EPS = 1e-8


def compute_distance_nn(df_train, df_test, df_synth, quantile=0.05):
    """Compute distance to closest real record for each synthetic record.
    Normalize using holdout test data."""

    # fit nearest neighbors to training data
    knn = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="l2", n_jobs=-1)
    knn.fit(df_train)

    # nearest-neighbor search for test and synthetic data
    dist_test, _ = knn.kneighbors(df_test)
    dist_synth, _ = knn.kneighbors(df_synth)

    # normalize DCR using a quantile of test data
    # use smoothing factor to avoid division by zero
    dist_test = np.square(dist_test)
    dist_synth = np.square(dist_synth)
    bound = np.maximum(np.quantile(dist_test[~np.isnan(dist_test)], 0.95), EPS)
    norm_dist_test = np.where(dist_test <= bound, dist_test / bound, 1)
    norm_dist_synth = np.where(dist_synth <= bound, dist_synth / bound, 1)

    return norm_dist_test, norm_dist_synth
