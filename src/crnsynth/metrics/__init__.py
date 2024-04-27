from crnsynth.metrics.base import BaseMetric
from crnsynth.metrics.privacy.catcap import CategoricalCAPScore
from crnsynth.metrics.privacy.dcr import DistanceClosestRecord
from crnsynth.metrics.privacy.nndr import NearestNeighborDistanceRatio
from crnsynth.metrics.utility.contingency import ContingencySimilarityScore
from crnsynth.metrics.utility.correlation import (
    CorrelationSimilarityScore,
    FeatureCorrelation,
)

PRIVACY_METRICS = [
    DistanceClosestRecord(),
    NearestNeighborDistanceRatio(),
    CategoricalCAPScore(),
]

UTILITY_METRICS = [
    ContingencySimilarityScore(),
    FeatureCorrelation(),
    CorrelationSimilarityScore(),
]

ALL_METRICS = {"privacy": PRIVACY_METRICS, "utility": UTILITY_METRICS}
