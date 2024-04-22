from crnsynth.metrics.base_metric import BaseMetric
from crnsynth.metrics.privacy.catcap import CategoricalCAPScore
from crnsynth.metrics.privacy.dcr import DistanceClosestRecord
from crnsynth.metrics.privacy.nndr import NearestNeighborDistanceRatio

PRIVACY_METRICS = [
    DistanceClosestRecord(),
    NearestNeighborDistanceRatio(),
    CategoricalCAPScore(),
]

ALL_METRICS = {"privacy": PRIVACY_METRICS}
