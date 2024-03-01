from crnsynth2.metrics.base_metric import BaseMetric
from crnsynth2.metrics.privacy.dcr import DistanceClosestRecord
from crnsynth2.metrics.privacy.nndr import NearestNeighborDistanceRatio

PRIVACY_METRICS = [DistanceClosestRecord(), NearestNeighborDistanceRatio()]

DEFAULT_METRICS = {"privacy": PRIVACY_METRICS}
