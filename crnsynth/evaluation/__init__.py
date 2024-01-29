from synthcity.metrics.eval import standard_metrics

from .custom_performance_metrics.target_classification import (
    LinearClassication,
    TreeClassication,
)
from .custom_privacy_metrics.catcap import CategoricalCAPScore
from .custom_privacy_metrics.catknn import CategoricalKNNScore
from .custom_privacy_metrics.dcr import DistanceClosestRecord
from .custom_privacy_metrics.nndr import NearestNeighborDistanceRatio
from .custom_similarity_metrics.contingency import ContingencySimilarityScore
from .custom_similarity_metrics.correlation import (
    CorrelationSimilarityScore,
    FeatureCorrelation,
)
from .custom_similarity_metrics.cox_beta_scores import CoxBetaScore
from .custom_similarity_metrics.median_survival_scores import (
    MedianSurvivalScore,
    PredictedMedianSurvivalScore,
)
from .custom_similarity_metrics.survival_curve_scores import SurvivalCurvesDistanceScore

__all__ = [
    "CategoricalCAPScore",
    "CategoricalKNNScore",
    "DistanceClosestRecord",
    "NearestNeighborDistanceRatio",
    "ContingencySimilarityScore",
    "CorrelationSimilarityScore",
    "CoxBetaScore",
    "MedianSurvivalScore",
    "PredictedMedianSurvivalScore",
    "SurvivalCurvesDistanceScore",
    "FeatureCorrelation",
    "RandomForestClassication",
    "LinearClassication",
]

# register custom metrics to synthcity
standard_metrics.append(CategoricalCAPScore)
standard_metrics.append(CategoricalKNNScore)
standard_metrics.append(DistanceClosestRecord)
standard_metrics.append(NearestNeighborDistanceRatio)
standard_metrics.append(CoxBetaScore)
standard_metrics.append(MedianSurvivalScore)
standard_metrics.append(PredictedMedianSurvivalScore)
standard_metrics.append(SurvivalCurvesDistanceScore)
standard_metrics.append(ContingencySimilarityScore)
standard_metrics.append(CorrelationSimilarityScore)
standard_metrics.append(FeatureCorrelation)
standard_metrics.append(TreeClassication)
standard_metrics.append(LinearClassication)
