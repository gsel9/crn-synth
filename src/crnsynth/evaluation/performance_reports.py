from pathlib import Path
 
from synthcity.metrics import Metrics 
from synthcity.metrics.scores import ScoreEvaluator  
from synthcity.plugins.core.dataloader import GenericDataLoader

from crnsynth.utils.experiment import remove_dir


PERFORMANCE_METRICS = {
    "sanity": [
        "data_mismatch",
        "common_rows_proportion",
        "nearest_syn_neighbor_distance",
        "close_values_probability",
        "distant_values_probability"
    ],
    "stats": [
        "jensenshannon_dist",
        "chi_squared_test",
        "feature_corr",
        "inv_kl_divergence",
        "ks_test",
        "max_mean_discrepancy",
        "wasserstein_dist",
        "prdc",
        "alpha_precision"
    ],
    "performance": [
        "linear_model", 
        "mlp", 
        "xgb", 
        "feat_rank_distance"
    ],
    "detection": [
        "detection_xgb",
        "detection_mlp",
        "detection_gmm",
        "detection_linear"
    ],
    "privacy": [
        "delta-presence",
        "k-anonymization",
        "k-map",
        "distinct l-diversity",
        "identifiability_score",
        #"DomiasMIA_BNAF",
        #"DomiasMIA_KDE",
        #"DomiasMIA_prior"
    ]
}


# TODO: enable custom selection of score stats
def create_score_report(data_real, data_fake, metrics=None, reduce="mean"):
    """ Create score report for a single synthetic dataset when compared 
    to real data.
    
    Args:
        data_real (_type_): _description_
        data_fake (_type_): _description_
        metrics (_type_, optional): _description_. Defaults to None.
        reduce (str, optional): _description_. Defaults to "mean".

    Returns:
        _type_: _description_
    """
    if metrics is None:
        metrics = PERFORMANCE_METRICS

    eval = Metrics.evaluate(
        X_gt=GenericDataLoader(data_real),
        X_syn=GenericDataLoader(data_fake),
        metrics=metrics,
        workspace=Path("./tmp"),
    )
    # remove cache dir
    remove_dir("./tmp")

    scores = eval[reduce].to_dict()

    errors = eval["errors"].to_dict()
    duration = eval["durations"].to_dict()
    direction = eval["direction"].to_dict()

    report = ScoreEvaluator()

    for key in scores:
        report.add(key, scores[key], errors[key], duration[key], direction[key])

    return report.to_dataframe()
