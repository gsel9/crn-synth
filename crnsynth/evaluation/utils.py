from pathlib import Path

from crnsynth.evaluation import (
    CategoricalCAPScore,
    CategoricalKNNScore,
    ContingencySimilarityScore,
    CorrelationSimilarityScore,
    CoxBetaScore,
    MedianSurvivalScore,
    PredictedMedianSurvivalScore,
    SurvivalCurvesDistanceScore,
)


def update_measures_from_config(column_config):
    """Configure evaluation metrics."""

    ContingencySimilarityScore.update_cls_params(
        {"CATEGORICAL_COLS": column_config["categorical_cols"]}
    )
    CorrelationSimilarityScore.update_cls_params(
        {"NUMERICAL_COLS": column_config["numerical_cols"]}
    )

    MedianSurvivalScore.update_cls_params(
        {"DURATION_COL": column_config["duration"], "EVENT_COL": column_config["event"]}
    )
    SurvivalCurvesDistanceScore.update_cls_params(
        {"DURATION_COL": column_config["duration"], "EVENT_COL": column_config["event"]}
    )

    CoxBetaScore.update_cls_params(
        {
            "CLIP_VALUE": column_config["clip_value"],
            "FEATURE_COLS": column_config["feature_cols"],
            "DURATION_COL": column_config["duration"],
            "TARGET_COL": column_config["target"],
            "EVENT_COL": column_config["event"],
        }
    )
    PredictedMedianSurvivalScore.update_cls_params(
        {
            "CLIP_VALUE": column_config["clip_value"],
            "FEATURE_COLS": column_config["feature_cols"],
            "DURATION_COL": column_config["duration"],
            "TARGET_COL": column_config["target"],
            "EVENT_COL": column_config["event"],
        }
    )

    CategoricalCAPScore.update_cls_params(
        {
            "CATEGORICAL_COLS": column_config["categorical_cols"],
            "FRAC_SENSITIVE": column_config["frac_sensitive_cols"],
        }
    )

    CategoricalKNNScore.update_cls_params(
        {
            "CATEGORICAL_COLS": column_config["categorical_cols"],
            "FRAC_SENSITIVE": column_config["frac_sensitive_cols"],
        }
    )


def remove_dir(path_to_dir):
    """Removes a possibly non-empty directory.

    Args:
        datasets_real:
        datasets_fake:
        generator_names:
        metrics:

    Returns:

    """

    path_to_dir = Path(path_to_dir)

    if path_to_dir.is_dir():
        # empty the directory
        for child in path_to_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                remove_dir(child)

        path_to_dir.rmdir()
