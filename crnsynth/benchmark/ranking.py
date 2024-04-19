"""Ranking of generators based on performance scores."""
import typing

import numpy as np
import pandas as pd


# NOTE: some metrics have zero as the optimal score (ie not direction)
def generator_rankings(
    score_reports: pd.DataFrame,
    weights: typing.Dict[str, np.ndarray] = None,
    normalize: typing.Optional[bool] = True,
    statistic: str = "mean",
) -> pd.DataFrame:
    """Rank generators according to performance.

    Args:
        score_reports: Results from score_reports.
        statistic: Defaults to using the mean performance scores.
        weights: Metric weights for each score report.
        normalize: TODO. Defaults to doing normalization.

    Returns:
        Ranking scores per generator (lower is better).
    """

    scores = {}

    for name in score_reports.index.unique():
        scores[name] = score_reports.loc[name, statistic].values

        if weights is not None:
            scores[name] *= weights[name]

    scores["direction"] = score_reports.loc[name, "direction"].values
    scores = pd.DataFrame(scores, index=score_reports.loc[name, "measure"])

    to_min = scores[scores["direction"] == "minimize"].drop(["direction"], axis=1)
    to_max = scores[scores["direction"] == "maximize"].drop(["direction"], axis=1)

    # add 1 to adjust to python count logic
    min_rankings = pd.DataFrame(
        np.argsort(to_min.values) + 1, index=to_min.index, columns=to_min.columns
    )
    max_rankings = pd.DataFrame(
        np.argsort(to_max.values)[:, ::-1] + 1,
        index=to_max.index,
        columns=to_max.columns,
    )

    total_rankings = pd.concat([min_rankings, max_rankings]).sum(axis=0)

    if normalize:
        return (total_rankings / total_rankings.sum()).sort_values()

    return total_rankings.sort_values()


def generator_rankings_by_group(
    score_reports, grouper, weights_by_group=None, normalize=False
):
    rankings = {}

    for name in score_reports[grouper].unique():
        weights = weights_by_group[name] if weights_by_group is not None else None

        grp_rankings = generator_rankings(
            score_reports[score_reports[grouper] == name],
            weights=weights,
            normalize=normalize,
        )
        rankings[name] = grp_rankings

    return pd.DataFrame(rankings)
