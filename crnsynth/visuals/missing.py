import matplotlib.pyplot as plt
import numpy as np


def plot_missing(
    reference, comparison, figsize=(12, 5), width=0.4, bar_shift=0.2, sort=True
):
    fig, axis = plt.subplots(1, 1, figsize=figsize)

    if sort:
        sort_idx = np.argsort(reference)
        reference = reference[sort_idx]
        comparison = comparison[sort_idx]

    axis.axhline(y=1.0, c="k", linestyle="--", alpha=0.5)

    axis.bar(np.arange(reference.size) - bar_shift, reference, width, label="reference")
    axis.bar(
        np.arange(reference.size) + bar_shift, comparison, width, label="comparison"
    )

    axis.set_ylabel("Fraction of missing values")
    axis.set_xlabel("Variable number")

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    fig.tight_layout()
