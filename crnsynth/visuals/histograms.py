import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_histograms(data, columns, dropna=True, card_thresh=2):
    for column in columns:
        values = data[column].dropna() if dropna else data[column]

        if np.unique(values).size < card_thresh:
            continue

        plt.figure(figsize=(10, 6))
        sns.histplot(values)

        plt.xlabel(f"Histogram for {column}")
        plt.ylabel("Count")
        plt.show()


def plot_cumulative_histograms(
    data, columns, figsize=None, data_ref=None, dropna=False
):
    if figsize is None:
        figsize = (8, len(columns) * 4)

    fig, axes = plt.subplots(len(columns), 1, figsize=figsize)

    for i, axis in enumerate(axes.ravel()):
        col_data = data[columns[i]].dropna() if dropna else data[columns[i]]
        sorted_data = col_data.sort_values().values

        axis.step(
            sorted_data, np.arange(sorted_data.size), c="darkorange", label="data"
        )

        if data_ref is not None:
            sorted_data_ref = data_ref[columns[i]].sort_values().values
            axis.step(
                sorted_data_ref,
                np.arange(sorted_data_ref.size),
                c="k",
                label="reference data",
            )

        axis.set_xlabel(f"Values {columns[i]}")
        axis.set_ylabel("Cumulative frequency")
        axis.legend()

    fig.tight_layout()
