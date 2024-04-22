import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_correlations_difference(data1, data2, vmax=0.5):
    sns.set_theme(style="white")

    corr1 = data1.corr()
    corr2 = data2.corr()

    mask = np.triu(np.ones_like(corr1, dtype=bool))

    _, axis = plt.subplots(figsize=(8, 5))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr1 - corr2,
        mask=mask,
        cmap=cmap,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=axis,
    )


def plot_correlations(data, vmax=0.5):
    sns.set_theme(style="white")

    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    _, axis = plt.subplots(figsize=(8, 5))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=axis,
    )
