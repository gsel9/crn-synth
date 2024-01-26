"""Plotting functions"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from synthesis.evaluation.efficacy import (
    ClassifierComparison,
    FeatureImportanceComparison,
    KaplanMeierComparison,
)
from synthesis.evaluation.metrics import AssociationsComparison, MarginalComparison

from crnsynth.process import postprocess


def plot_heatmap(
    data_matrix,
    figsize=(8, 5),
    xlabel="",
    ylabel="",
    xticklabels="",
    yticklabels="",
    fig_title=None,
):
    fig, axis = plt.subplots(figsize=figsize)

    sns.heatmap(
        data_matrix,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        vmin=np.min(data_matrix),
        vmax=np.max(data_matrix),
        robust=True,
        annot=True,
        square=True,
        linewidth=0.5,
        cbar=False,
        annot_kws={"size": 13},
        fmt=".3f",
        ax=axis,
        cmap="cividis",
    )

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    # if fig_title is not None:
    #    fig.suptitle(fig_title, fontsize=15, y=0.5)

    # plt.tight_layout()


def plot_grouped_barplot(
    data,
    value_key,
    name_key,
    group_key,
    fig_title=None,
    figsize=(8, 5),
    group_width=1,
    bar_width=0.2,
):
    fig, axis = plt.subplots(figsize=figsize)

    groups = data[group_key].unique()
    shifts = np.linspace(
        -1 / (groups.size + group_width), 1 / (groups.size + group_width), groups.size
    )

    x_coords = np.arange(data[name_key].nunique())
    colors = sns.color_palette(palette="pastel", n_colors=groups.size)

    for i, group in enumerate(groups):
        axis.bar(
            x_coords + shifts[i],
            data[data[group_key] == group][value_key].values,
            width=bar_width,
            label=group,
            color=colors[i],
        )

    axis.set_ylabel("Ranking score (smaller is better)")
    axis.set_xlabel("Measure category")

    axis.set_xticks(x_coords)
    axis.set_xticklabels(data[name_key].unique())

    # handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, groups)]
    fig.legend(
        loc="center left", bbox_to_anchor=(0.9, 0.5), ncol=1, fancybox=True, shadow=True
    )

    if fig_title is not None:
        axis.set_title(fig_title)

    return fig, axis


def plot_partial_effects_comparison(
    cox, cox_ref, covariate, values, y="survival_function", figsize=(14, 5)
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    cox.plot_partial_effects_on_outcome(
        covariates=covariate,
        values=values,
        plot_baseline=False,
        y=y,
        ax=axes[0],
        title="estimate",
    )

    cox_ref.plot_partial_effects_on_outcome(
        covariates=covariate,
        values=values,
        plot_baseline=False,
        y=y,
        ax=axes[1],
        title="reference",
    )

    axes[0].set_ylabel(y)
    axes[0].set_xlabel(covariate)
    axes[1].set_xlabel(covariate)

    return fig, axes


def plot_coef_comparison(
    params, confidence, ref_params, ref_confidence, param_labels, figsize=(10, 10)
):
    y = np.arange(ref_params.size)

    fig, axis = plt.subplots(1, 1, figsize=figsize)

    axis.errorbar(
        ref_params,
        y + 0.15,
        xerr=ref_confidence,
        capsize=4,
        marker="o",
        linestyle="",
        color="k",
        alpha=0.8,
        label="reference",
    )
    axis.errorbar(
        params,
        y - 0.15,
        xerr=confidence,
        capsize=4,
        marker="o",
        linestyle="",
        color="maroon",
        alpha=0.8,
        label="estimate",
    )

    axis.axvline(x=0, linestyle="--", c="k", alpha=0.5)
    axis.set_yticks(y)
    axis.set_yticklabels(param_labels)
    axis.set_xlabel("Coefficient value")
    axis.grid(True, alpha=0.4)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
        fancybox=True,
        shadow=True,
    )

    return fig, axis


def plot_density_comparison(data_original, data_synthetic, x, hue=None, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4), sharey=True)

    # ensure consisten hue order for both figures
    hue_order = np.sort(data_original[hue].unique())

    sns.histplot(
        data=data_original,
        x=x,
        ax=ax[0],
        stat="probability",
        hue=hue,
        hue_order=hue_order,
        kde=True,
        label="original",
    )
    sns.histplot(
        data=data_synthetic,
        x=x,
        ax=ax[1],
        stat="probability",
        hue=hue,
        kde=True,
        hue_order=hue_order,
        label="synthetic",
    )
    ax[0].set_title("original")
    ax[1].set_title("synthetic")
    if save_path:
        plt.savefig(save_path)
    return fig, ax


def plot_kaplan_meier_comparison(
    df_original, df_synth, time_column, event_column, group_column, save_path=None
):
    """Wrapper function to compare Kaplan Meier curves for original and synthetic data."""
    group_column = "PD-L1 result"
    km_comp = KaplanMeierComparison(
        time_column=time_column, event_column=event_column, group_column=group_column
    )
    km_comp.fit(df_original, df_synth)
    km_comp.plot()
    if save_path:
        plt.savefig(save_path)


def plot_marginal_comparison(
    df_original, df_synth, exclude_columns=None, save_path=None
):
    """Wrapper function to compare marginal distributions for original and synthetic data."""
    mc = MarginalComparison(exclude_columns=exclude_columns)
    mc.fit(df_original, df_synth)
    mc.plot()
    if save_path:
        plt.savefig(save_path)


def plot_association_comparison(df_original, df_synth, save_path=None):
    """Wrapper function to compare associations for original and synthetic data."""
    ac = AssociationsComparison()
    ac.fit(df_original, df_synth)
    ac.plot()
    if save_path:
        plt.savefig(save_path)


def plot_random_forest_comparison(
    df_original, df_synth, df_test, y_column, exclude_columns=None, save_path=None
):
    """Wrapper function to compare random forest ROC-AUC and PR-AUC scores for original and synthetic data."""
    # convert target to numerical
    df_original, df_synth, df_test = postprocess.encode_label_numeric(
        y_column, df_original, df_synth, df_test
    )

    rf_comp = ClassifierComparison(
        y_column=y_column, default_clf="rf", exclude_columns=exclude_columns
    )
    rf_comp.fit(df_original, df_synth)
    rf_comp.plot(df_test)
    plt.suptitle("Random Forest")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)


def plot_feature_importance_comparison(
    df_original, df_synth, y_column, top=30, exclude_columns=None, save_path=None
):
    """Wrapper function to compare feature importance of an Random Forest model for original and synthetic data."""
    # convert target to numerical
    df_original, df_synth, _ = postprocess.encode_label_numeric(
        y_column, df_original, df_synth
    )

    fi_comp = FeatureImportanceComparison(
        default_clf="rf", y_column=y_column, exclude_columns=exclude_columns
    )
    fi_comp.fit(df_original, df_synth)
    fi_comp.plot(None, top=top)

    if save_path:
        plt.savefig(save_path)


def plot_logistic_regression_comparison(
    df_original, df_synth, df_test, y_column, exclude_columns=None, save_path=None
):
    """Wrapper function to compare logistic regression ROC-AUC and PR-AUC scores for original and synthetic data."""
    # convert target to numerical
    df_original, df_synth, df_test = postprocess.encode_label_numeric(
        y_column, df_original, df_synth, df_test
    )

    lr_comp = ClassifierComparison(
        default_clf="lr", y_column=y_column, exclude_columns=exclude_columns
    )
    lr_comp.fit(df_original, df_synth)
    lr_comp.plot(df_test)
    plt.suptitle("Logistic Regression")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    return lr_comp


def plot_logreg_coef_comparison(
    df_original, df_synth, y_column, exclude_columns=None, top=30, save_path=None
):
    """Wrapper function to compare logistic regression coefficients for original and synthetic data."""
    # convert target to numerical
    df_original, df_synth, _ = postprocess.encode_label_numeric(
        y_column, df_original, df_synth
    )

    lr_coef_comp = FeatureImportanceComparison(
        default_clf="lr", y_column=y_column, exclude_columns=exclude_columns
    )
    lr_coef_comp.fit(df_original, df_synth)
    lr_coef_comp.plot(None, top=top)

    if save_path:
        plt.savefig(save_path)
    return lr_coef_comp


def plot_distances(
    distances_test, distances_synth, title, palette=None, quantile=0.5, save_path=None
):
    """Plot histogram of distances for test-train and test-synth. Useful for investigating DCR / NNDR privacy metrics."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.set_theme(style="white")
    sns.despine()

    if palette is None:
        palette = sns.color_palette()[:2]

    # plot histograms
    df_distances = pd.concat(
        [
            pd.DataFrame({"distance": distances_test.flatten(), "data": "test"}),
            pd.DataFrame({"distance": distances_synth.flatten(), "data": "synth"}),
        ]
    )
    sns.histplot(
        df_distances,
        ax=ax,
        x="distance",
        hue="data",
        kde=True,
        bins=50,
        palette=palette,
        stat="count",
        hue_order=["test", "synth"],
    )

    # plot lines for quantile
    quantile_test = np.quantile(distances_test, quantile)
    quantile_synth = np.quantile(distances_synth, quantile)

    ax.axvline(
        np.median(distances_test), color=palette[0], linestyle="dashed", linewidth=1.5
    )
    ax.text(
        quantile_test,
        0.9,
        f"{quantile} quantile test",
        color=palette[0],
        ha="right",
        va="top",
        rotation=90,
        transform=ax.get_xaxis_transform(),
    )
    ax.axvline(
        np.median(distances_synth), color=palette[1], linestyle="dashed", linewidth=1.5
    )
    ax.text(
        quantile_synth,
        0.9,
        f"{quantile} quantile synth",
        color=palette[1],
        ha="right",
        va="top",
        rotation=90,
        transform=ax.get_xaxis_transform(),
    )

    ax.set_title(title)
    ax.set_xlabel("distance")
    ax.set_ylabel("count")

    if save_path:
        plt.savefig(save_path)
