"""Compare multiple synthetic data generation methods."""
from pathlib import Path

import pandas as pd

from crnsynth.benchmark.review import SyntheticDataReview
from crnsynth.serialization.paths import create_output_dir
from crnsynth.serialization.save import save_csv, save_generator
from crnsynth.synthesization.synthesization import generate_synth_data


def benchmark_generators(
    data_real: pd.DataFrame,
    data_holdout: pd.DataFrame,
    generators: list,
    path_out: Path,
    reviewer: SyntheticDataReview = None,
    n_records=None,
    fname_param=None,
    verbose=1,
):
    """
    Benchmark multiple synthetic data generators.

    Args:
        data_real (pd.DataFrame): Real data to synthesize.
        data_holdout (pd.DataFrame): Holdout data for reviewer.
        generators (list): List of generators to benchmark.
        path_out (Path): Output path to save synthetic data and generators.
        reviewer (SyntheticDataReview): Reviewer to evaluate synthetic data.
        n_records (int): Number of records to generate.
        fname_param (str): Parameter to include in filename if present in generator.
        verbose (int): Verbosity level.

    Returns:
        pd.DataFrame: Scores from reviewer.
    """

    # Convert path_out to a Path object if it's a string
    path_out = Path(path_out) if isinstance(path_out, str) else path_out

    # create output directories
    create_output_dir(path_out)

    # prepare scores
    scores = []

    # create synthetic data with each generator
    for i, generator in enumerate(generators):
        if verbose:
            print(f"Running generator {generator}")
        data_synth, generator = generate_synth_data(
            data_real, generator, n_records=n_records, verbose=verbose
        )

        # create filename
        fname = f"{i}_{generator.name}"
        # add parameter value to filename
        if fname_param is not None and fname_param in generator.__dict__.keys():
            fname += f"_{fname_param}{generator.__dict__[fname_param]}"

        # save synthetic data and generator
        save_csv(data_synth, path_out / f"synthetic_data/{fname}.csv")
        generator.save(path_out / f"generators/{fname}.pkl")

        if verbose:
            print(f"Saved synthetic data and generator for {fname} at {path_out}")

        # run reviewer
        if reviewer:
            if verbose:
                print(f"Running reviewer for {fname}")

            reviewer.compute(data_real, data_synth, data_holdout)
            scores.append(reviewer.score_as_dataframe(fname).T)

    # convert reviews to single dataframe and save to disk
    if reviewer:
        df_scores = pd.concat(scores).T.rename_axis("metric")
        df_scores.to_csv(path_out / "reports/scores.csv")

        if verbose:
            print(f"Saved scores at {path_out / 'reports/scores.csv'}")
