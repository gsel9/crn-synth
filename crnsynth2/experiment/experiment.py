"""Run multiple synthesis experiments and save the results to disk"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from joblib import Parallel, delayed

from crnsynth2.generators.base import BaseGenerator
from crnsynth2.metrics.base import BaseMetric
from crnsynth2.process.pipeline.synthpipe import BaseSynthPipe


def create_dir(path_to_dir, sub_dirs):
    """Ensure output directory for results exists."""

    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

    for subdir in sub_dirs:
        if not os.path.exists(path_to_dir / subdir):
            os.makedirs(path_to_dir / subdir)


class SynthExperiment:
    """Run a set of synthesis configurations"""

    def __init__(
        self,
        experiment_name: str,
        generators: list,
        metrics: list,
        synth_pipe: BaseSynthPipe,
        path_out: Union[str, Path],
        verbose=1,
    ):
        # TODO: add optional save output and optional metric computation
        # TODO: compute metrics over already generated datasets instead of generators
        self.experiment_name = experiment_name
        self.generators = generators
        self.metrics = metrics
        self.synth_pipe = synth_pipe
        self.path_out = path_out
        self.verbose = verbose

        # learned attributes
        self.config_ = {}
        self.scores_ = {}

    def run(
        self, data_real: pd.DataFrame, data_holdout: Union[pd.DataFrame, None] = None
    ):
        """Run the synthesis experiment"""
        # create output directory
        create_dir(self.path_out, sub_dirs=["synthetic_data", "generators"])

        # TODO: add option to run in parallel

        # iterate over generators
        for generator in self.generators:
            if self.verbose:
                print(f"Running synthesis experiment for {generator.name}")

            # run the synthesis experiment
            self.synth_pipe.set_generator(generator)
            data_synth = self.synth_pipe.run(data_real)

            # compute metrics
            self.scores_[generator.name] = {}
            for metric in self.metrics:
                self.scores_[generator.name][metric.name] = metric.compute(
                    data_real, data_synth, data_holdout
                )

            # save the synth data and generator
            fname = f"{self.experiment_name}_{generator.name}"
            data_synth.to_csv(
                Path(self.path_out) / "synthetic_data" / f"{fname}.csv", index=False
            )
            generator.save(Path(self.path_out) / "generators" / f"{fname}.pkl")

        # save scores
        df_scores = pd.DataFrame.from_dict(self.scores_, orient="index")
        df_scores.to_csv(
            Path(self.path_out) / f"{self.experiment_name}_scores.csv", index=False
        )

        # save experiment configuration
        self._save_config()

    def _save_config(self):
        """Save configuration and hyperparameter settings used to disk"""
        self.config_["experiment_name"] = self.experiment_name
        self.config_["synth_pipe"] = self.synth_pipe.__dict__
        self.config_["generators"] = {}
        for generator in self.generators:
            self.config_["generators"][generator.name] = generator.__dict__

        with open(
            Path(self.path_out) / f"{self.experiment_name}_config.json", "w"
        ) as outfile:
            json.dump(self.config_, outfile, indent=4)


class SynthExperimentConfig:
    """Run a synthesis experiment"""

    def __init__(self, generators, config_experiment, config_generator):
        self.generators = generators
        self.config_experiment = config_experiment
        self.config_generator = config_generator

    def run(self):
        """Run the synthesis experiment"""
        # check the configuration file and assign main parameters to the object
        self.check_config_experiment(self.config_experiment)

        # load the real data based on the configuration file
        data_real = self.load_data()
        print(data_real.head())

        # iterate over generators and experiment values
        for generator_name in self.generators:
            for experiment_values in self.param_grid:
                # update the generator configuration with the experiment values
                generator_params = self.config_generator["generator"][generator_name]
                generator_params.update(experiment_values)

                # run the synthesis experiment
                Parallel(n_jobs=self.n_jobs)(
                    delayed(self._run_experiment)(generator_name)
                    for generator_name in self.generators
                )

    def load_data(self):
        """Load the real data"""
        # check for optional kwargs in the config file
        kwargs = {}
        if "delimiter" in self.config_experiment["data"]:
            kwargs["delimiter"] = self.config_experiment["data"]["delimiter"]
        if "index_column" in self.config_experiment["data"]:
            kwargs["index_col"] = self.config_experiment["data"]["index_column"]

        # load data
        data_real = pd.read_csv(self.config_experiment["data"]["path"], **kwargs)

        if self.verbose:
            print("Loaded real data from:", self.config_experiment["data"]["path"])
        return data_real

    def _run_experiment(self, data_real, config_generator):
        if self.n_jobs == 1:
            self._run_single_job(data_real, config_generator)
        else:
            self._run_parallel(data_real, config_generator)

    def check_config_experiment(self, config_experiment):
        """Check the configuration file for the synthesis experiment"""

        def _check_keys(config, keys, main_key=None):
            """Check if all keys are present in the configuration file"""

            # subset based on main key, e.g. experiment, data, output
            if main_key:
                if main_key not in config:
                    raise ValueError(
                        f"Key '{main_key}' not found in the configuration file"
                    )
                config = config[main_key]

            for key in keys:
                if key not in config:
                    raise ValueError(f"Key '{key}' not found in the configuration file")

        # check if all required keys are present in the configuration file
        _check_keys(
            config_experiment, ["experiment_id", "param_grid"], main_key="experiment"
        )
        _check_keys(config_experiment, ["dataset_path"], main_key="data")
        _check_keys(
            config_experiment,
            ["save_score_report", "save_generator", "save_synthetic", "save_config"],
            main_key="output",
        )

        # check if param_grid is a dictionary
        if not isinstance(config_experiment["experiment"]["param_grid"], dict):
            raise ValueError("The experiment_values key must be a dictionary")

        # check if the data path exists
        assert os.path.exists(
            config_experiment["data"]["path"]
        ), f"The path {config_experiment['data']['path']} does not exist"

        # check if the data path ends with .csv - for now only support csv files
        assert config_experiment["data"]["path"].endswith(
            ".csv"
        ), "The path to the data file must end with .csv"

        # assign main parameters to the object
        self.set_params(config_experiment)

    def set_params(self, config_experiment):
        """Set the parameters of the synthesis experiment"""
        self.experiment_id = config_experiment["experiment"]["experiment_id"]
        self.param_grid = config_experiment["experiment"]["param_grid"]
        self.n_jobs = config_experiment["experiment"]["n_jobs"]
        self.verbose = config_experiment["experiment"]["verbose"]

        if self.verbose:
            print("Running synthesis experiment")
            print("Generators:", self.generators)
            print("Experiment values:", self.param_grid)


def load_configs(path):
    """Load the configuration files"""

    def load_json(path_to_file, verbose=1):
        with open(path_to_file, "r") as infile:
            loaded_file = json.load(infile)

        # sanity check
        assert loaded_file is not None

        if verbose > 0:
            print("Loaded:", path_to_file)

        return loaded_file

    config_experiment = load_json(path / "experiment_config.json")
    config_generator = load_json(path / "generator_config.json")

    return config_experiment, config_generator
