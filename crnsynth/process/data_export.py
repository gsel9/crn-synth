"""Functionality writing data to disk."""

# generic
import json
import os

# third party
from synthcity.utils.serialization import save_to_file

# local
from crnsynth.configs import config
from crnsynth.process.util import make_json_serializeable


def save_generator(path_to_file, generator, save_kwargs={}, verbose=1):
    save_to_file(path_to_file, generator, **save_kwargs)

    if verbose > 0:
        print("Saved to disk:", path_to_file)


def save_csv(path_to_file, dataframe, save_kwargs={}, verbose=1):
    dataframe.to_csv(path_to_file, **save_kwargs)

    if verbose > 0:
        print("Saved to disk:", path_to_file)


def save_json(path_to_file, data, verbose=1):
    with open(path_to_file, "w", encoding="utf-8") as outfile:
        json.dump(
            data, outfile, ensure_ascii=False, indent=4, default=make_json_serializeable
        )

    if verbose > 0:
        print("Saved to disk:", path_to_file)


def save_output_from_experiment(
    path_to_dir,
    exp_name,
    file_suffix,
    generator=None,
    data_synth=None,
    score_report=None,
    run_config=None,
):
    # path_to_exp = config.PATH_RESULTS / exp_name
    filename = exp_name + "_" + file_suffix

    create_experiment_dir(path_to_dir)

    if generator is not None:
        save_generator(path_to_dir / f"generators/{filename}.pkl", generator)

    if data_synth is not None:
        save_csv(path_to_dir / f"synthetic_data/{filename}.csv", data_synth)

    if score_report is not None:
        save_csv(path_to_dir / f"reports/{filename}.csv", score_report)

    if run_config is not None:
        save_json(path_to_dir / f"configs/{filename}.json", run_config)


def save_experiment_from_config(
    path_to_dir, run_config, generator=None, data_synth=None, score_report=None
):
    # output management
    config_base = run_config["base"]

    if generator is not None:
        file_suffix = "{}_{}".format(generator.name(), config_base["run_id"])
    else:
        file_suffix = config_base["run_id"]

    generator = generator if config_base["save_generator"] else None
    data_synth = data_synth if config_base["save_synthetic"] else None
    run_config = run_config if config_base["save_config"] else None
    score_report = score_report if config_base["save_score_report"] else None

    save_output_from_experiment(
        path_to_dir=path_to_dir,
        exp_name=config_base["experiment_id"],
        file_suffix=file_suffix,
        generator=generator,
        data_synth=data_synth,
        run_config=run_config,
        score_report=score_report,
    )


def create_experiment_dir(path_to_dir):
    """Ensure output directory for results exists."""

    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

    for subdir in config.DEFAULT_DIRS:
        if not os.path.exists(path_to_dir / subdir):
            os.makedirs(path_to_dir / subdir)
