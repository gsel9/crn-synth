"""Utility functions for the experiment module"""
import os
from typing import Any, Dict, Iterable, Union

from sklearn.model_selection import ParameterGrid


def init_synthpipes(synth_pipes, generators, param_grid=None):
    """Initialize synthesis pipelines with generators and parameter grid."""

    def _check_iterable(x):
        # check if iterable
        if not isinstance(x, (list, tuple)):
            return [x]
        return x

    synth_pipes = _check_iterable(synth_pipes)
    generators = _check_iterable(generators)
    param_grid = ParameterGrid(param_grid) if param_grid is not None else None

    # create all possible combinations of synth_pipes, generators and param_grid
    synth_pipes_exp = []
    for synth_pipe in synth_pipes:
        for generator in generators:
            sp = synth_pipe.__copy__().set_generator(generator)

            # loop over parameter options
            if param_grid is not None:
                for params in param_grid:
                    # assign to synthpipe and generator (only if attribute already exists)
                    sp.set_params(**params)
                    sp.generator.set_params(**params)
                    synth_pipes_exp.append(sp)
    return synth_pipes_exp


def create_dir(path_to_dir, sub_dirs):
    """Ensure output directory for results exists."""

    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

    for subdir in sub_dirs:
        if not os.path.exists(path_to_dir / subdir):
            os.makedirs(path_to_dir / subdir)
