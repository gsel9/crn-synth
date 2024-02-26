"""Synthesis pipeline"""
import warnings
from typing import Tuple, Union

import pandas as pd

from crnsynth2.generators.base_generator import BaseGenerator

warnings.filterwarnings("default")


class BaseSynthPipe:
    """Base class for synthesis pipelines. Handles data processing and generation of synthetic data.
    Use AdvancedSynthPipe to inherit more functionality."""

    def __init__(
        self,
        generator: BaseGenerator,
        random_state: Union[int, None] = None,
        verbose: int = 1,
    ) -> None:
        """Initialize synthesis pipeline.

        Args:
            generator: Generator to use for synthesizing data
            random_state: Random state
            verbose: Verbosity level
        """
        self.generator = generator
        self.random_state = random_state
        self.verbose = verbose

    @property
    def name(self):
        """Return the name of the pipeline"""
        raise NotImplementedError("name() method not implemented")

    def process_data(self, data_real: pd.DataFrame) -> pd.DataFrame:
        """Process real data"""
        raise NotImplementedError("Should implement .process_data()")

    def fit(self, data_real: pd.DataFrame) -> "BaseSynthPipe":
        """Fit generator on processed real data"""
        self.generator.fit(data_real)
        return self

    def generate(self, n_records: int) -> pd.DataFrame:
        """Generate records using fitted generator"""
        return self.generator.generate(n_records)

    def postprocess_synthetic_data(self, data_synth: pd.DataFrame) -> pd.DataFrame:
        """Postprocess synthetic data"""
        raise NotImplementedError("Should implement .postprocess_synthetic_data()")

    def run(self, data_real, n_records: Union[int, None] = None):
        """Run all steps in synthesis pipeline. User can run these steps one by one themselves as well."""
        # process real data
        data_real = self.process_data(data_real)

        # fit generator on processed real data
        self.fit(data_real)

        # generate records using fitted generator
        n_records = n_records if n_records is not None else data_real.shape[0]
        data_synth = self.generate(n_records)

        # postprocess synthetic data
        data_synth = self.postprocess_synthetic_data(data_synth)
        return data_synth

    def set_generator(self, generator: BaseGenerator) -> None:
        """Set generator for synthesis"""
        self.generator = generator
