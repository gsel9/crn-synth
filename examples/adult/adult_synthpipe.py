from synthesis.transformers.generalization import (
    bin_numeric_column,
    sample_from_binned_column,
)

from crnsynth2.synthpipes.dp_synthpipe import DPParam, DPPipeline
from examples.adult.adult_config import AGE_BOUNDS, HOURS_PER_WEEK_BOUNDS

# parameters to compute differentially private which are used for the reverse generalization
DP_PARAMS = [
    DPParam(stat_name="mean", column="age", epsilon=0.025, bounds=AGE_BOUNDS),
    DPParam(stat_name="std", column="age", epsilon=0.025, bounds=AGE_BOUNDS),
    DPParam(
        stat_name="mean",
        column="hours-per-week",
        epsilon=0.025,
        bounds=HOURS_PER_WEEK_BOUNDS,
    ),
    DPParam(
        stat_name="std",
        column="hours-per-week",
        epsilon=0.025,
        bounds=HOURS_PER_WEEK_BOUNDS,
    ),
]


class AdultSynthPipe(DPPipeline):
    def __init__(
        self,
        generator=None,
        dp_params=DP_PARAMS,
        random_state=None,
        holdout_size=0.2,
        target_column=None,
        generalize=True,
        verbose=2,
    ) -> None:
        super().__init__(
            generator=generator,
            dp_params=dp_params,
            random_state=random_state,
            holdout_size=holdout_size,
            generalize=generalize,
            target_column=target_column,
            verbose=verbose,
        )

    def process_data(self, data_real):
        # reduce columns
        columns_subset = [
            "age",
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "hours-per-week",
            "native-country",
            "income",
        ]
        data_real = data_real[columns_subset]

        return data_real

    def postprocess_synthetic_data(self, data_synth):
        # no postprocessing needed
        return data_synth

    def _generalize_data(self, data_real):
        data_real = bin_numeric_column(
            df=data_real,
            column_name="age",
            n_bins=5,
            col_min=AGE_BOUNDS[0],
            col_max=AGE_BOUNDS[1],
            strategy="quantile",
        )

        data_real = bin_numeric_column(
            df=data_real,
            column_name="hours-per-week",
            n_bins=5,
            col_min=HOURS_PER_WEEK_BOUNDS[0],
            col_max=HOURS_PER_WEEK_BOUNDS[1],
            strategy="quantile",
        )
        return data_real

    def _reverse_generalization(self, data_synth):
        data_synth = sample_from_binned_column(
            df=data_synth,
            column_name="age",
            numeric_type="int",
            mean=self._get_dp_param("mean", "age"),
            std=self._get_dp_param("std", "age"),
            random_state=self.random_state,
        )

        data_synth = sample_from_binned_column(
            df=data_synth,
            column_name="hours-per-week",
            numeric_type="int",
            mean=self._get_dp_param("mean", "hours-per-week"),
            std=self._get_dp_param("std", "hours-per-week"),
            random_state=self.random_state,
        )
        return data_synth
