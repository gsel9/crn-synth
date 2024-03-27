from synthesis.transformers.generalization import (
    bin_numeric_column,
    sample_from_binned_column,
)

from crnsynth2.process.generalize_mech import NumericGeneralizationMech
from crnsynth2.synthpipes.generalize_synthpipe import GeneralizedSynthPipe
from examples.adult.adult_config import AGE_BOUNDS, HOURS_PER_WEEK_BOUNDS

# # parameters to compute differentially private which are used for the reverse generalization
# DP_PARAMS = [
#     DPParam(stat_name="mean", column="age", epsilon=0.025, bounds=AGE_BOUNDS),
#     DPParam(stat_name="std", column="age", epsilon=0.025, bounds=AGE_BOUNDS),
#     DPParam(
#         stat_name="mean",
#         column="hours-per-week",
#         epsilon=0.025,
#         bounds=HOURS_PER_WEEK_BOUNDS,
#     ),
#     DPParam(
#         stat_name="std",
#         column="hours-per-week",
#         epsilon=0.025,
#         bounds=HOURS_PER_WEEK_BOUNDS,
#     ),
# ]

GEN_MECHS = [
    NumericGeneralizationMech(column="age", epsilon=0.05, bins=5, bounds=AGE_BOUNDS),
    NumericGeneralizationMech(
        column="hours-per-week", epsilon=0.05, bins=5, bounds=HOURS_PER_WEEK_BOUNDS
    ),
]


class AdultSynthPipe(GeneralizedSynthPipe):
    def __init__(
        self,
        generator=None,
        random_state=None,
        holdout_size=0.2,
        target_column=None,
        generalizers=None,
        verbose=2,
    ) -> None:
        if generalizers is None:
            generalizers = GEN_MECHS

        super().__init__(
            generator=generator,
            random_state=random_state,
            holdout_size=holdout_size,
            generalizers=generalizers,
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
