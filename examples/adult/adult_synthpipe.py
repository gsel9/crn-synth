from deprecated.synthpipes import GeneralizedSynthPipe

from crnsynth2.process.generalize_mech import NumericGeneralizationMech
from examples.adult.adult_config import AGE_BOUNDS, HOURS_PER_WEEK_BOUNDS

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
