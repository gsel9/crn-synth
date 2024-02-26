from synthesis.transformers.generalization import (
    bin_numeric_column,
    sample_from_binned_column,
)

from crnsynth2.synthpipes.generalized_synthpipe import GeneralizedSynthPipe


class AdultSynthPipe(GeneralizedSynthPipe):
    def __init__(
        self,
        generator,
        random_state=None,
        test_size=0.2,
        target_column=None,
        generalize=True,
        verbose=2,
    ) -> None:
        super().__init__(
            generator=generator,
            random_state=random_state,
            test_size=test_size,
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
            col_min=17,
            col_max=90,
            strategy="quantile",
        )

        data_real = bin_numeric_column(
            df=data_real,
            column_name="hours-per-week",
            n_bins=5,
            col_min=1,
            col_max=99,
            strategy="quantile",
        )
        return data_real

    def _reverse_generalization(self, data_synth):
        data_synth = sample_from_binned_column(
            df=data_synth,
            column_name="age",
            numeric_type="int",
            mean=38,
            std=13,
            random_state=self.random_state,
        )

        data_synth = sample_from_binned_column(
            df=data_synth,
            column_name="hours-per-week",
            numeric_type="int",
            mean=40,
            std=10,
            random_state=self.random_state,
        )
        return data_synth
