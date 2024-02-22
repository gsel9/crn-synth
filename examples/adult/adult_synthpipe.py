from synthesis.transformers.generalization import (
    bin_numeric_column,
    sample_from_binned_column,
)

from crnsynth2.process.pipeline.synthpipe import BaseSynthPipe


class AdultSynthPipe(BaseSynthPipe):
    def __init__(
        self,
        generator,
        data_name="adult",
        target_column="income",
        test_size=0.2,
        output_train_format=False,
        generalize=True,
        data_loader_name="generic",
        random_state=None,
        warn=True,
        verbose=2,
    ) -> None:
        super().__init__(
            generator=generator,
            data_name=data_name,
            target_column=target_column,
            test_size=test_size,
            output_train_format=output_train_format,
            generalize=generalize,
            data_loader_name=data_loader_name,
            random_state=random_state,
            warn=warn,
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

        return super().process_data(data_real)

    def _generalize_data(self, data_real):
        data_real = data_real.bin_numeric_column(
            column_name="age", n_bins=5, col_min=17, col_max=90, strategy="quantile"
        ).bin_numeric_column(
            column_name="hours-per-week",
            n_bins=5,
            col_min=1,
            col_max=99,
            strategy="quantile",
        )
        return super()._generalize_data(data_real)

    def _reverse_generalization(self, data_synth):
        data_synth = data_synth.sample_from_binned_column(
            column_name="age",
            numeric_type="int",
            mean=38,
            std=13,
            random_state=self.random_state,
        ).sample_from_binned_column(
            column_name="hours-per-week",
            numeric_type="int",
            mean=40,
            std=10,
            random_state=self.random_state,
        )
        return super()._reverse_generalization(data_synth)
