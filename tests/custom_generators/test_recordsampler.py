import pandas as pd
import pytest

from crnsynth.synth.custom_generators.recordsampler import RecordSampler


def test_recordsampler():
    """Test the RecordSampler plugin.

    Supposed to re-sample records from the original dataset."""
    record_sampler = RecordSampler(random_state=42)

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    record_sampler.fit(df)
    df_synth = record_sampler.generate().dataframe()

    assert len(df_synth) == len(
        df
    ), "RecordSampler did not create a dataset of the same size as the original dataset."

    # check if all records from synth occur at least once in original dataset
    for i in range(len(df_synth)):
        assert (
            df_synth.iloc[i].to_list() in df.values.tolist()
        ), "RecordSampler did not create a dataset with records from the original dataset."

    # test value error if raised when count is greater than original dataset size
    record_sampler.replace = False
    with pytest.raises(ValueError):
        record_sampler.generate(4)
