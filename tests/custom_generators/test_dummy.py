import pandas as pd
import pytest

from crnsynth.synth.custom_generators.dummy import DummySampler


def test_dummysampler():
    """Test the DummySampler plugin.

    Supposed to create an exact copy of the original dataset."""
    dummy = DummySampler()

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    dummy.fit(df)
    df_synth = dummy.generate().dataframe()

    assert df_synth.equals(
        df
    ), "DummySampler did not create an exact copy of the original dataset."

    # test value error if raised when count is different from original dataset size
    with pytest.raises(ValueError):
        dummy.generate(2)
