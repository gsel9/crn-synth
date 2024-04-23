import pandas as pd
import pytest

from crnsynth.generators.privbayes import PrivBayes


@pytest.fixture
def data_real():
    return pd.DataFrame(
        {
            "column1": [1, 2, 3, 4],
            "column2": [5, 6, 7, 8],
            "column3": [9, 10, 11, 12],
            "column4": [13, 14, 15, 16],
        }
    )


def test_privbayes_generator(data_real):
    generator = PrivBayes(epsilon=0.1)
    generator.fit(data_real)
    data_synth = generator.generate(4)
    assert data_synth.shape == data_real.shape
    assert data_synth.columns.tolist() == data_real.columns.tolist()
    assert data_synth.dtypes.tolist() == data_real.dtypes.tolist()
    assert data_synth.isnull().sum().sum() == 0
    assert data_synth.notna().all().all()
