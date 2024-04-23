import pandas as pd
import pytest

from crnsynth.generators.dummy import DummyGenerator


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


def test_dummy_generator_copy_data(data_real):
    """Test that the DummyGenerator with sample_with_replace == False and all records see if it copies the data."""
    generator = DummyGenerator(sample_with_replace=False)
    generator.fit(data_real)
    data_synth = generator.generate(data_real.shape[0])
    assert data_real.equals(data_synth)


def test_dummy_generator_with_replace(data_real):
    """Test that the DummyGenerator with sample_with_replace == True and all records see if it does not copy the data."""
    generator = DummyGenerator(sample_with_replace=True, random_state=42)
    generator.fit(data_real)
    data_synth = generator.generate(data_real.shape[0])
    assert not data_real.equals(data_synth)


def test_dummy_generator_without_replace_too_many_rcords(data_real):
    """Test that the DummyGenerator with sample_with_replace == False and more records than the original dataset raises an error."""
    generator = DummyGenerator(sample_with_replace=False)
    generator.fit(data_real)
    with pytest.raises(ValueError):
        generator.generate(data_real.shape[0] + 1)


def test_dummy_generator_without_replace(data_real):
    """Test that the DummyGenerator with sample_with_replace == False and less records than the original dataset copies the data."""
    generator = DummyGenerator(sample_with_replace=False)
    generator.fit(data_real)
    data_synth = generator.generate(data_real.shape[0] - 1)
    assert data_synth.shape[0] == data_real.shape[0] - 1
    assert data_real.columns.tolist() == data_synth.columns.tolist()
    assert data_real.dtypes.tolist() == data_synth.dtypes.tolist()
    assert data_synth.isnull().sum().sum() == 0
    assert data_synth.notna().all().all()
