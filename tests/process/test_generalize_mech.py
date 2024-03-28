import numpy as np
import pandas as pd
import pytest

from crnsynth2.process.generalize_mech import NumericGeneralizationMech


def test_numeric_generalization_mech_initialization_with_uniform_inverse():
    mech = NumericGeneralizationMech(
        column="test", epsilon=1.0, bins=10, bounds=(0, 1), inverse="uniform"
    )
    assert mech.epsilon is None


def test_numeric_generalization_mech_initialization_with_truncated_normal_inverse():
    mech = NumericGeneralizationMech(
        column="test", epsilon=1.0, bins=10, bounds=(0, 1), inverse="truncated_normal"
    )
    assert mech.epsilon == 1.0


def test_numeric_generalization_mech_fit_with_invalid_bounds():
    mech = NumericGeneralizationMech(
        column="test", epsilon=1.0, bins=10, bounds=(1, 0), inverse="uniform"
    )
    data = pd.DataFrame({"test": np.random.rand(100)})
    with pytest.raises(ValueError):
        mech.fit(data)


def test_numeric_generalization_mech_fit_with_non_numeric_data():
    mech = NumericGeneralizationMech(
        column="test", epsilon=1.0, bins=10, bounds=(0, 1), inverse="uniform"
    )
    data = pd.DataFrame({"test": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        mech.fit(data)


def test_numeric_generalization_mech_transform_with_uniform_inverse():
    mech = NumericGeneralizationMech(
        column="test", epsilon=1.0, bins=10, bounds=(0, 1), inverse="uniform"
    )
    data = pd.DataFrame({"test": np.random.rand(100)})
    mech.fit(data)
    transformed_data = mech.transform(data)
    assert transformed_data["test"].between(0, 10).all()


def test_numeric_generalization_mech_inverse_transform_with_uniform_inverse():
    mech = NumericGeneralizationMech(
        column="test", epsilon=1.0, bins=10, bounds=(0, 1), inverse="uniform"
    )
    data = pd.DataFrame({"test": np.random.rand(100)})
    mech.fit(data)
    transformed_data = mech.transform(data)
    inverse_transformed_data = mech.inverse_transform(transformed_data)
    assert inverse_transformed_data["test"].between(0, 1).all()


def test_numeric_generalization_mech_inverse_transform_with_truncated_normal_inverse():
    mech = NumericGeneralizationMech(
        column="test", epsilon=1.0, bins=10, bounds=(0, 1), inverse="truncated_normal"
    )
    data = pd.DataFrame({"test": np.random.rand(100)})
    mech.fit(data)
    transformed_data = mech.transform(data)
    inverse_transformed_data = mech.inverse_transform(transformed_data)
    assert inverse_transformed_data["test"].between(0, 1).all()


# test inverse with nan value
def test_numeric_generalization_mech_inverse_transform_with_nan_value():
    mech = NumericGeneralizationMech(
        column="test", epsilon=1.0, bins=10, bounds=(0, 1), inverse="truncated_normal"
    )
    data = pd.DataFrame({"test": np.random.rand(100)})
    data.loc[0, "test"] = np.nan
    mech.fit(data)
    transformed_data = mech.transform(data)
    inverse_transformed_data = mech.inverse_transform(transformed_data)
    assert (
        inverse_transformed_data["test"][~inverse_transformed_data.test.isna()]
        .between(0, 1)
        .all()
    )
    assert np.isnan(inverse_transformed_data.loc[0, "test"])
