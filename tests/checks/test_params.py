"""Test parameter checks"""

import numpy as np
import pandas as pd
import pytest

from crnsynth.checks import params


def test_check_param_consistency():
    """Test parameter consistency check"""

    class A:
        random_state = None

    class B:
        random_state = 42

    classes = [A, B]

    expected_param_value = 42
    # catch assertion error - class A is inconsistent
    with pytest.raises(AssertionError):
        params.check_param_consistency(
            classes,
            parameter_name="random_state",
            parameter_value=expected_param_value,
            force_value=False,
        )

    # catch warning - class A is inconsistent, correct value
    with pytest.warns(UserWarning):
        params.check_param_consistency(
            classes,
            parameter_name="random_state",
            parameter_value=expected_param_value,
            force_value=True,
        )

        # check if parameter value is corrected
        assert A.random_state == expected_param_value
        assert B.random_state == expected_param_value
