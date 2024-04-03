"""Check parameters in classes and functions"""
from typing import Any, Union


def check_param_consistency(classes, parameter_name: str, parameter_value: Any):
    """Check parameter consistency across classes.

    For example: check_param_consistency(classes, "random_state", 42) will check if all classes have random_state = 42.
    """
    for cls in classes:
        if hasattr(cls, parameter_name):
            assert getattr(cls, parameter_name) == parameter_value, (
                f"Parameter {parameter_name} is not consistent across classes.\n"
                f"Class: {cls} has {parameter_name} = {getattr(cls, parameter_name)} "
                f"while desired {parameter_name} = {parameter_value}."
            )
