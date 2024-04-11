"""Check parameters in classes and functions"""
import warnings
from typing import Any, Union

warnings.simplefilter(action="default")


def set_class_param(
    cls, parameter_name: str, parameter_value: Any, check_has_param: bool = True
):
    """Set parameter value in classes if it has the parameter."""
    if not check_has_param:
        setattr(cls, parameter_name, parameter_value)
    else:
        if hasattr(cls, parameter_name):
            setattr(cls, parameter_name, parameter_value)


def check_param_consistency(
    classes, parameter_name: str, parameter_value: Any, force_value: bool = False
):
    """Check parameter consistency across classes.

    For example: check_param_consistency(classes, "random_state", 42) will check if all classes have random_state = 42.
    If force_value is True, only raise a warning and set the value to 42 if the class has a different value.
    """
    for cls in classes:
        if hasattr(cls, parameter_name):
            # check if parameter value is consistent and raise warning if not
            if not force_value:
                assert getattr(cls, parameter_name) == parameter_value, (
                    f"Parameter {parameter_name} is inconsistent in {cls}"
                    f"with value {getattr(cls, parameter_name)}"
                )
            else:
                if getattr(cls, parameter_name) != parameter_value:
                    warnings.warn(
                        f"Parameter {parameter_name} is inconsistent in {cls} with value {getattr(cls, parameter_name)}. "
                        f"Setting to {parameter_value}"
                    )
                    set_class_param(cls, parameter_name, parameter_value)
    return classes
