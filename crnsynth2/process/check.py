from typing import Union


def check_consistent_random_state(classes, random_state: Union[None, int]) -> int:
    """Check random state is same for all classes."""
    for cls in classes:
        if hasattr(cls, "random_state"):
            assert cls.random_state == random_state, (
                "Random state is not consistent across classes.\n"
                "Class: {cls} has random state = {cls.random_state} "
                "while global random state = {random_state}."
            )
