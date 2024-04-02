"""Utility functions"""


def reduce_dict(dictionary, keys_to_keep):
    """Reduce dictionary to only include specified keys."""
    return {k: v for k, v in dictionary.items() if k in keys_to_keep}
