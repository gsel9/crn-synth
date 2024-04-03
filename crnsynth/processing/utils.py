"""Utility functions for processing data."""


def reduce_dict(dictionary, keys_to_keep):
    """Reduce dictionary to only include specified keys."""
    return {k: v for k, v in dictionary.items() if k in keys_to_keep}
