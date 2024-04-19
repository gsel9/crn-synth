"""Utility functions for processing data."""


def flatten_dict(d, parent_key="", sep="_"):
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def reduce_dict(dictionary, keys_to_keep):
    """Reduce dictionary to only include specified keys."""
    return {k: v for k, v in dictionary.items() if k in keys_to_keep}
