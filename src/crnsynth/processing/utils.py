"""Utility functions for processing data."""
from pathlib import Path


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


def remove_workspace(path_to_dir):
    # removes a possibly non-empty directory 

    path_to_dir = Path(path_to_dir)

    if path_to_dir.is_dir():

        # empty the directory 
        for child in path_to_dir.iterdir():
            if child.is_file():
                child.unlink()
            
            else:
                remove_workspace(child)

        path_to_dir.rmdir()