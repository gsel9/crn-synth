from pathlib import Path


def remove_dir(path_to_dir):
    """Removes a possibly non-empty directory.

    Args:
        datasets_real:
        datasets_fake:
        generator_names:
        metrics:

    Returns:

    """

    path_to_dir = Path(path_to_dir)

    if path_to_dir.is_dir():
        # empty the directory
        for child in path_to_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                remove_dir(child)

        path_to_dir.rmdir()
