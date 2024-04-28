from pathlib import Path
import numpy as np 


def remove_dir(path_to_dir):
    # removes a possibly non-empty directory 

    path_to_dir = Path(path_to_dir)

    if path_to_dir.is_dir():

        # empty the directory 
        for child in path_to_dir.iterdir():
            if child.is_file():
                child.unlink()
            
            else:
                remove_dir(child)

        path_to_dir.rmdir()


