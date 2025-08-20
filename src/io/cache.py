""""""

from datetime import datetime
from pathlib import Path
import pickle

import numpy as np

from . import config as cfg


def save(data, filename=None, directory=None):
    """Save an arbitrary data object to a file. Uses numpy.save() if
    input is a numpy array, otherwise, uses pickle.dump().


    Parameters
    ----------
    data : arbitrary
        The data to be saved.

    
    Optional parameters
    -------------------
    filename : str or None
        Name of file to save. If None, uses 'data_cached_YYYMMDD_HHMM'
        where the date/time is the current write time.

    directory : str or pathlib.Path or None
        Directory within which to save. If None, gets from config.

    """

    if filename is None:
        filename = f"data_cached_{datetime.now().strftime('%Y%m%d_%H%M')}"

    if directory is None:
        directory = cfg.data_path["cache"]

    Path(directory).mkdir(parents=True, exist_ok=True)

    if type(data) in [np.ndarray]:
        np.save(Path(directory, filename), data)
    else:
        with open(Path(directory, filename), "wb") as file:
            pickle.dump(data, file)

    print(f"Saved: {str(Path(directory, filename))}")


def load(filename, directory=None):
    """Load an arbitrary data object from a file saved using numpy or pickle,
    depending on whether the file extension is .npy or .pkl, respectively.


    Parameters
    ----------
    filename : str
        Name of file to load.

    directory : str or pathlib.Path or None
        Directory within which to save. If None, gets from config.

    """

    if directory is None:
        directory = cfg.data_path["cache"]

    if filename.endswith(".npy"):
        data = np.load(Path(directory, filename))
    else:
        with open(Path(directory, filename), "rb") as file:
            data = pickle.load(file)

    print(f"Loaded: {str(Path(directory, filename))} (type: {repr(type(data))})")

    return data


def write_txt(data, filename, directory=None):
    """Write data to a text file."""

    if not filename.endswith(".txt"):
        filename += ".txt"

    if not data.endswith("\n"):
        data += "\n"

    with open(Path(directory, filename), "w+") as file:
        file.write(data)

    print(f"Saved: {str(Path(directory, filename))}")

