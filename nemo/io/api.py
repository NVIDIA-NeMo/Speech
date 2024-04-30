import pickle
from pathlib import Path
from typing import Any, Type, TypeVar

import fiddle as fdl

from nemo.io.pl import TrainerCheckpoint

CkptType = TypeVar("CkptType")


def load(path: Path, output_type: Type[CkptType] = Any) -> CkptType:
    """
    Loads a configuration from a pickle file and constructs an object of the specified type.

    Args:
        path (Path): The path to the pickle file or directory containing 'io.pkl'.
        output_type (Type[CkptType]): The type of the object to be constructed from the loaded data.

    Returns
    -------
        CkptType: An instance of the specified type constructed from the loaded configuration.

    Raises
    ------
        FileNotFoundError: If the specified file does not exist.

    Example:
        loaded_model = load("/path/to/model", output_type=MyModel)
    """
    del output_type  # Just for type-hint

    _path = Path(path)
    if hasattr(_path, 'is_dir') and _path.is_dir():
        _path = Path(_path) / "io.pkl"
    elif hasattr(_path, 'isdir') and _path.isdir:
        _path = Path(_path) / "io.pkl"

    if not _path.is_file():
        raise FileNotFoundError(f"No such file: '{_path}'")

    with open(_path, "rb") as f:
        config = pickle.load(f)

    return fdl.build(config)


def load_ckpt(path: Path) -> TrainerCheckpoint:
    """
    Loads a TrainerCheckpoint from a pickle file or directory.

    Args:
        path (Path): The path to the pickle file or directory containing 'io.pkl'.

    Returns
    -------
        TrainerCheckpoint: The loaded TrainerCheckpoint instance.

    Example:
        checkpoint: TrainerCheckpoint = load_ckpt("/path/to/checkpoint")
    """
    return load(path, output_type=TrainerCheckpoint)
