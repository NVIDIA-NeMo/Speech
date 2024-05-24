import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar

import fiddle as fdl
import pytorch_lightning as pl

from nemo.lightning.io.mixin import ConnectorMixin, ConnT, ModelConnector
from nemo.lightning.io.pl import TrainerCheckpoint

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


def model_importer(
    target: Type[ConnectorMixin], ext: str, default_path: Optional[str] = None
) -> Callable[[Type[ConnT]], Type[ConnT]]:
    """
    Registers an importer for a model with a specified file extension and an optional default path.

    Args:
        target (Type[ConnectorMixin]): The model class to which the importer will be attached.
        ext (str): The file extension associated with the model files to be imported.
        default_path (Optional[str]): The default path where the model files are located, if any.

    Returns
    -------
        Callable[[Type[ConnT]], Type[ConnT]]: A decorator function that registers the importer
        to the model class.

    Example:
        @model_importer(MyModel, "hf", default_path="path/to/default")
        class MyModelHfImporter(io.ModelConnector):
            ...
    """
    return target.register_importer(ext, default_path=default_path)


def model_exporter(
    target: Type[ConnectorMixin], ext: str, default_path: Optional[str] = None
) -> Callable[[Type[ConnT]], Type[ConnT]]:
    """
    Registers an exporter for a model with a specified file extension and an optional default path.

    Args:
        target (Type[ConnectorMixin]): The model class to which the exporter will be attached.
        ext (str): The file extension associated with the model files to be exported.
        default_path (Optional[str]): The default path where the model files will be saved, if any.

    Returns
    -------
        Callable[[Type[ConnT]], Type[ConnT]]: A decorator function that registers the exporter
        to the model class.

    Example:
        @model_exporter(MyModel, "hf", default_path="path/to/default")
        class MyModelHFExporter(io.ModelConnector):
            ...
    """
    return target.register_exporter(ext, default_path=default_path)


def import_ckpt(
    model: pl.LightningModule, source: str, output_path: Optional[Path] = None, overwrite: bool = False
) -> Path:
    """
    Imports a checkpoint into a model using the model's associated importer, typically for
    the purpose of fine-tuning a community model trained in an external framework, such as
    Hugging Face. This function leverages the ConnectorMixin interface to integrate external
    checkpoint data seamlessly into the specified model instance.

    The importer component of the model reads the checkpoint data from the specified source
    and transforms it into the right format. This is particularly useful for adapting
    models that have been pre-trained in different environments or frameworks to be fine-tuned
    or further developed within the current system. The function allows for specifying an output
    path for the imported checkpoint; if not provided, the importer's default path will be used.
    The 'overwrite' parameter enables the replacement of existing data at the output path, which
    is useful when updating models with new data and discarding old checkpoint files.

    For instance, using `import_ckpt(Mistral7BModel(), "hf")` initiates the import process
    by searching for a registered model importer tagged with "hf". In NeMo, `HFMistral7BImporter`
    is registered under this tag via:
    `@io.model_importer(Mistral7BModel, "hf", default_path="mistralai/Mistral-7B-v0.1")`.
    This links `Mistral7BModel` to `HFMistral7BImporter`, designed for HuggingFace checkpoints.
    The importer then processes and integrates these checkpoints into `Mistral7BModel` for further
    fine-tuning.

    Args:
        model (pl.LightningModule): The model into which the checkpoint will be imported.
            This model must implement the ConnectorMixin, which includes the necessary
            importer method for checkpoint integration.
        source (str): The source from which the checkpoint will be imported. This can be
            a file path, URL, or any other string identifier that the model's importer
            can recognize.
        output_path (Optional[Path]): The path where the imported checkpoint will be stored.
            If not specified, the importer's default path is used.
        overwrite (bool): If set to True, existing files at the output path will be overwritten.
            This is useful for model updates where retaining old checkpoint files is not required.

    Returns
    -------
        Path: The path where the checkpoint has been saved after import. This path is determined
            by the importer, based on the provided output_path and its internal logic.

    Raises
    ------
        ValueError: If the model does not implement ConnectorMixin, indicating a lack of
            necessary importer functionality.

    Example:
        model = Mistral7BModel()
        imported_path = import_ckpt(model, "hf")
    """
    if not isinstance(model, ConnectorMixin):
        raise ValueError("Model must be an instance of ConnectorMixin")

    importer: ModelConnector = model.importer(source)
    return importer(overwrite=overwrite, output_path=output_path)


def load_connector_from_trainer_ckpt(path: Path, target: str) -> ModelConnector:
    model: pl.LightningModule = load_ckpt(path).model

    if not isinstance(model, ConnectorMixin):
        raise ValueError("Model must be an instance of ConnectorMixin")

    return model.exporter(target, path)


def export_ckpt(
    path: Path,
    target: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    load_connector: Callable[[Path, str], ModelConnector] = load_connector_from_trainer_ckpt,
) -> Path:
    """
    Exports a checkpoint from a model using the model's associated exporter, typically for
    the purpose of sharing a model that has been fine-tuned or customized within NeMo.
    This function leverages the ConnectorMixin interface to seamlessly integrate
    the model's state into an external checkpoint format.

    The exporter component of the model reads the model's state from the specified path and
    exports it into the format specified by the 'target' identifier. This is particularly
    useful for adapting models that have been developed or fine-tuned within the current system
    to be compatible with other environments or frameworks. The function allows for specifying
    an output path for the exported checkpoint; if not provided, the exporter's default path
    will be used. The 'overwrite' parameter enables the replacement of existing data at the
    output path, which is useful when updating models with new data and discarding old checkpoint
    files.

    Args:
        path (Path): The path to the model's checkpoint file from which data will be exported.
        target (str): The identifier for the exporter that defines the format of the export.
        output_path (Optional[Path]): The path where the exported checkpoint will be saved.
            If not specified, the exporter's default path is used.
        overwrite (bool): If set to True, existing files at the output path will be overwritten.
            This is useful for model updates where retaining old checkpoint files is not required.
        load_connector (Callable[[Path, str], ModelConnector]): A function to load the appropriate
            exporter based on the model and target format. Defaults to `load_connector_from_trainer_ckpt`.

    Returns
    -------
        Path: The path where the checkpoint has been saved after export. This path is determined
            by the exporter, based on the provided output_path and its internal logic.

    Raises
    ------
        ValueError: If the model does not implement ConnectorMixin, indicating a lack of
            necessary exporter functionality.

    Example:
        nemo_ckpt_path = Path("/path/to/model.ckpt")
        export_path = export_ckpt(nemo_ckpt_path, "hf")
    """
    exporter: ModelConnector = load_connector(path, target)
    _output_path = output_path or Path(path) / target

    return exporter(overwrite=overwrite, output_path=_output_path)
