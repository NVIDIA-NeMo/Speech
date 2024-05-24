import functools
import inspect
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

import fiddle as fdl
from cloudpickle import dump
from typing_extensions import Self

from nemo.lightning.io.capture import IOProtocol
from nemo.lightning.io.connector import ModelConnector

ConnT = TypeVar('ConnT', bound=ModelConnector)


class IOMixin:
    """
    A mixin class designed to capture the arguments passed to the `__init__` method,
    facilitating the re-creation of the object through `io.reinit` method using stored configurations.

    This class intercepts the initialization of an object to store the arguments in a configuration
    object, which can be serialized and later used to reinitialize the object to its original state.
    It utilizes `fdl.Config` from the Fiddle library to create a structured configuration object
    that holds the initialization parameters. This configuration object is crucial for enabling
    serialization and deserialization of the parameters, thus allowing the object to be reconstructed
    at a later time with the same initial state.

    Attributes
    ----------
        __io__ (fdl.Config[Self]): A configuration object that stores the captured initialization
        parameters in a structured format. This object is an instance of `fdl.Config`, which allows
        for the serialization and deserialization of the parameters, enabling the object to be
        reconstructed at a later time with the same initial state.

    Examples
    --------
        from nemo.lightning import io

        class ExampleClass(io.IOMixin):
            def __init__(self, param1, param2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

        # Creating an instance of ExampleClass
        example = ExampleClass('value1', 'value2')
        example_copy = io.reinit(example)


    Note:
        For more information on `fdl.Config`, refer to the Fiddle library documentation at
        [Fiddle Config Documentation](https://fiddle.readthedocs.io/en/latest/api_reference/core.html#config).

    """

    __io__ = fdl.Config[Self]

    def __new__(cls, *args, **kwargs):
        """
        Overrides the default object creation process to wrap the `__init__` method, allowing
        initialization arguments to be captured and stored in the `__io__` attribute.

        Args:
            *args: Variable length argument list for the `__init__` method.
            **kwargs: Arbitrary keyword arguments for the `__init__` method.

        Returns
        -------
            The newly created object instance.
        """
        original_init = cls.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            cfg_kwargs = self.io_transform_args(original_init, *args, **kwargs)
            self.__io__ = self.io_init(**cfg_kwargs)
            original_init(self, *args, **kwargs)

        cls.__init__ = wrapped_init
        output = object().__new__(cls)

        return output

    def io_transform_args(self, init_fn, *args, **kwargs) -> Dict[str, Any]:
        """
        Transforms and captures the arguments passed to the `__init__` method, filtering out
        any arguments that are instances of `IOProtocol` or are dataclass fields with default
        factories.

        Args:
            init_fn (Callable): The original `__init__` method of the class.
            *args: Variable length argument list for the `__init__` method.
            **kwargs: Arbitrary keyword arguments for the `__init__` method.

        Returns
        -------
            Dict[str, Any]: A dictionary of the captured and transformed arguments.
        """
        sig = inspect.signature(init_fn)
        bound_args = sig.bind_partial(self, *args, **kwargs)
        bound_args.apply_defaults()
        config_kwargs = {k: v for k, v in bound_args.arguments.items() if k != "self"}

        to_del = []
        for key in config_kwargs:
            if isinstance(config_kwargs[key], IOProtocol):
                config_kwargs[key] = config_kwargs[key].__io__
            if is_dataclass(self):
                # Check if the arg is a factory (dataclasses.field)
                if config_kwargs[key].__class__.__name__ == "_HAS_DEFAULT_FACTORY_CLASS":
                    to_del.append(key)

        for key in to_del:
            del config_kwargs[key]

        return config_kwargs

    def io_init(self, **kwargs) -> fdl.Config[Self]:
        """
        Initializes the configuration object (`__io__`) with the captured arguments.

        Args:
            **kwargs: A dictionary of arguments that were captured during object initialization.

        Returns
        -------
            fdl.Config[Self]: The initialized configuration object.
        """
        return fdl.Config(type(self), **kwargs)

    def io_dump(self, output: Path):
        """
        Serializes the configuration object (`__io__`) to a file, allowing the object state to be
        saved and later restored.

        Args:
            output (Path): The path to the file where the configuration object will be serialized.
        """
        config_path = Path(output) / "io.pkl"
        with open(config_path, "wb") as f:
            dump(self.__io__, f)


class ConnectorMixin:
    """
    A mixin class that provides methods to register and retrieve model connectors for importing
    and exporting models. This class supports dynamic registration of connectors based on file
    extensions, which facilitates the customization and extension of model serialization and
    deserialization processes.

    Attributes
    ----------
        _IMPORTERS (Dict[str, Type[ModelConnector]]): A dictionary mapping file extensions to
            model connector classes that handle the import process.
        _EXPORTERS (Dict[str, Type[ModelConnector]]): A dictionary mapping file extensions to
            model connector classes that handle the export process.
    """

    _IMPORTERS: Dict[str, Type[ModelConnector]] = {}
    _EXPORTERS: Dict[str, Type[ModelConnector]] = {}

    @classmethod
    def import_from(cls, path: str) -> Self:
        """
        Creates an instance of a model by using the appropriate importer based on the file
        extension of the provided path.

        Args:
            path (str): The path to the model file to be imported.

        Example:
            from nemo.collections import llm
            model = llm.Mistral7BModel.import_from("hf")

        Returns
        -------
            Self: An instance of the model initialized from the imported data.
        """
        output = cls._get_connector(path).init()
        output.ckpt_path = output.import_ckpt_path(path)

        return output

    @classmethod
    def register_importer(cls, ext: str, default_path: Optional[str] = None) -> Callable[[Type[ConnT]], Type[ConnT]]:
        """
        A class method decorator to register a model connector as an importer for a specific file
        extension.

        Args:
            ext (str): The file extension to associate with the model connector.
            default_path (Optional[str]): The default path to use if no path is specified during import.

        Returns
        -------
            Callable[[Type[ConnT]], Type[ConnT]]: The decorator that registers the model connector.
        """

        def decorator(connector: Type[ConnT]) -> Type[ConnT]:
            cls._IMPORTERS[ext] = connector
            if default_path:
                connector.default_path = default_path
            return connector

        return decorator

    @classmethod
    def register_exporter(cls, ext: str, default_path: Optional[str] = None) -> Callable[[Type[ConnT]], Type[ConnT]]:
        """
        A class method decorator to register a model connector as an exporter for a specific file
        extension.

        Args:
            ext (str): The file extension to associate with the model connector.
            default_path (Optional[str]): The default path to use if no path is specified during export.

        Returns
        -------
            Callable[[Type[ConnT]], Type[ConnT]]: The decorator that registers the model connector.
        """

        def decorator(connector: Type[ConnT]) -> Type[ConnT]:
            cls._EXPORTERS[ext] = connector
            if default_path:
                connector.default_path = default_path
            return connector

        return decorator

    @classmethod
    def importer(cls, path: str) -> ModelConnector:
        """
        Retrieves the appropriate model connector for importing based on the extension of the
        provided path.

        Args:
            path (str): The path to the model file to be imported.

        Returns
        -------
            ModelConnector: The model connector instance capable of handling the import.
        """
        return cls._get_connector(path, importer=True)

    @classmethod
    def exporter(cls, ext: str, path: Union[str, Path]) -> ModelConnector:
        """
        Retrieves the appropriate model connector for exporting based on the extension.

        Args:
            ext (str): The file extension associated with the model connector.
            path (Union[str, Path]): The path where the model will be exported.

        Returns
        -------
            ModelConnector: The model connector instance capable of handling the export.
        """
        return cls._get_connector(ext, path, importer=False)

    def import_ckpt(self, path: str, overwrite: bool = False, base_path: Optional[Path] = None) -> Path:
        """
        Imports a checkpoint from a specified path, potentially overwriting existing files.

        Args:
            path (str): The path to the checkpoint file to be imported.
            overwrite (bool): Flag to determine if existing files should be overwritten (default is False).
            base_path (Optional[Path]): The base path where the checkpoint file is located; used to resolve
                                        relative paths.

        Returns
        -------
            Path: The path to the imported checkpoint.

        Raises
        ------
            FileNotFoundError: If the checkpoint file does not exist at the specified path.
        """
        connector = self._get_connector(path)
        ckpt_path: Path = connector.local_path(base_path=base_path)
        ckpt_path = connector(ckpt_path, overwrite=overwrite)

        return ckpt_path

    @classmethod
    def _get_connector(cls, ext, path=None, importer=True) -> ModelConnector:
        """
        Retrieves the appropriate model connector based on the file extension and path,
        distinguishing between importers and exporters.

        Args:
            ext (str): The file extension or a URI that may include a protocol specifier.
            path (Optional[Union[str, Path]]): The path where the model file is located or will be saved.
            importer (bool): Flag to determine if the connector is for importing (True) or exporting (False).

        Returns
        -------
            ModelConnector: The model connector instance capable of handling the import or export.

        Raises
        ------
            ValueError: If no connector is found for the specified extension or if no default path is provided
                        when required.
        """
        _path = None
        if "://" in ext:
            ext, _path = ext.split("://")
        else:
            _path = path

        connector = cls._IMPORTERS.get(ext) if importer else cls._EXPORTERS.get(ext)
        if not connector:
            raise ValueError(f"No connector found for extension '{ext}'")

        if not _path:
            if not connector.default_path:
                raise ValueError(f"No default path specified for extension '{ext}'. ", "Please provide a path")

            return connector()

        return connector(_path)
