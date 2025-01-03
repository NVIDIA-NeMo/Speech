# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import inspect
import json
import shutil
import threading
import types
import uuid
from copy import deepcopy
from dataclasses import is_dataclass
from pathlib import Path
from pydoc import locate
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
from cloudpickle import dump
from cloudpickle import load as pickle_load
from fiddle._src import config as config_lib
from fiddle._src import partial
from fiddle._src.experimental import serialization
from typing_extensions import Self

from nemo.lightning.io.artifact.base import Artifact
from nemo.lightning.io.capture import IOProtocol
from nemo.lightning.io.connector import ModelConnector
from nemo.lightning.io.fdl_torch import enable as _enable_ext
from nemo.utils import logging

ConnT = TypeVar("ConnT", bound=ModelConnector)
CkptType = TypeVar("CkptType")
_enable_ext()


# Thread-local storage for artifacts directory
_thread_local = threading.local()


def _ordered_arguments_with_default(data: config_lib.Config) -> Dict[Union[int, str], Any]:
    result = config_lib.ordered_arguments(data, include_defaults=True)
    for key, arg in result.items():
        if isinstance(arg, config_lib.Config):
            ordered_arg = _ordered_arguments_with_default(arg)
            result[key] = ordered_arg

    if "__fn_or_cls__" in result:
        raise ValueError(
            "It is not supported to dump objects of functions/classes " "that have a __fn_or_cls__ parameter."
        )

    result["_target_"] = (
        f"{inspect.getmodule(config_lib.get_callable(data)).__name__}" f".{config_lib.get_callable(data).__qualname__}"
    )
    if isinstance(data, partial.Partial):
        result["_partial_"] = True

    return result


def _config_representer_with_defaults(dumper, data, type_name="Config"):
    """Returns a YAML representation of `data`."""
    value = _ordered_arguments_with_default(data)
    return dumper.represent_data(value)


def _partial_representer_with_defaults(dumper, data):
    return _config_representer_with_defaults(dumper, data, type_name="Partial")


def _safe_object_representer(dumper, data):
    """
    Represent a given object as YAML using the specified dumper.

    This function is a fallback for objects that don't have specific representers.
    If the object has __qualname__ attr,
    the __target__ is set to f"{inspect.getmodule(obj).__name__}.{obj.__qualname__}".
    If the object does not have a __qualname__ attr, the __target__ is set from its __class__ attr.
    The __call__ key is used to indicate whether the target should be called to create an instance.

    Args:
        dumper (yaml.Dumper): The YAML dumper to use for serialization.
        data (Any): The data to serialize. This can be any Python object,
            but if it's a class or a class instance, special handling will be applied.

    Returns:
        str: The YAML representation of the data.
    """
    try:
        obj = data
        target = f"{inspect.getmodule(obj).__name__}.{obj.__qualname__}"
        call = False
    except AttributeError:
        obj = data.__class__
        target = f"{inspect.getmodule(obj).__name__}.{obj.__qualname__}"
        call = True

    value = {
        "_target_": target,  # type: ignore
        "_call_": call,
    }
    return dumper.represent_data(value)


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

    __io__: fdl.Config[Self]

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
        cls = _io_wrap_init(cls)
        output = object().__new__(cls)

        return output

    def __init_subclass__(cls):
        _io_register_serialization(cls)

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
        return _io_transform_args(self, init_fn, *args, **kwargs)

    def io_init(self, **kwargs) -> fdl.Config[Self]:
        """
        Initializes the configuration object (`__io__`) with the captured arguments.

        Args:
            **kwargs: A dictionary of arguments that were captured during object initialization.

        Returns
        -------
            fdl.Config[Self]: The initialized configuration object.
        """
        return _io_init(self, **kwargs)

    @classmethod
    def io_artifacts(cls) -> List[Artifact]:
        """Initialize io artifacts"""
        return []

    def io_dump(self, output: Path, yaml_attrs: list[str]):
        """
        Serializes the configuration object (`__io__`) to a file, allowing the object state to be
        saved and later restored. Also creates an artifacts directory and stores it in a thread-local
        global variable. If the artifacts directory is empty at the end, it is deleted.

        Args:
            output (Path): The path to the directory where the configuration object and artifacts
                           will be stored.
        """
        output_path = Path(output)
        local_artifacts_dir = "."
        artifacts_dir = output_path / local_artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Store artifacts directory in thread-local storage
        _thread_local.local_artifacts_dir = local_artifacts_dir
        _thread_local.output_path = output_path

        config_path = output_path / "io.json"
        with open(config_path, "w") as f:
            io = deepcopy(self.__io__)
            _artifact_transform_save(io, output_path, local_artifacts_dir)
            json = serialization.dump_json(io)
            f.write(json)

        yaml_configs = self._io_dump_yaml(io, attrs=yaml_attrs)
        for attr, serialized_str in yaml_configs.items():
            _path = output_path / f"{attr}.yaml"
            _path.write_text(serialized_str)

        # Clear thread-local storage after io_dump is complete
        del _thread_local.local_artifacts_dir
        del _thread_local.output_path

        # Check if artifacts directory is empty and delete if so
        if not any(artifacts_dir.iterdir()):
            shutil.rmtree(artifacts_dir)

    def _io_dump_yaml(self, io: config_lib.Config, attrs: list[str]):
        import yaml

        original_representers = yaml.SafeDumper.yaml_representers.copy()

        from nemo_run.config import Config, Partial
        from nemo_run.core.serialization.yaml import YamlSerializer

        yaml.SafeDumper.add_representer(config_lib.Config, _config_representer_with_defaults)
        yaml.SafeDumper.add_representer(partial.Partial, _partial_representer_with_defaults)
        yaml.SafeDumper.add_representer(Config, _config_representer_with_defaults)
        yaml.SafeDumper.add_representer(Partial, _partial_representer_with_defaults)

        yaml.SafeDumper.add_multi_representer(object, _safe_object_representer)

        serializer = YamlSerializer()
        result = {}
        for attr in attrs:
            result[attr] = serializer.serialize(getattr(io, attr))

        yaml.SafeDumper.yaml_representers = original_representers
        return result


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
        output.ckpt_path = output.import_ckpt(path)

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
            cls._IMPORTERS[str(cls) + ext] = connector
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
            cls._EXPORTERS[str(cls) + ext] = connector
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

    def import_ckpt(self, path: str, overwrite: bool = False, base_path: Optional[Path] = None, **kwargs) -> Path:
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
        connector = self._get_connector(path, **kwargs)
        ckpt_path: Path = connector.local_path(base_path=base_path)
        ckpt_path = connector(ckpt_path, overwrite=overwrite)
        connector.on_import_ckpt(self)
        return ckpt_path

    @classmethod
    def _get_connector(
        cls, ext: Union[str, Path], path: Optional[Union[str, Path]] = None, importer: bool = True, **kwargs
    ) -> ModelConnector:
        """
        Retrieves the appropriate model connector based on the file extension and path,
        distinguishing between importers and exporters.

        Args:
            ext (Union[str, Path]): The file extension or a URI that may include a protocol specifier.
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
        ext = str(ext)
        if "://" in ext:
            ext, _path = ext.split("://")
        else:
            _path = str(path)

        connector = cls._IMPORTERS.get(str(cls) + ext) if importer else cls._EXPORTERS.get(str(cls) + ext)
        if not connector:
            raise ValueError(f"No connector found for extension '{ext}' for {cls}")

        if not _path:
            if not connector.default_path:
                raise ValueError(f"No default path specified for extension '{ext}'. ", "Please provide a path")

            return connector()

        return connector(_path, **kwargs)


def track_io(target, artifacts: Optional[List[Artifact]] = None):
    """
    Adds IO functionality to the target object or eligible classes in the target module
    by wrapping __init__ and registering serialization methods.

    Args:
        target (object or types.ModuleType): The target object or module to modify.

    Returns:
        object or types.ModuleType: The modified target with IO functionality added to eligible classes.

    Examples:
        >>> from nemo.collections.common import tokenizers
        >>> modified_tokenizers = track_io(tokenizers)
        >>> ModifiedWordTokenizer = track_io(tokenizers.WordTokenizer)
    """

    def _add_io_to_class(cls):
        if inspect.isclass(cls) and hasattr(cls, "__init__") and not hasattr(cls, "__io__"):
            if cls in [str, int, float, tuple, list, dict, bool, type(None)]:
                return cls

            cls = _io_wrap_init(cls)
            _io_register_serialization(cls)
            cls.__io_artifacts__ = artifacts or []
        return cls

    def _process_module(module):
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and _is_defined_in_module_or_submodules(obj, module):
                setattr(module, name, _add_io_to_class(obj))
        return module

    def _is_defined_in_module_or_submodules(obj, module):
        return obj.__module__ == module.__name__ or obj.__module__.startswith(f"{module.__name__}.")

    if isinstance(target, types.ModuleType):
        return _process_module(target)
    elif inspect.isclass(target):
        return _add_io_to_class(target)
    else:
        raise TypeError("Target must be a module or a class")


def _io_transform_args(self, init_fn, *args, **kwargs) -> Dict[str, Any]:
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
    config_kwargs = {k: v for k, v in bound_args.arguments.items() if k != "self"}

    to_del = []
    for key in config_kwargs:
        if isinstance(config_kwargs[key], IOProtocol):
            config_kwargs[key] = config_kwargs[key].__io__
        if is_dataclass(config_kwargs[key]):
            config_kwargs[key] = fdl_dc.convert_dataclasses_to_configs(config_kwargs[key], allow_post_init=True)
            # Check if the arg is a factory (dataclasses.field)
        if config_kwargs[key].__class__.__name__ == "_HAS_DEFAULT_FACTORY_CLASS":
            to_del.append(key)

    for key in to_del:
        del config_kwargs[key]

    return config_kwargs


def _io_init(self, **kwargs) -> fdl.Config[Self]:
    """
    Initializes the configuration object (`__io__`) with the captured arguments.

    Args:
        **kwargs: A dictionary of arguments that were captured during object initialization.

    Returns
    -------
        fdl.Config[Self]: The initialized configuration object.
    """
    try:
        return fdl.Config(type(self), **kwargs)
    except Exception as e:
        error_msg = (
            f"Error creating fdl.Config for {type(self).__name__}: {str(e)}\n"
            f"Arguments that caused the error: {kwargs}\n"
            f"This may be due to unsupported argument types or nested configurations."
        )
        raise RuntimeError(error_msg) from e


def _io_wrap_init(cls):
    """Wraps the __init__ method of a class to add IO functionality."""
    original_init = cls.__init__

    if getattr(cls, "__wrapped_init__", False):
        return cls

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        if hasattr(self, "io_transform_args"):
            cfg_kwargs = self.io_transform_args(original_init, *args, **kwargs)
        else:
            cfg_kwargs = _io_transform_args(self, original_init, *args, **kwargs)
        if hasattr(self, "io_init"):
            self.__io__ = self.io_init(**cfg_kwargs)
        else:
            self.__io__ = _io_init(self, **cfg_kwargs)

        original_init(self, *args, **kwargs)

    cls.__init__ = wrapped_init
    cls.__wrapped_init__ = True
    return cls


def _io_register_serialization(cls):
    serialization.register_node_traverser(
        cls,
        flatten_fn=_io_flatten_object,
        unflatten_fn=_io_unflatten_object,
        path_elements_fn=_io_path_elements_fn,
    )


def _io_flatten_object(instance):
    try:
        serialization.dump_json(instance.__io__)
    except (serialization.UnserializableValueError, AttributeError) as e:
        if not hasattr(_thread_local, "local_artifacts_dir") or not hasattr(_thread_local, "output_path"):
            raise e

        local_artifact_path = Path(_thread_local.local_artifacts_dir) / f"{uuid.uuid4()}"
        output_path = _thread_local.output_path
        artifact_path = output_path / local_artifact_path
        with open(artifact_path, "wb") as f:
            dump(getattr(instance, "__io__", instance), f)
        return (str(local_artifact_path),), None

    return instance.__io__.__flatten__()


def _io_unflatten_object(values, metadata):
    if not hasattr(_thread_local, "output_dir"):
        return fdl.Config.__unflatten__(values, metadata)

    output_dir = _thread_local.output_dir

    if len(values) == 1:
        pickle_path = values[0]
        with open(Path(output_dir) / pickle_path, "rb") as f:
            return pickle_load(f)

    return fdl.Config.__unflatten__(values, metadata)


def _io_path_elements_fn(x):
    try:
        serialization.dump_json(x.__io__)
    except (serialization.UnserializableValueError, AttributeError):
        return (serialization.IdentityElement(),)

    return x.__io__.__path_elements__()


def _artifact_transform_save(cfg: fdl.Config, output_path: Path, relative_dir: Path = "."):
    for artifact in getattr(cfg.__fn_or_cls__, "__io_artifacts__", []):
        # Allow optional artifacts
        if artifact.skip or (not hasattr(cfg, artifact.attr) and not artifact.required):
            continue

        if not hasattr(cfg, artifact.attr) and artifact.required:
            raise ValueError(f"Artifact '{artifact.attr}' is required but not provided")

        current_val = getattr(cfg, artifact.attr)
        if current_val is None:
            if artifact.required:
                raise ValueError(f"Artifact '{artifact.attr}' is required but not provided")
            continue
        ## dump artifact and return the relative path
        new_val = artifact.dump(current_val, output_path, relative_dir)
        setattr(cfg, artifact.attr, new_val)

    for attr in dir(cfg):
        try:
            if isinstance(getattr(cfg, attr), fdl.Config):
                _artifact_transform_save(getattr(cfg, attr), output_path=output_path, relative_dir=relative_dir)
        except ValueError:
            pass


def _artifact_transform_load(cfg: fdl.Config, path: Path):
    for artifact in getattr(cfg.__fn_or_cls__, "__io_artifacts__", []):
        # We expect an artifact.attr to be a string or a fdl.Config.
        # Some parameteres can be a string or a filepath. When those parameters are just strings,
        # we will represent it with a fdl.Config, and will skip the rest of the loop (base-dir adjustment).
        current_val = getattr(cfg, artifact.attr)
        if isinstance(current_val, fdl.Config):
            # artifact.attr is a string not a path.
            setattr(cfg, artifact.attr, fdl.build(current_val).attr)
            continue

        if artifact.skip:
            continue
        current_val = getattr(cfg, artifact.attr)
        # __init__ arguments can be None
        if current_val is None:
            continue
        ## replace local path with absolute one
        new_val = str(Path(path) / current_val)
        setattr(cfg, artifact.attr, new_val)

    for attr in dir(cfg):
        try:
            if isinstance(getattr(cfg, attr), fdl.Config):
                _artifact_transform_load(getattr(cfg, attr), path=path)
        except ValueError:
            pass


def drop_unexpected_params(config: fdl.Config) -> bool:
    """
    Analyzes config to detect unexpected keyword arguments -- for example, deprecated parameters -- and
    updates the config by dropping them. Returns True if the config gets updated and False otherwise.

    Args:
        config (fdl.Config): The configuration object to analyze.
    """

    updated = False

    def analyze(config: fdl.Config, prefix: str):

        if isinstance(config, fdl.Config):
            signature = inspect.signature(config.__fn_or_cls__)

            accept_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())

            if not accept_kwargs:
                to_drop = [param for param in config.__arguments__ if param not in signature.parameters]

                if to_drop:
                    nonlocal updated
                    updated = True
                    logging.warning(f"Deprecated parameters to drop from {prefix}: {to_drop}")
                    for param in to_drop:
                        del config.__arguments__[param]
            else:
                logging.debug(f"Skip analyzing {prefix} as it accepts arbitrary keyword arguments.")

            # Proceed recursively for all arguments
            for key, value in config.__arguments__.items():
                analyze(value, prefix + "." + key)

    analyze(config, "<root>")

    return updated


def load(path: Path, output_type: Type[CkptType] = Any, subpath: Optional[str] = None, build: bool = True) -> CkptType:
    """
    Loads a configuration from a pickle file and constructs an object of the specified type.

    Args:
        path (Path): The path to the pickle file or directory containing 'io.pkl'.
        output_type (Type[CkptType]): The type of the object to be constructed from the loaded data.
        subpath (Optional[str]): Subpath to selectively load only specific objects inside the output_type.
                                 Defaults to None.

    Returns
    -------
        CkptType: An instance of the specified type constructed from the loaded configuration.

    Raises
    ------
        FileNotFoundError: If the specified file does not exist.

    Example:
        loaded_model = load("/path/to/model", output_type=MyModel)
    """
    _path = Path(path)
    _thread_local.output_dir = _path

    if hasattr(_path, "is_dir") and _path.is_dir():
        _path = Path(_path) / "io.json"
    elif hasattr(_path, "isdir") and _path.isdir:
        _path = Path(_path) / "io.json"

    if not _path.is_file():
        raise FileNotFoundError(f"No such file: '{_path}'")

    if subpath:
        subpath = "<root>." + subpath

    ## add IO functionality to custom objects present in the json file
    with open(_path) as f:
        j = json.load(f)
    for obj, val in j.get("objects", {}).items():
        clss = ".".join([val["type"]["module"], val["type"]["name"]])
        if subpath and "paths" in val:
            if all(map(lambda p: subpath not in p, val["paths"])):
                continue

        if not serialization.find_node_traverser(locate(clss)):
            track_io(locate(clss))

    with open(_path, "rb") as f:
        json_config = json.loads(f.read())

    root_key = None
    for obj, val in json_config.get("objects", {}).items():
        if "paths" in val and subpath in val["paths"]:
            root_key = obj
            break

    if subpath and not root_key:
        logging.warning(f"Could not find {subpath} for {output_type} in {_path}")

    if root_key:
        json_config["root"]["key"] = root_key

    config = serialization.Deserialization(json_config).result
    _artifact_transform_load(config, path)

    drop_unexpected_params(config)

    if not build:
        return config

    return fdl.build(config)
