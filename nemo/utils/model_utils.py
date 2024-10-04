# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import copy
import fnmatch
import importlib
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, is_dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import wrapt

from nemo.utils import AppState, logging
from nemo.utils.data_utils import resolve_cache_dir  # imported for compatibility: model_utils.resolve_cache_dir()
from nemo.utils.data_utils import is_datastore_path

# TODO @blisc: Perhaps refactor instead of import guarding

_HAS_HYDRA = True

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
    from omegaconf import errors as omegaconf_errors
    from packaging import version
except ModuleNotFoundError:
    _HAS_HYDRA = False


MODEL_CONFIG = "model_config.yaml"
_VAL_TEST_FASTPATH_KEY = 'ds_item'


class ArtifactPathType(Enum):
    """
    ArtifactPathType refers to the type of the path that the artifact is located at.

    LOCAL_PATH: A user local filepath that exists on the file system.
    TAR_PATH: A (generally flattened) filepath that exists inside of an archive (that may have its own full path).
    """

    LOCAL_PATH = 0
    TAR_PATH = 1


@dataclass
class ArtifactItem:
    path: str = ""
    path_type: ArtifactPathType = ArtifactPathType.LOCAL_PATH
    hashed_path: Optional[str] = None


def detect_prefix(names: List[str]) -> str:
    """Detect model config prefix for a list of file names.

    Useful to identify prefix used within .nemo tarball checkpoint."""
    model_config = fnmatch.filter(names, f"*{MODEL_CONFIG}")
    assert len(model_config) == 1, f"Exactly one model config path expected, found: {model_config}."
    prefix = model_config[0].removesuffix(MODEL_CONFIG)
    return prefix


def load_config(model_file: str) -> DictConfig:
    """Load model config from extracted directory or '.nemo' tarball."""
    if os.path.isfile(model_file):
        with tempfile.TemporaryDirectory() as tmp, tarfile.open(model_file, "r:") as tar:
            prefix = detect_prefix(tar.getnames())
            tar.extract(f"{prefix}{MODEL_CONFIG}", path=tmp)
            model_config = OmegaConf.load(os.path.join(tmp, MODEL_CONFIG))
    elif os.path.isdir(model_file):
        model_config = OmegaConf.load(os.path.join(model_file, MODEL_CONFIG))
    else:
        raise FileNotFoundError(model_file)

    return model_config


def unwrap_model(model, module_instances: Union[Type, Tuple[Type]]):
    """Unwrap model from wrapper classes like Float16Module, for example."""

    # TODO: Import this from megatron.core once moved there from megatron.training.
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared


def resolve_dataset_name_from_cfg(cfg: 'DictConfig') -> Optional[str]:
    """
    Parses items of the provided sub-config to find the first potential key that
    resolves to an existing file or directory.

    # Fast-path Resolution
    In order to handle cases where we need to resolve items that are not paths, a fastpath
    key can be provided as defined in the global `_VAL_TEST_FASTPATH_KEY`.

    This key can be used in two ways :

    ## _VAL_TEST_FASTPATH_KEY points to another key in the config

    If this _VAL_TEST_FASTPATH_KEY points to another key in this config itself,
    then we assume we want to loop through the values of that key.

    This allows for any key in the config to become a fastpath key.

    Example:
    validation_ds:
        splits: "val"
        ...
        <_VAL_TEST_FASTPATH_KEY>: "splits"  <-- this points to the key name "splits"

    Then we can write the following when overriding in hydra:
    ```python
    python train_file.py ... \
        model.validation_ds.splits=[val1, val2, dev1, dev2] ...
    ```

    ## _VAL_TEST_FASTPATH_KEY itself acts as the resolved key

    If this _VAL_TEST_FASTPATH_KEY does not point to another key in the config, then
    it is assumed that the items of this key itself are used for resolution.

    Example:
    validation_ds:
        ...
        <_VAL_TEST_FASTPATH_KEY>: "val"  <-- this points to the key name "splits"

    Then we can write the following when overriding in hydra:
    ```python
    python train_file.py ... \
        model.validation_ds.<_VAL_TEST_FASTPATH_KEY>=[val1, val2, dev1, dev2] ...
    ```

    # IMPORTANT NOTE:
    It <can> potentially mismatch if there exist more than 2 valid paths, and the
    first path does *not* resolve the the path of the data file (but does resolve to
    some other valid path).

    To avoid this side-effect, place the data path as the first item on the config file.

    Args:
        cfg: DictConfig (Sub-config) that should be parsed.

    Returns:
        A str representing the `key` of the config which hosts the filepath(s),
        or None in case path could not be resolved.
    """
    if _VAL_TEST_FASTPATH_KEY in cfg and cfg[_VAL_TEST_FASTPATH_KEY] is not None:
        fastpath_key = cfg[_VAL_TEST_FASTPATH_KEY]

        if isinstance(fastpath_key, str) and fastpath_key in cfg:
            return cfg[fastpath_key]
        else:
            return _VAL_TEST_FASTPATH_KEY

    for key, value in cfg.items():
        if type(value) in [list, tuple, ListConfig]:
            # Count the number of valid paths in the list
            values_are_paths = 0
            for val_i in value:
                val_i = str(val_i)
                if os.path.exists(val_i) or os.path.isdir(val_i) or is_datastore_path(val_i):
                    values_are_paths += 1
                else:
                    # reset counter and break inner loop
                    break

            if values_are_paths == len(value):
                return key

        else:
            if os.path.exists(str(value)) or os.path.isdir(str(value)) or is_datastore_path(str(value)):
                return key

    return None


def parse_dataset_as_name(name: str) -> str:
    """
    Constructs a valid prefix-name from a provided file path.

    Args:
        name: str path to some valid data/manifest file or a python object that
            will be used as a name for the data loader (via str() cast).

    Returns:
        str prefix used to identify uniquely this data/manifest file.
    """
    if os.path.exists(str(name)) or os.path.isdir(str(name)) or is_datastore_path(str(name)):
        name = Path(name).stem
    else:
        name = str(name)

    # cleanup name
    name = name.replace('-', '_')

    if 'manifest' in name:
        name = name.replace('manifest', '')

    if 'dataset' in name:
        name = name.replace('dataset', '')

    # Test if the manifes/dataset name was simply `manifest.yaml` or `dataset.yaml`: Invalid names.
    if name == '':
        raise ValueError(
            "Provided dataset / manifest filename was `manifest.json` or `dataset.json`.\n"
            "Such a name is invalid, since multiple datasets/manifests can share the same name,\n"
            "thereby overriding their results during logging. Please pick a more discriptive filename \n"
            "for the provided dataset / manifest file."
        )

    if '_' != name[-1]:
        name = name + '_'

    return name


def unique_names_check(name_list: Optional[List[str]]):
    """
    Performs a uniqueness check on the name list resolved, so that it can warn users
    about non-unique keys.

    Args:
        name_list: List of strings resolved for data loaders.
    """
    if name_list is None:
        return

    # Name uniqueness checks
    names = set()
    for name in name_list:
        if name in names:
            logging.warning(
                "Name resolution has found more than one data loader having the same name !\n"
                "In such cases, logs will nor be properly generated. "
                "Please rename the item to have unique names.\n"
                f"Resolved name : {name}"
            )
        else:
            names.add(name)  # we need just hash key check, value is just a placeholder


def resolve_validation_dataloaders(model: 'ModelPT'):
    """
    Helper method that operates on the ModelPT class to automatically support
    multiple dataloaders for the validation set.

    It does so by first resolving the path to one/more data files via `resolve_dataset_name_from_cfg()`.
    If this resolution fails, it assumes the data loader is prepared to manually support / not support
    multiple data loaders and simply calls the appropriate setup method.

    If resolution succeeds:
        Checks if provided path is to a single file or a list of files.
        If a single file is provided, simply tags that file as such and loads it via the setup method.
        If multiple files are provided:
            Inject a new manifest path at index "i" into the resolved key.
            Calls the appropriate setup method to set the data loader.
            Collects the initialized data loader in a list and preserves it.
            Once all data loaders are processed, assigns the list of loaded loaders to the ModelPT.
            Finally assigns a list of unique names resolved from the file paths to the ModelPT.

    Args:
        model: ModelPT subclass, which requires >=1 Validation Dataloaders to be setup.
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/Omegaconf and it was not installed.")
        exit(1)
    cfg = copy.deepcopy(model._cfg)
    dataloaders = []

    # process val_loss_idx
    if 'val_dl_idx' in cfg.validation_ds:
        cfg = OmegaConf.to_container(cfg)
        val_dl_idx = cfg['validation_ds'].pop('val_dl_idx')
        cfg = OmegaConf.create(cfg)
    else:
        val_dl_idx = 0

    # Set val_loss_idx
    model._val_dl_idx = val_dl_idx

    ds_key = resolve_dataset_name_from_cfg(cfg.validation_ds)

    if ds_key is None or val_dl_idx < 0:
        logging.debug(
            "Could not resolve file path from provided config - {}. "
            "Disabling support for multi-dataloaders.".format(cfg.validation_ds)
        )

        model.setup_validation_data(cfg.validation_ds)
        return

    ds_values = cfg.validation_ds[ds_key]

    if isinstance(ds_values, (list, tuple, ListConfig)):

        for ds_value in ds_values:
            if isinstance(ds_value, (dict, DictConfig)):
                # this is a nested dataset
                cfg.validation_ds = ds_value
            else:
                cfg.validation_ds[ds_key] = ds_value

            model.setup_validation_data(cfg.validation_ds)
            dataloaders.append(model._validation_dl)

        model._validation_dl = dataloaders
        if len(ds_values) > 0 and isinstance(ds_values[0], (dict, DictConfig)):
            # using the name of each of the nested dataset
            model._validation_names = [ds.name for ds in ds_values]
        else:
            ds_names = cfg.validation_ds.get('name', [])
            if len(ds_names) > 0:
                if len(ds_names) != len(ds_values):
                    raise ValueError(
                        f"Number of names ({len(ds_names)}) does not match number of datasets ({len(ds_values)}). Got {ds_names} and {ds_values}"
                    )
                model._validation_names = [parse_dataset_as_name(n) for n in ds_names]
            else:
                model._validation_names = [parse_dataset_as_name(ds) for ds in ds_values]
        unique_names_check(name_list=model._validation_names)

        return

    else:
        model.setup_validation_data(cfg.validation_ds)
        ds_names = cfg.validation_ds.get('name', None)
        if ds_names is not None:
            if not isinstance(ds_names, str):
                raise ValueError(f"`name` must be a string for single manifest, got {ds_names}")
            model._validation_names = [parse_dataset_as_name(ds_names)]
        else:
            model._validation_names = [parse_dataset_as_name(ds_values)]
        unique_names_check(name_list=model._validation_names)


def resolve_test_dataloaders(model: 'ModelPT'):
    """
    Helper method that operates on the ModelPT class to automatically support
    multiple dataloaders for the test set.

    It does so by first resolving the path to one/more data files via `resolve_dataset_name_from_cfg()`.
    If this resolution fails, it assumes the data loader is prepared to manually support / not support
    multiple data loaders and simply calls the appropriate setup method.

    If resolution succeeds:
        Checks if provided path is to a single file or a list of files.
        If a single file is provided, simply tags that file as such and loads it via the setup method.
        If multiple files are provided:
            Inject a new manifest path at index "i" into the resolved key.
            Calls the appropriate setup method to set the data loader.
            Collects the initialized data loader in a list and preserves it.
            Once all data loaders are processed, assigns the list of loaded loaders to the ModelPT.
            Finally assigns a list of unique names resolved from the file paths to the ModelPT.

    Args:
        model: ModelPT subclass, which requires >=1 Test Dataloaders to be setup.
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/Omegaconf and it was not installed.")
        exit(1)
    cfg = copy.deepcopy(model._cfg)
    dataloaders = []

    # process test_loss_idx
    if 'test_dl_idx' in cfg.test_ds:
        cfg = OmegaConf.to_container(cfg)
        test_dl_idx = cfg['test_ds'].pop('test_dl_idx')
        cfg = OmegaConf.create(cfg)
    else:
        test_dl_idx = 0

    # Set val_loss_idx
    model._test_dl_idx = test_dl_idx

    ds_key = resolve_dataset_name_from_cfg(cfg.test_ds)

    if ds_key is None:
        logging.debug(
            "Could not resolve file path from provided config - {}. "
            "Disabling support for multi-dataloaders.".format(cfg.test_ds)
        )

        model.setup_test_data(cfg.test_ds)
        return

    ds_values = cfg.test_ds[ds_key]

    if isinstance(ds_values, (list, tuple, ListConfig)):

        for ds_value in ds_values:
            if isinstance(ds_value, (dict, DictConfig)):
                # this is a nested dataset
                cfg.test_ds = ds_value
            else:
                cfg.test_ds[ds_key] = ds_value

            model.setup_test_data(cfg.test_ds)
            dataloaders.append(model._test_dl)

        model._test_dl = dataloaders
        if len(ds_values) > 0 and isinstance(ds_values[0], (dict, DictConfig)):
            # using the name of each of the nested dataset
            model._test_names = [ds.name for ds in ds_values]
        else:
            ds_names = cfg.test_ds.get('name', [])
            if len(ds_names) > 0:
                if len(ds_names) != len(ds_values):
                    raise ValueError(
                        f"Number of names ({len(ds_names)}) does not match number of datasets ({len(ds_values)}). Got {ds_names} and {ds_values}"
                    )
                model._test_names = [parse_dataset_as_name(n) for n in ds_names]
            else:
                model._test_names = [parse_dataset_as_name(ds) for ds in ds_values]

        unique_names_check(name_list=model._test_names)
        return

    else:
        model.setup_test_data(cfg.test_ds)
        ds_names = cfg.test_ds.get('name', None)
        if ds_names is not None:
            if not isinstance(ds_names, str):
                raise ValueError(f"`name` must be a string for single manifest, got {ds_names}")
            model._test_names = [parse_dataset_as_name(ds_names)]
        else:
            model._test_names = [parse_dataset_as_name(ds_values)]

        unique_names_check(name_list=model._test_names)


@wrapt.decorator
def wrap_training_step(wrapped, instance: 'pl.LightningModule', args, kwargs):
    output_dict = wrapped(*args, **kwargs)

    if isinstance(output_dict, dict) and output_dict is not None and 'log' in output_dict:
        log_dict = output_dict.pop('log')
        instance.log_dict(log_dict, on_step=True)

    return output_dict


def convert_model_config_to_dict_config(cfg: Union['DictConfig', 'NemoConfig']) -> 'DictConfig':
    """
    Converts its input into a standard DictConfig.
    Possible input values are:
    -   DictConfig
    -   A dataclass which is a subclass of NemoConfig

    Args:
        cfg: A dict-like object.

    Returns:
        The equivalent DictConfig
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/Omegaconf and it was not installed.")
        exit(1)
    if not isinstance(cfg, (OmegaConf, DictConfig)) and is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if not isinstance(cfg, DictConfig):
        raise ValueError(f"cfg constructor argument must be of type DictConfig/dict but got {type(cfg)} instead.")

    config = OmegaConf.to_container(cfg, resolve=True)
    config = OmegaConf.create(config)
    return config


def _convert_config(cfg: 'OmegaConf'):
    """Recursive function convertint the configuration from old hydra format to the new one."""
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/Omegaconf and it was not installed.")
        exit(1)

    # Get rid of cls -> _target_.
    if 'cls' in cfg and '_target_' not in cfg:
        cfg._target_ = cfg.pop('cls')

    # Get rid of params.
    if 'params' in cfg:
        params = cfg.pop('params')
        for param_key, param_val in params.items():
            cfg[param_key] = param_val

    # Recursion.
    try:
        for _, sub_cfg in cfg.items():
            if isinstance(sub_cfg, DictConfig):
                _convert_config(sub_cfg)
    except omegaconf_errors.OmegaConfBaseException as e:
        logging.warning(f"Skipped conversion for config/subconfig:\n{cfg}\n Reason: {e}.")


def maybe_update_config_version(cfg: 'DictConfig'):
    """
    Recursively convert Hydra 0.x configs to Hydra 1.x configs.

    Changes include:
    -   `cls` -> `_target_`.
    -   `params` -> drop params and shift all arguments to parent.
    -   `target` -> `_target_` cannot be performed due to ModelPT injecting `target` inside class.

    Args:
        cfg: Any Hydra compatible DictConfig

    Returns:
        An updated DictConfig that conforms to Hydra 1.x format.
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/Omegaconf and it was not installed.")
        exit(1)
    if cfg is not None and not isinstance(cfg, DictConfig):
        try:
            temp_cfg = OmegaConf.create(cfg)
            cfg = temp_cfg
        except omegaconf_errors.OmegaConfBaseException:
            # Cannot be cast to DictConfig, skip updating.
            return cfg

    # Make a copy of model config.
    cfg = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg, False)

    # Convert config.
    _convert_config(cfg)

    # Update model config.
    OmegaConf.set_struct(cfg, True)

    return cfg


@lru_cache(maxsize=1024)
def import_class_by_path(path: str):
    """
    Recursive import of class by path string.
    """
    paths = path.split('.')
    path = ".".join(paths[:-1])
    class_name = paths[-1]
    mod = __import__(path, fromlist=[class_name])
    mod = getattr(mod, class_name)
    return mod


def resolve_subclass_pretrained_model_info(base_class) -> List['PretrainedModelInfo']:
    """
    Recursively traverses the inheritance graph of subclasses to extract all pretrained model info.
    First constructs a set of unique pretrained model info by performing DFS over the inheritance graph.
    All model info belonging to the same class is added together.

    Args:
        base_class: The root class, whose subclass graph will be traversed.

    Returns:
        A list of unique pretrained model infos belonging to all of the inherited subclasses of
        this baseclass.
    """
    list_of_models = set()

    def recursive_subclass_walk(cls):
        for subclass in cls.__subclasses__():
            # step into its immediate subclass
            recursive_subclass_walk(subclass)

            subclass_models = subclass.list_available_models()

            if subclass_models is not None and len(subclass_models) > 0:
                # Inject subclass info into pretrained model info
                # if not already overriden by subclass
                for model_info in subclass_models:
                    # If subclass manually injects class_, dont override.
                    if model_info.class_ is None:
                        model_info.class_ = subclass

                for model_info in subclass_models:
                    list_of_models.add(model_info)

    recursive_subclass_walk(base_class)

    list_of_models = list(sorted(list_of_models))
    return list_of_models


def check_lib_version(lib_name: str, checked_version: str, operator) -> Tuple[Optional[bool], str]:
    """
    Checks if a library is installed, and if it is, checks the operator(lib.__version__, checked_version) as a result.
    This bool result along with a string analysis of result is returned.

    If the library is not installed at all, then returns None instead, along with a string explaining
    that the library is not installed

    Args:
        lib_name: lower case str name of the library that must be imported.
        checked_version: semver string that is compared against lib.__version__.
        operator: binary callable function func(a, b) -> bool; that compares lib.__version__ against version in
            some manner. Must return a boolean.

    Returns:
        A tuple of results:
        -   Bool or None. Bool if the library could be imported, and the result of
            operator(lib.__version__, checked_version) or False if __version__ is not implemented in lib.
            None is passed if the library is not installed at all.
        -   A string analysis of the check.
    """
    try:
        if '.' in lib_name:
            mod = import_class_by_path(lib_name)
        else:
            mod = importlib.import_module(lib_name)

        if hasattr(mod, '__version__'):
            lib_ver = version.Version(mod.__version__)
            match_ver = version.Version(checked_version)

            if operator(lib_ver, match_ver):
                msg = f"Lib {lib_name} version is satisfied !"
                return True, msg
            else:
                msg = (
                    f"Lib {lib_name} version ({lib_ver}) is not {operator.__name__} than required version {checked_version}.\n"
                    f"Please upgrade the lib using either pip or conda to the latest version."
                )
                return False, msg
        else:
            msg = (
                f"Lib {lib_name} does not implement __version__ in its init file. "
                f"Could not check version compatibility."
            )
            return False, msg
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass

    msg = f"Lib {lib_name} has not been installed. Please use pip or conda to install this package."
    return None, msg


def uninject_model_parallel_rank(filepath):
    filepath = str(filepath)
    if any([s for s in ['mp_rank', 'tp_rank', 'fsdp_shard'] if s in filepath]):
        dirname = os.path.dirname(os.path.dirname(filepath))
        basename = os.path.basename(filepath)
        filepath = os.path.join(dirname, basename)
        return filepath
    else:
        return filepath


def inject_model_parallel_rank(filepath, fsdp_sharded_ckpt=False):
    """
    Injects tensor/pipeline model parallel ranks into the filepath.
    Does nothing if not using model parallelism.
    """
    # first make sure filepath does not have rank
    filepath = uninject_model_parallel_rank(filepath)

    app_state = AppState()
    dirname = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
        fsdp_shard = f'_fsdp_shard_{app_state.data_parallel_rank:05d}' if fsdp_sharded_ckpt else ''
        if app_state.pipeline_model_parallel_size is None or app_state.pipeline_model_parallel_size == 1:
            filepath = f'{dirname}/mp_rank_{app_state.tensor_model_parallel_rank:02d}{fsdp_shard}/{basename}'
        else:
            filepath = f'{dirname}/tp_rank_{app_state.tensor_model_parallel_rank:02d}_pp_rank_{app_state.pipeline_model_parallel_rank:03d}/{basename}'
        return filepath
    else:
        fsdp_shard = f'/fsdp_shard_{app_state.data_parallel_rank:05d}' if fsdp_sharded_ckpt else ''
        return f'{dirname}{fsdp_shard}/{basename}'


def ckpt_to_dir(filepath: Union[str, Path]) -> Path:
    """PTL considers checkpoints as .ckpt files.
    This method removes the extension and returns a path
    to be used as a directory for distributed checkpoints
    """

    filepath = Path(filepath)

    # if it is already a distributed checkpoint, then return
    if filepath.suffix != ".ckpt" and filepath.is_dir():
        return filepath

    # adding this assert because we will later remove directories based on the return value of this method
    assert filepath.suffix == ".ckpt", f'filepath: {filepath} must have .ckpt extension'

    # create a new path whose name is the original filepath without the .ckpt extension
    checkpoint_dir = filepath.with_name(filepath.stem)

    return checkpoint_dir


def save_artifacts(model, output_dir: str, use_abspath: bool = False) -> None:
    """Save all model artifacts and tokenizer config to a given output directory."""
    app_state = AppState()
    model_file = app_state.model_restore_path
    model_cfg = copy.deepcopy(model.cfg)
    if not hasattr(model, "artifacts"):
        if hasattr(model_cfg, "tokenizer"):
            OmegaConf.save(model_cfg.tokenizer, os.path.join(output_dir, "tokenizer_config.yaml"))
        return

    # Setup model file handling context: directory or tarball
    if os.path.isfile(model_file):
        model_file_handler = tarfile.open
        kwargs = {"name": model_file, "mode": "r:"}
    elif os.path.isdir(model_file):
        model_file_handler = contextlib.nullcontext
        kwargs = {}
    else:
        raise FileNotFoundError(model_file)

    # Copy or extract artifacts depending on the context
    with model_file_handler(**kwargs) as maybe_tar:
        if maybe_tar is not None:
            prefix = detect_prefix(maybe_tar.getnames())
        for arti_name, arti_item in model.artifacts.items():
            _, arti_file = arti_item.path.split("nemo:")
            arti_path = os.path.join(output_dir, arti_name)
            if maybe_tar is not None:
                maybe_tar.extract(f"{prefix}{arti_file}", path=output_dir)
                os.rename(os.path.join(output_dir, arti_file), arti_path)
            else:
                shutil.copy(os.path.join(model_file, arti_file), arti_path)
            # Store artifact path as basename by default. Otherwise save absolute path but bear in mind
            # that in this case output directory should be permanent for correct artifact recovery later
            arti_path = os.path.abspath(arti_path) if use_abspath else os.path.basename(arti_path)
            OmegaConf.update(model_cfg, arti_name, arti_path)

    if hasattr(model_cfg, "tokenizer"):
        OmegaConf.save(model_cfg.tokenizer, os.path.join(output_dir, "tokenizer_config.yaml"))
