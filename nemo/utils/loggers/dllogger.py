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

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lightning_utilities.core.apply_func import apply_to_collection
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.parsing import AttributeDict

from nemo.utils import logging

try:
    import dllogger
    from dllogger import Verbosity

    HAVE_DLLOGGER = True
except (ImportError, ModuleNotFoundError):
    HAVE_DLLOGGER = False

try:
    from lightning_fabric.utilities.logger import _convert_params, _flatten_dict, _sanitize_callable_params

    PL_LOGGER_UTILITIES = True
except (ImportError, ModuleNotFoundError):
    PL_LOGGER_UTILITIES = False


@dataclass
class DLLoggerParams:
    verbose: Optional[bool] = False
    stdout: Optional[bool] = False
    json_file: Optional[str] = "./dllogger.json"


class DLLogger(Logger):
    @property
    def name(self):
        return self.__class__.__name__

    @property
    def version(self):
        return None

    def __init__(self, stdout: bool, verbose: bool, json_file: str):
        if not HAVE_DLLOGGER:
            raise ImportError(
                "DLLogger was not found. Please see the README for installation instructions: "
                "https://github.com/NVIDIA/dllogger"
            )
        if not PL_LOGGER_UTILITIES:
            raise ImportError(
                "DLLogger utilities were not found. You probably need to update PyTorch Lightning>=1.9.0. "
                "pip install pytorch-lightning -U"
            )
        verbosity = Verbosity.VERBOSE if verbose else Verbosity.DEFAULT
        backends = []
        if json_file:
            Path(json_file).parent.mkdir(parents=True, exist_ok=True)
            backends.append(dllogger.JSONStreamBackend(verbosity, json_file))
        if stdout:
            backends.append(dllogger.StdOutBackend(verbosity))

        if not backends:
            logging.warning(
                "Neither stdout nor json_file DLLogger parameters were specified." "DLLogger will not log anything."
            )
        dllogger.init(backends=backends)

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        if isinstance(params, Namespace):
            params = vars(params)
        elif isinstance(params, AttributeDict):
            params = dict(params)
        params = apply_to_collection(params, (DictConfig, ListConfig), OmegaConf.to_container, resolve=True)
        params = apply_to_collection(params, Path, str)
        params = _sanitize_callable_params(_flatten_dict(_convert_params(params)))
        dllogger.log(step="PARAMETER", data=params)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if step is None:
            step = tuple()

        dllogger.log(step=step, data=metrics)

    def save(self):
        dllogger.flush()
