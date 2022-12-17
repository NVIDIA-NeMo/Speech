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

from pathlib import Path

from pytorch_lightning.loggers import Logger

from nemo.utils import logging

try:
    import dllogger
    from dllogger import Verbosity

    HAVE_DLLOGGER = True
except (ImportError, ModuleNotFoundError):
    HAVE_DLLOGGER = False


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

    def log_hyperparams(self, params, *args, **kwargs):
        dllogger.log(step="PARAMETER", data=params)

    def log_metrics(self, metrics, step=None):
        if step is None:
            step = tuple()

        dllogger.log(step=step, data=metrics)

    def save(self):
        dllogger.flush()
