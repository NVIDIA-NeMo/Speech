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

import argparse
import functools
import os
import sys
from typing import Any, Callable, Optional

from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.core.config_store import ConfigStore
from hydra.types import TaskFunction
from omegaconf import DictConfig, OmegaConf


def _get_gpu_name():
    try:
        import pynvml
    except (ImportError, ModuleNotFoundError):
        return None

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    cuda_capability, _ = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    pynvml.nvmlShutdown()
    if cuda_capability == 8:
        return "a100"
    elif cuda_capability == 9:
        return "h100"
    else:
        return None


OmegaConf.register_new_resolver("gpu_name", _get_gpu_name)

# multiple interpolated values in the config
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)

# sum interpolated values in the config
OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)


def hydra_runner(
    config_path: Optional[str] = ".", config_name: Optional[str] = None, schema: Optional[Any] = None
) -> Callable[[TaskFunction], Any]:
    """
    Decorator used for passing the Config paths to main function.
    Optionally registers a schema used for validation/providing default values.

    Args:
        config_path: Optional path that will be added to config search directory.
            NOTE: The default value of `config_path` has changed between Hydra 1.0 and Hydra 1.1+.
            Please refer to https://hydra.cc/docs/next/upgrades/1.0_to_1.1/changes_to_hydra_main_config_path/
            for details.
        config_name: Pathname of the config file.
        schema: Structured config  type representing the schema used for validation/providing default values.
    """

    def decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def wrapper(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            # Check it config was passed.
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args = get_args_parser()

                # Parse arguments in order to retrieve overrides
                parsed_args = args.parse_args()  # type: argparse.Namespace

                # Get overriding args in dot string format
                overrides = parsed_args.overrides  # type: list

                # Disable the creation of .hydra subdir
                # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory
                overrides.append("hydra.output_subdir=null")
                # Hydra logging outputs only to stdout (no log file).
                # https://hydra.cc/docs/configure_hydra/logging
                overrides.append("hydra/job_logging=stdout")

                # Set run.dir ONLY for ExpManager "compatibility" - to be removed.
                overrides.append("hydra.run.dir=.")

                # Check if user set the schema.
                if schema is not None:
                    # Create config store.
                    cs = ConfigStore.instance()

                    # Get the correct ConfigStore "path name" to "inject" the schema.
                    if parsed_args.config_name is not None:
                        path, name = os.path.split(parsed_args.config_name)
                        # Make sure the path is not set - as this will disable validation scheme.
                        if path != '':
                            sys.stderr.write(
                                f"ERROR Cannot set config file path using `--config-name` when "
                                "using schema. Please set path using `--config-path` and file name using "
                                "`--config-name` separately.\n"
                            )
                            sys.exit(1)
                    else:
                        name = config_name

                    # Register the configuration as a node under the name in the group.
                    cs.store(name=name, node=schema)  # group=group,

                # Wrap a callable object with name `parse_args`
                # This is to mimic the ArgParser.parse_args() API.
                def parse_args(self, args=None, namespace=None):
                    return parsed_args

                parsed_args.parse_args = parse_args

                # no return value from run_hydra() as it may sometime actually run the task_function
                # multiple times (--multirun)
                # argparse_wrapper = _argparse_wrapper(args)
                argparse_wrapper = parsed_args

                _run_hydra(
                    args=argparse_wrapper,
                    args_parser=args,
                    task_function=task_function,
                    config_path=config_path,
                    config_name=config_name,
                )

        return wrapper

    return decorator
