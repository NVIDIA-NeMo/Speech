# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Union

import pandas as pd
from lightning_utilities.core.apply_func import apply_to_collection
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.parsing import AttributeDict
from torch import Tensor

from nemo.utils import logging

try:
    from clearml import OutputModel, Task

    HAVE_CLEARML_LOGGER = True
except (ImportError, ModuleNotFoundError):
    HAVE_CLEARML_LOGGER = False


@dataclass
class ClearMLParams:
    project: Optional[str] = None
    task: Optional[str] = None
    connect_pytorch: Optional[bool] = False
    model_name: Optional[str] = None
    tags: Optional[List[str]] = None
    log_model: Optional[bool] = False
    log_cfg: Optional[bool] = False
    log_metrics: Optional[bool] = False


class ClearMLLogger(Logger):
    @property
    def name(self) -> str:
        return self.clearml_task.name

    @property
    def version(self) -> str:
        return self.clearml_task.id

    def __init__(
        self, clearml_cfg: DictConfig, log_dir: str, prefix: str, save_best_model: bool, postfix: str = ".nemo"
    ) -> None:
        if not HAVE_CLEARML_LOGGER:
            raise ImportError(
                "Found create_clearml_logger is True."
                "But ClearML not found. Please see the README for installation instructions:"
                "https://github.com/allegroai/clearml"
            )

        self.clearml_task = None
        self.clearml_model = None
        self.clearml_cfg = clearml_cfg
        self.path_nemo_model = os.path.abspath(
            os.path.expanduser(os.path.join(log_dir, "checkpoints", prefix + postfix))
        )
        self.save_best_model = save_best_model
        self.prefix = prefix
        self.previos_best_model_path = None
        self.last_metrics = None
        self.save_blocked = True

        self.project_name = os.getenv("CLEARML_PROJECT", clearml_cfg.project if clearml_cfg.project else "NeMo")
        self.task_name = os.getenv("CLEARML_TASK", clearml_cfg.task if clearml_cfg.task else f"Trainer {self.prefix}")

        tags = ["NeMo"]
        if clearml_cfg.tags:
            tags.extend(clearml_cfg.tags)

        self.clearml_task: Task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            auto_connect_frameworks={"pytorch": clearml_cfg.connect_pytorch},
            output_uri=True,
            tags=tags,
        )

        if clearml_cfg.model_name:
            model_name = clearml_cfg.model_name
        elif self.prefix:
            model_name = self.prefix
        else:
            model_name = self.task_name

        if clearml_cfg.log_model:
            self.clearml_model: OutputModel = OutputModel(
                name=model_name, task=self.clearml_task, tags=tags, framework="NeMo"
            )

    def log_hyperparams(self, params, *args, **kwargs) -> None:
        if self.clearml_model and self.clearml_cfg.log_cfg:
            if isinstance(params, Namespace):
                params = vars(params)
            elif isinstance(params, AttributeDict):
                params = dict(params)
            params = apply_to_collection(params, (DictConfig, ListConfig), OmegaConf.to_container, resolve=True)
            params = apply_to_collection(params, Path, str)
            params = OmegaConf.to_yaml(params)
            self.clearml_model.update_design(config_text=params)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        if self.clearml_model and self.clearml_cfg.log_metrics:
            metrics = {
                k: {
                    "value": str(v.item() if type(v) == Tensor else v),
                    "type": str(type(v.item() if type(v) == Tensor else v)),
                }
                for k, v in metrics.items()
            }
            self.last_metrics = metrics

    def log_table(
        self,
        key: str,
        columns: List[str] = None,
        data: List[List[Any]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
    ) -> None:
        table: Optional[Union[pd.DataFrame, List[List[Any]]]] = None

        if dataframe is not None:
            table = dataframe
            if columns is not None:
                table.columns = columns

        if data is not None:
            table = data
            assert len(columns) == len(table[0]), "number of column names should match the total number of columns"
            table.insert(0, columns)

        if table is not None:
            self.clearml_task.logger.report_table(title=key, series=key, iteration=step, table_plot=table)

    def after_save_checkpoint(self, checkpoint_callback: Checkpoint) -> None:
        if self.clearml_model:
            if self.save_best_model:
                if self.save_blocked:
                    self.save_blocked = False
                    return None
                if not os.path.exists(checkpoint_callback.best_model_path):
                    return None
                if self.previos_best_model_path == checkpoint_callback.best_model_path:
                    return None
                self.previos_best_model_path = checkpoint_callback.best_model_path
            self._log_model(self.path_nemo_model)

    def finalize(self, status: Literal["success", "failed", "aborted"] = "success") -> None:
        if status == "success":
            self.clearml_task.mark_completed()
        elif status == "failed":
            self.clearml_task.mark_failed()
        elif status == "aborted":
            self.clearml_task.mark_stopped()

    def _log_model(self, save_path: str) -> None:
        if self.clearml_model:
            if os.path.exists(save_path):
                self.clearml_model.update_weights(
                    weights_filename=save_path,
                    upload_uri=self.clearml_task.storage_uri or self.clearml_task._get_default_report_storage_uri(),
                    auto_delete_file=False,
                    is_package=True,
                )

                if self.clearml_cfg.log_metrics and self.last_metrics:
                    self.clearml_model.set_all_metadata(self.last_metrics)

                self.save_blocked = True
            else:
                logging.warning((f"Logging model enabled, but cant find .nemo file!" f" Path: {save_path}"))
