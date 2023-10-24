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

from dataclasses import dataclass
from typing import Any, Optional

from hydra.core.config_store import ConfigStore

__all__ = ['TrainerConfig']


cs = ConfigStore.instance()


@dataclass
class TrainerConfig:
    """
    Configuration of PyTorch Lightning Trainer.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    ..warning:
        Picked just few params of the PTL trainer for now. This needs to be discussed.
    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
    """

    logger: Any = True
    callbacks: Optional[Any] = None
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    num_nodes: int = 1
    enable_progress_bar: bool = True
    overfit_batches: Any = 0.0
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False
    accumulate_grad_batches: Any = 1
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = -1
    min_steps: Optional[int] = None
    limit_train_batches: Any = 1.0
    limit_val_batches: Any = 1.0
    limit_test_batches: Any = 1.0
    val_check_interval: Any = 1.0
    log_every_n_steps: int = 50
    accelerator: Optional[str] = 'auto'
    sync_batchnorm: bool = False
    precision: Any = 32
    num_sanity_val_steps: int = 2
    profiler: Optional[Any] = None
    benchmark: bool = False
    deterministic: bool = False
    use_distributed_sampler: bool = True
    detect_anomaly: bool = False
    plugins: Optional[Any] = None  # Optional[Union[str, list]]
    limit_predict_batches: float = 1.0
    gradient_clip_algorithm: str = 'norm'
    max_time: Optional[Any] = None  # can be one of Union[str, timedelta, Dict[str, int], None]
    reload_dataloaders_every_n_epochs: int = 0
    devices: Any = 'auto'
    strategy: Any = 'auto'
    enable_checkpointing: bool = False
    enable_model_summary: bool = True
    inference_mode: bool = True
    barebones: bool = False


# Register the trainer config.
cs.store(
    group="trainer", name="trainer", node=TrainerConfig,
)
