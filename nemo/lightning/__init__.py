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

from typing import Union

from lightning_fabric.plugins.environments import slurm
from pytorch_lightning import plugins as _pl_plugins

# This is here to import it once, which improves the speed of launch when in debug-mode
from nemo.utils.import_utils import safe_import

safe_import("transformer_engine")

from nemo.lightning.base import get_vocab_size, teardown
from nemo.lightning.fabric.fabric import Fabric
from nemo.lightning.fabric.plugins import FabricMegatronMixedPrecision
from nemo.lightning.fabric.strategies import FabricMegatronStrategy
from nemo.lightning.nemo_logger import NeMoLogger
from nemo.lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from nemo.lightning.pytorch.optim import LRSchedulerModule, MegatronOptimizerModule, OptimizerModule, lr_scheduler
from nemo.lightning.pytorch.plugins import MegatronDataSampler, MegatronMixedPrecision
from nemo.lightning.pytorch.plugins import data_sampler as _data_sampler
from nemo.lightning.pytorch.strategies import FSDPStrategy, MegatronStrategy
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.lightning.pytorch.trainer import Trainer
from nemo.lightning.resume import AutoResume


# We monkey patch because nvidia uses a naming convention for SLURM jobs
def _is_slurm_interactive_mode():
    job_name = slurm.SLURMEnvironment.job_name()
    return job_name is None or job_name.endswith("bash") or job_name.endswith("interactive")


slurm._is_slurm_interactive_mode = _is_slurm_interactive_mode  # noqa: SLF001


_pl_plugins._PLUGIN_INPUT = Union[_pl_plugins._PLUGIN_INPUT, _data_sampler.DataSampler]  # noqa: SLF001


__all__ = [
    "AutoResume",
    "Fabric",
    "FabricMegatronMixedPrecision",
    "FabricMegatronStrategy",
    "LRSchedulerModule",
    "MegatronStrategy",
    "MegatronDataSampler",
    "MegatronMixedPrecision",
    "MegatronOptimizerModule",
    "FSDPStrategy",
    "RestoreConfig",
    "lr_scheduler",
    "NeMoLogger",
    "ModelCheckpoint",
    "OptimizerModule",
    "Trainer",
    "get_vocab_size",
    "teardown",
]
