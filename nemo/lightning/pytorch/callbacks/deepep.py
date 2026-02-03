# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks.callback import Callback
from megatron.core.transformer import TransformerConfig


class DeepEPCallback(Callback):
    """
    A PyTorch Lightning callback to enable DeepEP if the hardware is supported.
    Per official documentation https://github.com/deepseek-ai/DeepEP,
    DeepEP is supported for Ampere (SM80), Hopper (SM90), Blackwell (SM100),
    and newer architecture GPUs.

    Adding this callback is equivalent to setting the following flags in the recipe function:

    recipe.model.config.moe_token_dispatcher_type = "flex"
    recipe.model.config.moe_enable_deepep = True
    recipe.model.config.moe_shared_expert_overlap = False

    Since the recipe function may be run on a different machine, this callback is needed so that
    configs are set during run time.
    """

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Enable DeepEP if GPU is Ampere (SM80) or newer"""
        # SM80 = Ampere (major 8), SM90 = Hopper (major 9), SM100 = Blackwell (major 10), SM120 (major 12)
        if torch.cuda.get_device_properties(0).major < 8:
            return

        if hasattr(trainer.model, "config") and isinstance(trainer.model.config, TransformerConfig):
            self._apply_deepep_cfgs(trainer.model.config)
            if hasattr(trainer.model, '__io__'):
                self._apply_deepep_cfgs(trainer.model.__io__.config)

    def _apply_deepep_cfgs(self, dest_cfg):
        # apply optimizations into dest_cfg
        dest_cfg.moe_token_dispatcher_type = "flex"
        dest_cfg.moe_enable_deepep = True
        dest_cfg.moe_shared_expert_overlap = False
