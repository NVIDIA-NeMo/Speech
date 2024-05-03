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

import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron.griffin.griffin_model import GriffinModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

try:

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    TransformerConfig = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False


class MegatronGriffinModel(MegatronGPTModel):
    """
    Megatron Griffin pretraining.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        # build the transformer config
        # TODO: add type hint once pip package is out

        self.vocab_size = cfg.get('vocab_size', 256000)
        self.cfg = cfg
        super().__init__(cfg=cfg, trainer=trainer)
        self.mcore_gpt = True

    def model_provider_func(self, pre_process, post_process):
        model = GriffinModel(
            config=self.transformer_config,
            max_sequence_length=self.cfg.get('encoder_seq_length', 512),
            vocab_size=self.cfg.get('vocab_size', 256000),
            position_embedding_type=self.cfg.get('position_embedding_type', 'rope'),
            logits_soft_cap=self.cfg.get('logits_soft_cap', 30.0),
            rotary_percent=self.cfg.get('rotary_percentage', 0.5),
            rotary_base=self.cfg.get('rotary_base', 10000),
        )

        return model

    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None):

        output_tensor = self.model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, labels=labels
        )
        return output_tensor

    def build_transformer_config(self):
        transformer_config = super().build_transformer_config()
        transformer_config.gated_linear_unit = self.cfg.get('gated_linear_unit', True)
        transformer_config.layernorm_zero_centered_gamma = self.cfg.get('layernorm_zero_centered_gamma', True)

        return transformer_config

    def on_validation_epoch_end(self):

        averaged_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
        return averaged_loss

    def sharded_state_dict(self, prefix: str = ''):
        return None

    def _reset_activation_checkpointing_args(self):
        return

    def _restore_activation_checkpointing_args(self):
        return

    def _reset_sequence_parallelism_args(self):
        return

    def _restore_sequence_parallelism_args(self):
        return
