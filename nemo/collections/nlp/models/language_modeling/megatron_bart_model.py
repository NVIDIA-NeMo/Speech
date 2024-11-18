# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from lightning.pytorch.trainer.trainer import Trainer
from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model

__all__ = ["MegatronBARTModel"]


class MegatronBARTModel(MegatronT5Model):
    """
    Megatron BART pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

    @property
    def model_name(self):
        """Allows child classes to implement models with different data regime"""
        return "BART"

    def _validate_cfg(self):
        """Class-specific cfg validation"""
        if self._cfg.data.get('dataset_type', None) != 'bart':
            raise ValueError(
                f"cfg.data.dataset_type = {self._cfg.data.get('dataset_type', None)} but 'bart' is expected"
            )

        if self.num_sentinel_tokens != 0:
            raise ValueError(
                f"cfg.tokenizer.num_sentinel_tokens = {self.num_sentinel_tokens} but 0 is expected for 'bart'"
            )

    @property
    def _build_train_valid_test_datasets_kwargs(self):
        """allows child classes to add kwargs to dataset building"""
        return dict(
            delete_mask_prob=self._cfg.data.get('delete_mask_prob', 0.0),
        )

    def list_available_models(self):
        pass
