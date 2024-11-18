# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import Optional

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.data.question_answering.data_processor.qa_processing import (
    EVALUATION_MODE,
    INFERENCE_MODE,
    TRAINING_MODE,
)
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.utils import logging
from nemo.utils.decorators import deprecated_warning


class BaseQAModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None, no_lm_init=True):
        # deprecation warning
        deprecated_warning("BaseQAModel")

        self.cfg = cfg
        super().__init__(cfg=cfg, trainer=trainer, no_lm_init=no_lm_init)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.file:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return

        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, mode=TRAINING_MODE)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.file:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return

        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, mode=EVALUATION_MODE)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.file is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return

        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, mode=EVALUATION_MODE)

    def setup_inference_data(self, input_file, batch_size=1, num_samples=-1, num_workers=2):
        dataloader_cfg = {
            "batch_size": batch_size,
            "file": input_file,
            "shuffle": False,
            "num_samples": num_samples,
            'num_workers': num_workers,
            'pin_memory': False,
            'drop_last': False,
        }
        dataloader_cfg = OmegaConf.create(dataloader_cfg)
        inference_dl = self._setup_dataloader_from_config(cfg=dataloader_cfg, mode=INFERENCE_MODE)

        return inference_dl

    def _setup_dataloader_from_config(self, cfg: DictConfig, mode: str):
        raise NotImplementedError()

    @torch.no_grad()
    def _get_per_sample_perplexity(self, logits, labels):
        """Returns average perplexity for each sample in the batch"""

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        unreduced_loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        unreduced_loss = unreduced_loss.reshape(labels.shape)
        mask_0 = unreduced_loss != 0
        per_sample_perplexity = torch.exp((unreduced_loss * mask_0).sum(axis=1) / mask_0.sum(axis=1))

        return per_sample_perplexity
