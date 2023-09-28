# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import (
    TextToTextGLUEDataset,
    TextToTextXNLIDataset,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_sft_model import MegatronT5SFTModel
from nemo.utils import logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

__all__ = ['MegatronT5GLUEModel']


class MegatronT5GLUEModel(MegatronT5SFTModel):
    """GLUE Model that Inherits from MegatronT5SFTModel and overrides the dataset building."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

    def _build_dataset(self, data_cfg, check_implict_grad_acc=False):
        if (
            check_implict_grad_acc
            and data_cfg.global_batch_size > data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size()
        ):
            raise ValueError(
                f'You are trying to use "implicit gradient accumulation" of {data_cfg.global_batch_size // (data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size())} in your validation/test datasets. This is not supported. Please set global_batch_size equal to micro_batch_size * data_parallel_world_size.'
            )
        if data_cfg.task_name == 'xnli':
            dataset = TextToTextXNLIDataset(
                data_cfg.file_path,
                task_name=data_cfg.task_name,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                lang_list=self.cfg.eval_languages,
            )
        else:
            dataset = TextToTextGLUEDataset(
                data_cfg.file_path,
                task_name=data_cfg.task_name,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
            )
        return dataset

    def build_train_valid_test_datasets(self, stage):
        logging.info('Building GLUE/XNLI datasets.')
        if stage != 'test':
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._validation_ds = [self._build_dataset(self.cfg.data.validation_ds, check_implict_grad_acc=True)]
            logging.info(f'Length of val dataset: {len(self._validation_ds[0])}')

        if stage != 'validate':
            if hasattr(self.cfg.data, 'test_ds'):
                # Wrap this in a list since the general finetuning parent class supports multi-validation.
                self._test_ds = [self._build_dataset(self.cfg.data.test_ds, check_implict_grad_acc=True)]
                logging.info(f'Length of test dataset: {len(self._test_ds[0])}')

        if stage == 'validate' or stage == 'test':
            return
        self._train_ds = self._build_dataset(self.cfg.data.train_ds, check_implict_grad_acc=False)
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Finished building GLUE/XNLI datasets.')
