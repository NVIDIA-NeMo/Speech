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
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.data.language_modeling.t0_dataset import T0Dataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_sft_model import MegatronT5SFTModel
from nemo.utils import AppState, logging

try:
    from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False


try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

__all__ = ['MegatronT0Model']


class MegatronT0Model(MegatronT5SFTModel):
    """T0 (https://arxiv.org/abs/2110.08207) Model that Inherits from MegatronT5SFTModel and overrides the dataset building."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

    def setup(self, stage=None):
        # NOTE: super().__init__ will try and setup train/val/test datasets, but we sidestep this using a if self._train_ds is not None condition
        # We then set things up for real only once setup() of this class is called.
        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples

        if stage == 'predict':
            return

        # If the user wants to manually override train and validation dataloaders before calling `.fit()`
        if self._train_dl is not None and self._validation_dl is not None:
            return
        self.build_train_valid_test_datasets(stage=stage)
        if hasattr(self, '_train_ds'):
            self.setup_training_dataloader()
        if hasattr(self, '_validation_ds'):
            self._validation_dl = self.setup_eval_dataloader(self._validation_ds, self.cfg.data.validation_ds)
        if hasattr(self.cfg.data, 'test_ds'):
            self._test_dl = self.setup_test_data(self._test_ds, self.cfg.data.test_ds)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            self.enc_dec_model.sync_initial_word_embeddings()
            # Only synchronize position embeddings if using absolute position embeddings in both the encoder and decoder.
            if (
                self.cfg.encoder.get("position_embedding_type", "learned_absolute") == "learned_absolute"
                and self.cfg.decoder.get("position_embedding_type", "learned_absolute") == "learned_absolute"
            ):
                self.enc_dec_model.sync_initial_position_embeddings()

    def _build_dataset(self, data_cfg, check_implict_grad_acc=False, is_train=True):
        if (
            check_implict_grad_acc
            and data_cfg.global_batch_size > data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size()
        ):
            raise ValueError(
                f'You are trying to use "implicit gradient accumulation" of {data_cfg.global_batch_size // (data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size())} in your validation/test datasets. This is not supported. Please set global_batch_size equal to micro_batch_size * data_parallel_world_size.'
            )
        datasets = []
        # Determine if we are using a single dataset or a list of datasets.
        is_list_config = isinstance(data_cfg.file_names, ListConfig)
        if not is_list_config:
            raise ValueError(f"T0 train/validation datasets must be provided as a list of individual JSONL files.")

        if is_train:
            # Construct the data prefix list for `get_datasets_weights_and_num_samples()` that is of the format [weight1,file_name1,weight2,file_name2,...]
            if data_cfg.concat_sampling_probabilities is None or not isinstance(
                data_cfg.concat_sampling_probabilities, ListConfig
            ):
                raise ValueError(
                    f"concat_sampling_probabilities must be a ListConfig with the same number of files in file_names. Found: {data_cfg.concat_sampling_probabilities}"
                )

            if len(data_cfg.get('concat_sampling_probabilities', None)) != len(data_cfg.file_names):
                raise ValueError(
                    f"concat_sampling_probabilities must be of the same size as file_names. Provided size {len(data_cfg.concat_sampling_probabilities)}, number of datasets {len(data_cfg.file_names)}"
                )

            data_prefix = []
            for weight, prefix in zip(data_cfg.concat_sampling_probabilities, data_cfg.file_names):
                data_prefix.append(weight)
                data_prefix.append(prefix)

            if self.trainer.max_steps is None or self.trainer.max_steps <= 0:
                raise ValueError(
                    f'Trainer max_steps must be set to a positive integer. Found {self.trainer.max_steps}'
                )
            num_train_samples = [self.trainer.max_steps * self.cfg.global_batch_size]
            _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
            num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])
        else:
            num_train_samples_per_dataset = [[None]] * len(data_cfg.file_names)

        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            dataset = T0Dataset(
                file_path=file_path,
                tokenizer=self.tokenizer,
                max_src_seq_length=data_cfg.max_src_seq_length,
                max_tgt_seq_length=data_cfg.max_tgt_seq_length,
                add_bos_to_input=data_cfg.get('add_bos_to_input', False),
                add_eos_to_input=data_cfg.get('add_eos_to_input', False),
                replace_bos_with_pad=data_cfg.get('replace_bos_with_pad', False),
                max_num_samples=num_samples[0],
                seed=data_cfg.get('seed', 1234),
            )
            datasets.append(dataset)

        if is_train:
            dataset = BlendableDataset(
                datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
            )
            return dataset
        else:
            return datasets

    def training_step(self, dataloader_iter, batch_idx):
        return super().training_step(dataloader_iter, batch_idx)

    # Override the parent batch reconfiguring logic.
    def _reconfigure_and_process_inference_batch(self, batch):
        global_batch_per_gpu = batch['text_enc'].size(0)
        # This should happen only on the last batch of the validation/test dataset with drop_last=False.
        if global_batch_per_gpu != self.cfg.data.validation_ds.global_batch_size:
            app_state = AppState()
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
                micro_batch_size=global_batch_per_gpu,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        return batch

    def build_train_valid_test_datasets(self, stage):
        if stage != 'test':
            logging.info('Building T0 validation datasets.')
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._validation_ds = self._build_dataset(
                self.cfg.data.validation_ds, check_implict_grad_acc=True, is_train=False
            )
            logging.info(f'Length of val dataset: {len(self._validation_ds[0])}')

        if stage != 'validate':
            if hasattr(self.cfg.data, 'test_ds'):
                logging.info('Building T0 test datasets.')
                # Wrap this in a list since the general finetuning parent class supports multi-validation.
                self._test_ds = self._build_dataset(self.cfg.data.test_ds, check_implict_grad_acc=True, is_train=False)
                logging.info(f'Length of test dataset: {len(self._test_ds[0])}')

        if stage == 'validate' or stage == 'test':
            return
        logging.info('Building T0 traing datasets.')
        self._train_ds = self._build_dataset(self.cfg.data.train_ds, check_implict_grad_acc=False)
        logging.info(f'Length of train dataset: {len(self._train_ds)}')

    def build_data_loader(
        self, dataset, data_cfg, consumed_samples=0,
    ):
        """Buld dataloader given an input dataset."""
        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        if isinstance(dataset, BlendableDataset):
            collate_fn = dataset.datasets[0].collate_fn
        else:
            collate_fn = dataset.collate_fn

        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=data_cfg.micro_batch_size,
            global_batch_size=data_cfg.global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
        )

    def setup_training_dataloader(self):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            self._train_dl = self.build_data_loader(
                dataset=self._train_ds, data_cfg=self.cfg.data.train_ds, consumed_samples=consumed_samples,
            )

    def setup_eval_dataloader(self, datasets, data_cfg):
        dataloaders = []
        for dataset in datasets:
            eval_dl = self.build_data_loader(dataset=dataset, data_cfg=data_cfg, consumed_samples=0,)
            dataloaders.append(eval_dl)
        return dataloaders

    # TODO: Temporary overrides of finetune model. This needs to removed in the finetune model.
    def on_train_start(self) -> None:
        super().on_train_start()

    def on_validation_start(self) -> None:
        super().on_validation_start()

    def on_test_start(self) -> None:
        super().on_test_start()
