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

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron.bert_model import BertModel
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_params_for_weight_decay_optimization,
)
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import ChannelType, MaskType, NeuralType
from nemo.utils import AppState, logging

try:
    from apex.transformer import parallel_state, tensor_parallel

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


class MegatronBertModel(MegatronBaseModel):
    """
    Megatron Bert pretraining.
    Model returns [batch, seq, hidden] shape
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__(cfg, trainer=trainer, no_lm_init=False)
        self.cfg = cfg

        # used in NVIDIA NGC PyTorch containers
        # buffer used during train_step for logging average loss over gradient accumulation steps
        self._reduced_lm_loss_buffer = []
        self._reduced_sop_loss_buffer = []

        num_tokentypes = 2 if cfg.bert_binary_head else 0

        self.model = BertModel(
            vocab_size=self.padded_vocab_size,
            hidden_size=cfg.hidden_size,
            max_position_embeddings=cfg.max_position_embeddings,
            num_layers=cfg.num_layers,
            num_attention_heads=cfg.num_attention_heads,
            apply_query_key_layer_scaling=cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=cfg.get('kv_channels', None),
            ffn_hidden_size=cfg.ffn_hidden_size,
            num_tokentypes=num_tokentypes,
            parallel_output=True,
            pre_process=cfg.get('pre_process', True),
            post_process=cfg.get('post_process', True),
            init_method_std=cfg.get('init_method_std', 0.02),
            fp16_lm_cross_entropy=cfg.get('fp16_lm_cross_entropy', False),
            use_cpu_initialization=cfg.get('use_cpu_initialization', False),
            hidden_dropout=cfg.get('hidden_dropout', 0.1),
            precision=cfg.get('precision', 16),
            fp32_residual_connection=cfg.get('fp32_residual_connection', False),
            activations_checkpoint_method=cfg.get('activations_checkpoint_method', None),
            activations_checkpoint_num_layers=cfg.get('activations_checkpoint_num_layers', 1),
            layernorm_epsilon=cfg.get('layernorm_epsilon', 1e-5),
            masked_softmax_fusion=cfg.get('masked_softmax_fusion', True),
            bias_gelu_fusion=cfg.get('bias_gelu_fusion', True),
            onnx_safe=cfg.get('onnx_safe', False),
            add_binary_head=cfg.bert_binary_head,
            megatron_legacy=cfg.get('megatron_legacy', False),
        )
        # not using amp o2
        self.megatron_amp_o2 = False

    def forward(self, input_ids, attention_mask, token_type_ids, lm_labels=None):
        output_tensor = self.model(input_ids, attention_mask, token_type_ids=token_type_ids, lm_labels=lm_labels)

        # Return the output tensor of encoder and transpose from [seq_len, batch, hidden] to [batch, seq_len, hidden]
        if torch.is_tensor(output_tensor):
            output_tensor = output_tensor.transpose(1, 0).contiguous()
        else:
            lm_loss_, sop_logits = output_tensor

            lm_loss_ = lm_loss_.transpose(1, 0).contiguous()
            if sop_logits is not None:
                sop_logits = sop_logits.transpose(1, 0).contiguous()

            output_tensor = (lm_loss_, sop_logits)

        return output_tensor

    def training_step(self, batch, batch_idx):
        tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = self.process_batch(batch)
        if not self.cfg.bert_binary_head:
            types = None
        output_tensor = self(tokens, padding_mask, token_type_ids=types, lm_labels=lm_labels)
        loss_dict = self.loss_func(loss_mask, sentence_order, output_tensor)
        if 'sop loss' in loss_dict:
            lm_loss = loss_dict['lm loss']
            sop_loss = loss_dict['sop loss']
            loss = lm_loss + sop_loss
            reduced_loss = average_losses_across_data_parallel_group([loss, lm_loss, sop_loss])
            self._reduced_loss_buffer.append(reduced_loss[0])
            self._reduced_lm_loss_buffer.append(reduced_loss[1])
            self._reduced_sop_loss_buffer.append(reduced_loss[2])
        else:
            lm_loss = loss_dict['lm loss']
            loss = lm_loss
            reduced_loss = average_losses_across_data_parallel_group([loss, lm_loss])
            self._reduced_loss_buffer.append(reduced_loss[0])
            self._reduced_lm_loss_buffer.append(reduced_loss[1])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            if len(self._reduced_sop_loss_buffer) > 0:
                average_reduced_lm_loss = sum(self._reduced_lm_loss_buffer) / len(self._reduced_lm_loss_buffer)
                average_reduced_sop_loss = sum(self._reduced_sop_loss_buffer) / len(self._reduced_sop_loss_buffer)
                self.log('reduced_lm_train_loss', average_reduced_lm_loss, prog_bar=True)
                self.log('reduced_sop_train_loss', average_reduced_sop_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self.log(
                'consumed_samples',
                self.compute_consumed_samples(self.trainer.global_step - self.init_global_step),
                prog_bar=True,
            )
            self._reduced_loss_buffer = []
            self._reduced_lm_loss_buffer = []
            self._reduced_sop_loss_buffer = []
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = self.process_batch(batch)
        if not self.cfg.bert_binary_head:
            types = None
        output_tensor = self(tokens, padding_mask, token_type_ids=types, lm_labels=lm_labels)
        loss_dict = self.loss_func(loss_mask, sentence_order, output_tensor)
        if 'sop loss' in loss_dict:
            lm_loss = loss_dict['lm loss']
            sop_loss = loss_dict['sop loss']
            loss = lm_loss + sop_loss
        else:
            lm_loss = loss_dict['lm loss']
            loss = lm_loss
        reduced_loss = average_losses_across_data_parallel_group([loss])
        return reduced_loss

    def validation_epoch_end(self, outputs):
        averaged_loss = torch.stack(outputs).mean()
        self.log('val_loss', averaged_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def loss_func(self, loss_mask, sentence_order, output_tensor):
        lm_loss_, sop_logits = output_tensor

        lm_loss_ = lm_loss_.float()
        loss_mask = loss_mask.float()
        lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

        if sop_logits is not None:
            sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(), sentence_order.view(-1), ignore_index=-1)
            sop_loss = sop_loss.float()
            return {'lm loss': lm_loss, 'sop loss': sop_loss}
            # loss = lm_loss + sop_loss
            # averaged_losses = average_losses_across_data_parallel_group(
            #     [lm_loss, sop_loss])
            # return loss, {'lm loss': averaged_losses[0],
            #               'sop loss': averaged_losses[1]}

        else:
            return {'lm loss': lm_loss}
            # loss = lm_loss
            # averaged_losses = average_losses_across_data_parallel_group(
            #     [lm_loss])
            # return loss, {'lm loss': averaged_losses[0]}

    def process_batch(self, batch):
        """Build the batch."""
        # Items and their type.
        keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
        datatype = torch.int64

        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens = data_b['text'].long()
        types = data_b['types'].long()
        sentence_order = data_b['is_random'].long()
        loss_mask = data_b['loss_mask'].float()
        lm_labels = data_b['labels'].long()
        padding_mask = data_b['padding_mask'].long()
        return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask

    def _build_train_valid_test_datasets(self):
        logging.info('Building Bert datasets.')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size / self.cfg.tensor_model_parallel_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = self.trainer.max_steps * self.trainer.accumulate_grad_batches
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            train_valid_test_num_samples[
                1
            ] = 1  # This is to make sure we only have one epoch on every validation iteration

        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self.cfg,
            trainer=self.trainer,
            data_prefix=self.cfg.data.data_prefix,
            data_impl=self.cfg.data.data_impl,
            splits_string=self.cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            max_seq_length=self.cfg.data.seq_length,
            masked_lm_prob=self.cfg.data.masked_lm_prob,
            short_seq_prob=self.cfg.data.short_seq_prob,
            seed=self.cfg.seed,
            skip_warmup=self.cfg.data.get('skip_warmup', True),
            binary_head=self.cfg.bert_binary_head,
            max_seq_length_dec=None,
            dataset_type='standard_bert',
            tokenizer=self.tokenizer.tokenizer,
        )

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building Bert datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.data.num_workers, pin_memory=True,
        )

    def setup(self, stage=None):
        resume_checkpoint_path = self.trainer._checkpoint_connector.resume_from_checkpoint_fit_path

        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0

        self.init_consumed_samples = init_consumed_samples

        if stage == 'predict':
            return
        # TODO: consider adding a ModelPT guard to check if model is being restored.
        # allowing restored models to optionally setup datasets
        self._build_train_valid_test_datasets()
        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        self.setup_test_data(self.cfg.data)

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def compute_consumed_samples(self, steps_since_resume=0):
        app_state = AppState()
        consumed_samples = (
            self.init_consumed_samples
            + steps_since_resume
            * app_state.data_parallel_size
            * self.cfg.micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return int(consumed_samples)

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []
        for vocab in ['cased', 'uncased']:
            result.append(
                PretrainedModelInfo(
                    pretrained_model_name=f"megatron_bert_345m_{vocab}",
                    location=f"https://api.ngc.nvidia.com/v2/models/nvidia/nemo/megatron_bert_345m_{vocab}/versions/1/files/megatron_bert_345m_{vocab}.nemo",
                    description=f"345M parameter BERT Megatron model with {vocab} vocab.",
                )
            )
        for vocab_size in ['50k', '30k']:
            for vocab in ['cased', 'uncased']:
                result.append(
                    PretrainedModelInfo(
                        pretrained_model_name=f"biomegatron345m_biovocab_{vocab_size}_{vocab}",
                        location=f"https://api.ngc.nvidia.com/v2/models/nvidia/nemo/biomegatron345m_biovocab_{vocab_size}_{vocab}/versions/1/files/BioMegatron345m-biovocab-{vocab_size}-{vocab}.nemo",
                        description="Megatron 345m parameters model with biomedical vocabulary ({vocab_size} size) {vocab}, pre-trained on PubMed biomedical text corpus.",
                    )
                )
        for vocab in ['cased', 'uncased']:
            result.append(
                PretrainedModelInfo(
                    pretrained_model_name=f"biomegatron-bert-345m-{vocab}",
                    location=f"https://api.ngc.nvidia.com/v2/models/nvidia/nemo/biomegatron345m{vocab}/versions/1/files/BioMegatron345m{vocab.capitalize()}.nemo",
                    description=f"Megatron pretrained on {vocab} biomedical dataset PubMed with 345 million parameters.",
                )
            )
        return result

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        self._optimizer_param_groups = get_params_for_weight_decay_optimization([self.model])

    # Required for ONNX export
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
        }

    # Required for ONNX export
    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        sz = (max_batch, max_dim)
        input_ids = torch.randint(low=0, high=2048, size=sz, device=sample.device)
        token_type_ids = torch.randint(low=0, high=1, size=sz, device=sample.device)
        attention_mask = torch.randint(low=0, high=1, size=sz, device=sample.device)
        input_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        return tuple([input_dict])
