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

import os

import pytest
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging

try:
    from megatron.core.num_microbatches_calculator import get_num_microbatches, update_num_microbatches

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches, update_num_microbatches

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


def reset_microbatch_calculator():
    try:
        import megatron.core.num_microbatches_calculator as mb

    except (ImportError, ModuleNotFoundError):
        logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
        import apex.transformer.pipeline_parallel.utils as mb

    mb._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None


@pytest.fixture()
def model_cfg(test_data_dir):

    model_cfg = {
        'precision': 16,
        'micro_batch_size': 4,
        'global_batch_size': 16,
        'rampup_batch_size': [4, 4, 100],
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'resume_from_checkpoint': None,
        'encoder_seq_length': 512,
        'max_position_embeddings': 512,
        'num_layers': 1,
        'hidden_size': 128,
        'ffn_hidden_size': 512,
        'num_attention_heads': 2,
        'init_method_std': 0.02,
        'hidden_dropout': 0.1,
        'kv_channels': None,
        'apply_query_key_layer_scaling': True,
        'layernorm_epsilon': 1e-5,
        'make_vocab_size_divisible_by': 128,
        'pre_process': True,
        'post_process': True,
        'persist_layer_norm': True,
        'gradient_as_bucket_view': True,
        'tokenizer': {
            'library': 'megatron',
            'type': 'GPT2BPETokenizer',
            'model': None,
            'vocab_file': os.path.join(test_data_dir, 'nlp/gpt_vocab_merges/vocab.json'),
            'merge_file': os.path.join(test_data_dir, 'nlp/gpt_vocab_merges/merges.txt'),
            'delimiter': None,
        },
        'native_amp_init_scale': 4294967296,
        'native_amp_growth_interval': 1000,
        'hysteresis': 2,
        'fp32_residual_connection': False,
        'fp16_lm_cross_entropy': False,
        'megatron_amp_O2': False,
        'seed': 1234,
        'use_cpu_initialization': False,
        'onnx_safe': False,
        'apex_transformer_log_level': 30,
        'activations_checkpoint_method': None,
        'activations_checkpoint_num_layers': 1,
        'data': {
            'data_prefix': '???',
            'index_mapping_dir': None,
            'data_impl': 'mmap',
            'splits_string': '900,50,50',
            'seq_length': 512,
            'skip_warmup': True,
            'num_workers': 2,
            'dataloader_type': 'single',
            'reset_position_ids': False,
            'reset_attention_mask': False,
            'eod_mask_loss': False,
        },
        'optim': {
            'name': 'fused_adam',
            'lr': 2e-4,
            'weight_decay': 0.01,
            'betas': [0.9, 0.98],
            'sched': {'name': 'CosineAnnealing', 'warmup_steps': 500, 'constant_steps': 50000, 'min_lr': '2e-5'},
        },
    }

    return model_cfg


@pytest.fixture()
def trainer_cfg():

    trainer_cfg = {
        'devices': 1,
        'num_nodes': 1,
        'accelerator': 'gpu',
        'precision': 16,
        'logger': False,
        'enable_checkpointing': False,
        'use_distributed_sampler': False,
        'max_epochs': 1,
        'max_steps': 150,
        'log_every_n_steps': 10,
        'val_check_interval': 100,
        'limit_val_batches': 50,
        'limit_test_batches': 500,
        'accumulate_grad_batches': 1,
        'gradient_clip_val': 1.0,
    }

    return trainer_cfg


@pytest.fixture()
def gpt_model(model_cfg, trainer_cfg):

    strategy = NLPDDPStrategy()
    trainer = Trainer(strategy=strategy, **trainer_cfg)
    cfg = DictConfig(model_cfg)

    reset_microbatch_calculator()
    model = MegatronGPTModel(cfg, trainer)

    return model


@pytest.fixture()
def rampup_batch_size():

    return [4, 4, 100]


@pytest.fixture()
def rampup_batch_size_schedule():

    return [4, 8, 12, 16]


@pytest.mark.run_only_on('GPU')
class TestRampupBatchSize:
    @pytest.mark.unit
    def test_rampup_bs(self, gpt_model, rampup_batch_size):

        assert gpt_model.cfg.rampup_batch_size == rampup_batch_size

    @pytest.mark.unit
    def test_rampup_bs_schedule(self, gpt_model, trainer_cfg, rampup_batch_size_schedule):
        micro_batch_size = gpt_model.cfg.micro_batch_size
        num_devices = trainer_cfg["devices"]
        num_nodes = trainer_cfg["num_nodes"]
        max_steps = trainer_cfg["max_steps"]

        global_batch_size_schedule = []
        step, consumed_samples = 0, 0
        while step <= max_steps:
            step += 1
            current_global_batch_size = get_num_microbatches() * micro_batch_size * num_devices * num_nodes
            consumed_samples += current_global_batch_size
            update_num_microbatches(consumed_samples=consumed_samples, consistency_check=True)

            if current_global_batch_size not in global_batch_size_schedule:
                global_batch_size_schedule.append(current_global_batch_size)

        reset_microbatch_calculator()

        assert global_batch_size_schedule == rampup_batch_size_schedule
