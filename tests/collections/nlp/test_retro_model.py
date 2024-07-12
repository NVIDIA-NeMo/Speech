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

import json

import pytest
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_retro_model import MegatronRetroModel
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


@pytest.fixture()
def retro_workdir_path(test_data_dir):

    config_file = {
        "retro_bert_tokenizer_type": "BertWordPieceLowerCase",
        "retro_bert_vocab_file": "",
        "retro_block_size": 1000,
        "retro_gpt_chunk_length": 64,
        "retro_gpt_data_cache_path": None,
        "retro_gpt_data_path": "",
        "retro_gpt_eval_interval": 2000,
        "retro_gpt_eval_iters": 100,
        "retro_gpt_global_batch_size": 8,
        "retro_gpt_merge_file": None,
        "retro_gpt_seed": 1234,
        "retro_gpt_seq_length": 2048,
        "retro_gpt_split": "98,2,0",
        "retro_gpt_tokenizer_model": "spm_tok_ende_4k/tokenizer.model",
        "retro_gpt_tokenizer_type": "GPTSentencePieceTokenizer",
        "retro_gpt_train_samples": 5000,
        "retro_gpt_valid_samples": 5000,
        "retro_gpt_vocab_file": None,
        "retro_neighbor_dirs": {"test": None, "train": None, "valid": None},
    }

    # save config to json file in retro_workdir_path
    retro_workdir_path = test_data_dir + "/nlp"
    config_file_path = retro_workdir_path + "/config.json"
    out_file = open(config_file_path, 'w')
    json.dump(config_file, out_file)
    out_file.close()

    return retro_workdir_path


@pytest.fixture()
def model_cfg(test_data_dir, retro_workdir_path):

    # set model configs
    model_cfg = {
        'mcore_gpt': True,
        'precision': '16',
        'micro_batch_size': 4,
        'global_batch_size': 8,
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'resume_from_checkpoint': None,
        'encoder_seq_length': 2048,
        'max_position_embeddings': 2048,
        'num_layers': 12,
        'hidden_size': 768,
        'ffn_hidden_size': 3072,
        'num_attention_heads': 12,
        'init_method_std': 0.023,
        'hidden_dropout': 0.1,
        'kv_channels': 64,
        # 'apply_query_key_layer_scaling': False,
        'apply_query_key_layer_scaling': True,
        'layernorm_epsilon': 1e-5,
        'make_vocab_size_divisible_by': 128,
        'pre_process': True,
        'post_process': True,
        'persist_layer_norm': True,
        'bias': True,
        'activation': 'gelu',
        'transformer_block_type': 'pre_ln',
        'retro': {
            # 'retro_project_dir': os.path.join('tests/.data/test_data/nlp/retro_workdir_dummy'),
            # 'retro_project_dir': os.path.join(test_data_dir, 'nlp/retro_workdir_dummy'),
            # 'retro_project_dir': '/lustre/fsw/coreai_dlalgo_genai/huvu/data/retro/pretrain_data/micro-wiki-core-unittest',
            'retro_project_dir': retro_workdir_path,
            'retro_encoder_num_layers': 2,
            'retro_encoder_hidden_dropout': 0.1,
            'retro_encoder_attention_dropout': 0.1,
            'retro_num_neighbors': 2,
            'retro_num_retrieved_chunks': 2,
            'retro_verify_neighbor_count': True,
        },
        'tokenizer': {
            'library': 'megatron',
            'type': None,
            'model': None,
            'vocab_file': None,
            'merge_file': None,
            'delimiter': None,
            'sentencepiece_legacy': False,
        },
        'native_amp_init_scale': 4294967296,
        'native_amp_growth_interval': 1000,
        'hysteresis': 2,
        'fp32_residual_connection': False,
        'fp16_lm_cross_entropy': False,
        'megatron_amp_O2': True,
        'seed': 1234,
        'use_cpu_initialization': False,
        'onnx_safe': False,
        'apex_transformer_log_level': 30,
        'activations_checkpoint_method': None,
        'activations_checkpoint_num_layers': None,
        'data': {
            'data_prefix': 'None',
            'index_mapping_dir': None,
            'data_impl': 'mmap',
            'splits_string': '98,2,0',
            'seq_length': 2048,
            'skip_warmup': True,
            'num_workers': 2,
            'dataloader_type': 'single',
            'reset_position_ids': False,
            'reset_attention_mask': False,
            'eod_mask_loss': False,
            'shuffle_documents': False,
            'retro_data': {
                'retro_block_size': 10000,
                'retro_chunk_length': 64,
                'retro_split_preprocessing': "98,2,0",
                'retro_neighbor_dirs': None,
            },
        },
        'optim': {
            'name': 'distributed_fused_adam',
            'lr': 6.0e-4,
            'weight_decay': 0.1,
            'betas': [0.9, 0.95],
            'sched': {'name': 'CosineAnnealing', 'warmup_steps': None, 'constant_steps': None, 'min_lr': '6.0e-5'},
        },
    }
    return model_cfg


@pytest.fixture()
def trainer_cfg():

    trainer_cfg = {
        'devices': 1,
        'num_nodes': 1,
        'accelerator': 'gpu',
        'precision': '16',
        'logger': False,
        'enable_checkpointing': False,
        'use_distributed_sampler': False,
        'max_epochs': -1,
        'max_steps': 750000,
        'log_every_n_steps': 10,
        'val_check_interval': 100,
        'limit_val_batches': 50,
        'limit_test_batches': 500,
        'accumulate_grad_batches': 1,
        'gradient_clip_val': 1.0,
    }

    return trainer_cfg


@pytest.fixture()
def retro_model(model_cfg, trainer_cfg):

    strategy = NLPDDPStrategy()

    trainer = Trainer(strategy=strategy, **trainer_cfg)

    cfg = DictConfig(model_cfg)

    model = MegatronRetroModel(cfg=cfg, trainer=trainer)

    return model


@pytest.mark.run_only_on('GPU')
class TestRETROModel:
    @pytest.mark.unit
    def test_constructor(self, retro_model):
        assert isinstance(retro_model, MegatronRetroModel)

        num_weights = retro_model.num_weights
        ## due to recent change in M-LM RETRO model, the exact number of parameters of RETRO is not determined.
        ## temporary skip checking for number of parameters, will be added after M-LM RETRO side is concluded
        # assert num_weights == 306868224 # using "tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model" tokenizer
        # assert num_weights == 113405952  # using "spm_tok_ende_4k/tokenizer.model" tokenizer

    @pytest.mark.unit
    def test_forward(self, retro_model):

        # create dummy input
        batch_size = 4
        neighbors = 2
        seq_length = 2048
        chunk_length = 64
        num_chunks = seq_length // chunk_length
        retrieved_chunk_size = chunk_length * 2
        vocab_size = 2000
        eos_id = vocab_size - 2

        # set input for forward
        all_tokens = torch.randint(0, vocab_size, (batch_size, seq_length + 1)).cuda()
        tokens = all_tokens[:, :-1]
        labels = all_tokens[:, 1:]
        attention_mask, _, text_position_ids = get_ltor_masks_and_position_ids(tokens, eos_id, False, False, False)
        context_input_ids = torch.randint(
            0, vocab_size, (batch_size * num_chunks * neighbors, retrieved_chunk_size)
        ).cuda()
        _, _, context_position_ids = get_ltor_masks_and_position_ids(  # neighbor_tokens is already a 2D array
            context_input_ids, eos_id, False, False, False
        )
        context_mask = None

        # set model to eval mode
        retro_model.eval()

        # forward step
        with torch.no_grad():
            out = retro_model(
                tokens=tokens.cuda(),
                text_position_ids=text_position_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                labels=labels.cuda(),
                context_input_ids=context_input_ids.cuda(),
                context_position_ids=context_position_ids.cuda(),
                context_mask=context_mask,
            )

        assert out.shape == torch.Size([batch_size, seq_length])
