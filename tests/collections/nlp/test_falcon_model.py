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

import pytest
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


@pytest.fixture()
def model_cfg(test_data_dir):

    model_cfg = {
        'mcore_gpt': True,
        'micro_batch_size': 4,
        'global_batch_size': 8,
        'rampup_batch_size': None,
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'virtual_pipeline_model_parallel_size': None,
        'encoder_seq_length': 512,
        'max_position_embeddings': 512,
        'num_layers': 1,
        'hidden_size': 128,
        'ffn_hidden_size': 512,
        'num_attention_heads': 2,
        'num_query_groups': 1,
        'init_method_std': 0.02,
        'use_scaled_init_method': True,
        'hidden_dropout': 0.0,
        'attention_dropout': 0.0,
        'ffn_dropout': 0,
        'kv_channels': None,
        'apply_query_key_layer_scaling': False,
        'normalization': 'layernorm',
        'layernorm_epsilon': 1e-05,
        'do_layer_norm_weight_decay': False,
        'make_vocab_size_divisible_by': 128,
        'pre_process': True,
        'post_process': True,
        'persist_layer_norm': True,
        'bias': False,
        'activation': 'gelu',
        'headscale': False,
        'transformer_block_type': 'pre_ln',
        'openai_gelu': False,
        'normalize_attention_scores': True,
        'position_embedding_type': 'rope',
        'rotary_percentage': 1.0,
        'attention_type': 'multihead',
        'share_embeddings_and_output_weights': False,
        'overlap_p2p_comm': False,
        'batch_p2p_comm': True,
        'seq_len_interpolation_factor': None,
        'tokenizer': {'library': 'huggingface', 'type': 'tiiuae/falcon-40b', 'use_fast': True},
        'native_amp_init_scale': 4294967296,
        'native_amp_growth_interval': 1000,
        'hysteresis': 2,
        'fp32_residual_connection': False,
        'fp16_lm_cross_entropy': False,
        'megatron_amp_O2': False,
        'grad_allreduce_chunk_size_mb': 125,
        'grad_div_ar_fusion': True,
        'gradient_accumulation_fusion': False,
        'bias_activation_fusion': False,
        'bias_dropout_add_fusion': False,
        'masked_softmax_fusion': True,
        'get_attention_mask_from_fusion': True,
        'seed': 1234,
        'resume_from_checkpoint': None,
        'use_cpu_initialization': False,
        'onnx_safe': False,
        'apex_transformer_log_level': 30,
        'gradient_as_bucket_view': True,
        'sync_batch_comm': False,
        'activations_checkpoint_granularity': None,
        'activations_checkpoint_method': None,
        'activations_checkpoint_num_layers': None,
        'num_micro_batches_with_partial_activation_checkpoints': None,
        'activations_checkpoint_layers_per_pipeline': None,
        'sequence_parallel': False,
        'transformer_engine': True,
        'fp8': False,
        'fp8_e4m3': False,
        'fp8_hybrid': False,
        'fp8_margin': 0,
        'fp8_interval': 1,
        'fp8_amax_history_len': 1,
        'fp8_amax_compute_algo': 'most_recent',
        'reduce_amax': True,
        'use_emha': False,
        'ub_tp_comm_overlap': False,
        'ub_tp_comm_overlap_cfg': None,
        'use_flash_attention': False,
        'nsys_profile': {'enabled': False, 'start_step': 10, 'end_step': 10, 'ranks': [0], 'gen_shape': False},
        'optim': {
            'name': 'distributed_fused_adam',
            'lr': '2e-4',
            'weight_decay': 0.01,
            'betas': [0.9, 0.98],
            'sched': {'name': 'CosineAnnealing', 'warmup_steps': 500, 'constant_steps': 50000, 'min_lr': '2e-5'},
        },
        'gc_interval': 0,
        'precision': 'bf16',
        'new_decoder_architecture': False,
        'parallel_attention': True,
        'name': 'megatron_falcon_gpt',
        'target': 'nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel',
    }
    return model_cfg


@pytest.fixture()
def trainer_cfg():

    trainer_cfg = {
        'devices': 1,
        'num_nodes': 1,
        'accelerator': 'gpu',
        'precision': 'bf16',
        'logger': False,
        'enable_checkpointing': False,
        'use_distributed_sampler': False,
        'max_epochs': 1000,
        'max_steps': 100000,
        'log_every_n_steps': 10,
        'val_check_interval': 100,
        'limit_val_batches': 50,
        'limit_test_batches': 500,
        'accumulate_grad_batches': 1,
        'gradient_clip_val': 1.0,
    }

    return trainer_cfg


@pytest.fixture()
def precision():
    return 'bf16'


@pytest.fixture()
def falcon_gpt_model(model_cfg, trainer_cfg, precision):
    model_cfg['precision'] = precision
    trainer_cfg['precision'] = precision

    strategy = NLPDDPStrategy()

    trainer = Trainer(strategy=strategy, **trainer_cfg)

    cfg = DictConfig(model_cfg)

    model = MegatronGPTModel(cfg=cfg, trainer=trainer)

    return model


@pytest.fixture()
def test_text():
    test_text = [
        "hello, world",
        "four score and seven years ago",
        "Your time is limited",
        "If you set goals rediculously high",
    ]
    return test_text


@pytest.mark.run_only_on('GPU')
class TestFalconGPTModel:
    @pytest.mark.unit
    def test_constructor(self, falcon_gpt_model):
        assert isinstance(falcon_gpt_model, MegatronGPTModel)

        num_weights = falcon_gpt_model.num_weights
        assert num_weights == 16827136

    @pytest.mark.unit
    def test_tokenizer(self, falcon_gpt_model, test_text):

        assert isinstance(falcon_gpt_model.tokenizer, AutoTokenizer)
        assert falcon_gpt_model.tokenizer.name == 'PreTrainedTokenizerFast'
        assert falcon_gpt_model.tokenizer.vocab_size == 65024

        ids = [falcon_gpt_model.tokenizer.text_to_ids(text) for text in test_text]

        true_ids = [
            [30835, 23, 1079],
            [18584, 5179, 273, 5144, 909, 2323],
            [4560, 601, 304, 3991],
            [1424, 299, 889, 4258, 2400, 276, 20201, 986],
        ]
        assert sum([id_list == true_id_list for id_list, true_id_list in zip(ids, true_ids)]) == 4

    @pytest.mark.parametrize(
        "precision",
        [
            32,
            16,
            pytest.param(
                "bf16",
                marks=pytest.mark.skipif(
                    not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 8,
                    reason='bfloat16 is not supported on this device',
                ),
            ),
        ],
    )
    @pytest.mark.unit
    def test_forward(self, falcon_gpt_model, test_text):

        dtype = falcon_gpt_model.torch_dtype

        falcon_gpt_model.eval()

        ids = [falcon_gpt_model.tokenizer.text_to_ids(text) for text in test_text]

        id_tensors = [torch.unsqueeze(torch.LongTensor(id_list), dim=0) for id_list in ids]

        masks_and_position_ids = [
            get_ltor_masks_and_position_ids(id_tensor, falcon_gpt_model.tokenizer.eos_id, False, False, False)
            for id_tensor in id_tensors
        ]
        output_tensors = []
        with torch.no_grad():
            for tokens, attn_mask_and_pos_ids in zip(id_tensors, masks_and_position_ids):
                attn_mask, _, pos_ids = attn_mask_and_pos_ids
                assert tokens.shape == pos_ids.shape
                assert attn_mask.shape[2] == attn_mask.shape[3] == tokens.shape[1] == pos_ids.shape[1]
                with torch.autocast('cuda', dtype=dtype):
                    output_tensor = falcon_gpt_model.forward(
                        tokens=tokens.cuda(),
                        text_position_ids=pos_ids.cuda(),
                        attention_mask=attn_mask.cuda(),
                        labels=None,
                    )
                # output is [b s h]
                assert output_tensor.shape[0] == 1
                assert output_tensor.shape[1] == tokens.shape[1]
                assert output_tensor.shape[2] == falcon_gpt_model.padded_vocab_size
                assert output_tensor.dtype == dtype
                output_tensors.append(output_tensor)
