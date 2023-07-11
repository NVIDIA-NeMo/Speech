# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


"""
A script to convert the Mosaic MPT-7B checkpoint on HuggingFace to Megatron GPTModel
This script is hardcoded specifically for the MPT-7B pretrained model only, and is not
generalisable to any other models.

This script will load and convert the model entirely on CPU for OOM safety, but there
is an option to put the model onto GPU before the save down, which sets the map_location
to cuda for the restore_from call. You can do this by adding --cuda to this script call.

This script requires that you have downloaded the 2 .bin weight files for MPT-7B from
HuggingFace located here: https://huggingface.co/mosaicml/mpt-7b/tree/main
These files MUST have the following file names and be saved somewhere where this script
can read them:
    pytorch_model-00001-of-00002.bin
    pytorch_model-00002-of-00002.bin

This script will generate a Megatron model with TP=1 and PP=1. If you need different TP/PP
values, then after running this script, please use the script located below to set whatever
TP/PP values you want:
    NeMo/examples/nlp/language_modeling/megatron_change_num_partitions.py

* Please note: when using the above script, you MUST also pass the `-–megatron_legacy` flag
  Failure to do this will result in a corrupt model! *

This script also requires a baseline config file from which to override default parameters.
You can specify the location of this file using the -c argument. You can use any Nemo config
file which is appropriate, but in the default case, we highly recommend you use the following:
    NeMo/examples/nlp/language_modeling/conf/megatron_gpt_config.yaml


Here is an example usage command:

```python
python scripts/nlp_language_modeling/convert_mpt_7b_hf_to_nemo.py -c /path/to/megatron_gpt_config.yaml -i /path/to/mpt_7b -o /path/to/save
```

"""


import argparse
import os

import pytorch_lightning as pl
import torch
import yaml
from omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron import GPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.utils import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', required=True, type=str, help='path to the two MPT-7B .bin weight files from HuggingFace'
    )
    parser.add_argument(
        '-c', '--config', required=True, type=str, help='the path to the megatron_gpt_config.yaml file'
    )
    parser.add_argument(
        '-o', '--output', required=False, default=None, type=str, help='path to dir where to store output .nemo file'
    )
    parser.add_argument('--cuda', action='store_true', help='put Nemo model onto GPU prior to savedown')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.critical(f'Input directory [ {args.input} ] does not exist or cannot be found. Aborting.')
        exit(255)

    if not os.path.exists(args.config):
        logging.critical(f'Path to config file [ {args.config} ] does not exist or cannot be found. Aborting.')
        exit(255)

    with open(args.config, 'r', encoding='utf_8') as fr:
        orig_cfg = yaml.safe_load(fr)

    model_dict = orig_cfg['model']
    if 'tokenizer' in model_dict:
        del model_dict['tokenizer']
    if 'data' in model_dict:
        del model_dict['data']

    override_model_dict = {
        'micro_batch_size': 1,
        'global_batch_size': 4,
        'rampup_batch_size': None,
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'virtual_pipeline_model_parallel_size': None,
        'megatron_amp_O2': True,
        'transformer_engine': False,
        'use_cpu_initialization': False,
        'hidden_size': 4096,
        'encoder_seq_length': 2048,
        'max_position_embeddings': 2048,
        'num_layers': 32,
        'num_attention_heads': 32,
        'ffn_hidden_size': 4 * 4096,
        'precision': 'bf16',
        'layernorm_epsilon': 1e-5,
        'pre_process': True,
        'post_process': True,
        'num_tokentypes': 0,
        'apply_query_key_layer_scaling': False,
        'parallel_output': False,
        'bias': False,
        'bias_dropout_add_fusion': False,
        'bias_activation_fusion': False,
        'transformer_block_type': 'pre_ln',
        'normalization': 'low_precision_layernorm',
        'fp32_residual_connection': False,
        'hidden_dropout': 0,
        'attention_dropout': 0,
        'ffn_dropout': 0,
        'megatron_legacy': True,
        'share_embeddings_and_output_weights': True,
        'sequence_parallel': False,
        'position_embedding_type': 'alibi',
        'normalize_attention_scores': True,
        'use_flash_attention': False,
        'override_vocab_size': 50432,
    }
    tokeniser_dict = {
        'library': 'huggingface',
        'type': 'EleutherAI/gpt-neox-20b',
        'use_fast': True,
    }
    trainer_dict = {
        'devices': 1,
        'num_nodes': 1,
        'accelerator': 'gpu' if args.cuda else 'cpu',
        'precision': 'bf16',
        'logger': False,  # logger provided by exp_manager
        'enable_checkpointing': False,
        'replace_sampler_ddp': False,
        'max_epochs': -1,  # PTL default. In practice, max_steps will be reached first.
        'max_steps': 100000,  # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
        'log_every_n_steps': 10,
        'val_check_interval': 100,
        'limit_val_batches': 50,
        'limit_test_batches': 500,
        'accumulate_grad_batches': 1,
        'gradient_clip_val': 1.0,
        'benchmark': False,
        'enable_model_summary': False,
    }

    model_dict.update(override_model_dict)
    model_dict['tokenizer'] = tokeniser_dict

    omega_cfg = OmegaConf.create(model_dict)

    trainer = pl.Trainer(**trainer_dict)

    model = MegatronGPTModel(omega_cfg, trainer)

    model_keys = list(model.state_dict().keys())
    model_dtypes = list(set([model.state_dict()[x].dtype for x in model_keys]))

    if not (len(model_dtypes) == 1 and model_dtypes[0] is torch.bfloat16):
        model = model.bfloat16()

    if args.cuda:
        model = model.cuda()

    mpt_1 = torch.load(os.path.join(args.input, 'pytorch_model-00001-of-00002.bin'), map_location="cpu")
    mpt_2 = torch.load(os.path.join(args.input, 'pytorch_model-00002-of-00002.bin'), map_location="cpu")
    mpt_dict = {**mpt_1, **mpt_2}
    del mpt_1, mpt_2

    def convert_state_dict(state_dict, amp=False):
        def get_new_key(old_key):
            if old_key == 'transformer.wte.weight':
                return 'language_model.embedding.word_embeddings.weight'
            elif old_key == 'transformer.norm_f.weight':
                return 'language_model.encoder.final_layernorm.weight'
            else:
                p1 = old_key.replace('transformer.blocks.', 'language_model.encoder.layers.')
                p2 = p1.replace('norm_1.weight', 'input_layernorm.weight')
                p3 = p2.replace('attn.Wqkv.weight', 'self_attention.query_key_value.weight')
                p4 = p3.replace('attn.out_proj.weight', 'self_attention.dense.weight')
                p5 = p4.replace('norm_2.weight', 'post_attention_layernorm.weight')
                p6 = p5.replace('ffn.up_proj.weight', 'mlp.dense_h_to_4h.weight')
                p7 = p6.replace('ffn.down_proj.weight', 'mlp.dense_4h_to_h.weight')

                return p7

        new_dict = {}

        for old_key, val in state_dict.items():
            new_key = get_new_key(old_key)
            if amp:
                new_key = 'module.' + new_key

            new_dict[new_key] = val

        return new_dict

    convert_dict = convert_state_dict(mpt_dict, amp=model_dict['megatron_amp_O2'])

    if model_dict['megatron_amp_O2']:
        missing_keys, unexpected_keys = model.model.load_state_dict(convert_dict, strict=True)
    else:
        missing_keys, unexpected_keys = super(GPTModel, model.model).load_state_dict(convert_dict, strict=True)

    if len(missing_keys) > 0:
        logging.critical('Missing keys were detected during the load, something has gone wrong. Aborting.')
        logging.critical(f'Missing keys: \n{missing_keys}')
        exit(255)

    if len(unexpected_keys) > 0:
        logging.warning('Unexpected keys were detected which should not happen. Please investigate.')
        logging.warning(f'Unexpected keys: \n{unexpected_keys}')

    if args.output is None:
        args.output = os.path.dirname(os.path.abspath(__file__))

    model.save_to(os.path.join(args.output, 'megatron_mpt_7b_base_tp1_pp1.nemo'))
