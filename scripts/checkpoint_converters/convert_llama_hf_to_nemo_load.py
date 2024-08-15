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

r"""
Conversion script to convert Huggingface LLaMA checkpoints into nemo checkpoint.
  Example to run this conversion script:
    python convert_llama_hf_to_nemo.py \
     --input_name_or_path <path_to_hf_checkpoints_folder> \
     --input_state_dict <path_to_saved_state_dict> \
     --output_path <path_to_output_nemo_file> \
     --precision bf16
     --llama31 True
"""

import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.collections.nlp.parts.utils_funcs import load_state_dict_helper, torch_dtype_from_precision
from nemo.utils import logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface LLaMA checkpoints",
    )
    parser.add_argument(
        "--input_state_dict",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface LLaMA checkpoints",
    )

    parser.add_argument(
        "--llama31",
        type=bool,
        default=True,
        required=False,
        help="Apply scaling for RoPE frequencies",
    )

    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), '../../examples/nlp/language_modeling/conf/megatron_llama_config.yaml'
        ),
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--precision", type=str, default="16", help="Model precision")
    args = parser.parse_args()
    return args


def load_config(args, llama_config):
    nemo_config = OmegaConf.load(args.hparams_file).model

    if llama_config.get('rope_theta', None):
        nemo_config['rotary_base'] = llama_config['rope_theta']
    nemo_config.encoder_seq_length = llama_config['max_position_embeddings']
    nemo_config.num_layers = int(llama_config['num_hidden_layers'])
    nemo_config.hidden_size = llama_config['hidden_size']
    nemo_config.ffn_hidden_size = llama_config['intermediate_size']
    nemo_config.num_attention_heads = llama_config['num_attention_heads']
    nemo_config.max_position_embeddings = llama_config['max_position_embeddings']
    nemo_config.init_method_std = llama_config['initializer_range']
    nemo_config.layernorm_epsilon = llama_config['rms_norm_eps']
    if 'num_key_value_heads' in llama_config:
        nemo_config.num_query_groups = llama_config['num_key_value_heads']
    nemo_config.use_cpu_initialization = True
    nemo_config.activation = 'fast-swiglu'
    nemo_config.megatron_amp_O2 = True  # True
    nemo_config.scale_positional_embedding = args.llama31

    # Tokenizer config
    if 'tokenizer_model' in llama_config:
        nemo_config.tokenizer.model = llama_config['tokenizer_model']
    else:
        # Llama3 uses converted TikToken Tokenizer
        tokenizer_dict = {
            'library': 'huggingface',
            'type': args.input_name_or_path,
            'use_fast': True,
        }
        nemo_config.tokenizer = tokenizer_dict

    if llama_config['rope_scaling'] is not None:
        if llama_config['rope_scaling']['type'] == 'linear':
            nemo_config['seq_len_interpolation_factor'] = llama_config['rope_scaling']['factor']
        else:
            raise ValueError("Only linear rope scaling type is supported now")
    if llama_config['rope_theta'] is not None:
        nemo_config['rotary_base'] = llama_config['rope_theta']

    base = 128
    while llama_config['vocab_size'] % base != 0:
        base //= 2
    nemo_config.make_vocab_size_divisible_by = base

    return nemo_config


def convert(args):
    logging.info(f"loading checkpoint {args.input_name_or_path}")
    import torch

    model = LlamaForCausalLM.from_pretrained(
        args.input_name_or_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    hf_config = vars(model.config)
    if os.path.exists(f'{args.input_name_or_path}/tokenizer.model'):
        tokenizer = LlamaTokenizer.from_pretrained(args.input_name_or_path)
        hf_config['tokenizer_model'] = str(tokenizer.vocab_file)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.input_name_or_path)
    print(f"hf_config: {hf_config}")
    print("named parameters:")
    for name, param in model.named_parameters():
        print(f"- {name}")

    nemo_config = load_config(args, hf_config)

    if args.precision in ["32", "16"]:
        precision = int(float(args.precision))
    elif args.precision in ["bf16", "bf16-mixed"]:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = args.precision
        else:
            logging.warning("BF16 is not supported on this device. Using FP16 instead.")
            precision = args.precision[2:]  # prune bf in string
    else:
        precision = args.precision

    plugins = []
    if precision in [16, '16', 'bf16', '16-mixed', 'bf16-mixed']:
        scaler = None
        if precision in [16, '16', '16-mixed']:
            scaler = GradScaler(
                init_scale=nemo_config.get('native_amp_init_scale', 2**32),
                growth_interval=nemo_config.get('native_amp_growth_interval', 1000),
                hysteresis=nemo_config.get('hysteresis', 2),
            )
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'

        if nemo_config.get('megatron_amp_O2', False):
            print('HALF PRECISION')
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    nemo_config.precision = precision
    nemo_config.micro_batch_size = 1
    print(f"nemo_config: {nemo_config}")

    # Remove precision arg, since with PTL >= 2.1 both precision and precision plugin cannot exist together.
    trainer = Trainer(plugins=plugins, accelerator='cpu', strategy=NLPDDPStrategy())

    hidden_size = hf_config["hidden_size"]
    head_num = hf_config["num_attention_heads"]
    head_size = hidden_size // head_num
    num_layers = hf_config["num_hidden_layers"]

    mcore_gpt = nemo_config.mcore_gpt

    assert mcore_gpt == nemo_config.get(
        'transformer_engine', False
    ), "mcore_gpt transformer_engine must be enabled (or disabled) together."

    param_to_weights = lambda param: param.float()

    print('start init model')
    del model
    import time

    st = time.perf_counter()
    model = MegatronGPTModel(nemo_config, trainer)
    print(f'Model init took {time.perf_counter() - st} sec')
    from functools import reduce
    from glob import glob

    weights = glob(f'{args.input_state_dict}/*.pt')
    st = time.perf_counter()
    for weight_file in sorted(weights):
        filename = os.path.basename(weight_file)
        str_list = filename.split('.')
        weight_type = str_list[-2]
        str_name = '.'.join(str_list[1:-1])
        print(f'-- Assign weight_type={weight_type} to {str_name}')
        if nemo_config.get('megatron_amp_O2', False):
            current = reduce(getattr, [model, 'model', 'module'] + str_list[:-2])
        else:
            current = reduce(getattr, [model, 'model'] + str_list[:-2])
        load = torch.load(weight_file)
        if nemo_config.get('megatron_amp_O2', False):
            if precision == 'bf16':
                target_precision = torch.bfloat16
            elif precision == 16:
                target_precision = torch.float16
            load = load.to(target_precision)

        if weight_type == 'weight':
            assert current.weight.shape == load.shape
            assert current.weight.dtype == load.dtype
            current.weight = torch.nn.Parameter(load)
            assert current.weight.norm() == load.norm()
        elif weight_type == 'layer_norm_weight':
            assert current.layer_norm_weight.dtype == load.dtype
            assert current.layer_norm_weight.shape == load.shape
            current.layer_norm_weight = torch.nn.Parameter(load)
            assert current.layer_norm_weight.norm() == load.norm()
        else:
            raise ValueError(f'Unsupported weight type = {weight_type}')
        del load

    print(f'Finish loading model in {time.perf_counter() - st} sec. Start to save model')
    st = time.perf_counter()
    print(f'Model save took {time.perf_counter() - st} sec.')

    model._save_restore_connector = NLPSaveRestoreConnector()

    # We make sure that the tokenizer can be instantiated later regardless of args.input_name_or_path
    if 'tokenizer_model' not in hf_config:
        if args.llama31:
            if hf_config['num_hidden_layers'] == 32:
                model.cfg.tokenizer.update(type='meta-llama/Meta-Llama-3.1-8B')
            elif hf_config['num_hidden_layers'] == 80:
                model.cfg.tokenizer.update(type='meta-llama/Meta-Llama-3.1-70B')
            elif hf_config['num_hidden_layers'] == 126:
                model.cfg.tokenizer.update(type='meta-llama/Meta-Llama-3.1-8B')  # 405B tokenizer is the same as 8B
            else:
                logging.warning("Unexpected model config for Llama3. Tokenizer config has not been modified.")
        else:
            if hf_config['num_hidden_layers'] == 32:
                model.cfg.tokenizer.update(type='meta-llama/Meta-Llama-3-8B')
            elif hf_config['num_hidden_layers'] == 80:
                model.cfg.tokenizer.update(type='meta-llama/Meta-Llama-3-70B')
            else:
                logging.warning("Unexpected model config for Llama3. Tokenizer config has not been modified.")

    # cast to target precision and disable cpu init
    dtype = torch_dtype_from_precision(precision)
    model = model.to(dtype=dtype)
    model.cfg.use_cpu_initialization = False

    model.save_to(args.output_path)
    logging.info(f'NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
