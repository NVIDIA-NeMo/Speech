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
Example to run this conversion script:
```
    python /opt/NeMo/scripts/checkpoint_converters/convert_bert_hf_to_nemo.py \
     --input_name_or_path /path/to/hf/checkpoints/folder \
     --output_path /path/to/output/nemo/file.nemo \
     --mcore True \
     --precision bf16
```
"""

import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf
from transformers import AutoModel
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging


def adjust_nemo_config(model_config, ref_config, mcore_bert=True):
    model_config.tokenizer["type"] = ref_config["_name_or_path"]
    model_config.tokenizer["library"] = "huggingface"
    model_config.tokenizer["use_fast"] = True
    model_config["max_position_embeddings"] = ref_config['max_position_embeddings']
    model_config["num_layers"] = ref_config["num_hidden_layers"]
    model_config["hidden_size"] = ref_config["hidden_size"]
    model_config["ffn_hidden_size"] = ref_config["intermediate_size"]
    model_config["num_attention_heads"] = ref_config["num_attention_heads"]
    model_config["layernorm_epsilon"] = ref_config["layer_norm_eps"]
    model_config["normalization"] = "layernorm"
    model_config["transformer_block_type"] = "post_ln"
    model_config["apply_query_key_layer_scaling"] = False
    model_config["megatron_legacy"] = False
    model_config["mcore_bert"] = mcore_bert
    return model_config


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input_name_or_path", type=str, default="thenlper/gte-large")
    parser.add_argument("--mcore", type=bool, default=True)
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_bert_config.yaml",
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument(
        "--post_process", type=bool, default=False, required=False, help="Whether to have the postprocessing modules"
    )
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )

    args = parser.parse_args()
    return args


def convert(args):
    logging.info(f"Loading checkpoint from HF: `{args.input_name_or_path}`")
    hf_model = AutoModel.from_pretrained(args.input_name_or_path)

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.model = adjust_nemo_config(nemo_config.model, hf_model.config.to_dict(), mcore_bert=args.mcore)

    nemo_config.trainer["precision"] = args.precision
    trainer = MegatronTrainerBuilder(nemo_config).create_trainer()
    model = MegatronBertModel(nemo_config.model, trainer)

    if not args.post_process:
        (
            model.model.module.lm_head,
            model.model.module.encoder.final_layernorm,
            model.model.module.binary_head,
            model.model.module.output_layer,
        ) = (
            None,
            None,
            None,
            None,
        )

    nemo_state_dict = {}
    hf_config = hf_model.config.to_dict()
    hidden_size = hf_config["hidden_size"]
    head_num = hf_config["num_attention_heads"]
    head_size = hidden_size // head_num
    num_layers = hf_config["num_hidden_layers"]

    param_to_weights = lambda param: param.float()
    for l in range(num_layers):
        print(f"converting layer {l}")
        old_tensor_shape = hf_model.state_dict()[f'encoder.layer.{l}.attention.self.query.weight'].size()
        new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
        new_q_tensor_shape_bias = (head_num, head_size)

        q = hf_model.state_dict()[f'encoder.layer.{l}.attention.self.query.weight'].view(*new_q_tensor_shape)
        k = hf_model.state_dict()[f'encoder.layer.{l}.attention.self.key.weight'].view(*new_q_tensor_shape)
        v = hf_model.state_dict()[f'encoder.layer.{l}.attention.self.value.weight'].view(*new_q_tensor_shape)
        bias_q = hf_model.state_dict()[f'encoder.layer.{l}.attention.self.query.bias'].view(*new_q_tensor_shape_bias)
        bias_k = hf_model.state_dict()[f'encoder.layer.{l}.attention.self.key.bias'].view(*new_q_tensor_shape_bias)
        bias_v = hf_model.state_dict()[f'encoder.layer.{l}.attention.self.value.bias'].view(*new_q_tensor_shape_bias)

        qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:])
        qkv_biases = torch.empty((0, head_size))
        for i in range(head_num):
            qkv_weights = torch.cat((qkv_weights, q[i : i + 1, :, :]))
            qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
            qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
            qkv_biases = torch.cat((qkv_biases, bias_q[i : i + 1]))
            qkv_biases = torch.cat((qkv_biases, bias_k[i : i + 1]))
            qkv_biases = torch.cat((qkv_biases, bias_v[i : i + 1]))

        qkv_weights = qkv_weights.reshape([head_size * (3 * head_num), hidden_size])
        qkv_biases = qkv_biases.reshape([head_size * (3 * head_num)])

        if args.mcore:
            qkv_weights_base_name = f'model.encoder.layers.{l}.self_attention.linear_qkv.weight'
            qkv_biases_base_name = f'model.encoder.layers.{l}.self_attention.linear_qkv.bias'
        else:
            qkv_weights_base_name = f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.weight'
            qkv_biases_base_name = f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.bias'
        nemo_state_dict[qkv_weights_base_name] = param_to_weights(qkv_weights)
        nemo_state_dict[qkv_biases_base_name] = param_to_weights(qkv_biases)

        # attention dense
        dense_weight = hf_model.state_dict()[f'encoder.layer.{l}.attention.output.dense.weight']
        dense_bias = hf_model.state_dict()[f'encoder.layer.{l}.attention.output.dense.bias']
        if args.mcore:
            dense_weight_base_name = f'model.encoder.layers.{l}.self_attention.linear_proj.weight'
            dense_bias_base_name = f'model.encoder.layers.{l}.self_attention.linear_proj.bias'
        else:
            dense_weight_base_name = f'model.language_model.encoder.layers.{l}.self_attention.dense.weight'
            dense_bias_base_name = f'model.language_model.encoder.layers.{l}.self_attention.dense.bias'
        nemo_state_dict[dense_weight_base_name] = param_to_weights(dense_weight)
        nemo_state_dict[dense_bias_base_name] = param_to_weights(dense_bias)

        # LayerNorm1
        LayerNorm1_weight = hf_model.state_dict()[f'encoder.layer.{l}.attention.output.LayerNorm.weight']
        LayerNorm1_bias = hf_model.state_dict()[f'encoder.layer.{l}.attention.output.LayerNorm.bias']
        if args.mcore:
            LayerNorm1_weight_base_name = f'model.encoder.layers.{l}.post_att_layernorm.weight'
            LayerNorm1_bias_base_name = f'model.encoder.layers.{l}.post_att_layernorm.bias'
        else:
            LayerNorm1_weight_base_name = f'model.language_model.encoder.layers.{l}.input_layernorm.weight'
            LayerNorm1_bias_base_name = f'model.language_model.encoder.layers.{l}.input_layernorm.bias'
        nemo_state_dict[LayerNorm1_weight_base_name] = param_to_weights(LayerNorm1_weight)
        nemo_state_dict[LayerNorm1_bias_base_name] = param_to_weights(LayerNorm1_bias)

        # MLP 1
        MLP1_weight = hf_model.state_dict()[f'encoder.layer.{l}.intermediate.dense.weight']
        MLP1_bias = hf_model.state_dict()[f'encoder.layer.{l}.intermediate.dense.bias']
        if args.mcore:
            MLP1_weight_base_name = f'model.encoder.layers.{l}.mlp.linear_fc1.weight'
            MLP1_bias_base_name = f'model.encoder.layers.{l}.mlp.linear_fc1.bias'
        else:
            MLP1_weight_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.weight'
            MLP1_bias_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.bias'
        nemo_state_dict[MLP1_weight_base_name] = param_to_weights(MLP1_weight)
        nemo_state_dict[MLP1_bias_base_name] = param_to_weights(MLP1_bias)

        # MLP 2
        MLP2_weight = hf_model.state_dict()[f'encoder.layer.{l}.output.dense.weight']
        MLP2_bias = hf_model.state_dict()[f'encoder.layer.{l}.output.dense.bias']
        if args.mcore:
            MLP2_weight_base_name = f'model.encoder.layers.{l}.mlp.linear_fc2.weight'
            MLP2_bias_base_name = f'model.encoder.layers.{l}.mlp.linear_fc2.bias'
        else:
            MLP2_weight_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.weight'
            MLP2_bias_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.bias'
        nemo_state_dict[MLP2_weight_base_name] = param_to_weights(MLP2_weight)
        nemo_state_dict[MLP2_bias_base_name] = param_to_weights(MLP2_bias)

        # LayerNorm2
        LayerNorm2_weight = hf_model.state_dict()[f'encoder.layer.{l}.output.LayerNorm.weight']
        LayerNorm2_bias = hf_model.state_dict()[f'encoder.layer.{l}.output.LayerNorm.bias']
        if args.mcore:
            LayerNorm2_weight_base_name = f'model.encoder.layers.{l}.post_mlp_layernorm.weight'
            LayerNorm2_bias_base_name = f'model.encoder.layers.{l}.post_mlp_layernorm.bias'
        else:
            LayerNorm2_weight_base_name = f'model.language_model.encoder.layers.{l}.post_attention_layernorm.weight'
            LayerNorm2_bias_base_name = f'model.language_model.encoder.layers.{l}.post_attention_layernorm.bias'
        nemo_state_dict[LayerNorm2_weight_base_name] = param_to_weights(LayerNorm2_weight)
        nemo_state_dict[LayerNorm2_bias_base_name] = param_to_weights(LayerNorm2_bias)

        nemo_state_dict[f'model.encoder.layers.{l}.self_attention.linear_proj._extra_state'] = model.state_dict()[
            f'model.encoder.layers.{l}.self_attention.linear_proj._extra_state'
        ]
        nemo_state_dict[f'model.encoder.layers.{l}.self_attention.linear_qkv._extra_state'] = model.state_dict()[
            f'model.encoder.layers.{l}.self_attention.linear_qkv._extra_state'
        ]
        nemo_state_dict[f'model.encoder.layers.{l}.mlp.linear_fc1._extra_state'] = model.state_dict()[
            f'model.encoder.layers.{l}.mlp.linear_fc1._extra_state'
        ]
        nemo_state_dict[f'model.encoder.layers.{l}.mlp.linear_fc2._extra_state'] = model.state_dict()[
            f'model.encoder.layers.{l}.mlp.linear_fc2._extra_state'
        ]

    # Non-layer dependent keys
    word_embeddings_weight = hf_model.state_dict()['embeddings.word_embeddings.weight']
    position_embeddings_weight = hf_model.state_dict()['embeddings.position_embeddings.weight']
    token_type_embeddings_weight = hf_model.state_dict()['embeddings.token_type_embeddings.weight']
    LayerNorm_weight = hf_model.state_dict()['embeddings.LayerNorm.weight']
    LayerNorm_bias = hf_model.state_dict()['embeddings.LayerNorm.bias']
    pooler_dense = hf_model.state_dict()['pooler.dense.weight']
    pooler_bias = hf_model.state_dict()['pooler.dense.bias']

    if args.mcore:
        word_embeddings_weight_base_name = "model.embedding.word_embeddings.weight"
        position_embeddings_weight_base_name = "model.embedding.position_embeddings.weight"
        token_type_embeddings_weight_base_name = "model.embedding.tokentype_embeddings.weight"
        LayerNorm_weight_base_name = "model.encoder.initial_layernorm.weight"
        LayerNorm_bias_base_name = "model.encoder.initial_layernorm.bias"
        pooler_dense_base_name = "model.pooler.dense.weight"
        pooler_bias_base_name = "model.pooler.dense.bias"
    else:
        word_embeddings_weight_base_name = "model.language_model.embedding.word_embeddings.weight"
        position_embeddings_weight_base_name = "model.language_model.embedding.position_embeddings.weight"
        token_type_embeddings_weight_base_name = "model.language_model.embedding.tokentype_embeddings.weight"
        LayerNorm_weight_base_name = "model.language_model.encoder.initial_layernorm.weight"
        LayerNorm_bias_base_name = "model.language_model.encoder.initial_layernorm.bias"
        pooler_dense_base_name = "model.language_model.pooler.dense.weight"
        pooler_bias_base_name = "model.language_model.pooler.dense.bias"

    nemo_state_dict[word_embeddings_weight_base_name] = param_to_weights(word_embeddings_weight)
    nemo_state_dict[position_embeddings_weight_base_name] = param_to_weights(position_embeddings_weight)
    nemo_state_dict[token_type_embeddings_weight_base_name] = param_to_weights(token_type_embeddings_weight)
    nemo_state_dict[LayerNorm_weight_base_name] = param_to_weights(LayerNorm_weight)
    nemo_state_dict[LayerNorm_bias_base_name] = param_to_weights(LayerNorm_bias)
    nemo_state_dict[pooler_dense_base_name] = param_to_weights(pooler_dense)
    nemo_state_dict[pooler_bias_base_name] = param_to_weights(pooler_bias)

    # Padding to new vocab size
    if args.mcore:
        original_embedding = nemo_state_dict['model.embedding.word_embeddings.weight']
    else:
        original_embedding = nemo_state_dict['model.language_model.embedding.word_embeddings.weight']
    vocab_size = original_embedding.size(0)
    if model.padded_vocab_size > vocab_size:
        zeros_to_add = torch.zeros(
            model.padded_vocab_size - vocab_size,
            original_embedding.size(1),
            dtype=original_embedding.dtype,
            device=original_embedding.device,
        )
        # Concatenate the two tensors along rows
        padded_embedding = torch.cat([original_embedding, zeros_to_add], dim=0)
        if args.mcore:
            nemo_state_dict['model.embedding.word_embeddings.weight'] = padded_embedding
        else:
            nemo_state_dict['model.language_model.embedding.word_embeddings.weight'] = padded_embedding

    modified_dict = {}
    for key, value in nemo_state_dict.items():
        if key.startswith('model.'):
            new_key = 'model.module.' + key[len('model.') :]
            modified_dict[new_key] = value
        else:
            modified_dict[key] = value

    nemo_state_dict = modified_dict

    model.load_state_dict(nemo_state_dict, strict=True)
    dtype = torch_dtype_from_precision(args.precision)
    model = model.to(dtype=dtype)
    model.save_to(args.output_path)
    logging.info(f'NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    os.environ['NVTE_FLASH_ATTN'] = '0'  # Bert doesn't support FLASH_ATTN
    args = get_args()
    convert(args)
