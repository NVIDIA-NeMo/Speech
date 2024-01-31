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
    python convert_bert_hf_to_nemo.py \
     --input_name_or_path "thenlper/gte-large" \
     --output_path /path/to/output/nemo/file.nemo \
     --precision 32
```
"""

import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer

from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.utils import logging


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def create_rename_keys(num_hidden_layers):
    rename_keys = []
    for i in range(num_hidden_layers):
        # encoder layers: attention mechanism, 2 feedforward neural networks, and 2 layernorms
        rename_keys.extend(
            [
                (
                    f"encoder.layer.{i}.attention.self.query.weight",
                    f"model.language_model.encoder.layers.{i}.self_attention.query.weight",
                ),
                (
                    f"encoder.layer.{i}.attention.self.query.bias",
                    f"model.language_model.encoder.layers.{i}.self_attention.query.bias",
                ),
                (
                    f"encoder.layer.{i}.attention.self.key.weight",
                    f"model.language_model.encoder.layers.{i}.self_attention.key.weight",
                ),
                (
                    f"encoder.layer.{i}.attention.self.key.bias",
                    f"model.language_model.encoder.layers.{i}.self_attention.key.bias",
                ),
                (
                    f"encoder.layer.{i}.attention.self.value.weight",
                    f"model.language_model.encoder.layers.{i}.self_attention.value.weight",
                ),
                (
                    f"encoder.layer.{i}.attention.self.value.bias",
                    f"model.language_model.encoder.layers.{i}.self_attention.value.bias",
                ),
                (
                    f"encoder.layer.{i}.attention.output.dense.weight",
                    f"model.language_model.encoder.layers.{i}.self_attention.dense.weight",
                ),
                (
                    f"encoder.layer.{i}.attention.output.dense.bias",
                    f"model.language_model.encoder.layers.{i}.self_attention.dense.bias",
                ),
                (
                    f"encoder.layer.{i}.attention.output.LayerNorm.weight",
                    f"model.language_model.encoder.layers.{i}.input_layernorm.weight",
                ),
                (
                    f"encoder.layer.{i}.attention.output.LayerNorm.bias",
                    f"model.language_model.encoder.layers.{i}.input_layernorm.bias",
                ),
                (
                    f"encoder.layer.{i}.intermediate.dense.weight",
                    f"model.language_model.encoder.layers.{i}.mlp.dense_h_to_4h.weight",
                ),
                (
                    f"encoder.layer.{i}.intermediate.dense.bias",
                    f"model.language_model.encoder.layers.{i}.mlp.dense_h_to_4h.bias",
                ),
                (
                    f"encoder.layer.{i}.output.dense.weight",
                    f"model.language_model.encoder.layers.{i}.mlp.dense_4h_to_h.weight",
                ),
                (
                    f"encoder.layer.{i}.output.dense.bias",
                    f"model.language_model.encoder.layers.{i}.mlp.dense_4h_to_h.bias",
                ),
                (
                    f"encoder.layer.{i}.output.LayerNorm.weight",
                    f"model.language_model.encoder.layers.{i}.post_attention_layernorm.weight",
                ),
                (
                    f"encoder.layer.{i}.output.LayerNorm.bias",
                    f"model.language_model.encoder.layers.{i}.post_attention_layernorm.bias",
                ),
            ]
        )

    # Non-layer dependent keys
    rename_keys.extend(
        [
            ("embeddings.word_embeddings.weight", "model.language_model.embedding.word_embeddings.weight"),
            ("embeddings.position_embeddings.weight", "model.language_model.embedding.position_embeddings.weight"),
            ("embeddings.token_type_embeddings.weight", "model.language_model.embedding.tokentype_embeddings.weight"),
            ("embeddings.LayerNorm.weight", "model.language_model.encoder.initial_layernorm.weight"),
            ("embeddings.LayerNorm.bias", "model.language_model.encoder.initial_layernorm.bias"),
            ("pooler.dense.weight", "model.language_model.pooler.dense.weight"),
            ("pooler.dense.bias", "model.language_model.pooler.dense.bias"),
        ]
    )

    return rename_keys


def rename_model_keys(model_state_dict, rename_keys):
    """
    Rename keys in the model's state dictionary based on the provided mappings.

    Parameters:
    model_state_dict (dict): The state dictionary of the model.
    rename_keys (list): A list of tuples with the mapping (old_key, new_key).

    Returns:
    dict: A new state dictionary with updated key names.
    """

    # Create a new state dictionary with updated key names
    new_state_dict = {}

    # Track keys from the original state dict to ensure all are processed
    remaining_keys = set(model_state_dict.keys())

    # Iterate over the rename mappings
    for old_key, new_key in rename_keys:
        if old_key in model_state_dict:
            # Rename the key and remove it from the tracking set
            new_state_dict[new_key] = model_state_dict[old_key]
            remaining_keys.remove(old_key)
        else:
            print(f"Warning: Key '{old_key}' not found in the model state dictionary.")

    # Check if any keys were not converted from old to new
    for old_key in remaining_keys:
        print(f"Warning: Key '{old_key}' was not converted.")

    return new_state_dict


def adjust_tensor_shapes(model, nemo_state_dict):
    """
    Adapt tensor shapes in the state dictionary to ensure compatibility with a different model structure.

    Parameters:
    nemo_state_dict (dict): The state dictionary of the model.

    Returns:
    dict: The updated state dictionary with modified tensor shapes for compatibility.
    """

    # Note: For 'key' and 'value' weights and biases, NeMo uses a consolidated tensor 'query_key_value'.
    for key_ in list(nemo_state_dict.keys()):
        if "self_attention.query" in key_:
            key_q = key_
            key_k = key_.replace('self_attention.query', 'self_attention.key')
            key_v = key_.replace('self_attention.query', 'self_attention.value')
            key_new = key_.replace('self_attention.query', 'self_attention.query_key_value')
            value_new = torch.concat((nemo_state_dict[key_q], nemo_state_dict[key_k], nemo_state_dict[key_v]), dim=0)
            nemo_state_dict[key_new] = value_new
            del nemo_state_dict[key_q], nemo_state_dict[key_k], nemo_state_dict[key_v]

    # Padding to new vocab size
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
        nemo_state_dict['model.language_model.embedding.word_embeddings.weight'] = padded_embedding

    return nemo_state_dict


def adjust_nemo_config(model_config, ref_config):
    model_config.tokenizer["type"] = "intfloat/e5-large-unsupervised"  # ref_config["_input_name_or_path"]
    model_config["num_layers"] = ref_config["num_hidden_layers"]
    model_config["hidden_size"] = ref_config["hidden_size"]
    model_config["ffn_hidden_size"] = ref_config["intermediate_size"]
    model_config["num_attention_heads"] = ref_config["num_attention_heads"]
    model_config["layernorm_epsilon"] = ref_config["layer_norm_eps"]
    model_config["normalization"] = "layernorm"
    model_config["transformer_block_type"] = "post_ln"
    model_config["apply_query_key_layer_scaling"] = False
    model_config["skip_head"] = True
    model_config["megatron_legacy"] = True
    return model_config


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input_name_or_path", type=str, default="thenlper/gte-large")
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_bert_config.yaml",
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument(
        "--precision", type=str, default="32", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )

    args = parser.parse_args()
    return args


def convert(args):
    logging.info(f"Loading checkpoint from HF: `{args.input_name_or_path}`")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.input_name_or_path)
    hf_model = AutoModel.from_pretrained(args.input_name_or_path)

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.model = adjust_nemo_config(nemo_config.model, hf_model.config.to_dict())

    nemo_config.trainer["precision"] = args.precision
    trainer = MegatronTrainerBuilder(nemo_config).create_trainer()
    model = MegatronBertModel(nemo_config.model, trainer)

    old_state_dict = hf_model.state_dict()
    rename_keys = create_rename_keys(nemo_config.model.num_layers)
    new_state_dict = rename_model_keys(model_state_dict=old_state_dict, rename_keys=rename_keys)
    nemo_state_dict = adjust_tensor_shapes(model, new_state_dict)
    model.load_state_dict(nemo_state_dict, strict=True)

    logging.info(f'=' * 50)
    # Verifications
    input_texts = [
        'query: how much protein should a female eat',
        'query: summit define',
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]

    # Tokenize the input texts
    batch_dict = hf_tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict_cuda = {k: v.cuda() for k, v in batch_dict.items()}
    hf_model = hf_model.cuda().eval()
    model = model.eval()
    with torch.no_grad():
        hf_outputs = hf_model(**batch_dict_cuda)
        embeddings_hf = average_pool(hf_outputs.last_hidden_state, batch_dict_cuda['attention_mask'])
        embeddings_hf = F.normalize(embeddings_hf, p=2, dim=1)

        outputs = model(**batch_dict_cuda)
        embeddings = average_pool(outputs[0], batch_dict_cuda['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    # Print difference between two embeddings
    print("Difference between reference embedding and converted embedding results:")
    print(embeddings - embeddings_hf)

    model.save_to(args.output_path)
    logging.info(f'NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
