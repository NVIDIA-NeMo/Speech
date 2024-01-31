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
     --input_name_or_path /path/to/input/nemo/file.nemo \
     --output_path /path/to/output/huggingface/file \
     --precision 32
```
"""

from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from transformers import AutoTokenizer, BertConfig, BertModel

from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
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
    rename_keys (list): A list of tuples with the mapping (new_key, old_key).

    Returns:
    dict: A new state dictionary with updated key names.
    """

    # Create a new state dictionary with updated key names
    new_state_dict = {}

    # Track keys from the original state dict to ensure all are processed
    remaining_keys = set(model_state_dict.keys())

    # Iterate over the rename mappings
    for new_key, old_key in rename_keys:
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


def adjust_tensor_shapes(model_state_dict):
    """
    Adapt tensor shapes in the state dictionary to ensure compatibility with a different model structure.

    Parameters:
    nemo_state_dict (dict): The state dictionary of the model.

    Returns:
    dict: The updated state dictionary with modified tensor shapes for compatibility.
    """

    # Note: For 'key' and 'value' weights and biases, NeMo uses a consolidated tensor 'query_key_value'.
    for key_ in list(model_state_dict.keys()):
        if "self_attention.query_key_value" in key_:
            key_q = key_.replace('self_attention.query_key_value', 'self_attention.query')
            key_k = key_.replace('self_attention.query_key_value', 'self_attention.key')
            key_v = key_.replace('self_attention.query_key_value', 'self_attention.value')
            local_dim = model_state_dict[key_].shape[0] // 3
            q, k, v = model_state_dict[key_].split(local_dim)
            model_state_dict[key_q] = q
            model_state_dict[key_k] = k
            model_state_dict[key_v] = v
            del model_state_dict[key_]

    return model_state_dict


def convert_config(ref_config, hf_state_dict):
    vocab_size = hf_state_dict['embeddings.word_embeddings.weight'].shape[0]
    new_config = {
        "vocab_size": vocab_size,
        "num_hidden_layers": ref_config["num_layers"],
        "hidden_size": ref_config["hidden_size"],
        "intermediate_size": ref_config["ffn_hidden_size"],
        "num_attention_heads": ref_config["num_attention_heads"],
        "layer_norm_eps": ref_config["layernorm_epsilon"],
        "max_position_embeddings": ref_config["max_position_embeddings"],
    }
    hf_config = BertConfig(**new_config)
    return hf_config


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path", type=str, required=True, help="Path to .nemo file",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output HF model path",
    )

    args = parser.parse_args()
    return args


def convert(args):
    logging.info(f"Loading checkpoint from: `{args.input_name_or_path}`")
    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    nemo_model = MegatronBertModel.restore_from(args.input_name_or_path, trainer=dummy_trainer)
    nemo_config = nemo_model.cfg

    old_state_dict = nemo_model.state_dict()
    rename_keys = create_rename_keys(nemo_config.num_layers)
    new_state_dict = adjust_tensor_shapes(old_state_dict)
    hf_state_dict = rename_model_keys(model_state_dict=new_state_dict, rename_keys=rename_keys)

    hf_config = convert_config(nemo_config, hf_state_dict)
    hf_model = BertModel(hf_config)

    hf_model.load_state_dict(hf_state_dict, strict=True)

    logging.info(f'=' * 50)
    # Verifications
    input_texts = [
        'query: how much protein should a female eat',
        'query: summit define',
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]

    # Tokenize the input texts
    hf_tokenizer = AutoTokenizer.from_pretrained(nemo_config.tokenizer["type"])
    batch_dict = hf_tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict_cuda = {k: v.cuda() for k, v in batch_dict.items()}
    hf_model = hf_model.cuda().eval()
    nemo_model = nemo_model.eval()
    with torch.no_grad():
        hf_outputs = hf_model(**batch_dict_cuda)
        embeddings_hf = average_pool(hf_outputs.last_hidden_state, batch_dict_cuda['attention_mask'])
        embeddings_hf = F.normalize(embeddings_hf, p=2, dim=1)

        outputs = nemo_model(**batch_dict_cuda)
        embeddings = average_pool(outputs[0], batch_dict_cuda['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    # Print difference between two embeddings
    print("Difference between reference embedding and converted embedding results:")
    print(embeddings - embeddings_hf)

    hf_model.save_pretrained(args.output_path)
    logging.info(f'Full HF model model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
