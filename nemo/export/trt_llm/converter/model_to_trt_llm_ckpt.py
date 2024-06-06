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


import logging
import math
import multiprocessing
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_torch, torch_to_numpy
from tqdm import tqdm

from nemo.export.trt_llm.converter.utils import split_and_save_weight

LOGGER = logging.getLogger("NeMo")

layer_names = {
    "position_embedding": "embedding.position_embeddings.weight",
    "word_embedding": "embedding.word_embeddings.weight",
    "output_layer": "output_layer.weight",
    "final_layernorm.weight": "final_layernorm.weight",
    "final_layernorm.bias": "final_layernorm.bias",
}


def extract_layers_with_prefix(model_, prefix):
    length_to_trim = len(prefix)
    model_state = model_.get("state_dict", model_)
    return {key[length_to_trim:]: model_state[key] for key in model_state.keys() if prefix in key}


def get_layer_name(layer_type: str, prefix: str):
    layer_dict = layer_names
    if layer_type in layer_dict:
        return prefix + layer_dict[layer_type]
    else:
        raise ValueError(f"Unknown layer type {layer_type}")


def get_layer_prefix(layer_names, is_mcore):
    transformer_layer_prefix = None

    for layer_name in layer_names:
        if 'self_attention' in layer_name:
            transformer_layer_prefix = layer_name.split('layers')[0]
            break
    assert transformer_layer_prefix is not None, "Cannot extract transformer layer prefix from {layer_name}"
    if is_mcore:
        model_prefix = transformer_layer_prefix.split('decoder')[0]
    else:
        model_prefix = transformer_layer_prefix.split('encoder')[0]
    assert model_prefix is not None, "Cannot extract model prefix from {layer_name}"

    return model_prefix, transformer_layer_prefix


def rename_key_dist_ckpt(old_key: str, layer: int):
    new_key = old_key

    if "layers." in old_key:
        split_key = old_key.split(".")
        split_key.insert(1, str(layer))
        new_key = ".".join(split_key)

        if "self_attention" in new_key:
            new_key = new_key.replace("self_attention", "attention")
        if "attention.linear_qkv.layer_norm_weight" in new_key:
            new_key = new_key.replace("attention.linear_qkv.layer_norm_weight", "input_layernorm.weight")
        if "attention.linear_qkv.layer_norm_bias" in new_key:
            new_key = new_key.replace("attention.linear_qkv.layer_norm_bias", "input_layernorm.bias")
        if "mlp.linear_fc1.layer_norm_weight" in new_key:
            new_key = new_key.replace("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight")
        if "mlp.linear_fc1.layer_norm_bias" in new_key:
            new_key = new_key.replace("mlp.linear_fc1.layer_norm_bias", "post_attention_layernorm.bias")

    return new_key


@torch.no_grad()
def convert_model_to_trt_llm_ckpt(
    nemo_model_config,
    model,
    nemo_export_dir,
    storage_type,
    inference_tp_size,
    decoder_type,
    use_parallel_embedding,
    processes,
):

    # if checkpoints files could be found - start preparing output dir
    out_dir = create_export_dir(nemo_export_dir)
    storage_type = str_dtype_to_torch(storage_type)
    is_mcore = nemo_model_config.get("mcore_gpt", False)

    # load position_embedding from rank 0
    model_state_dict = model.get("state_dict", model)

    prefix, transformer_layer_prefix = get_layer_prefix(model_state_dict.keys(), is_mcore)

    has_position_embedding = get_layer_name("position_embedding", prefix) in model_state_dict
    has_lm_head = get_layer_name("output_layer", prefix) in model_state_dict
    share_embeddings_and_output = nemo_model_config.get("share_embeddings_and_output_weights", False)
    embedding_scaling = nemo_model_config.get("apply_embedding_scaling", False)
    hidden_size = nemo_model_config["hidden_size"]

    num_layers = nemo_model_config["num_layers"]
    training_tp_size = 1
    training_pp_size = 1
    num_kv_heads = nemo_model_config.get("num_query_groups", 0)
    multi_query_mode = nemo_model_config.get("multi_query_mode", False)
    num_attention_heads = nemo_model_config["num_attention_heads"]
    kv_channels = nemo_model_config.get("kv_channels", None)

    if num_kv_heads == 0:
        if multi_query_mode:
            num_kv_heads = 1
        else:
            num_kv_heads = num_attention_heads

    export_config = {
        "apply_layernorm_1p": nemo_model_config.get("normalization", "") == "layernorm1p",
        "tp_size": training_tp_size,
        "split_gated_activation": nemo_model_config.get("activation", "gelu")
        in ["swiglu", "geglu", "fast-swiglu", "fast-geglu"]
        and (decoder_type == "gptnext" or is_mcore),
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "kv_channels": kv_channels,
        "use_attention_nemo_shape": True,
        "transpose_weights": True,
        "use_parallel_embedding": use_parallel_embedding,
    }

    # split_factor: in how many parts a TP training node is split
    split_factor = inference_tp_size
    model_level_weights = defaultdict(list)

    def handle_model_level_weights(model, tp_idx: int, pp_idx: int):
        if tp_idx == 0 and pp_idx == 0:
            if has_position_embedding:
                val = model[get_layer_name("position_embedding", prefix)]
                val = torch_to_numpy(val.to(storage_type).cpu())
                model_level_weights["transformer.position_embedding.weight"].append(val)
        if pp_idx == 0:
            val = model.get("state_dict", model)[get_layer_name("word_embedding", prefix)]
            if embedding_scaling:
                val = val * float(math.sqrt(hidden_size))

            vocab_size = val.shape[0]
            if use_parallel_embedding:
                # Pad vocab_size first
                if vocab_size % inference_tp_size != 0:
                    vocab_size_padded = pad_vocab_size(vocab_size, inference_tp_size)
                    pad_width = vocab_size_padded - vocab_size
                    val = torch.nn.functional.pad(val, (0, 0, 0, pad_width), value=0)

            val = torch_to_numpy(val.to(storage_type).cpu())
            model_level_weights["transformer.vocab_embedding.weight"].append(val)
            if share_embeddings_and_output:
                val = model.get("state_dict", model)[get_layer_name("word_embedding", prefix)]
                val = torch_to_numpy(val.to(storage_type).cpu())
                model_level_weights["lm_head.weight"].append(val)
        if has_lm_head and pp_idx == training_pp_size - 1:
            val = model.get("state_dict", model)[get_layer_name("output_layer", prefix)]
            val = torch_to_numpy(val.to(storage_type).cpu())
            model_level_weights["lm_head.weight"].append(val)

    weights_dict = {}

    tp_rank = 0

    handle_model_level_weights(model, 0, 0)
    model = extract_layers_with_prefix(model, transformer_layer_prefix)

    starmap_args = []
    for key, val in model.items():
        if "_extra_state" not in key:
            if len(val.size()) == 1:
                starmap_args.append(
                    (
                        tp_rank,
                        out_dir,
                        split_factor,
                        # Let's rename/map the key to the old layer name previously. You can try printing out
                        # the rename_key output of the old llama checkpoint and compare.
                        rename_key_dist_ckpt(key, 0),
                        # Since the state dict value has the full layers, let's select the ith layer weights/biases here.
                        [val],
                        storage_type,
                        None,
                        export_config,
                    )
                )
            else:
                for i in range(num_layers):
                    starmap_args.append(
                        (
                            tp_rank,
                            out_dir,
                            split_factor,
                            # Let's rename/map the key to the old layer name previously. You can try printing out
                            # the rename_key output of the old llama checkpoint and compare.
                            rename_key_dist_ckpt(key, i),
                            # Since the state dict value has the full layers, let's select the ith layer weights/biases here.
                            [val[i]],
                            storage_type,
                            None,
                            export_config,
                        )
                    )

    starmap_args = tqdm(starmap_args, desc="saving weights")

    if processes > 1:
        with multiprocessing.Pool(processes) as pool:
            weights_dicts = pool.starmap(split_and_save_weight, starmap_args)
            weights_dict_local = {k: v for d in weights_dicts for k, v in d.items()}
    else:
        # simpler for debug situations
        for starmap_arg in starmap_args:
            weights_dict_local = split_and_save_weight(*starmap_arg)

    weights_dict.update(weights_dict_local)

    for key, values in model_level_weights.items():
        model_level_weights[key] = np.concatenate(values, axis=0)
        weights_dict[key] = model_level_weights[key]

    return weights_dict


def create_export_dir(nemo_export_dir):
    out_dir = Path(nemo_export_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    return out_dir
