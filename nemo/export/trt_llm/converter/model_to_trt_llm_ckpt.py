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

from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.export.trt_llm.converter.utils import save_val, split_and_save_weight, weights_dict

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


def rename_key(new_key: str):
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


def rename_key_dist_ckpt(old_key: str, layer: int):
    new_key = old_key
    if "layers." in old_key:
        split_key = old_key.split(".")
        split_key.insert(1, str(layer))
        new_key = ".".join(split_key)

    return rename_key(new_key)


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

            vocab_size = val.shape[0]
            if use_parallel_embedding:
                # Pad vocab_size first
                if vocab_size % inference_tp_size != 0:
                    vocab_size_padded = pad_vocab_size(vocab_size, inference_tp_size)
                    pad_width = vocab_size_padded - vocab_size
                    val = torch.nn.functional.pad(val, (0, 0, 0, pad_width), value=0)

            val = torch_to_numpy(val.to(storage_type).cpu())
            model_level_weights["transformer.vocab_embedding.weight"].append(val)
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


def _get_layer_index(split_key):
    for index, key in enumerate(split_key):
        if key == "layers":
            return index + 1
    raise ValueError(f"Unknown layer name format: {split_key}")


def rename_layer_num(param_name, layer_num):
    split_key = param_name.split(".")
    layer_index = int(_get_layer_index(split_key))
    split_key[layer_index] = str(layer_num)
    return ".".join(split_key)


def get_layer_num(param_name):
    split_key = param_name.split(".")
    layer_index = int(_get_layer_index(split_key))
    return int(split_key[layer_index])


@torch.no_grad()
def dist_model_to_trt_llm_ckpt(
    model,
    nemo_model_config,
    inference_tp_size,
    inference_pp_size,
    tokenizer_vocab_size,
):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.utils import VocabUtility

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
    pp_last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_is_last = parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    pp_is_first = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
    vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
    if not vp_size:
        vp_size = 1

    reshard_model = False
    if inference_tp_size != tp_size or inference_pp_size != pp_size:
        LOGGER.info("Training/Generation model parallelism resharding enabled")
        if inference_pp_size == 1 and pp_size > 1 and inference_tp_size == tp_size:
            reshard_model = True
        else:
            raise NotImplementedError(
                f"NeMo currently only supports PP>1 -> PP=1 resharding, other types of resharding will come in future releases."
            )

    num_layers = nemo_model_config["num_layers"]
    is_mcore = nemo_model_config.get("mcore_gpt", False)
    storage_type = torch_dtype_from_precision(nemo_model_config.precision)
    sample_state_dict = model[0].state_dict() if vp_size > 1 else model.state_dict()
    prefix, transformer_layer_prefix = get_layer_prefix(sample_state_dict, is_mcore)
    assert is_mcore, "Only megatron-core inflight model conversion is supported"

    export_config = {
        "apply_layernorm_1p": nemo_model_config.get("normalization", "") == "layernorm1p",
        "tp_size": tp_size,
        "split_gated_activation": nemo_model_config.get("activation", "gelu")
        in ["swiglu", "geglu", "fast-swiglu", "fast-geglu"],
        "num_attention_heads": nemo_model_config["num_attention_heads"],
        "num_kv_heads": nemo_model_config.get('num_query_groups', nemo_model_config['num_attention_heads']),
        "convert_on_device": True,
        "use_attention_nemo_shape": True,
        "transpose_weights": True,
    }

    starmap_config = {
        "tp_rank": None,
        "saved_dir": None,  # unused
        "split_factor": 0,
        "storage_type": storage_type,
        "act_range": None,
        "config": export_config,
    }

    tl_params = {}
    model_level_params = {}
    starmap_args = []
    layers_per_pp = num_layers // pp_size
    layers_per_chunk = layers_per_pp // vp_size

    if vp_size > 1:  # consolidate params across model chunks
        for idx, model_chunk in enumerate(model):
            for key, val in model_chunk.state_dict().items():
                if torch.is_tensor(val):
                    if 'layers' in key:
                        key2 = rename_layer_num(key, get_layer_num(key) + idx * pp_size * layers_per_chunk)
                        tl_params[key2] = val
                    else:
                        model_level_params[key] = val
    else:
        for key, val in model.state_dict().items():
            if torch.is_tensor(val):
                if 'decoder.layers' in key:
                    tl_params[key] = val
                else:
                    model_level_params[key] = val

    if vp_size > 1 or reshard_model:
        # gather layers across pp ranks
        gathered_params = {}
        for key, val in tl_params.items():
            weight_list = [torch.zeros_like(val) for _ in range(pp_size)]
            torch.distributed.all_gather(weight_list, val, group=pp_group)
            for idx in range(pp_size):
                layer_num = get_layer_num(key) + idx * layers_per_chunk
                key2 = rename_layer_num(key, layer_num)
                if not reshard_model:  # Save only layers of 1 single PP stage
                    layers_start = layers_per_pp * pp_rank
                    layers_end = layers_per_pp * (pp_rank + 1) - 1
                    if layer_num >= layers_start and layer_num <= layers_end:
                        key2 = rename_layer_num(key, layer_num % layers_per_pp)
                        gathered_params[key2] = weight_list[idx]
                else:
                    gathered_params[key2] = weight_list[idx]
        tl_params = gathered_params

    # ----------------Convert layer level weights----------------
    layer_params = extract_layers_with_prefix(tl_params, transformer_layer_prefix)
    layer_params = {k: v for k, v in layer_params.items() if k.startswith("layers.")}
    for key, val in layer_params.items():
        starmap_args.append(starmap_config | {'key': rename_key(key), 'vals': val})

    def broadcast_item(item, group, src_rank):
        item = [item]
        torch.distributed.broadcast_object_list(item, src_rank, group=group)
        return item[0]

    def try_get_model_level_weight(src_key_or_tensor, pp_src_idx):
        have_tensor = False
        if torch.distributed.get_rank() == pp_src_idx:
            if isinstance(src_key_or_tensor, str):
                tensor = model_level_params.get(src_key_or_tensor, None)
                have_tensor = torch.is_tensor(tensor)
            else:
                assert torch.is_tensor(src_key_or_tensor)
                tensor = src_key_or_tensor
                have_tensor = True
        if reshard_model:
            have_tensor = broadcast_item(have_tensor, pp_group, pp_src_idx)
        if not have_tensor:
            return None

        if reshard_model:  # Broadcast tensor to all PP groups
            if torch.distributed.get_rank() == pp_src_idx:
                shape = tensor.shape
            else:
                shape = [None]
            shape = broadcast_item(shape, pp_group, pp_src_idx)
            if torch.distributed.get_rank() != pp_src_idx:
                tensor = torch.zeros(shape, dtype=storage_type).cuda()
            torch.distributed.broadcast(tensor.contiguous(), pp_src_idx, group=pp_group)
        return tensor

    # ----------------Convert Final Layernorm----------------
    if pp_is_last or reshard_model:
        ln_f = try_get_model_level_weight(
            get_layer_name("final_layernorm.weight", transformer_layer_prefix), pp_last_rank
        )
        if ln_f is not None:
            starmap_args.append(starmap_config | {'key': "final_layernorm.weight", 'vals': ln_f})

        ln_f_bias = try_get_model_level_weight(
            get_layer_name("final_layernorm.bias", transformer_layer_prefix), pp_last_rank
        )
        if ln_f_bias is not None:
            starmap_args.append(starmap_config | {'key': "final_layernorm.bias", 'vals': ln_f_bias})

    # ----------------Convert Embeddings----------------
    def get_remove_vocab_padding(tensor_name):
        tensor = model_level_params.get(tensor_name, None)
        if tensor is None:
            return None

        if tp_size > 1:  # Gather padded tensor chunks
            vocab_size_padded = tensor.shape[0] * tp_size
            vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                vocab_size_padded, tp_rank, tp_size
            )
            dim_size = list(tensor.size())
            dim_size[0] = vocab_size_padded
            gathered_tensor = torch.zeros(dim_size, dtype=tensor.dtype, device=torch.cuda.current_device())
            gathered_tensor[vocab_start_index:vocab_end_index] = tensor
            torch.distributed.all_reduce(gathered_tensor, group=tp_group)
            tensor = gathered_tensor
        unpadded = tensor[:tokenizer_vocab_size]
        if tp_size > 1:  # Split gathered tensor for tensor parallel embedding
            vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                tokenizer_vocab_size, tp_rank, tp_size
            )
            unpadded = unpadded[vocab_start_index:vocab_end_index]
        return unpadded.T  # TRTLLM expects (vocab_size, hidden_size) so need extra transpose

    if pp_is_first or reshard_model:
        vocab_embed = get_remove_vocab_padding(get_layer_name("word_embedding", prefix))
        vocab_embed = try_get_model_level_weight(vocab_embed, pp_first_rank)
        save_val(vocab_embed, dir=None, key='transformer.vocab_embedding.weight', tp_num=None)

    if pp_is_last or reshard_model:
        lm_head = get_remove_vocab_padding(get_layer_name("output_layer", prefix))
        lm_head = try_get_model_level_weight(lm_head, pp_last_rank)
        save_val(lm_head, dir=None, key='lm_head.weight', tp_num=None)

    for starmap_arg in tqdm(starmap_args, desc="saving weights"):
        split_and_save_weight(**starmap_arg)

    return weights_dict


def create_export_dir(nemo_export_dir):
    out_dir = Path(nemo_export_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    return out_dir
