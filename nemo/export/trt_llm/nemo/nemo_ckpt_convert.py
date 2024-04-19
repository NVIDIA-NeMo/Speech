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


import configparser
import logging
import math
import multiprocessing
import os
import typing
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import tensorstore  # This is important even though not used. Otherwise zarr raises error.
import torch
import zarr
from tensorrt_llm._utils import np_bfloat16, str_dtype_to_torch, torch_to_numpy
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer, LlamaConfig

from nemo.export.tarutils import TarPath, ZarrPathStore
from nemo.export.trt_llm.nemo.convert import save_weight_torch, split_and_save_weight
from nemo.export.trt_llm.nemo.nemo import UnpackedNemoCheckpointDir, extract_layers_with_prefix, nemo_to_llm_config
from nemo.export.trt_llm.nemo.sentencepiece_tokenizer import SentencePieceTokenizer

LOGGER = logging.getLogger("NeMo")

layer_names = {
    "position_embedding": "embedding.position_embeddings.weight",
    "word_embedding": "embedding.word_embeddings.weight",
    "output_layer": "output_layer.weight",
    "final_layernorm.weight": "final_layernorm.weight",
    "final_layernorm.bias": "final_layernorm.bias",
}


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


def get_layer_index(split_key):
    index = 0
    for key in split_key:
        if key == "layers":
            return index + 1
        index += 1


def rename_key(old_key: str, pp_rank: int, num_layers: int, pp_size: int):
    new_key = old_key

    if "layers." in old_key:
        split_key = old_key.split(".")
        layer_index = get_layer_index(split_key)
        split_key[layer_index] = str(int(split_key[layer_index]) + pp_rank * num_layers // pp_size)
        new_key = ".".join(split_key)

        if "self_attention" in new_key:
            new_key = new_key.replace("self_attention", "attention")
        if "attention.linear_qkv.layer_norm_weight" in new_key:
            new_key = new_key.replace("attention.linear_qkv.layer_norm_weight", "input_layernorm.weight")
        if "mlp.linear_fc1.layer_norm_weight" in new_key:
            new_key = new_key.replace("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight")

    return new_key


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


def load_sharded_metadata(checkpoint_dir: Union[Path, TarPath], torch_tensor=True):
    sharded_state_dict = {}
    for subdir in checkpoint_dir.iterdir():
        if not subdir.is_dir() or not (subdir / '.zarray').exists():
            continue
        key = subdir.name

        zstore = ZarrPathStore(subdir)
        arr = zarr.open(zstore, 'r')

        if torch_tensor:
            # sharded_state_dict[key] = torch.from_numpy(arr[:].astype("float32")).to(dtype=torch.bfloat16)
            if arr.dtype.name == "bfloat16":
                sharded_state_dict[key] = torch.from_numpy(arr[:].view(np.int16)).view(torch.bfloat16)
            else:
                sharded_state_dict[key] = torch.from_numpy(arr[:])
        else:
            sharded_state_dict[key] = arr[:]

    return sharded_state_dict


@torch.no_grad()
def convert_dist_checkpoint(unpacked_checkpoints_dir: UnpackedNemoCheckpointDir, args):
    nemo_model_config = unpacked_checkpoints_dir.model_config
    checkpoints_path = unpacked_checkpoints_dir.checkpoints_dir / "model_weights"

    # if checkpoints files could be found - start preparing output dir
    out_dir = create_out_dir(args)

    storage_type = str_dtype_to_torch(args.storage_type)
    is_mcore = nemo_model_config.get("mcore_gpt", False)

    # load position_embedding from rank 0
    model = load_sharded_metadata(checkpoints_path)
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
    inference_tp_size = args.tensor_parallelism
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
        and (args.decoder_type == "gptnext" or is_mcore),
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "kv_channels": kv_channels,
        "use_attention_nemo_shape": True,
        "transpose_weights": True,
    }

    # split_factor: in how many parts a TP training node is split
    split_factor = inference_tp_size
    model_level_weights = defaultdict(list)

    def handle_model_level_weights(model, tp_idx: int, pp_idx: int):
        if tp_idx == 0 and pp_idx == 0:
            if has_position_embedding:
                val = model[get_layer_name("position_embedding", prefix)]
                val = torch_to_numpy(val.to(storage_type).cpu())
                model_level_weights["model.wpe.bin"].append(val)
        if pp_idx == 0:
            val = model.get("state_dict", model)[get_layer_name("word_embedding", prefix)]
            if embedding_scaling:
                val = val * float(math.sqrt(hidden_size))

            val = torch_to_numpy(val.to(storage_type).cpu())
            model_level_weights["model.wte.bin"].append(val)
            if share_embeddings_and_output:
                val = model.get("state_dict", model)[get_layer_name("word_embedding", prefix)]
                val = torch_to_numpy(val.to(storage_type).cpu())
                model_level_weights["model.lm_head.weight.bin"].append(val)
        if has_lm_head and pp_idx == training_pp_size - 1:
            val = model.get("state_dict", model)[get_layer_name("output_layer", prefix)]
            val = torch_to_numpy(val.to(storage_type).cpu())
            model_level_weights["model.lm_head.weight.bin"].append(val)

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

    if args.processes > 1:
        with multiprocessing.Pool(args.processes) as pool:
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
    vocab_size = model_level_weights["model.wte.bin"].shape[0]

    if nemo_model_config["tokenizer"].get("library", None) == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(
            nemo_model_config["tokenizer"]["type"], use_fast=nemo_model_config["tokenizer"].get("use_fast", False)
        )
    else:
        tokenizer_config = update_tokenizer_paths(nemo_model_config["tokenizer"], unpacked_checkpoints_dir)
        copy_tokenizer_files(tokenizer_config, out_dir)

        tokenizer_config["model"] = os.path.join(out_dir, "tokenizer.model")
        tokenizer = build_tokenizer(tokenizer_config)

    llm_config = nemo_to_llm_config(
        nemo_model_config, vocab_size, tokenizer.eos_token_id, tokenizer.bos_token_id, args.decoder_type,
    )

    llm_config.is_mcore = is_mcore

    config = configparser.ConfigParser()
    decoder_name_dict = {"llama": "llama", "falcon": "falcon"}
    model_name = decoder_name_dict[args.decoder_type] if args.decoder_type in decoder_name_dict else "gpt"

    config[model_name] = {k: str(v) for k, v in vars(llm_config).items()}
    config[model_name]["storage_dtype"] = args.storage_type
    config_path = out_dir / "config.ini"
    with config_path.open("w") as config_file:
        config.write(config_file)

    return weights_dict, llm_config, tokenizer


@torch.no_grad()
def convert_nemo_model(nemo_model, nemo_model_config, storage_type_str, decoder_type=None):
    from megatron.core import parallel_state

    is_mcore = nemo_model_config.get("mcore_gpt", False)

    nemo_model_state_dict = nemo_model.state_dict()
    prefix, transformer_layer_prefix = get_layer_prefix(nemo_model_state_dict, is_mcore)
    has_position_embedding = get_layer_name("position_embedding", prefix) in nemo_model_state_dict
    has_lm_head = get_layer_name("output_layer", prefix) in nemo_model_state_dict
    has_final_layer_bias = get_layer_name("final_layernorm.bias", transformer_layer_prefix) in nemo_model_state_dict

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    # split_factor = 1
    storage_type = str_dtype_to_torch(storage_type_str)

    num_layers = nemo_model_config["num_layers"]
    training_tp_size = nemo_model_config.get("tensor_model_parallel_size", 1)
    training_pp_size = nemo_model_config.get("pipeline_model_parallel_size", 1)
    num_kv_heads = nemo_model_config.get("num_query_groups", 0)
    multi_query_mode = nemo_model_config.get("multi_query_mode", False)
    num_attention_heads = nemo_model_config["num_attention_heads"]

    # pp currently unsupported so reshard away PP
    is_pp_resharding = False
    if pp_size > 1:
        is_pp_resharding = True

    if num_kv_heads == 0:
        if multi_query_mode:
            num_kv_heads = 1
        else:
            num_kv_heads = num_attention_heads

    export_config = {
        "apply_layernorm_1p": nemo_model_config.get("normalization", "") == "layernorm1p",
        "tp_size": training_tp_size,
        "split_gated_activation": "swiglu" in nemo_model_config.get("activation", "gelu")
        and (decoder_type == "gptnext" or is_mcore),
        "num_attention_heads": nemo_model_config["num_attention_heads"],
        "num_kv_heads": num_kv_heads,
        "use_attention_nemo_shape": True,
        "transpose_weights": True,
        "from_nemo_model": True,
    }

    # Gather meta data from first and last PP stage
    if is_pp_resharding:
        has_lm_head = torch.tensor(has_lm_head).cuda()
        src_rank = torch.distributed.get_global_rank(pp_group, pp_size - 1)
        torch.distributed.broadcast(has_lm_head, src_rank, group=pp_group)
        has_lm_head = has_lm_head.item()

        has_position_embedding = torch.tensor(has_position_embedding).cuda()
        src_rank = torch.distributed.get_global_rank(pp_group, 0)
        torch.distributed.broadcast(has_position_embedding, src_rank, group=pp_group)
        has_position_embedding = has_position_embedding.item()

        has_final_layer_bias = torch.tensor(has_final_layer_bias).cuda()
        src_rank = torch.distributed.get_global_rank(pp_group, pp_size - 1)
        torch.distributed.broadcast(has_final_layer_bias, src_rank, group=pp_group)
        has_final_layer_bias = has_final_layer_bias.item()

    trt_inflight_weights = {}
    starmap_args = []

    def handle_model_level_weights(model, tp_idx: int, pp_idx: int):
        def _handle_weights(src_key: str, dst_key: str, pp_src_idx: int, tensor_dim: int):
            src_pp_global_rank = torch.distributed.get_global_rank(pp_group, pp_src_idx)
            # Broadcast the shape
            if pp_idx == pp_src_idx:
                gathered_tensor = model.get("state_dict", model)[src_key].type(storage_type).cuda()
                shape = torch.IntTensor(list(gathered_tensor.shape)).cuda()
            else:
                shape = torch.zeros(tensor_dim, dtype=torch.int32).cuda()
            torch.distributed.broadcast(shape, src_pp_global_rank, group=pp_group)

            # Collect the tensor
            if pp_idx != pp_src_idx:
                gathered_tensor = torch.zeros(*shape, dtype=storage_type).cuda()
            torch.distributed.broadcast(gathered_tensor, src_pp_global_rank, group=pp_group)

            if "final_layernorm" not in src_key:
                gathered_tensor = gathered_tensor.to(storage_type).cpu()
                trt_inflight_weights[dst_key] = torch_to_numpy(gathered_tensor)
            else:
                starmap_args.append(
                    {
                        "tp_rank": tp_idx,
                        "saved_dir": trt_inflight_weights,
                        "split_factor": 1,
                        "key": dst_key,
                        "vals": [gathered_tensor],
                        "storage_type": storage_type,
                        "act_range": None,
                        "config": export_config,
                    }
                )

        if has_lm_head:
            _handle_weights(get_layer_name("output_layer", prefix), "model.lm_head.weight.bin", pp_size - 1, 2)
        if has_position_embedding:
            _handle_weights(get_layer_name("position_embedding", prefix), "model.wpe.bin", 0, 2)

        _handle_weights(get_layer_name("word_embedding", prefix), "model.wte.bin", 0, 2)
        _handle_weights(
            get_layer_name("final_layernorm.weight", transformer_layer_prefix),
            "final_layernorm.weight",
            pp_size - 1,
            1,
        )

        if has_final_layer_bias:
            _handle_weights(
                get_layer_name("final_layernorm.bias", transformer_layer_prefix),
                "final_layernorm.bias",
                pp_size - 1,
                1,
            )

        torch.cuda.empty_cache()

    models = []

    handle_model_level_weights(nemo_model_state_dict, tp_rank, pp_rank)
    layers = extract_layers_with_prefix(nemo_model_state_dict, transformer_layer_prefix)
    models.append(layers)

    for key in models[0].keys():
        # Skip final_layernorm.
        if not key.startswith("layers."):
            continue
        if "_extra_state" not in key:
            starmap_args.append(
                {
                    "tp_rank": tp_rank,
                    "saved_dir": trt_inflight_weights,
                    "split_factor": 1,
                    "key": rename_key(key, pp_rank, num_layers, training_pp_size),
                    "vals": [model[key] for model in models],
                    "storage_type": storage_type,
                    "act_range": None,
                    "config": export_config,
                }
            )
    starmap_args = tqdm(starmap_args, desc="saving weights")
    for starmap_arg in starmap_args:
        save_weight_torch(**starmap_arg)

    # Collect weights from different pp stages
    # Assume each rank has the same number of layers
    if is_pp_resharding:
        collect_pp_weights = {}
        for key, val in trt_inflight_weights.items():
            # Skip embedding and final layer
            if not key.startswith("model.layers"):
                continue
            # Convert numpy array to torch tensor and gather weights
            curr_weight = trt_inflight_weights[key]
            if curr_weight.dtype != np_bfloat16:
                curr_weight = torch.tensor(curr_weight).cuda()
            else:
                curr_weight = torch.tensor(curr_weight.view(np.int16)).view(torch.bfloat16).cuda()
            weight_list = [torch.zeros_like(curr_weight) for _ in range(pp_size)]
            torch.distributed.all_gather(weight_list, curr_weight, group=pp_group)
            # Collect weights name
            for rank in range(pp_size):
                split_key = key.split(".")
                layer_index = get_layer_index(split_key)
                split_key[layer_index] = str(int(split_key[layer_index]) + (rank - pp_rank) * num_layers // pp_size)
                new_key = ".".join(split_key)
                collect_pp_weights[new_key] = torch_to_numpy(weight_list[rank].to(storage_type).cpu())

        trt_inflight_weights.update(collect_pp_weights)

    vocab_size = trt_inflight_weights["model.wte.bin"].shape[0] * tp_size

    llm_config = nemo_to_llm_config(
        nemo_model_config,
        vocab_size,
        None,
        None,
        decoder_type=decoder_type,  # how to get eos_id and bos_id from different tokenizer?
    )
    llm_config.is_mcore = is_mcore

    config = configparser.ConfigParser()
    model_name = "llama" if isinstance(llm_config, LlamaConfig) else "gpt"
    config[model_name] = {k: str(v) for k, v in vars(llm_config).items()}
    config[model_name]["storage_dtype"] = storage_type_str

    return trt_inflight_weights, llm_config


def create_out_dir(args):
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    return out_dir


def update_tokenizer_paths(tokenizer_config: typing.Dict, unpacked_checkpoints_dir):
    def _update_config_entry(key, file_pattern):
        old_path = tokenizer_config[key]
        if old_path is None:
            return
        old_path = Path(old_path)
        new_path = unpacked_checkpoints_dir.get_tokenizer_file_path("tokenizer", key, file_pattern)
        if new_path:
            LOGGER.debug(f"Update tokenizer {key} {old_path} -> {new_path}")
            tokenizer_config[key] = new_path
        elif not old_path.exists():
            LOGGER.warning(f"Tokenizer {key}'s path {old_path} does not exists: set it to None")
            tokenizer_config[key] = None

    _update_config_entry("model", "*.model")
    _update_config_entry("vocab_file", "*vocab*")
    _update_config_entry("merge_file", "*merge*.txt")

    return tokenizer_config


def copy_tokenizer_files(config, out_dir):
    basenames = {
        "model": "tokenizer",
        "vocab_file": "vocab",
        "merge_file": "merges",
    }

    for key in basenames.keys():
        if config[key] is None:
            continue

        path = config[key]

        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            LOGGER.debug(f"Tokenizer {key}: {path} file not found")
            continue

        dst_path = out_dir / f"{basenames[key]}{path.suffix}"
        LOGGER.debug(f"Copy tokenizer {key}: {path}->{dst_path}")

        # Copy 'path' to 'dst_path' without shutil.copy(...) because 'path' may be a TarPath
        with path.open('rb') as infile:
            with open(dst_path, 'wb') as outfile:
                outfile.write(infile.read())


def build_tokenizer(tokenizer):
    if isinstance(tokenizer, dict):
        tokenizer_config = tokenizer
        if tokenizer_config["library"] == "sentencepiece":
            return SentencePieceTokenizer(model_path=tokenizer_config["model"])
        elif "GPT2" in tokenizer_config["type"]:
            tokenizer = GPT2Tokenizer(tokenizer_config["vocab_file"], tokenizer_config["merge_file"])
        else:
            raise ValueError(f'Tokenizer type {tokenizer_config["library"]} not handled')

        if tokenizer.bos_token_id is None:
            tokenizer.add_special_tokens({"bos_token": "<s>"})
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
    else:
        try:
            # If NeMo tokenizer, monkey patch interface
            from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

            if isinstance(tokenizer, TokenizerSpec):

                def batch_encode_patch(self, ids):
                    if torch.is_tensor(ids):
                        ids = ids.cpu().numpy()
                    return self.ids_to_text(ids)

                tokenizer.bos_token_id = tokenizer.bos_id
                tokenizer.eos_token_id = tokenizer.eos_id
                tokenizer.encode = tokenizer.text_to_ids
                TokenizerSpec.batch_decode = batch_encode_patch
        except:
            raise TypeError(f'Unsupported tokenizer build input: {type(tokenizer)}')

    return tokenizer
