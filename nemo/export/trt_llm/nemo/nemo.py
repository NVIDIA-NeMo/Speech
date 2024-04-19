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


import functools
import logging
import pathlib
import typing

import torch
import yaml
from transformers import FalconConfig, GPT2Config, LlamaConfig

from nemo.export.tarutils import TarPath
from nemo.export.trt_llm.nemo.convert import cpu_map_location, gpu_map_location

LOGGER = logging.getLogger("NeMo")


def nemo_to_llm_config(nemo_model_config, vocab_size, eos_id, bos_id, decoder_type):
    convertion_dict = {
        "activation_function": "activation",
        "layer_norm_epsilon": "layernorm_epsilon",
        "n_embd": "hidden_size",
        "n_head": "num_attention_heads",
        "n_layer": "num_layers",
        "n_positions": "max_position_embeddings",
        "rotary_pct": "rotary_percentage",
        "rotary_base": "rotary_base",
        "rotary_scaling": "seq_len_interpolation_factor",
        "position_embedding_type": "position_embedding_type",
        "bias": "bias",
        "intermediate_size": "ffn_hidden_size",
        "num_kv_heads": "num_query_groups",
        "moe_num_experts": "num_moe_experts",
        "moe_top_k": "moe_router_topk",
        "moe_renorm_mode": "moe_renorm_mode",
        "kv_channels": "kv_channels",
        "norm_epsilon": "layernorm_epsilon",
    }

    kwargs = {key: nemo_model_config[value] for key, value in convertion_dict.items() if value in nemo_model_config}
    kwargs["vocab_size"] = vocab_size
    kwargs["eos_token_id"] = eos_id
    kwargs["bos_token_id"] = eos_id if decoder_type == 'falcon' else bos_id  # in HF falcon eos==bos
    if "moe_num_experts" not in kwargs:
        kwargs["moe_num_experts"] = 0
    config_dict = {"llama": LlamaConfig, "falcon": FalconConfig, "gemma": LlamaConfig}
    llm_config = config_dict[decoder_type] if decoder_type in config_dict else GPT2Config

    return llm_config(**kwargs)


def add_special_tokens_to_tokenizer(tokenizer):
    # Need to add cls, sep, mask tokens to the tokenizer if they don't exist.
    # If cls, sep and mask are not attributes of the tokenizer, add it.
    if not hasattr(tokenizer, "cls_token"):
        tokenizer.add_special_tokens({"cls_token": "<cls>"})
    if not hasattr(tokenizer.tokenizer, "sep_id"):
        tokenizer.add_special_tokens({"sep_token": "<sep>"})
    if not hasattr(tokenizer.tokenizer, "mask_id"):
        tokenizer.add_special_tokens({"mask_token": "<mask>"})

    # bos, eos, pad and unk may be present in the provided spm .model file, if they are, use it.
    if not hasattr(tokenizer, "pad_token"):
        if hasattr(tokenizer.tokenizer, "pad_id") and tokenizer.tokenizer.pad_id() > 0:
            tokenizer.pad_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.pad_id())
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    else:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    if not hasattr(tokenizer, "bos_token"):
        if hasattr(tokenizer.tokenizer, "bos_id") and tokenizer.tokenizer.bos_id() > 0:
            tokenizer.bos_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.bos_id())
        else:
            tokenizer.add_special_tokens({"bos_token": "<bos>"})
    else:
        tokenizer.add_special_tokens({"bos_token": "<s>"})

    if not hasattr(tokenizer, "eos_token"):
        if hasattr(tokenizer.tokenizer, "eos_id") and tokenizer.tokenizer.eos_id() > 0:
            tokenizer.eos_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.eos_id())
        else:
            tokenizer.add_special_tokens({"eos_token": "<eos>"})
    else:
        tokenizer.add_special_tokens({"eos_token": "</s>"})


def extract_layers_with_prefix(model_, prefix):
    length_to_trim = len(prefix)
    model_state = model_.get("state_dict", model_)
    return {key[length_to_trim:]: model_state[key] for key in model_state.keys() if prefix in key}


class UnpackedNemoCheckpointDir:
    def __init__(
        self, checkpoints_dir: typing.Union[pathlib.Path, TarPath], load_checkpoints_to_cpu: bool = False,
    ):
        assert isinstance(checkpoints_dir, (pathlib.Path, TarPath))
        self._checkpoints_dir = checkpoints_dir
        self._load_checkpoints_to_cpu = load_checkpoints_to_cpu

    @property
    @functools.lru_cache
    def model_config(self):
        model_config = None

        model_config_filename = "model_config.yaml"
        model_configs_paths = list(self._checkpoints_dir.rglob(model_config_filename))
        if model_configs_paths:
            if len(model_configs_paths) > 1:
                raise RuntimeError(
                    f"There are more than single {model_config_filename} in"
                    f" {self._checkpoints_dir}:"
                    f" {', '.join(map(lambda p: p.as_posix(), model_configs_paths))}"
                )
            model_config_path = model_configs_paths[0]
            LOGGER.debug("Loading model config from %s", model_config_path)
            with model_config_path.open("r") as model_config_file:
                model_config = yaml.load(model_config_file, Loader=yaml.SafeLoader)
        else:
            LOGGER.debug("Searching model config in checkpoints")
            # try to obtain from checkpoint
            checkpoint_name = self.checkpoint_name
            checkpoints_paths = sorted(self._checkpoints_dir.rglob(checkpoint_name))
            if checkpoints_paths:
                # assume that parallel ranks 0 checkpoint should have model config embedded
                checkpoint_path = checkpoints_paths[0]

                map_location_fn = cpu_map_location if self._load_checkpoints_to_cpu else gpu_map_location

                model_00 = torch.load(checkpoint_path, map_location=map_location_fn)
                if "hyper_parameters" in model_00 and "cfg" in model_00["hyper_parameters"]:
                    model_config = model_00["hyper_parameters"]["cfg"]
                    LOGGER.debug("Loaded model config from checkpoint %s", checkpoint_path)
                else:
                    LOGGER.debug("Could not find model config in checkpoint %s", checkpoint_path)

                del model_00

        if model_config is None:
            LOGGER.warning("Could not find checkpoint with NeMo model config in %s", self._checkpoints_dir)

        LOGGER.debug("Loaded model config %s", model_config)

        return model_config

    @property
    def checkpoints_dir(self):
        return self._checkpoints_dir

    def get_checkpoints_paths(self, tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
        """Injects tensor/pipeline model parallel ranks into the filepath.
        Does nothing if not using model parallelism.
        """
        checkpoint_path_without_rank = self.checkpoints_dir / self.checkpoint_name

        def _inject_parallel_ranks(tp_rank, pp_rank):
            if tensor_model_parallel_size > 1 or pipeline_model_parallel_size > 1:
                if pipeline_model_parallel_size is None or pipeline_model_parallel_size == 1:
                    checkpoint_path = (
                        checkpoint_path_without_rank.parent
                        / f"mp_rank_{tp_rank:02d}"
                        / checkpoint_path_without_rank.name
                    )
                else:
                    checkpoint_path = (
                        checkpoint_path_without_rank.parent
                        / f"tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}"
                        / checkpoint_path_without_rank.name
                    )
                return checkpoint_path
            else:
                return checkpoint_path_without_rank

        return [
            [
                _inject_parallel_ranks(tp_rank=tp_rank, pp_rank=pp_rank)
                for pp_rank in range(pipeline_model_parallel_size)
            ]
            for tp_rank in range(tensor_model_parallel_size)
        ]

    @property
    @functools.lru_cache
    def checkpoint_name(self):
        patterns = [
            "model_weights.ckpt",  # older megatron checkpoints
            "*last.ckpt",  # newer format of checkpoints
        ]
        for pattern in patterns:
            model_files = sorted(list(self._checkpoints_dir.rglob(pattern)))
            if model_files:
                return model_files[0].name

        raise ValueError(f"Could not find checkpoint files in {self._checkpoints_dir}")

    @functools.lru_cache
    def get_tokenizer_file_path(self, tokenizer_key, file_key, default_filename_pattern):
        model_config = self.model_config
        file_property = None
        if tokenizer_key in model_config and file_key in model_config[tokenizer_key]:
            file_property = model_config[tokenizer_key][file_key]
        elif file_key in model_config:
            file_property = model_config[file_key]

        LOGGER.debug("model_config[%s][%s]=%s", tokenizer_key, file_key, file_property)

        if file_property and file_property.startswith("nemo:"):
            filename = file_property.split("nemo:")[1]
            filename_pattern = f"*{filename}"
        elif file_property and file_property.startswith("/artifacts/"):
            filename = pathlib.Path(file_property).name
            filename_pattern = f"*{filename}"
        elif file_property is None or file_property == "None":
            filename_pattern = None
        else:
            filename_pattern = default_filename_pattern
            LOGGER.warning(
                f"Tokenizer file from config: {tokenizer_key}.{file_key}={file_property} "
                f"looks like unsupported path. Pattern {filename_pattern} will be used."
            )

        file_path = None
        if filename_pattern is not None:
            files_paths = list(self._checkpoints_dir.glob(filename_pattern))
            if files_paths:
                assert len(files_paths) == 1
                file_path = files_paths[0]

        return file_path
