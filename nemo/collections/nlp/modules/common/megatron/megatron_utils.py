# Copyright 2020 The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os
import shutil
from typing import Dict, List, Optional

import torch
import wget
from torch.hub import _get_torch_home

from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

__all__ = [
    "get_megatron_lm_model",
    "get_megatron_lm_models_list",
    "get_megatron_checkpoint",
    "is_lower_cased_megatron",
    "get_megatron_tokenizer",
    "get_megatron_pretrained_bert_models",
]


torch_home = _get_torch_home()

if not isinstance(torch_home, str):
    logging.info("Torch home not found, caching megatron in cwd")
    torch_home = os.getcwd()

MEGATRON_CACHE = os.path.join(torch_home, "megatron")


CONFIGS = {"345m": {"hidden_size": 1024, "num_attention_heads": 16, "num_layers": 24, "max_position_embeddings": 512}}

MEGATRON_CONFIG_MAP = {
    "megatron-gpt-345m": {
        "config": CONFIGS["345m"],
        "checkpoint": "models/nvidia/megatron_lm_345m/versions/v0.0/files/release/mp_rank_00/model_optim_rng.pt",
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
        "merges_file": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
        "do_lower_case": False,
        "tokenizer_name": "gpt2",
    },
    "megatron-bert-345m-uncased": {
        "config": CONFIGS["345m"],
        "checkpoint": "https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.0/files/release/mp_rank_00/model_optim_rng.pt",  # pylint: disable=line-too-long
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        "do_lower_case": True,
        "tokenizer_name": "bert-large-uncased",
    },
    "megatron-bert-345m-cased": {
        "config": CONFIGS["345m"],
        "checkpoint": "https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/files/release/mp_rank_00/model_optim_rng.pt",  # pylint: disable=line-too-long
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        "do_lower_case": False,
        "tokenizer_name": "bert-large-cased",
    },
    "megatron-bert-uncased": {
        "config": None,
        "checkpoint": None,
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        "do_lower_case": True,
        "tokenizer_name": "bert-large-uncased",
    },
    "megatron-bert-cased": {
        "config": None,
        "checkpoint": None,
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        "do_lower_case": False,
        "tokenizer_name": "bert-large-cased",
    },
    "biomegatron-bert-345m-uncased": {
        "config": CONFIGS["345m"],
        "checkpoint": "https://api.ngc.nvidia.com/v2/models/nvidia/biomegatron345muncased/versions/0/files/MegatronBERT.pt",  # pylint: disable=line-too-long
        "vocab": "https://api.ngc.nvidia.com/v2/models/nvidia/biomegatron345muncased/versions/0/files/vocab.txt",
        "do_lower_case": True,
        "tokenizer_name": "bert-large-uncased",
    },
    "biomegatron-bert-345m-cased": {
        "config": CONFIGS["345m"],
        "checkpoint": "https://api.ngc.nvidia.com/v2/models/nvidia/biomegatron345mcased/versions/0/files/MegatronBERT.pt",  # pylint: disable=line-too-long
        "vocab": "https://api.ngc.nvidia.com/v2/models/nvidia/biomegatron345mcased/versions/0/files/vocab.txt",
        "do_lower_case": False,
        "tokenizer_name": "bert-large-cased",
    },
}


def compute_model_parallel_rank(local_rank: int, model_parallel_size: int) -> int:
    """Calculates the model_parallel_rank from the local rank and the model parallel size

    Args:
        local_rank (int): The local rank of the process.
        model_parallel_size (int): The number of ranks in the model parallel group.

    Returns:
        int: The model parallel rank corresponding to the given local rank.
    """
    return local_rank % model_parallel_size


def get_megatron_pretrained_bert_models() -> List[str]:
    """Retrieves the names of all available pretrained Megatron-BERT models.

    This function uses the NeMo MegatronBertModel class to list all available
    pretrained model configurations, extracting each model's name.

    Returns:
        List[str]: A list of pretrained Megatron-BERT model names.
    """

    all_pretrained_megatron_bert_models = [model.pretrained_model_name for model in list_available_models()]
    return all_pretrained_megatron_bert_models


def get_megatron_lm_models_list() -> List[str]:
    """
    Returns the list of supported Megatron-LM models
    """
    return list(MEGATRON_CONFIG_MAP.keys())


def get_megatron_config(pretrained_model_name: str) -> Dict[str, int]:
    """
    Returns Megatron-LM model config file

    Args:
        pretrained_model_name (str): pretrained model name

    Returns:
        config (dict): contains model configuration: number of hidden layers, number of attention heads, etc
    """
    _check_megatron_name(pretrained_model_name)
    return MEGATRON_CONFIG_MAP[pretrained_model_name]["config"]


def _check_megatron_name(pretrained_model_name: str) -> None:
    megatron_model_list = get_megatron_lm_models_list()
    if pretrained_model_name not in megatron_model_list:
        raise ValueError(f'For Megatron-LM models, choose from the following list: {megatron_model_list}')


def get_megatron_vocab_file(pretrained_model_name: str) -> str:
    """
    Gets vocabulary file from cache or downloads it

    Args:
        pretrained_model_name: pretrained model name

    Returns:
        path: path to the vocab file
    """
    _check_megatron_name(pretrained_model_name)
    url = MEGATRON_CONFIG_MAP[pretrained_model_name]["vocab"]

    path = os.path.join(MEGATRON_CACHE, pretrained_model_name + "_vocab")
    path = _download(path, url)
    return path


def get_megatron_merges_file(pretrained_model_name: str) -> str:
    """
    Gets merge file from cache or downloads it

    Args:
        pretrained_model_name: pretrained model name

    Returns:
        path: path to the vocab file
    """
    if 'gpt' not in pretrained_model_name.lower():
        return None
    _check_megatron_name(pretrained_model_name)
    url = MEGATRON_CONFIG_MAP[pretrained_model_name]["merges_file"]

    path = os.path.join(MEGATRON_CACHE, pretrained_model_name + "_merges")
    path = _download(path, url)
    return path


def get_megatron_checkpoint(pretrained_model_name: str) -> str:
    """
    Gets checkpoint file from cache or downloads it
    Args:
        pretrained_model_name: pretrained model name
    Returns:
        path: path to model checkpoint
    """
    _check_megatron_name(pretrained_model_name)
    url = MEGATRON_CONFIG_MAP[pretrained_model_name]["checkpoint"]
    path = os.path.join(MEGATRON_CACHE, pretrained_model_name)
    return _download(path, url)


def _download(path: str, url: str):
    """
    Gets a file from cache or downloads it

    Args:
        path: path to the file in cache
        url: url to the file
    Returns:
        path: path to the file in cache
    """
    if url is None:
        return None

    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and not os.path.exists(path):
        os.makedirs(MEGATRON_CACHE, exist_ok=True)
        logging.info(f"Downloading from {url} to {path}")
        downloaded_path = wget.download(url)
        if not os.path.exists(downloaded_path):
            raise FileNotFoundError(f"Downloaded file not found: {downloaded_path}")
        shutil.move(downloaded_path, path)
    # wait until the master process downloads the file and writes it to the cache dir
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return path


def is_lower_cased_megatron(pretrained_model_name):
    """
    Returns if the megatron is cased or uncased

    Args:
        pretrained_model_name (str): pretrained model name
    Returns:
        do_lower_cased (bool): whether the model uses lower cased data
    """
    _check_megatron_name(pretrained_model_name)
    return MEGATRON_CONFIG_MAP[pretrained_model_name]["do_lower_case"]


def get_megatron_tokenizer(pretrained_model_name: str):
    """
    Takes a pretrained_model_name for megatron such as "megatron-bert-cased" and returns the according
    tokenizer name for tokenizer instantiating.

    Args:
        pretrained_model_name: pretrained_model_name for megatron such as "megatron-bert-cased"
    Returns:
        tokenizer name for tokenizer instantiating
    """
    _check_megatron_name(pretrained_model_name)
    return MEGATRON_CONFIG_MAP[pretrained_model_name]["tokenizer_name"]


def list_available_models() -> Optional[PretrainedModelInfo]:
    """
    This function returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
    Returns:
        List of available pre-trained models.
    """
    result = []
    for vocab in ['cased', 'uncased']:
        result.append(
            PretrainedModelInfo(
                pretrained_model_name=f"megatron_bert_345m_{vocab}",
                # pylint: disable=C0301
                location=f"https://api.ngc.nvidia.com/v2/models/nvidia/nemo/megatron_bert_345m_{vocab}/versions/1/files/megatron_bert_345m_{vocab}.nemo",
                description=f"345M parameter BERT Megatron model with {vocab} vocab.",
            )
        )
    for vocab_size in ['50k', '30k']:
        for vocab in ['cased', 'uncased']:
            result.append(
                PretrainedModelInfo(
                    pretrained_model_name=f"biomegatron345m_biovocab_{vocab_size}_{vocab}",
                    # pylint: disable=C0301
                    location=f"https://api.ngc.nvidia.com/v2/models/nvidia/nemo/biomegatron345m_biovocab_{vocab_size}_{vocab}/versions/1/files/BioMegatron345m-biovocab-{vocab_size}-{vocab}.nemo",
                    # pylint: disable=C0301
                    description="Megatron 345m parameters model with biomedical vocabulary ({vocab_size} size) {vocab}, pre-trained on PubMed biomedical text corpus.",
                )
            )
    for vocab in ['cased', 'uncased']:
        result.append(
            PretrainedModelInfo(
                pretrained_model_name=f"biomegatron-bert-345m-{vocab}",
                # pylint: disable=C0301
                location=f"https://api.ngc.nvidia.com/v2/models/nvidia/nemo/biomegatron345m{vocab}/versions/1/files/BioMegatron345m{vocab.capitalize()}.nemo",
                # pylint: disable=C0301
                description=f"Megatron pretrained on {vocab} biomedical dataset PubMed with 345 million parameters.",
            )
        )
    return result
