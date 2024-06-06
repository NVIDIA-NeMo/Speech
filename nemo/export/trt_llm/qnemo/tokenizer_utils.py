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

import os

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo.export.trt_llm.nemo_ckpt_loader.sentencepiece_tokenizer import SentencePieceTokenizer

# TODO: use get_nmt_tokenizer helper below to instantiate tokenizer once environment / dependencies get stable
# from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

TOKENIZER_CONFIG_FILE = "tokenizer_config.yaml"


def get_nmt_tokenizer(nemo_checkpoint_path: str):
    """Build tokenizer from Nemo tokenizer config."""

    print(f"Initializing tokenizer from {TOKENIZER_CONFIG_FILE}")
    tokenizer_cfg = OmegaConf.load(os.path.join(nemo_checkpoint_path, TOKENIZER_CONFIG_FILE))

    library = tokenizer_cfg.library
    legacy = tokenizer_cfg.get("sentencepiece_legacy", library == "sentencepiece")

    if library == "huggingface":
        print(f"Getting HuggingFace AutoTokenizer with pretrained_model_name: {tokenizer_cfg.type}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg["type"], use_fast=tokenizer_cfg.get("use_fast", False))
    elif library == "sentencepiece":
        print(f"Getting SentencePieceTokenizer with model: {tokenizer_cfg.model}")
        tokenizer = SentencePieceTokenizer(
            model_path=os.path.join(nemo_checkpoint_path, tokenizer_cfg.model), legacy=legacy
        )
    else:
        raise NotImplementedError("Currently we only support 'huggingface' and 'sentencepiece' tokenizer libraries.")

    return tokenizer
