# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo.export.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.export.tiktoken_tokenizer import TiktokenTokenizer

# TODO: use get_nmt_tokenizer helper below to instantiate tokenizer once environment / dependencies get stable
# from nemo.collections.common.tokenizers.tokenizer_utils import get_nmt_tokenizer

TOKENIZER_CONFIG_FILE = "tokenizer_config.yaml"
TOKENIZER_DIR = "tokenizer"
LOGGER = logging.getLogger("NeMo")


def get_nmt_tokenizer(nemo_checkpoint_path: str):
    """Build tokenizer from Nemo tokenizer config."""

    LOGGER.info(f"Initializing tokenizer from {TOKENIZER_CONFIG_FILE}")
    tokenizer_cfg = OmegaConf.load(os.path.join(nemo_checkpoint_path, TOKENIZER_CONFIG_FILE))

    library = tokenizer_cfg.library
    legacy = tokenizer_cfg.get("sentencepiece_legacy", library == "sentencepiece")

    if library == "huggingface":
        LOGGER.info(f"Getting HuggingFace AutoTokenizer with pretrained_model_name: {tokenizer_cfg.type}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg["type"], use_fast=tokenizer_cfg.get("use_fast", False))
    elif library == "sentencepiece":
        LOGGER.info(f"Getting SentencePieceTokenizer with model: {tokenizer_cfg.model}")
        tokenizer = SentencePieceTokenizer(
            model_path=os.path.join(nemo_checkpoint_path, tokenizer_cfg.model), legacy=legacy
        )
    elif library == "tiktoken":
        print(f"Getting TiktokenTokenizer with file: {tokenizer_cfg.vocab_file}")
        tokenizer = TiktokenTokenizer(vocab_file=os.path.join(nemo_checkpoint_path, tokenizer_cfg.vocab_file))
    else:
        raise NotImplementedError("Currently we only support 'huggingface' and 'sentencepiece' tokenizer libraries.")

    return tokenizer
