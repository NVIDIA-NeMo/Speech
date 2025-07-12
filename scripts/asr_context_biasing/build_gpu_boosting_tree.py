# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from omegaconf import MISSING
from torch.nn.utils.rnn import pad_sequence

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import (
    BoostingTreeModelConfig,
    GPUBoostingTreeModel,
)
from nemo.collections.common.tokenizers import AggregateTokenizer
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class BuildWordBoostingTreeConfig(BoostingTreeModelConfig):
    """
    Build GPU-accelerated phrase boosting tree (btree) to be used with greedy and beam search decoders of ASR models.
    """

    asr_pretrained_name: Optional[str] = None  # Name of a pretrained model
    asr_model_path: Optional[str] = None  # The path to '.nemo' ASR checkpoint
    save_to: str = MISSING  # The path to save the GPU-accelerated word boosting graph

    # evaluation of obtained boosting tree with test_sentences (optional)
    test_boosting_tree: bool = False  # Whether to test the GPU-accelerated word boosting tree after building it
    test_sentences: List[str] = field(
        default_factory=list
    )  # The phrases to test boosting tree ["hello world","nvlink","nvlinz","omniverse cloud now","acupuncture"]


@hydra_runner(config_path=None, config_name='BuildWordBoostingTreeConfig', schema=BuildWordBoostingTreeConfig)
def main(cfg: BuildWordBoostingTreeConfig):

    # 1. load asr model to obtain tokenizer
    if cfg.asr_model_path is None and cfg.asr_pretrained_name is None:
        raise ValueError("Either asr_model_path or asr_pretrained_name must be provided")
    elif cfg.asr_model_path is not None:
        asr_model = nemo_asr.models.ASRModel.restore_from(cfg.asr_model_path, map_location=torch.device('cpu'))
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(cfg.asr_pretrained_name)

    is_aggregate_tokenizer = isinstance(asr_model.tokenizer, AggregateTokenizer)

    # 2. Build GPU-accelerated word boosting tree from config
    gpu_boosting_model = GPUBoostingTreeModel.from_config(cfg, tokenizer=asr_model.tokenizer)

    # 3. save gpu boosting tree to nemo file
    gpu_boosting_model.save_to(cfg.save_to)

    # 4. test gpu boosting tree model
    logging.info("testing gpu boosting tree model...")
    if cfg.test_boosting_tree and cfg.test_sentences:
        gpu_boosting_model_loaded = GPUBoostingTreeModel.from_nemo(
            cfg.save_to, vocab_size=len(asr_model.tokenizer.vocab), use_triton=cfg.use_triton
        )
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        gpu_boosting_model_loaded = gpu_boosting_model_loaded.cuda()

        if not is_aggregate_tokenizer:
            sentences_ids = [asr_model.tokenizer.text_to_ids(sentence) for sentence in cfg.test_sentences]
            sentences_tokens = [asr_model.tokenizer.text_to_tokens(sentence) for sentence in cfg.test_sentences]
        else:
            sentences_ids = [
                asr_model.tokenizer.text_to_ids(sentence, cfg.source_lang) for sentence in cfg.test_sentences
            ]
            sentences_tokens = []  # aggregate tokenizer does not support text_to_tokens

        boosting_scores = gpu_boosting_model_loaded(
            labels=pad_sequence([torch.LongTensor(sentence) for sentence in sentences_ids], batch_first=True).to(
                device
            ),
            labels_lengths=torch.LongTensor([len(sentence) for sentence in sentences_ids]).to(device),
            bos=False,
            eos=False if not is_aggregate_tokenizer else True,
        )

        logging.info(f"[info]: boosting_scores: {boosting_scores}")
        logging.info(f"[info]: test_sentences: {cfg.test_sentences}")
        logging.info(f"[info]: test_sentences_tokens: {sentences_tokens}")
        logging.info(f"[info]: test_sentences_ids: {sentences_ids}")


if __name__ == '__main__':
    main()
