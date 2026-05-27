# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2020 AWSLABS, AMAZON.
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

from typing import List

import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForMaskedLM, AutoTokenizer

__all__ = ['MLMScorer']


class MLMScorer:
    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Creates MLM scorer from https://arxiv.org/abs/1910.14659.
        Args:
            model_name: HuggingFace pretrained model name
            device: either 'cpu' or 'cuda'
        """
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.device = device
        self.MASK_LABEL = self.tokenizer.mask_token

    def score_sentences(self, sentences: List[str]):
        """
        returns list of MLM scores for each sentence in list.
        """
        return [self.score_sentence(sentence) for sentence in sentences]

    def score_sentence(self, sentence: str):
        """
        returns MLM score for sentence.
        """
        assert type(sentence) == str

        tokens = self.tokenizer.tokenize(sentence)
        mask_idx = []
        token_type = []
        attn_mask = []
        ids = []
        for m_idx, _ in enumerate(tokens):
            masked = self.__mask_text__(m_idx, tokens)
            mask_idx.append(m_idx)
            ids.append(self.tokenizer.encode(masked))
            id_len = len(ids[-1])
            token_type.append([0] * id_len)
            attn_mask.append([1] * id_len)

        data = {
            'input_ids': torch.tensor(ids, device=self.device),
            'attention_mask': torch.tensor(attn_mask, device=self.device),
            'token_type_ids': torch.tensor(token_type, device=self.device),
        }

        with torch.no_grad():
            outputs = self.model(**data)
            logits = outputs.logits

        scores = []
        scores_log_prob = 0.0

        for i, m_idx in enumerate(mask_idx):
            preds = logits[i].squeeze(0)
            probs = softmax(preds, dim=1)
            token_id = self.tokenizer.convert_tokens_to_ids([tokens[m_idx]])[0]
            log_prob = np.log(probs[m_idx + 1, token_id].cpu().numpy()).item()
            scores.append(log_prob)
            scores_log_prob += log_prob

        return scores_log_prob

    def __mask_text__(self, idx: int, tokens: List[str]):
        """
        replaces string at index idx in list `tokens` with a masked token and returns the modified list.
        """
        masked = tokens.copy()
        masked[idx] = self.MASK_LABEL
        return masked
