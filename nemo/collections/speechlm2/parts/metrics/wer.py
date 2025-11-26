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
from collections import defaultdict

import torch
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging


class WER:
    """
    Computes WER on text predictions.
    By default, uses Whisper's EnglishTextNormalizer on hypotheses and references.
    """

    def __init__(self, normalize: bool = True, normalizer=None, verbose: bool = True):
        self.verbose = verbose
        if normalize:
            if normalizer is None:
                self.normalizer = EnglishTextNormalizer()
            else:
                self.normalizer = normalizer
        else:
            self.normalizer = _identity

        self._refs = defaultdict(list)
        self._hyps = defaultdict(list)

    def reset(self):
        self._refs.clear()
        self._hyps.clear()
        return self

    def update(self, name: str, refs: list[str], hyps: list[str]) -> None:
        for ref, hyp in zip(refs, hyps):
            self._refs[name].append(self.normalizer(ref))
            self._hyps[name].append(self.normalizer(hyp))
        if self.verbose and refs and hyps:
            logging.info(f"[REF]\t{refs[0]}\n[HYP]\t{hyps[0]}")

    def compute(self) -> dict[str, torch.Tensor]:
        corpus_metric = {}
        for name in self._refs.keys():
            metric = torch.tensor(word_error_rate(self._hyps[name], self._refs[name]))
            corpus_metric[f"wer_{name}"] = metric
        corpus_metric["wer"] = torch.stack(list(corpus_metric.values())).mean()
        self.reset()
        return corpus_metric


def _identity(x):
    return x
