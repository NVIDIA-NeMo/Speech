# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import re
from dataclasses import dataclass

import torch
from whisper_normalizer.english import EnglishTextNormalizer

_whisper_normalizer = EnglishTextNormalizer()


def clean_pred_text(text: str) -> str:
    """Clean prediction text for fair WER comparison.

    First strips model-specific tokens (turn markers, timestamps, pad tokens)
    that the Whisper normalizer doesn't know about, then applies
    ``EnglishTextNormalizer`` — the same normalizer used by the offline eval
    metrics in ``speechlm2.parts.metrics.wer``.
    """
    if not text:
        return ""
    # Strip model-specific tokens
    text = text.lstrip('^')
    text = re.sub(r'</?s>', '', text)
    text = re.sub(r'<\$[\d.]+\$>', '', text)
    text = re.sub(r'<\|[\d.]+\|>', '', text)
    text = re.sub(r'<SPECIAL_12>', '', text)
    # Normalize with Whisper's EnglishTextNormalizer (same as offline eval)
    return _whisper_normalizer(text)


@dataclass
class PipelineOutput:
    """Output of the S2S pipeline's :meth:`run` method.

    Every list field is indexed by stream id — entry *i* holds the result
    for the *i*-th input audio file.
    """

    texts: list[str] | None = None
    asr_texts: list[str] | None = None
    texts_with_timestamps: list[str] | None = None
    asr_texts_with_timestamps: list[str] | None = None
    raw_texts: list[str] | None = None
    raw_asr_texts: list[str] | None = None
    token_texts: list[torch.Tensor | None] | None = None
    token_asr_texts: list[torch.Tensor | None] | None = None
    token_function_texts: list[torch.Tensor | None] | None = None
    token_lengths: list[int | None] | None = None
    audio_filepaths: list[str | None] | None = None
    debug_data: list[list] | None = None
