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

import pytest
import torch

from nemo.collections.asr.inference.model_wrappers.ctc_inference_wrapper import CTCInferenceWrapper
from nemo.collections.asr.inference.utils.bpe_decoder import BPEDecoder
from nemo.collections.asr.inference.utils.text_segment import TextSegment, Word
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig


@pytest.fixture(scope="module")
def bpe_decoder():
    asr_model = CTCInferenceWrapper(
        model_name="stt_en_conformer_ctc_small",
        decoding_cfg=CTCDecodingConfig(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return BPEDecoder(
        vocabulary=asr_model.get_vocabulary(),
        tokenizer=asr_model.tokenizer,
        confidence_aggregator=min,
        asr_supported_puncts=asr_model.supported_punctuation(),
        word_boundary_tolerance=0.0,  # Set to 0.0 for easy testing
        token_duration_in_secs=asr_model.get_model_stride(in_secs=True),
    )


class TestBPEDecoder:

    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text",
        [
            "the quick brown fox jumps over the lazy dog",
            "lorem ipsum dolor sit amet",
            "this a test sentence",
        ],
    )
    def test_group_tokens_into_words(self, bpe_decoder, text):
        ground_truth_words = text.split()
        tokens = bpe_decoder.tokenizer.text_to_ids(text)
        n_tokens = len(tokens)
        timestamps = [float(i) for i in range(n_tokens)]
        confidences = [0.1] * n_tokens

        words, need_merge = bpe_decoder.group_tokens_into_words(tokens, timestamps, confidences)
        assert len(words) == len(ground_truth_words)
        prev_word_end = -1
        for word, ground_truth_word in zip(words, ground_truth_words):
            assert isinstance(word, Word)
            assert word.text == ground_truth_word
            assert word.conf == 0.1
            assert word.end > word.start and word.start >= prev_word_end
            prev_word_end = word.end
        assert need_merge == False

    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text",
        [
            "the quick brown fox jumps over the lazy dog",
            "lorem ipsum dolor sit amet",
            "this a test sentence",
        ],
    )
    def test_group_tokens_into_segment(self, bpe_decoder, text):
        tokens = bpe_decoder.tokenizer.text_to_ids(text)
        n_tokens = len(tokens)
        timestamps = [float(i) for i in range(n_tokens)]
        confidences = [0.1] * n_tokens

        segment, need_merge = bpe_decoder.group_tokens_into_segment(tokens, timestamps, confidences)
        assert isinstance(segment, TextSegment)
        assert need_merge == False
        assert segment.text == text
        assert segment.start == 0.0
        assert segment.end == (n_tokens - 1) * bpe_decoder.token_duration_in_secs
        assert segment.conf == 0.1
