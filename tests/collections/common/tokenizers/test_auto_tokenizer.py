# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import pytest
from omegaconf import DictConfig

from nemo.collections.asr.parts.mixins import mixins


@pytest.fixture(scope="module")
def sample_auto_tokenizer(test_data_dir):
    vocab_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128")
    cfg = DictConfig({'dir': vocab_dir, 'type': 'wpe'})

    class _TmpASRBPE(mixins.ASRBPEMixin):
        def register_artifact(self, _, vocab_path):
            return vocab_path

    asrbpe = _TmpASRBPE()
    asrbpe._setup_tokenizer(cfg)
    return asrbpe.tokenizer


class TestAutoTokenizer:
    @pytest.mark.unit
    def test_tokens_to_text_with_list(self, sample_auto_tokenizer):
        tokens = ["k"]
        text = sample_auto_tokenizer.tokens_to_text(tokens)
        assert text == "k"

        tokens_multi = ["a", "b"]
        text_multi = sample_auto_tokenizer.tokens_to_text(tokens_multi)
        assert "a" in text_multi and "b" in text_multi

    @pytest.mark.unit
    def test_tokens_to_text_with_single_str(self, sample_auto_tokenizer):
        # Passing a single token as string must succeed and not raise TypeError
        text = sample_auto_tokenizer.tokens_to_text("k")
        assert text == "k"

    @pytest.mark.unit
    def test_tokens_to_text_with_tuple(self, sample_auto_tokenizer):
        # Passing an iterable tuple of tokens must succeed
        tokens = ("k",)
        text = sample_auto_tokenizer.tokens_to_text(tokens)
        assert text == "k"

    @pytest.mark.unit
    def test_tokens_to_ids_with_single_str_and_iterables(self, sample_auto_tokenizer):
        # tokens_to_ids should always return a list of ints
        ids_from_list = sample_auto_tokenizer.tokens_to_ids(["k"])
        assert isinstance(ids_from_list, list)
        assert len(ids_from_list) == 1
        assert isinstance(ids_from_list[0], int)

        ids_from_str = sample_auto_tokenizer.tokens_to_ids("k")
        assert isinstance(ids_from_str, list)
        assert ids_from_str == ids_from_list

        ids_from_tuple = sample_auto_tokenizer.tokens_to_ids(("k",))
        assert isinstance(ids_from_tuple, list)
        assert ids_from_tuple == ids_from_list
