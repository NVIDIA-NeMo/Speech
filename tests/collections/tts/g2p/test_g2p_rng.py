# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import random

import pytest
import torch

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer, IPATokenizer
from nemo.collections.tts.g2p.models import base as g2p_base
from nemo.collections.tts.g2p.models.en_us_arpabet import EnglishG2p
from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p
from nemo.collections.tts.parts.utils.tts_dataset_utils import set_g2p_rng

pytestmark = pytest.mark.unit

PHONEME_DICT_PATH_EN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phoneme_dict", "test_dict_en.txt")
TEXT = " ".join(["hello world nvidia jones airport"] * 8)


def _ipa_g2p():
    return IpaG2p(
        PHONEME_DICT_PATH_EN, locale="en-US", use_chars=True, phoneme_probability=0.5, apply_to_oov_word=lambda x: x
    )


def _arpabet_g2p():
    phoneme_dict = {
        "hello": [["HH", "AH0", "L", "OW1"]],
        "world": [["W", "ER1", "L", "D"]],
        "nvidia": [["EH0", "N", "V", "IH1", "D", "IY0", "AH0"]],
        "jones": [["JH", "OW1", "N", "Z"]],
        "airport": [["EH1", "R", "P", "AO2", "R", "T"]],
    }
    return EnglishG2p(phoneme_dict=phoneme_dict, phoneme_probability=0.5, apply_to_oov_word=lambda x: x)


@pytest.fixture(params=[_ipa_g2p, _arpabet_g2p], ids=["IpaG2p", "EnglishG2p"])
def make_g2p(request):
    return request.param


def _phonemize(g2p, times: int = 3):
    return [tuple(map(str, g2p(TEXT))) for _ in range(times)]


def test_default_rng_is_seeded_per_process(make_g2p):
    torch.manual_seed(0)
    random.seed(0)
    first = _phonemize(make_g2p())
    torch.manual_seed(0)
    random.seed(1)  # the global RNG plays no part
    second = _phonemize(make_g2p())

    assert first == second
    # phoneme_probability=0.5 keeps some words as graphemes, so the output varies between calls.
    assert len(set(first)) > 1


def test_injected_rng_drives_phoneme_probability(make_g2p):
    g2p = make_g2p()
    g2p.rng = random.Random(1)
    expected = _phonemize(g2p)

    torch.manual_seed(123)  # the default per-process seed plays no part either
    other = make_g2p()
    other.rng = random.Random(1)

    assert _phonemize(other) == expected


def test_rng_from_another_process_is_replaced(make_g2p, monkeypatch):
    g2p = make_g2p()
    injected = random.Random(1)
    g2p.rng = injected
    assert g2p.rng is injected

    monkeypatch.setattr(g2p_base.os, "getpid", lambda: -1)

    assert g2p.rng is not injected


def test_set_g2p_rng_reaches_aggregated_tokenizers():
    g2p = _ipa_g2p()
    tokenizer = AggregatedTTSTokenizer([IPATokenizer(g2p=g2p, locale="en-US")], ["english_phoneme"])
    rng = random.Random(0)

    set_g2p_rng(tokenizer, rng)
    set_g2p_rng(None, rng)  # tokenizers that are not configured are skipped

    assert g2p.rng is rng
