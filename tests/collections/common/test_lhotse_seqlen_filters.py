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
from copy import deepcopy

import numpy as np
import pytest
from lhotse import SupervisionSegment
from lhotse.testing.dummies import dummy_cut

from nemo.collections.common.data.lhotse.sampling import (
    DurationFilter,
    TokenCountFilter,
    TokenPerSecondFilter,
    TokenPerTokenFilter,
)
from nemo.collections.common.data.lhotse.text_adapters import NeMoSFTExample, SourceTargetTextExample, TextExample


@pytest.fixture
def cut():
    c = dummy_cut(0, duration=1.0, supervisions=[SupervisionSegment("", "", 0, 1.0, text="dummy")])
    c.supervisions[0].tokens = [1, 37, 12, 2]
    return c


def test_cut_duration_filter(cut):
    f = DurationFilter(0, 10)
    assert f(cut) == True

    f = DurationFilter(0, 0.5)
    assert f(cut) == False

    f = DurationFilter(1.5, 2.0)
    assert f(cut) == False


def test_cut_token_per_second_filter(cut):
    f = TokenPerSecondFilter(tps_min=0.0, tps_max=5.0)
    assert f(cut) == True

    f = TokenPerSecondFilter(tps_min=0.0, tps_max=1.0)
    assert f(cut) == False

    f = TokenPerSecondFilter(tps_min=10.0, tps_max=12.0)
    assert f(cut) == False


def test_cut_passes_by_token_count_and_tpt_filter(cut):
    assert TokenCountFilter(1, 10, measure_total_length=True)(cut) == True
    assert TokenPerTokenFilter(1, 10)(cut) == True


def test_cut_passes_by_token_count_and_tpt_filter(cut):
    assert TokenCountFilter(1, 10, measure_total_length=True)(cut) == True
    assert TokenPerTokenFilter(1, 10)(cut) == True


@pytest.fixture
def src_tgt_example():
    return SourceTargetTextExample(
        source=TextExample("", tokens=np.array([1, 37, 12, 2])),
        target=TextExample("", tokens=np.array([1, 1823, 1245, 2446, 1038, 2])),
    )


def test_src_tgt_token_filter_requires_prompt_formatting(src_tgt_example):
    with pytest.raises(RuntimeError):
        TokenCountFilter(0, 1, True)(src_tgt_example)


def test_src_tgt_passes_by_duration_filter(src_tgt_example):
    assert DurationFilter(1, 10)(src_tgt_example) == True
    assert TokenPerSecondFilter(1, 10)(src_tgt_example) == True


def test_src_tgt_token_filter(src_tgt_example):
    example = deepcopy(src_tgt_example)
    example.input_ids = np.concatenate((example.source.tokens, example.target.tokens))
    example.context_ids = example.source.tokens
    example.answer_ids = example.target.tokens

    """
    Input length measurement / encoder-decoder models / 2D bucketing
    """
    f = TokenCountFilter(1, 5, measure_total_length=False)
    assert f(example) == True

    f = TokenCountFilter(1, 3, measure_total_length=False)
    assert f(example) == False

    f = TokenCountFilter(10, 30, measure_total_length=False)
    assert f(example) == False

    """
    Total length measurement / decoder-only models / 1D bucketing
    """
    f = TokenCountFilter(1, 5, measure_total_length=True)
    assert f(example) == False

    f = TokenCountFilter(1, 20, measure_total_length=True)
    assert f(example) == True

    f = TokenCountFilter(1, 3, measure_total_length=True)
    assert f(example) == False

    f = TokenCountFilter(20, 30, measure_total_length=True)
    assert f(example) == False


@pytest.fixture
def nemo_sft_example():
    example = NeMoSFTExample(
        data={
            "system": "",
            "mask": "User",
            "dataset": "",
            "conversations": [
                {
                    "from": "User",
                    "value": "Hi, how are you?",
                },
                {
                    "from": "Assistant",
                    "value": "Good day, I'm a useful assistant.",
                },
            ],
        },
    )
    return example


def test_nemo_sft_token_filter_requires_prompt_formatting(nemo_sft_example):
    with pytest.raises(RuntimeError):
        TokenCountFilter(0, 1, True)(nemo_sft_example)


def test_nemo_sft_passes_by_duration_filter(nemo_sft_example):
    assert DurationFilter(1, 10)(nemo_sft_example) == True
    assert TokenPerSecondFilter(1, 10)(nemo_sft_example) == True


def test_nemo_sft_token_filter(nemo_sft_example):
    example = deepcopy(nemo_sft_example)
    example.input_ids = np.array([1, 123, 3425, 123, 2345, 324, 54, 2])
    example.context_ids = np.array([1, 123, 3425])
    example.answer_ids = np.array([123, 2345, 324, 54, 2])

    """
    Input length measurement / encoder-decoder models / 2D bucketing
    """
    f = TokenCountFilter(1, 5, measure_total_length=False)
    assert f(example) == True

    f = TokenCountFilter(1, 2, measure_total_length=False)
    assert f(example) == False

    f = TokenCountFilter(10, 30, measure_total_length=False)
    assert f(example) == False

    """
    Total length measurement / decoder-only models / 1D bucketing
    """
    f = TokenCountFilter(1, 5, measure_total_length=True)
    assert f(example) == False

    f = TokenCountFilter(1, 20, measure_total_length=True)
    assert f(example) == True

    f = TokenCountFilter(1, 3, measure_total_length=True)
    assert f(example) == False

    f = TokenCountFilter(10, 30, measure_total_length=True)
    assert f(example) == False
