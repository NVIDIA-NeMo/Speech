# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import pytest
from lhotse import CutSet, SupervisionSegment
from lhotse.dataset.iterable_dataset import IdentityDataset
from lhotse.testing.dummies import dummy_cut
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.sampling import TokenPerSecondFilter, TokenPerTokenFilter
from nemo.collections.common.data.lhotse.text_adapters import SourceTargetTextExample, TextExample
from nemo.collections.common.tokenizers.char_tokenizer import CharTokenizer


@pytest.mark.unit
@pytest.mark.parametrize("filter_type", [TokenPerSecondFilter, TokenPerTokenFilter])
@pytest.mark.parametrize(
    "lower,upper,expected",
    [
        (None, None, [True, True, True]),
        (None, 2.0, [True, True, False]),
        (2.0, None, [False, True, True]),
        (2.0, 2.0, [False, True, False]),
        (None, [1.0, 2.0], [True, True, True]),
        (2.0, [1.0, 2.0], [False, True, True]),
        (None, OmegaConf.create([1.0, 2.0]), [True, True, True]),
    ],
)
def test_optional_ratio_bounds(filter_type, lower, upper, expected):
    ratio_filter = filter_type(lower, upper)
    examples = []
    for index, ratio in enumerate((1, 2, 3)):
        if filter_type is TokenPerSecondFilter:
            example = dummy_cut(index, duration=1.0)
            example.supervisions = [SupervisionSegment(str(index), example.recording_id, 0.0, 1.0)]
            example.supervisions[0].tokens = list(range(ratio))
        else:
            example = SourceTargetTextExample(source=TextExample("a"), target=TextExample("b"))
            example.context_ids = np.array([0])
            example.answer_ids = np.arange(ratio)
        examples.append(example)

    assert [ratio_filter(example) for example in examples] == expected


@pytest.mark.unit
@pytest.mark.parametrize("filter_type", [TokenPerSecondFilter, TokenPerTokenFilter])
def test_optional_ratio_bounds_disabled(filter_type):
    ratio_filter = filter_type(None, None)
    assert not ratio_filter.enabled
    # Disabled filters must not require tokenization or prompt formatting.
    assert ratio_filter(dummy_cut(0))
    assert ratio_filter(SourceTargetTextExample(source=TextExample("a"), target=TextExample("b")))


@pytest.mark.unit
@pytest.mark.parametrize("filter_type", [TokenPerSecondFilter, TokenPerTokenFilter])
@pytest.mark.parametrize("lower,upper", [(3.0, 2.0), (float("nan"), 2.0), (1.0, float("nan"))])
def test_invalid_ratio_bounds(filter_type, lower, upper):
    with pytest.raises(ValueError, match="minimum.*maximum"):
        filter_type(lower, upper)


@pytest.mark.unit
@pytest.mark.parametrize("min_tps,max_tps,expected_ids", [(-1, None, [0, 1, 2]), (2, None, [1, 2]), (-1, 2.0, [0, 1])])
def test_dataloader_optional_ratio_bounds(tmp_path, min_tps, max_tps, expected_ids):
    cuts = []
    for index, text in enumerate(("a", "aa", "aaa")):
        cut = dummy_cut(index, duration=1.0)
        cut.supervisions = [SupervisionSegment(str(index), cut.recording_id, 0.0, 1.0, text=text)]
        cuts.append(cut)
    cuts_path = tmp_path / "cuts.jsonl"
    CutSet.from_cuts(cuts).to_file(cuts_path)
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("'a'\n")

    dataloader = get_lhotse_dataloader_from_config(
        config=OmegaConf.create(
            {
                "cuts_path": str(cuts_path),
                "batch_size": 2,
                "use_bucketing": False,
                "drop_last": False,
                "min_tps": min_tps,
                "max_tps": max_tps,
                "max_tpt": None,
            }
        ),
        global_rank=0,
        world_size=1,
        dataset=IdentityDataset(),
        tokenizer=CharTokenizer(str(vocab_path)),
    )

    batches = list(dataloader)
    assert len(batches[0]) == 2
    assert [cut.id for batch in batches for cut in batch] == [cuts[index].id for index in expected_ids]
