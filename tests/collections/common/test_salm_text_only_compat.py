# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from lhotse import CutSet

from nemo.collections.common.data.lhotse.text_adapters import NeMoSFTExample
from nemo.collections.speechlm2.data.salm_dataset import SALMDataset


class _Tokenizer:
    pad = 0
    unk_id = 1


@pytest.mark.unit
@pytest.mark.parametrize("pack_audio", [False, True])
def test_strict_salm_batching_accepts_prompt_formatted_text_only_examples(pack_audio):
    example = NeMoSFTExample(data={"conversations": []})
    example.input_ids = torch.tensor([7, 8], dtype=torch.long)
    example.mask = torch.tensor([False, True])

    batch = SALMDataset(_Tokenizer(), pack_audio=pack_audio)[CutSet([example])]

    assert batch["audio_lens"].numel() == 0
    assert batch["input_ids"].tolist() == [[7, 8]]
    assert list(batch["conversations"]) == [example]
    if pack_audio:
        assert batch["packed_audio_samples"].numel() == 0
        assert batch["audio_cu_seqlens"].tolist() == [0]
    else:
        assert batch["audios"].numel() == 0


@pytest.mark.unit
@pytest.mark.parametrize("pack_sequences", [False, True])
def test_text_only_examples_with_multispeaker_cfg_have_no_speaker_targets(
    pack_sequences,
):
    example = NeMoSFTExample(data={"conversations": []})
    example.input_ids = torch.tensor([7, 8], dtype=torch.long)
    example.mask = torch.tensor([False, True])

    dataset = SALMDataset(
        _Tokenizer(),
        pack_sequences=pack_sequences,
        multispeaker_cfg={"num_speakers": 2},
    )
    batch = dataset[CutSet([example])]

    assert batch["audio_lens"].numel() == 0
    assert list(batch["conversations"]) == [example]
    assert "spk_targets" not in batch
    if pack_sequences:
        assert batch["input_ids"].tolist() == [7, 8]
        assert batch["text_cu_seqlens"].tolist() == [0, 2]
        assert batch["audio_cu_seqlens"].tolist() == [0]
    else:
        assert batch["input_ids"].tolist() == [[7, 8]]
        assert batch["audios"].numel() == 0
