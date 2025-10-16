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
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import DummyManifest

from nemo.collections.asr.data.audio_to_text_lhotse_speaker import LhotseSpeechToTextSpkBpeDataset
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory) -> SentencePieceTokenizer:
    tmpdir = tmp_path_factory.mktemp("klingon_tokens")
    text_path = tmpdir / "text.txt"
    text_path.write_text("\n".join(map(chr, range(ord('a'), ord('z')))))
    model_path, vocab_path = create_spt_model(
        text_path, vocab_size=32, sample_size=-1, do_lower_case=False, output_dir=str(tmpdir)
    )
    return SentencePieceTokenizer(model_path)


def test_lhotse_asr_speaker_dataset(tokenizer):
    # 3 cuts of duration 1s with audio and a single supervision with text 'irrelevant'
    cuts = DummyManifest(CutSet, begin_id=0, end_id=2, with_data=True)

    # cuts[0] is the default case: audio + single untokenized superivision

    # cuts[1]: audio + two supervisions
    cuts[1].supervisions = [
        SupervisionSegment(
            id="cuts1-sup0", recording_id=cuts[1].recording_id, start=0, duration=0.5, text="first", speaker="0"
        ),
        SupervisionSegment(
            id="cuts1-sup1", recording_id=cuts[1].recording_id, start=0.5, duration=0.5, text="second", speaker="1"
        ),
    ]

    dataset = LhotseSpeechToTextSpkBpeDataset(cfg={}, tokenizer=tokenizer)
    batch = dataset[cuts]

    assert isinstance(batch, tuple)
    assert len(batch) == 6
    assert all(isinstance(t, torch.Tensor) for t in batch)

    audio, audio_lens, tokens, token_lens, spk_targets, bg_spk_targets = batch

    assert audio.shape == (2, 16000)
    assert audio_lens.tolist() == [16000] * 2

    assert tokens.shape == (2, 11)
    assert tokens[0].tolist() == [1, 10, 19, 19, 6, 13, 6, 23, 2, 15, 21]
    assert tokens[1].tolist() == [1, 20, 6, 4, 16, 15, 5, 0, 0, 0, 0] or tokens[1].tolist() == [
        1,
        7,
        10,
        19,
        20,
        21,
        0,
        0,
        0,
        0,
        0,
    ]
    assert token_lens.tolist() == [11, 7] or token_lens.tolist() == [11, 6]

    assert spk_targets.shape == (2, 13)
    assert spk_targets[0].long().tolist() == [1] * 13
    assert spk_targets[1].long().sum().item() in [6, 7]

    assert bg_spk_targets.shape == (2, 13)
    assert bg_spk_targets[0].long().tolist() == [0] * 13
    assert bg_spk_targets[1].long().sum().item() in [6, 7]

    assert (spk_targets[1] + bg_spk_targets[1]).long().tolist() == [1] * 13
