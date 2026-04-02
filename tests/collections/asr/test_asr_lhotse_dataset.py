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

from unittest.mock import patch

import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.dataset import AudioSamples
from lhotse.testing.dummies import DummyManifest

from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
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


def test_lhotse_asr_dataset(tokenizer):
    # 3 cuts of duration 1s with audio and a single supervision with text 'irrelevant'
    cuts = DummyManifest(CutSet, begin_id=0, end_id=3, with_data=True)

    # cuts[0] is the default case: audio + single untokenized superivision

    # cuts[1]: audio + single pre-tokenized superivision
    cuts[1].supervisions[0].tokens = tokenizer.text_to_ids(cuts[1].supervisions[0].text)

    # cuts[2]: audio + two supervisions
    cuts[2].supervisions = [
        SupervisionSegment(id="cuts2-sup0", recording_id=cuts[2].recording_id, start=0, duration=0.5, text="first"),
        SupervisionSegment(id="cuts2-sup1", recording_id=cuts[2].recording_id, start=0.5, duration=0.5, text="second"),
    ]

    dataset = LhotseSpeechToTextBpeDataset(tokenizer=tokenizer)
    batch = dataset[cuts]

    assert isinstance(batch, tuple)
    assert len(batch) == 4
    assert all(isinstance(t, torch.Tensor) for t in batch)

    audio, audio_lens, tokens, token_lens = batch

    assert audio.shape == (3, 16000)
    assert audio_lens.tolist() == [16000] * 3

    assert tokens.shape == (3, 13)
    assert tokens[0].tolist() == [1, 10, 19, 19, 6, 13, 6, 23, 2, 15, 21, 0, 0]
    assert tokens[1].tolist() == tokens[0].tolist()
    assert tokens[2].tolist() == [1, 7, 10, 19, 20, 21, 1, 20, 6, 4, 16, 15, 5]

    assert token_lens.tolist() == [11, 11, 13]


def test_lhotse_asr_dataset_metadata(tokenizer):

    cuts = DummyManifest(CutSet, begin_id=0, end_id=2, with_data=True)

    cuts[0].id = "cuts0"
    cuts[1].id = "cuts1"
    cuts[0].supervisions = [
        SupervisionSegment(id="cuts0-sup0", recording_id=cuts[0].recording_id, start=0.2, duration=0.5, text="first"),
    ]
    cuts[1].supervisions = [
        SupervisionSegment(id="cuts1-sup0", recording_id=cuts[1].recording_id, start=0, duration=1, text=""),
    ]

    datasets_metadata = LhotseSpeechToTextBpeDataset(tokenizer=tokenizer, return_cuts=True)
    batch = datasets_metadata[cuts]
    assert isinstance(batch, tuple)
    assert len(batch) == 5

    _, _, _, _, cuts_metadata = batch

    assert cuts_metadata[0].supervisions[0].text == "first"
    assert cuts_metadata[1].supervisions[0].text == ""
    assert cuts_metadata[0].id == "cuts0"
    assert cuts_metadata[1].id == "cuts1"

    assert cuts_metadata[0].supervisions[0].duration == 0.5
    assert cuts_metadata[0].supervisions[0].start == 0.2

    assert cuts_metadata[1].supervisions[0].duration == 1
    assert cuts_metadata[1].supervisions[0].start == 0.0


def test_lhotse_asr_dataset_ais_batch_loading_enabled(tokenizer, monkeypatch):
    """Test that USE_AIS_GET_BATCH=true passes use_batch_loader=True to AudioSamples."""
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")

    with patch.object(AudioSamples, "__init__", return_value=None) as mock_init:
        mock_init.side_effect = lambda *args, **kwargs: None
        try:
            dataset = LhotseSpeechToTextBpeDataset(tokenizer=tokenizer)
        except Exception:
            pass
        # Check that AudioSamples was called with use_batch_loader=True
        mock_init.assert_called_with(fault_tolerant=True, use_batch_loader=True)


def test_lhotse_asr_dataset_ais_batch_loading_disabled(tokenizer, monkeypatch):
    """Test that without USE_AIS_GET_BATCH, use_batch_loader=False is passed to AudioSamples."""
    monkeypatch.delenv("USE_AIS_GET_BATCH", raising=False)

    with patch.object(AudioSamples, "__init__", return_value=None) as mock_init:
        mock_init.side_effect = lambda *args, **kwargs: None
        try:
            dataset = LhotseSpeechToTextBpeDataset(tokenizer=tokenizer)
        except Exception:
            pass
        # Check that AudioSamples was called with use_batch_loader=False
        mock_init.assert_called_with(fault_tolerant=True, use_batch_loader=False)


def test_lhotse_asr_dataset_ais_batch_loading_fallback(tokenizer, monkeypatch):
    """Test fallback when Lhotse doesn't support use_batch_loader (< 1.32.0)."""
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")

    call_args = []

    original_init = AudioSamples.__init__

    def mock_init(self, *args, **kwargs):
        call_args.append(kwargs.copy())
        if "use_batch_loader" in kwargs:
            raise TypeError("unexpected keyword argument 'use_batch_loader'")
        return original_init(self, *args, **kwargs)

    with patch.object(AudioSamples, "__init__", mock_init):
        dataset = LhotseSpeechToTextBpeDataset(tokenizer=tokenizer)

    # First call should have use_batch_loader=True, second call should not
    assert call_args[0] == {"fault_tolerant": True, "use_batch_loader": True}
    assert call_args[1] == {"fault_tolerant": True}
