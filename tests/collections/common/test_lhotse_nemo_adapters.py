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

import numpy as np
import pytest
from lhotse import AudioSource, CutSet, MonoCut, MultiCut, Recording, SupervisionSegment
from lhotse.serialization import save_to_jsonl
from lhotse.testing.dummies import DummyManifest, dummy_multi_cut, dummy_supervision

from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator


@pytest.fixture
def nemo_manifest_path(tmp_path_factory):
    """2 utterances of length 1s as a NeMo manifest."""
    tmpdir = tmp_path_factory.mktemp("nemo_data")
    cuts = DummyManifest(CutSet, begin_id=0, end_id=2, with_data=True).save_audios(tmpdir, progress_bar=False)
    nemo = []
    for c in cuts:
        nemo.append(
            {
                "audio_filepath": c.recording.sources[0].source,
                "text": "irrelevant",
                "duration": c.duration,
                "lang": "en",
            }
        )
    p = tmpdir / "nemo_manifest.json"
    save_to_jsonl(nemo, p)
    return p


@pytest.fixture
def nemo_manifest_path_multichannel(tmp_path_factory):
    """2 utterances of length 1s with 3 channels as a NeMo manifest."""
    tmpdir = tmp_path_factory.mktemp("nemo_data")
    cuts = CutSet.from_cuts(
        dummy_multi_cut(idx, supervisions=[dummy_supervision(idx)], channel=[0, 1, 2], with_data=True)
        for idx in range(0, 2)
    ).save_audios(tmpdir, progress_bar=False)
    nemo = []
    for c in cuts:
        nemo.append(
            {
                "audio_filepath": c.recording.sources[0].source,
                "text": "irrelevant",
                "duration": c.duration,
                "lang": "en",
            }
        )
    p = tmpdir / "nemo_manifest_multichannel.json"
    save_to_jsonl(nemo, p)
    return p


def test_lazy_nemo_iterator(nemo_manifest_path):
    cuts = CutSet(LazyNeMoIterator(nemo_manifest_path))

    assert len(cuts) == 2

    for c in cuts:
        assert isinstance(c, MonoCut)
        assert c.start == 0.0
        assert c.duration == 1.0
        assert c.num_channels == 1
        assert c.sampling_rate == 16000
        assert c.num_samples == 16000

        assert c.has_recording
        assert isinstance(c.recording, Recording)
        assert c.recording.duration == 1.0
        assert c.recording.num_channels == 1
        assert c.recording.num_samples == 16000
        assert len(c.recording.sources) == 1
        assert isinstance(c.recording.sources[0], AudioSource)
        assert c.recording.sources[0].type == "file"

        audio = c.load_audio()
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (1, 16000)
        assert audio.dtype == np.float32

        assert len(c.supervisions) == 1
        s = c.supervisions[0]
        assert isinstance(s, SupervisionSegment)
        assert s.start == 0
        assert s.duration == 1
        assert s.channel == 0
        assert s.text == "irrelevant"
        assert s.language == "en"


def test_lazy_nemo_iterator_multichannel(nemo_manifest_path_multichannel):
    cuts = CutSet(LazyNeMoIterator(nemo_manifest_path_multichannel))

    assert len(cuts) == 2

    for c in cuts:
        assert isinstance(c, MultiCut)
        assert c.start == 0.0
        assert c.duration == 1.0
        assert c.num_channels == 3
        assert c.channel == [0, 1, 2]  # cuts have three channels
        assert c.sampling_rate == 16000
        assert c.num_samples == 16000

        assert c.has_recording
        assert isinstance(c.recording, Recording)
        assert c.recording.duration == 1.0
        assert c.recording.num_channels == 3
        assert c.recording.num_samples == 16000
        assert len(c.recording.sources) == 1
        assert isinstance(c.recording.sources[0], AudioSource)
        assert c.recording.sources[0].type == "file"
        assert c.recording.sources[0].channels == c.channel  # recording has same channels as the cut

        audio = c.load_audio()
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (c.num_channels, 16000)  # audio has same num_channels as the cut
        assert audio.dtype == np.float32

        assert len(c.supervisions) == 1
        s = c.supervisions[0]
        assert isinstance(s, SupervisionSegment)
        assert s.start == 0
        assert s.duration == 1
        assert s.channel == c.channel  # supervision has same channels as the cut
        assert s.text == "irrelevant"
        assert s.language == "en"


@pytest.fixture
def nemo_offset_manifest_path(tmp_path_factory):
    """
    4 utterances of length 0.5s as a NeMo manifest.
    They are dervied from two audio files of 1s duration, so
    two of them have offset 0 and the other two have offset 0.5.
    """
    tmpdir = tmp_path_factory.mktemp("nemo_data_offset")
    cuts = (
        DummyManifest(CutSet, begin_id=0, end_id=2, with_data=True)
        .save_audios(tmpdir, progress_bar=False)
        .cut_into_windows(duration=0.5, hop=0.5)
    )
    nemo = []
    for c in cuts:
        nemo.append(
            {
                "audio_filepath": c.recording.sources[0].source,
                "text": "irrelevant",
                "offset": c.start,
                "duration": c.duration,
                "lang": "en",
            }
        )
    p = tmpdir / "nemo_manifest.json"
    save_to_jsonl(nemo, p)
    return p


def test_lazy_nemo_iterator_with_offset(nemo_offset_manifest_path):
    cuts = CutSet(LazyNeMoIterator(nemo_offset_manifest_path))

    assert len(cuts) == 4

    for idx, c in enumerate(cuts):
        # Note we originally had 1 cut per 1s audio file.
        # Then we cut them into 0.5s cuts, so we have 4 cuts in total,
        # 2 of them start at 0s and the other 2 start at 0.5s.
        is_even = idx % 2 == 0

        assert isinstance(c, MonoCut)
        if is_even:
            assert c.start == 0.0
        else:
            assert c.start == 0.5
        assert c.duration == 0.5
        assert c.num_channels == 1
        assert c.sampling_rate == 16000
        assert c.num_samples == 8000

        assert c.has_recording
        assert isinstance(c.recording, Recording)
        assert c.recording.duration == 1.0
        assert c.recording.num_channels == 1
        assert c.recording.num_samples == 16000
        assert len(c.recording.sources) == 1
        assert isinstance(c.recording.sources[0], AudioSource)
        assert c.recording.sources[0].type == "file"

        audio = c.load_audio()
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (1, 8000)
        assert audio.dtype == np.float32

        assert len(c.supervisions) == 1
        s = c.supervisions[0]
        assert isinstance(s, SupervisionSegment)
        assert s.start == 0
        assert s.duration == 0.5
        assert s.channel == 0
        assert s.text == "irrelevant"
        assert s.language == "en"


def test_lazy_nemo_iterator_with_offset_metadata_only(nemo_offset_manifest_path):
    cuts = CutSet(LazyNeMoIterator(nemo_offset_manifest_path, metadata_only=True))

    assert len(cuts) == 4

    for idx, c in enumerate(cuts):
        # Note we originally had 1 cut per 1s audio file.
        # Then we cut them into 0.5s cuts, so we have 4 cuts in total,
        # 2 of them start at 0s and the other 2 start at 0.5s.
        is_even = idx % 2 == 0

        assert isinstance(c, MonoCut)
        if is_even:
            assert c.start == 0.0
        else:
            assert c.start == 0.5
        assert c.duration == 0.5
        assert c.num_channels == 1
        assert c.sampling_rate == 16000
        assert c.num_samples == 8000

        # With metadata_only=True we can't actually check what's in the Recording.
        # The metadata for it may be incorrect (but is correct for the actual Cut),
        # but we don't have to perform any I/O to read the file for info.
        assert c.has_recording
        assert isinstance(c.recording, Recording)
        if is_even:
            assert c.recording.duration == 0.5
            assert c.recording.num_samples == 8000
        else:
            assert c.recording.duration == 1.0
            assert c.recording.num_samples == 16000
        assert c.recording.num_channels == 1
        assert len(c.recording.sources) == 1
        assert isinstance(c.recording.sources[0], AudioSource)
        assert c.recording.sources[0].type == "dummy"

        with pytest.raises(AssertionError):
            c.load_audio()

        assert len(c.supervisions) == 1
        s = c.supervisions[0]
        assert isinstance(s, SupervisionSegment)
        assert s.start == 0
        assert s.duration == 0.5
        assert s.channel == 0
        assert s.text == "irrelevant"
        assert s.language == "en"
