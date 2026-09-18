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
from pathlib import Path

import pytest
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.cut import MixedCut
from lhotse.testing.dummies import DummyManifest, dummy_cut, dummy_recording
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.cutset import LazyConcatCuts, read_cutset_from_config


def _make_cuts(durations):
    """Create a CutSet with MonoCuts of given durations backed by in-memory audio."""
    cuts = []
    for i, dur in enumerate(durations):
        num_samples = int(dur * 16000)
        rec = Recording(
            id=f"rec-{i}",
            sources=[],
            sampling_rate=16000,
            num_samples=num_samples,
            duration=dur,
        )
        cut = MonoCut(
            id=f"cut-{i}",
            start=0.0,
            duration=dur,
            channel=0,
            recording=rec,
            supervisions=[SupervisionSegment(id=f"sup-{i}", recording_id=f"rec-{i}", start=0, duration=dur)],
        )
        cuts.append(cut)
    return CutSet.from_cuts(cuts)


@pytest.fixture(scope="session")
def cutset_path(tmp_path_factory) -> Path:
    """10 utterances of length 1s as a Lhotse CutSet."""
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10, with_data=True)
    for c in cuts:
        c.features = None
        c.custom = None
        c.supervisions[0].custom = None

    tmp_path = tmp_path_factory.mktemp("data")
    p = tmp_path / "cuts.jsonl.gz"
    pa = tmp_path / "audio"
    cuts.save_audios(pa).to_file(p)
    return p


@pytest.fixture(scope="session")
def cutset_varying_durations_path(tmp_path_factory) -> Path:
    """5 utterances with varying durations: 0.2, 0.4, 0.6, 0.8, 1.0."""
    cuts = DummyManifest(CutSet, begin_id=0, end_id=5, with_data=True)
    trimmed = []
    for i, c in enumerate(cuts):
        c.features = None
        c.custom = None
        c.supervisions[0].custom = None
        dur = (i + 1) * 0.2
        trimmed.append(c.truncate(duration=dur))
    cuts = CutSet.from_cuts(trimmed)

    tmp_path = tmp_path_factory.mktemp("data_varying")
    p = tmp_path / "cuts.jsonl.gz"
    pa = tmp_path / "audio"
    cuts.save_audios(pa).to_file(p)
    return p


# ---------------------------------------------------------------------------
# LazyConcatCuts unit tests (pure iterator, no config parsing)
# ---------------------------------------------------------------------------


class TestLazyConcatCuts:
    def test_basic_concat(self):
        """Cuts are concatenated until max_duration is reached."""
        source = _make_cuts([1.0, 1.0, 1.0, 1.0])
        result = list(LazyConcatCuts(source, max_duration=2.5))
        # [1+1=2.0, 1+1=2.0] — each pair fits within 2.5
        assert len(result) == 2
        for c in result:
            assert c.duration == pytest.approx(2.0, abs=0.01)
            assert isinstance(c, MixedCut)

    def test_exact_boundary(self):
        """Cuts whose total equals max_duration exactly are grouped together."""
        source = _make_cuts([1.0, 1.0, 1.0])
        result = list(LazyConcatCuts(source, max_duration=2.0))
        # [1+1=2.0, 1.0 alone]
        assert len(result) == 2
        assert result[0].duration == pytest.approx(2.0, abs=0.01)
        assert result[1].duration == pytest.approx(1.0, abs=0.01)

    def test_single_cut_exceeding_max(self):
        """A single cut longer than max_duration is yielded as-is, never dropped."""
        source = _make_cuts([5.0, 0.5, 0.5])
        result = list(LazyConcatCuts(source, max_duration=2.0))
        assert len(result) == 2
        assert result[0].duration == pytest.approx(5.0, abs=0.01)
        assert result[1].duration == pytest.approx(1.0, abs=0.01)

    def test_all_exceed_max(self):
        """When every cut exceeds max_duration, each is yielded individually."""
        source = _make_cuts([3.0, 4.0, 5.0])
        result = list(LazyConcatCuts(source, max_duration=2.0))
        assert len(result) == 3
        assert result[0].duration == pytest.approx(3.0, abs=0.01)
        assert result[1].duration == pytest.approx(4.0, abs=0.01)
        assert result[2].duration == pytest.approx(5.0, abs=0.01)

    def test_gap_insertion(self):
        """When gap > 0, silence is inserted between concatenated cuts."""
        source = _make_cuts([1.0, 1.0])
        result = list(LazyConcatCuts(source, max_duration=5.0, gap=0.5))
        assert len(result) == 1
        # 1.0 + 0.5 (gap) + 1.0 = 2.5
        assert result[0].duration == pytest.approx(2.5, abs=0.01)

    def test_gap_affects_capacity(self):
        """Gap duration is counted toward max_duration when deciding whether to append."""
        source = _make_cuts([1.0, 1.0, 1.0])
        # Without gap: 1+1+1=3.0 fits in 3.0
        # With gap=0.5: 1+0.5+1=2.5, next would be 2.5+0.5+1=4.0 > 3.0
        result = list(LazyConcatCuts(source, max_duration=3.0, gap=0.5))
        assert len(result) == 2
        assert result[0].duration == pytest.approx(2.5, abs=0.01)
        assert result[1].duration == pytest.approx(1.0, abs=0.01)

    def test_empty_source(self):
        """Empty source produces no output."""
        source = CutSet.from_cuts([])
        result = list(LazyConcatCuts(source, max_duration=10.0))
        assert len(result) == 0

    def test_single_cut(self):
        """A single cut is yielded as-is."""
        source = _make_cuts([2.0])
        result = list(LazyConcatCuts(source, max_duration=10.0))
        assert len(result) == 1
        assert result[0].duration == pytest.approx(2.0, abs=0.01)

    def test_re_iterable(self):
        """LazyConcatCuts supports multiple iterations (required by CutSet.repeat)."""
        source = _make_cuts([1.0, 1.0, 1.0])
        lazy = LazyConcatCuts(source, max_duration=2.5)
        first = list(lazy)
        second = list(lazy)
        assert len(first) == len(second) == 2

    def test_supervisions_preserved(self):
        """Supervisions from all constituent cuts are preserved after concatenation."""
        source = _make_cuts([1.0, 1.0])
        result = list(LazyConcatCuts(source, max_duration=5.0))
        assert len(result) == 1
        sups = result[0].supervisions
        assert len(sups) == 2

    def test_cutset_wrapping(self):
        """LazyConcatCuts works when wrapped in a CutSet."""
        source = _make_cuts([1.0, 1.0, 1.0, 1.0])
        wrapped = CutSet(LazyConcatCuts(source, max_duration=2.5))
        result = list(wrapped)
        assert len(result) == 2

    def test_cuts_with_target_audio(self):
        """Concatenation works when cuts carry a custom target_audio Recording."""
        cuts = []
        for i in range(3):
            c = dummy_cut(i, recording=dummy_recording(i, with_data=True))
            c.target_audio = dummy_recording(100 + i, with_data=True)
            c.supervisions = [
                SupervisionSegment(id=f"sup-{i}", recording_id=c.recording_id, start=0, duration=c.duration)
            ]
            cuts.append(c)
        source = CutSet.from_cuts(cuts)

        result = list(LazyConcatCuts(source, max_duration=2.5))

        # 3 cuts of 1s each -> [1+1=2.0, 1.0]
        assert len(result) == 2

        # The concatenated cut is a MixedCut; each track still has its own target_audio.
        concat_cut = result[0]
        assert isinstance(concat_cut, MixedCut)
        for track in concat_cut.tracks:
            assert hasattr(track.cut, "target_audio"), "target_audio lost after append"
            assert isinstance(track.cut.target_audio, Recording)
            assert track.cut.target_audio.duration == pytest.approx(1.0, abs=0.01)

        # Source audio is still loadable on the concatenated cut.
        audio = concat_cut.load_audio()
        assert audio.shape[0] == 1
        assert audio.shape[1] == pytest.approx(2 * 16000, abs=16)

        # target_audio is loadable on the concatenated MixedCut directly.
        ta = concat_cut.load_target_audio()
        assert ta.shape == (1, 2 * 16000)

        # target_audio is also loadable on individual tracks.
        for track in concat_cut.tracks:
            ta = track.cut.load_target_audio()
            assert ta.shape == (1, 16000)

        # The non-concatenated (single) cut keeps load_target_audio() working directly.
        single_cut = result[1]
        ta = single_cut.load_target_audio()
        assert ta.shape == (1, 16000)

    def test_cuts_with_target_audio_different_sr(self):
        """Concatenation works when source is 16kHz and target_audio is 24kHz."""
        cuts = []
        for i in range(2):
            c = dummy_cut(i, recording=dummy_recording(i, with_data=True, sampling_rate=16000))
            c.target_audio = dummy_recording(100 + i, with_data=True, sampling_rate=24000)
            c.supervisions = [
                SupervisionSegment(id=f"sup-{i}", recording_id=c.recording_id, start=0, duration=c.duration)
            ]
            cuts.append(c)
        source = CutSet.from_cuts(cuts)

        result = list(LazyConcatCuts(source, max_duration=5.0))

        # Both cuts fit within 5.0s -> single concatenated MixedCut
        assert len(result) == 1
        concat_cut = result[0]
        assert isinstance(concat_cut, MixedCut)

        # Source audio is 16kHz.
        audio = concat_cut.load_audio()
        assert audio.shape == (1, 2 * 16000)

        # target_audio is loadable on the concatenated MixedCut at 24kHz.
        ta = concat_cut.load_target_audio()
        assert ta.shape == (1, 2 * 24000)

        # Each track's target_audio retains its 24kHz sampling rate.
        for track in concat_cut.tracks:
            assert track.cut.target_audio.sampling_rate == 24000
            ta = track.cut.load_target_audio()
            assert ta.shape == (1, 24000)

    def test_cuts_with_target_audio_and_gap(self):
        """load_target_audio() works on MixedCut with gap (PaddingCut between data cuts)."""
        cuts = []
        for i in range(2):
            c = dummy_cut(i, recording=dummy_recording(i, with_data=True, sampling_rate=16000))
            c.target_audio = dummy_recording(100 + i, with_data=True, sampling_rate=24000)
            c.supervisions = [
                SupervisionSegment(id=f"sup-{i}", recording_id=c.recording_id, start=0, duration=c.duration)
            ]
            cuts.append(c)
        source = CutSet.from_cuts(cuts)

        result = list(LazyConcatCuts(source, max_duration=5.0, gap=0.5))

        assert len(result) == 1
        concat_cut = result[0]
        assert isinstance(concat_cut, MixedCut)
        # 1.0 + 0.5 gap + 1.0 = 2.5s
        assert concat_cut.duration == pytest.approx(2.5, abs=0.01)

        # Source audio: 2.5s at 16kHz
        audio = concat_cut.load_audio()
        assert audio.shape == (1, int(2.5 * 16000))

        # Target audio: 2.5s at 24kHz (gap filled with silence)
        ta = concat_cut.load_target_audio()
        assert ta.shape == (1, int(2.5 * 24000))

    def test_varying_durations(self):
        """Greedy packing with varying-length cuts."""
        source = _make_cuts([0.5, 1.5, 0.8, 0.7, 2.0])
        result = list(LazyConcatCuts(source, max_duration=2.5))
        # 0.5+1.5=2.0 (fits, next would be 2.0+0.8=2.8 > 2.5)
        # 0.8+0.7=1.5 (fits, next would be 1.5+2.0=3.5 > 2.5)
        # 2.0 alone
        assert len(result) == 3
        assert result[0].duration == pytest.approx(2.0, abs=0.01)
        assert result[1].duration == pytest.approx(1.5, abs=0.01)
        assert result[2].duration == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# Integration tests via read_cutset_from_config (end-to-end with config)
# ---------------------------------------------------------------------------


def test_lhotse_concat_from_config(cutset_path):
    """End-to-end: read lhotse_concat config and get concatenated cuts."""
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "lhotse_concat",
                    "max_duration": 2.5,
                    "input_cfg": [
                        {
                            "type": "lhotse",
                            "cuts_path": str(cutset_path),
                        }
                    ],
                }
            ],
            "force_finite": True,
        }
    )

    cuts, is_tarred = read_cutset_from_config(config)
    result = list(cuts)

    # 10 cuts of 1s each, max_duration=2.5 => pairs of 2, so 5 results
    assert len(result) == 5
    for c in result:
        assert c.duration == pytest.approx(2.0, abs=0.01)
        assert isinstance(c, MixedCut)


def test_lhotse_concat_from_config_with_gap(cutset_path):
    """End-to-end with gap parameter."""
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "lhotse_concat",
                    "max_duration": 5.0,
                    "gap": 0.5,
                    "input_cfg": [
                        {
                            "type": "lhotse",
                            "cuts_path": str(cutset_path),
                        }
                    ],
                }
            ],
            "force_finite": True,
        }
    )

    cuts, is_tarred = read_cutset_from_config(config)
    result = list(cuts)

    # 10 cuts of 1s with gap=0.5, max_duration=5.0
    # 1 + (0.5+1)*3 = 5.5 > 5.0, so 3 per group: 1 + 0.5 + 1 + 0.5 + 1 = 4.0
    # Group of 3 = 4.0s, then group of 3 = 4.0s, then group of 3 = 4.0s, then 1 alone = 1.0s
    assert len(result) == 4
    for c in result[:-1]:
        assert c.duration == pytest.approx(4.0, abs=0.01)
    assert result[-1].duration == pytest.approx(1.0, abs=0.01)


def test_lhotse_concat_with_tags(cutset_path):
    """Tags are propagated through lhotse_concat to inner cuts."""
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "lhotse_concat",
                    "max_duration": 2.5,
                    "tags": {"language": "en"},
                    "input_cfg": [
                        {
                            "type": "lhotse",
                            "cuts_path": str(cutset_path),
                        }
                    ],
                }
            ],
            "force_finite": True,
        }
    )

    cuts, is_tarred = read_cutset_from_config(config)
    result = list(cuts)

    assert len(result) > 0


def test_lhotse_concat_varying_durations(cutset_varying_durations_path):
    """End-to-end with varying-duration cuts."""
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "lhotse_concat",
                    "max_duration": 1.5,
                    "input_cfg": [
                        {
                            "type": "lhotse",
                            "cuts_path": str(cutset_varying_durations_path),
                        }
                    ],
                }
            ],
            "force_finite": True,
        }
    )

    cuts, is_tarred = read_cutset_from_config(config)
    result = list(cuts)

    # Durations: 0.2, 0.4, 0.6, 0.8, 1.0
    # 0.2+0.4+0.6 = 1.2 (next: 1.2+0.8=2.0 > 1.5)
    # 0.8 (next: 0.8+1.0=1.8 > 1.5)
    # 1.0 alone
    assert len(result) == 3
    assert result[0].duration == pytest.approx(1.2, abs=0.01)
    assert result[1].duration == pytest.approx(0.8, abs=0.01)
    assert result[2].duration == pytest.approx(1.0, abs=0.01)


def test_lhotse_concat_audio_loadable(cutset_path):
    """Concatenated cuts can actually load audio."""
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "lhotse_concat",
                    "max_duration": 2.5,
                    "input_cfg": [
                        {
                            "type": "lhotse",
                            "cuts_path": str(cutset_path),
                        }
                    ],
                }
            ],
            "force_finite": True,
        }
    )

    cuts, _ = read_cutset_from_config(config)
    first_cut = next(iter(cuts))

    audio = first_cut.load_audio()
    # 2 cuts of 1s at 16kHz
    assert audio.shape[1] == pytest.approx(2 * 16000, abs=16)
