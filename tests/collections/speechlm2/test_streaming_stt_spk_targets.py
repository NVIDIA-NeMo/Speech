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

"""RTTM-derived ``spk_targets`` in StreamingSTTDataset."""

import pytest
import torch
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.audio import AudioSource
from lhotse.testing.dummies import dummy_recording
from omegaconf import OmegaConf

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.streaming_stt_dataset import StreamingSTTBatch, StreamingSTTDataset
from nemo.collections.speechlm2.parts.alignments import get_word_alignments_for_batch

# Two speakers who both start at 0 s, so the order in which they appear does not settle their columns, and their RTTM
# labels are not the tags: `fix_speaker_activity` picks the order. b, whose turns hold the words of the first, third
# and last tags, is <spk:0>, with 58 active frames to a's 60. The two orders cost nearly the same. A two-speaker
# LibriSpeech simulation, its speakers renamed.
NEAR_TIE_SECONDS = 11.294
NEAR_TIE_TEXT = (
    "<spk:0> sancho said they had what led <spk:1> under <spk:0> he cruel <spk:1> i did not know what he meant now "
    "it is a remarkable thing that i have always had <spk:0> seeing this"
)
NEAR_TIE_TURNS = [
    ("a", 0.000, 0.145),
    ("b", 0.000, 2.010),
    ("a", 2.830, 0.430),
    ("b", 3.179, 0.580),
    ("b", 4.261, 1.020),
    ("a", 4.918, 3.920),
    ("b", 10.344, 0.950),
    ("a", 11.036, 0.258),
]


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer("Qwen/Qwen3-1.7B", use_fast=True)
    tok.add_special_tokens({"additional_special_tokens": ["<blank>"] + [f"<spk:{i}>" for i in range(4)]})
    return tok


def _cfg(**overrides):
    ms = {
        "enable": True,
        "num_speakers": 4,
        "sample_rate": 16000,
        "window_stride": 0.01,
        "subsampling_factor": 8,
        "max_alignment_permutations": 720,
    }
    ms.update(overrides.pop("multispeaker_cfg", {}))
    base = {
        "sample_rate": 16000,
        "frame_length_in_secs": 0.08,
        "chunk_size": 14,
        "blank_token": "<blank>",
        "words_per_group": 1,
        "multispeaker_cfg": ms,
    }
    base.update(overrides)
    return OmegaConf.create(base)


class TestSpkTargetsConfig:
    @pytest.mark.unit
    @pytest.mark.parametrize("budget,expected", [(720, 6), (120, 5), (24, 4), (2, 2), (1, 1)])
    def test_permutation_budget_maps_to_a_speaker_cap(self, tokenizer, budget, expected):
        # `fix_speaker_activity` takes a speaker count but the reference config expresses the limit
        # as a permutation budget. At 8 active speakers the uncapped search measured 48.7 s per cut;
        # capped at 6 it is instant.
        ds = StreamingSTTDataset(
            cfg=_cfg(multispeaker_cfg={"max_alignment_permutations": budget}), tokenizer=tokenizer
        )
        assert ds._ms.max_permutable == expected

    @pytest.mark.unit
    def test_budget_none_disables_the_cap(self, tokenizer):
        ds = StreamingSTTDataset(cfg=_cfg(multispeaker_cfg={"max_alignment_permutations": None}), tokenizer=tokenizer)
        assert ds._ms.max_permutable is None

    @pytest.mark.unit
    def test_absent_multispeaker_cfg_leaves_the_path_inert(self, tokenizer):
        cfg = _cfg()
        del cfg.multispeaker_cfg
        ds = StreamingSTTDataset(cfg=cfg, tokenizer=tokenizer)
        assert ds._multispeaker_enabled is False
        assert ds._speaker_token_template is None
        assert ds._build_speaker_activities(cuts=[], text=[]) is None

    @pytest.mark.unit
    def test_words_per_group_above_one_is_rejected(self, tokenizer):
        # A speaker change on a non-first word of a group would be silently dropped and those
        # words attributed to the previous speaker, so refuse rather than corrupt the targets.
        with pytest.raises(ValueError, match="words_per_group=1"):
            StreamingSTTDataset(cfg=_cfg(words_per_group=3), tokenizer=tokenizer)

    @pytest.mark.unit
    def test_multi_token_speaker_tag_is_rejected(self):
        # Without registration `<spk:0>` is 6 tokens in Qwen3, which would swamp the loss and make
        # every speaker change cost six emissions.
        bare = AutoTokenizer("Qwen/Qwen3-1.7B", use_fast=True)
        bare.add_special_tokens({"additional_special_tokens": ["<blank>"]})
        with pytest.raises(ValueError, match="single special token"):
            StreamingSTTDataset(cfg=_cfg(), tokenizer=bare)

    @pytest.mark.unit
    @pytest.mark.parametrize("num_speakers,expected", [(8, 4), (4, 4), (2, 2)])
    def test_probe_counts_registered_tags_not_columns(self, tokenizer, num_speakers, expected):
        # The model registers `speaker_tokens.max_speakers` tags (4 here), which may be fewer than
        # the target columns: the reference fuses 8 columns and emits 4 tags. `<spk:4>` is six
        # tokens, so probing every column index would reject that configuration.
        ds = StreamingSTTDataset(cfg=_cfg(multispeaker_cfg={"num_speakers": num_speakers}), tokenizer=tokenizer)
        assert ds._num_speaker_tags == expected


class TestMultiSpeakerConfigDataclass:
    @pytest.mark.unit
    def test_shared_with_salm_and_back_compatible(self):
        # One definition, importable from both places: SALM code and tests predate the move.
        from nemo.collections.speechlm2.data.salm_dataset import MultiSpeakerConfig as FromSalm
        from nemo.collections.speechlm2.parts.multispeaker import MultiSpeakerConfig as Shared

        assert FromSalm is Shared

    @pytest.mark.unit
    def test_reference_defaults_are_unchanged(self):
        from nemo.collections.speechlm2.parts.multispeaker import MultiSpeakerConfig

        cfg = MultiSpeakerConfig()
        # `no_rttm_to_ones` was retired: an unlabelled cut now gets the missing-RTTM
        # sentinel unconditionally, matching SALM.
        assert cfg.num_speakers == 4
        assert not hasattr(cfg, 'no_rttm_to_ones')
        assert (cfg.num_sample_per_mel_frame, cfg.num_mel_frame_per_target_frame) == (160, 8)

    @pytest.mark.unit
    def test_from_dict_collapses_yaml_knobs_into_frame_rates(self):
        from nemo.collections.speechlm2.parts.multispeaker import MultiSpeakerConfig

        cfg = MultiSpeakerConfig.from_dict(
            {"num_speakers": 4, "window_stride": 0.01, "sample_rate": 16000, "subsampling_factor": 8}
        )
        assert cfg.num_sample_per_mel_frame == 160
        assert cfg.num_mel_frame_per_target_frame == 8

    @pytest.mark.unit
    def test_from_dict_none_returns_none(self):
        from nemo.collections.speechlm2.parts.multispeaker import MultiSpeakerConfig

        assert MultiSpeakerConfig.from_dict(None) is None


class TestBatchSchema:
    @pytest.mark.unit
    def test_batch_exposes_reference_named_fields(self):
        # Names mirror the SALM/phPEE reference so recipes and code port across unchanged.
        batch = StreamingSTTBatch()
        assert batch.spk_targets is None
        assert batch.spk_target_length is None


class TestTargetsIgnoreBatchPadding:
    """A row's speaker targets must not depend on the batch it is drawn into.

    ``collate_audio`` pads every cut with silence to the longest one in its batch. The targets were built on the
    padded cut, whose silent tail could flip the column order that ``fix_speaker_activity`` picks, and could mark the
    frame after the cut's end. They are built on the cut as sampled, then zero-padded to the batch width.
    """

    @staticmethod
    def _cut(tmp_path, uid, seconds, text, turns=None, loads=True):
        if loads:
            recording = dummy_recording(uid, duration=seconds, with_data=True)
        else:  # its audio file does not exist
            recording = Recording(
                id=f"missing-{uid}",
                sources=[AudioSource(type="file", channels=[0], source=str(tmp_path / f"{uid}.wav"))],
                sampling_rate=16000,
                num_samples=round(seconds * 16000),
                duration=seconds,
            )
        custom = {}
        if turns is not None:
            rttm = tmp_path / f"{uid}.rttm"
            rttm.write_text(
                "".join(
                    f"SPEAKER rec 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n" for spk, start, dur in turns
                )
            )
            custom["rttm_filepath"] = str(rttm)
        supervision = SupervisionSegment(
            id=f"sup-{uid}", recording_id=recording.id, start=0.0, duration=seconds, text=text
        )
        return MonoCut(
            id=f"cut-{uid}",
            start=0.0,
            duration=seconds,
            channel=0,
            recording=recording,
            supervisions=[supervision],
            custom=custom,
        )

    @staticmethod
    def _batch(dataset, cuts):
        batch = dataset[CutSet(cuts)]
        if dataset.defer_get_batch:
            # What StreamingSTTModel.training_step does with a deferred batch, with the alignments of its forced
            # aligner; the targets do not depend on them.
            batch = dataset.get_batch_data(
                cuts=batch.cuts,
                audios=batch.audios,
                audio_lens=batch.audio_lens,
                alignments=get_word_alignments_for_batch(batch.cuts),
                text=batch.text,
            )
        return batch

    @pytest.mark.unit
    @pytest.mark.parametrize("defer", [False, True], ids=["eager", "defer_get_batch"])
    def test_a_near_tie_keeps_its_columns_when_the_batch_pads_it(self, tokenizer, tmp_path, defer):
        """Padded by 1 s, the cut's columns swapped: the other order won on the padded activity."""
        dataset = StreamingSTTDataset(cfg=_cfg(), tokenizer=tokenizer, defer_get_batch=defer)
        cut = self._cut(tmp_path, 1, NEAR_TIE_SECONDS, NEAR_TIE_TEXT, NEAR_TIE_TURNS)
        longer = self._cut(tmp_path, 2, NEAR_TIE_SECONDS + 1.0, "<spk:0> hello world")

        alone = self._batch(dataset, [cut]).spk_targets
        padded = self._batch(dataset, [cut, longer]).spk_targets  # `cut` gets 1 s of silence

        frames = alone.shape[1]
        assert alone[0].sum(0).tolist() == [58.0, 60.0, 0.0, 0.0]  # b, then a
        assert padded.shape[1] > frames
        assert padded[0].sum(0).tolist() == [58.0, 60.0, 0.0, 0.0]  # it was [60, 58, 0, 0]: a, then b
        assert torch.equal(padded[0, :frames], alone[0])
        assert not padded[0, frames:].any()
        if defer:
            # The model rebuilds the targets from the batch's cuts, so those are the cuts as sampled.
            assert [c.duration for c in dataset[CutSet([cut, longer])].cuts] == [cut.duration, longer.duration]

    @pytest.mark.unit
    def test_a_cut_the_collator_drops_leaves_the_others_their_own_targets(self, tokenizer, tmp_path):
        """With ``fault_tolerant=True`` the collator drops a cut whose audio fails to load (here, a missing file).
        Each survivor is matched to the cut it was sampled as, not to the cut at its position."""
        dataset = StreamingSTTDataset(cfg=_cfg(), tokenizer=tokenizer)
        broken = self._cut(tmp_path, 3, 5.0, "<spk:0> lost", [("c", 0.0, 5.0)], loads=False)
        cut = self._cut(tmp_path, 1, NEAR_TIE_SECONDS, NEAR_TIE_TEXT, NEAR_TIE_TURNS)
        longer = self._cut(tmp_path, 2, NEAR_TIE_SECONDS + 1.0, "<spk:0> hello world")

        alone = self._batch(dataset, [cut]).spk_targets
        batch = self._batch(dataset, [broken, cut, longer])

        frames = alone.shape[1]
        assert batch.text == [NEAR_TIE_TEXT, "<spk:0> hello world"]
        assert batch.spk_targets.shape[0] == 2
        assert batch.spk_targets[0].sum(0).tolist() == [58.0, 60.0, 0.0, 0.0]
        assert torch.equal(batch.spk_targets[0, :frames], alone[0])
        assert not batch.spk_targets[0, frames:].any()
        assert bool((batch.spk_targets[1] == -1.0).all())  # `longer` has no RTTM: the sentinel

    @pytest.mark.unit
    @pytest.mark.parametrize("defer", [False, True], ids=["eager", "defer_get_batch"])
    def test_rows_keep_their_values_and_the_batch_width(self, tokenizer, tmp_path, defer):
        """What building on the cuts as sampled leaves as it was: rows whose column order is clear and whose speakers
        are quiet at the cut's end, a sentinel row across the whole width, and the width of the longest cut."""
        dataset = StreamingSTTDataset(cfg=_cfg(), tokenizer=tokenizer, defer_get_batch=defer)
        two = self._cut(
            tmp_path, 1, 2.0, "<spk:0> hello there <spk:1> good morning", [("a", 0.08, 0.8), ("b", 1.04, 0.64)]
        )
        unlabelled = self._cut(tmp_path, 2, 1.2, "<spk:0> hi")
        longest = self._cut(tmp_path, 3, 3.2, "<spk:0> one two <spk:1> three", [("x", 0.0, 1.6), ("y", 1.76, 1.04)])

        batch = self._batch(dataset, [two, unlabelled, longest])

        expected = torch.zeros(3, 40, 4)  # 3.2 s at 80 ms per frame
        expected[0, 2:12, 0] = 1.0  # a, 0.08-0.88 s
        expected[0, 14:22, 1] = 1.0  # b, 1.04-1.68 s
        expected[1] = -1.0  # no RTTM
        expected[2, 0:21, 0] = 1.0  # x, 0-1.6 s
        expected[2, 23:36, 1] = 1.0  # y, 1.76-2.8 s
        assert torch.equal(batch.spk_targets, expected)

    @pytest.mark.unit
    def test_cuts_that_share_an_id_keep_their_own_targets(self, tokenizer, tmp_path):
        """The survivors are matched by id, so cuts that share one (two sources may name their cuts alike) are
        matched in order."""
        dataset = StreamingSTTDataset(cfg=_cfg(), tokenizer=tokenizer)
        first = self._cut(
            tmp_path, 1, 2.0, "<spk:0> hello there <spk:1> good morning", [("a", 0.08, 0.8), ("b", 1.04, 0.64)]
        )
        second = self._cut(tmp_path, 2, 1.2, "<spk:0> hi")
        second.id = first.id

        batch = self._batch(dataset, [first, second])

        assert batch.text == ["<spk:0> hello there <spk:1> good morning", "<spk:0> hi"]
        assert batch.spk_targets[0, :, 0].nonzero().flatten().tolist() == list(range(2, 12))
        assert batch.spk_targets[0, :, 1].nonzero().flatten().tolist() == list(range(14, 22))
        assert bool((batch.spk_targets[1] == -1.0).all())  # no RTTM

    @pytest.mark.unit
    @pytest.mark.parametrize("defer", [False, True], ids=["eager", "defer_get_batch"])
    def test_a_mixed_cut_comes_back_whole(self, tokenizer, tmp_path, defer):
        """The collator pads a cut that already is a ``MixedCut`` by appending a silent track, so taking a track back
        would drop the others. The survivor is the requested cut itself, all its tracks included."""
        dataset = StreamingSTTDataset(cfg=_cfg(), tokenizer=tokenizer, defer_get_batch=defer)
        near_tie = self._cut(tmp_path, 1, NEAR_TIE_SECONDS, NEAR_TIE_TEXT, NEAR_TIE_TURNS)
        overlay = self._cut(tmp_path, 2, 3.0, "<spk:0> in the background")
        mixed = near_tie.mix(overlay, offset_other_by=2.0, preserve_id="left")  # near_tie's id, text and RTTM
        longer = self._cut(tmp_path, 3, NEAR_TIE_SECONDS + 1.0, "<spk:0> hello world")

        alone = self._batch(dataset, [mixed]).spk_targets
        padded = self._batch(dataset, [mixed, longer]).spk_targets  # `mixed` gets 1 s of silence

        frames = alone.shape[1]
        assert padded[0].sum(0).tolist() == [58.0, 60.0, 0.0, 0.0]  # it was [60, 58, 0, 0]
        assert torch.equal(padded[0, :frames], alone[0])
        if defer:
            assert list(dataset[CutSet([mixed, longer])].cuts) == [mixed, longer]  # both tracks of `mixed`

    @pytest.mark.unit
    def test_a_row_ends_at_its_own_last_frame(self, tokenizer, tmp_path):
        """``spk_target_length`` counts each row's own frames, as SALM's does, not the batch width. Past them is
        padding: a speaker who talks until the cut's end is not marked in the frame after it."""
        dataset = StreamingSTTDataset(cfg=_cfg(), tokenizer=tokenizer)
        talker = self._cut(tmp_path, 1, 2.0, "<spk:0> hello there", [("a", 1.2, 0.8)])
        unlabelled = self._cut(tmp_path, 2, 1.2, "<spk:0> hi")
        longest = self._cut(tmp_path, 3, 3.2, "<spk:0> one two", [("x", 0.0, 1.6)])

        batch = self._batch(dataset, [talker, unlabelled, longest])

        assert batch.spk_target_length.tolist() == [25, 15, 40]  # it was [40, 40, 40]
        assert batch.spk_targets[0, :, 0].nonzero().flatten().tolist() == list(range(16, 25))  # it ended at 25
