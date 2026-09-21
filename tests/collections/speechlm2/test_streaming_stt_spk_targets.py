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
from omegaconf import OmegaConf

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.streaming_stt_dataset import StreamingSTTBatch, StreamingSTTDataset


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
