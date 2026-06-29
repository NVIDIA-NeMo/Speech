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
from omegaconf import OmegaConf

from nemo.collections.asr.inference.utils.pipeline_eval import calculate_asr_laal
from nemo.collections.asr.parts.utils.eval_utils import compute_laal

# EoU enabled (stop_history_eou >= 0) so ASR LAAL is computed.
CFG = OmegaConf.create({"metrics": {"asr": {"gt_text_attr_name": "text"}}, "endpointing": {"stop_history_eou": 800}})


class TestCalculateAsrLaal:
    @pytest.mark.unit
    def test_matches_direct_compute_laal(self):
        # Two finalized steps: "hello world" committed at 2.0 s, "foo" at 5.0 s of audio elapsed.
        # Each word inherits its step's delay (ms); LAAL is computed against the reference word count.
        durations = {"a.wav": 10.0}  # seconds -> 10000 ms
        manifest = [{"audio_filepath": "a.wav", "text": "hello world foo bar"}]  # 4 reference words
        output = {0: {"audio_filepath": "a.wav", "asr_segments": [("hello world", 2.0), (" foo", 5.0)]}}

        expected = compute_laal([2000.0, 2000.0, 5000.0], 10000.0, 4)
        got = calculate_asr_laal(output, durations, manifest, CFG)
        assert got == pytest.approx(expected)

    @pytest.mark.unit
    def test_delay_capped_at_duration(self):
        # A delay beyond the audio duration is clamped to the duration.
        durations = {"a.wav": 3.0}  # 3000 ms
        manifest = [{"audio_filepath": "a.wav", "text": "hello world"}]
        output = {0: {"audio_filepath": "a.wav", "asr_segments": [("hello world", 99.0)]}}

        expected = compute_laal([3000.0, 3000.0], 3000.0, 2)
        assert calculate_asr_laal(output, durations, manifest, CFG) == pytest.approx(expected)

    @pytest.mark.unit
    def test_no_manifest_returns_none(self):
        output = {0: {"audio_filepath": "a.wav", "asr_segments": [("hi", 1.0)]}}
        assert calculate_asr_laal(output, {"a.wav": 1.0}, None, CFG) is None

    @pytest.mark.unit
    def test_no_reference_returns_none(self):
        # Stream's audio is absent from the manifest -> nothing to score.
        output = {0: {"audio_filepath": "missing.wav", "asr_segments": [("hi", 1.0)]}}
        manifest = [{"audio_filepath": "other.wav", "text": "hello"}]
        assert calculate_asr_laal(output, {"missing.wav": 1.0}, manifest, CFG) is None

    @pytest.mark.unit
    def test_eou_disabled_returns_none(self):
        # EoU disabled (stop_history_eou < 0) -> one segment per utterance, no latency signal -> skipped.
        cfg = OmegaConf.create(
            {"metrics": {"asr": {"gt_text_attr_name": "text"}}, "endpointing": {"stop_history_eou": -1}}
        )
        manifest = [{"audio_filepath": "a.wav", "text": "hello world"}]
        output = {0: {"audio_filepath": "a.wav", "asr_segments": [("hello world", 2.0)]}}
        assert calculate_asr_laal(output, {"a.wav": 10.0}, manifest, cfg) is None
