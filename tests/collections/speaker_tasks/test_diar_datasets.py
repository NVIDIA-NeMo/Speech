# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import json
import os
import random
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch.cuda

from nemo.collections.asr.data.audio_to_diar_label import (
    AudioToSpeechE2ESpkDiarDataset,
    get_frame_targets_from_rttm,
    get_subsegments_to_timestamps,
)
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures, WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.speaker_utils import get_vad_out_from_rttm_line, read_rttm_lines
from nemo.collections.common.parts.preprocessing.collections import EndtoEndDiarizationSpeechLabel
from nemo.core.neural_types import StringType


def is_rttm_length_too_long(rttm_file_path, wav_len_in_sec):
    """
    Check if the maximum RTTM duration exceeds the length of the provided audio file.

    Args:
        rttm_file_path (str): Path to the RTTM file.
        wav_len_in_sec (float): Length of the audio file in seconds.

    Returns:
        bool: True if the maximum RTTM duration is less than or equal to the length of the audio file, False otherwise.
    """
    rttm_lines = read_rttm_lines(rttm_file_path)
    max_rttm_sec = 0
    for line in rttm_lines:
        start, dur = get_vad_out_from_rttm_line(line)
        max_rttm_sec = max(max_rttm_sec, start + dur)
    return max_rttm_sec <= wav_len_in_sec


class TestAudioToSpeechE2ESpkDiarDataset:
    @staticmethod
    def _selection_dataset(
        feat_per_sec=10,
        start_guard_sec=0.2,
        min_first_speaker_sec=0.5,
        splice_silence_sec=0.2,
        max_speakers=4,
    ):
        dataset = object.__new__(AudioToSpeechE2ESpkDiarDataset)
        dataset.soft_label_thres = 0.5
        dataset.feat_per_sec = feat_per_sec
        dataset.subsegment_start_guard_sec = start_guard_sec
        dataset.subsegment_min_first_spk_sec = min_first_speaker_sec
        dataset.subsegment_splice_silence_sec = splice_silence_sec
        dataset.subsegment_nspk_bias = 1.0
        dataset.max_spks = max_speakers
        dataset.soft_targets = False
        return dataset

    @classmethod
    def _runtime_dataset(cls, session_len_sec=0.5, max_speakers=4, sample_rate=100):
        dataset = cls._selection_dataset(max_speakers=max_speakers)
        dataset.round_digits = 2
        dataset.session_len_sec = session_len_sec
        dataset.subsegment_single_chunk_min_len_sec = 0.5
        dataset.subsegment_two_chunk_min_len_sec = 0.2
        dataset.subsegment_two_chunks_rate = 0.0
        dataset.min_subsegment_duration = 0.03
        dataset.featurizer = SimpleNamespace(sample_rate=sample_rate, int_values=False)
        dataset.n_fft = 16
        dataset.hop_length = 10
        dataset.stft_pad_amount = 16
        dataset.subsampling_factor = 1
        dataset.diar_frame_length = 0.1
        dataset.use_asr_style_frame_count = True
        dataset.floor_decimal = 100
        return dataset

    @staticmethod
    @contextmanager
    def _preserve_random_state():
        python_random_state = random.getstate()
        torch_random_state = torch.random.get_rng_state()
        try:
            yield
        finally:
            random.setstate(python_random_state)
            torch.random.set_rng_state(torch_random_state)

    @staticmethod
    def _configured_dataset(tmp_path, **kwargs):
        session_len_sec = kwargs.pop("session_len_sec", 60.0)
        window_stride = kwargs.pop("window_stride", 0.1)
        manifest_path = tmp_path / "minimums_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "audio_filepath": str(tmp_path / "audio.wav"),
                    "duration": 1.0,
                    "rttm_filepath": str(tmp_path / "audio.rttm"),
                }
            ),
            encoding="utf-8",
        )
        return AudioToSpeechE2ESpkDiarDataset(
            manifest_filepath=str(manifest_path),
            soft_label_thres=0.5,
            session_len_sec=session_len_sec,
            num_spks=2,
            featurizer=SimpleNamespace(sample_rate=100),
            fb_featurizer=SimpleNamespace(n_fft=16, hop_length=10, stft_pad_amount=16),
            window_stride=window_stride,
            global_rank=0,
            soft_targets=False,
            device="cpu",
            validate_manifest_paths=False,
            **kwargs,
        )

    @staticmethod
    def _target(frames):
        return torch.tensor(
            [[("A" in frame), ("B" in frame), ("C" in frame)] for frame in frames],
            dtype=torch.bool,
        )

    def _is_eligible(self, dataset, frames, start=0, end=None):
        target = self._target(frames)
        end = len(frames) if end is None else end
        info = dataset._prepare_subsegment_activity(target)
        return dataset._source_chunk_eligibility(info, torch.tensor([start]), torch.tensor([end])).item()

    @staticmethod
    def _eligibility_reference(dataset, activity_info, starts, ends):
        if starts.numel() == 0:
            return torch.zeros(0, dtype=torch.bool)
        starts = starts.long()
        ends = ends.long()
        num_frames, num_speakers = activity_info.activity.shape
        if num_speakers == 0:
            return torch.ones_like(starts, dtype=torch.bool)

        guard_frames = dataset._seconds_to_feature_frames(dataset.subsegment_start_guard_sec)
        minimum_frames = dataset._seconds_to_feature_frames(dataset.subsegment_min_first_spk_sec)
        candidate_silent = activity_info.active_count[starts] == 0
        effective_start = activity_info.next_active[starts]
        has_speech = effective_start < ends
        history_start = torch.clamp(starts - guard_frames, min=0)
        history = activity_info.prefix[starts] - activity_info.prefix[history_start]
        history_total = history.sum(dim=1)
        silence_history_safe = history_total == 0
        all_silence_eligible = candidate_silent & ~has_speech & silence_history_safe

        safe_effective_start = effective_start.clamp(max=max(num_frames - 1, 0))
        first_speaker = activity_info.first_speaker[safe_effective_start]
        exactly_one_speaker = activity_info.active_count[safe_effective_start] == 1
        first_speaker_history = history.gather(dim=1, index=first_speaker.unsqueeze(1)).squeeze(1)
        history_safe = torch.where(
            candidate_silent,
            silence_history_safe,
            history_total - first_speaker_history == 0,
        )
        competitor = torch.minimum(activity_info.next_competitor[safe_effective_start], ends)
        accumulated = (
            activity_info.prefix[competitor, first_speaker] - activity_info.prefix[safe_effective_start, first_speaker]
        )
        condition_a = accumulated >= minimum_frames
        run_end = safe_effective_start + minimum_frames
        safe_run_end = run_end.clamp(max=num_frames)
        run_activity = (
            activity_info.prefix[safe_run_end, first_speaker]
            - activity_info.prefix[safe_effective_start, first_speaker]
        )
        condition_b = (run_end <= ends) & (run_activity >= minimum_frames)
        return all_silence_eligible | (has_speech & exactly_one_speaker & history_safe & (condition_a | condition_b))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "subsegments, expected_shape, expected_dtype",
        [((), (0, 2), torch.long)],
    )
    def test_empty_subsegments_to_timestamps(self, subsegments, expected_shape, expected_dtype):
        timestamps = get_subsegments_to_timestamps(subsegments)

        assert timestamps.shape == expected_shape
        assert timestamps.dtype == expected_dtype

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "audio_rows,expected_padded_rows",
        [
            (
                ((1.0, 2.0), (3.0, 4.0, 5.0), (6.0, 7.0, 8.0, 9.0)),
                ((1.0, 2.0, 0.0, 0.0), (3.0, 4.0, 5.0, 0.0), (6.0, 7.0, 8.0, 9.0)),
            )
        ],
    )
    def test_collate_pads_audio_and_preserves_metadata(self, audio_rows, expected_padded_rows):
        dataset = object.__new__(AudioToSpeechE2ESpkDiarDataset)
        batch = [
            (
                torch.tensor(audio_row),
                torch.tensor(len(audio_row)),
                torch.ones(len(audio_row), 2),
                torch.tensor([len(audio_row)]),
                ["speaker_A", "speaker_B"],
            )
            for audio_row in audio_rows
        ]

        audio_signal, _, _, _, speaker_names = dataset.eesd_train_collate_fn(batch)

        assert torch.equal(audio_signal, torch.tensor(expected_padded_rows))
        assert speaker_names == [["speaker_A", "speaker_B"]] * len(audio_rows)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "output_name,element_type",
        [("speaker_names", StringType)],
    )
    def test_output_types_include_speaker_names(self, output_name, element_type):
        dataset = object.__new__(AudioToSpeechE2ESpkDiarDataset)

        assert output_name in dataset.output_types
        assert isinstance(dataset.output_types[output_name].elements_type, element_type)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "session_len_sec,sample_duration,expected_samples",
        [(0.5, 0.75, 50), (-1.0, 0.75, 75)],
        ids=["configured-session-length", "manifest-session-length"],
    )
    def test_public_sample_has_five_fields(self, session_len_sec, sample_duration, expected_samples):
        dataset = self._runtime_dataset(session_len_sec=session_len_sec, max_speakers=2)
        dataset.collection = [
            SimpleNamespace(audio_file="unused.wav", rttm_file=None, duration=sample_duration, offset=0.0)
        ]
        dataset.subsegment_mode = False
        dataset.featurizer.process = lambda audio_file, offset, duration: torch.ones(round(duration * 100))

        sample = dataset[0]

        assert len(sample) == 5
        audio, audio_len, targets, target_len, speaker_names = sample
        assert audio.shape == (expected_samples,)
        assert audio_len.item() == expected_samples
        assert targets.shape == (target_len.item(), 2)
        assert speaker_names == [None, None]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "rttm_starts, rttm_ends, rttm_speakers, duration, frame_rate, max_spks, "
            "expected_target_shape, expected_collated_shape"
        ),
        [
            (
                (0.0, 0.2, 0.4),
                (0.2, 0.4, 0.6),
                (0, 1, 2),
                1.0,
                10,
                -1,
                (10, 3),
                (2, 3, 3),
            )
        ],
    )
    def test_unlimited_speaker_targets_and_collation(
        self,
        rttm_starts,
        rttm_ends,
        rttm_speakers,
        duration,
        frame_rate,
        max_spks,
        expected_target_shape,
        expected_collated_shape,
    ):
        frame_targets = get_frame_targets_from_rttm(
            rttm_timestamps=(rttm_starts, rttm_ends, rttm_speakers),
            offset=0.0,
            duration=duration,
            round_digits=2,
            feat_per_sec=frame_rate,
            max_spks=max_spks,
        )
        assert frame_targets.shape == expected_target_shape
        assert torch.equal(frame_targets.sum(dim=0), torch.tensor([2.0, 2.0, 2.0]))

        unlimited_dataset = self._selection_dataset(max_speakers=-1)
        batch = [
            (
                torch.ones(4),
                torch.tensor(4),
                torch.ones(2, 2),
                torch.tensor([2]),
                unlimited_dataset._build_speaker_names({0: "speaker_A", 1: "speaker_B"}),
            ),
            (
                torch.ones(6),
                torch.tensor(6),
                torch.ones(3, 3),
                torch.tensor([3]),
                unlimited_dataset._build_speaker_names({0: "speaker_A", 1: "speaker_B", 2: "speaker_C"}),
            ),
        ]
        _, _, targets, _, speaker_names = unlimited_dataset.eesd_train_collate_fn(batch)

        assert targets.shape == expected_collated_shape
        assert torch.count_nonzero(targets[0, :, 2]) == 0
        assert speaker_names == [
            ["speaker_A", "speaker_B", None],
            ["speaker_A", "speaker_B", "speaker_C"],
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "seconds,expected_frames",
        [(0.25, 2), (0.50, 4)],
    )
    def test_seconds_to_frames_uses_ceil_at_nonstandard_rate(self, seconds, expected_frames):
        dataset = self._selection_dataset(feat_per_sec=7)
        assert dataset._seconds_to_feature_frames(seconds) == expected_frames

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "soft_targets,expected_values",
        [
            (False, [0.0, 1.0, 1.0]),
            (True, [0.25, 0.50, 0.75]),
        ],
        ids=["hard-thresholds", "soft-values"],
    )
    def test_segment_targets_respect_soft_targets_setting(self, monkeypatch, soft_targets, expected_values):
        dataset = self._selection_dataset()
        averaged_targets = torch.tensor([[0.25, 0.50, 0.75]])
        dataset.soft_targets = soft_targets
        monkeypatch.setattr(dataset, "get_soft_targets_seg", lambda **kwargs: averaged_targets)

        targets = dataset._get_segment_targets(torch.empty(0), torch.tensor([1]))
        assert torch.equal(targets, torch.tensor([expected_values]))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("seed,num_frames,num_speakers,active_threshold,candidate_pool_size," "num_candidates,max_candidate_length"),
        [
            (3, 200, 6, 0.85, 190, 120, 30),
            (17, 80, 3, 0.70, 75, 50, 18),
        ],
        ids=["sparse-six-speaker", "dense-three-speaker"],
    )
    def test_optimized_eligibility_matches_full_history_reference(
        self,
        seed,
        num_frames,
        num_speakers,
        active_threshold,
        candidate_pool_size,
        num_candidates,
        max_candidate_length,
    ):
        dataset = self._selection_dataset()
        generator = torch.Generator().manual_seed(seed)
        target = torch.rand(num_frames, num_speakers, generator=generator) > active_threshold
        info = dataset._prepare_subsegment_activity(target)
        starts = torch.randperm(candidate_pool_size, generator=generator)[:num_candidates]
        lengths = torch.randint(1, max_candidate_length, (num_candidates,), generator=generator)
        ends = torch.minimum(starts + lengths, torch.tensor(len(target)))

        expected = self._eligibility_reference(dataset, info, starts, ends)
        actual = dataset._source_chunk_eligibility(info, starts, ends)

        assert torch.equal(actual, expected)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "frames,start,start_guard_sec,expected_eligible",
        [
            (("-",) * 8, 2, 0.2, True),
            (("B", "-", "-", "-", "-"), 2, 0.2, False),
            (("-", "-", "A", "A", "A", "A", "A"), 0, 0.2, True),
            (("-", "-", "AB", "AB", "AB", "AB", "AB"), 0, 0.2, False),
            (("A", "A", "-", "-", "A", "A", "A", "B"), 0, 0.2, True),
            (("A", "A", "-", "A", "A", "B"), 0, 0.2, False),
            (("A", "A", "A", "A", "A"), 0, 1.0, True),
        ],
        ids=[
            "safe-all-silence",
            "unsafe-guard-history",
            "leading-silence-with-evidence",
            "simultaneous-effective-start",
            "exact-accumulated-threshold",
            "insufficient-evidence",
            "recording-start-clamps-guard",
        ],
    )
    def test_ats_named_boundaries(self, frames, start, start_guard_sec, expected_eligible):
        dataset = self._selection_dataset(start_guard_sec=start_guard_sec)
        assert self._is_eligible(dataset, frames, start=start) is expected_eligible

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "splice_silence_sec,num_frames,num_speakers,activity_points,"
            "safe_end_index,unsafe_end_index,safe_start_index,unsafe_start_index"
        ),
        [
            (0.2, 8, 2, ((0, 0), (3, 1)), 3, 2, 1, 2),
            (0.1, 6, 2, ((0, 0), (2, 1)), 2, 1, 1, 2),
        ],
        ids=["two-frame-silence-gap", "one-frame-silence-gap"],
    )
    def test_exact_splice_silence_threshold_is_accepted(
        self,
        splice_silence_sec,
        num_frames,
        num_speakers,
        activity_points,
        safe_end_index,
        unsafe_end_index,
        safe_start_index,
        unsafe_start_index,
    ):
        dataset = self._selection_dataset(splice_silence_sec=splice_silence_sec)
        target = torch.zeros(num_frames, num_speakers, dtype=torch.bool)
        for frame_index, speaker_index in activity_points:
            target[frame_index, speaker_index] = True
        info = dataset._prepare_subsegment_activity(target)

        safe_end, safe_start = dataset._splice_silence_masks(info)

        assert safe_end[safe_end_index]
        assert not safe_end[unsafe_end_index]
        assert safe_start[safe_start_index]
        assert not safe_start[unsafe_start_index]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "kwargs,error",
        [
            ({"subsegment_single_chunk_min_len_sec": 0.0}, ValueError),
            ({"subsegment_single_chunk_min_len_sec": True}, TypeError),
            ({"subsegment_two_chunk_min_len_sec": float("nan")}, ValueError),
            ({"subsegment_two_chunk_min_len_sec": "10"}, TypeError),
            ({"subsegment_two_chunks_rate": -0.1}, ValueError),
            ({"subsegment_two_chunks_rate": True}, TypeError),
            ({"subsegment_nspk_bias": 0.9}, ValueError),
            ({"subsegment_nspk_bias": "2"}, TypeError),
            ({"subsegment_start_guard_sec": -0.1}, ValueError),
            ({"subsegment_min_first_spk_sec": 0.0}, ValueError),
            ({"subsegment_splice_silence_sec": 0.0}, ValueError),
            (
                {
                    "subsegment_mode": True,
                    "session_len_sec": 10.0,
                    "subsegment_single_chunk_min_len_sec": 11.0,
                },
                ValueError,
            ),
            (
                {
                    "subsegment_mode": True,
                    "session_len_sec": 10.0,
                    "subsegment_single_chunk_min_len_sec": 5.0,
                    "subsegment_two_chunk_min_len_sec": 6.0,
                    "subsegment_two_chunks_rate": 0.5,
                },
                ValueError,
            ),
        ],
    )
    def test_subsegment_configuration_validation(self, tmp_path, kwargs, error):
        with pytest.raises(error):
            self._configured_dataset(tmp_path, **kwargs)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "minimum_attribute,minimum_seconds,expected_frames",
        [
            ("subsegment_single_chunk_min_len_sec", 0.50, 4),
            ("subsegment_two_chunk_min_len_sec", 0.255, 2),
        ],
    )
    def test_chunk_minimum_frame_conversion_uses_ceiling(self, minimum_attribute, minimum_seconds, expected_frames):
        dataset = self._selection_dataset(feat_per_sec=7)
        setattr(dataset, minimum_attribute, minimum_seconds)
        assert dataset._seconds_to_feature_frames(getattr(dataset, minimum_attribute)) == expected_frames

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("num_frames,num_speakers,activity_segments,max_speakers,seed," "max_chunk_len,min_chunk_len"),
        [
            (
                30,
                3,
                ((0, 3, 0), (5, 8, 1), (10, 13, 2), (20, 23, 0)),
                2,
                0,
                15,
                5,
            )
        ],
        ids=["three-speaker-source-with-two-speaker-capacity"],
    )
    def test_single_chunk_rejects_windows_over_speaker_capacity(
        self,
        num_frames,
        num_speakers,
        activity_segments,
        max_speakers,
        seed,
        max_chunk_len,
        min_chunk_len,
    ):
        dataset = self._selection_dataset(min_first_speaker_sec=0.2, max_speakers=max_speakers)
        target = torch.zeros(num_frames, num_speakers, dtype=torch.bool)
        for start, end, speaker in activity_segments:
            target[start:end, speaker] = True
        info = dataset._prepare_subsegment_activity(target)

        with self._preserve_random_state():
            random.seed(seed)
            bounds = dataset._sample_single_chunk_bounds(info, max_len=max_chunk_len, min_len=min_chunk_len)

        assert bounds is not None
        selected = torch.cat([target[start:end] for start, end in bounds])
        assert selected.any(dim=0).sum() <= dataset.max_spks

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "frames,min_first_speaker_sec,max_chunk_len,min_chunk_len,expected_bounds",
        [(("A",) * 4, 0.3, 10, 6, [(0, 4)])],
        ids=["source-shorter-than-minimum"],
    )
    def test_short_clean_source_returns_full_available_length(
        self,
        frames,
        min_first_speaker_sec,
        max_chunk_len,
        min_chunk_len,
        expected_bounds,
    ):
        dataset = self._selection_dataset(min_first_speaker_sec=min_first_speaker_sec)
        info = dataset._prepare_subsegment_activity(self._target(frames))

        bounds = dataset._sample_single_chunk_bounds(info, max_len=max_chunk_len, min_len=min_chunk_len)

        assert bounds == expected_bounds

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "frames",
        [
            ("AB",) * 4,
            ("A", "A", "B", "B"),
        ],
        ids=["overlap-fails-ats", "speaker-union-exceeds-capacity"],
    )
    def test_short_source_still_obeys_ats_and_capacity(self, frames):
        dataset = self._selection_dataset(min_first_speaker_sec=0.2, max_speakers=1)
        info = dataset._prepare_subsegment_activity(self._target(frames))
        assert dataset._sample_single_chunk_bounds(info, max_len=10, min_len=6) is None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "num_frames,num_speakers,activity_segments,seed,total_len,"
            "min_chunk_len,min_first_speaker_sec,splice_silence_sec"
        ),
        [
            (
                80,
                2,
                ((0, 5, 0), (15, 20, 1), (30, 35, 0), (45, 50, 1), (60, 65, 0)),
                0,
                24,
                8,
                0.3,
                0.2,
            )
        ],
        ids=["alternating-speakers-with-safe-silence-gaps"],
    )
    def test_two_chunk_selection_obeys_start_and_silence_rules(
        self,
        num_frames,
        num_speakers,
        activity_segments,
        seed,
        total_len,
        min_chunk_len,
        min_first_speaker_sec,
        splice_silence_sec,
    ):
        dataset = self._selection_dataset(
            min_first_speaker_sec=min_first_speaker_sec,
            splice_silence_sec=splice_silence_sec,
        )
        target = torch.zeros(num_frames, num_speakers, dtype=torch.bool)
        for start, end, speaker in activity_segments:
            target[start:end, speaker] = 1
        info = dataset._prepare_subsegment_activity(target)

        with self._preserve_random_state():
            random.seed(seed)
            torch.manual_seed(seed)
            bounds = dataset._sample_two_chunk_bounds(info, total_len=total_len, min_chunk_len=min_chunk_len)

        assert bounds is not None
        (start1, end1), (start2, end2) = bounds
        starts = torch.tensor([start1, start2])
        ends = torch.tensor([end1, end2])
        safe_end, safe_start = dataset._splice_silence_masks(info)
        assert dataset._source_chunk_eligibility(info, starts, ends).all()
        assert safe_end[end1] and safe_start[start2]
        assert not target[start2].any()
        assert end1 - start1 >= min_chunk_len and end2 - start2 >= min_chunk_len
        assert (end1 - start1) + (end2 - start2) == total_len
        assert end2 <= start1 or start2 >= end1

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("num_frames,num_speakers,activity_segments,max_speakers,seed," "total_len,min_chunk_len"),
        [
            (
                100,
                3,
                ((0, 5, 0), (20, 25, 1), (40, 45, 2), (60, 65, 0), (80, 85, 1)),
                2,
                0,
                30,
                10,
            )
        ],
        ids=["three-speaker-source-with-two-speaker-union-capacity"],
    )
    def test_two_chunk_selection_rejects_unions_over_speaker_capacity(
        self,
        num_frames,
        num_speakers,
        activity_segments,
        max_speakers,
        seed,
        total_len,
        min_chunk_len,
    ):
        dataset = self._selection_dataset(
            min_first_speaker_sec=0.3,
            splice_silence_sec=0.2,
            max_speakers=max_speakers,
        )
        target = torch.zeros(num_frames, num_speakers, dtype=torch.bool)
        for start, end, speaker in activity_segments:
            target[start:end, speaker] = 1
        info = dataset._prepare_subsegment_activity(target)

        with self._preserve_random_state():
            random.seed(seed)
            torch.manual_seed(seed)
            bounds = dataset._sample_two_chunk_bounds(info, total_len=total_len, min_chunk_len=min_chunk_len)

        assert bounds is not None
        selected = torch.cat([target[start:end] for start, end in bounds])
        assert selected.any(dim=0).sum() <= dataset.max_spks

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("num_frames,num_speakers,selected_first_length,selected_start," "total_len,min_chunk_len,expected_bounds"),
        [(9, 2, 5, 0, 10, 3, [(0, 5), (5, 9)])],
        ids=["nine-frame-source-for-ten-frame-request"],
    )
    def test_two_chunk_shortened_fallback_works_when_source_is_shorter(
        self,
        monkeypatch,
        num_frames,
        num_speakers,
        selected_first_length,
        selected_start,
        total_len,
        min_chunk_len,
        expected_bounds,
    ):
        dataset = self._selection_dataset()
        target = torch.zeros(num_frames, num_speakers, dtype=torch.bool)
        info = dataset._prepare_subsegment_activity(target)
        monkeypatch.setattr(random, "randint", lambda low, high: selected_first_length)
        monkeypatch.setattr(random, "randrange", lambda size: selected_start)

        bounds = dataset._sample_two_chunk_bounds(info, total_len=total_len, min_chunk_len=min_chunk_len)

        assert bounds == expected_bounds
        assert sum(end - start for start, end in bounds) == num_frames
        selected = torch.cat([target[start:end] for start, end in bounds])
        assert selected.shape[0] == num_frames

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "audio_file,source_offset,bounds,expected_calls,expected_chunk_lengths",
        [
            (
                "audio.wav",
                5.0,
                [(10, 20), (40, 55)],
                [("audio.wav", 6.0, 1.0), ("audio.wav", 9.0, 1.5)],
                (100, 150),
            ),
            (
                ["left.wav", "right.wav"],
                0.25,
                [(0, 5)],
                [(["left.wav", "right.wav"], 0.25, 0.5)],
                (50,),
            ),
        ],
        ids=["two-noncontiguous-ranges", "single-multichannel-range"],
    )
    def test_load_audio_chunks_reads_only_selected_ranges(
        self,
        monkeypatch,
        audio_file,
        source_offset,
        bounds,
        expected_calls,
        expected_chunk_lengths,
    ):
        dataset = self._selection_dataset()

        class Featurizer:
            sample_rate = 100
            int_values = False

            @staticmethod
            def process_segment(segment):
                return torch.from_numpy(segment.samples)

        dataset.featurizer = Featurizer()
        calls = []

        def fake_from_file(audio_file, target_sr, int_values, offset, duration):
            calls.append((audio_file, offset, duration))
            value = float(len(calls))
            return AudioSegment(
                np.full(round(duration * target_sr), value, dtype=np.float32),
                target_sr,
            )

        monkeypatch.setattr(AudioSegment, "from_file", staticmethod(fake_from_file))

        audio = dataset._load_audio_chunks(
            audio_file,
            offset=source_offset,
            bounds=bounds,
        )

        assert calls == expected_calls
        assert audio.shape == (sum(expected_chunk_lengths),)
        start = 0
        for value, chunk_length in enumerate(expected_chunk_lengths, start=1):
            assert torch.all(audio[start : start + chunk_length] == value)
            start += chunk_length

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "audio_file,rttm_line,sample_duration,session_len_sec,"
            "single_chunk_min_len_sec,two_chunk_min_len_sec,selected_bounds,"
            "expected_audio_samples,expected_target_frames,expected_speaker_names"
        ),
        [
            (
                "audio.wav",
                "SPEAKER session 1 2.0 1.0 <NA> <NA> speaker_A <NA> <NA>\n",
                6.0,
                2.0,
                1.0,
                0.5,
                [(20, 40)],
                200,
                20,
                ["speaker_A", None, None, None],
            )
        ],
        ids=["select-before-loading-one-chunk"],
    )
    def test_create_subsegment_plans_before_loading_audio(
        self,
        tmp_path,
        audio_file,
        rttm_line,
        sample_duration,
        session_len_sec,
        single_chunk_min_len_sec,
        two_chunk_min_len_sec,
        selected_bounds,
        expected_audio_samples,
        expected_target_frames,
        expected_speaker_names,
    ):
        rttm_path = tmp_path / "audio.rttm"
        rttm_path.write_text(rttm_line, encoding="utf-8")
        sample = SimpleNamespace(
            audio_file=audio_file,
            rttm_file=str(rttm_path),
            duration=sample_duration,
        )
        dataset = self._selection_dataset()
        dataset.round_digits = 2
        dataset.session_len_sec = session_len_sec
        dataset.subsegment_single_chunk_min_len_sec = single_chunk_min_len_sec
        dataset.subsegment_two_chunk_min_len_sec = two_chunk_min_len_sec
        dataset.subsegment_two_chunks_rate = 0
        dataset.min_subsegment_duration = 0.03
        dataset.featurizer = SimpleNamespace(sample_rate=100)
        dataset._sample_two_chunk_bounds = lambda *args, **kwargs: pytest.fail(
            "Two-chunk selection must not run when its rate is zero"
        )

        def fake_single_chunk(activity_info, max_len, min_len):
            assert max_len == round(session_len_sec * dataset.feat_per_sec)
            assert min_len == round(single_chunk_min_len_sec * dataset.feat_per_sec)
            return selected_bounds

        dataset._sample_single_chunk_bounds = fake_single_chunk
        dataset.get_segment_timestamps = lambda duration, sample_rate: torch.tensor([expected_target_frames])
        dataset.get_frame_count_from_time_series_length = lambda length: expected_target_frames
        dataset.get_soft_targets_seg = lambda feat_level_target, target_len: feat_level_target
        load_calls = []

        def fake_load(audio_file, offset, bounds):
            load_calls.append((audio_file, offset, bounds))
            return torch.zeros(expected_audio_samples)

        dataset._load_audio_chunks = fake_load

        audio, audio_len, targets, target_len, speaker_names = dataset._create_subsegment(
            sample,
            offset=0,
        )

        assert load_calls == [(audio_file, 0, selected_bounds)]
        assert audio.shape == (expected_audio_samples,)
        assert audio_len == expected_audio_samples
        assert targets.shape == (expected_target_frames, dataset.max_spks)
        assert target_len == expected_target_frames
        assert speaker_names == expected_speaker_names

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "audio_file,rttm_line,sample_duration,expected_bounds,"
            "expected_audio_samples,expected_target_frames,expected_speaker_names"
        ),
        [
            (
                "short.wav",
                "SPEAKER session 1 0.0 0.4 <NA> <NA> speaker_A <NA> <NA>\n",
                0.4,
                [(0, 4)],
                40,
                4,
                ["speaker_A", None, None, None],
            )
        ],
        ids=["clean-source-shorter-than-minimum"],
    )
    def test_create_subsegment_loads_full_short_source(
        self,
        tmp_path,
        audio_file,
        rttm_line,
        sample_duration,
        expected_bounds,
        expected_audio_samples,
        expected_target_frames,
        expected_speaker_names,
    ):
        rttm_path = tmp_path / "short.rttm"
        rttm_path.write_text(rttm_line, encoding="utf-8")
        sample = SimpleNamespace(
            audio_file=audio_file,
            rttm_file=str(rttm_path),
            duration=sample_duration,
        )
        dataset = self._selection_dataset(min_first_speaker_sec=0.2)
        dataset.round_digits = 2
        dataset.session_len_sec = 2.0
        dataset.subsegment_single_chunk_min_len_sec = 1.0
        dataset.subsegment_two_chunk_min_len_sec = 0.3
        dataset.subsegment_two_chunks_rate = 0
        dataset.min_subsegment_duration = 0.03
        dataset.featurizer = SimpleNamespace(sample_rate=100)
        dataset.get_segment_timestamps = lambda duration, sample_rate: torch.tensor([expected_target_frames])
        dataset.get_frame_count_from_time_series_length = lambda length: expected_target_frames
        dataset.get_soft_targets_seg = lambda feat_level_target, target_len: feat_level_target
        load_calls = []

        def fake_load(audio_file, offset, bounds):
            load_calls.append((audio_file, offset, bounds))
            return torch.zeros(expected_audio_samples)

        dataset._load_audio_chunks = fake_load

        audio, audio_len, targets, target_len, speaker_names = dataset._create_subsegment(
            sample,
            offset=0,
        )

        assert load_calls == [(audio_file, 0, expected_bounds)]
        assert audio.shape == (expected_audio_samples,)
        assert audio_len == expected_audio_samples
        assert targets.shape == (expected_target_frames, dataset.max_spks)
        assert target_len == expected_target_frames
        assert speaker_names == expected_speaker_names

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "session_len_sec,sample_duration,sample_rate,max_speakers,"
            "observed_speakers,expected_samples,expected_width"
        ),
        [
            (0.5, 1.0, 100, 4, 2, 50, 4),
            (0.0, 0.75, 100, -1, 3, 75, 3),
            (0.091, 1.0, 100, 4, 2, 9, 4),
            (0.1, 1.0, 10, 4, 2, 1, 4),
        ],
        ids=[
            "configured-duration-fixed-width",
            "sample-duration-dynamic-width",
            "rounded-waveform-duration",
            "single-sample-waveform",
        ],
    )
    def test_fallback_noise_contract(
        self,
        session_len_sec,
        sample_duration,
        sample_rate,
        max_speakers,
        observed_speakers,
        expected_samples,
        expected_width,
    ):
        dataset = self._runtime_dataset(
            session_len_sec=session_len_sec,
            max_speakers=max_speakers,
            sample_rate=sample_rate,
        )

        audio, audio_len, targets, target_len, speaker_names = dataset._fallback_noise_subsegment(
            SimpleNamespace(duration=sample_duration),
            observed_speakers=observed_speakers,
        )

        waveform_duration = expected_samples / sample_rate
        expected_target_len = dataset.get_segment_timestamps(
            duration=waveform_duration,
            sample_rate=sample_rate,
        )
        expected_target_len = torch.clamp(
            expected_target_len,
            max=dataset.get_frame_count_from_time_series_length(expected_samples),
        )
        assert audio.shape == (expected_samples,)
        assert audio_len.dtype == torch.long and audio_len.item() == expected_samples
        assert audio.abs().count_nonzero() > 0
        assert audio.square().mean().sqrt().item() == pytest.approx(1e-3, rel=1e-5)
        assert torch.equal(target_len, expected_target_len)
        assert targets.shape == (target_len.item(), expected_width)
        assert targets.is_floating_point() and targets.count_nonzero() == 0
        assert speaker_names == [None] * expected_width

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "session_len_sec,sample_duration,sample_rate,error_match",
        [
            (0.0, 0.0, 100, "duration"),
            (0.0, -1.0, 100, "duration"),
            (0.0, float("nan"), 100, "duration"),
            (0.0, 0.001, 100, "sample count"),
            (0.0, 1.0, 0, "sample rate"),
            (0.0, 1.0, float("inf"), "sample rate"),
            (0.0, 1.0, 1, "feature frame rate"),
        ],
    )
    def test_fallback_noise_rejects_invalid_duration(self, session_len_sec, sample_duration, sample_rate, error_match):
        dataset = self._runtime_dataset(session_len_sec=session_len_sec, sample_rate=sample_rate)

        with pytest.raises(ValueError, match=error_match):
            dataset._fallback_noise_subsegment(SimpleNamespace(duration=sample_duration))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "segments,session_len_sec,max_speakers,expected_samples",
        [
            (((0.0, 1.0, "speaker_A"), (0.0, 1.0, "speaker_B")), 0.5, 4, 50),
            (((0.0, 0.5, "speaker_A"), (0.5, 0.5, "speaker_B")), 0.0, 1, 100),
        ],
        ids=["ats-rejection", "speaker-capacity-rejection"],
    )
    def test_rejected_subsegment_uses_noise_without_source_io(
        self, tmp_path, segments, session_len_sec, max_speakers, expected_samples
    ):
        rttm_path = tmp_path / "rejected.rttm"
        rttm_path.write_text(
            "\n".join(
                f"SPEAKER session 1 {start} {duration} <NA> <NA> {speaker} <NA> <NA>"
                for start, duration, speaker in segments
            )
            + "\n",
            encoding="utf-8",
        )
        sample = SimpleNamespace(
            audio_file="must-not-load.wav",
            rttm_file=str(rttm_path),
            duration=1.0,
        )
        dataset = self._runtime_dataset(session_len_sec=session_len_sec, max_speakers=max_speakers)
        dataset._load_audio_chunks = lambda *args, **kwargs: pytest.fail("Rejected candidates must not load audio")

        audio, audio_len, targets, target_len, speaker_names = dataset._create_subsegment(
            sample,
            offset=0,
        )

        assert audio.shape == (expected_samples,)
        assert audio_len.item() == expected_samples
        assert targets.shape[0] == target_len.item()
        assert targets.count_nonzero() == 0
        assert speaker_names == [None] * targets.shape[1]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "loaded_samples",
        [0, 2],
        ids=["empty-load", "sub-minimum-load"],
    )
    def test_too_short_loaded_audio_uses_fallback_noise(self, loaded_samples):
        dataset = self._runtime_dataset(session_len_sec=0.5, max_speakers=2)
        dataset._load_audio_chunks = lambda *args, **kwargs: torch.ones(loaded_samples)
        sample = SimpleNamespace(audio_file="unused.wav", rttm_file=None, duration=1.0)

        audio, audio_len, targets, target_len, speaker_names = dataset._create_subsegment(sample, offset=0.0)

        assert audio.shape == (50,)
        assert audio_len.item() == 50
        assert audio.square().mean().sqrt().item() == pytest.approx(1e-3, rel=1e-5)
        assert targets.shape == (target_len.item(), 2)
        assert targets.count_nonzero() == 0
        assert speaker_names == [None, None]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "max_speakers,observed_speakers,valid_width,valid_audio_shape,reverse_batch,expected_audio_shape",
        [
            (4, 2, 4, (4,), False, (2, 50)),
            (-1, 2, 3, (4, 2), True, (2, 50, 2)),
        ],
        ids=["fixed-speaker-width", "dynamic-speaker-width-mixed-channels"],
    )
    def test_collate_keeps_valid_and_fallback_cardinality(
        self,
        max_speakers,
        observed_speakers,
        valid_width,
        valid_audio_shape,
        reverse_batch,
        expected_audio_shape,
    ):
        dataset = self._runtime_dataset(session_len_sec=0.5, max_speakers=max_speakers)
        fallback = dataset._fallback_noise_subsegment(
            SimpleNamespace(duration=1.0),
            observed_speakers=observed_speakers,
        )
        valid_names = [f"speaker_{index}" for index in range(valid_width)]
        valid = (
            torch.ones(valid_audio_shape),
            torch.tensor(4),
            torch.ones(3, valid_width),
            torch.tensor([3]),
            valid_names,
        )
        batch = [valid, fallback] if reverse_batch else [fallback, valid]

        audio, audio_len, targets, target_len, speaker_names = dataset.eesd_train_collate_fn(batch)

        fallback_index = 1 if reverse_batch else 0
        assert len(batch) == audio.shape[0] == audio_len.shape[0] == targets.shape[0] == target_len.shape[0] == 2
        assert audio.shape == expected_audio_shape
        assert audio_len[fallback_index].item() == 50
        assert targets[fallback_index].count_nonzero() == 0
        assert speaker_names[fallback_index] == [None] * targets.shape[-1]

    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling_factor", [1, 4])
    def test_e2e_speaker_diar_dataset(self, test_data_dir, subsampling_factor):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/diarizer/lsm_val.json'))

        batch_size = 4
        num_samples = 8
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_dict_list = []
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
            with open(manifest_path, 'r', encoding='utf-8') as mfile:
                for ix, line in enumerate(mfile):
                    if ix >= num_samples:
                        break

                    line = line.replace("tests/data/", test_data_dir + "/").replace("\n", "")
                    f.write(f"{line}\n")
                    data_dict = json.loads(line)
                    data_dict_list.append(data_dict)

            f.seek(0)
            featurizer = WaveformFeaturizer(sample_rate=16000, int_values=False, augmentor=None)
            fb_featurizer = FilterbankFeatures(
                sample_rate=featurizer.sample_rate,
                n_window_size=int(0.025 * featurizer.sample_rate),
                n_window_stride=int(0.01 * featurizer.sample_rate),
                dither=False,
            )

            dataset = AudioToSpeechE2ESpkDiarDataset(
                manifest_filepath=f.name,
                soft_label_thres=0.5,
                session_len_sec=90,
                num_spks=4,
                featurizer=featurizer,
                window_stride=0.01,
                global_rank=0,
                soft_targets=False,
                subsampling_factor=subsampling_factor,
                device=device,
                fb_featurizer=fb_featurizer,
            )
            assert dataset.subsampling_factor == subsampling_factor
            dataloader_instance = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=dataset.eesd_train_collate_fn,
                drop_last=False,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            assert len(dataloader_instance) == (num_samples / batch_size)  # Check if the number of batches is correct
            batch_counts = len(dataloader_instance)

            deviation_thres_rate = 0.01  # 1% deviation allowed
            for batch_index, batch in enumerate(dataloader_instance):
                audio_signals, audio_signal_len, targets, target_lens, speaker_names = batch
                if batch_index != batch_counts - 1:
                    assert audio_signals.shape[0] == batch_size, "Batch size does not match the expected value"
                assert len(speaker_names) == audio_signals.shape[0]
                assert all(len(sample_names) == targets.shape[-1] for sample_names in speaker_names)
                for sample_index in range(audio_signals.shape[0]):
                    dataloader_audio_in_sec = audio_signal_len[sample_index].item()
                    data_dur_in_sec = abs(
                        data_dict_list[batch_size * batch_index + sample_index]['duration'] * featurizer.sample_rate
                        - dataloader_audio_in_sec
                    )
                    assert (
                        data_dur_in_sec <= deviation_thres_rate * dataloader_audio_in_sec
                    ), "Duration deviation exceeds 1%"
                assert not torch.isnan(audio_signals).any(), "audio_signals tensor contains NaN values"
                assert not torch.isnan(audio_signal_len).any(), "audio_signal_len tensor contains NaN values"
                assert not torch.isnan(targets).any(), "targets tensor contains NaN values"
                assert not torch.isnan(target_lens).any(), "target_lens tensor contains NaN values"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "audio_names,repeat_count,validate_paths,create_files,expected_checked_names",
        [
            ("audio.wav", 3, True, True, ["audio.wav", "audio.rttm"]),
            (("left.wav", "right.wav"), 1, True, True, ["left.wav", "right.wav", "audio.rttm"]),
            ("missing.wav", 1, False, False, []),
        ],
        ids=["cached-repeated-paths", "multichannel-paths", "validation-disabled"],
    )
    def test_manifest_path_validation(
        self,
        tmp_path,
        monkeypatch,
        audio_names,
        repeat_count,
        validate_paths,
        create_files,
        expected_checked_names,
    ):
        audio_names = (audio_names,) if isinstance(audio_names, str) else audio_names
        audio_paths = [tmp_path / name for name in audio_names]
        rttm_path = tmp_path / "audio.rttm"
        manifest_path = tmp_path / "manifest.json"
        if create_files:
            for path in (*audio_paths, rttm_path):
                path.touch()
        audio_value = str(audio_paths[0]) if len(audio_paths) == 1 else [str(path) for path in audio_paths]
        entry = {
            "audio_filepath": audio_value,
            "duration": 1.0,
            "rttm_filepath": str(rttm_path),
            "uniq_id": "session",
        }
        manifest_path.write_text(
            "\n".join(json.dumps(entry) for _ in range(repeat_count)),
            encoding="utf-8",
        )
        checked_paths = []
        original_exists = os.path.exists
        tracked_paths = {str(path) for path in (*audio_paths, rttm_path)}

        def counting_exists(path):
            if path in tracked_paths:
                checked_paths.append(path)
            return original_exists(path)

        monkeypatch.setattr(os.path, "exists", counting_exists)

        collection = EndtoEndDiarizationSpeechLabel(
            manifests_files=str(manifest_path),
            validate_manifest_paths=validate_paths,
        )

        assert len(collection) == repeat_count
        assert checked_paths == [str(tmp_path / name) for name in expected_checked_names]
