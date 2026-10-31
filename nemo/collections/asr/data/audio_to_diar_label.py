# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import math
import os
import random
from numbers import Real
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.speaker_utils import convert_rttm_line, get_subsegments
from nemo.collections.common.parts.preprocessing.collections import EndtoEndDiarizationSpeechLabel
from nemo.core.classes import Dataset
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType, ProbsType, StringType
from nemo.utils import logging


_SpeakerNames = List[Optional[str]]
_EESDSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, _SpeakerNames]
_EESDBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[_SpeakerNames]]


def get_subsegments_to_timestamps(
    subsegments: List[Tuple[float, float]], feat_per_sec: int = 100, max_end_ts: float = None, decimals=2
):
    """
    Convert subsegment timestamps to scale timestamps by multiplying with the feature rate (`feat_per_sec`)
    and rounding. Segment is consisted of many subsegments and sugsegments are equivalent to `frames`
    in end-to-end speaker diarization models.

    Args:
        subsegments (List[Tuple[float, float]]):
            A list of tuples where each tuple contains the start and end times of a subsegment
            (frames in end-to-end models).
            >>> subsegments = [[t0_start, t0_duration], [t1_start, t1_duration],..., [tN_start, tN_duration]]
        feat_per_sec (int, optional):
            The number of feature frames per second. Defaults to 100.
        max_end_ts (float, optional):
            The maximum end timestamp to clip the results. If None, no clipping is applied. Defaults to None.
        decimals (int, optional):
            The number of decimal places to round the timestamps. Defaults to 2.

    Example:
        Segments starting from 0.0 and ending at 69.2 seconds.
        If hop-length is 0.08 and the subsegment (frame) length is 0.16 seconds,
        there are 864 = (69.2 - 0.16)/0.08 + 1 subsegments (frames in end-to-end models) in this segment.
        >>> subsegments = [[[0.0, 0.16], [0.08, 0.16], ..., [69.04, 0.16], [69.12, 0.08]]

    Returns:
        ts (torch.tensor):
            A tensor containing the scaled and rounded timestamps for each subsegment.
    """
    if len(subsegments) == 0:
        # Each row stores the start and end frame indices.
        return torch.zeros((0, 2), dtype=torch.long)
    seg_ts = (torch.tensor(subsegments) * feat_per_sec).float()
    ts_round = torch.round(seg_ts, decimals=decimals)
    ts = ts_round.long()
    ts[:, 1] = ts[:, 0] + ts[:, 1]
    if max_end_ts is not None:
        ts = np.clip(ts, 0, int(max_end_ts * feat_per_sec))
    return ts


def extract_frame_info_from_rttm(offset, duration, rttm_lines, round_digits=3):
    """
    Extracts RTTM lines containing speaker labels, start time, and end time for a given audio segment.

    Args:
        uniq_id (str): Unique identifier for the audio file and corresponding RTTM file.
        offset (float): The starting time offset for the segment of interest.
        duration (float): The duration of the segment of interest.
        rttm_lines (list): List of RTTM lines in string format.
        round_digits (int, optional): Number of decimal places to round the start and end times. Defaults to 3.

    Returns:
        rttm_mat (tuple): A tuple containing lists of start times, end times, and speaker labels.
        sess_to_global_spkids (dict): A mapping from session-specific speaker indices to global speaker identifiers.
    """
    rttm_stt, rttm_end = offset, offset + duration
    stt_list, end_list, speaker_list, speaker_set = [], [], [], []
    sess_to_global_spkids = dict()

    for rttm_line in rttm_lines:
        start, end, speaker = convert_rttm_line(rttm_line)

        # Skip invalid RTTM lines where the start time is greater than the end time.
        if start > end:
            continue

        # Check if the RTTM segment overlaps with the specified segment of interest.
        if (end > rttm_stt and start < rttm_end) or (start < rttm_end and end > rttm_stt):
            # Adjust the start and end times to fit within the segment of interest.
            start, end = max(start, rttm_stt), min(end, rttm_end)
        else:
            continue

        # Round the start and end times to the specified number of decimal places.
        end_list.append(round(end, round_digits))
        stt_list.append(round(start, round_digits))

        # Assign a unique index to each speaker and maintain a mapping.
        if speaker not in speaker_set:
            speaker_set.append(speaker)
        speaker_list.append(speaker_set.index(speaker))
        sess_to_global_spkids.update({speaker_set.index(speaker): speaker})

    rttm_mat = (stt_list, end_list, speaker_list)
    return rttm_mat, sess_to_global_spkids


def get_frame_targets_from_rttm(
    rttm_timestamps: list,
    offset: float,
    duration: float,
    round_digits: int,
    feat_per_sec: int,
    max_spks: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a multi-dimensional vector sequence containing speaker timestamp information in RTTM.
    The unit-length is the frame shift length of the acoustic feature. The feature-level annotations
    `feat_level_target` will later be converted to base-segment level diarization label.

    Args:
        rttm_timestamps (list): Lists of start times, end times, and speaker indices for the RTTM segments.
        offset (float): Start time of the selected audio region in seconds.
        duration (float): Duration of the selected audio region in seconds.
        round_digits (int): Decimal precision associated with the prepared RTTM timestamps.
        feat_per_sec (int): Number of feature frames per second, as determined by the preprocessing window stride.
        max_spks (int): Maximum number of target speakers. Use ``-1`` to preserve all speakers.
        dtype (torch.dtype): Output target dtype.

    Returns:
        feat_level_target (torch.Tensor): Speaker labels for each feature-level frame.
    """
    stt_list, end_list, speaker_list = rttm_timestamps
    sorted_speakers = sorted(list(set(speaker_list)))
    total_fr_len = int(duration * feat_per_sec)
    if max_spks == -1:
        num_target_speakers = max(1, len(sorted_speakers))
    else:
        num_target_speakers = max_spks
    if max_spks != -1 and len(sorted_speakers) > max_spks:
        logging.warning(
            f"Number of speakers in RTTM file {len(sorted_speakers)} exceeds the maximum number of speakers: "
            f"{max_spks}! Only {max_spks} first speakers remain, and this will affect frame metrics!"
        )
    feat_level_target = torch.zeros(total_fr_len, num_target_speakers, dtype=dtype)
    for count, (stt, end, spk_rttm_key) in enumerate(zip(stt_list, end_list, speaker_list)):
        if end < offset or stt > offset + duration:
            continue
        stt, end = max(offset, stt), min(offset + duration, end)
        spk = spk_rttm_key
        if spk < num_target_speakers:
            stt_fr, end_fr = int((stt - offset) * feat_per_sec), int((end - offset) * feat_per_sec)
            feat_level_target[stt_fr:end_fr, spk] = 1
    return feat_level_target


class _SubsegmentActivity(NamedTuple):
    """Compact, reusable frame-activity data used while planning source chunks.

    Attributes:
        activity: Boolean speaker activity with shape ``(T, S)``.
        prefix: Per-speaker activity prefix counts with shape ``(T + 1, S)``;
            ``prefix[t, s]`` counts active frames for speaker ``s`` before ``t``.
        total_prefix: Total active speaker-frame prefix counts with shape
            ``(T + 1,)``. Overlap contributes one count per active speaker.
        active_count: Number of active speakers at each frame, shape ``(T,)``.
        next_active: First active frame at or after each frame, or ``T`` when
            no future speech exists, shape ``(T,)``.
        first_speaker: First active speaker index at each frame. It is meaningful
            only when ``active_count == 1``, shape ``(T,)``.
        next_competitor: First frame at or after each frame where someone other
            than ``first_speaker`` is active, or ``T`` if none exists, shape ``(T,)``.
    """

    activity: torch.Tensor
    prefix: torch.Tensor
    total_prefix: torch.Tensor
    active_count: torch.Tensor
    next_active: torch.Tensor
    first_speaker: torch.Tensor
    next_competitor: torch.Tensor


class AudioToSpeechE2ESpkDiarDataset(Dataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    RTTM files and number of speakers. This Dataset class is designed for
    training or fine-tuning speaker embedding extractor and diarization decoder
    at the same time.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath: Comma-separated path or paths to input JSON manifests.
        soft_label_thres: Speaker-activity threshold used both to convert soft
            feature targets to Boolean planning activity and, when
            ``soft_targets`` is false, to produce hard diarization targets.
        session_len_sec: Maximum selected duration in seconds. In subsegment
            mode, a positive value is the combined one- or two-chunk budget;
            a non-positive value selects the complete annotated region.
        num_spks: Maximum target speaker width. ``-1`` disables the speaker
            limit and lets each sample retain all observed speakers; collation
            then pads samples to the largest width in the batch.
        featurizer: Waveform featurizer used to load and augment audio.
        fb_featurizer: Filterbank featurizer that defines STFT frame geometry.
        window_stride: Acoustic feature stride in seconds.
        global_rank: Distributed-process rank retained by the dataset.
        soft_targets: If true, return averaged speaker-activity probabilities;
            otherwise threshold them with ``soft_label_thres``.
        subsampling_factor: Number of acoustic frames represented by one
            diarization target step.
        device: Device identifier retained by the dataset.
        subsegment_mode: If true, plan ATS-safe source chunks from RTTM activity
            before loading audio; otherwise load the ordinary contiguous region.
        subsegment_single_chunk_min_len_sec: Minimum one-chunk candidate length
            in seconds. A shorter non-empty source is still considered whole.
        subsegment_two_chunk_min_len_sec: Minimum length in seconds of each
            chunk in a two-chunk plan.
        subsegment_two_chunks_rate: Probability of attempting two-chunk
            sampling before falling back to one chunk.
        subsegment_nspk_bias: Multiplicative sampling weight per active speaker.
            ``1.0`` samples uniformly; larger values favor more-speaker plans.
        subsegment_start_guard_sec: Look-back interval that must contain no
            competing speaker activity at a candidate source-chunk start.
        subsegment_min_first_spk_sec: Minimum initial activity required from the
            first speaker before a competing speaker can enter.
        subsegment_splice_silence_sec: Required silent suffix of chunk one and
            silent prefix of chunk two at a splice.
        validate_manifest_paths: If true, validate unique audio and RTTM paths
            while loading the manifest.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        output_types = {
            "audio_signal": NeuralType(('B', 'T'), AudioSignal()),
            "audio_length": NeuralType(('B'), LengthsType()),
            "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
            "target_len": NeuralType(('B'), LengthsType()),
            "speaker_names": NeuralType(('B', 'C'), StringType()),
        }

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        soft_label_thres: float,
        session_len_sec: float,
        num_spks: int,
        featurizer: Any,
        fb_featurizer: Any,
        window_stride: float,
        global_rank: int,
        soft_targets: bool,
        device: Union[str, torch.device],
        subsampling_factor: int = 8,
        subsegment_mode: bool = False,
        subsegment_single_chunk_min_len_sec: float = 15.0,
        subsegment_two_chunk_min_len_sec: float = 10.0,
        subsegment_two_chunks_rate: float = 0.0,
        subsegment_nspk_bias: float = 1.0,
        subsegment_start_guard_sec: float = 0.25,
        subsegment_min_first_spk_sec: float = 0.50,
        subsegment_splice_silence_sec: float = 0.10,
        validate_manifest_paths: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(subsegment_mode, bool):
            raise TypeError(f"subsegment_mode must be bool, got {subsegment_mode!r}")
        if not isinstance(validate_manifest_paths, bool):
            raise TypeError(f"validate_manifest_paths must be bool, got {validate_manifest_paths!r}")
        self.round_digits = 2
        self.min_subsegment_duration = 0.03
        self.collection = EndtoEndDiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            round_digits=self.round_digits,
            validate_manifest_paths=validate_manifest_paths,
        )
        self.featurizer = featurizer
        self.fb_featurizer = fb_featurizer
        # STFT and subsampling factor parameters
        self.n_fft = self.fb_featurizer.n_fft
        self.hop_length = self.fb_featurizer.hop_length
        self.stft_pad_amount = self.fb_featurizer.stft_pad_amount
        self.subsampling_factor = subsampling_factor
        # Annotation and target length parameters
        self.feat_per_sec = int(1 / window_stride)
        self.diar_frame_length = round(subsampling_factor * window_stride, self.round_digits)
        self.session_len_sec = session_len_sec
        self.soft_label_thres = soft_label_thres
        self.max_spks = num_spks
        self.use_asr_style_frame_count = True
        self.soft_targets = soft_targets
        self.floor_decimal = 10**self.round_digits
        self.device = device
        self.global_rank = global_rank
        self.subsegment_mode = subsegment_mode

        if isinstance(subsegment_two_chunks_rate, bool) or not isinstance(subsegment_two_chunks_rate, Real):
            raise TypeError(f"subsegment_two_chunks_rate must be numeric, got {subsegment_two_chunks_rate!r}")
        if not math.isfinite(float(subsegment_two_chunks_rate)) or not 0.0 <= subsegment_two_chunks_rate <= 1.0:
            raise ValueError(
                f"subsegment_two_chunks_rate must be finite and between 0 and 1, got {subsegment_two_chunks_rate!r}"
            )
        self.subsegment_two_chunks_rate = float(subsegment_two_chunks_rate)
        for name, value in (
            ("subsegment_single_chunk_min_len_sec", subsegment_single_chunk_min_len_sec),
            ("subsegment_two_chunk_min_len_sec", subsegment_two_chunk_min_len_sec),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be numeric, got {value!r}")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        self.subsegment_single_chunk_min_len_sec = float(subsegment_single_chunk_min_len_sec)
        self.subsegment_two_chunk_min_len_sec = float(subsegment_two_chunk_min_len_sec)

        if isinstance(subsegment_nspk_bias, bool) or not isinstance(subsegment_nspk_bias, Real):
            raise TypeError(f"subsegment_nspk_bias must be numeric, got {subsegment_nspk_bias!r}")
        if not math.isfinite(float(subsegment_nspk_bias)) or subsegment_nspk_bias < 1.0:
            raise ValueError(f"subsegment_nspk_bias must be finite and >= 1.0, got {subsegment_nspk_bias!r}")
        self.subsegment_nspk_bias = float(subsegment_nspk_bias)

        for name, value, allow_zero in (
            ("subsegment_start_guard_sec", subsegment_start_guard_sec, True),
            ("subsegment_min_first_spk_sec", subsegment_min_first_spk_sec, False),
            ("subsegment_splice_silence_sec", subsegment_splice_silence_sec, False),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be numeric, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if value < 0 or (not allow_zero and value == 0):
                comparison = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be {comparison}, got {value!r}")
        self.subsegment_start_guard_sec = float(subsegment_start_guard_sec)
        self.subsegment_min_first_spk_sec = float(subsegment_min_first_spk_sec)
        self.subsegment_splice_silence_sec = float(subsegment_splice_silence_sec)

        if self.subsegment_mode and self.session_len_sec > 0:
            session_len_frames = int(self.session_len_sec * self.feat_per_sec)
            single_chunk_min_frames = self._seconds_to_feature_frames(self.subsegment_single_chunk_min_len_sec)
            two_chunk_min_frames = self._seconds_to_feature_frames(self.subsegment_two_chunk_min_len_sec)
            if single_chunk_min_frames > session_len_frames:
                raise ValueError(
                    "subsegment_single_chunk_min_len_sec "
                    f"({self.subsegment_single_chunk_min_len_sec}s -> {single_chunk_min_frames} frames) "
                    f"cannot exceed session_len_sec "
                    f"({self.session_len_sec}s -> {session_len_frames} frames)"
                )
            if self.subsegment_two_chunks_rate > 0 and 2 * two_chunk_min_frames > session_len_frames:
                raise ValueError(
                    "twice subsegment_two_chunk_min_len_sec "
                    f"({self.subsegment_two_chunk_min_len_sec}s -> {two_chunk_min_frames} frames each) "
                    f"cannot exceed session_len_sec "
                    f"({self.session_len_sec}s -> {session_len_frames} frames) when two-chunk sampling is enabled"
                )

    def __len__(self):
        return len(self.collection)

    def get_frame_count_from_time_series_length(self, seq_len):
        """
        This function is used to get the sequence length of the audio signal. This is required to match
        the feature frame length with ASR (STT) models. This function is copied from
        NeMo/nemo/collections/asr/parts/preprocessing/features.py::FilterbankFeatures::get_seq_len.

        Args:
            seq_len (int):
                The sequence length of the time-series data.

        Returns:
            seq_len (int):
                The sequence length of the feature frames.
        """
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length).to(dtype=torch.long)
        frame_count = int(np.ceil(seq_len / self.subsampling_factor))
        return frame_count

    def get_uniq_id_with_range(self, sample, deci=3):
        """
        Generate unique training sample ID from unique file ID, offset and duration. The start-end time added
        unique ID is required for identifying the sample since multiple short audio samples are generated from a single
        audio file. The start time and end time of the audio stream uses millisecond units if `deci=3`.

        Args:
            sample:
                `EndtoEndDiarizationSpeechLabel` instance from collections.

        Returns:
            uniq_id (str):
                Unique sample ID which includes start and end time of the audio stream.
                Example: abc1001_3122_6458
        """
        bare_uniq_id = os.path.splitext(os.path.basename(sample.rttm_file))[0]
        offset = str(int(round(sample.offset, deci) * pow(10, deci)))
        endtime = str(int(round(sample.offset + sample.duration, deci) * pow(10, deci)))
        uniq_id = f"{bare_uniq_id}_{offset}_{endtime}"
        return uniq_id

    def _target_speaker_width(self, observed_speakers: int = 0) -> int:
        """Resolve the target width for fixed and unlimited speaker modes.

        Args:
            observed_speakers: Number of speakers represented by the current
                sample or selected column set.

        Returns:
            ``max_spks`` when it is configured, otherwise at least one column
            and enough columns to retain every observed speaker.
        """
        if self.max_spks == -1:
            return max(1, observed_speakers)
        return self.max_spks

    def _build_speaker_names(
        self,
        sess_to_global_spkids: Dict[int, str],
        columns: Optional[Sequence[int]] = None,
    ) -> _SpeakerNames:
        """Build RTTM speaker names aligned with target-matrix columns.

        Args:
            sess_to_global_spkids: Mapping from source target-column indices to
                RTTM speaker names, as returned by
                ``extract_frame_info_from_rttm``.
            columns: Optional ordered subset of source columns retained after
                subsegment speaker selection. If omitted, source column
                positions are preserved.

        Returns:
            Speaker name for each output target column, with ``None`` in unused
            columns. Fixed-width mode returns ``max_spks`` entries; unlimited
            mode returns a non-empty width derived from ``columns`` or the
            largest observed source column.
        """
        if columns is None:
            observed_speakers = max(sess_to_global_spkids, default=-1) + 1
            speaker_names = [None] * self._target_speaker_width(observed_speakers)
            for column, speaker_name in sess_to_global_spkids.items():
                if column < len(speaker_names):
                    speaker_names[column] = speaker_name
        else:
            speaker_names = [None] * self._target_speaker_width(len(columns))
            for new_column, old_column in enumerate(columns):
                if new_column < len(speaker_names):
                    speaker_names[new_column] = sess_to_global_spkids.get(old_column)
        return speaker_names

    def parse_rttm_for_targets_and_lens(
        self,
        rttm_file: Optional[str],
        offset: float,
        duration: float,
        target_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, _SpeakerNames]:
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function converts (start, end, speaker_id) format into base-scale (the finest scale) segment level
        diarization label in a matrix form.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]

        Args:
            rttm_file: RTTM annotation path, or ``None``/an empty string for an
                unannotated region.
            offset: Start of the selected manifest region in seconds.
            duration: Selected region duration in seconds.
            target_len: One-element tensor containing the diarization-step
                length expected for the waveform.

        Returns:
            A pair containing diarization targets of shape ``(T, S)`` and RTTM
            speaker names aligned with their columns. ``S`` is ``max_spks`` in
            fixed-width mode and the observed non-empty width when
            ``max_spks == -1``. Missing RTTM input produces all-zero targets
            and all-``None`` metadata.
        """
        if rttm_file in [None, '']:
            num_seg = torch.max(target_len)
            num_target_speakers = self._target_speaker_width()
            targets = torch.zeros(num_seg, num_target_speakers)
            return targets, [None] * num_target_speakers

        with open(rttm_file, 'r') as f:
            rttm_lines = f.readlines()

        rttm_timestamps, sess_to_global_spkids = extract_frame_info_from_rttm(offset, duration, rttm_lines)

        fr_level_target = get_frame_targets_from_rttm(
            rttm_timestamps=rttm_timestamps,
            offset=offset,
            duration=duration,
            round_digits=self.round_digits,
            feat_per_sec=self.feat_per_sec,
            max_spks=self.max_spks,
        )

        step_target = self._get_segment_targets(feat_level_target=fr_level_target, target_len=target_len)
        speaker_names = self._build_speaker_names(sess_to_global_spkids)
        return step_target, speaker_names

    def _get_segment_targets(self, feat_level_target: torch.Tensor, target_len: torch.Tensor) -> torch.Tensor:
        """Aggregate feature-frame activity into configured diarization targets.

        Args:
            feat_level_target: Speaker activity of shape ``(T_feature, S)``.
            target_len: One-element tensor containing the requested number of
                diarization steps.

        Returns:
            Averaged soft activity when ``soft_targets`` is true. Otherwise,
            activity thresholded by ``soft_label_thres`` and converted to
            floating-point hard labels.
        """
        soft_target_seg = self.get_soft_targets_seg(feat_level_target=feat_level_target, target_len=target_len)
        if self.soft_targets:
            return soft_target_seg
        return (soft_target_seg >= self.soft_label_thres).float()

    def get_soft_targets_seg(self, feat_level_target, target_len):
        """
        Generate the final targets for the actual diarization step.
        Here, frame level means step level which is also referred to as segments.
        We follow the original paper and refer to the step level as "frames".

        Args:
            feat_level_target (torch.tensor):
                Tensor variable containing hard-labels of speaker activity in each feature-level segment.
            target_len (torch.tensor):
                Numbers of ms segments

        Returns:
            soft_target_seg (torch.tensor):
                Tensor variable containing soft-labels of speaker activity in each step-level segment.
        """
        num_seg = torch.max(target_len)
        stride = int(self.feat_per_sec * self.diar_frame_length)
        if stride <= 1:
            return feat_level_target[:num_seg, :].clone()

        targets = feat_level_target.new_zeros((num_seg, feat_level_target.shape[1]))
        for index in range(num_seg):
            if index == 0:
                seg_stt_feat = 0
            else:
                seg_stt_feat = stride * index - 1 - int(stride / 2)
            if index == num_seg - 1:
                seg_end_feat = feat_level_target.shape[0]
            else:
                seg_end_feat = stride * index - 1 + int(stride / 2)
            targets[index] = torch.mean(feat_level_target[seg_stt_feat : seg_end_feat + 1, :], axis=0)
        return targets

    def get_segment_timestamps(
        self,
        duration: float,
        offset: float = 0,
        sample_rate: int = 16000,
    ):
        """
        Get start and end time of segments in each scale.

        Args:
            sample:
                `EndtoEndDiarizationSpeechLabel` instance from preprocessing.collections
        Returns:
            segment_timestamps (torch.tensor):
                Tensor containing Multiscale segment timestamps.
            target_len (torch.tensor):
                Number of segments for each scale. This information is used for reshaping embedding batch
                during forward propagation.
        """
        stride = int(self.feat_per_sec * self.diar_frame_length)
        if stride <= 1:
            num_frames = int(np.ceil((1 + duration * sample_rate) / int(sample_rate / self.feat_per_sec)))
            return torch.tensor([num_frames])

        subsegments = get_subsegments(
            offset=offset,
            window=round(self.diar_frame_length * 2, self.round_digits),
            shift=self.diar_frame_length,
            duration=duration,
            min_subsegment_duration=self.min_subsegment_duration,
            use_asr_style_frame_count=self.use_asr_style_frame_count,
            sample_rate=sample_rate,
            feat_per_sec=self.feat_per_sec,
        )
        if self.use_asr_style_frame_count:
            effective_dur = (
                np.ceil((1 + duration * sample_rate) / int(sample_rate / self.feat_per_sec)).astype(int)
                / self.feat_per_sec
            )
        else:
            effective_dur = duration
        ts_tensor = get_subsegments_to_timestamps(
            subsegments, self.feat_per_sec, decimals=2, max_end_ts=(offset + effective_dur)
        )
        target_len = torch.tensor([ts_tensor.shape[0]])
        return target_len

    def _seconds_to_feature_frames(self, seconds: float) -> int:
        """Convert a duration to the smallest containing feature-frame count.

        Args:
            seconds: Non-negative duration in seconds.

        Returns:
            Feature-frame count computed with
            ``ceil(seconds * feat_per_sec)``.
        """
        return math.ceil(seconds * self.feat_per_sec)

    def _prepare_subsegment_activity(self, frame_level_target: torch.Tensor) -> _SubsegmentActivity:
        """Build compact activity metadata once for all chunk-planning checks.

        Boolean targets are retained directly. Other target dtypes are
        converted to activity with the strict comparison
        ``frame_level_target > soft_label_thres``.

        Args:
            frame_level_target: Boolean or soft target tensor of shape ``(T, S)``.

        Returns:
            Reusable activity, prefix-count, and next-event metadata.
        """
        activity = (
            frame_level_target
            if frame_level_target.dtype == torch.bool
            else frame_level_target > self.soft_label_thres
        )
        num_frames, num_speakers = activity.shape
        prefix = torch.zeros(num_frames + 1, num_speakers, dtype=torch.int32, device=activity.device)
        total_prefix = torch.zeros(num_frames + 1, dtype=torch.int32, device=activity.device)
        if num_frames == 0 or num_speakers == 0:
            empty_frames = torch.zeros(num_frames, dtype=torch.int32, device=activity.device)
            empty_indices = torch.zeros(num_frames, dtype=torch.long, device=activity.device)
            return _SubsegmentActivity(
                activity=activity,
                prefix=prefix,
                total_prefix=total_prefix,
                active_count=empty_frames,
                next_active=empty_indices,
                first_speaker=empty_indices,
                next_competitor=empty_indices,
            )

        activity_int = activity.to(torch.int32)
        prefix[1:] = activity_int.cumsum(dim=0, dtype=torch.int32)
        active_count = activity_int.sum(dim=1, dtype=torch.int32)
        total_prefix[1:] = active_count.cumsum(dim=0, dtype=torch.int32)
        frame_indices = torch.arange(num_frames, dtype=torch.int32, device=activity.device)
        next_active = self._next_true_indices(active_count > 0, frame_indices).long()
        first_speaker = activity_int.argmax(dim=1)

        other_speaker_activity = active_count.unsqueeze(1) > activity_int
        marked_other = torch.where(other_speaker_activity, frame_indices.unsqueeze(1), num_frames)
        next_other = torch.flip(
            torch.cummin(torch.flip(marked_other, dims=(0,)), dim=0).values,
            dims=(0,),
        )
        next_competitor = next_other.gather(dim=1, index=first_speaker.unsqueeze(1)).squeeze(1).long()

        return _SubsegmentActivity(
            activity=activity,
            prefix=prefix,
            total_prefix=total_prefix,
            active_count=active_count,
            next_active=next_active,
            first_speaker=first_speaker,
            next_competitor=next_competitor,
        )

    @staticmethod
    def _chunk_speaker_presence(
        activity_info: _SubsegmentActivity,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        """Return speaker presence for aligned source-chunk bounds.

        Args:
            activity_info: Reusable compact activity metadata.
            starts: Start indices of shape ``(N,)``.
            ends: Exclusive end indices of shape ``(N,)``.

        Returns:
            Boolean speaker-presence tensor of shape ``(N, S)``.
        """
        return activity_info.prefix[ends] - activity_info.prefix[starts] > 0

    @staticmethod
    def _next_true_indices(mask: torch.Tensor, frame_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Find the first true value at or after each position.

        Args:
            mask: One-dimensional Boolean tensor of length ``T``.
            frame_indices: Optional compact integer frame indices matching
                ``mask``.

        Returns:
            Integer tensor of length ``T``. Each value is the first true index
            at or after that position, or ``T`` when no such index exists.
        """
        length = mask.shape[0]
        if frame_indices is None:
            frame_indices = torch.arange(length, dtype=torch.int32, device=mask.device)
        marked = torch.where(mask, frame_indices, length)
        return torch.flip(torch.cummin(torch.flip(marked, dims=(0,)), dim=0).values, dims=(0,))

    def _source_chunk_eligibility(
        self,
        activity_info: _SubsegmentActivity,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the complete ATS-safe start policy for source chunks.

        A speech-bearing candidate must begin, after any leading silence, with
        exactly one speaker. The look-back guard may contain only that speaker
        for an active start, and must be fully silent for a silent start. The
        first speaker must then satisfy the configured minimum either in total
        before a competitor enters or as an uninterrupted initial run. An
        all-silence chunk is also eligible when its look-back guard is silent.

        Args:
            activity_info: Reusable compact activity metadata.
            starts: Candidate source-chunk starts of shape ``(N,)``.
            ends: Exclusive source-chunk ends of shape ``(N,)``.

        Returns:
            Boolean eligibility mask of shape ``(N,)``.
        """
        if starts.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=starts.device)

        starts = starts.long()
        ends = ends.long()
        num_frames, num_speakers = activity_info.activity.shape
        if num_speakers == 0:
            return torch.ones_like(starts, dtype=torch.bool)

        guard_frames = self._seconds_to_feature_frames(self.subsegment_start_guard_sec)
        minimum_frames = self._seconds_to_feature_frames(self.subsegment_min_first_spk_sec)
        candidate_silent = activity_info.active_count[starts] == 0
        effective_start = activity_info.next_active[starts]
        has_speech = effective_start < ends

        history_start = torch.clamp(starts - guard_frames, min=0)
        history_total = activity_info.total_prefix[starts] - activity_info.total_prefix[history_start]
        silence_history_safe = history_total == 0
        all_silence_eligible = candidate_silent & ~has_speech & silence_history_safe

        safe_effective_start = effective_start.clamp(max=max(num_frames - 1, 0))
        first_speaker = activity_info.first_speaker[safe_effective_start]
        exactly_one_speaker = activity_info.active_count[safe_effective_start] == 1
        first_speaker_history = (
            activity_info.prefix[starts, first_speaker] - activity_info.prefix[history_start, first_speaker]
        )
        active_history_safe = history_total - first_speaker_history == 0
        history_safe = torch.where(candidate_silent, silence_history_safe, active_history_safe)

        competitor = torch.minimum(activity_info.next_competitor[safe_effective_start], ends)
        accumulated_activity = (
            activity_info.prefix[competitor, first_speaker] - activity_info.prefix[safe_effective_start, first_speaker]
        )
        condition_a = accumulated_activity >= minimum_frames

        initial_run_end = safe_effective_start + minimum_frames
        run_within_chunk = initial_run_end <= ends
        safe_initial_run_end = initial_run_end.clamp(max=num_frames)
        initial_run_activity = (
            activity_info.prefix[safe_initial_run_end, first_speaker]
            - activity_info.prefix[safe_effective_start, first_speaker]
        )
        condition_b = run_within_chunk & (initial_run_activity >= minimum_frames)

        active_eligible = has_speech & exactly_one_speaker & history_safe & (condition_a | condition_b)
        return all_silence_eligible | active_eligible

    def _splice_silence_masks(self, activity_info: _SubsegmentActivity) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return valid chunk-one ends and chunk-two starts for splice silence.

        Args:
            activity_info: Reusable compact activity metadata.

        Returns:
            Two Boolean tensors of shape ``(T + 1,)``. The first marks
            boundaries preceded by the configured silent suffix; the second
            marks boundaries followed by the configured silent prefix.
        """
        num_frames = activity_info.activity.shape[0]
        silence_frames = self._seconds_to_feature_frames(self.subsegment_splice_silence_sec)
        boundaries = torch.arange(num_frames + 1, dtype=torch.long, device=activity_info.activity.device)
        total_prefix = activity_info.total_prefix

        safe_end = torch.zeros(num_frames + 1, dtype=torch.bool, device=boundaries.device)
        end_boundaries = boundaries[boundaries >= silence_frames]
        safe_end[end_boundaries] = (total_prefix[end_boundaries] - total_prefix[end_boundaries - silence_frames]) == 0

        safe_start = torch.zeros(num_frames + 1, dtype=torch.bool, device=boundaries.device)
        start_boundaries = boundaries[boundaries + silence_frames <= num_frames]
        safe_start[start_boundaries] = (
            total_prefix[start_boundaries + silence_frames] - total_prefix[start_boundaries]
        ) == 0
        return safe_end, safe_start

    def _sample_single_chunk_bounds(
        self, activity_info: _SubsegmentActivity, max_len: int, min_len: int
    ) -> Optional[List[Tuple[int, int]]]:
        """Filter all starts by ATS safety and capacity, then sample one chunk.

        A non-empty source no longer than ``min_len`` is evaluated as the sole
        full-source candidate rather than rejected for its duration. Fixed
        speaker capacity is enforced unless ``max_spks == -1``. When
        ``subsegment_nspk_bias`` exceeds one, candidates are weighted
        exponentially by their number of active speakers.

        Args:
            activity_info: Reusable compact activity metadata.
            max_len: Maximum selected length in feature frames.
            min_len: Minimum selected length in feature frames.

        Returns:
            A one-element ``[(start, end)]`` list, or ``None`` when no
            candidate survives.
        """
        num_frames, num_speakers = activity_info.activity.shape
        if num_frames == 0:
            return None

        if num_frames <= min_len:
            candidates = torch.zeros(1, dtype=torch.long, device=activity_info.activity.device)
            ends = torch.full_like(candidates, num_frames)
        else:
            num_candidates = num_frames - min_len + 1
            candidates = torch.arange(num_candidates, dtype=torch.long, device=activity_info.activity.device)
            ends = torch.clamp(candidates + max_len, max=num_frames)
        eligible = self._source_chunk_eligibility(activity_info, candidates, ends)
        candidates = candidates[eligible]
        ends = ends[eligible]

        speaker_counts = None
        capacity_limited = self.max_spks != -1
        if (capacity_limited and num_speakers > self.max_spks) or self.subsegment_nspk_bias > 1.0:
            speaker_presence = self._chunk_speaker_presence(activity_info, candidates, ends)
            speaker_counts = speaker_presence.sum(dim=1)
            if capacity_limited and num_speakers > self.max_spks:
                within_capacity = speaker_counts <= self.max_spks
                candidates = candidates[within_capacity]
                speaker_counts = speaker_counts[within_capacity]

        if candidates.numel() == 0:
            return None
        if self.subsegment_nspk_bias == 1.0:
            start = candidates[random.randrange(candidates.numel())].item()
        else:
            weights = self.subsegment_nspk_bias**speaker_counts
            start = candidates[torch.multinomial(weights, 1).item()].item()
        return [(start, min(start + max_len, num_frames))]

    def _sample_two_chunk_bounds(
        self,
        activity_info: _SubsegmentActivity,
        total_len: int,
        min_chunk_len: int,
    ) -> Optional[List[Tuple[int, int]]]:
        """Sample one two-chunk plan with a shortened-second-chunk fallback.

        The method samples one first-chunk length and one compatible first
        chunk. Both chunks obey the source-start policy, cannot overlap in the
        source, and meet the configured splice-silence masks. It first tries to
        use all remaining target length for chunk two. If no full-length second
        candidate survives, it evaluates one maximum geometrically available
        shorter chunk per splice-safe second start and samples among the
        longest eligible candidates. It does not retry another first length or
        first chunk. Speaker capacity is skipped when ``max_spks == -1``;
        speaker-count bias still applies.

        Args:
            activity_info: Reusable compact activity metadata.
            total_len: Combined output length in feature frames.
            min_chunk_len: Minimum length of either chunk in feature frames.

        Returns:
            ``[(start1, end1), (start2, end2)]`` in output concatenation order,
            or ``None``.
        """
        num_frames = activity_info.activity.shape[0]
        if total_len < 2 * min_chunk_len:
            return None

        splice_safe_end, splice_safe_start = self._splice_silence_masks(activity_info)
        length1 = random.randint(min_chunk_len, total_len - min_chunk_len)
        remaining = total_len - length1

        end1_candidates = torch.where(splice_safe_end)[0]
        start1_candidates = end1_candidates - length1
        valid_start = start1_candidates >= 0
        start1_candidates = start1_candidates[valid_start]
        end1_candidates = end1_candidates[valid_start]
        if start1_candidates.numel() == 0:
            return None

        first_eligible = self._source_chunk_eligibility(activity_info, start1_candidates, end1_candidates)
        start1_candidates = start1_candidates[first_eligible]
        end1_candidates = end1_candidates[first_eligible]
        if start1_candidates.numel() == 0:
            return None

        first_speaker_presence = self._chunk_speaker_presence(activity_info, start1_candidates, end1_candidates)
        first_speaker_counts = first_speaker_presence.sum(dim=1)
        if self.max_spks != -1:
            within_capacity = first_speaker_counts <= self.max_spks
            start1_candidates = start1_candidates[within_capacity]
            end1_candidates = end1_candidates[within_capacity]
            first_speaker_presence = first_speaker_presence[within_capacity]
            first_speaker_counts = first_speaker_counts[within_capacity]
        if start1_candidates.numel() == 0:
            return None

        if self.subsegment_nspk_bias == 1.0:
            first_index = random.randrange(start1_candidates.numel())
        else:
            first_weights = self.subsegment_nspk_bias**first_speaker_counts
            first_index = torch.multinomial(first_weights, 1).item()
        start1 = start1_candidates[first_index].item()
        end1 = end1_candidates[first_index].item()
        first_speakers = first_speaker_presence[first_index]

        start2_candidates = torch.where(splice_safe_start[:-1])[0]
        available = torch.where(
            start2_candidates < start1,
            start1 - start2_candidates,
            torch.where(
                start2_candidates >= end1,
                num_frames - start2_candidates,
                torch.zeros_like(start2_candidates),
            ),
        )

        full_length = available >= remaining
        full_starts = start2_candidates[full_length]
        full_ends = full_starts + remaining
        full_eligible = self._source_chunk_eligibility(activity_info, full_starts, full_ends)
        full_starts = full_starts[full_eligible]
        full_ends = full_ends[full_eligible]
        full_speaker_presence = self._chunk_speaker_presence(activity_info, full_starts, full_ends)
        full_union_counts = (full_speaker_presence | first_speakers.unsqueeze(0)).sum(dim=1)
        if self.max_spks != -1:
            full_capacity = full_union_counts <= self.max_spks
            full_starts = full_starts[full_capacity]
            full_union_counts = full_union_counts[full_capacity]
        if full_starts.numel() > 0:
            if self.subsegment_nspk_bias == 1.0:
                second_index = random.randrange(full_starts.numel())
            else:
                second_weights = self.subsegment_nspk_bias**full_union_counts
                second_index = torch.multinomial(second_weights, 1).item()
            start2 = full_starts[second_index].item()
            return [(start1, end1), (start2, start2 + remaining)]

        length2_candidates = torch.minimum(available, torch.full_like(available, remaining))
        long_enough = length2_candidates >= min_chunk_len
        short_starts = start2_candidates[long_enough]
        short_lengths = length2_candidates[long_enough]
        short_ends = short_starts + short_lengths
        short_eligible = self._source_chunk_eligibility(activity_info, short_starts, short_ends)
        short_starts = short_starts[short_eligible]
        short_lengths = short_lengths[short_eligible]
        short_ends = short_ends[short_eligible]
        short_speaker_presence = self._chunk_speaker_presence(activity_info, short_starts, short_ends)
        short_union_counts = (short_speaker_presence | first_speakers.unsqueeze(0)).sum(dim=1)
        if self.max_spks != -1:
            short_capacity = short_union_counts <= self.max_spks
            short_starts = short_starts[short_capacity]
            short_lengths = short_lengths[short_capacity]
            short_union_counts = short_union_counts[short_capacity]
        if short_starts.numel() == 0:
            return None

        longest_length = short_lengths.max()
        longest = short_lengths == longest_length
        short_starts = short_starts[longest]
        short_union_counts = short_union_counts[longest]
        if self.subsegment_nspk_bias == 1.0:
            second_index = random.randrange(short_starts.numel())
        else:
            second_weights = self.subsegment_nspk_bias**short_union_counts
            second_index = torch.multinomial(second_weights, 1).item()
        start2 = short_starts[second_index].item()
        return [(start1, end1), (start2, start2 + longest_length.item())]

    def _load_audio_chunks(
        self,
        audio_file: Union[str, List[str]],
        offset: float,
        bounds: Sequence[Tuple[int, int]],
    ) -> torch.Tensor:
        """Load only selected source ranges and augment the combined waveform.

        Planning is complete before this method performs waveform I/O. Each
        feature-frame range is converted to an absolute audio offset and
        duration, trimmed or zero-padded to its exact expected sample count,
        concatenated in ``bounds`` order, and passed through the waveform
        featurizer once so random augmentation is consistent across chunks.

        Args:
            audio_file: Audio path or paths accepted by ``AudioSegment``.
            offset: Manifest-segment offset in seconds.
            bounds: Ordered sequence of ``(start, end)`` feature-frame pairs.

        Returns:
            Waveform tensor containing only the selected ranges.
        """
        sample_rate = self.featurizer.sample_rate
        samples_per_frame = sample_rate / self.feat_per_sec
        chunks = []
        for start, end in bounds:
            segment = AudioSegment.from_file(
                audio_file,
                target_sr=sample_rate,
                int_values=self.featurizer.int_values,
                offset=offset + start / self.feat_per_sec,
                duration=(end - start) / self.feat_per_sec,
            )
            samples = segment.samples
            expected_samples = round((end - start) * samples_per_frame)
            if samples.shape[0] < expected_samples:
                pad_width = [(0, expected_samples - samples.shape[0])]
                pad_width.extend([(0, 0)] * (samples.ndim - 1))
                samples = np.pad(samples, pad_width)
            chunks.append(samples[:expected_samples])

        combined = AudioSegment(np.concatenate(chunks, axis=0), sample_rate)
        return self.featurizer.process_segment(combined)

    def _fallback_noise_subsegment(self, sample: Any, observed_speakers: int = 0) -> _EESDSample:
        """Create a full-duration negative example without loading source audio.

        Args:
            sample: Manifest item whose duration is used when
                ``session_len_sec`` is non-positive.
            observed_speakers: Speaker width to preserve when
                ``max_spks == -1``.

        Returns:
            Exact-length, zero-mean Gaussian noise normalized to RMS ``1e-3``;
            its scalar length; full-length all-zero floating-point targets;
            the one-element target length; and aligned all-``None`` speaker
            names. Target length follows ordinary timestamp generation and
            frontend frame-count clamping.

        Raises:
            ValueError: If the resolved duration is not positive and finite, or
                if the sample rate cannot represent one feature frame, or if
                the duration rounds to a non-positive waveform sample count.
        """
        duration = self.session_len_sec if self.session_len_sec > 0 else sample.duration
        if not isinstance(duration, Real) or not math.isfinite(float(duration)) or duration <= 0:
            raise ValueError(f"Fallback subsegment duration must be finite and > 0, got {duration!r}")

        sample_rate = self.featurizer.sample_rate
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, Real)
            or not math.isfinite(float(sample_rate))
            or sample_rate <= 0
        ):
            raise ValueError(f"Fallback subsegment sample rate must be finite and > 0, got {sample_rate!r}")
        if int(sample_rate / self.feat_per_sec) <= 0:
            raise ValueError(
                f"Fallback subsegment sample rate ({sample_rate!r}) must be at least "
                f"the feature frame rate ({self.feat_per_sec})"
            )
        num_samples = round(duration * sample_rate)
        if num_samples <= 0:
            raise ValueError(
                f"Fallback subsegment sample count must be > 0, got {num_samples} "
                f"for duration={duration!r} and sample_rate={sample_rate!r}"
            )

        audio_signal = torch.randn(num_samples, dtype=torch.float32)
        if num_samples > 1:
            audio_signal -= audio_signal.mean()
        rms = audio_signal.square().mean().sqrt()
        if not torch.isfinite(rms) or rms <= 0:
            audio_signal = torch.arange(num_samples, dtype=torch.float32) - (num_samples - 1) / 2
            if num_samples == 1:
                audio_signal.fill_(1)
            rms = audio_signal.square().mean().sqrt()
        audio_signal *= 1e-3 / rms

        audio_signal_length = torch.tensor(num_samples, dtype=torch.long)
        waveform_duration = num_samples / sample_rate
        target_len = self.get_segment_timestamps(duration=waveform_duration, sample_rate=sample_rate)
        frontend_frame_count = max(0, self.get_frame_count_from_time_series_length(num_samples))
        target_len = torch.clamp(target_len, max=frontend_frame_count)
        target_width = self._target_speaker_width(observed_speakers)
        return (
            audio_signal,
            audio_signal_length,
            torch.zeros((int(target_len.item()), target_width), dtype=torch.float32),
            target_len,
            [None] * target_width,
        )

    def _create_subsegment(self, sample: Any, offset: float) -> _EESDSample:
        """Plan, load, and label one ATS-safe training subsegment.

        Planning uses RTTM-derived Boolean activity before waveform I/O. The
        method attempts two-chunk sampling according to its configured rate,
        falls back to single-chunk sampling, and loads only accepted source
        ranges. Missing RTTM input is treated as all-silence activity. A sample
        with no valid bounds or too little loaded audio is replaced by a
        full-duration negative example from ``_fallback_noise_subsegment``.

        Speaker columns inactive in the selected bounds are removed before
        fixed-width padding. ``max_spks == -1`` retains all selected speakers
        without padding to a configured cap. Final target aggregation respects
        ``soft_targets``: averaged activity is returned in soft mode and
        thresholded floating-point activity in hard mode.

        Args:
            sample: Manifest item containing audio path(s), optional RTTM path,
                duration, and offset metadata.
            offset: Manifest-segment offset in seconds.

        Returns:
            ``(audio_signal, audio_length, targets, target_length,
            speaker_names)`` with speaker names aligned to target columns.
        """
        duration = sample.duration
        if not isinstance(duration, Real) or not math.isfinite(float(duration)) or duration <= 0:
            return self._fallback_noise_subsegment(sample)
        if sample.rttm_file in [None, '']:
            rttm_lines = []
        else:
            with open(sample.rttm_file, 'r') as f:
                rttm_lines = f.readlines()

        rttm_timestamps, sess_to_global_spkids = extract_frame_info_from_rttm(offset, duration, rttm_lines)
        num_speakers = len(sess_to_global_spkids)
        frame_level_target = get_frame_targets_from_rttm(
            rttm_timestamps=rttm_timestamps,
            offset=offset,
            duration=duration,
            round_digits=self.round_digits,
            feat_per_sec=self.feat_per_sec,
            max_spks=max(1, num_speakers),
            dtype=torch.bool,
        )
        activity_info = self._prepare_subsegment_activity(frame_level_target)

        if self.session_len_sec > 0:
            max_len_frames = int(self.session_len_sec * self.feat_per_sec)
            single_chunk_min_len_frames = self._seconds_to_feature_frames(self.subsegment_single_chunk_min_len_sec)
            two_chunk_min_len_frames = self._seconds_to_feature_frames(self.subsegment_two_chunk_min_len_sec)
            bounds = None
            if random.random() < self.subsegment_two_chunks_rate:
                bounds = self._sample_two_chunk_bounds(
                    activity_info,
                    total_len=max_len_frames,
                    min_chunk_len=two_chunk_min_len_frames,
                )
            if bounds is None:
                bounds = self._sample_single_chunk_bounds(
                    activity_info,
                    max_len=max_len_frames,
                    min_len=single_chunk_min_len_frames,
                )
        else:
            bounds = [(0, frame_level_target.shape[0])] if frame_level_target.shape[0] > 0 else None
            if bounds is not None and self.max_spks != -1 and frame_level_target.any(dim=0).sum() > self.max_spks:
                bounds = None

        if bounds is None:
            return self._fallback_noise_subsegment(sample, num_speakers)

        frame_level_target = torch.cat([frame_level_target[start:end] for start, end in bounds])
        spks_tokeep = torch.where(frame_level_target.any(dim=0))[0].tolist()
        if self.max_spks != -1 and len(spks_tokeep) > self.max_spks:
            raise RuntimeError(
                f"Selected subsegment has {len(spks_tokeep)} speakers, exceeding max_spks={self.max_spks}"
            )

        if spks_tokeep:
            frame_level_target = frame_level_target[:, spks_tokeep].float()
            speaker_names = self._build_speaker_names(sess_to_global_spkids, columns=spks_tokeep)
        else:
            frame_level_target = torch.zeros((frame_level_target.shape[0], 1))
            speaker_names = [None]
        if self.max_spks != -1 and frame_level_target.shape[1] < self.max_spks:
            pad_width = self.max_spks - frame_level_target.shape[1]
            frame_level_target = torch.nn.functional.pad(frame_level_target, (0, pad_width), 'constant', 0)
            speaker_names.extend([None] * (self.max_spks - len(speaker_names)))

        del activity_info
        audio_signal = self._load_audio_chunks(sample.audio_file, offset, bounds)
        min_viable_samples = int(self.min_subsegment_duration * self.featurizer.sample_rate)
        if audio_signal.shape[0] < min_viable_samples:
            return self._fallback_noise_subsegment(sample, len(speaker_names))

        audio_signal_length = torch.tensor(audio_signal.shape[0]).long()
        session_len_sec = audio_signal.shape[0] / self.featurizer.sample_rate
        target_len = self.get_segment_timestamps(duration=session_len_sec, sample_rate=self.featurizer.sample_rate)
        target_len = torch.clamp(target_len, max=self.get_frame_count_from_time_series_length(audio_signal.shape[0]))
        targets = self._get_segment_targets(feat_level_target=frame_level_target, target_len=target_len)
        targets = targets[:target_len, :]
        return audio_signal, audio_signal_length, targets, target_len, speaker_names

    def __getitem__(self, index: int) -> _EESDSample:
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0
        offset = sample.offset
        if self.subsegment_mode:
            return self._create_subsegment(sample, offset)

        if self.session_len_sec < 0:
            session_len_sec = sample.duration
        else:
            session_len_sec = min(sample.duration, self.session_len_sec)

        audio_signal = self.featurizer.process(sample.audio_file, offset=offset, duration=session_len_sec)

        # We should resolve the length mis-match from the round-off errors between these two variables:
        # `session_len_sec` and `audio_signal.shape[0]`
        session_len_sec = (
            np.floor(audio_signal.shape[0] / self.featurizer.sample_rate * self.floor_decimal) / self.floor_decimal
        )
        audio_signal = audio_signal[: round(self.featurizer.sample_rate * session_len_sec)]
        audio_signal_length = torch.tensor(audio_signal.shape[0]).long()

        # Target length should be following the ASR feature extraction convention: Use self.get_frame_count_from_time_series_length.
        target_len = self.get_segment_timestamps(duration=session_len_sec, sample_rate=self.featurizer.sample_rate)
        target_len = torch.clamp(target_len, max=self.get_frame_count_from_time_series_length(audio_signal.shape[0]))

        targets, speaker_names = self.parse_rttm_for_targets_and_lens(
            rttm_file=sample.rttm_file, offset=offset, duration=session_len_sec, target_len=target_len
        )
        targets = targets[:target_len, :]
        return audio_signal, audio_signal_length, targets, target_len, speaker_names

    def eesd_train_collate_fn(self, batch: Sequence[_EESDSample]) -> _EESDBatch:
        """Collate a five-field EESD batch without dropping samples.

        Waveforms, target sequences, and dynamic speaker dimensions are padded
        to batch maxima. Speaker names are padded with ``None`` so metadata
        remains aligned with the final target speaker columns.

        Args:
            batch: Sequence of five-field EESD samples.

        Returns:
            ``(audio_signal, feature_length, targets, target_lens,
            speaker_names)``. The first four fields are padded or stacked
            tensors. ``speaker_names`` contains one aligned metadata list per
            input sample.
        """
        audio_signal = tuple(item[0] for item in batch)
        targets = tuple(item[2] for item in batch)
        audio_signal_list, feature_length_list = [], []
        target_len_list, targets_list = [], []
        speaker_names_list = []

        max_raw_feat_len = max([x.shape[0] for x in audio_signal])
        max_target_len = max([x.shape[0] for x in targets])
        max_target_speakers = max([x.shape[1] for x in targets])
        max_ch = max(feat.shape[1] if feat.ndim > 1 else 1 for feat in audio_signal)
        for feat, feat_len, tgt, segment_ct, spk_names in batch:
            seq_len = tgt.shape[0]
            if max_ch > 1 and feat.ndim == 1:
                feat = feat.unsqueeze(1)
            if len(feat.shape) > 1:
                pad_feat = (0, 0, 0, max_raw_feat_len - feat.shape[0])
            else:
                pad_feat = (0, max_raw_feat_len - feat.shape[0])
            if feat.shape[0] < feat_len:
                feat_len_pad = feat_len - feat.shape[0]
                feat = torch.nn.functional.pad(feat, (0, feat_len_pad))
            pad_tgt = (0, max_target_speakers - tgt.shape[1], 0, max_target_len - seq_len)
            padded_feat = torch.nn.functional.pad(feat, pad_feat)
            padded_tgt = torch.nn.functional.pad(tgt, pad_tgt)
            if max_ch > 1 and padded_feat.shape[1] < max_ch:
                feat_ch_pad = max_ch - padded_feat.shape[1]
                padded_feat = torch.nn.functional.pad(padded_feat, (0, feat_ch_pad))
            audio_signal_list.append(padded_feat)
            feature_length_list.append(feat_len.clone().detach())
            target_len_list.append(segment_ct.clone().detach())
            targets_list.append(padded_tgt)
            speaker_names_list.append(
                list(spk_names[:max_target_speakers]) + [None] * max(0, max_target_speakers - len(spk_names))
            )
        audio_signal = torch.stack(audio_signal_list)
        feature_length = torch.stack(feature_length_list)
        target_lens = torch.stack(target_len_list).squeeze(1)
        targets = torch.stack(targets_list)
        return audio_signal, feature_length, targets, target_lens, speaker_names_list
