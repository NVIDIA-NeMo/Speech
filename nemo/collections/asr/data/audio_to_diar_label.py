# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from nemo.collections.asr.parts.utils.speaker_utils import convert_rttm_line, get_subsegments
from nemo.collections.common.parts.preprocessing.collections import EndtoEndDiarizationSpeechLabel
from nemo.core.classes import Dataset
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType, ProbsType
from nemo.utils import logging


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
):
    """
    Create a multi-dimensional vector sequence containing speaker timestamp information in RTTM.
    The unit-length is the frame shift length of the acoustic feature. The feature-level annotations
    `feat_level_target` will later be converted to base-segment level diarization label.

    Args:
        rttm_timestamps (list):
            List containing start and end time for each speaker segment label.
            stt_list, end_list and speaker_list are contained.
        feat_per_sec (int):
            Number of feature frames per second.
            This quantity is determined by window_stride variable in preprocessing module.
        target_spks (tuple):
            Speaker indices that are generated from combinations. If there are only one or two speakers,
            only a single target_spks variable is generated.

    Returns:
        feat_level_target (torch.tensor):
            Tensor containing label for each feature level frame.
    """
    stt_list, end_list, speaker_list = rttm_timestamps
    sorted_speakers = sorted(list(set(speaker_list)))
    total_fr_len = int(duration * feat_per_sec)
    if len(sorted_speakers) > max_spks:
        logging.warning(
            f"Number of speakers in RTTM file {len(sorted_speakers)} exceeds the maximum number of speakers: "
            f"{max_spks}! Only {max_spks} first speakers remain, and this will affect frame metrics!"
        )
    feat_level_target = torch.zeros(total_fr_len, max_spks)
    for count, (stt, end, spk_rttm_key) in enumerate(zip(stt_list, end_list, speaker_list)):
        if end < offset or stt > offset + duration:
            continue
        stt, end = max(offset, stt), min(offset + duration, end)
        spk = spk_rttm_key
        if spk < max_spks:
            stt_fr, end_fr = int((stt - offset) * feat_per_sec), int((end - offset) * feat_per_sec)
            feat_level_target[stt_fr:end_fr, spk] = 1
    return feat_level_target


class _AudioToSpeechE2ESpkDiarDataset(Dataset):
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
        manifest_filepath (str):
            Path to input manifest json files.
        multiargs_dict (dict):
            Dictionary containing the parameters for multiscale segmentation and clustering.
        soft_label_thres (float):
            Threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating audio_signal from the raw waveform.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        output_types = {
            "audio_signal": NeuralType(('B', 'T'), AudioSignal()),
            "audio_length": NeuralType(('B'), LengthsType()),
            "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
            "target_len": NeuralType(('B'), LengthsType()),
        }

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        soft_label_thres: float,
        session_len_sec: float,
        num_spks: int,
        featurizer,
        fb_featurizer,
        window_stride: float,
        min_subsegment_duration: float = 0.03,
        global_rank: int = 0,
        dtype=torch.float16,
        round_digits: int = 2,
        soft_targets: bool = False,
        subsampling_factor: int = 8,
        device: str = 'cpu',
    ):
        super().__init__()
        self.collection = EndtoEndDiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            round_digits=round_digits,
        )
        self.featurizer = featurizer
        self.fb_featurizer = fb_featurizer
        # STFT and subsampling factor parameters
        self.n_fft = self.fb_featurizer.n_fft
        self.hop_length = self.fb_featurizer.hop_length
        self.stft_pad_amount = self.fb_featurizer.stft_pad_amount
        self.subsampling_factor = subsampling_factor
        # Annotation and target length parameters
        self.round_digits = round_digits
        self.feat_per_sec = int(1 / window_stride)
        self.diar_frame_length = round(subsampling_factor * window_stride, round_digits)
        self.session_len_sec = session_len_sec
        self.soft_label_thres = soft_label_thres
        self.max_spks = num_spks
        self.min_subsegment_duration = min_subsegment_duration
        self.dtype = dtype
        self.use_asr_style_frame_count = True
        self.soft_targets = soft_targets
        self.round_digits = 2
        self.floor_decimal = 10**self.round_digits
        self.device = device
        self.global_rank = global_rank

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

    def parse_rttm_for_targets_and_lens(self, rttm_file, offset, duration, target_len):
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function converts (start, end, speaker_id) format into base-scale (the finest scale) segment level
        diarization label in a matrix form.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]
        """
        if rttm_file in [None, '']:
            num_seg = torch.max(target_len)
            targets = torch.zeros(num_seg, self.max_spks)
            return targets

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

        soft_target_seg = self.get_soft_targets_seg(feat_level_target=fr_level_target, target_len=target_len)
        if self.soft_targets:
            step_target = soft_target_seg
        else:
            step_target = (soft_target_seg >= self.soft_label_thres).float()
        return step_target

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
        targets = torch.zeros(num_seg, self.max_spks)
        stride = int(self.feat_per_sec * self.diar_frame_length)
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

    def __getitem__(self, index):
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0
        offset = sample.offset
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

        targets = self.parse_rttm_for_targets_and_lens(
            rttm_file=sample.rttm_file, offset=offset, duration=session_len_sec, target_len=target_len
        )
        targets = targets[:target_len, :]
        return audio_signal, audio_signal_length, targets, target_len


def _eesd_train_collate_fn(self, batch):
    """
    Collate a batch of variables needed for training the end-to-end speaker diarization (EESD) model
    from raw waveforms to diarization labels. The following variables are included in the training/validation batch:

    Args:
        batch (tuple):
            A tuple containing the variables for diarization training.

    Returns:
        audio_signal (torch.Tensor):
            A tensor containing the raw waveform samples (time series) loaded from the `audio_filepath`
            in the input manifest file.
        feature_length (torch.Tensor):
            A tensor containing the lengths of the raw waveform samples.
        targets (torch.Tensor):
            Groundtruth speaker labels for the given input embedding sequence.
        target_lens (torch.Tensor):
            A tensor containing the number of segments for each sample in the batch, necessary for
            reshaping inputs to the EESD model.
    """
    packed_batch = list(zip(*batch))
    audio_signal, feature_length, targets, target_len = packed_batch
    audio_signal_list, feature_length_list = [], []
    target_len_list, targets_list = [], []

    max_raw_feat_len = max([x.shape[0] for x in audio_signal])
    max_target_len = max([x.shape[0] for x in targets])
    if max([len(feat.shape) for feat in audio_signal]) > 1:
        max_ch = max([feat.shape[1] for feat in audio_signal])
    else:
        max_ch = 1
    for feat, feat_len, tgt, segment_ct in batch:
        seq_len = tgt.shape[0]
        if len(feat.shape) > 1:
            pad_feat = (0, 0, 0, max_raw_feat_len - feat.shape[0])
        else:
            pad_feat = (0, max_raw_feat_len - feat.shape[0])
        if feat.shape[0] < feat_len:
            feat_len_pad = feat_len - feat.shape[0]
            feat = torch.nn.functional.pad(feat, (0, feat_len_pad))
        pad_tgt = (0, 0, 0, max_target_len - seq_len)
        padded_feat = torch.nn.functional.pad(feat, pad_feat)
        padded_tgt = torch.nn.functional.pad(tgt, pad_tgt)
        if max_ch > 1 and padded_feat.shape[1] < max_ch:
            feat_ch_pad = max_ch - padded_feat.shape[1]
            padded_feat = torch.nn.functional.pad(padded_feat, (0, feat_ch_pad))
        audio_signal_list.append(padded_feat)
        feature_length_list.append(feat_len.clone().detach())
        target_len_list.append(segment_ct.clone().detach())
        targets_list.append(padded_tgt)
        audio_signal = torch.stack(audio_signal_list)
    feature_length = torch.stack(feature_length_list)
    target_lens = torch.stack(target_len_list).squeeze(1)
    targets = torch.stack(targets_list)
    return audio_signal, feature_length, targets, target_lens


class AudioToSpeechE2ESpkDiarDataset(_AudioToSpeechE2ESpkDiarDataset):
    """
    Dataset class for loading a JSON file containing paths to audio files,
    RTTM (Rich Transcription Time Marked) files, and the number of speakers.
    This class is designed for training or fine-tuning a speaker embedding
    extractor and diarization decoder simultaneously.

    The JSON manifest file should have entries in the following format:

    Example:
    {
        "audio_filepath": "/path/to/audio_0.wav",
        "num_speakers": 2,
        "rttm_filepath": "/path/to/diar_label_0.rttm"
    }
    ...
    {
        "audio_filepath": "/path/to/audio_n.wav",
        "num_speakers": 2,
        "rttm_filepath": "/path/to/diar_label_n.rttm"
    }

    Args:
        manifest_filepath (str):
            Path to the input manifest JSON file containing paths to audio and RTTM files.
        soft_label_thres (float):
            Threshold for assigning soft labels to segments based on RTTM file information.
        session_len_sec (float):
            Duration of each session (in seconds) for training or fine-tuning.
        num_spks (int):
            Number of speakers in the audio files.
        featurizer:
            Instance of a featurizer for generating features from the raw waveform.
        window_stride (float):
            Window stride (in seconds) for extracting acoustic features, used to calculate
            the number of feature frames.
        global_rank (int):
            Global rank of the current process (used for distributed training).
        soft_targets (bool):
            Whether or not to use soft targets during training.

    Methods:
        eesd_train_collate_fn(batch):
            Collates a batch of data for end-to-end speaker diarization training.
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        soft_label_thres: float,
        session_len_sec: float,
        num_spks: int,
        featurizer,
        fb_featurizer,
        window_stride,
        global_rank: int,
        soft_targets: bool,
        device: str,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            soft_label_thres=soft_label_thres,
            session_len_sec=session_len_sec,
            num_spks=num_spks,
            featurizer=featurizer,
            fb_featurizer=fb_featurizer,
            window_stride=window_stride,
            global_rank=global_rank,
            soft_targets=soft_targets,
            device=device,
        )

    def eesd_train_collate_fn(self, batch):
        """Collate a batch of data for end-to-end speaker diarization training."""
        return _eesd_train_collate_fn(self, batch)
