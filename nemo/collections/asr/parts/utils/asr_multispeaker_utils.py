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
import json
import logging
import math
import random
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Union

import torch.utils.data
from cytoolz import groupby
from lhotse import AudioSource, Recording, SupervisionSegment, SupervisionSet
from lhotse.cut import Cut, MixedCut, MixTrack, MonoCut
from lhotse.lazy import LazyJsonlIterator
from lhotse.utils import compute_num_samples, uuid4


def find_first_nonzero(mat: torch.Tensor, max_cap_val=-1, thres: float = 0.5) -> torch.Tensor:
    """
    Finds the first nonzero value in the matrix, discretizing it to the specified maximum capacity.

    Args:
        mat (Tensor): A torch tensor representing the matrix.
        max_cap_val (int): The maximum capacity to which the matrix values will be discretized.
        thres (float): The threshold value for discretizing the matrix values.

    Returns:
        mask_max_indices (Tensor): A torch tensor representing the discretized matrix with the first
        nonzero value in each row.
    """
    # Discretize the matrix to the specified maximum capacity
    labels_discrete = mat.clone()
    labels_discrete[labels_discrete < thres] = 0
    labels_discrete[labels_discrete >= thres] = 1

    # non zero values mask
    non_zero_mask = labels_discrete != 0
    # operations on the mask to find first nonzero values in the rows
    mask_max_values, mask_max_indices = torch.max(non_zero_mask, dim=1)
    # if the max-mask is zero, there is no nonzero value in the row
    mask_max_indices[mask_max_values == 0] = max_cap_val
    return mask_max_indices


def find_best_permutation(match_score: torch.Tensor, speaker_permutations: torch.Tensor) -> torch.Tensor:
    """
    Finds the best permutation indices based on the match score.

    Args:
        match_score (torch.Tensor): A tensor containing the match scores for each permutation.
            Shape: (batch_size, num_permutations)
        speaker_permutations (torch.Tensor): A tensor containing all possible speaker permutations.
            Shape: (num_permutations, num_speakers)

    Returns:
        torch.Tensor: A tensor containing the best permutation indices for each batch.
            Shape: (batch_size, num_speakers)
    """
    batch_best_perm = torch.argmax(match_score, axis=1)
    rep_speaker_permutations = speaker_permutations.repeat(batch_best_perm.shape[0], 1).to(match_score.device)
    perm_size = speaker_permutations.shape[0]
    global_inds_vec = (
        torch.arange(0, perm_size * batch_best_perm.shape[0], perm_size).to(batch_best_perm.device) + batch_best_perm
    )
    return rep_speaker_permutations[global_inds_vec.to(rep_speaker_permutations.device), :]


def reconstruct_labels(labels: torch.Tensor, batch_perm_inds: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs the labels using the best permutation indices with matrix operations.

    Args:
        labels (torch.Tensor): A tensor containing the original labels.
            Shape: (batch_size, num_frames, num_speakers)
        batch_perm_inds (torch.Tensor): A tensor containing the best permutation indices for each batch.
            Shape: (batch_size, num_speakers)

    Returns:
        torch.Tensor: A tensor containing the reconstructed labels using the best permutation indices.
            Shape: (batch_size, num_frames, num_speakers)
    """
    # Expanding batch_perm_inds to align with labels dimensions
    batch_size, num_frames, num_speakers = labels.shape
    batch_perm_inds_exp = batch_perm_inds.unsqueeze(1).expand(-1, num_frames, -1)

    # Reconstructing the labels using advanced indexing
    reconstructed_labels = torch.gather(labels, 2, batch_perm_inds_exp)
    return reconstructed_labels


def get_ats_targets(
    labels: torch.Tensor,
    preds: torch.Tensor,
    speaker_permutations: torch.Tensor,
    thres: float = 0.5,
    tolerance: float = 0,
) -> torch.Tensor:
    """
    Sorts labels and predictions to get the optimal of all arrival-time ordered permutations.

    Args:
        labels (torch.Tensor): A tensor containing the original labels.
            Shape: (batch_size, num_frames, num_speakers)
        preds (torch.Tensor): A tensor containing the predictions.
            Shape: (batch_size, num_frames, num_speakers)
        speaker_permutations (torch.Tensor): A tensor containing all possible speaker permutations.
            Shape: (num_permutations, num_speakers)
        thres (float): The threshold value for discretizing the matrix values. Default is 0.5.
        tolerance (float): The tolerance for comparing the first speech frame indices. Default is 0.

    Returns:
        torch.Tensor: A tensor containing the reconstructed labels using the best permutation indices.
            Shape: (batch_size, num_frames, num_speakers)
    """
    # Find the first nonzero frame index for each speaker in each batch
    nonzero_ind = find_first_nonzero(
        mat=labels, max_cap_val=labels.shape[1], thres=thres
    )  # (batch_size, num_speakers)

    # Sort the first nonzero frame indices for arrival-time ordering
    sorted_values = torch.sort(nonzero_ind)[0]  # (batch_size, num_speakers)
    perm_size = speaker_permutations.shape[0]  # Scalar value (num_permutations)
    permed_labels = labels[:, :, speaker_permutations]  # (batch_size, num_frames, num_permutations, num_speakers)
    permed_nonzero_ind = find_first_nonzero(
        mat=permed_labels, max_cap_val=labels.shape[1]
    )  # (batch_size, num_permutations, num_speakers)

    # Compare the first frame indices of sorted labels with those of the permuted labels using tolerance
    perm_compare = (
        torch.abs(sorted_values.unsqueeze(1) - permed_nonzero_ind) <= tolerance
    )  # (batch_size, num_permutations, num_speakers)
    perm_mask = torch.all(perm_compare, dim=2).float()  # (batch_size, num_permutations)
    preds_rep = torch.unsqueeze(preds, 2).repeat(
        1, 1, perm_size, 1
    )  # Exapnd the preds: (batch_size, num_frames, num_permutations, num_speakers)

    # Compute the match score for each permutation by comparing permuted labels with preds
    match_score = (
        torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2) * perm_mask
    )  # (batch_size, num_permutations)
    batch_perm_inds = find_best_permutation(match_score, speaker_permutations)  # (batch_size, num_speakers)
    max_score_permed_labels = reconstruct_labels(labels, batch_perm_inds)  # (batch_size, num_frames, num_speakers)
    return max_score_permed_labels  # (batch_size, num_frames, num_speakers)


def get_pil_targets(labels: torch.Tensor, preds: torch.Tensor, speaker_permutations: torch.Tensor) -> torch.Tensor:
    """
    Sorts labels and predictions to get the optimal permutation based on the match score.

    Args:
        labels (torch.Tensor): A tensor containing the ground truth labels.
            Shape: (batch_size, num_speakers, num_classes)
        preds (torch.Tensor): A tensor containing the predicted values.
            Shape: (batch_size, num_speakers, num_classes)
        speaker_permutations (torch.Tensor): A tensor containing all possible speaker permutations.
            Shape: (num_permutations, num_speakers)

    Returns:
        torch.Tensor: A tensor of permuted labels that best match the predictions.
            Shape: (batch_size, num_speakers, num_classes)
    """
    permed_labels = labels[:, :, speaker_permutations]  # (batch_size, num_classes, num_permutations, num_speakers)
    # Repeat preds to match permutations for comparison
    preds_rep = torch.unsqueeze(preds, 2).repeat(
        1, 1, speaker_permutations.shape[0], 1
    )  # (batch_size, num_speakers, num_permutations, num_classes)
    match_score = torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2)  # (batch_size, num_permutations)
    batch_perm_inds = find_best_permutation(match_score, speaker_permutations)  # (batch_size, num_speakers)
    # Reconstruct labels based on the best permutation for each batch
    max_score_permed_labels = reconstruct_labels(labels, batch_perm_inds)  # (batch_size, num_speakers, num_classes)
    return max_score_permed_labels  # (batch_size, num_speakers, num_classes)


def find_segments_from_rttm(
    recording_id: str,
    rttms: SupervisionSet,
    start_after: float,
    end_before: float,
    adjust_offset: bool = True,
    tolerance: float = 0.001,
):
    """
    Finds segments from the given rttm file.
    This function is designed to replace rttm

    Args:
        recording_id (str): The recording ID in string format.
        rttms (SupervisionSet): The SupervisionSet instance.
        start_after (float): The start time after which segments are selected.
        end_before (float): The end time before which segments are selected.
        adjust_offset (bool): Whether to adjust the offset of the segments.
        tolerance (float): The tolerance for time matching. 0.001 by default.
    Returns:
        segments (List[SupervisionSegment]): A list of SupervisionSegment instances.
    """
    segment_by_recording_id = rttms._segments_by_recording_id
    if segment_by_recording_id is None:
        from cytoolz import groupby

        segment_by_recording_id = groupby(lambda seg: seg.recording_id, rttms)

    return [
        # We only modify the offset - the duration remains the same, as we're only shifting the segment
        # relative to the Cut's start, and not truncating anything.
        segment.with_offset(-start_after) if adjust_offset else segment
        for segment in segment_by_recording_id.get(recording_id, [])
        if segment.start < end_before + tolerance and segment.end > start_after + tolerance
    ]


def get_mask_from_segments(
    segments: list,
    a_cut: Optional[Union[MonoCut, MixedCut]],
    speaker_to_idx_map: torch.Tensor,
    num_speakers: int = 4,
    feat_per_sec: int = 100,
):
    """
    Generate mask matrix from segments list.
    This function is needed for speaker diarization with ASR model trainings.

    Args:
        segments: A list of Lhotse Supervision segments iterator.
        cut (MonoCut, MixedCut): Lhotse MonoCut or MixedCut instance.
        speaker_to_idx_map (dict): A dictionary mapping speaker names to indices.
        num_speakers (int): max number of speakers for all cuts ("mask" dim0), 4 by default
        feat_per_sec (int): number of frames per second, 100 by default, 0.01s frame rate

    Returns:
        mask (Tensor): A numpy array of shape (num_speakers, encoder_hidden_len).
            Dimension: (num_speakers, num_frames)
    """
    # get targets with 0.01s frame rate
    num_samples = round(a_cut.duration * feat_per_sec)
    mask = torch.zeros((num_samples, num_speakers))
    for rttm_sup in segments:
        speaker_idx = speaker_to_idx_map[rttm_sup.speaker]

        stt = max(rttm_sup.start, 0)
        ent = min(rttm_sup.end, a_cut.duration)
        stf = int(stt * feat_per_sec)
        enf = int(ent * feat_per_sec)
        mask[stf:enf, speaker_idx] = 1.0
    return mask


def get_soft_mask(feat_level_target, num_frames, stride):
    """
    Get soft mask from feat_level_target with stride.
    This function is needed for speaker diarization with ASR model trainings.

    Args:
        feat_level_target (Tensor): A numpy array of shape (num_frames, num_speakers).
            Dimension: (num_frames, num_speakers)
        num_sample (int): The total number of samples.
        stride (int): The stride for the mask.

    Returns:
        mask: The soft mask of shape (num_frames, num_speakers).
            Dimension: (num_frames, num_speakers)
    """

    num_speakers = feat_level_target.shape[1]
    mask = torch.zeros(num_frames, num_speakers)

    for index in range(num_frames):
        if index == 0:
            seg_stt_feat = 0
        else:
            seg_stt_feat = stride * index - 1 - int(stride / 2)
        if index == num_frames - 1:
            seg_end_feat = feat_level_target.shape[0]
        else:
            seg_end_feat = stride * index - 1 + int(stride / 2)
        mask[index] = torch.mean(feat_level_target[seg_stt_feat : seg_end_feat + 1, :], axis=0)
    return mask


def get_hidden_length_from_sample_length(
    num_samples: int, num_sample_per_mel_frame: int = 160, num_mel_frame_per_asr_frame: int = 8
) -> int:
    """
    Calculate the hidden length from the given number of samples.
    This function is needed for speaker diarization with ASR model trainings.

    This function computes the number of frames required for a given number of audio samples,
    considering the number of samples per mel frame and the number of mel frames per ASR frame.

    Please refer to the following function for more on feature frame length calculation:
        NeMo/nemo/collections/asr/parts/preprocessing/features.py::FilterbankFeatures::get_seq_len

    Parameters:
        num_samples (int): The total number of audio samples.
        num_sample_per_mel_frame (int, optional): The number of samples per mel frame. Default is 160.
        num_mel_frame_per_asr_frame (int, optional): The number of mel frames per ASR frame. Default is 8.

    Returns:
        hidden_length (int): The calculated hidden length in terms of the number of frames.
    """
    mel_frame_count = math.ceil(num_samples / num_sample_per_mel_frame)
    hidden_length = math.ceil(mel_frame_count / num_mel_frame_per_asr_frame)
    return int(hidden_length)


def speaker_to_target(
    a_cut,
    num_speakers: Optional[int] = None,
    num_sample_per_mel_frame: int = 160,
    num_mel_frame_per_asr_frame: int = 8,
    boundary_segments: bool = False,
    soft_label: bool = False,
    soft_thres: float = 0.5,
    return_text: bool = False,
):
    '''
    Get rttm samples corresponding to one cut, generate speaker mask numpy.ndarray with shape (num_speaker, hidden_length)
    This function is needed for speaker diarization with ASR model trainings.

    Args:
        a_cut (MonoCut, MixedCut): Lhotse Cut instance which is MonoCut or MixedCut instance.
        num_speakers (int): max number of speakers for all cuts ("mask" dim0), 4 by default
        num_sample_per_mel_frame (int): number of sample per mel frame, sample_rate / 1000 * window_stride, 160 by default (10ms window stride)
        num_mel_frame_per_asr_frame (int): encoder subsampling_factor, 8 by default
        boundary_segments (bool): set to True to include segments containing the boundary of the cut, False by default for multi-speaker ASR training
        soft_label (bool): set to True to use soft label that enables values in [0, 1] range, False by default and leads to binary labels.
        soft_thres (float): the threshold for the soft label, 0.5 by default.
        return_text (bool): set to True to return the text of the speakers (if it is available), False by default.

    Returns:
        mask (Tensor): speaker mask with shape (num_speaker, hidden_lenght)
    '''
    # get cut-related segments from rttms
    if isinstance(a_cut, MixedCut):
        cut_list = [track.cut for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
        offsets = [track.offset for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
    elif isinstance(a_cut, MonoCut):
        cut_list = [a_cut]
        offsets = [0]
    else:
        raise ValueError(f"Unsupported cut type type{a_cut}: only MixedCut and MonoCut are supported")

    segments_total = []

    for i, cut in enumerate(cut_list):
        if cut.custom.get('rttm_filepath', None):
            rttms = SupervisionSet.from_rttm(cut.rttm_filepath)
        elif cut.supervisions:
            rttms = SupervisionSet(cut.supervisions)
        else:
            logging.warning(f"No rttm or supervisions found for cut {cut.id}")
            continue

        start = cut.offset if hasattr(cut, 'offset') else cut.start
        end = start + cut.duration
        recording_id = rttms[0].recording_id if len(rttms) > 0 else cut.recording_id
        if boundary_segments:  # segments with seg_start < total_end and seg_end > total_start are included
            segments_iterator = find_segments_from_rttm(
                recording_id=recording_id, rttms=rttms, start_after=start, end_before=end, tolerance=0.0
            )
        else:  # segments with seg_start > total_start and seg_end < total_end are included
            segments_iterator = rttms.find(
                recording_id=recording_id, start_after=start, end_before=end, adjust_offset=True
            )  # , tolerance=0.0)

        for seg in segments_iterator:
            if seg.start < 0:
                seg.duration += seg.start
                seg.start = 0
            if seg.end > cut.duration:
                seg.duration -= seg.end - cut.duration
            seg.start += offsets[i]
            segments_total.append(seg)
    # apply arrival time sorting to the existing segments
    segments_total.sort(key=lambda rttm_sup: rttm_sup.start)

    seen = set()
    seen_add = seen.add
    speaker_ats = [s.speaker for s in segments_total if not (s.speaker in seen or seen_add(s.speaker))]

    speaker_to_idx_map = {spk: idx for idx, spk in enumerate(speaker_ats)}

    if num_speakers is None:
        num_speakers_dim = len(speaker_ats)
    else:
        if len(speaker_ats) > num_speakers:
            logging.warning(
                "Number of speakers in the target %s is greater than "
                "the maximum number of speakers %s. Truncating extra speakers. "
                "Set the `num_speakers` to higher value to avoid this warning.",
                len(speaker_ats),
                num_speakers,
            )
        num_speakers_dim = max(len(speaker_ats), num_speakers)
    # initialize mask matrices (num_speaker, encoder_hidden_len)
    feat_per_sec = int(a_cut.sampling_rate / num_sample_per_mel_frame)  # 100 by default
    num_samples = get_hidden_length_from_sample_length(
        a_cut.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame
    )
    frame_mask = get_mask_from_segments(segments_total, a_cut, speaker_to_idx_map, num_speakers_dim, feat_per_sec)
    soft_mask = get_soft_mask(frame_mask, num_samples, num_mel_frame_per_asr_frame)

    if soft_label:
        mask = soft_mask
    else:
        mask = (soft_mask > soft_thres).float()

    if return_text:
        speaker2text = defaultdict(list)
        for seg in segments_total:
            speaker2text[seg.speaker].append(seg.text)
        texts = [' '.join(speaker2text[speaker]) for speaker in speaker_ats]
        return mask, texts
    else:
        return mask


def read_seglst(seglst_filepath: str, session_id: Optional[str] = None):
    """
    Read the seglst file and return a list of SupervisionSegment.
    """
    with open(seglst_filepath, 'r', encoding='utf-8') as f:
        seglst = json.load(f)
        return [
            SupervisionSegment(
                id=f'{seg["session_id"]}-sup{i:05d}',
                recording_id=seg['session_id'] if session_id is None else session_id,
                start=float(seg['start_time']),
                duration=float(seg['end_time']) - float(seg['start_time']),
                text=seg['words'],
                speaker=seg['speaker'],
            )
            for i, seg in enumerate(seglst)
        ]


class MultiSpeakerMixtureGenerator:
    """
    This class is used to simulate multi-speaker audio data,
    which can be used for multi-speaker ASR and speaker diarization training.
    """

    def __init__(
        self,
        manifest_filepath,
        sample_rate,
        simulator_type,
        min_duration=0.1,
        max_duration=50.0,
        min_delay=0.5,
        random_seed=42,
        num_speakers=2,
        global_rank=0,
        world_size=1,
    ):
        """
        Args:
            cuts (CutSet): The cutset that contains single-speaker audio cuts.
                Please make sure that the cuts have the 'speaker_id' attribute.
            num_speakers (int): The number of speakers in the simulated audio.
                We only simulate the samples with the fixed number of speakers.
                The variation of the number of speakers is controlled by the weights in Lhotse dataloader config.
            simulator_type (str): The type of simulator to use.
                - 'lsmix': LibriSpeechMix-style training sample.
                - 'meeting': Meeting-style training sample.
                - 'conversation': Conversation-style training sample.
            speaker_distribution (list): The distribution of speakers in the simulated audio.
                The length of the list is the maximum number of speakers.
                The list elements are the weights for each speaker.
            min_delay (float): The minimum delay between speakers
                to avoid the same starting time for multiple speakers.
        """
        self.random_seed = random_seed
        self.global_rank = global_rank
        self.world_size = world_size

        self.manifest_filepath = manifest_filepath
        self.manifests = list(LazyJsonlIterator(manifest_filepath))
        self.sample_rate = sample_rate

        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_delay = min_delay
        self.simulator_type = simulator_type
        self.max_speakers = num_speakers

        print("======  simulator_type", simulator_type)

        type2simulator = {'lsmix': self.LibriSpeechMixSimulator, 'mixture_loader': self.MultiSpeakerMixtureLoader}

        self.simulator = type2simulator[simulator_type]

        if simulator_type == 'lsmix':
            self.spk2manifests = groupby(lambda x: x["speaker_id"], self.manifests)
            self.speaker_ids = list(self.spk2manifests.keys())

        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        return self.simulator()

    def LibriSpeechMixSimulator(self):
        """
        This function simulates a LibriSpeechMix-style training sample.
        Ref:
            Paper: https://arxiv.org/abs/2003.12687
            Github: https://github.com/NaoyukiKanda/LibriSpeechMix
        """
        # Sample the speakers
        sampled_speaker_ids = random.sample(self.speaker_ids, self.max_speakers)
        # Sample the cuts for each speaker
        mono_cuts = []
        for speaker_id in sampled_speaker_ids:
            manifest = random.choice(self.spk2manifests[speaker_id])
            mono_cuts.append(self._json_to_cut(manifest))
            mono_cuts[-1].supervisions.append(
                SupervisionSegment(
                    id=uuid4(),
                    recording_id=uuid4(),
                    start=0.0,
                    duration=mono_cuts[-1].duration,
                    text=mono_cuts[-1].custom['text'],
                    speaker=speaker_id,
                )
            )

        tracks = []
        offset = 0.0
        for speaker_id, mono_cut in zip(sampled_speaker_ids, mono_cuts):
            tracks.append(MixTrack(cut=deepcopy(mono_cut), type=type(mono_cut), offset=offset))
            offset += random.uniform(self.min_delay, mono_cut.duration)

        mixed_cut = MixedCut(
            id='lsmix_' + '_'.join([track.cut.id for track in tracks]) + '_' + str(uuid4()), tracks=tracks
        )

        return mixed_cut

    def MultiSpeakerMixtureLoader(self):
        """
        Load a multi-speaker mixture from the manifest,
          and generate a mixed cut with a random duration.
        The timestamps and transcript are from the seglst file,
        where the format is:
            {
                "session_id": "session_id",
                "speaker": "speaker_id",
                "words": "transcript",
                "start_time": "start_time",
                "end_time": "end_time",
                "duration": "duration"
                ...
            }
        Supervisions are generated from the seglst file and sorted by start time.
        """

        manifest = random.choice(self.manifests)
        audio_filepath = manifest['audio_filepath']
        seglst_filepath = manifest['seglst_filepath']

        supervisions = read_seglst(seglst_filepath, session_id=manifest['session_id'])
        supervisions = sorted(supervisions, key=lambda x: x.start)

        segment_offset, segment_duration = self._get_offset_and_duration(supervisions)

        json_dict = {
            'audio_filepath': audio_filepath,
            'duration': segment_duration,
            'offset': segment_offset,
            'supervisions': find_segments_from_rttm(
                recording_id=supervisions[0].recording_id,
                rttms=SupervisionSet(supervisions),
                start_after=segment_offset,
                end_before=segment_offset + segment_duration,
                adjust_offset=False,
            ),
        }
        cut = self._json_to_cut(json_dict)

        return cut

    def _get_offset_and_duration(self, supervisions):
        """
        Get a random offset and duration of the segment.
        supervisions should be sorted by start time
        """
        non_overlap_supervisions_indices = self._get_non_overlap_supervisions_indices(supervisions)
        # find the start and the end of the segment
        start_idx = random.choice(non_overlap_supervisions_indices)
        end_idx = start_idx
        offset = supervisions[start_idx].start
        for i in range(start_idx + 1, len(supervisions)):
            end_idx = i
            if supervisions[i].end - offset <= self.min_duration:
                pass
            else:
                if i in non_overlap_supervisions_indices:
                    break
        segment_offset = offset
        segment_duration = supervisions[end_idx].end - offset

        return segment_offset, segment_duration

    def _get_non_overlap_supervisions_indices(self, supervisions):
        """
        Get the indices of the non-overlapping supervisions.
        supervisions should be sorted by start time
        """
        non_overlap_supervisions_indices = []
        max_end = -1
        for i in range(len(supervisions)):
            if supervisions[i].start >= max_end:
                non_overlap_supervisions_indices.append(i)
                max_end = max(max_end, supervisions[i].end)
        return non_overlap_supervisions_indices

    def _json_to_cut(self, json_dict):
        """
        Convert a json dictionary to a Cut instance.
        """
        audio_path = json_dict["audio_filepath"]
        duration = json_dict["duration"]
        offset = json_dict.get("offset", 0.0)
        supervisions = json_dict.get("supervisions", [])
        cut = self._create_cut(
            audio_path=audio_path,
            offset=offset,
            duration=duration,
            sampling_rate=json_dict.get("sampling_rate", None),
        )
        # Note that start=0 and not start=offset because supervision's start if relative to the
        # start of the cut; and cut.start is already set to offset

        if json_dict.get("text") is not None and json_dict.get("text") != "":
            cut_text = json_dict.get("text")
        else:
            cut_text = " ".join(json_dict.get("words", []))
            if cut_text == " ":
                cut_text = ""

        cut.supervisions.extend(supervisions)
        cut.custom = json_dict
        cut.duration = duration
        return cut

    def _create_cut(
        self,
        audio_path: str,
        offset: float,
        duration: float,
        sampling_rate: int | None = None,
        channel: int = 0,
    ) -> Cut:

        recording = self._create_recording(audio_path, duration, sampling_rate)
        cut = recording.to_cut()
        if isinstance(cut.channel, list) and len(cut.channel) > 1:
            cut.channel = [channel]
        if offset is not None:
            cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
            cut.id = f"{cut.id}-{round(offset * 1e2):06d}-{round(duration * 1e2):06d}"
        return cut

    def _create_recording(
        self,
        audio_path: str,
        duration: float,
        sampling_rate: int | None = None,
    ) -> Recording:
        if sampling_rate is not None:
            # TODO(pzelasko): It will only work with single-channel audio in the current shape.
            return Recording(
                id=audio_path,
                sources=[AudioSource(type="file", channels=[0], source=audio_path)],
                sampling_rate=sampling_rate,
                num_samples=compute_num_samples(duration, sampling_rate),
                duration=duration,
                channel_ids=[0],
            )
        else:
            return Recording.from_file(audio_path)
