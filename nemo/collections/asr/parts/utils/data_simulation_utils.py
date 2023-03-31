# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import os
import shutil
from collections import defaultdict
from typing import IO, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import beta, gamma
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_ctm, write_manifest, write_text
from nemo.collections.asr.parts.utils.speaker_utils import labels_to_rttmfile
from nemo.utils import logging


def get_cleaned_base_path(output_dir: str, overwrite_output: bool = True) -> str:
    """
    Delete output directory if it exists or throw warning.

    Args:
        output_dir (str): Path to output directory
        overwrite_output (bool): If True, delete output directory if it exists

    Returns:
        basepath (str): Path to base-path directory for writing output files
    """
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        if overwrite_output:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.mkdir(output_dir)
        else:
            raise Exception("Output directory is nonempty and overwrite_output = false")
    elif not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # only add root if paths are relative
    if not os.path.isabs(output_dir):
        ROOT = os.getcwd()
        basepath = os.path.join(ROOT, output_dir)
    else:
        basepath = output_dir
    return basepath


def binary_search_alignments(
    inds: List[int], max_audio_read_sec: float, min_alignment_count: int, alignments: List[float],
) -> int:
    """
    Binary search to find the index of the alignment that satisfies the maximum audio read duration, 
    `max_audio_read_sec`. This is used to avoid reading the short audio files.
    NOTE: `offset_max` should be at least 1 to avoid feeding max=0 to random sampling function.

    Args:
        inds (list): List of indices to search from
        max_audio_read_sec (float): Maximum audio read duration
        min_alignment_count (int): Minimum number of alignments to read
        audio_manifest (dict): Dictionary containing the audio file's alignments

    Returns:
        offset_max (int) Index of the alignment that satisfies the maximum audio read duration
    """
    # Start from the left end (at index 0) and -1 * min_alignment_count for the right end
    left, right = 0, len(inds) - 1 - min_alignment_count
    while left < right:
        mid = left + (right - left) // 2
        dur_left = alignments[-1 * min_alignment_count] - alignments[inds[mid]]
        if dur_left < max_audio_read_sec:
            right = mid - 1
        elif dur_left > max_audio_read_sec:
            left = mid + 1
        else:
            break
    mid_out = left + (right - left) // 2
    # If mid_out is on the boundary, move it to the left.
    if alignments[-1 * min_alignment_count] - alignments[inds[mid_out]] < max_audio_read_sec:
        mid_out -= 1
    offset_max = max(mid_out, 1)
    return offset_max


def get_subset_of_audio_manifest(
    audio_manifest: dict, offset_index: int, max_audio_read_sec: float, min_alignment_count: int,
) -> dict:
    """
    Get a subset of `audio_manifest` for faster audio-file reading.

    Args:
        audio_manifest (dict): Audio manifest dictionary.
            keys: 'offset', 'duration', 'alignments', 'words'
        offset_index (int): Index of the offset.
        max_audio_read_sec (float): Maximum audio read duration.
        min_alignment_count (int): Minimum number of alignments to read.

    Returns:
        audio_manifest (dict): Subset of `audio_manifest` is returned for `words` and `alignments` keys.
    """
    alignment_array = np.array(audio_manifest['alignments'])
    alignment_array_pr = np.array(alignment_array[offset_index:]) - alignment_array[offset_index]
    subset_alignments = alignment_array_pr[alignment_array_pr < max_audio_read_sec]
    if len(subset_alignments) < min_alignment_count:
        # Cases where the word next to the offset is longer than the max_audio_read_sec.
        logging.warning(
            f"subset_alignments of {audio_manifest['audio_filepath']} \n"
            f"has subset alignment length:{len(subset_alignments)} at offset_index:{offset_index}, "
            f"word:{audio_manifest['words'][offset_index:offset_index+min_alignment_count]}, "
            f"alignments:{alignment_array_pr[:min_alignment_count]} which is longer than _max_audio_read_sec:{[0, max_audio_read_sec]}."
            " Truncating the alignements."
        )
        # Attach the `_max_audio_read_sec` to the `subset_alignments` to truncate the alignment timestamp.
        subset_alignments = np.concatenate([subset_alignments, np.array([max_audio_read_sec])])
    audio_manifest['offset'], audio_manifest['duration'] = (
        alignment_array[offset_index],
        subset_alignments[-1] - subset_alignments[0],
    )
    audio_manifest['alignments'] = subset_alignments.tolist()
    audio_manifest['words'] = audio_manifest['words'][offset_index : offset_index + len(subset_alignments)]
    return audio_manifest


def read_audio_from_buffer(
    audio_manifest: dict,
    buffer_dict: dict,
    offset_index: int,
    device: torch.device,
    max_audio_read_sec: float = 2.5,
    min_alignment_count: int = 2,
    read_subset: bool = True,
) -> Tuple[torch.Tensor, int, dict]:
    """
    Read from the provided file path while maintaining a hash-table that saves loading time.
    Also, this function only reads a subset of the audio file if `read_subset` is True for faster audio-file reading.

    Args:
        audio_manifest (dict): Audio manifest dictionary.
            keys: 'audio_filepath', 'duration', 'alignments', 'words'
        buffer_dict (dict): Hash-table that saves loaded audio files.
        offset_index (int): Index of the offset for the audio file.
        device (torch.device): Device to load the audio file.
        max_audio_read_sec (float): Maximum audio read duration.
        min_alignment_count (int): Minimum number of alignments to read.
        read_subset (bool): If True, read a subset of the audio file.
                            To control the length of the audio file, use data_simulator.session_params.max_audio_read_sec.
                            Note that using large value (greater than 3~4 sec) for `max_audio_read_sec` will slow down the generation process.
                            If False, read the entire audio file.

    Returns:
        audio_file (torch.Tensor): Time-series audio data in a tensor.
        sr (int): Sample rate of the audio file.
        audio_manifest (dict): (modified) audio manifest dictionary.
    """
    audio_file_id = f"{audio_manifest['audio_filepath']}#{offset_index}"
    if audio_file_id in buffer_dict:
        audio_file, sr, audio_manifest = buffer_dict[audio_file_id]
    else:
        if read_subset:
            audio_manifest = get_subset_of_audio_manifest(
                audio_manifest=audio_manifest,
                offset_index=offset_index,
                max_audio_read_sec=max_audio_read_sec,
                min_alignment_count=min_alignment_count,
            )
            segment = AudioSegment.from_file(
                audio_file=audio_manifest['audio_filepath'],
                offset=audio_manifest['offset'],
                duration=audio_manifest['duration'],
            )
        else:
            segment = AudioSegment.from_file(audio_file=audio_manifest['audio_filepath'])
        audio_file, sr = torch.from_numpy(segment.samples).to(device), segment.sample_rate
        if read_subset and segment.duration < (audio_manifest['alignments'][-1] - audio_manifest['alignments'][0]):
            audio_manifest['alignments'][-1] = min(segment.duration, audio_manifest['alignments'][-1])
        if audio_file.ndim > 1:
            audio_file = torch.mean(audio_file, 1, False).to(device)
        buffer_dict[audio_file_id] = (audio_file, sr, audio_manifest)
    return audio_file, sr, audio_manifest


def perturb_audio(
    audio: torch.Tensor, sr: int, augmentor: Optional[AudioAugmentor] = None, device: Optional[torch.device] = None
) -> torch.Tensor:

    """
    Perturb the audio (segment or session) using audio augmentor.

    Args:
        audio (torch.Tensor): Time-series signal of the segment
        sr (int): Sample rate of the original audio file
        augmentor (AudioAugmentor): Audio augmentor to use
        device (torch.device): Device to load the audio file

    Returns:
        audio (torch.Tensor): Perturbed audio (time-series signal) of the segment
    """
    if augmentor is None:
        return audio
    device = device if device is not None else torch.device('cpu')
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    audio_segment = AudioSegment(audio, sample_rate=sr)
    augmentor.perturb(audio_segment)
    audio_segment = torch.from_numpy(audio_segment.samples).to(device)
    return audio_segment


def normalize_audio(array: torch.Tensor) -> torch.Tensor:
    """
    Normalize the audio signal to avoid clipping.

    Args:
        array (torch.Tensor): Time-series audio data in a tensor.

    Returns:
        (torch.Tensor): Normalized audio signal.
    """
    return array / (1.0 * torch.max(torch.abs(array)))


def get_power_of_audio_file(audio_file: str, end_audio_file: int, running_len_samples: int, device: torch.device):
    """
    Calculate the power of the audio signal.

    Args:
        audio_file (torch.Tensor): Time-series audio data in a tensor.
        end_audio_file (int): End index of the audio file.
        running_len_samples (int): Running length of the audio file.
        device (torch.device): Device to use.

    Returns:
        (float): Power of the audio signal.
    """
    return torch.mean(audio_file[: end_audio_file - running_len_samples] ** 2).to(device)


def get_scaled_audio_signal(
    audio_file: torch.Tensor,
    end_audio_file: int,
    running_len_samples: int,
    desired_avg_power_noise: float,
    device: torch.device,
):
    """
    Scale the audio signal to the desired average power.

    Args:
        audio_file (torch.Tensor): Time-series audio data in a tensor.
        end_audio_file (int): End index of the audio file.
        running_len_samples (int): Running length of the audio file.
        desired_avg_power_noise (float): Desired average power of the audio file.
        device (torch.device): Device to use.

    Returns:
        scaled_audio_file (torch.Tensor): Scaled audio signal.
    """
    pow_audio_file = get_power_of_audio_file(
        audio_file=audio_file, end_audio_file=end_audio_file, running_len_samples=running_len_samples, device=device
    )
    scaled_audio_file = audio_file[: end_audio_file - running_len_samples] * torch.sqrt(
        desired_avg_power_noise / pow_audio_file
    ).to(device)
    return scaled_audio_file


def get_desired_avg_power_noise(
    power_array: float, snr_min: float, snr_max: float, background_noise_snr: float,
):
    """
    Calculate the desired average power of the noise.

    Args:
        power_array (float): Power of the audio signal.
        snr_min (float): Minimum SNR.
        snr_max (float): Maximum SNR.
        background_noise_snr (float): SNR of the background noise.

    Returns:
        desired_avg_power_noise (float): Desired average power of the noise.
    """
    if (snr_min is not None) and (snr_max is not None) and (snr_min <= snr_max):
        desired_snr = np.random.uniform(snr_min, snr_max)
    else:
        desired_snr = background_noise_snr
    ratio = 10 ** (desired_snr / 20)
    desired_avg_power_noise = power_array / ratio
    return desired_avg_power_noise, desired_snr


def get_background_noise(
    len_array: int,
    power_array: float,
    noise_samples: list,
    audio_read_buffer_dict: dict,
    snr_min: float,
    snr_max: float,
    background_noise_snr: float,
    seed: int,
    device: torch.device,
):
    """
    Augment with background noise (inserting ambient background noise up to the desired SNR for the full clip).

    Args:
        len_array (int): Length of background noise required.
        power_array (float): Power of the audio signal.
        noise_samples (list): List of noise samples.
        audio_read_buffer_dict (dict): Dictionary containing audio read buffer.
        snr_min (float): Minimum SNR.
        snr_max (float): Maximum SNR.
        background_noise_snr (float): SNR of the background noise.
        seed (int): Seed for random number generator.
        device (torch.device): Device to use.
    
    Returns:
        bg_array (tensor): Tensor containing background noise.
        desired_snr (float): Desired SNR for adding background noise.
    """
    np.random.seed(seed)
    bg_array = torch.zeros(len_array).to(device)
    desired_avg_power_noise, desired_snr = get_desired_avg_power_noise(
        power_array=power_array, snr_min=snr_min, snr_max=snr_max, background_noise_snr=background_noise_snr
    )
    running_len_samples = 0

    while running_len_samples < len_array:  # build background audio stream (the same length as the full file)
        file_id = np.random.randint(len(noise_samples))
        audio_file, sr, audio_manifest = read_audio_from_buffer(
            audio_manifest=noise_samples[file_id],
            buffer_dict=audio_read_buffer_dict,
            offset_index=0,
            device=device,
            read_subset=False,
        )
        if running_len_samples + len(audio_file) < len_array:
            end_audio_file = running_len_samples + len(audio_file)
        else:
            end_audio_file = len_array
        scaled_audio_file = get_scaled_audio_signal(
            audio_file=audio_file,
            end_audio_file=end_audio_file,
            running_len_samples=running_len_samples,
            desired_avg_power_noise=desired_avg_power_noise,
            device=device,
        )

        bg_array[running_len_samples:end_audio_file] = scaled_audio_file
        running_len_samples = end_audio_file

    return bg_array, desired_snr


def get_random_offset_index(
    audio_manifest: dict,
    audio_read_buffer_dict: dict,
    offset_min: int = 0,
    max_audio_read_sec: float = 2.5,
    min_alignment_count: int = 2,
) -> int:
    """
    Get an index for randomly accessing the silence in alignment timestamps. 

    Args:
        audio_manifest (dict): Audio manifest dictionary.
            keys: 'audio_filepath', 'duration', 'alignments', 'words'
        audio_read_buffer_dict (dict): Dictionary containing audio read buffer.
        offset_min (int): Minimum offset index. (Default: 0)
        max_audio_read_sec (float): Maximum audio read duration in seconds. (Default: 2.5)
        min_alignment_count (int): Minimum number of alignment timestamps. (Default: 2)

    Returns: 
        (int): Random offset index smaller than `offset_count`.
    """
    if len(audio_manifest['alignments']) <= min_alignment_count:
        raise ValueError(
            f"Audio file {audio_manifest['audio_filepath']} has less than {min_alignment_count} alignment timestamps."
        )
    index_file_id = f"{audio_manifest['audio_filepath']}#index"

    # Avoid multiple indexings of the same audio file by using a hash-table.
    if index_file_id in audio_read_buffer_dict:
        (sil_inds, offset_max) = audio_read_buffer_dict[index_file_id]
    else:
        # Find all silence indices
        sil_inds = np.where((np.array(audio_manifest['words']) == '') == True)[0]
        if audio_manifest['alignments'][-1] - audio_manifest['alignments'][0] < max_audio_read_sec:
            # The total duration is already short, therefore skip range search.
            offset_max = 1
        else:
            # Find the range that satisfies `max_audio_read_sec` duration.
            offset_max = binary_search_alignments(
                inds=sil_inds,
                max_audio_read_sec=max_audio_read_sec,
                min_alignment_count=min_alignment_count,
                alignments=audio_manifest['alignments'],
            )

        audio_read_buffer_dict[index_file_id] = (sil_inds, offset_max)

    # If the audio file is shorter than the max_audio_read_sec, then we don't need to read a subset of the audio file.
    if (
        len(sil_inds) <= min_alignment_count
        or (audio_manifest['alignments'][-1] - audio_manifest['alignments'][0]) < max_audio_read_sec
    ):
        return offset_min
    else:
        offset_index = np.random.randint(offset_min, offset_max)
        return sil_inds[offset_index]


def get_speaker_ids(sess_idx: int, speaker_samples: dict, permutated_speaker_inds: list) -> List[str]:
    """
    Randomly select speaker IDs from the loaded manifest file.

    Args:
        sess_idx (int): Session index in integer.
        speaker_samples (dict): Dictionary mapping speaker ID to their list of samples.
        permutated_speaker_inds (list): List of permutated speaker indices.

    Returns:
        speaker_ids (list): List of speaker IDs
    """
    all_speaker_ids = list(speaker_samples.keys())
    idx_list = permutated_speaker_inds[sess_idx, :]
    speaker_ids = [all_speaker_ids[i] for i in idx_list]
    return speaker_ids


def build_speaker_samples_map(manifest: dict) -> dict:
    """
    Build a dictionary for mapping speaker ID to their list of samples

    Returns:
        speaker_samples (Dict[list]):
            Dictionary mapping speaker ID to their list of samples
    """
    speaker_samples = defaultdict(list)
    logging.info("Building speaker to samples map...")
    for sample in tqdm(manifest, total=len(manifest)):
        speaker_id = sample['speaker_id']
        speaker_samples[speaker_id].append(sample)
    return speaker_samples


def read_noise_manifest(add_bg: bool, background_manifest: str):
    """
    Read the noise manifest file and sample the noise manifest.

    Args:
        add_bg (bool): Whether to add background noise.
        background_manifest (str): Path to the background noise manifest file.

    Returns:
        noise_manifest (list): List of the entire noise source samples.
    """
    noise_manifest = []
    if add_bg is True:
        if background_manifest is not None:
            background_manifest_list = background_manifest
            if isinstance(background_manifest_list, str):
                background_manifest_list = [background_manifest_list]
            for background_manifest in background_manifest_list:
                if os.path.exists(background_manifest):
                    noise_manifest += read_manifest(background_manifest)
                else:
                    raise FileNotFoundError(f"Noise manifest file: {background_manifest} file not found.")
        else:
            raise FileNotFoundError(
                f"Noise manifest file is {background_manifest}. Please provide a valid noise manifest file/list if add_bg=True."
            )
    return noise_manifest


def get_speaker_samples(speaker_ids: List[str], speaker_samples: dict) -> Dict[str, list]:
    """
    Get a list of the samples for each of the specified speakers.

    Args:
        speaker_ids (list): LibriSpeech speaker IDs for each speaker in the current session.
        speaker_samples (dict): Dictionary mapping speaker ID to their list of samples.
    
    Returns:
        speaker_wav_align_map (dict): Dictionary containing speaker IDs and their corresponding wav filepath and alignments.
    """
    speaker_wav_align_map = defaultdict(list)
    for sid in speaker_ids:
        speaker_wav_align_map[sid] = speaker_samples[sid]
    return speaker_wav_align_map


def add_silence_to_alignments(audio_manifest: dict):
    """
    Add silence to the beginning of the alignments and words.

    Args:
        audio_manifest (dict): Audio manifest dictionary.
            keys: 'audio_filepath', 'duration', 'alignments', 'words'

    Returns:
        audio_manifest (dict): Audio manifest dictionary with silence added to the beginning.
    """
    if type(audio_manifest['words'][0]) == str and len(audio_manifest['words'][0]) > 0:
        audio_manifest['words'].insert(0, "")
        audio_manifest['alignments'].insert(0, 0.0)
    return audio_manifest


def load_speaker_sample(
    speaker_wav_align_map: List[dict], speaker_ids: List[str], speaker_turn: int, min_alignment_count: int,
) -> str:
    """
    Load a sample for the selected speaker ID.
    The first alignment and word must be silence that determines the start of the alignments.

    Args:
        speaker_wav_align_map (dict): Dictionary containing speaker IDs and their corresponding wav filepath and alignments.
        speaker_ids (list): LibriSpeech speaker IDs for each speaker in the current session.
        speaker_turn (int): Current speaker turn.
        output_precision (int): Precision of the output alignments in integer.
        min_alignment_count (int): Minimum number of alignments in the audio file.
    
    Returns:
        audio_manifest (dict): Audio manifest dictionary containing the wav filepath, words and alignments.
    """
    speaker_id = speaker_ids[speaker_turn]
    file_id = np.random.randint(0, max(len(speaker_wav_align_map[str(speaker_id)]) - 1, 1))
    audio_manifest = speaker_wav_align_map[str(speaker_id)][file_id]

    # Check if the alignment file has at least 2 words.
    if len(audio_manifest['alignments']) < min_alignment_count:
        raise ValueError(
            f"Alignment file {audio_manifest['audio_filepath']} has an inappropriate length of {len(audio_manifest['alignments'])} < 2."
        )

    # Check whether the first word is silence and insert a silence token if the first token is not silence.
    if audio_manifest['words'][0] != "":
        audio_manifest = add_silence_to_alignments(audio_manifest)

    audio_manifest = copy.deepcopy(audio_manifest)
    return audio_manifest


def get_split_points_in_alignments(
    words: List[str],
    alignments: List[float],
    split_buffer: float,
    sr: int,
    sentence_audio_len: int,
    new_start: float = 0,
):
    """
    Collect split points in the alignment based on silence.
    Silence is defined as a blank symbol between two words that is longer than 2 * split_buffer.

    Args:
        words (List[str]): List of words in the sentence.
        alignments (List[float]): List of alignment timestamps in the sentence.
        split_buffer (float): Buffer length in seconds.
        sr (int): Sample rate of the audio.
        sentence_audio_len (int): Length of the sentence audio in samples.
        new_start (float): Start of the sentence audio in seconds.

    Returns:
        splits (List[List[int]]): List of integer split points in the sentence audio.
    """
    splits = []
    for i in range(len(words)):
        if words[i] == "" and i != 0 and i != len(words) - 1:
            silence_length = alignments[i] - alignments[i - 1]
            if silence_length > 2 * split_buffer:  # split utterance on silence
                new_end = alignments[i - 1] + split_buffer
                splits.append(
                    [int(new_start * sr), int(new_end * sr),]
                )
                new_start = alignments[i] - split_buffer
    # The last split point should be added
    splits.append([int(new_start * sr), sentence_audio_len])
    return splits


def per_speaker_normalize(
    sentence_audio: torch.Tensor, splits: List[List[int]], speaker_turn: int, volume: List[float], device: torch.device
) -> torch.Tensor:
    """
    Normalize time-series audio signal per speaker.

    Args:
        sentence_audio (torch.Tensor): Time-series audio signal.
        splits (List[List[int]]): List of integer split points in the sentence audio.
        speaker_turn (int): Speaker ID of the current speaker.
        volume (List[float]): List of volume levels for each speaker.
        device (torch.device): Device to use for computations.

    Returns:
        sentence_audio (torch.Tensor): Normalized time-series audio signal.
    """
    split_length = torch.tensor(0).to(device).double()
    split_sum = torch.tensor(0).to(device).double()
    for split in splits:
        split_length += len(sentence_audio[split[0] : split[1]])
        split_sum += torch.sum(sentence_audio[split[0] : split[1]] ** 2)
    average_rms = torch.sqrt(split_sum * 1.0 / split_length)
    sentence_audio = sentence_audio / (1.0 * average_rms) * volume[speaker_turn]
    return sentence_audio


class DataAnnotator(object):
    """
    Class containing the functions that create RTTM, CTM, JSON files.

    Arguments in config:
    
    data_simulator:
        session_config:
            num_speakers (int): Number of unique speakers per multispeaker audio session
            session_params:
            split_buffer (float): Split RTTM labels if greater than twice this amount of silence (to avoid long gaps between 
                                    utterances as being labelled as speech)
        outputs:
            output_dir (str): Output directory for audio sessions and corresponding label files
            output_filename (str): Output filename for the wav and RTTM files
            overwrite_output (bool): If true, delete the output directory if it exists
            output_precision (int): Number of decimal places in output files
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: OmegaConf configuration loaded from yaml file.
        """
        self._params = cfg
        self._files = {}
        self._init_file_write()
        self._init_filelist_lists()

    def _init_file_write(self):
        """
        Initialize file writing arguments
        """
        self._file_base_str = "synthetic"
        self._file_types = ["wav", "rttm", "json", "ctm", "txt", "meta"]
        self._annotation_types = ["rttm", "json", "ctm"]

    def _init_filelist_lists(self):
        """
        Initialize lists to store the filelists for each file type
        """
        self.annote_lists = {}
        for file_type in self._file_types:
            self.annote_lists[f"{file_type}_list"] = []

    def init_annotation_lists(self):
        """
        Initialize lists to store the annotations for each file type
        """
        for file_type in self._file_types:
            self.annote_lists[file_type] = []

    def create_new_rttm_entry(
        self, words: List[str], alignments: List[float], start: int, end: int, speaker_id: int
    ) -> List[str]:

        """
        Create new RTTM entries (to write to output rttm file)

        Args:
            words (list): List of words in the current audio file.
            alignments (list): List of alignments (timestamps) for the current audio file.
            start (int): Current start of the audio file being inserted.
            end (int): End of the audio file being inserted.
            speaker_id (int): LibriSpeech speaker ID for the current entry.
        
        Returns:
            rttm_list (list): List of rttm entries
        """
        rttm_list = []
        new_start = start
        # look for split locations
        for i in range(len(words)):
            if words[i] == "" and i != 0 and i != len(words) - 1:
                silence_length = alignments[i] - alignments[i - 1]
                if (
                    silence_length > 2 * self._params.data_simulator.session_params.split_buffer
                ):  # split utterance on silence
                    new_end = start + alignments[i - 1] + self._params.data_simulator.session_params.split_buffer
                    t_stt = round(float(new_start), self._params.data_simulator.outputs.output_precision)
                    t_end = round(float(new_end), self._params.data_simulator.outputs.output_precision)
                    rttm_list.append(f"{t_stt} {t_end} {speaker_id}")
                    new_start = start + alignments[i] - self._params.data_simulator.session_params.split_buffer

        t_stt = round(float(new_start), self._params.data_simulator.outputs.output_precision)
        t_end = round(float(end), self._params.data_simulator.outputs.output_precision)
        rttm_list.append(f"{t_stt} {t_end} {speaker_id}")
        return rttm_list

    def create_new_json_entry(
        self,
        text: List[str],
        wav_filename: str,
        start: float,
        length: float,
        speaker_id: int,
        rttm_filepath: str,
        ctm_filepath: str,
    ) -> dict:
        """
        Create new JSON entries (to write to output json file).

        Args:
            text (list): string of text for the current entry.
            wav_filename (str): Filename of the wav file.
            start (float): Start time of the current entry.
            length (float): Length of the current entry.
            speaker_id (int): speaker ID for the current entry.
            rttm_filepath (str): Path to the RTTM file.
            ctm_filepath (str): Path to the CTM file.

        Returns:
            meta (dict): JSON entry dictionary.
        """
        start = round(float(start), self._params.data_simulator.outputs.output_precision)
        length = round(float(length), self._params.data_simulator.outputs.output_precision)
        meta = {
            "audio_filepath": wav_filename,
            "offset": start,
            "duration": length,
            "label": speaker_id,
            "text": text,
            "num_speakers": self._params.data_simulator.session_config.num_speakers,
            "rttm_filepath": rttm_filepath,
            "ctm_filepath": ctm_filepath,
            "uem_filepath": None,
        }
        return meta

    def create_new_ctm_entry(
        self, words: List[str], alignments: List[float], session_name: str, speaker_id: int, start: int
    ) -> List[str]:
        """
        Create new CTM entry (to write to output ctm file)

        Args:
            words (list): List of words in the current audio file.
            alignments (list): List of alignments (timestamps) for the current audio file.
            session_name (str): Current session name.
            speaker_id (int): LibriSpeech speaker ID for the current entry.
            start (int): Current start of the audio file being inserted.
        
        Returns:
            arr (list): List of ctm entries
        """
        arr = []
        start = float(round(start, self._params.data_simulator.outputs.output_precision))
        for i in range(len(words)):
            word = words[i]
            if (
                word != ""
            ):  # note that using the current alignments the first word is always empty, so there is no error from indexing the array with i-1
                prev_align = 0 if i == 0 else alignments[i - 1]
                align1 = round(float(prev_align + start), self._params.data_simulator.outputs.output_precision)
                align2 = round(float(alignments[i] - prev_align), self._params.data_simulator.outputs.output_precision)
                text = f"{session_name} {speaker_id} {align1} {align2} {word} 0\n"
                arr.append((align1, text))
        return arr

    def add_to_filename_lists(self, basepath: str, filename: str):
        """
        Add the current filename to the list of filenames for each file type.

        Args:
            basepath (str): Basepath for output files.
            filename (str): Base filename for all output files.
        """
        full_base_filepath = os.path.join(basepath, filename)
        for file_type in self._file_types:
            self.annote_lists[f"{file_type}_list"].append(f"{full_base_filepath}.{file_type}")

    def write_filelist_files(self, basepath):
        """
        Write all filelist files.

        Args:
            basepath (str): Basepath for output files.
        """
        for file_type in self._file_types:
            with open(f"{basepath}/{self._file_base_str}_{file_type}.list", "w") as list_file:
                list_file.write("\n".join(self.annote_lists[f"{file_type}_list"]))
            list_file.close()

    def write_annotation_files(self, basepath: str, filename: str, meta_data: dict):
        """
        Write all annotation files: RTTM, JSON, CTM, TXT, and META.

        Args:
            basepath (str): Basepath for output files.
            filename (str): Base filename for all output files.
            meta_data (dict): Metadata for the current session.
            rttm_list (list): List of RTTM entries.
            json_list (list): List of JSON entries.
            ctm_list (list): List of CTM entries.
        """
        labels_to_rttmfile(self.annote_lists['rttm'], filename, self._params.data_simulator.outputs.output_dir)
        write_manifest(os.path.join(basepath, filename + '.json'), self.annote_lists['json'])
        write_ctm(os.path.join(basepath, filename + '.ctm'), self.annote_lists['ctm'])
        write_text(os.path.join(basepath, filename + '.txt'), self.annote_lists['ctm'])
        write_manifest(os.path.join(basepath, filename + '.meta'), [meta_data])


class SpeechSampler(object):
    """
    Class for sampling speech samples for Multispeaker Audio Session Simulator 

    Args:
        cfg: OmegaConf configuration loaded from yaml file.

    Variables for sampling speech:
        self.running_speech_len_samples (int): Running total of speech samples in the current audio session.
        self.running_silence_len_samples (int): Running total of silence samples in the current audio session.
        self.running_overlap_len_samples (int): Running total of overlap samples in the current audio session.

        self.sess_silence_mean (int) : Targeted mean number of silence samples in the current audio session.
        self.per_silence_min_len (int): Minimum number of silence samples in the silence segment.
        self.per_silence_max_len (int): Maximum number of silence samples in the silence segment.

        self.sess_overlap_mean (int): Targeted mean number of overlap samples in the current audio session.
        self.per_overlap_min_len (int): Minimum number of overlap samples in the overlap segment.
        self.per_overlap_max_len (int): Maximum number of overlap samples in the overlap segment.

    data_simulator: 
        session_params: 
            mean_silence (float): Mean proportion of silence to speaking time in the audio session. Should be in range [0, 1).
            mean_silence_var (float): Variance for mean silence in all audio sessions. 
                                    This value should be 0 <= mean_silence_var < mean_silence * (1 - mean_silence).
            per_silence_var (float):  Variance for each silence in an audio session, set large values (e.g., 20) for de-correlation.
            per_silence_min (float): Minimum duration for each silence, default to 0.
            per_silence_max (float): Maximum duration for each silence, default to -1 for no maximum.
            
            mean_overlap (float): Mean proportion of overlap in the overall non-silence duration. Should be in range [0, 1) and 
                                recommend [0, 0.15] range for accurate results.
            mean_overlap_var (float): Variance for mean overlap in all audio sessions. 
                                    This value should be 0 <= mean_overlap_var < mean_overlap * (1 - mean_overlap).
            per_overlap_var (float): Variance for per overlap in each session, set large values to de-correlate silence lengths 
                                    with the latest speech segment lengths
            per_overlap_min (float): Minimum per overlap duration in seconds
            per_overlap_max (float): Maximum per overlap duration in seconds, set -1 for no maximum 
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: OmegaConf configuration loaded from yaml file.
        """
        self._params = cfg

        self.running_speech_len_samples = 0
        self.running_silence_len_samples = 0
        self.running_overlap_len_samples = 0

        self.sess_silence_mean = None
        self.per_silence_min_len = 0
        self.per_silence_max_len = 0

        self.sess_overlap_mean = None
        self.per_overlap_min_len = 0
        self.per_overlap_max_len = 0

        self.mean_overlap = float(self._params.data_simulator.session_params.mean_overlap)
        self.mean_overlap_var = float(self._params.data_simulator.session_params.mean_overlap_var)

        self.mean_silence = float(self._params.data_simulator.session_params.mean_silence)
        self.mean_silence_var = float(self._params.data_simulator.session_params.mean_silence_var)

        self.per_silence_var = float(self._params.data_simulator.session_params.per_silence_var)
        self.per_overlap_var = float(self._params.data_simulator.session_params.per_overlap_var)

        self.num_noise_files = int(self._params.data_simulator.background_noise.num_noise_files)

    def _mean_var_to_a_and_b(self, mean: float, var: float) -> Tuple[float, float]:
        """
        Convert mean and variance to a and b parameters for beta distribution.

        Args:
            mean (float): Mean of the beta distribution.
            var (float): Variance of the beta distribution.

        Returns:
            Tuple[float, float]: a and b parameters for beta distribution.
        """
        a = mean ** 2 * (1 - mean) / var - mean
        b = mean * (1 - mean) ** 2 / var - (1 - mean)
        return a, b

    def _init_silence_params(self):
        """
        Initialize parameters for silence insertion in the current session.
        """
        self.running_speech_len_samples = 0
        self.running_silence_len_samples = 0

        self.per_silence_min_len = int(
            max(0, self._params.data_simulator.session_params.per_silence_min) * self._params.data_simulator.sr
        )
        if self._params.data_simulator.session_params.per_silence_max > 0:
            self.per_silence_max_len = int(
                self._params.data_simulator.session_params.per_silence_max * self._params.data_simulator.sr
            )
        else:
            self.per_silence_max_len = int(
                self._params.data_simulator.session_config.session_length * self._params.data_simulator.sr
            )

    def _init_overlap_params(self):
        """
        Initialize parameters for overlap insertion in the current session.
        """
        self.running_overlap_len_samples = 0

        self.per_overlap_min_len = int(
            max(0, self._params.data_simulator.session_params.per_overlap_min) * self._params.data_simulator.sr
        )
        if self._params.data_simulator.session_params.per_overlap_max > 0:
            self.per_overlap_max_len = int(
                self._params.data_simulator.session_params.per_overlap_max * self._params.data_simulator.sr
            )
        else:
            self.per_overlap_max_len = int(
                self._params.data_simulator.session_config.session_length * self._params.data_simulator.sr
            )

    def silence_vs_overlap_selector(self, running_len_samples: int, non_silence_len_samples: int) -> bool:
        """
        Compare the current silence ratio to the current overlap ratio. Switch to either silence or overlap mode according 
        to the amount of the gap between current ratio and session mean in config.

        Args:
            running_len_samples (int): Length of the current session in samples.
            non_silence_len_samples (int): Length of the signal that is not silence in samples.

        Returns:
            add_overlap (bool): True if the current silence ratio is less than the current overlap ratio, False otherwise.
        """
        if running_len_samples > 0:
            self.current_silence_ratio = (running_len_samples - self.running_speech_len_samples) / running_len_samples
            self.current_overlap_ratio = self.running_overlap_len_samples / non_silence_len_samples
        else:
            self.current_silence_ratio, self.current_overlap_ratio = 0, 0

        # self.silence_discrepancy = max(0, self.sess_silence_mean - self.current_silence_ratio)
        # self.overlap_discrepancy = max(0, self.sess_overlap_mean - self.current_overlap_ratio)
        # threshold = self.silence_discrepancy / (self.overlap_discrepancy + self.silence_discrepancy + 1e-10)
        # add_overlap = np.random.rand() > threshold
        self.silence_discrepancy = self.current_silence_ratio - self.sess_silence_mean
        self.overlap_discrepancy = self.current_overlap_ratio - self.sess_overlap_mean
        add_overlap = bool(self.overlap_discrepancy < self.silence_discrepancy)
        return add_overlap

    def get_session_silence_mean(self):
        """
        Get the target mean silence for current session using re-parameterized Beta distribution.
        The following constraints are applied to make a > 0 and b > 0:

            0 < mean_silence < 1
            0 < mean_silence_var < mean_silence * (1 - mean_silence)

        Args:
            silence_mean (float): 
                Target mean silence for the current session
        """
        self._init_silence_params()
        mean, var = self.mean_silence, self.mean_silence_var
        if var > 0:
            a, b = self._mean_var_to_a_and_b(mean, var)
            if a < 0 or b < 0:
                raise ValueError(
                    f"Beta(a, b), a = {a:.3f} and b = {b:.3f} should be both greater than 0. "
                    f"Invalid `mean_silence_var` value {var} for sampling from Beta distribution. "
                    f"`mean_silence_var` should be less than `mean_silence * (1 - mean_silence)`. "
                    f"Please check `mean_silence_var` and try again."
                )
            self.sess_silence_mean = beta(a, b).rvs()
        else:
            self.sess_silence_mean = mean
        return self.sess_silence_mean

    def get_session_overlap_mean(self):
        """
        Get the target mean overlap for current session using re-parameterized Beta distribution.
        The following constraints are applied to make a > 0 and b > 0:

            0 < mean_overlap < 1
            0 < mean_overlap_var < mean_overlap * (1 - mean_overlap)

        Returns:
            overlap_mean (float):
                Target mean overlap for the current session
        """
        self._init_overlap_params()
        mean, var = self.mean_overlap, self.mean_overlap_var
        if var > 0:
            a, b = self._mean_var_to_a_and_b(mean, var)
            if a < 0 or b < 0:
                raise ValueError(
                    f"Beta(a, b), a = {a:.3f} and b = {b:.3f} should be both greater than 0. "
                    f"Invalid `mean_overlap_var` value {var} for sampling from Beta distribution. "
                    f"`mean_overlap_var` should be less than `mean_overlap * (1 - mean_overlap)`. "
                    f"Please check `mean_overlap_var` and try again."
                )
            self.sess_overlap_mean = beta(a, b).rvs()
        else:
            self.sess_overlap_mean = mean
        return self.sess_overlap_mean

    def sample_from_silence_model(self, running_len_samples: int) -> int:
        """
        Sample from the silence model to determine the amount of silence to add between sentences.
        Gamma distribution is employed for modeling the highly skewed distribution of silence length distribution.
        When we add silence between sentences, we want to ensure that the proportion of silence meets the `sess_silence_mean`.
        Thus, [Session Silence Mean] = [Total Running Silence Time] / [Total Running Session Time] equation holds. We employ the following 
        formula to determine the amount of silence to add, which is `silence_mean`:

            self.sess_silence_mean = (silence_mean + self.running_silence_len_samples) / (silence_mean + running_len_samples)

        The above equation is setting `silence_mean` to yield the desired silence ratio `self.sess_silence_mean`. 
        We use the above `silence_mean` value to sample silence-length for each silence occurrence.

        Args:
            running_len_samples (int): 
                Running length of the session (in terms of number of samples).
            session_len_samples (int):
                Targeted total session length (in terms of number of samples).

        Returns:
            silence_amount (int): Amount of silence to add between sentences (in terms of number of samples).
        """
        silence_mean = ((self.sess_silence_mean * running_len_samples) - self.running_silence_len_samples) / (
            1 - self.sess_silence_mean
        )
        silence_mean = max(self.per_silence_min_len, min(silence_mean, self.per_silence_max_len))
        if silence_mean > 0:
            self.per_silence_var = self._params.data_simulator.session_params.per_silence_var
            silence_amount = (
                int(
                    gamma(
                        a=(silence_mean ** 2) / self.per_silence_var, scale=self.per_silence_var / silence_mean
                    ).rvs()
                )
                if self.per_silence_var > 0
                else int(silence_mean)
            )
            silence_amount = max(self.per_silence_min_len, min(silence_amount, self.per_silence_max_len))
        else:
            silence_amount = 0
        return silence_amount

    def sample_from_overlap_model(self, non_silence_len_samples: int):
        """
        Sample from the overlap model to determine the amount of overlap between segments.
        Gamma distribution is employed for modeling  the highly skewed distribution of overlap length distribution.
        When we add an overlap occurrence, we want to meet the desired overlap ratio defined by `self.sess_overlap_mean`.
        Thus, [Session Overlap Mean] = [Total Running Overlap Speech Time] / [Total Running Non-Silence Speech Time].
        Let `overlap_mean` be the desired overlap amount, then the mean and variance of the gamma distribution is given by:

            self.sess_overlap_mean = (overlap_mean + self.running_overlap_len_samples) / (non_silence_len_samples - overlap_mean)

        The above equation is setting `overlap_mean` to yield the desired overlap ratio `self.sess_overlap_mean`. 
        We use the above `overlap_mean` value to sample overlap-length for each overlap occurrence.
        
        Args:
            non_silence_len_samples (int): 
                The total amount of non-silence (speech) region regardless of overlap status

        Returns:
            desired_overlap_amount (int): 
                Amount of overlap between segments (in terms of number of samples).
        """
        overlap_mean = ((self.sess_overlap_mean * non_silence_len_samples) - self.running_overlap_len_samples) / (
            1 + self.sess_overlap_mean
        )
        overlap_mean = max(self.per_overlap_min_len, min(max(0, overlap_mean), self.per_overlap_max_len))

        if overlap_mean > 0:
            desired_overlap_amount = (
                int(gamma(a=overlap_mean ** 2 / self.per_overlap_var, scale=self.per_overlap_var / overlap_mean).rvs())
                if self.per_overlap_var > 0
                else int(overlap_mean)
            )
            desired_overlap_amount = max(
                self.per_overlap_min_len, min(desired_overlap_amount, self.per_overlap_max_len)
            )
        else:
            desired_overlap_amount = 0
        return desired_overlap_amount

    def sample_noise_manifest(self, noise_manifest: dict) -> list:
        """
        Sample noise manifest to a specified count `num_noise_files` for the current simulated audio session.

        Args:
            noise_manifest (list): 
                List of noise source samples to be sampled from.

        Returns:
            sampled_noise_manifest (list):
                List of noise samples to be used for the current session.
        """
        num_noise_files = min(len(noise_manifest), self.num_noise_files)
        sampled_noise_manifest = []
        if num_noise_files > 0:
            selected_noise_ids = np.random.choice(range(len(noise_manifest)), num_noise_files, replace=False)
            for k in selected_noise_ids:
                sampled_noise_manifest.append(noise_manifest[k])
        return sampled_noise_manifest
