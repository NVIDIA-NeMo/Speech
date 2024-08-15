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

import concurrent
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from scipy.signal import convolve
from scipy.signal.windows import cosine, hamming, hann
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.utils.data_simulation_utils import (
    DataAnnotator,
    SpeechSampler,
    build_speaker_samples_map,
    get_background_noise,
    get_cleaned_base_path,
    get_random_offset_index,
    get_speaker_ids,
    get_speaker_samples,
    get_split_points_in_alignments,
    load_speaker_sample,
    normalize_audio,
    per_speaker_normalize,
    perturb_audio,
    read_audio_from_buffer,
    read_noise_manifest,
)
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.asr.parts.utils.speaker_utils import get_overlap_range, is_overlap, merge_float_intervals
from nemo.utils import logging

try:
    import pyroomacoustics as pra
    from pyroomacoustics.directivities import CardioidFamily, DirectionVector, DirectivityPattern

    PRA = True
except ImportError:
    PRA = False
try:
    from gpuRIR import att2t_SabineEstimator, beta_SabineEstimation, simulateRIR, t2n

    GPURIR = True
except ImportError:
    GPURIR = False


class MultiSpeakerSimulator(object):
    """
    Multispeaker Audio Session Simulator - Simulates multispeaker audio sessions using single-speaker audio files and
    corresponding word alignments.

    Change Log:
    v1.0: Dec 2022
        - First working verison, supports multispeaker simulation with overlaps, silence and RIR
        v1.0.1: Feb 2023
            - Multi-GPU support for speed up
            - Faster random sampling routine
            - Fixed sentence duration bug
            - Silence and overlap length sampling algorithms are updated to guarantee `mean_silence` approximation
        v1.0.2: March 2023
            - Added support for segment-level gain perturbation and session-level white-noise perturbation
            - Modified speaker sampling mechanism to include as many speakers as possible in each data-generation run
            - Added chunking mechanism to avoid freezing in multiprocessing processes

    v1.1.0 March 2023
        - Faster audio-file loading with maximum audio duration parameter
        - Re-organized MultiSpeakerSimulator class and moved util functions to util files.
        v1.1.1 March 2023
            - Changed `silence_mean` to use exactly the same sampling equation as `overlap_mean`.


    Args:
        cfg: OmegaConf configuration loaded from yaml file.

    Parameters:
      manifest_filepath (str): Manifest file with paths to single speaker audio files
      sr (int): Sampling rate of the input audio files from the manifest
      random_seed (int): Seed to random number generator

    session_config:
      num_speakers (int): Number of unique speakers per multispeaker audio session
      num_sessions (int): Number of sessions to simulate
      session_length (int): Length of each simulated multispeaker audio session (seconds). Short sessions
                            (e.g. ~240 seconds) tend to fall short of the expected overlap-ratio and silence-ratio.

    session_params:
      max_audio_read_sec (int): The maximum audio length in second when loading an audio file.
                                The bigger the number, the slower the reading speed. Should be greater than 2.5 second.
      sentence_length_params (list): k,p values for a negative_binomial distribution which is sampled to get the
                                     sentence length (in number of words)
      dominance_var (float): Variance in speaker dominance (where each speaker's dominance is sampled from a normal
                             distribution centered on 1/`num_speakers`, and then the dominance values are together
                             normalized to 1)
      min_dominance (float): Minimum percentage of speaking time per speaker (note that this can cause the dominance of
                             the other speakers to be slightly reduced)
      turn_prob (float): Probability of switching speakers after each utterance

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
      start_window (bool): Whether to window the start of sentences to smooth the audio signal (and remove silence at
                            the start of the clip)
      window_type (str): Type of windowing used when segmenting utterances ("hamming", "hann", "cosine")
      window_size (float): Length of window at the start or the end of segmented utterance (seconds)
      start_buffer (float): Buffer of silence before the start of the sentence (to avoid cutting off speech or starting
                            abruptly)
      split_buffer (float): Split RTTM labels if greater than twice this amount of silence (to avoid long gaps between
                            utterances as being labelled as speech)
      release_buffer (float): Buffer before window at end of sentence (to avoid cutting off speech or ending abruptly)
      normalize (bool): Normalize speaker volumes
      normalization_type (str): Normalizing speakers ("equal" - same volume per speaker, "var" - variable volume per
                                speaker)
      normalization_var (str): Variance in speaker volume (sample from standard deviation centered at 1)
      min_volume (float): Minimum speaker volume (only used when variable normalization is used)
      max_volume (float): Maximum speaker volume (only used when variable normalization is used)
      end_buffer (float): Buffer at the end of the session to leave blank

    outputs:
      output_dir (str): Output directory for audio sessions and corresponding label files
      output_filename (str): Output filename for the wav and RTTM files
      overwrite_output (bool): If true, delete the output directory if it exists
      output_precision (int): Number of decimal places in output files

    background_noise:
      add_bg (bool): Add ambient background noise if true
      background_manifest (str): Path to background noise manifest file
      snr (int): SNR for background noise (using average speaker power), set `snr_min` and `snr_max` values to enable random SNR
      snr_min (int):  Min random SNR for background noise (using average speaker power), set `null` to use fixed SNR
      snr_max (int):  Max random SNR for background noise (using average speaker power), set `null` to use fixed SNR

    segment_augmentor:
      add_seg_aug (bool): Set True to enable augmentation on each speech segment (Default: False)
      segmentor:
        gain:
            prob (float): Probability range (uniform distribution) gain augmentation for individual segment
            min_gain_dbfs (float): minimum gain in terms of dB
            max_gain_dbfs (float): maximum gain in terms of dB

    session_augmentor:
      add_sess_aug: (bool) set True to enable audio augmentation on the whole session (Default: False)
      segmentor:
        white_noise:
            prob (float): Probability of adding white noise (Default: 1.0)
            min_level (float): minimum gain in terms of dB
            max_level (float): maximum gain in terms of dB

    speaker_enforcement:
      enforce_num_speakers (bool): Enforce that all requested speakers are present in the output wav file
      enforce_time (list): Percentage of the way through the audio session that enforcement mode is triggered (sampled
                           between time 1 and 2)

    segment_manifest: (parameters for regenerating the segment manifest file)
      window (float): Window length for segmentation
      shift (float): Shift length for segmentation
      step_count (int): Number of the unit segments you want to create per utterance
      deci (int): Rounding decimals for segment manifest file
    """

    def __init__(self, cfg):
        self._params = cfg
        self.annotator = DataAnnotator(cfg)
        self.sampler = SpeechSampler(cfg)
        # internal params
        self._manifest = read_manifest(self._params.data_simulator.manifest_filepath)
        self._speaker_samples = build_speaker_samples_map(self._manifest)
        self._noise_samples = []
        self._sentence = None
        self._text = ""
        self._words = []
        self._alignments = []
        # minimum number of alignments for a manifest to be considered valid
        self._min_alignment_count = 2
        self._merged_speech_intervals = []
        # keep track of furthest sample per speaker to avoid overlapping same speaker
        self._furthest_sample = [0 for n in range(self._params.data_simulator.session_config.num_speakers)]
        # use to ensure overlap percentage is correct
        self._missing_overlap = 0
        # creating manifests during online data simulation
        self.base_manifest_filepath = None
        self.segment_manifest_filepath = None
        self._max_audio_read_sec = self._params.data_simulator.session_params.max_audio_read_sec
        self._turn_prob_min = self._params.data_simulator.session_params.get("turn_prob_min", 0.5)
        # variable speaker volume
        self._volume = None
        self._speaker_ids = None
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._audio_read_buffer_dict = {}
        self.add_missing_overlap = self._params.data_simulator.session_params.get("add_missing_overlap", False)

        if (
            self._params.data_simulator.segment_augmentor.get("augmentor", None)
            and self._params.data_simulator.segment_augmentor.add_seg_aug
        ):
            self.segment_augmentor = process_augmentations(
                augmenter=self._params.data_simulator.segment_augmentor.augmentor
            )
        else:
            self.segment_augmentor = None

        if (
            self._params.data_simulator.session_augmentor.get("augmentor", None)
            and self._params.data_simulator.session_augmentor.add_sess_aug
        ):
            self.session_augmentor = process_augmentations(
                augmenter=self._params.data_simulator.session_augmentor.augmentor
            )
        else:
            self.session_augmentor = None

        # Error check the input arguments for simulation
        self._check_args()

        # Initialize speaker permutations to maximize the number of speakers in the created dataset
        self._permutated_speaker_inds = self._init_speaker_permutations(
            num_sess=self._params.data_simulator.session_config.num_sessions,
            num_speakers=self._params.data_simulator.session_config.num_speakers,
            all_speaker_ids=self._speaker_samples.keys(),
            random_seed=self._params.data_simulator.random_seed,
        )

        # Intialize multiprocessing related variables
        self.num_workers = self._params.get("num_workers", 1)
        self.multiprocessing_chunksize = self._params.data_simulator.get('multiprocessing_chunksize', 10000)
        self.chunk_count = self._init_chunk_count()

    def _init_speaker_permutations(self, num_sess: int, num_speakers: int, all_speaker_ids: List, random_seed: int):
        """
        Initialize the speaker permutations for the number of speakers in the session.
        When generating the simulated sessions, we want to include as many speakers as possible.
        This function generates a set of permutations that can be used to sweep all speakers in
        the source dataset to make sure we maximize the total number of speakers included in
        the simulated sessions.

        Args:
            num_sess (int): Number of sessions to generate
            num_speakers (int): Number of speakers in each session
            all_speaker_ids (list): List of all speaker IDs

        Returns:
            permuted_inds (np.array):
                Array of permuted speaker indices to use for each session
                Dimensions: (num_sess, num_speakers)
        """
        np.random.seed(random_seed)
        all_speaker_id_counts = len(list(all_speaker_ids))

        # Calculate how many permutations are needed
        perm_set_count = int(np.ceil(num_speakers * num_sess / all_speaker_id_counts))

        target_count = num_speakers * num_sess
        for count in range(perm_set_count):
            if target_count < all_speaker_id_counts:
                seq_len = target_count
            else:
                seq_len = all_speaker_id_counts
            if seq_len <= 0:
                raise ValueError(f"seq_len is {seq_len} at count {count} and should be greater than 0")

            if count == 0:
                permuted_inds = np.random.permutation(len(all_speaker_ids))[:seq_len]
            else:
                permuted_inds = np.hstack((permuted_inds, np.random.permutation(len(all_speaker_ids))[:seq_len]))
            target_count -= seq_len

        logging.info(f"Total {all_speaker_id_counts} speakers in the source dataset.")
        logging.info(f"Initialized speaker permutations for {num_sess} sessions with {num_speakers} speakers each.")
        return permuted_inds.reshape(num_sess, num_speakers)

    def _init_chunk_count(self):
        """
        Initialize the chunk count for multi-processing to prevent over-flow of job counts.
        The multi-processing pipeline can freeze if there are more than approximately 10,000 jobs
        in the pipeline at the same time.
        """
        return int(np.ceil(self._params.data_simulator.session_config.num_sessions / self.multiprocessing_chunksize))

    def _check_args(self):
        """
        Checks YAML arguments to ensure they are within valid ranges.
        """
        if self._params.data_simulator.session_config.num_speakers < 1:
            raise Exception("At least one speaker is required for making audio sessions (num_speakers < 1)")
        if (
            self._params.data_simulator.session_params.turn_prob < 0
            or self._params.data_simulator.session_params.turn_prob > 1
        ):
            raise Exception("Turn probability is outside of [0,1]")
        if (
            self._params.data_simulator.session_params.turn_prob < 0
            or self._params.data_simulator.session_params.turn_prob > 1
        ):
            raise Exception("Turn probability is outside of [0,1]")
        elif (
            self._params.data_simulator.session_params.turn_prob < self._turn_prob_min
            and self._params.data_simulator.speaker_enforcement.enforce_num_speakers == True
        ):
            logging.warning(
                "Turn probability is less than {self._turn_prob_min} while enforce_num_speakers=True, which may result in excessive session lengths. Forcing turn_prob to 0.5."
            )
            self._params.data_simulator.session_params.turn_prob = self._turn_prob_min
        if self._params.data_simulator.session_params.max_audio_read_sec < 2.5:
            raise Exception("Max audio read time must be greater than 2.5 seconds")

        if self._params.data_simulator.session_params.sentence_length_params[0] <= 0:
            raise Exception(
                "k (number of success until the exp. ends) in Sentence length parameter value must be a positive number"
            )

        if not (0 < self._params.data_simulator.session_params.sentence_length_params[1] <= 1):
            raise Exception("p (success probability) value in sentence length parameter must be in range (0,1]")

        if (
            self._params.data_simulator.session_params.mean_overlap < 0
            or self._params.data_simulator.session_params.mean_overlap > 1
        ):
            raise Exception("Mean overlap is outside of [0,1]")
        if (
            self._params.data_simulator.session_params.mean_silence < 0
            or self._params.data_simulator.session_params.mean_silence > 1
        ):
            raise Exception("Mean silence is outside of [0,1]")
        if self._params.data_simulator.session_params.mean_silence_var < 0:
            raise Exception("Mean silence variance is not below 0")
        if (
            self._params.data_simulator.session_params.mean_silence > 0
            and self._params.data_simulator.session_params.mean_silence_var
            >= self._params.data_simulator.session_params.mean_silence
            * (1 - self._params.data_simulator.session_params.mean_silence)
        ):
            raise Exception("Mean silence variance should be lower than mean_silence * (1-mean_silence)")
        if self._params.data_simulator.session_params.per_silence_var < 0:
            raise Exception("Per silence variance is below 0")

        if self._params.data_simulator.session_params.mean_overlap_var < 0:
            raise Exception("Mean overlap variance is not larger than 0")
        if (
            self._params.data_simulator.session_params.mean_overlap > 0
            and self._params.data_simulator.session_params.mean_overlap_var
            >= self._params.data_simulator.session_params.mean_overlap
            * (1 - self._params.data_simulator.session_params.mean_overlap)
        ):
            raise Exception("Mean overlap variance should be lower than mean_overlap * (1-mean_overlap)")
        if self._params.data_simulator.session_params.per_overlap_var < 0:
            raise Exception("Per overlap variance is not larger than 0")

        if (
            self._params.data_simulator.session_params.min_dominance < 0
            or self._params.data_simulator.session_params.min_dominance > 1
        ):
            raise Exception("Minimum dominance is outside of [0,1]")
        if (
            self._params.data_simulator.speaker_enforcement.enforce_time[0] < 0
            or self._params.data_simulator.speaker_enforcement.enforce_time[0] > 1
        ):
            raise Exception("Speaker enforcement start is outside of [0,1]")
        if (
            self._params.data_simulator.speaker_enforcement.enforce_time[1] < 0
            or self._params.data_simulator.speaker_enforcement.enforce_time[1] > 1
        ):
            raise Exception("Speaker enforcement end is outside of [0,1]")

        if (
            self._params.data_simulator.session_params.min_dominance
            * self._params.data_simulator.session_config.num_speakers
            > 1
        ):
            raise Exception("Number of speakers times minimum dominance is greater than 1")

        if (
            self._params.data_simulator.session_params.window_type not in ['hamming', 'hann', 'cosine']
            and self._params.data_simulator.session_params.window_type is not None
        ):
            raise Exception("Incorrect window type provided")

        if len(self._manifest) == 0:
            raise Exception("Manifest file is empty. Check that the source path is correct.")

    def clean_up(self):
        """
        Clear the system memory. Cache data for audio files and alignments are removed.
        """
        self._sentence = None
        self._words = []
        self._alignments = []
        self._audio_read_buffer_dict = {}
        torch.cuda.empty_cache()

    def _get_speaker_dominance(self) -> List[float]:
        """
        Get the dominance value for each speaker, accounting for the dominance variance and
        the minimum per-speaker dominance.

        Returns:
            dominance (list): Per-speaker dominance
        """
        dominance_mean = 1.0 / self._params.data_simulator.session_config.num_speakers
        dominance = np.random.normal(
            loc=dominance_mean,
            scale=self._params.data_simulator.session_params.dominance_var,
            size=self._params.data_simulator.session_config.num_speakers,
        )
        dominance = np.clip(dominance, a_min=0, a_max=np.inf)
        # normalize while maintaining minimum dominance
        total = np.sum(dominance)
        if total == 0:
            for i in range(len(dominance)):
                dominance[i] += self._params.data_simulator.session_params.min_dominance
        # scale accounting for min_dominance which has to be added after
        dominance = (dominance / total) * (
            1
            - self._params.data_simulator.session_params.min_dominance
            * self._params.data_simulator.session_config.num_speakers
        )
        for i in range(len(dominance)):
            dominance[i] += self._params.data_simulator.session_params.min_dominance
            if (
                i > 0
            ):  # dominance values are cumulative to make it easy to select the speaker using a random value in [0,1]
                dominance[i] = dominance[i] + dominance[i - 1]
        return dominance

    def _increase_speaker_dominance(
        self, base_speaker_dominance: List[float], factor: int
    ) -> Tuple[List[float], bool]:
        """
        Increase speaker dominance for unrepresented speakers (used only in enforce mode).
        Increases the dominance for these speakers by the input factor (and then re-normalizes the probabilities to 1).

        Args:
            base_speaker_dominance (list): Dominance values for each speaker.
            factor (int): Factor to increase dominance of unrepresented speakers by.
        Returns:
            dominance (list): Per-speaker dominance
            enforce (bool): Whether to keep enforce mode turned on
        """
        increase_percent = []
        for i in range(self._params.data_simulator.session_config.num_speakers):
            if self._furthest_sample[i] == 0:
                increase_percent.append(i)
        # ramp up enforce counter until speaker is sampled, then reset once all speakers have spoken
        if len(increase_percent) > 0:
            # extract original per-speaker probabilities
            dominance = np.copy(base_speaker_dominance)
            for i in range(len(dominance) - 1, 0, -1):
                dominance[i] = dominance[i] - dominance[i - 1]
            # increase specified speakers by the desired factor
            for i in increase_percent:
                dominance[i] = dominance[i] * factor
            # renormalize
            dominance = dominance / np.sum(dominance)
            for i in range(1, len(dominance)):
                dominance[i] = dominance[i] + dominance[i - 1]
            enforce = True
        else:  # no unrepresented speakers, so enforce mode can be turned off
            dominance = base_speaker_dominance
            enforce = False
        return dominance, enforce

    def _set_speaker_volume(self):
        """
        Set the volume for each speaker (either equal volume or variable speaker volume).
        """
        if self._params.data_simulator.session_params.normalization_type == 'equal':
            self._volume = np.ones(self._params.data_simulator.session_config.num_speakers)
        elif self._params.data_simulator.session_params.normalization_type == 'variable':
            self._volume = np.random.normal(
                loc=1.0,
                scale=self._params.data_simulator.session_params.normalization_var,
                size=self._params.data_simulator.session_config.num_speakers,
            )
            self._volume = np.clip(
                np.array(self._volume),
                a_min=self._params.data_simulator.session_params.min_volume,
                a_max=self._params.data_simulator.session_params.max_volume,
            ).tolist()

    def _get_next_speaker(self, prev_speaker: int, dominance: List[float]) -> int:
        """
        Get the next speaker (accounting for turn probability and dominance distribution).

        Args:
            prev_speaker (int): Previous speaker turn.
            dominance (list): Dominance values for each speaker.
        Returns:
            prev_speaker/speaker_turn (int): Speaker turn
        """
        if self._params.data_simulator.session_config.num_speakers == 1:
            prev_speaker = 0 if prev_speaker is None else prev_speaker
            return prev_speaker
        else:
            if (
                np.random.uniform(0, 1) > self._params.data_simulator.session_params.turn_prob
                and prev_speaker is not None
            ):
                return prev_speaker
            else:
                speaker_turn = prev_speaker
                while speaker_turn == prev_speaker:  # ensure another speaker goes next
                    rand = np.random.uniform(0, 1)
                    speaker_turn = 0
                    while rand > dominance[speaker_turn]:
                        speaker_turn += 1
                return speaker_turn

    def _get_window(self, window_amount: int, start: bool = False):
        """
        Get window curve to alleviate abrupt change of time-series signal when segmenting audio samples.

        Args:
            window_amount (int): Window length (in terms of number of samples).
            start (bool): If true, return the first half of the window.

        Returns:
            window (tensor): Half window (either first half or second half)
        """
        if self._params.data_simulator.session_params.window_type == 'hamming':
            window = hamming(window_amount * 2)
        elif self._params.data_simulator.session_params.window_type == 'hann':
            window = hann(window_amount * 2)
        elif self._params.data_simulator.session_params.window_type == 'cosine':
            window = cosine(window_amount * 2)
        else:
            raise Exception("Incorrect window type provided")

        window = torch.from_numpy(window).to(self._device)

        # return the first half or second half of the window
        if start:
            return window[:window_amount]
        else:
            return window[window_amount:]

    def _get_start_buffer_and_window(self, first_alignment: int) -> Tuple[int, int]:
        """
        Get the start cutoff and window length for smoothing the start of the sentence.

        Args:
            first_alignment (int): Start of the first word (in terms of number of samples).
        Returns:
            start_cutoff (int): Amount into the audio clip to start
            window_amount (int): Window length
        """
        window_amount = int(self._params.data_simulator.session_params.window_size * self._params.data_simulator.sr)
        start_buffer = int(self._params.data_simulator.session_params.start_buffer * self._params.data_simulator.sr)

        if first_alignment < start_buffer:
            window_amount = 0
            start_cutoff = 0
        elif first_alignment < start_buffer + window_amount:
            window_amount = first_alignment - start_buffer
            start_cutoff = 0
        else:
            start_cutoff = first_alignment - start_buffer - window_amount

        return start_cutoff, window_amount

    def _get_end_buffer_and_window(
        self, current_sample_cursor: int, remaining_dur_samples: int, remaining_len_audio_file: int
    ) -> Tuple[int, int]:
        """
        Get the end buffer and window length for smoothing the end of the sentence.

        Args:
            current_sample_cursor (int): Current location in the target file (in terms of number of samples).
            remaining_dur_samples (int): Remaining duration in the target file (in terms of number of samples).
            remaining_len_audio_file (int): Length remaining in audio file (in terms of number of samples).
        Returns:
            release_buffer (int): Amount after the end of the last alignment to include
            window_amount (int): Window length
        """
        window_amount = int(self._params.data_simulator.session_params.window_size * self._params.data_simulator.sr)
        release_buffer = int(
            self._params.data_simulator.session_params.release_buffer * self._params.data_simulator.sr
        )

        if current_sample_cursor + release_buffer > remaining_dur_samples:
            release_buffer = remaining_dur_samples - current_sample_cursor
            window_amount = 0
        elif current_sample_cursor + window_amount + release_buffer > remaining_dur_samples:
            window_amount = remaining_dur_samples - current_sample_cursor - release_buffer

        if remaining_len_audio_file < release_buffer:
            release_buffer = remaining_len_audio_file
            window_amount = 0
        elif remaining_len_audio_file < release_buffer + window_amount:
            window_amount = remaining_len_audio_file - release_buffer

        return release_buffer, window_amount

    def _check_missing_speakers(self, num_missing: int = 0):
        """
        Check if any speakers were not included in the clip and display a warning.

        Args:
            num_missing (int): Number of missing speakers.
        """
        for k in range(len(self._furthest_sample)):
            if self._furthest_sample[k] == 0:
                num_missing += 1
        if num_missing != 0:
            warnings.warn(
                f"{self._params.data_simulator.session_config.num_speakers - num_missing}"
                f"speakers were included in the clip instead of the requested amount of "
                f"{self._params.data_simulator.session_config.num_speakers}"
            )

    def _add_file(
        self,
        audio_manifest: dict,
        audio_file,
        sentence_word_count: int,
        max_word_count_in_sentence: int,
        max_samples_in_sentence: int,
        random_offset: bool = False,
    ) -> Tuple[int, torch.Tensor]:
        """
        Add audio file to current sentence (up to the desired number of words).
        Uses the alignments to segment the audio file.
        NOTE: 0 index is always silence in `audio_manifest['words']`, so we choose `offset_idx=1` as the first word

        Args:
            audio_manifest (dict): Line from manifest file for current audio file
            audio_file (tensor): Current loaded audio file
            sentence_word_count (int): Running count for number of words in sentence
            max_word_count_in_sentence (int): Maximum count for number of words in sentence
            max_samples_in_sentence (int): Maximum length for sentence in terms of samples

        Returns:
            sentence_word_count+current_word_count (int): Running word count
            len(self._sentence) (tensor): Current length of the audio file
        """
        # In general, random offset is not needed since random silence index has already been chosen
        if random_offset:
            offset_idx = np.random.randint(low=1, high=len(audio_manifest['words']))
        else:
            offset_idx = 1

        first_alignment = int(audio_manifest['alignments'][offset_idx - 1] * self._params.data_simulator.sr)
        start_cutoff, start_window_amount = self._get_start_buffer_and_window(first_alignment)
        if not self._params.data_simulator.session_params.start_window:  # cut off the start of the sentence
            start_window_amount = 0

        # Ensure the desired number of words are added and the length of the output session isn't exceeded
        sentence_samples = len(self._sentence)

        remaining_dur_samples = max_samples_in_sentence - sentence_samples
        remaining_duration = max_word_count_in_sentence - sentence_word_count
        prev_dur_samples, dur_samples, curr_dur_samples = 0, 0, 0
        current_word_count = 0
        word_idx = offset_idx
        silence_count = 1
        while (
            current_word_count < remaining_duration
            and dur_samples < remaining_dur_samples
            and word_idx < len(audio_manifest['words'])
        ):
            dur_samples = int(audio_manifest['alignments'][word_idx] * self._params.data_simulator.sr) - start_cutoff

            # check the length of the generated sentence in terms of sample count (int).
            if curr_dur_samples + dur_samples > remaining_dur_samples:
                # if the upcoming loop will exceed the remaining sample count, break out of the loop.
                break

            word = audio_manifest['words'][word_idx]

            if silence_count > 0 and word == "":
                break

            self._words.append(word)
            self._alignments.append(
                float(sentence_samples * 1.0 / self._params.data_simulator.sr)
                - float(start_cutoff * 1.0 / self._params.data_simulator.sr)
                + audio_manifest['alignments'][word_idx]
            )

            if word == "":
                word_idx += 1
                silence_count += 1
                continue
            elif self._text == "":
                self._text += word
            else:
                self._text += " " + word

            word_idx += 1
            current_word_count += 1
            prev_dur_samples = dur_samples
            curr_dur_samples += dur_samples

        # add audio clip up to the final alignment
        if self._params.data_simulator.session_params.window_type is not None:  # cut off the start of the sentence
            if start_window_amount > 0:  # include window
                window = self._get_window(start_window_amount, start=True)
                self._sentence = self._sentence.to(self._device)
                self._sentence = torch.cat(
                    (
                        self._sentence,
                        torch.multiply(audio_file[start_cutoff : start_cutoff + start_window_amount], window),
                    ),
                    0,
                )
            self._sentence = torch.cat(
                (
                    self._sentence,
                    audio_file[start_cutoff + start_window_amount : start_cutoff + prev_dur_samples],
                ),
                0,
            ).to(self._device)

        else:
            self._sentence = torch.cat(
                (self._sentence, audio_file[start_cutoff : start_cutoff + prev_dur_samples]), 0
            ).to(self._device)

        # windowing at the end of the sentence
        if (
            word_idx < len(audio_manifest['words'])
        ) and self._params.data_simulator.session_params.window_type is not None:
            release_buffer, end_window_amount = self._get_end_buffer_and_window(
                prev_dur_samples,
                remaining_dur_samples,
                len(audio_file[start_cutoff + prev_dur_samples :]),
            )
            self._sentence = torch.cat(
                (
                    self._sentence,
                    audio_file[start_cutoff + prev_dur_samples : start_cutoff + prev_dur_samples + release_buffer],
                ),
                0,
            ).to(self._device)

            if end_window_amount > 0:  # include window
                window = self._get_window(end_window_amount, start=False)
                sig_start = start_cutoff + prev_dur_samples + release_buffer
                sig_end = start_cutoff + prev_dur_samples + release_buffer + end_window_amount
                windowed_audio_file = torch.multiply(audio_file[sig_start:sig_end], window)
                self._sentence = torch.cat((self._sentence, windowed_audio_file), 0).to(self._device)

        del audio_file
        return sentence_word_count + current_word_count, len(self._sentence)

    def _build_sentence(
        self,
        speaker_turn: int,
        speaker_ids: List[str],
        speaker_wav_align_map: Dict[str, list],
        max_samples_in_sentence: int,
    ):
        """
        Build a new sentence by attaching utterance samples together until the sentence has reached a desired length.
        While generating the sentence, alignment information is used to segment the audio.

        Args:
            speaker_turn (int): Current speaker turn.
            speaker_ids (list): LibriSpeech speaker IDs for each speaker in the current session.
            speaker_wav_align_map (dict): Dictionary containing speaker IDs and their corresponding wav filepath and alignments.
            max_samples_in_sentence (int): Maximum length for sentence in terms of samples
        """
        # select speaker length
        sl = (
            np.random.negative_binomial(
                self._params.data_simulator.session_params.sentence_length_params[0],
                self._params.data_simulator.session_params.sentence_length_params[1],
            )
            + 1
        )

        # initialize sentence, text, words, alignments
        self._sentence = torch.zeros(0, dtype=torch.float64, device=self._device)
        self._text = ""
        self._words, self._alignments = [], []
        sentence_word_count, sentence_samples = 0, 0

        # build sentence
        while sentence_word_count < sl and sentence_samples < max_samples_in_sentence:
            audio_manifest = load_speaker_sample(
                speaker_wav_align_map=speaker_wav_align_map,
                speaker_ids=speaker_ids,
                speaker_turn=speaker_turn,
                min_alignment_count=self._min_alignment_count,
            )

            offset_index = get_random_offset_index(
                audio_manifest=audio_manifest,
                audio_read_buffer_dict=self._audio_read_buffer_dict,
                offset_min=0,
                max_audio_read_sec=self._max_audio_read_sec,
                min_alignment_count=self._min_alignment_count,
            )

            audio_file, sr, audio_manifest = read_audio_from_buffer(
                audio_manifest=audio_manifest,
                buffer_dict=self._audio_read_buffer_dict,
                offset_index=offset_index,
                device=self._device,
                max_audio_read_sec=self._max_audio_read_sec,
                min_alignment_count=self._min_alignment_count,
                read_subset=True,
            )

            # Step 6-2: Add optional perturbations to the specific audio segment (i.e. to `self._sentnece`)
            if self._params.data_simulator.segment_augmentor.add_seg_aug:
                audio_file = perturb_audio(audio_file, sr, self.segment_augmentor, device=self._device)

            sentence_word_count, sentence_samples = self._add_file(
                audio_manifest, audio_file, sentence_word_count, sl, max_samples_in_sentence
            )

        # per-speaker normalization (accounting for active speaker time)
        if self._params.data_simulator.session_params.normalize and torch.max(torch.abs(self._sentence)) > 0:
            splits = get_split_points_in_alignments(
                words=self._words,
                alignments=self._alignments,
                split_buffer=self._params.data_simulator.session_params.split_buffer,
                sr=self._params.data_simulator.sr,
                sentence_audio_len=len(self._sentence),
            )
            self._sentence = per_speaker_normalize(
                sentence_audio=self._sentence,
                splits=splits,
                speaker_turn=speaker_turn,
                volume=self._volume,
                device=self._device,
            )

    def _add_silence_or_overlap(
        self,
        speaker_turn: int,
        prev_speaker: int,
        start: int,
        length: int,
        session_len_samples: int,
        prev_len_samples: int,
        enforce: bool,
    ) -> int:
        """
        Returns new overlapped (or shifted) start position after inserting overlap or silence.

        Args:
            speaker_turn (int): The integer index of the current speaker turn.
            prev_speaker (int): The integer index of the previous speaker turn.
            start (int): Current start of the audio file being inserted.
            length (int): Length of the audio file being inserted.
            session_len_samples (int): Maximum length of the session in terms of number of samples
            prev_len_samples (int): Length of previous sentence (in terms of number of samples)
            enforce (bool): Whether speaker enforcement mode is being used
        Returns:
            new_start (int): New starting position in the session accounting for overlap or silence
        """
        running_len_samples = start + length
        # `length` is the length of the current sentence to be added, so not included in self.sampler.running_speech_len_samples
        non_silence_len_samples = self.sampler.running_speech_len_samples + length

        # compare silence and overlap ratios
        add_overlap = self.sampler.silence_vs_overlap_selector(running_len_samples, non_silence_len_samples)

        # choose overlap if this speaker is not the same as the previous speaker and add_overlap is True.
        if prev_speaker != speaker_turn and prev_speaker is not None and add_overlap:
            desired_overlap_amount = self.sampler.sample_from_overlap_model(non_silence_len_samples)
            new_start = start - desired_overlap_amount

            # avoid overlap at start of clip
            if new_start < 0:
                desired_overlap_amount -= 0 - new_start
                self._missing_overlap += 0 - new_start
                new_start = 0

            # if same speaker ends up overlapping from any previous clip, pad with silence instead
            if new_start < self._furthest_sample[speaker_turn]:
                desired_overlap_amount -= self._furthest_sample[speaker_turn] - new_start
                self._missing_overlap += self._furthest_sample[speaker_turn] - new_start
                new_start = self._furthest_sample[speaker_turn]

            prev_start = start - prev_len_samples
            prev_end = start
            new_end = new_start + length

            # check overlap amount to calculate the actual amount of generated overlaps
            overlap_amount = 0
            if is_overlap([prev_start, prev_end], [new_start, new_end]):
                overlap_range = get_overlap_range([prev_start, prev_end], [new_start, new_end])
                overlap_amount = max(overlap_range[1] - overlap_range[0], 0)

            if overlap_amount < desired_overlap_amount:
                self._missing_overlap += desired_overlap_amount - overlap_amount
            self.sampler.running_overlap_len_samples += overlap_amount

        # if we are not adding overlap, add silence
        else:
            silence_amount = self.sampler.sample_from_silence_model(running_len_samples)
            if start + length + silence_amount > session_len_samples and not enforce:
                new_start = max(session_len_samples - length, start)
            else:
                new_start = start + silence_amount
        return new_start

    def _get_session_meta_data(self, array: np.ndarray, snr: float) -> dict:
        """
        Get meta data for the current session.

        Args:
            array (np.ndarray): audio array
            snr (float): signal-to-noise ratio

        Returns:
            dict: meta data
        """
        meta_data = {
            "duration": array.shape[0] / self._params.data_simulator.sr,
            "silence_mean": self.sampler.sess_silence_mean,
            "overlap_mean": self.sampler.sess_overlap_mean,
            "bg_snr": snr,
            "speaker_ids": self._speaker_ids,
            "speaker_volumes": list(self._volume),
        }
        return meta_data

    def _get_session_silence_from_rttm(self, rttm_list: List[str], running_len_samples: int):
        """
        Calculate the total speech and silence duration in the current session using RTTM file.

        Args:
            rttm_list (list):
                List of RTTM timestamps
            running_len_samples (int):
                Total number of samples generated so far in the current session

        Returns:
            sess_speech_len_rttm (int):
                The total number of speech samples in the current session
            sess_silence_len_rttm (int):
                The total number of silence samples in the current session
        """
        all_sample_list = []
        for x_raw in rttm_list:
            x = [token for token in x_raw.split()]
            all_sample_list.append([float(x[0]), float(x[1])])

        self._merged_speech_intervals = merge_float_intervals(all_sample_list)
        total_speech_in_secs = sum([x[1] - x[0] for x in self._merged_speech_intervals])
        total_silence_in_secs = running_len_samples / self._params.data_simulator.sr - total_speech_in_secs
        sess_speech_len = int(total_speech_in_secs * self._params.data_simulator.sr)
        sess_silence_len = int(total_silence_in_secs * self._params.data_simulator.sr)
        return sess_speech_len, sess_silence_len

    def _add_sentence_to_array(
        self, start: int, length: int, array: torch.Tensor, is_speech: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Add a sentence to the session array containing time-series signal.

        Args:
            start (int): Starting position in the session
            length (int): Length of the sentence
            array (torch.Tensor): Session array
            is_speech (torch.Tensor): Session array containing speech/non-speech labels

        Returns:
            array (torch.Tensor): Session array in torch.Tensor format
            is_speech (torch.Tensor): Session array containing speech/non-speech labels in torch.Tensor format
        """
        end = start + length
        if end > len(array):  # only occurs in enforce mode
            array = torch.nn.functional.pad(array, (0, end - len(array)))
            is_speech = torch.nn.functional.pad(is_speech, (0, end - len(is_speech)))
        array[start:end] += self._sentence
        is_speech[start:end] = 1
        return array, is_speech, end

    def _generate_session(
        self,
        idx: int,
        basepath: str,
        filename: str,
        speaker_ids: List[str],
        speaker_wav_align_map: Dict[str, list],
        noise_samples: list,
        device: torch.device,
        enforce_counter: int = 2,
    ):
        """
        _generate_session function without RIR simulation.
        Generate a multispeaker audio session and corresponding label files.

        Args:
            idx (int): Index for current session (out of total number of sessions).
            basepath (str): Path to output directory.
            filename (str): Filename for output files.
            speaker_ids (list): List of speaker IDs that will be used in this session.
            speaker_wav_align_map (dict): Dictionary containing speaker IDs and their corresponding wav filepath and alignments.
            noise_samples (list): List of randomly sampled noise source files that will be used for generating this session.
            device (torch.device): Device to use for generating this session.
            enforce_counter (int): In enforcement mode, dominance is increased by a factor of enforce_counter for unrepresented speakers
        """
        random_seed = self._params.data_simulator.random_seed
        np.random.seed(random_seed + idx)

        self._device = device
        speaker_dominance = self._get_speaker_dominance()  # randomly determine speaker dominance
        base_speaker_dominance = np.copy(speaker_dominance)
        self._set_speaker_volume()

        running_len_samples, prev_len_samples = 0, 0
        prev_speaker = None
        self.annotator.init_annotation_lists()
        self._noise_samples = noise_samples
        self._furthest_sample = [0 for n in range(self._params.data_simulator.session_config.num_speakers)]
        self._missing_silence = 0

        # hold enforce until all speakers have spoken
        enforce_time = np.random.uniform(
            self._params.data_simulator.speaker_enforcement.enforce_time[0],
            self._params.data_simulator.speaker_enforcement.enforce_time[1],
        )
        enforce = self._params.data_simulator.speaker_enforcement.enforce_num_speakers

        session_len_samples = int(
            (self._params.data_simulator.session_config.session_length * self._params.data_simulator.sr)
        )
        array = torch.zeros(session_len_samples).to(self._device)
        is_speech = torch.zeros(session_len_samples).to(self._device)

        self.sampler.get_session_silence_mean()
        self.sampler.get_session_overlap_mean()

        while running_len_samples < session_len_samples or enforce:
            # Step 1: Prepare parameters for sentence generation
            # Enforce speakers depending on running length
            if running_len_samples > enforce_time * session_len_samples and enforce:
                speaker_dominance, enforce = self._increase_speaker_dominance(base_speaker_dominance, enforce_counter)
                if enforce:
                    enforce_counter += 1

            # Step 2: Select a speaker
            speaker_turn = self._get_next_speaker(prev_speaker, speaker_dominance)

            # Calculate parameters for building a sentence (only add if remaining length >  specific time)
            max_samples_in_sentence = session_len_samples - running_len_samples
            if enforce:
                max_samples_in_sentence = float('inf')
            elif (
                max_samples_in_sentence
                < self._params.data_simulator.session_params.end_buffer * self._params.data_simulator.sr
            ):
                break

            # Step 3: Generate a sentence
            self._build_sentence(speaker_turn, speaker_ids, speaker_wav_align_map, max_samples_in_sentence)
            length = len(self._sentence)

            # Step 4: Generate a timestamp for either silence or overlap
            start = self._add_silence_or_overlap(
                speaker_turn=speaker_turn,
                prev_speaker=prev_speaker,
                start=running_len_samples,
                length=length,
                session_len_samples=session_len_samples,
                prev_len_samples=prev_len_samples,
                enforce=enforce,
            )
            # step 5: add sentence to array
            array, is_speech, end = self._add_sentence_to_array(
                start=start,
                length=length,
                array=array,
                is_speech=is_speech,
            )

            # Step 6: Build entries for output files
            new_rttm_entries = self.annotator.create_new_rttm_entry(
                words=self._words,
                alignments=self._alignments,
                start=start / self._params.data_simulator.sr,
                end=end / self._params.data_simulator.sr,
                speaker_id=speaker_ids[speaker_turn],
            )

            self.annotator.annote_lists['rttm'].extend(new_rttm_entries)

            new_json_entry = self.annotator.create_new_json_entry(
                text=self._text,
                wav_filename=os.path.join(basepath, filename + '.wav'),
                start=start / self._params.data_simulator.sr,
                length=length / self._params.data_simulator.sr,
                speaker_id=speaker_ids[speaker_turn],
                rttm_filepath=os.path.join(basepath, filename + '.rttm'),
                ctm_filepath=os.path.join(basepath, filename + '.ctm'),
            )
            self.annotator.annote_lists['json'].append(new_json_entry)

            new_ctm_entries = self.annotator.create_new_ctm_entry(
                words=self._words,
                alignments=self._alignments,
                session_name=filename,
                speaker_id=speaker_ids[speaker_turn],
                start=float(start / self._params.data_simulator.sr),
            )

            self.annotator.annote_lists['ctm'].extend(new_ctm_entries)

            running_len_samples = np.maximum(running_len_samples, end)
            (
                self.sampler.running_speech_len_samples,
                self.sampler.running_silence_len_samples,
            ) = self._get_session_silence_from_rttm(
                rttm_list=self.annotator.annote_lists['rttm'], running_len_samples=running_len_samples
            )

            self._furthest_sample[speaker_turn] = running_len_samples
            prev_speaker = speaker_turn
            prev_len_samples = length

        # Step 7-1: Add optional perturbations to the whole session, such as white noise.
        if self._params.data_simulator.session_augmentor.add_sess_aug:
            # NOTE: This perturbation is not reflected in the session SNR in meta dictionary.
            array = perturb_audio(array, self._params.data_simulator.sr, self.session_augmentor, device=array.device)

        # Step 7-2: Additive background noise from noise manifest files
        if self._params.data_simulator.background_noise.add_bg:
            if len(self._noise_samples) > 0:
                avg_power_array = torch.mean(array[is_speech == 1] ** 2)
                bg, snr = get_background_noise(
                    len_array=len(array),
                    power_array=avg_power_array,
                    noise_samples=self._noise_samples,
                    audio_read_buffer_dict=self._audio_read_buffer_dict,
                    snr_min=self._params.data_simulator.background_noise.snr_min,
                    snr_max=self._params.data_simulator.background_noise.snr_max,
                    background_noise_snr=self._params.data_simulator.background_noise.snr,
                    seed=(random_seed + idx),
                    device=self._device,
                )
                array += bg
            else:
                raise ValueError('No background noise samples found in self._noise_samples.')
        else:
            snr = "N/A"

        # Step 7: Normalize and write to disk
        array = normalize_audio(array)

        if torch.is_tensor(array):
            array = array.cpu().numpy()
        sf.write(os.path.join(basepath, filename + '.wav'), array, self._params.data_simulator.sr)

        self.annotator.write_annotation_files(
            basepath=basepath,
            filename=filename,
            meta_data=self._get_session_meta_data(array=array, snr=snr),
        )

        # Step 8: Clean up memory
        del array
        self.clean_up()
        return basepath, filename

    def generate_sessions(self, random_seed: int = None):
        """
        Generate several multispeaker audio sessions and corresponding list files.

        Args:
            random_seed (int): random seed for reproducibility
        """
        logging.info(f"Generating Diarization Sessions")
        if random_seed is None:
            random_seed = self._params.data_simulator.random_seed
        np.random.seed(random_seed)

        output_dir = self._params.data_simulator.outputs.output_dir

        basepath = get_cleaned_base_path(
            output_dir, overwrite_output=self._params.data_simulator.outputs.overwrite_output
        )
        OmegaConf.save(self._params, os.path.join(output_dir, "params.yaml"))

        tp = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers)
        futures = []

        num_sessions = self._params.data_simulator.session_config.num_sessions
        source_noise_manifest = read_noise_manifest(
            add_bg=self._params.data_simulator.background_noise.add_bg,
            background_manifest=self._params.data_simulator.background_noise.background_manifest,
        )
        queue = []

        # add radomly sampled arguments to a list(queue) for multiprocessing
        for sess_idx in range(num_sessions):
            filename = self._params.data_simulator.outputs.output_filename + f"_{sess_idx}"
            speaker_ids = get_speaker_ids(
                sess_idx=sess_idx,
                speaker_samples=self._speaker_samples,
                permutated_speaker_inds=self._permutated_speaker_inds,
            )
            speaker_wav_align_map = get_speaker_samples(speaker_ids=speaker_ids, speaker_samples=self._speaker_samples)
            noise_samples = self.sampler.sample_noise_manifest(noise_manifest=source_noise_manifest)

            if torch.cuda.is_available():
                device = torch.device(f"cuda:{sess_idx % torch.cuda.device_count()}")
            else:
                device = self._device
            queue.append((sess_idx, basepath, filename, speaker_ids, speaker_wav_align_map, noise_samples, device))

        # for multiprocessing speed, we avoid loading potentially huge manifest list and speaker sample files into each process.
        if self.num_workers > 1:
            self._manifest = None
            self._speaker_samples = None

        # Chunk the sessions into smaller chunks for very large number of sessions (10K+ sessions)
        for chunk_idx in range(self.chunk_count):
            futures = []
            stt_idx, end_idx = (
                chunk_idx * self.multiprocessing_chunksize,
                min((chunk_idx + 1) * self.multiprocessing_chunksize, num_sessions),
            )
            for sess_idx in range(stt_idx, end_idx):
                self._furthest_sample = [0 for n in range(self._params.data_simulator.session_config.num_speakers)]
                self._audio_read_buffer_dict = {}
                if self.num_workers > 1:
                    futures.append(tp.submit(self._generate_session, *queue[sess_idx]))
                else:
                    futures.append(queue[sess_idx])

            if self.num_workers > 1:
                generator = concurrent.futures.as_completed(futures)
            else:
                generator = futures

            for future in tqdm(
                generator,
                desc=f"[{chunk_idx+1}/{self.chunk_count}] Waiting jobs from {stt_idx+1: 2} to {end_idx: 2}",
                unit="jobs",
                total=len(futures),
            ):
                if self.num_workers > 1:
                    basepath, filename = future.result()
                else:
                    self._noise_samples = self.sampler.sample_noise_manifest(
                        noise_manifest=source_noise_manifest,
                    )
                    basepath, filename = self._generate_session(*future)

                self.annotator.add_to_filename_lists(basepath=basepath, filename=filename)

                # throw warning if number of speakers is less than requested
                self._check_missing_speakers()

        tp.shutdown()
        self.annotator.write_filelist_files(basepath=basepath)
        logging.info(f"Data simulation has been completed, results saved at: {basepath}")


class RIRMultiSpeakerSimulator(MultiSpeakerSimulator):
    """
    RIR Augmented Multispeaker Audio Session Simulator - simulates multispeaker audio sessions using single-speaker
    audio files and corresponding word alignments, as well as simulated RIRs for augmentation.

    Args:
        cfg: OmegaConf configuration loaded from yaml file.

    Parameters (in addition to the base MultiSpeakerSimulator parameters):
    rir_generation:
      use_rir (bool): Whether to generate synthetic RIR
      toolkit (str): Which toolkit to use ("pyroomacoustics", "gpuRIR")
      room_config:
        room_sz (list): Size of the shoebox room environment (1d array for specific, 2d array for random range to be
                        sampled from)
        pos_src (list): Positions of the speakers in the simulated room environment (2d array for specific, 3d array
                        for random ranges to be sampled from)
        noise_src_pos (list): Position in room for the ambient background noise source
      mic_config:
        num_channels (int): Number of output audio channels
        pos_rcv (list): Microphone positions in the simulated room environment (1d/2d array for specific, 2d/3d array
                        for range assuming num_channels is 1/2+)
        orV_rcv (list or null): Microphone orientations (needed for non-omnidirectional microphones)
        mic_pattern (str): Microphone type ("omni" - omnidirectional) - currently only omnidirectional microphones are
                           supported for pyroomacoustics
      absorbtion_params: (Note that only `T60` is used for pyroomacoustics simulations)
        abs_weights (list): Absorption coefficient ratios for each surface
        T60 (float): Room reverberation time (`T60` is the time it takes for the RIR to decay by 60DB)
        att_diff (float): Starting attenuation (if this is different than att_max, the diffuse reverberation model is
                          used by gpuRIR)
        att_max (float): End attenuation when using the diffuse reverberation model (gpuRIR)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._check_args_rir()

    def _check_args_rir(self):
        """
        Checks RIR YAML arguments to ensure they are within valid ranges
        """

        if not (self._params.data_simulator.rir_generation.toolkit in ['pyroomacoustics', 'gpuRIR']):
            raise Exception("Toolkit must be pyroomacoustics or gpuRIR")
        if self._params.data_simulator.rir_generation.toolkit == 'pyroomacoustics' and not PRA:
            raise ImportError("pyroomacoustics should be installed to run this simulator with RIR augmentation")

        if self._params.data_simulator.rir_generation.toolkit == 'gpuRIR' and not GPURIR:
            raise ImportError("gpuRIR should be installed to run this simulator with RIR augmentation")

        if len(self._params.data_simulator.rir_generation.room_config.room_sz) != 3:
            raise Exception("Incorrect room dimensions provided")
        if self._params.data_simulator.rir_generation.mic_config.num_channels == 0:
            raise Exception("Number of channels should be greater or equal to 1")
        if len(self._params.data_simulator.rir_generation.room_config.pos_src) < 2:
            raise Exception("Less than 2 provided source positions")
        for sublist in self._params.data_simulator.rir_generation.room_config.pos_src:
            if len(sublist) != 3:
                raise Exception("Three coordinates must be provided for sources positions")
        if len(self._params.data_simulator.rir_generation.mic_config.pos_rcv) == 0:
            raise Exception("No provided mic positions")
        for sublist in self._params.data_simulator.rir_generation.room_config.pos_src:
            if len(sublist) != 3:
                raise Exception("Three coordinates must be provided for mic positions")

        if self._params.data_simulator.session_config.num_speakers != len(
            self._params.data_simulator.rir_generation.room_config.pos_src
        ):
            raise Exception("Number of speakers is not equal to the number of provided source positions")
        if self._params.data_simulator.rir_generation.mic_config.num_channels != len(
            self._params.data_simulator.rir_generation.mic_config.pos_rcv
        ):
            raise Exception("Number of channels is not equal to the number of provided microphone positions")

        if (
            not self._params.data_simulator.rir_generation.mic_config.orV_rcv
            and self._params.data_simulator.rir_generation.mic_config.mic_pattern != 'omni'
        ):
            raise Exception("Microphone orientations must be provided if mic_pattern != omni")
        if self._params.data_simulator.rir_generation.mic_config.orV_rcv is not None:
            if len(self._params.data_simulator.rir_generation.mic_config.orV_rcv) != len(
                self._params.data_simulator.rir_generation.mic_config.pos_rcv
            ):
                raise Exception("A different number of microphone orientations and microphone positions were provided")
            for sublist in self._params.data_simulator.rir_generation.mic_config.orV_rcv:
                if len(sublist) != 3:
                    raise Exception("Three coordinates must be provided for orientations")

    def _generate_rir_gpuRIR(self):
        """
        Create simulated RIR using the gpuRIR library

        Returns:
            RIR (tensor): Generated RIR
            RIR_pad (int): Length of padding added when convolving the RIR with an audio file
        """
        room_sz_tmp = np.array(self._params.data_simulator.rir_generation.room_config.room_sz)
        if room_sz_tmp.ndim == 2:  # randomize
            room_sz = np.zeros(room_sz_tmp.shape[0])
            for i in range(room_sz_tmp.shape[0]):
                room_sz[i] = np.random.uniform(room_sz_tmp[i, 0], room_sz_tmp[i, 1])
        else:
            room_sz = room_sz_tmp

        pos_src_tmp = np.array(self._params.data_simulator.rir_generation.room_config.pos_src)
        if pos_src_tmp.ndim == 3:  # randomize
            pos_src = np.zeros((pos_src_tmp.shape[0], pos_src_tmp.shape[1]))
            for i in range(pos_src_tmp.shape[0]):
                for j in range(pos_src_tmp.shape[1]):
                    pos_src[i] = np.random.uniform(pos_src_tmp[i, j, 0], pos_src_tmp[i, j, 1])
        else:
            pos_src = pos_src_tmp

        if self._params.data_simulator.background_noise.add_bg:
            pos_src = np.vstack((pos_src, self._params.data_simulator.rir_generation.room_config.noise_src_pos))

        mic_pos_tmp = np.array(self._params.data_simulator.rir_generation.mic_config.pos_rcv)
        if mic_pos_tmp.ndim == 3:  # randomize
            mic_pos = np.zeros((mic_pos_tmp.shape[0], mic_pos_tmp.shape[1]))
            for i in range(mic_pos_tmp.shape[0]):
                for j in range(mic_pos_tmp.shape[1]):
                    mic_pos[i] = np.random.uniform(mic_pos_tmp[i, j, 0], mic_pos_tmp[i, j, 1])
        else:
            mic_pos = mic_pos_tmp

        orV_rcv = self._params.data_simulator.rir_generation.mic_config.orV_rcv
        if orV_rcv:  # not needed for omni mics
            orV_rcv = np.array(orV_rcv)
        mic_pattern = self._params.data_simulator.rir_generation.mic_config.mic_pattern
        abs_weights = self._params.data_simulator.rir_generation.absorbtion_params.abs_weights
        T60 = self._params.data_simulator.rir_generation.absorbtion_params.T60
        att_diff = self._params.data_simulator.rir_generation.absorbtion_params.att_diff
        att_max = self._params.data_simulator.rir_generation.absorbtion_params.att_max
        sr = self._params.data_simulator.sr

        beta = beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights)  # Reflection coefficients
        Tdiff = att2t_SabineEstimator(att_diff, T60)  # Time to start the diffuse reverberation model [s]
        Tmax = att2t_SabineEstimator(att_max, T60)  # Time to stop the simulation [s]
        nb_img = t2n(Tdiff, room_sz)  # Number of image sources in each dimension
        RIR = simulateRIR(
            room_sz, beta, pos_src, mic_pos, nb_img, Tmax, sr, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern
        )
        RIR_pad = RIR.shape[2] - 1
        return RIR, RIR_pad

    def _generate_rir_pyroomacoustics(self) -> Tuple[torch.Tensor, int]:
        """
        Create simulated RIR using the pyroomacoustics library

        Returns:
            RIR (tensor): Generated RIR
            RIR_pad (int): Length of padding added when convolving the RIR with an audio file
        """

        rt60 = self._params.data_simulator.rir_generation.absorbtion_params.T60  # The desired reverberation time
        sr = self._params.data_simulator.sr

        room_sz_tmp = np.array(self._params.data_simulator.rir_generation.room_config.room_sz)
        if room_sz_tmp.ndim == 2:  # randomize
            room_sz = np.zeros(room_sz_tmp.shape[0])
            for i in range(room_sz_tmp.shape[0]):
                room_sz[i] = np.random.uniform(room_sz_tmp[i, 0], room_sz_tmp[i, 1])
        else:
            room_sz = room_sz_tmp

        pos_src_tmp = np.array(self._params.data_simulator.rir_generation.room_config.pos_src)
        if pos_src_tmp.ndim == 3:  # randomize
            pos_src = np.zeros((pos_src_tmp.shape[0], pos_src_tmp.shape[1]))
            for i in range(pos_src_tmp.shape[0]):
                for j in range(pos_src_tmp.shape[1]):
                    pos_src[i] = np.random.uniform(pos_src_tmp[i, j, 0], pos_src_tmp[i, j, 1])
        else:
            pos_src = pos_src_tmp

        # We invert Sabine's formula to obtain the parameters for the ISM simulator
        e_absorption, max_order = pra.inverse_sabine(rt60, room_sz)
        room = pra.ShoeBox(room_sz, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)

        if self._params.data_simulator.background_noise.add_bg:
            pos_src = np.vstack((pos_src, self._params.data_simulator.rir_generation.room_config.noise_src_pos))
        for pos in pos_src:
            room.add_source(pos)

        # currently only supports omnidirectional microphones
        mic_pattern = self._params.data_simulator.rir_generation.mic_config.mic_pattern
        if self._params.data_simulator.rir_generation.mic_config.mic_pattern == 'omni':
            mic_pattern = DirectivityPattern.OMNI
            dir_vec = DirectionVector(azimuth=0, colatitude=90, degrees=True)
        dir_obj = CardioidFamily(
            orientation=dir_vec,
            pattern_enum=mic_pattern,
        )

        mic_pos_tmp = np.array(self._params.data_simulator.rir_generation.mic_config.pos_rcv)
        if mic_pos_tmp.ndim == 3:  # randomize
            mic_pos = np.zeros((mic_pos_tmp.shape[0], mic_pos_tmp.shape[1]))
            for i in range(mic_pos_tmp.shape[0]):
                for j in range(mic_pos_tmp.shape[1]):
                    mic_pos[i] = np.random.uniform(mic_pos_tmp[i, j, 0], mic_pos_tmp[i, j, 1])
        else:
            mic_pos = mic_pos_tmp

        room.add_microphone_array(mic_pos.T, directivity=dir_obj)

        room.compute_rir()
        rir_pad = 0
        for channel in room.rir:
            for pos in channel:
                if pos.shape[0] - 1 > rir_pad:
                    rir_pad = pos.shape[0] - 1
        return room.rir, rir_pad

    def _convolve_rir(self, input, speaker_turn: int, RIR: torch.Tensor) -> Tuple[list, int]:
        """
        Augment one sentence (or background noise segment) using a synthetic RIR.

        Args:
            input (torch.tensor): Input audio.
            speaker_turn (int): Current speaker turn.
            RIR (torch.tensor): Room Impulse Response.
        Returns:
            output_sound (list): List of tensors containing augmented audio
            length (int): Length of output audio channels (or of the longest if they have different lengths)
        """
        output_sound = []
        length = 0
        for channel in range(self._params.data_simulator.rir_generation.mic_config.num_channels):
            if self._params.data_simulator.rir_generation.toolkit == 'gpuRIR':
                out_channel = convolve(input, RIR[speaker_turn, channel, : len(input)]).tolist()
            elif self._params.data_simulator.rir_generation.toolkit == 'pyroomacoustics':
                out_channel = convolve(input, RIR[channel][speaker_turn][: len(input)]).tolist()
            if len(out_channel) > length:
                length = len(out_channel)
            output_sound.append(torch.tensor(out_channel))
        return output_sound, length

    def _generate_session(
        self,
        idx: int,
        basepath: str,
        filename: str,
        speaker_ids: list,
        speaker_wav_align_map: dict,
        noise_samples: list,
        device: torch.device,
        enforce_counter: int = 2,
    ):
        """
        Generate a multispeaker audio session and corresponding label files.

        Args:
            idx (int): Index for current session (out of total number of sessions).
            basepath (str): Path to output directory.
            filename (str): Filename for output files.
            speaker_ids (list): List of speaker IDs that will be used in this session.
            speaker_wav_align_map (dict): Dictionary containing speaker IDs and their corresponding wav filepath and alignments.
            noise_samples (list): List of randomly sampled noise source files that will be used for generating this session.
            device (torch.device): Device to use for generating this session.
            enforce_counter (int): In enforcement mode, dominance is increased by a factor of enforce_counter for unrepresented speakers
        """
        random_seed = self._params.data_simulator.random_seed
        np.random.seed(random_seed + idx)

        self._device = device
        speaker_dominance = self._get_speaker_dominance()  # randomly determine speaker dominance
        base_speaker_dominance = np.copy(speaker_dominance)
        self._set_speaker_volume()

        running_len_samples, prev_len_samples = 0, 0  # starting point for each sentence
        prev_speaker = None
        self.annotator.init_annotation_lists()
        self._noise_samples = noise_samples
        self._furthest_sample = [0 for n in range(self._params.data_simulator.session_config.num_speakers)]

        # Room Impulse Response Generation (performed once per batch of sessions)
        if self._params.data_simulator.rir_generation.toolkit == 'gpuRIR':
            RIR, RIR_pad = self._generate_rir_gpuRIR()
        elif self._params.data_simulator.rir_generation.toolkit == 'pyroomacoustics':
            RIR, RIR_pad = self._generate_rir_pyroomacoustics()
        else:
            raise Exception("Toolkit must be pyroomacoustics or gpuRIR")

        # hold enforce until all speakers have spoken
        enforce_time = np.random.uniform(
            self._params.data_simulator.speaker_enforcement.enforce_time[0],
            self._params.data_simulator.speaker_enforcement.enforce_time[1],
        )
        enforce = self._params.data_simulator.speaker_enforcement.enforce_num_speakers

        session_len_samples = int(
            (self._params.data_simulator.session_config.session_length * self._params.data_simulator.sr)
        )
        array = torch.zeros((session_len_samples, self._params.data_simulator.rir_generation.mic_config.num_channels))
        is_speech = torch.zeros(session_len_samples)

        while running_len_samples < session_len_samples or enforce:
            # Step 1: Prepare parameters for sentence generation
            # Enforce speakers depending on running length
            if running_len_samples > enforce_time * session_len_samples and enforce:
                speaker_dominance, enforce = self._increase_speaker_dominance(base_speaker_dominance, enforce_counter)
                if enforce:
                    enforce_counter += 1

            # Step 2: Select a speaker
            speaker_turn = self._get_next_speaker(prev_speaker, speaker_dominance)

            # Calculate parameters for building a sentence (only add if remaining length >  specific time)
            max_samples_in_sentence = (
                session_len_samples - running_len_samples - RIR_pad
            )  # sentence will be RIR_len - 1 longer than the audio was pre-augmentation
            if enforce:
                max_samples_in_sentence = float('inf')
            elif (
                max_samples_in_sentence
                < self._params.data_simulator.session_params.end_buffer * self._params.data_simulator.sr
            ):
                break

            # Step 3: Generate a sentence
            self._build_sentence(speaker_turn, speaker_ids, speaker_wav_align_map, max_samples_in_sentence)
            augmented_sentence, length = self._convolve_rir(self._sentence, speaker_turn, RIR)

            # Step 4: Generate a time-stamp for either silence or overlap
            start = self._add_silence_or_overlap(
                speaker_turn=speaker_turn,
                prev_speaker=prev_speaker,
                start=running_len_samples,
                length=length,
                session_len_samples=session_len_samples,
                prev_len_samples=prev_len_samples,
                enforce=enforce,
            )
            # step 5: add sentence to array
            end = start + length
            if end > len(array):
                array = torch.nn.functional.pad(array, (0, 0, 0, end - len(array)))
                is_speech = torch.nn.functional.pad(is_speech, (0, end - len(is_speech)))
            is_speech[start:end] = 1

            for channel in range(self._params.data_simulator.rir_generation.mic_config.num_channels):
                len_ch = len(augmented_sentence[channel])  # accounts for how channels are slightly different lengths
                array[start : start + len_ch, channel] += augmented_sentence[channel]

            # Step 6: Build entries for output files
            new_rttm_entries = self.annotator.create_new_rttm_entry(
                self._words,
                self._alignments,
                start / self._params.data_simulator.sr,
                end / self._params.data_simulator.sr,
                speaker_ids[speaker_turn],
            )

            self.annotator.annote_lists['rttm'].extend(new_rttm_entries)

            new_json_entry = self.annotator.create_new_json_entry(
                self._text,
                os.path.join(basepath, filename + '.wav'),
                start / self._params.data_simulator.sr,
                length / self._params.data_simulator.sr,
                speaker_ids[speaker_turn],
                os.path.join(basepath, filename + '.rttm'),
                os.path.join(basepath, filename + '.ctm'),
            )
            self.annotator.annote_lists['json'].append(new_json_entry)

            new_ctm_entries = self.annotator.create_new_ctm_entry(
                filename, speaker_ids[speaker_turn], start / self._params.data_simulator.sr
            )
            self.annotator.annote_lists['ctm'].extend(new_ctm_entries)

            running_len_samples = np.maximum(running_len_samples, end)
            self._furthest_sample[speaker_turn] = running_len_samples
            prev_speaker = speaker_turn
            prev_len_samples = length

        # Step 7-1: Add optional perturbations to the whole session, such as white noise.
        if self._params.data_simulator.session_augmentor.add_sess_aug:
            # NOTE: This perturbation is not reflected in the session SNR in meta dictionary.
            array = perturb_audio(array, self._params.data_simulator.sr, self.session_augmentor)

        # Step 7-2: Additive background noise from noise manifest files
        if self._params.data_simulator.background_noise.add_bg:
            if len(self._noise_samples) > 0:
                avg_power_array = torch.mean(array[is_speech == 1] ** 2)
                bg, snr = get_background_noise(
                    len_array=len(array),
                    power_array=avg_power_array,
                    noise_samples=self._noise_samples,
                    audio_read_buffer_dict=self._audio_read_buffer_dict,
                    snr_min=self._params.data_simulator.background_noise.snr_min,
                    snr_max=self._params.data_simulator.background_noise.snr_max,
                    background_noise_snr=self._params.data_simulator.background_noise.snr,
                    seed=(random_seed + idx),
                    device=self._device,
                )
                array += bg
            length = array.shape[0]
            bg, snr = self._get_background(length, avg_power_array)
            augmented_bg, _ = self._convolve_rir(bg, -1, RIR)
            for channel in range(self._params.data_simulator.rir_generation.mic_config.num_channels):
                array[:, channel] += augmented_bg[channel][:length]
        else:
            snr = "N/A"

        # Step 7: Normalize and write to disk
        array = normalize_audio(array)

        if torch.is_tensor(array):
            array = array.cpu().numpy()
        sf.write(os.path.join(basepath, filename + '.wav'), array, self._params.data_simulator.sr)

        self.annotator.write_annotation_files(
            basepath=basepath,
            filename=filename,
            meta_data=self._get_session_meta_data(array=array, snr=snr),
        )

        del array
        self.clean_up()
        return basepath, filename
