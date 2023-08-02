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
import itertools
import multiprocessing
import os
import random
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union

import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from numpy.random import default_rng
from omegaconf import DictConfig, OmegaConf
from scipy.signal import convolve
from scipy.signal.windows import cosine, hamming, hann
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.audio_utils import db2mag, generate_approximate_noise_field, mag2db, pow2db, rms
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
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
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
                (self._sentence, audio_file[start_cutoff + start_window_amount : start_cutoff + prev_dur_samples],), 0,
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
                prev_dur_samples, remaining_dur_samples, len(audio_file[start_cutoff + prev_dur_samples :]),
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
                start=start, length=length, array=array, is_speech=is_speech,
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
                start=int(start / self._params.data_simulator.sr),
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
            basepath=basepath, filename=filename, meta_data=self._get_session_meta_data(array=array, snr=snr),
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
                    self._noise_samples = self.sampler.sample_noise_manifest(noise_manifest=source_noise_manifest,)
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
        dir_obj = CardioidFamily(orientation=dir_vec, pattern_enum=mic_pattern,)

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
            basepath=basepath, filename=filename, meta_data=self._get_session_meta_data(array=array, snr=snr),
        )

        del array
        self.clean_up()
        return basepath, filename


def check_angle(key: str, val: Union[float, Iterable[float]]) -> bool:
    """Check if the angle value is within the expected range. Input
    values are in degrees.

    Note:
        azimuth: angle between a projection on the horizontal (xy) plane and
                positive x axis. Increases counter-clockwise. Range: [-180, 180].
        elevation: angle between a vector an its projection on the horizontal (xy) plane.
                Positive above, negative below, i.e., north=+90, south=-90. Range: [-90, 90]
        yaw: rotation around the z axis. Defined accoding to right-hand rule.
            Range: [-180, 180]
        pitch: rotation around the yʹ axis. Defined accoding to right-hand rule.
            Range: [-90, 90]
        roll: rotation around the xʺ axis. Defined accoding to right-hand rule.
            Range: [-180, 180]

    Args:
        key: angle type
        val: values in degrees

    Returns:
        True if all values are within the expected range.
    """
    if np.isscalar(val):
        min_val = max_val = val
    else:
        min_val = min(val)
        max_val = max(val)

    if key == 'azimuth' and -180 <= min_val <= max_val <= 180:
        return True
    if key == 'elevation' and -90 <= min_val <= max_val <= 90:
        return True
    if key == 'yaw' and -180 <= min_val <= max_val <= 180:
        return True
    if key == 'pitch' and -90 <= min_val <= max_val <= 90:
        return True
    if key == 'roll' and -180 <= min_val <= max_val <= 180:
        return True

    raise ValueError(f'Invalid value for angle {key} = {val}')


def wrap_to_180(angle: float) -> float:
    """Wrap an angle to range ±180 degrees.

    Args:
        angle: angle in degrees

    Returns:
        Angle in degrees wrapped to ±180 degrees.
    """
    return angle - np.floor(angle / 360 + 1 / 2) * 360


class ArrayGeometry(object):
    """A class to simplify handling of array geometry.
    
    Supports translation and rotation of the array and calculation of
    spherical coordinates of a given point relative to the internal
    coordinate system of the array.

    Args:
        mic_positions: 3D coordinates, with shape (num_mics, 3)
        center: optional position of the center of the array. Defaults to the average of the coordinates.
        internal_cs: internal coordinate system for the array relative to the global coordinate system.
                    Defaults to (x, y, z), and is rotated with the array.
    """

    def __init__(
        self,
        mic_positions: Union[np.ndarray, List],
        center: Optional[np.ndarray] = None,
        internal_cs: Optional[np.ndarray] = None,
    ):
        if isinstance(mic_positions, Iterable):
            mic_positions = np.array(mic_positions)

        if not mic_positions.ndim == 2:
            raise ValueError(
                f'Expecting a 2D array specifying mic positions, but received {mic_positions.ndim}-dim array'
            )

        if not mic_positions.shape[1] == 3:
            raise ValueError(f'Expecting 3D positions, but received {mic_positions.shape[1]}-dim positions')

        mic_positions_center = np.mean(mic_positions, axis=0)
        self.centered_positions = mic_positions - mic_positions_center
        self.center = mic_positions_center if center is None else center

        # Internal coordinate system
        if internal_cs is None:
            # Initially aligned with the global
            self.internal_cs = np.eye(3)
        else:
            self.internal_cs = internal_cs

    @property
    def num_mics(self):
        """Return the number of microphones for the current array.
        """
        return self.centered_positions.shape[0]

    @property
    def positions(self):
        """Absolute positions of the microphones.
        """
        return self.centered_positions + self.center

    @property
    def internal_positions(self):
        """Positions in the internal coordinate system.
        """
        return np.matmul(self.centered_positions, self.internal_cs.T)

    @property
    def radius(self):
        """Radius of the array, relative to the center.
        """
        return max(np.linalg.norm(self.centered_positions, axis=1))

    @staticmethod
    def get_rotation(yaw: float = 0, pitch: float = 0, roll: float = 0) -> Rotation:
        """Get a Rotation object for given angles.

        All angles are defined according to the right-hand rule.

        Args:
            yaw: rotation around the z axis
            pitch: rotation around the yʹ axis
            roll: rotation around the xʺ axis

        Returns:
            A rotation object constructed using the provided angles.
        """
        check_angle('yaw', yaw)
        check_angle('pitch', pitch)
        check_angle('roll', roll)

        return Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True)

    def translate(self, to: np.ndarray):
        """Translate the array center to a new point.

        Translation does not change the centered positions or the internal coordinate system.

        Args:
            to: 3D point, shape (3,)
        """
        self.center = to

    def rotate(self, yaw: float = 0, pitch: float = 0, roll: float = 0):
        """Apply rotation on the mic array.

        This rotates the centered microphone positions and the internal
        coordinate system, it doesn't change the center of the array.

        All angles are defined according to the right-hand rule.
        For example, this means that a positive pitch will result in a rotation from z
        to x axis, which will result in a reduced elevation with respect to the global
        horizontal plane.

        Args:
            yaw: rotation around the z axis
            pitch: rotation around the yʹ axis
            roll: rotation around the xʺ axis
        """
        # construct rotation using TB angles
        rotation = self.get_rotation(yaw=yaw, pitch=pitch, roll=roll)

        # rotate centered positions
        self.centered_positions = rotation.apply(self.centered_positions)

        # apply the same transformation on the internal coordinate system
        self.internal_cs = rotation.apply(self.internal_cs)

    def new_rotated_array(self, yaw: float = 0, pitch: float = 0, roll: float = 0):
        """Create a new array by rotating this array.

        Args:
            yaw: rotation around the z axis
            pitch: rotation around the yʹ axis
            roll: rotation around the xʺ axis

        Returns:
            A new ArrayGeometry object constructed using the provided angles.
        """
        new_array = ArrayGeometry(mic_positions=self.positions, center=self.center, internal_cs=self.internal_cs)
        new_array.rotate(yaw=yaw, pitch=pitch, roll=roll)
        return new_array

    def spherical_relative_to_array(
        self, point: np.ndarray, use_internal_cs: bool = True
    ) -> Tuple[float, float, float]:
        """Return spherical coordinates of a point relative to the internal coordinate system.

        Args:
            point: 3D coordinate, shape (3,)
            use_internal_cs: Calculate position relative to the internal coordinate system.
                            If `False`, the positions will be calculated relative to the
                            external coordinate system centered at `self.center`.

        Returns:
            A tuple (distance, azimuth, elevation) relative to the mic array.
        """
        rel_position = point - self.center
        distance = np.linalg.norm(rel_position)

        if use_internal_cs:
            # transform from the absolute coordinate system to the internal coordinate system
            rel_position = np.matmul(self.internal_cs, rel_position)

        # get azimuth
        azimuth = np.arctan2(rel_position[1], rel_position[0]) / np.pi * 180
        # get elevation
        elevation = np.arcsin(rel_position[2] / distance) / np.pi * 180

        return distance, azimuth, elevation

    def __str__(self):
        with np.printoptions(precision=3, suppress=True):
            desc = f"{type(self)}:\ncenter =\n{self.center}\ncentered positions =\n{self.centered_positions}\nradius = \n{self.radius:.3}\nabsolute positions =\n{self.positions}\ninternal coordinate system =\n{self.internal_cs}\n\n"
        return desc

    def plot(self, elev=30, azim=-55, mic_size=25):
        """Plot microphone positions.

        Args:
            elev: elevation for the view of the plot
            azim: azimuth for the view of the plot
            mic_size: size of the microphone marker in the plot
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # show mic positions
        for m in range(self.num_mics):
            # show mic
            ax.scatter(
                self.positions[m, 0],
                self.positions[m, 1],
                self.positions[m, 2],
                marker='o',
                c='black',
                s=mic_size,
                depthshade=False,
            )
            # add label
            ax.text(self.positions[m, 0], self.positions[m, 1], self.positions[m, 2], str(m), c='red', zorder=10)

        # show the internal coordinate system
        ax.quiver(
            self.center[0],
            self.center[1],
            self.center[2],
            self.internal_cs[:, 0],
            self.internal_cs[:, 1],
            self.internal_cs[:, 2],
            length=self.radius,
            label='internal cs',
            normalize=False,
            linestyle=':',
            linewidth=1.0,
        )
        for dim, label in enumerate(['x′', 'y′', 'z′']):
            label_pos = self.center + self.radius * self.internal_cs[dim]
            ax.text(label_pos[0], label_pos[1], label_pos[2], label, tuple(self.internal_cs[dim]), c='blue')
        try:
            # Unfortunately, equal aspect ratio has been added very recently to Axes3D
            ax.set_aspect('equal')
        except NotImplementedError:
            logging.warning('Equal aspect ratio not supported by Axes3D')
        # Set view
        ax.view_init(elev=elev, azim=azim)
        # Set reasonable limits for all axes, even for the case of an unequal aspect ratio
        ax.set_xlim([self.center[0] - self.radius, self.center[0] + self.radius])
        ax.set_ylim([self.center[1] - self.radius, self.center[1] + self.radius])
        ax.set_zlim([self.center[2] - self.radius, self.center[2] + self.radius])

        ax.set_xlabel('x/m')
        ax.set_ylabel('y/m')
        ax.set_zlabel('z/m')
        ax.set_title('Microphone positions')
        ax.legend()
        plt.show()


def convert_placement_to_range(
    placement: dict, room_dim: Iterable[float], object_radius: float = 0
) -> List[List[float]]:
    """Given a placement dictionary, return ranges for each dimension.

    Args:
        placement: dictionary containing x, y, height, and min_to_wall
        room_dim: dimensions of the room, shape (3,)
        object_radius: radius of the object to be placed

    Returns
        List with a range of values for each dimensions.
    """
    if not np.all(np.array(room_dim) > 0):
        raise ValueError(f'Room dimensions must be positive: {room_dim}')

    if object_radius < 0:
        raise ValueError(f'Object radius must be non-negative: {object_radius}')

    placement_range = [None] * 3
    min_to_wall = placement.get('min_to_wall', 0)

    if min_to_wall < 0:
        raise ValueError(f'Min distance to wall must be positive: {min_to_wall}')

    for idx, key in enumerate(['x', 'y', 'height']):
        # Room dimension
        dim = room_dim[idx]
        # Construct the range
        val = placement.get(key)
        if val is None:
            # No constrained specified on the coordinate of the mic center
            min_val, max_val = 0, dim
        elif np.isscalar(val):
            min_val = max_val = val
        else:
            if len(val) != 2:
                raise ValueError(f'Invalid value for placement for dim {idx}/{key}: {str(placement)}')
            min_val, max_val = val

        # Make sure the array is not too close to a wall
        min_val = max(min_val, min_to_wall + object_radius)
        max_val = min(max_val, dim - min_to_wall - object_radius)

        if min_val > max_val or min(min_val, max_val) < 0:
            raise ValueError(f'Invalid range dim {idx}/{key}: min={min_val}, max={max_val}')

        placement_range[idx] = [min_val, max_val]

    return placement_range


class RIRCorpusGenerator(object):
    """Creates a corpus of RIRs based on a defined configuration of rooms and microphone array.

    RIRs are generated using `generate` method.
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: dictionary with parameters of the simulation
        """
        logging.info("Initialize RIRCorpusGenerator")
        self._cfg = cfg
        self.check_cfg()

    @property
    def cfg(self):
        """Property holding the internal config of the object.

        Note:
            Changes to this config are not reflected in the state of the object.
            Please create a new model with the updated config.
        """
        return self._cfg

    @property
    def sample_rate(self):
        return self._cfg.sample_rate

    @cfg.setter
    def cfg(self, cfg):
        """Property holding the internal config of the object.

        Note:
            Changes to this config are not reflected in the state of the object.
            Please create a new model with the updated config.
        """
        self._cfg = cfg

    def check_cfg(self):
        """
        Checks provided configuration to ensure it has the minimal required
        configuration the values are in a reasonable range.
        """
        # sample rate
        sample_rate = self.cfg.get('sample_rate')
        if sample_rate is None:
            raise ValueError('Sample rate not provided.')
        elif sample_rate < 0:
            raise ValueError(f'Sample rate must to be positive: {sample_rate}')

        # room configuration
        room_cfg = self.cfg.get('room')
        if room_cfg is None:
            raise ValueError('Room configuration not provided')

        if room_cfg.get('num') is None:
            raise ValueError('Number of rooms per subset not provided')

        if room_cfg.get('dim') is None:
            raise ValueError('Room dimensions not provided')

        for idx, key in enumerate(['width', 'length', 'height']):
            dim = room_cfg.dim.get(key)

            if dim is None:
                # not provided
                raise ValueError(f'Room {key} needs to be a scalar or a range, currently it is None')
            elif np.isscalar(dim) and dim <= 0:
                # fixed dimension
                raise ValueError(f'A fixed dimension must be positive for {key}: {dim}')
            elif len(dim) != 2 or not 0 < dim[0] < dim[1]:
                # not a valid range
                raise ValueError(f'Range must be specified with two positive increasing elements for {key}: {dim}')

        rt60 = room_cfg.get('rt60')
        if rt60 is None:
            # not provided
            raise ValueError(f'RT60 needs to be a scalar or a range, currently it is None')
        elif np.isscalar(rt60) and rt60 <= 0:
            # fixed dimension
            raise ValueError(f'RT60 must be positive: {rt60}')
        elif len(rt60) != 2 or not 0 < rt60[0] < rt60[1]:
            # not a valid range
            raise ValueError(f'RT60 range must be specified with two positive increasing elements: {rt60}')

        # mic array
        mic_cfg = self.cfg.get('mic_array')
        if mic_cfg is None:
            raise ValueError('Mic configuration not provided')

        if mic_cfg.get('positions') == 'random':
            # Only num_mics and placement are required
            mic_cfg_keys = ['num_mics', 'placement']
        else:
            mic_cfg_keys = ['positions', 'placement', 'orientation']

        for key in mic_cfg_keys:
            if key not in mic_cfg:
                raise ValueError(f'Mic array {key} not provided')

        # source
        source_cfg = self.cfg.get('source')
        if source_cfg is None:
            raise ValueError('Source configuration not provided')

        if source_cfg.get('num') is None:
            raise ValueError('Number of sources per room not provided')
        elif source_cfg.num <= 0:
            raise ValueError(f'Number of sources must be positive: {source_cfg.num}')

        if 'placement' not in source_cfg:
            raise ValueError('Source placement dictionary not provided')

        # anechoic
        if self.cfg.get('anechoic') is None:
            raise ValueError(f'Anechoic configuratio not provided.')

    def generate_room_params(self) -> dict:
        """Generate randomized room parameters based on the provided
        configuration.
        """
        # Prepare room sim parameters
        if not PRA:
            raise ImportError('pyroomacoustics is required for room simulation')

        room_cfg = self.cfg.room

        # Prepare rt60
        if room_cfg.rt60 is None:
            raise ValueError(f'Room RT60 needs to be a scalar or a range, currently it is None')

        if np.isscalar(room_cfg.rt60):
            assert room_cfg.rt60 > 0, f'RT60 should be positive: {room_cfg.rt60}'
            rt60 = room_cfg.rt60
        elif len(room_cfg.rt60) == 2:
            assert (
                0 < room_cfg.rt60[0] <= room_cfg.rt60[1]
            ), f'Expecting two non-decreasing values for RT60, received {room_cfg.rt60}'
            rt60 = self.random.uniform(low=room_cfg.rt60[0], high=room_cfg.rt60[1])
        else:
            raise ValueError(f'Unexpected value for RT60: {room_cfg.rt60}')

        # Generate a room with random dimensions
        num_retries = self.cfg.get('num_retries', 20)

        for n in range(num_retries):

            # width, length, height
            room_dim = np.zeros(3)

            # prepare dimensions
            for idx, key in enumerate(['width', 'length', 'height']):
                # get configured dimension
                dim = room_cfg.dim[key]

                # set a value
                if dim is None:
                    raise ValueError(f'Room {key} needs to be a scalar or a range, currently it is None')
                elif np.isscalar(dim):
                    assert dim > 0, f'Dimension should be positive for {key}: {dim}'
                    room_dim[idx] = dim
                elif len(dim) == 2:
                    assert 0 < dim[0] <= dim[1], f'Expecting two non-decreasing values for {key}, received {dim}'
                    # Reduce dimension if the previous attempt failed
                    room_dim[idx] = self.random.uniform(low=dim[0], high=dim[1] - n * (dim[1] - dim[0]) / num_retries)
                else:
                    raise ValueError(f'Unexpected value for {key}: {dim}')

            try:
                # Get parameters from size and RT60
                room_absorption, room_max_order = pra.inverse_sabine(rt60, room_dim)
                break
            except Exception as e:
                logging.debug('Inverse sabine failed: %s', str(e))
                # Inverse sabine may fail if the room is too large for the selected RT60.
                # Try again by generate a smaller room.
                room_absorption = room_max_order = None
                continue

        if room_absorption is None or room_max_order is None:
            raise RuntimeError(f'Evaluation of parameters failed for RT60 {rt60}s and room size {room_dim}.')

        # Return the required values
        room_params = {
            'dim': room_dim,
            'absorption': room_absorption,
            'max_order': room_max_order,
            'rt60_theoretical': rt60,
            'anechoic_absorption': self.cfg.anechoic.absorption,
            'anechoic_max_order': self.cfg.anechoic.max_order,
            'sample_rate': self.cfg.sample_rate,
        }
        return room_params

    def generate_array(self, room_dim: Iterable[float]) -> ArrayGeometry:
        """Generate array placement for the current room and config.

        Args:
            room_dim: dimensions of the room, [width, length, height]

        Returns:
            Randomly placed microphone array.
        """
        mic_cfg = self.cfg.mic_array

        if mic_cfg.positions == 'random':
            # Create a radom set of microphones
            num_mics = mic_cfg.num_mics
            mic_positions = []

            # Each microphone is placed individually
            placement_range = convert_placement_to_range(
                placement=mic_cfg.placement, room_dim=room_dim, object_radius=0
            )

            # Randomize mic placement
            for m in range(num_mics):
                position_m = [None] * 3
                for idx in range(3):
                    position_m[idx] = self.random.uniform(low=placement_range[idx][0], high=placement_range[idx][1])
                mic_positions.append(position_m)

            mic_array = ArrayGeometry(mic_positions)

        else:
            mic_array = ArrayGeometry(mic_cfg.positions)

            # Randomize center placement
            center = np.zeros(3)
            placement_range = convert_placement_to_range(
                placement=mic_cfg.placement, room_dim=room_dim, object_radius=mic_array.radius
            )

            for idx in range(len(center)):
                center[idx] = self.random.uniform(low=placement_range[idx][0], high=placement_range[idx][1])

            # Place the array at the configured center point
            mic_array.translate(to=center)

            # Randomize orientation
            orientation = dict()
            for key in ['yaw', 'roll', 'pitch']:
                # angle for current orientation
                angle = mic_cfg.orientation[key]

                if angle is None:
                    raise ValueError(f'Mic array {key} should be a scalar or a range, currently it is set to None.')

                # check it's within the expected range
                check_angle(key, angle)

                if np.isscalar(angle):
                    orientation[key] = angle
                elif len(angle) == 2:
                    assert angle[0] <= angle[1], f"Expecting two non-decreasing values for {key}, received {angle}"
                    # generate integer values, for easier bucketing, if necessary
                    orientation[key] = self.random.uniform(low=angle[0], high=angle[1])
                else:
                    raise ValueError(f'Unexpected value for orientation {key}: {angle}')

            # Rotate the array to match the selected orientation
            mic_array.rotate(**orientation)

        return mic_array

    def generate_source_position(self, room_dim: Iterable[float]) -> List[List[float]]:
        """Generate position for all sources in a room.

        Args:
            room_dim: dimensions of a 3D shoebox room

        Returns:
            List of source positions, with each position characterized with a 3D coordinate
        """
        source_cfg = self.cfg.source
        placement_range = convert_placement_to_range(placement=source_cfg.placement, room_dim=room_dim)
        source_position = []

        for n in range(source_cfg.num):
            # generate a random point withing the range
            s_pos = [None] * 3
            for idx in range(len(s_pos)):
                s_pos[idx] = self.random.uniform(low=placement_range[idx][0], high=placement_range[idx][1])
            source_position.append(s_pos)

        return source_position

    def generate(self):
        """Generate RIR corpus.
        
        This method will prepare randomized examples based on the current configuration,
        run room simulations and save results to output_dir.
        """
        logging.info("Generate RIR corpus")

        # Initialize
        self.random = default_rng(seed=self.cfg.random_seed)

        # Prepare output dir
        output_dir = self.cfg.output_dir
        if output_dir.endswith('.yaml'):
            output_dir = output_dir[:-5]

        # Create absolute path
        logging.info('Output dir set to: %s', output_dir)

        # Generate all cases
        for subset, num_rooms in self.cfg.room.num.items():

            output_dir_subset = os.path.join(output_dir, subset)
            examples = []

            if not os.path.exists(output_dir_subset):
                logging.info('Creating output directory: %s', output_dir_subset)
                os.makedirs(output_dir_subset)
            elif os.path.isdir(output_dir_subset) and len(os.listdir(output_dir_subset)) > 0:
                raise RuntimeError(f'Output directory {output_dir_subset} is not empty.')

            # Generate examples
            for n_room in range(num_rooms):

                # room info
                room_params = self.generate_room_params()

                # array placement
                mic_array = self.generate_array(room_params['dim'])

                # source placement
                source_position = self.generate_source_position(room_params['dim'])

                # file name for the file
                room_filepath = os.path.join(output_dir_subset, f'{subset}_room_{n_room:06d}.h5')

                # prepare example
                example = {
                    'room_params': room_params,
                    'mic_array': mic_array,
                    'source_position': source_position,
                    'room_filepath': room_filepath,
                }
                examples.append(example)

            # Simulation
            if (num_workers := self.cfg.get('num_workers')) is None:
                num_workers = os.cpu_count() - 1

            if num_workers > 1:
                logging.info(f'Simulate using {num_workers} workers')
                with multiprocessing.Pool(processes=num_workers) as pool:
                    metadata = list(tqdm(pool.imap(simulate_room_kwargs, examples), total=len(examples)))

            else:
                logging.info('Simulate using a single worker')
                metadata = []
                for example in tqdm(examples, total=len(examples)):
                    metadata.append(simulate_room(**example))

            # Save manifest
            manifest_filepath = os.path.join(output_dir, f'{subset}_manifest.json')

            if os.path.exists(manifest_filepath) and os.path.isfile(manifest_filepath):
                raise RuntimeError(f'Manifest config file exists: {manifest_filepath}')

            # Make all paths in the manifest relative to the output dir
            for data in metadata:
                data['room_filepath'] = os.path.relpath(data['room_filepath'], start=output_dir)

            write_manifest(manifest_filepath, metadata)

            # Generate plots with information about generated data
            plot_filepath = os.path.join(output_dir, f'{subset}_info.png')

            if os.path.exists(plot_filepath) and os.path.isfile(plot_filepath):
                raise RuntimeError(f'Plot file exists: {plot_filepath}')

            plot_rir_manifest_info(manifest_filepath, plot_filepath=plot_filepath)

        # Save used configuration for reference
        config_filepath = os.path.join(output_dir, 'config.yaml')
        if os.path.exists(config_filepath) and os.path.isfile(config_filepath):
            raise RuntimeError(f'Output config file exists: {config_filepath}')

        OmegaConf.save(self.cfg, config_filepath, resolve=True)


def simulate_room_kwargs(kwargs: dict) -> dict:
    """Wrapper around `simulate_room` to handle kwargs.
    
    `pool.map(simulate_room_kwargs, examples)` would be
    equivalent to `pool.starstarmap(simulate_room, examples)`
    if `starstarmap` would exist.

    Args:
        kwargs: kwargs that are forwarded to `simulate_room`

    Returns:
        Dictionary with metadata, see `simulate_room`
    """
    return simulate_room(**kwargs)


def simulate_room(
    room_params: dict, mic_array: ArrayGeometry, source_position: Iterable[Iterable[float]], room_filepath: str,
) -> dict:
    """Simulate room

    Args:
        room_params: parameters of the room to be simulated
        mic_array: defines positions of the microphones
        source_positions: positions for all sources to be simulated
        room_filepath: results are saved to this path

    Returns:
        Dictionary with metadata based on simulation setup
        and simulation results. Used to create the corresponding
        manifest file.
    """
    # room with the selected parameters
    room_sim = pra.ShoeBox(
        room_params['dim'],
        fs=room_params['sample_rate'],
        materials=pra.Material(room_params['absorption']),
        max_order=room_params['max_order'],
    )

    # same geometry for generating anechoic responses
    room_anechoic = pra.ShoeBox(
        room_params['dim'],
        fs=room_params['sample_rate'],
        materials=pra.Material(room_params['anechoic_absorption']),
        max_order=room_params['anechoic_max_order'],
    )

    # Compute RIRs
    for room in [room_sim, room_anechoic]:
        # place the array
        room.add_microphone_array(mic_array.positions.T)

        # place the sources
        for s_pos in source_position:
            room.add_source(s_pos)

        # generate RIRs
        room.compute_rir()

    # Get metadata for sources
    source_distance = []
    source_azimuth = []
    source_elevation = []
    for s_pos in source_position:
        distance, azimuth, elevation = mic_array.spherical_relative_to_array(s_pos)
        source_distance.append(distance)
        source_azimuth.append(azimuth)
        source_elevation.append(elevation)

    # RIRs
    rir_dataset = {
        'rir': convert_rir_to_multichannel(room_sim.rir),
        'anechoic': convert_rir_to_multichannel(room_anechoic.rir),
    }

    # Prepare metadata dict and return
    metadata = {
        'room_filepath': room_filepath,
        'sample_rate': room_params['sample_rate'],
        'dim': room_params['dim'],
        'rir_absorption': room_params['absorption'],
        'rir_max_order': room_params['max_order'],
        'rir_rt60_theory': room_sim.rt60_theory(),
        'rir_rt60_measured': room_sim.measure_rt60().mean(axis=0),  # average across mics for each source
        'anechoic_rt60_theory': room_anechoic.rt60_theory(),
        'anechoic_rt60_measured': room_anechoic.measure_rt60().mean(axis=0),  # average across mics for each source
        'anechoic_absorption': room_params['anechoic_absorption'],
        'anechoic_max_order': room_params['anechoic_max_order'],
        'mic_positions': mic_array.positions,
        'mic_center': mic_array.center,
        'source_position': source_position,
        'source_distance': source_distance,
        'source_azimuth': source_azimuth,
        'source_elevation': source_elevation,
        'num_sources': len(source_position),
    }

    # Save simulated RIR
    save_rir_simulation(room_filepath, rir_dataset, metadata)

    return convert_numpy_to_serializable(metadata)


def save_rir_simulation(filepath: str, rir_dataset: Dict[str, List[np.array]], metadata: dict):
    """Save simulated RIRs and metadata.

    Args:
        filepath: Path to the file where the data will be saved.
        rir_dataset: Dictionary with RIR data. Each item is a set of multi-channel RIRs.
        metadata: Dictionary with related metadata.
    """
    if os.path.exists(filepath):
        raise RuntimeError(f'Output file exists: {room_filepath}')

    num_sources = metadata['num_sources']

    with h5py.File(filepath, 'w') as h5f:
        # Save RIRs, each RIR set in a separate group
        for rir_key, rir_value in rir_dataset.items():
            if len(rir_value) != num_sources:
                raise ValueError(
                    f'Each RIR dataset should have exactly {num_sources} elements. Current RIR {key} has {len(rir_value)} elements'
                )

            rir_group = h5f.create_group(rir_key)

            # RIRs for different sources are saved under [group]['idx']
            for idx, rir in enumerate(rir_value):
                rir_group.create_dataset(f'{idx}', data=rir_value[idx])

        # Save metadata
        metadata_group = h5f.create_group('metadata')
        for key, value in metadata.items():
            metadata_group.create_dataset(key, data=value)


def load_rir_simulation(filepath: str, source: int = 0, rir_key: str = 'rir') -> Tuple[np.ndarray, float]:
    """Load simulated RIRs and metadata.

    Args:
        filepath: Path to simulated RIR data
        source: Index of a source.
        rir_key: String to denote which RIR to load, if there are multiple available.

    Returns:
        Multichannel RIR as ndarray with shape (num_samples, num_channels) and scalar sample rate.
    """
    with h5py.File(filepath, 'r') as h5f:
        # Load RIR
        rir = h5f[rir_key][f'{source}'][:]

        # Load metadata
        sample_rate = h5f['metadata']['sample_rate'][()]

    return rir, sample_rate


def convert_numpy_to_serializable(data: Union[dict, float, np.ndarray]) -> Union[dict, float, np.ndarray]:
    """Convert all numpy estries to list.
    Can be used to preprocess data before writing to a JSON file.

    Args:
        data: Dictionary, array or scalar.

    Returns:
        The same structure, but converted to list if
        the input is np.ndarray, so `data` can be seralized.
    """
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = convert_numpy_to_serializable(val)
    elif isinstance(data, list):
        data = [convert_numpy_to_serializable(d) for d in data]
    elif isinstance(data, np.ndarray):
        data = data.tolist()
    elif isinstance(data, np.integer):
        data = int(data)
    elif isinstance(data, np.floating):
        data = float(data)
    elif isinstance(data, np.generic):
        data = data.item()

    return data


def convert_rir_to_multichannel(rir: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Convert RIR to a list of arrays.

    Args:
        rir: list of lists, each element is a single-channel RIR

    Returns:
        List of multichannel RIRs
    """
    num_mics = len(rir)
    num_sources = len(rir[0])

    mc_rir = [None] * num_sources

    for n_source in range(num_sources):
        rir_len = [len(rir[m][n_source]) for m in range(num_mics)]
        max_len = max(rir_len)
        mc_rir[n_source] = np.zeros((max_len, num_mics))
        for n_mic, len_mic in enumerate(rir_len):
            mc_rir[n_source][:len_mic, n_mic] = rir[n_mic][n_source]

    return mc_rir


def plot_rir_manifest_info(filepath: str, plot_filepath: str = None):
    """Plot distribution of parameters from manifest file.

    Args:
        filepath: path to a RIR corpus manifest file
        plot_filepath: path to save the plot at
    """
    metadata = read_manifest(filepath)

    # source placement
    source_distance = []
    source_azimuth = []
    source_elevation = []
    source_height = []

    # room config
    rir_rt60_theory = []
    rir_rt60_measured = []
    anechoic_rt60_theory = []
    anechoic_rt60_measured = []

    # get the required data
    for data in metadata:
        # source config
        source_distance += data['source_distance']
        source_azimuth += data['source_azimuth']
        source_elevation += data['source_elevation']
        source_height += [s_pos[2] for s_pos in data['source_position']]

        # room config
        rir_rt60_theory.append(data['rir_rt60_theory'])
        rir_rt60_measured += data['rir_rt60_measured']
        anechoic_rt60_theory.append(data['anechoic_rt60_theory'])
        anechoic_rt60_measured += data['anechoic_rt60_measured']

    # plot
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 4, 1)
    plt.hist(source_distance, label='distance')
    plt.xlabel('distance / m')
    plt.ylabel('# examples')
    plt.title('Source-to-array center distance')

    plt.subplot(2, 4, 2)
    plt.hist(source_azimuth, label='azimuth')
    plt.xlabel('azimuth / deg')
    plt.ylabel('# examples')
    plt.title('Source-to-array center azimuth')

    plt.subplot(2, 4, 3)
    plt.hist(source_elevation, label='elevation')
    plt.xlabel('elevation / deg')
    plt.ylabel('# examples')
    plt.title('Source-to-array center elevation')

    plt.subplot(2, 4, 4)
    plt.hist(source_height, label='source height')
    plt.xlabel('height / m')
    plt.ylabel('# examples')
    plt.title('Source height')

    plt.subplot(2, 4, 5)
    plt.hist(rir_rt60_theory, label='theory')
    plt.xlabel('RT60 / s')
    plt.ylabel('# examples')
    plt.title('RT60 theory')

    plt.subplot(2, 4, 6)
    plt.hist(rir_rt60_measured, label='measured')
    plt.xlabel('RT60 / s')
    plt.ylabel('# examples')
    plt.title('RT60 measured')

    plt.subplot(2, 4, 7)
    plt.hist(anechoic_rt60_theory, label='theory')
    plt.xlabel('RT60 / s')
    plt.ylabel('# examples')
    plt.title('RT60 theory (anechoic)')

    plt.subplot(2, 4, 8)
    plt.hist(anechoic_rt60_measured, label='measured')
    plt.xlabel('RT60 / s')
    plt.ylabel('# examples')
    plt.title('RT60 measured (anechoic)')

    for n in range(8):
        plt.subplot(2, 4, n + 1)
        plt.grid()
        plt.legend(loc='lower left')

    plt.tight_layout()

    if plot_filepath is not None:
        plt.savefig(plot_filepath)
        plt.close()
        logging.info('Plot saved at %s', plot_filepath)


class RIRMixGenerator(object):
    """Creates a dataset of mixed signals at the microphone
    by combining target speech, background noise and interference.

    Correspnding signals are are generated and saved
    using the `generate` method.

    Input configuration is expexted to have the following structure
    ```
    sample_rate: sample rate used for simulation
    room:
        subset: manifest for RIR data
    target:
        subset: manifest for target source data
    noise:
        subset: manifest for noise data
    interference:
        subset: manifest for interference data
        interference_probability: probability that interference is present
        max_num_interferers: max number of interferers, randomly selected between 0 and max
    mix:
        subset:
            num: number of examples to generate
            rsnr: range of RSNR
            rsir: range of RSIR
        ref_mic: reference microphone
        ref_mic_rms: desired RMS at ref_mic
    ```
    """

    def __init__(self, cfg: DictConfig):
        """
        Instantiate a RIRMixGenerator object.

        Args:
            cfg: generator configuration defining data for room,
                 target signal, noise, interference and mixture
        """
        logging.info("Initialize RIRMixGenerator")
        self._cfg = cfg
        self.check_cfg()

        self.subsets = self.cfg.room.keys()
        logging.info('Initialized with %d subsets: %s', len(self.subsets), str(self.subsets))

        # load manifests
        self.metadata = dict()
        for subset in self.subsets:
            subset_data = dict()

            logging.info('Loading data for %s', subset)
            for key in ['room', 'target', 'noise', 'interference']:
                try:
                    subset_data[key] = read_manifest(self.cfg[key][subset])
                    logging.info('\t%-*s: \t%d files', 15, key, len(subset_data[key]))
                except Exception as e:
                    subset_data[key] = None
                    logging.info('\t%-*s: \t0 files', 15, key)
                    logging.warning('\t\tManifest data not loaded. Exception: %s', str(e))

            self.metadata[subset] = subset_data

        logging.info('Loaded all manifests')

        self.num_retries = self.cfg.get('num_retries', 5)

    @property
    def cfg(self):
        """Property holding the internal config of the object.

        Note:
            Changes to this config are not reflected in the state of the object.
            Please create a new model with the updated config.
        """
        return self._cfg

    @property
    def sample_rate(self):
        return self._cfg.sample_rate

    @cfg.setter
    def cfg(self, cfg):
        """Property holding the internal config of the object.

        Note:
            Changes to this config are not reflected in the state of the object.
            Please create a new model with the updated config.
        """
        self._cfg = cfg

    def check_cfg(self):
        """
        Checks provided configuration to ensure it has the minimal required
        configuration the values are in a reasonable range.
        """
        # sample rate
        sample_rate = self.cfg.get('sample_rate')
        if sample_rate is None:
            raise ValueError('Sample rate not provided.')
        elif sample_rate < 0:
            raise ValueError(f'Sample rate must be positive: {sample_rate}')

        # room configuration
        room_cfg = self.cfg.get('room')
        if not room_cfg:
            raise ValueError(
                'Room configuration not provided. Expecting RIR manifests in format {subset: path_to_manifest}'
            )

        # target configuration
        target_cfg = self.cfg.get('target')
        if not target_cfg:
            raise ValueError(
                'Target configuration not provided. Expecting audio manifests in format {subset: path_to_manifest}'
            )

        for key in ['azimuth', 'elevation', 'distance']:
            value = target_cfg.get(key)

            if value is None or np.isscalar(value):
                # no constraint or a fixed dimension is ok
                pass
            elif len(value) != 2 or not value[0] < value[1]:
                # not a valid range
                raise ValueError(f'Range must be specified with two positive increasing elements for {key}: {value}')

        # noise configuration
        noise_cfg = self.cfg.get('noise')
        if not noise_cfg:
            raise ValueError(
                'Noise configuration not provided. Expecting audio manifests in format {subset: path_to_manifest}'
            )

        # interference configuration
        interference_cfg = self.cfg.get('interference')
        if not interference_cfg:
            logging.info('Interference configuration not provided.')
        else:
            interference_probability = interference_cfg.get('interference_probability', 0)
            max_num_interferers = interference_cfg.get('max_num_interferers', 0)
            min_azimuth_to_target = interference_cfg.get('min_azimuth_to_target', 0)
            if interference_probability is not None:
                if interference_probability < 0:
                    raise ValueError(
                        f'Interference probability must be non-negative. Current value: {interference_prob}'
                    )
                elif interference_probability > 0:
                    assert (
                        max_num_interferers is not None and max_num_interferers > 0
                    ), f'Max number of interferers must be positive. Current value: {max_num_interferers}'
                    assert (
                        min_azimuth_to_target is not None and min_azimuth_to_target >= 0
                    ), f'Min azimuth to target must be non-negative'

        # mix configuration
        mix_cfg = self.cfg.get('mix')
        if not mix_cfg:
            raise ValueError('Mix configuration not provided. Expecting configuration for each subset.')
        if 'ref_mic' not in mix_cfg:
            raise ValueError('Reference microphone not defined.')
        if 'ref_mic_rms' not in mix_cfg:
            raise ValueError('Reference microphone RMS not defined.')

    def generate_target(self, subset: str) -> dict:
        """
        Prepare a dictionary with target configuration.

        The output dictionary contains the following information
        ```
            room_index: index of the selected room from the RIR corpus
            room_filepath: path to the room simulation file
            source: index of the selected source for the target
            rt60: reverberation time of the selected room
            num_mics: number of microphones
            azimuth: azimuth of the target source, relative to the microphone array
            elevation: elevation of the target source, relative to the microphone array
            distance: distance of the target source, relative to the microphone array
            audio_filepath: path to the audio file for the target source
            text: text for the target source audio signal, if available
            duration: duration of the target source audio signal
        ```

        Args:
            subset: string denoting a subset which will be used to selected target
                    audio and room parameters.
        
        Returns:
            Dictionary with target configuration, including room, source index, and audio information.
        """
        # Utility function
        def select_target_source(room_metadata, room_indices):
            """Find a room and a source that satisfies the constraints.
            """
            for room_index in room_indices:
                # Select room
                room_data = room_metadata[room_index]

                # Candidate sources
                sources = self.random.choice(room_data['num_sources'], size=self.num_retries, replace=False)

                # Select target source in this room
                for source in sources:
                    # Check constraints
                    constraints_met = []
                    for constraint in ['azimuth', 'elevation', 'distance']:
                        if self.cfg.target.get(constraint) is not None:
                            # Check that the selected source is in the range
                            source_value = room_data[f'source_{constraint}'][source]
                            if self.cfg.target[constraint][0] <= source_value <= self.cfg.target[constraint][1]:
                                constraints_met.append(True)
                            else:
                                constraints_met.append(False)
                                # No need to check the remaining constraints
                                break

                    # Check if a feasible source is found
                    if all(constraints_met):
                        # A feasible source has been found
                        return source, room_index

            return None, None

        # Prepare room & source position
        room_metadata = self.metadata[subset]['room']
        room_indices = self.random.choice(len(room_metadata), size=self.num_retries, replace=False)
        source, room_index = select_target_source(room_metadata, room_indices)

        if source is None:
            raise RuntimeError(f'Could not find a feasible source given target constraints {self.cfg.target}')

        room_data = room_metadata[room_index]

        # Optional: select subset of channels
        num_available_mics = len(room_data['mic_positions'])
        if 'mic_array' in self.cfg:
            num_mics = self.cfg.mic_array['num_mics']
            mic_selection = self.cfg.mic_array['selection']

            if mic_selection == 'random':
                logging.debug('Randomly selecting %d mics', num_mics)
                selected_mics = self.random.choice(num_available_mics, size=num_mics, replace=False)
            elif isinstance(mic_selection, Iterable):
                logging.debug('Using explicitly selected mics: %s', str(mic_selection))
                assert (
                    0 <= min(mic_selection) < num_available_mics
                ), f'Expecting mic_selection in range [0,{num_available_mics}), current value: {mic_selection}'
                selected_mics = np.array(mic_selection)
            else:
                raise ValueError(f'Unexpected value for mic_selection: {mic_selection}')
        else:
            logging.debug('Using all %d available mics', num_available_mics)
            num_mics = num_available_mics
            selected_mics = np.arange(num_mics)

        # Double-check the number of mics is as expected
        assert (
            len(selected_mics) == num_mics
        ), f'Expecting {num_mics} mics, but received {len(selected_mics)} mics: {selected_mics}'
        logging.debug('Selected mics: %s', str(selected_mics))

        # Calculate distance from the source to each microphone
        mic_positions = np.array(room_data['mic_positions'])[selected_mics]
        source_position = np.array(room_data['source_position'][source])
        distance_source_to_mic = np.linalg.norm(mic_positions - source_position, axis=1)

        # Handle relative paths
        room_filepath = room_data['room_filepath']
        if not os.path.isabs(room_filepath):
            manifest_dir = os.path.dirname(self.cfg.room[subset])
            room_filepath = os.path.join(manifest_dir, room_filepath)

        target_cfg = {
            'room_index': int(room_index),
            'room_filepath': room_filepath,
            'source': source,
            'rt60': room_data['rir_rt60_measured'][source],
            'selected_mics': selected_mics.tolist(),
            # Positions
            'source_position': source_position.tolist(),
            'mic_positions': mic_positions.tolist(),
            # Relative to center of the array
            'azimuth': room_data['source_azimuth'][source],
            'elevation': room_data['source_elevation'][source],
            'distance': room_data['source_distance'][source],
            # Relative to mics
            'distance_source_to_mic': distance_source_to_mic,
        }

        return target_cfg

    def generate_interference(self, subset: str, target_cfg: dict) -> List[dict]:
        """
        Prepare a list of dictionaries with interference configuration.

        Args:
            subset: string denoting a subset which will be used to select interference audio.
            target_cfg: dictionary with target configuration. This is used to determine
                        the minimal required duration for the noise signal.
        
        Returns:
            List of dictionary with interference configuration, including source index and audio information
            for one or more interference sources.
        """
        if (interference_metadata := self.metadata[subset]['interference']) is None:
            # No interference to be configured
            return None

        # Configure interfering sources
        max_num_sources = self.cfg.interference.get('max_num_interferers', 0)
        interference_probability = self.cfg.interference.get('interference_probability', 0)

        if (
            max_num_sources >= 1
            and interference_probability > 0
            and self.random.uniform(low=0.0, high=1.0) < interference_probability
        ):
            # interference present
            num_interferers = self.random.integers(low=1, high=max_num_sources + 1)
        else:
            # interference not present
            return None

        # Room setup: same room as target
        room_index = target_cfg['room_index']
        room_data = self.metadata[subset]['room'][room_index]
        feasible_sources = list(range(room_data['num_sources']))
        # target source is not eligible
        feasible_sources.remove(target_cfg['source'])

        # Constraints for interfering sources
        min_azimuth_to_target = self.cfg.interference.get('min_azimuth_to_target', 0)

        # Prepare interference configuration
        interference_cfg = []
        for n in range(num_interferers):

            # Select a source
            source = None
            while len(feasible_sources) > 0 and source is None:

                # Select a potential source for the target
                source = self.random.choice(feasible_sources)
                feasible_sources.remove(source)

                # Check azimuth separation
                if min_azimuth_to_target > 0:
                    source_azimuth = room_data['source_azimuth'][source]
                    azimuth_diff = wrap_to_180(source_azimuth - target_cfg['azimuth'])
                    if abs(azimuth_diff) < min_azimuth_to_target:
                        # Try again
                        source = None
                        continue

            if source is None:
                logging.warning('Could not select a feasible interference source %d of %s', n, num_interferers)

                # Return what we have for now or None
                return interference_cfg if interference_cfg else None

            # Current source setup
            interfering_source = {
                'source': source,
                'selected_mics': target_cfg['selected_mics'],
                'position': room_data['source_position'][source],
                'azimuth': room_data['source_azimuth'][source],
                'elevation': room_data['source_elevation'][source],
                'distance': room_data['source_distance'][source],
            }

            # Done with interference for this source
            interference_cfg.append(interfering_source)

        return interference_cfg

    def generate_mix(self, subset: str, target_cfg: dict) -> dict:
        """Generate scaling parameters for mixing
        the target speech at the microphone, background noise
        and interference signal at the microphone.

        The output dictionary contains the following information
        ```
            rsnr: reverberant signal-to-noise ratio
            rsir: reverberant signal-to-interference ratio
            ref_mic: reference microphone for calculating the metrics
            ref_mic_rms: RMS of the signal at the reference microphone
        ```

        Args:
            subset: string denoting the subset of configuration
            target_cfg: dictionary with target configuration

        Returns:
            Dictionary containing configured RSNR, RSIR, ref_mic
            and RMS on ref_mic.
        """
        mix_cfg = dict()

        for key in ['rsnr', 'rsir', 'ref_mic', 'ref_mic_rms', 'min_duration']:
            if key in self.cfg.mix[subset]:
                # Take the value from subset config
                value = self.cfg.mix[subset].get(key)
            else:
                # Take the global value
                value = self.cfg.mix.get(key)

            if value is None:
                mix_cfg[key] = None
            elif np.isscalar(value):
                mix_cfg[key] = value
            elif len(value) == 2:
                # Select from the given range, including the upper bound
                mix_cfg[key] = self.random.integers(low=value[0], high=value[1] + 1)
            else:
                # Select one of the multiple values
                mix_cfg[key] = self.random.choice(value)

        if mix_cfg['ref_mic'] == 'closest':
            # Select the closest mic as the reference
            mix_cfg['ref_mic'] = np.argmin(target_cfg['distance_source_to_mic'])

        # Configuration for saving individual components
        mix_cfg['save'] = OmegaConf.to_object(self.cfg.mix['save']) if 'save' in self.cfg.mix else {}

        return mix_cfg

    def generate(self):
        """Generate a corpus of microphone signals by mixing target, background noise
        and interference signals.

        This method will prepare randomized examples based on the current configuration,
        run simulations and save results to output_dir.
        """
        logging.info('Generate mixed signals')

        # Initialize
        self.random = default_rng(seed=self.cfg.random_seed)

        # Prepare output dir
        output_dir = self.cfg.output_dir
        if output_dir.endswith('.yaml'):
            output_dir = output_dir[:-5]

        # Create absolute path
        logging.info('Output dir set to: %s', output_dir)

        # Generate all cases
        for subset in self.subsets:

            output_dir_subset = os.path.join(output_dir, subset)
            examples = []

            if not os.path.exists(output_dir_subset):
                logging.info('Creating output directory: %s', output_dir_subset)
                os.makedirs(output_dir_subset)
            elif os.path.isdir(output_dir_subset) and len(os.listdir(output_dir_subset)) > 0:
                raise RuntimeError(f'Output directory {output_dir_subset} is not empty.')

            num_examples = self.cfg.mix[subset].num
            logging.info('Preparing %d examples for subset %s', num_examples, subset)

            # Generate examples
            for n_example in tqdm(range(num_examples), total=num_examples, desc=f'Preparing {subset}'):
                # prepare configuration
                target_cfg = self.generate_target(subset)
                interference_cfg = self.generate_interference(subset, target_cfg)
                mix_cfg = self.generate_mix(subset, target_cfg)

                # base file name
                base_output_filepath = os.path.join(output_dir_subset, f'{subset}_example_{n_example:09d}')

                # prepare example
                example = {
                    'sample_rate': self.sample_rate,
                    'target_cfg': target_cfg,
                    'interference_cfg': interference_cfg,
                    'mix_cfg': mix_cfg,
                    'base_output_filepath': base_output_filepath,
                }

                examples.append(example)

            # Audio data
            audio_metadata = {
                'target': self.metadata[subset]['target'],
                'target_dir': os.path.dirname(self.cfg.target[subset]),  # manifest_dir
                'noise': self.metadata[subset]['noise'],
                'noise_dir': os.path.dirname(self.cfg.noise[subset]),  # manifest_dir
            }

            if interference_cfg is not None:
                audio_metadata.update(
                    {
                        'interference': self.metadata[subset]['interference'],
                        'interference_dir': os.path.dirname(self.cfg.interference[subset]),  # manifest_dir
                    }
                )

            # Simulation
            if (num_workers := self.cfg.get('num_workers')) is None:
                num_workers = os.cpu_count() - 1

            if num_workers is not None and num_workers > 1:
                logging.info(f'Simulate using {num_workers} workers')
                examples_and_audio_metadata = zip(examples, itertools.repeat(audio_metadata, len(examples)))
                with multiprocessing.Pool(processes=num_workers) as pool:
                    metadata = list(
                        tqdm(
                            pool.imap(simulate_room_mix_helper, examples_and_audio_metadata),
                            total=len(examples),
                            desc=f'Simulating {subset}',
                        )
                    )
            else:
                logging.info('Simulate using a single worker')
                metadata = []
                for example in tqdm(examples, total=len(examples), desc=f'Simulating {subset}'):
                    metadata.append(simulate_room_mix(**example, audio_metadata=audio_metadata))

            # Save manifest
            manifest_filepath = os.path.join(output_dir, f'{os.path.basename(output_dir)}_{subset}.json')

            if os.path.exists(manifest_filepath) and os.path.isfile(manifest_filepath):
                raise RuntimeError(f'Manifest config file exists: {manifest_filepath}')

            # Make all paths in the manifest relative to the output dir
            for data in tqdm(metadata, total=len(metadata), desc=f'Making filepaths relative {subset}'):
                for key, val in data.items():
                    if key.endswith('_filepath') and val is not None:
                        data[key] = os.path.relpath(val, start=output_dir)

            write_manifest(manifest_filepath, metadata)

            # Generate plots with information about generated data
            plot_filepath = os.path.join(output_dir, f'{os.path.basename(output_dir)}_{subset}_info.png')

            if os.path.exists(plot_filepath) and os.path.isfile(plot_filepath):
                raise RuntimeError(f'Plot file exists: {plot_filepath}')

            plot_mix_manifest_info(manifest_filepath, plot_filepath=plot_filepath)

        # Save used configuration for reference
        config_filepath = os.path.join(output_dir, 'config.yaml')
        if os.path.exists(config_filepath) and os.path.isfile(config_filepath):
            raise RuntimeError(f'Output config file exists: {config_filepath}')

        OmegaConf.save(self.cfg, config_filepath, resolve=True)


def convolve_rir(signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve signal with a possibly multichannel IR in rir, i.e.,
    calculate the following for each channel m:

        signal_m = rir_m \ast signal

    Args:
        signal: single-channel signal (samples,)
        rir: single- or multi-channel IR, (samples,) or (samples, channels)

    Returns:
        out: same length as signal, same number of channels as rir, shape (samples, channels)
    """
    num_samples = len(signal)
    if rir.ndim == 1:
        # convolve and trim to length
        out = convolve(signal, rir)[:num_samples]
    elif rir.ndim == 2:
        num_channels = rir.shape[1]
        out = np.zeros((num_samples, num_channels))
        for m in range(num_channels):
            out[:, m] = convolve(signal, rir[:, m])[:num_samples]

    else:
        raise RuntimeError(f'RIR with {rir.ndim} not supported')

    return out


def calculate_drr(rir: np.ndarray, sample_rate: float, n_direct: List[int], n_0_ms=2.5) -> List[float]:
    """Calculate direct-to-reverberant ratio (DRR) from the measured RIR.
    
    Calculation is done as in eq. (3) from [1].

    Args:
        rir: room impulse response, shape (num_samples, num_channels)
        sample_rate: sample rate for the impulse response
        n_direct: direct path delay
        n_0_ms: window around n_direct for calculating the direct path energy

    Returns:
        Calculated DRR for each channel of the input RIR.

    References:
        [1] Eaton et al, The ACE challenge: Corpus description and performance evaluation, WASPAA 2015
    """
    # Define a window around the direct path delay
    n_0 = int(n_0_ms * sample_rate / 1000)

    len_rir, num_channels = rir.shape
    drr = [None] * num_channels
    for m in range(num_channels):

        # Window around the direct path
        dir_start = max(n_direct[m] - n_0, 0)
        dir_end = n_direct[m] + n_0

        # Power of the direct component
        pow_dir = np.sum(np.abs(rir[dir_start:dir_end, m]) ** 2) / len_rir

        # Power of the reverberant component
        pow_reverberant = (np.sum(np.abs(rir[0:dir_start, m]) ** 2) + np.sum(np.abs(rir[dir_end:, m]) ** 2)) / len_rir

        # DRR in dB
        drr[m] = pow2db(pow_dir / pow_reverberant)

    return drr


def normalize_max(x: np.ndarray, max_db: float = 0, eps: float = 1e-16) -> np.ndarray:
    """Normalize max input value to max_db full scale (±1).

    Args:
        x: input signal
        max_db: desired max magnitude compared to full scale
        eps: small regularization constant

    Returns:
        Normalized signal with max absolute value max_db. 
    """
    max_val = db2mag(max_db)
    return max_val * x / (np.max(np.abs(x)) + eps)


def simultaneously_active_rms(
    x: np.ndarray,
    y: np.ndarray,
    sample_rate: float,
    rms_threshold_db: float = -60,
    window_len_ms: float = 200,
    min_active_duration: float = 0.5,
) -> Tuple[float, float]:
    """Calculate RMS over segments where both input signals are active.
    
    Args:
        x: first input signal
        y: second input signal
        sample_rate: sample rate for input signals in Hz
        rms_threshold_db: threshold for determining activity of the signal, relative
                          to max absolute value
        window_len_ms: window length in milliseconds, used for calculating segmental RMS
        min_active_duration: minimal duration of the active segments

    Returns:
        RMS value over active segments for x and y.
    """
    if len(x) != len(y):
        raise RuntimeError(f'Expecting signals of same length: len(x)={len(x)}, len(y)={len(y)}')
    window_len = int(window_len_ms * sample_rate / 1000)
    rms_threshold = db2mag(rms_threshold_db)  # linear scale

    x_normalized = normalize_max(x)
    y_normalized = normalize_max(y)

    x_active_power = y_active_power = active_len = 0
    for start in range(0, len(x) - window_len, window_len):
        window = slice(start, start + window_len)

        # check activity on the scaled signal
        x_window_rms = rms(x_normalized[window])
        y_window_rms = rms(y_normalized[window])

        if x_window_rms > rms_threshold and y_window_rms > rms_threshold:
            # sum the power of the original non-scaled signal
            x_active_power += np.sum(np.abs(x[window]) ** 2)
            y_active_power += np.sum(np.abs(y[window]) ** 2)
            active_len += window_len

    if active_len < int(min_active_duration * sample_rate):
        raise RuntimeError(
            f'Signals are simultaneously active less than {min_active_duration} s: only {active_len/sample_rate} s'
        )

    # normalize
    x_active_power /= active_len
    y_active_power /= active_len

    return np.sqrt(x_active_power), np.sqrt(y_active_power)


def scaled_disturbance(
    signal: np.ndarray,
    disturbance: np.ndarray,
    sdr: float,
    sample_rate: float = None,
    ref_channel: int = 0,
    eps: float = 1e-16,
) -> np.ndarray:
    """
    Args:
        signal: numpy array, shape (num_samples, num_channels)
        disturbance: numpy array, same shape as signal
        sdr: desired signal-to-disturbance ration
        sample_rate: sample rate of the input signals
        ref_channel: ref mic used to calculate RMS
        eps: regularization constant

    Returns:
        Scaled disturbance, so that signal-to-disturbance ratio at ref_channel
        is approximately equal to input SDR during simultaneously active
        segment of signal and disturbance.
    """
    if signal.shape != disturbance.shape:
        raise ValueError(f'Signal and disturbance shapes do not match: {signal.shape} != {disturbance.shape}')

    # set scaling based on RMS at ref_mic
    signal_rms, disturbance_rms = simultaneously_active_rms(
        signal[:, ref_channel], disturbance[:, ref_channel], sample_rate=sample_rate
    )
    disturbance_gain = db2mag(-sdr) * signal_rms / (disturbance_rms + eps)
    # scale disturbance
    scaled_disturbance = disturbance_gain * disturbance
    return scaled_disturbance


def prepare_source_signal(
    signal_type: str,
    sample_rate: int,
    audio_data: List[dict],
    audio_dir: Optional[str] = None,
    min_duration: Optional[int] = None,
    ref_signal: Optional[np.ndarray] = None,
    mic_positions: Optional[np.ndarray] = None,
    num_retries: int = 10,
) -> tuple:
    """Prepare an audio signal for a source.

    Args:
        signal_type: 'point' or 'diffuse'
        sample_rate: Sampling rate for the signal
        audio_data: List of audio items, each is a dictionary with audio_filepath, duration, offset and optionally text
        audio_dir: Base directory for resolving paths, e.g., manifest basedir
        min_duration: Minimal duration to be loaded if ref_signal is not provided, in seconds
        ref_signal: Optional, used to determine the length of the signal
        mic_positions: Optional, used to prepare approximately diffuse signal
        num_retries: Number of retries when selecting the source files

    Returns:
        (audio_signal, metadata), where audio_signal is an ndarray and metadata is a dictionary
        with audio filepaths, durations and offsets
    """
    if not signal_type in ['point', 'diffuse']:
        raise ValueError(f'Unexpected signal type {signal_type}.')

    if audio_data is None:
        # No data to load
        return None

    metadata = {}

    if ref_signal is None:
        audio_signal = None
        # load at least one sample if min_duration is not provided
        samples_to_load = int(min_duration * sample_rate) if min_duration is not None else 1
        source_signals_metadata = {'audio_filepath': [], 'duration': [], 'offset': [], 'text': []}

        while samples_to_load > 0:
            # Select a random item and load the audio
            item = random.choice(audio_data)

            audio_filepath = item['audio_filepath']
            if not os.path.isabs(audio_filepath) and audio_dir is not None:
                audio_filepath = os.path.join(audio_dir, audio_filepath)

            # Load audio
            check_min_sample_rate(audio_filepath, sample_rate)
            audio_segment = AudioSegment.from_file(
                audio_file=audio_filepath,
                target_sr=sample_rate,
                duration=item['duration'],
                offset=item.get('offset', 0),
            )

            if signal_type == 'point':
                if audio_segment.num_channels > 1:
                    raise RuntimeError(
                        f'Expecting single-channel source signal, but received {audio_segment.num_channels}. File: {audio_filepath}'
                    )
            else:
                raise ValueError(f'Unexpected signal type {signal_type}.')

            source_signals_metadata['audio_filepath'].append(audio_filepath)
            source_signals_metadata['duration'].append(item['duration'])
            source_signals_metadata['duration'].append(item.get('offset', 0))
            source_signals_metadata['text'].append(item.get('text'))

            # not perfect, since different files may have different distributions
            segment_samples = normalize_max(audio_segment.samples)
            # concatenate
            audio_signal = (
                np.concatenate((audio_signal, segment_samples)) if audio_signal is not None else segment_samples
            )
            # remaining samples
            samples_to_load -= len(segment_samples)

        # Finally, we need only the metadata for the complete signal
        metadata = {
            'duration': sum(source_signals_metadata['duration']),
            'offset': 0,
        }

        # Add text only if all source signals have text
        if all([isinstance(tt, str) for tt in source_signals_metadata['text']]):
            metadata['text'] = ' '.join(source_signals_metadata['text'])
    else:
        # Load a signal with total_len samples and ensure it has enough simultaneous activity/overlap with ref_signal
        # Concatenate multiple files if necessary
        total_len = len(ref_signal)

        for n in range(num_retries):

            audio_signal = None
            source_signals_metadata = {'audio_filepath': [], 'duration': [], 'offset': []}

            if signal_type == 'point':
                samples_to_load = total_len
            elif signal_type == 'diffuse':
                # Load longer signal so it can be reshaped into (samples, mics) and
                # used to generate approximately diffuse noise field
                num_mics = len(mic_positions)
                samples_to_load = num_mics * total_len

            while samples_to_load > 0:
                # Select an audio file
                item = random.choice(audio_data)

                audio_filepath = item['audio_filepath']
                if not os.path.isabs(audio_filepath) and audio_dir is not None:
                    audio_filepath = os.path.join(audio_dir, audio_filepath)

                # Load audio signal
                check_min_sample_rate(audio_filepath, sample_rate)

                if (max_offset := item['duration'] - np.ceil(samples_to_load / sample_rate)) > 0:
                    # Load with a random offset if the example is longer than samples_to_load
                    offset = random.uniform(0, max_offset)
                    duration = -1
                else:
                    # Load the whole file
                    offset, duration = 0, item['duration']
                audio_segment = AudioSegment.from_file(
                    audio_file=audio_filepath, target_sr=sample_rate, duration=duration, offset=offset
                )

                # Prepare a single-channel signal
                if audio_segment.num_channels == 1:
                    # Take all samples
                    segment_samples = audio_segment.samples
                else:
                    # Take a random channel
                    selected_channel = random.choice(range(audio_segment.num_channels))
                    segment_samples = audio_segment.samples[:, selected_channel]

                source_signals_metadata['audio_filepath'].append(audio_filepath)
                source_signals_metadata['duration'].append(len(segment_samples) / sample_rate)
                source_signals_metadata['offset'].append(offset)

                # not perfect, since different files may have different distributions
                segment_samples = normalize_max(segment_samples)
                # concatenate
                audio_signal = (
                    np.concatenate((audio_signal, segment_samples)) if audio_signal is not None else segment_samples
                )
                # remaining samples
                samples_to_load -= len(segment_samples)

            if signal_type == 'diffuse' and num_mics > 1:
                try:
                    # Trim and reshape to num_mics to prepare num_mics source signals
                    audio_signal = audio_signal[: num_mics * total_len].reshape(num_mics, -1).T

                    # Make spherically diffuse noise
                    audio_signal = generate_approximate_noise_field(
                        mic_positions=np.array(mic_positions), noise_signal=audio_signal, sample_rate=sample_rate
                    )
                except Exception as e:
                    logging.info('Failed to generate approximate noise field: %s', str(e))
                    logging.info('Try again.')
                    # Try again
                    audio_signal, source_signals_metadata = None, {}
                    continue

            # Trim to length
            audio_signal = audio_signal[:total_len, ...]

            # Include the channel dimension if the reference includes it
            if ref_signal.ndim == 2 and audio_signal.ndim == 1:
                audio_signal = audio_signal[:, None]

            try:
                # Signal and ref_signal should be simultaneously active
                simultaneously_active_rms(ref_signal, audio_signal, sample_rate=sample_rate)
                # We have enough overlap
                break
            except Exception as e:
                # Signal and ref_signal are not overlapping, try again
                logging.info('Exception: %s', str(e))
                logging.info('Signals are not overlapping, try again.')
                audio_signal, source_signals_metadata = None, {}
                continue

    if audio_signal is None:
        logging.warning('Audio signal not set: %s.', signal_type)

    metadata['source_signals'] = source_signals_metadata

    return audio_signal, metadata


def check_min_sample_rate(filepath: str, sample_rate: float):
    """Make sure the file's sample rate is at least sample_rate.
    This will make sure that we have only downsampling if loading
    this file, while upsampling is not permitted.

    Args:
        filepath: path to a file
        sample_rate: desired sample rate
    """
    file_sample_rate = librosa.get_samplerate(path=filepath)
    if file_sample_rate < sample_rate:
        raise RuntimeError(
            f'Sample rate ({file_sample_rate}) is lower than the desired sample rate ({sample_rate}). File: {filepath}.'
        )


def simulate_room_mix(
    sample_rate: int,
    target_cfg: dict,
    interference_cfg: dict,
    mix_cfg: dict,
    audio_metadata: dict,
    base_output_filepath: str,
    max_amplitude: float = 0.999,
    eps: float = 1e-16,
) -> dict:
    """Simulate mixture signal at the microphone, including target, noise and
    interference signals and mixed at specific RSNR and RSIR.

    Args:
        sample_rate: Sample rate for all signals
        target_cfg: Dictionary with configuration of the target. Includes
                    room_filepath, source index, audio_filepath, duration
        noise_cfg: List of dictionaries, where each item includes audio_filepath,
                   offset and duration.
        interference_cfg: List of dictionaries, where each item contains source
                          index 
        mix_cfg: Dictionary with the mixture configuration. Includes RSNR, RSIR,
                 ref_mic and ref_mic_rms.
        audio_metadata: Dictionary with a list of files for target, noise and interference
        base_output_filepath: All output audio files will be saved with this prefix by
                              adding a diffierent suffix for each component, e.g., _mic.wav.
        max_amplitude: Maximum amplitude of the mic signal, used to prevent clipping.
        eps: Small regularization constant.

    Returns:
        Dictionary with metadata based on the mixture setup and
        simulation results. This corresponds to a line of the
        output manifest file.
    """
    # Local utilities
    def load_rir(
        room_filepath: str, source: int, selected_mics: list, sample_rate: float, rir_key: str = 'rir'
    ) -> np.ndarray:
        """Load a RIR and check that the sample rate is matching the desired sample rate

        Args:
            room_filepath: Path to a room simulation in an h5 file
            source: Index of the desired source
            sample_rate: Sample rate of the simulation
            rir_key: Key of the RIR to load from the simulation.

        Returns:
            Numpy array with shape (num_samples, num_channels)
        """
        rir, rir_sample_rate = load_rir_simulation(room_filepath, source=source, rir_key=rir_key)
        if rir_sample_rate != sample_rate:
            raise RuntimeError(
                f'RIR sample rate ({sample_rate}) is not matching the expected sample rate ({sample_rate}). File: {room_filepath}'
            )
        return rir[:, selected_mics]

    def get_early_rir(
        rir: np.ndarray, rir_anechoic: np.ndarray, sample_rate: int, early_duration: float = 0.050
    ) -> np.ndarray:
        """Return only the early part of the RIR.
        """
        early_len = int(early_duration * sample_rate)
        direct_path_delay = np.min(np.argmax(rir_anechoic, axis=0))
        rir_early = rir.copy()
        rir_early[direct_path_delay + early_len :, :] = 0
        return rir_early

    def save_audio(
        base_path: str,
        tag: str,
        audio_signal: Optional[np.ndarray],
        sample_rate: int,
        save: str = 'all',
        ref_mic: Optional[int] = None,
        format: str = 'wav',
        subtype: str = 'float',
    ):
        """Save audio signal and return filepath.
        """
        if (audio_signal is None) or (not save):
            return None

        if save == 'ref_mic':
            # save only ref_mic
            audio_signal = audio_signal[:, ref_mic]

        audio_filepath = base_path + f'_{tag}.{format}'
        sf.write(audio_filepath, audio_signal, sample_rate, subtype)

        return audio_filepath

    # Target RIRs
    target_rir = load_rir(
        target_cfg['room_filepath'],
        source=target_cfg['source'],
        selected_mics=target_cfg['selected_mics'],
        sample_rate=sample_rate,
    )
    target_rir_anechoic = load_rir(
        target_cfg['room_filepath'],
        source=target_cfg['source'],
        sample_rate=sample_rate,
        selected_mics=target_cfg['selected_mics'],
        rir_key='anechoic',
    )
    target_rir_early = get_early_rir(rir=target_rir, rir_anechoic=target_rir_anechoic, sample_rate=sample_rate)

    # Target signals
    target_signal, target_metadata = prepare_source_signal(
        signal_type='point',
        sample_rate=sample_rate,
        audio_data=audio_metadata['target'],
        audio_dir=audio_metadata['target_dir'],
        min_duration=mix_cfg['min_duration'],
    )
    source_signals_metadata = {'target': target_metadata['source_signals']}

    # Convolve target
    target_reverberant = convolve_rir(target_signal, target_rir)
    target_anechoic = convolve_rir(target_signal, target_rir_anechoic)
    target_early = convolve_rir(target_signal, target_rir_early)

    # Prepare noise signal
    noise, noise_metadata = prepare_source_signal(
        signal_type='diffuse',
        sample_rate=sample_rate,
        mic_positions=target_cfg['mic_positions'],
        audio_data=audio_metadata['noise'],
        audio_dir=audio_metadata['noise_dir'],
        ref_signal=target_reverberant,
    )
    source_signals_metadata['noise'] = noise_metadata['source_signals']

    # Prepare interference signal
    if interference_cfg is None:
        interference = None
    else:
        # Load interference signals
        interference = 0
        source_signals_metadata['interference'] = []
        for i_cfg in interference_cfg:
            # Load single-channel signal for directional interference
            i_signal, i_metadata = prepare_source_signal(
                signal_type='point',
                sample_rate=sample_rate,
                audio_data=audio_metadata['interference'],
                audio_dir=audio_metadata['interference_dir'],
                ref_signal=target_signal,
            )
            source_signals_metadata['interference'].append(i_metadata['source_signals'])
            # Load RIR from the same room as the target, but a difference source
            i_rir = load_rir(
                target_cfg['room_filepath'],
                source=i_cfg['source'],
                selected_mics=i_cfg['selected_mics'],
                sample_rate=sample_rate,
            )
            # Convolve interference
            i_reverberant = convolve_rir(i_signal, i_rir)
            # Sum
            interference += i_reverberant

    # Scale and add components of the signal
    mic = target_reverberant.copy()

    if noise is not None:
        noise = scaled_disturbance(
            signal=target_reverberant,
            disturbance=noise,
            sdr=mix_cfg['rsnr'],
            sample_rate=sample_rate,
            ref_channel=mix_cfg['ref_mic'],
        )
        # Update mic signal
        mic += noise

    if interference is not None:
        interference = scaled_disturbance(
            signal=target_reverberant,
            disturbance=interference,
            sdr=mix_cfg['rsir'],
            sample_rate=sample_rate,
            ref_channel=mix_cfg['ref_mic'],
        )
        # Update mic signal
        mic += interference

    # Set the final mic signal level
    mic_rms = rms(mic[:, mix_cfg['ref_mic']])
    global_gain = db2mag(mix_cfg['ref_mic_rms']) / (mic_rms + eps)
    mic_max = np.max(np.abs(mic))
    if (clipped_max := mic_max * global_gain) > max_amplitude:
        # Downscale the global gain to prevent clipping + adjust ref_mic_rms accordingly
        clipping_prevention_gain = max_amplitude / clipped_max
        global_gain *= clipping_prevention_gain
        mix_cfg['ref_mic_rms'] += mag2db(clipping_prevention_gain)

        logging.debug(
            'Clipping prevented for example %s (protection gain: %.2f dB)',
            base_output_filepath,
            mag2db(clipping_prevention_gain),
        )

    # save signals
    signals = {
        'mic': mic,
        'target_reverberant': target_reverberant,
        'target_anechoic': target_anechoic,
        'target_early': target_early,
        'noise': noise,
        'interference': interference,
    }

    metadata = {}

    for tag, signal in signals.items():

        if signal is not None:
            # scale all signal components with the global gain
            signal = global_gain * signal

        audio_filepath = save_audio(
            base_path=base_output_filepath,
            tag=tag,
            audio_signal=signal,
            sample_rate=sample_rate,
            save=mix_cfg['save'].get(tag, 'all'),
            ref_mic=mix_cfg['ref_mic'],
            format=mix_cfg['save'].get('format', 'wav'),
            subtype=mix_cfg['save'].get('subtype', 'float'),
        )

        if tag == 'mic':
            metadata['audio_filepath'] = audio_filepath
        else:
            metadata[tag + '_filepath'] = audio_filepath

    # Add metadata
    metadata.update(
        {
            'text': target_metadata.get('text'),
            'duration': target_metadata['duration'],
            'target_cfg': target_cfg,
            'interference_cfg': interference_cfg,
            'mix_cfg': mix_cfg,
            'ref_channel': mix_cfg.get('ref_mic'),
            'rt60': target_cfg.get('rt60'),
            'drr': calculate_drr(target_rir, sample_rate, n_direct=np.argmax(target_rir_anechoic, axis=0)),
            'rsnr': None if noise is None else mix_cfg['rsnr'],
            'rsir': None if interference is None else mix_cfg['rsir'],
            'source_signals': source_signals_metadata,
        }
    )

    return convert_numpy_to_serializable(metadata)


def simulate_room_mix_helper(example_and_audio_metadata: tuple) -> dict:
    """Wrapper around `simulate_room_mix` for pool.imap.

    Args:
        args: example and audio_metadata that are forwarded to `simulate_room_mix`

    Returns:
        Dictionary with metadata, see `simulate_room_mix`
    """
    example, audio_metadata = example_and_audio_metadata
    return simulate_room_mix(**example, audio_metadata=audio_metadata)


def plot_mix_manifest_info(filepath: str, plot_filepath: str = None):
    """Plot distribution of parameters from the manifest file.

    Args:
        filepath: path to a RIR corpus manifest file
        plot_filepath: path to save the plot at
    """
    metadata = read_manifest(filepath)

    # target info
    target_distance = []
    target_azimuth = []
    target_elevation = []
    target_duration = []

    # room config
    rt60 = []
    drr = []

    # noise
    rsnr = []
    rsir = []

    # get the required data
    for data in metadata:
        # target info
        target_distance.append(data['target_cfg']['distance'])
        target_azimuth.append(data['target_cfg']['azimuth'])
        target_elevation.append(data['target_cfg']['elevation'])
        target_duration.append(data['duration'])

        # room config
        rt60.append(data['rt60'])
        drr += data['drr']  # average DRR across all mics

        # noise
        if data['rsnr'] is not None:
            rsnr.append(data['rsnr'])

        if data['rsir'] is not None:
            rsir.append(data['rsir'])

    # plot
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 4, 1)
    plt.hist(target_distance, label='distance')
    plt.xlabel('distance / m')
    plt.ylabel('# examples')
    plt.title('Target-to-array distance')

    plt.subplot(2, 4, 2)
    plt.hist(target_azimuth, label='azimuth')
    plt.xlabel('azimuth / deg')
    plt.ylabel('# examples')
    plt.title('Target-to-array azimuth')

    plt.subplot(2, 4, 3)
    plt.hist(target_elevation, label='elevation')
    plt.xlabel('elevation / deg')
    plt.ylabel('# examples')
    plt.title('Target-to-array elevation')

    plt.subplot(2, 4, 4)
    plt.hist(target_duration, label='duration')
    plt.xlabel('time / s')
    plt.ylabel('# examples')
    plt.title('Target duration')

    plt.subplot(2, 4, 5)
    plt.hist(rt60, label='RT60')
    plt.xlabel('RT60 / s')
    plt.ylabel('# examples')
    plt.title('RT60')

    plt.subplot(2, 4, 6)
    plt.hist(drr, label='DRR')
    plt.xlabel('DRR / dB')
    plt.ylabel('# examples')
    plt.title('DRR [avg over mics]')

    if len(rsnr) > 0:
        plt.subplot(2, 4, 7)
        plt.hist(rsnr, label='RSNR')
        plt.xlabel('RSNR / dB')
        plt.ylabel('# examples')
        plt.title(f'RSNR [{100 * len(rsnr) / len(rt60):.0f}% ex]')

    if len(rsir):
        plt.subplot(2, 4, 8)
        plt.hist(rsir, label='RSIR')
        plt.xlabel('RSIR / dB')
        plt.ylabel('# examples')
        plt.title(f'RSIR [{100 * len(rsir) / len(rt60):.0f}% ex]')

    for n in range(8):
        plt.subplot(2, 4, n + 1)
        plt.grid()
        plt.legend(loc='lower left')

    plt.tight_layout()

    if plot_filepath is not None:
        plt.savefig(plot_filepath)
        plt.close()
        logging.info('Plot saved at %s', plot_filepath)
