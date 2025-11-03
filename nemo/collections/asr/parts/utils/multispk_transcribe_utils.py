# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import json
import math
import os
import time
from collections import OrderedDict
from copy import deepcopy
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import torch
from lhotse.dataset.collation import collate_matrices
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_diar_label import extract_frame_info_from_rttm, get_frame_targets_from_rttm
from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.collections.asr.modules.sortformer_modules import StreamingSortformerState
from nemo.collections.asr.parts.utils.diarization_utils import (
    OnlineEvaluation,
    get_color_palette,
    print_sentences,
    read_seglst,
    write_txt,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map as get_audio_rttm_map
from nemo.collections.asr.parts.utils.speaker_utils import get_uniqname_from_filepath, rttm_to_labels
from nemo.utils import logging


def measure_eta(func):
    """
    Measure the time taken to execute the function and print the ETA.

    Args:
        func (callable): The function to measure the ETA of.

    Returns:
        callable: The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record the end time
        eta = end_time - start_time  # Calculate the elapsed time
        logging.info(f"[ Step-{kwargs['step_num']} ] for '{func.__name__}': {eta:.4f} seconds")  # Print the ETA
        return result  # Return the original function's result

    return wrapper


def format_time(seconds: float) -> str:
    """
    Format the time in minutes and seconds.

    Args:
        seconds (float): The time in seconds.

    Returns:
        str: The time in minutes and seconds.
    """
    minutes = math.floor(seconds / 60)
    sec = seconds % 60
    return f"{minutes}:{sec:05.2f}"


def add_delay_for_real_time(
    cfg: Any,
    chunk_audio: torch.Tensor,
    session_start_time: float,
    feat_frame_count: int,
    loop_end_time: float,
    loop_start_time: float,
):
    """
    Add artificial delay for real-time mode by calculating the time difference between
    the current time and the session start time..

    Args:
        cfg (Any): The configuration object containing the parameters for the delay calculation.
        chunk_audio (torch.Tensor): The chunk audio tensor containing time-series audio data.
        session_start_time (float): The session start time in seconds.
        feat_frame_count (int): The number of features per second.
        loop_end_time (float): The loop end time in seconds.
        loop_start_time (float): The loop start time in seconds.
    """
    time_diff = max(0, (time.time() - session_start_time) - feat_frame_count * cfg.feat_len_sec)
    eta_min_sec = format_time(time.time() - session_start_time)
    logging.info(
        f"[   REAL TIME MODE   ] min:sec - {eta_min_sec} "
        f"Time difference for real-time mode: {time_diff:.4f} seconds"
    )
    time.sleep(
        max(
            0,
            (chunk_audio.shape[-1] - cfg.discarded_frames) * cfg.feat_len_sec
            - (loop_end_time - loop_start_time)
            - time_diff * cfg.finetune_realtime_ratio,
        )
    )


def write_seglst_file(seglst_dict_list: List[Dict[str, Any]], output_path: str):
    """
    Write a seglst file from the seglst dictionary list.

    Args:
        seglst_dict_list (List[Dict[str, Any]]): The list of seglst dictionaries.
            Example:
            [
                {
                    "session_id": "session_001",
                    "speaker": "speaker_1",
                    "words": "Write this to a SegLST file.",
                    "start_time": 12.34,
                    "end_time": 23.45,
                }, ...
            ]
        output_path (str): The path to the output file.
    """
    if len(seglst_dict_list) == 0:
        raise ValueError("seglst_dict_list is empty. No transcriptions were generated.")
    with open(output_path, 'w') as f:
        f.write(json.dumps(seglst_dict_list, indent=4) + '\n')
    logging.info(f"Saved the transcriptions of the streaming inference in\n:{output_path}")


def get_multi_talker_samples_from_manifest(cfg, manifest_file: str, feat_per_sec: float, max_spks: int):
    """
    Get the multi-talker samples from the manifest file and save it to a list named 'samples'.
    Also, save the rttm mask matrix to a list named 'rttms_mask_mats'.

    Args:
        cfg (DictConfig): The configuration object.
        manifest_file (str): The path to the manifest file.
        feat_per_sec (float): The number of features per second.
        max_spks (int): The maximum number of speakers.

    Returns:
        samples (list): The list of samples.
        rttms_mask_mats (list): The list of rttm mask matrices.
    """
    samples, rttms_mask_mats = [], []
    with open(manifest_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            item = json.loads(line)
            if 'audio_filepath' not in item:
                raise KeyError(f"Line {line_num}: 'audio_filepath' missing")
            if 'duration' not in item:
                raise KeyError(f"Line {line_num}: 'duration' missing")
            samples.append(item)
            if cfg.spk_supervision == "rttm":
                rttm_path = samples[-1]['rttm_filepath']
                if not rttm_path:
                    raise ValueError(f"Line {line_num}: rttm_filepath required when spk_supervision='rttm'")
                if not os.path.exists(rttm_path):
                    raise FileNotFoundError(f"Line {line_num}: RTTM file not found: {rttm_path}")

                with open(rttm_path, 'r', encoding='utf-8') as f:
                    rttm_lines = f.readlines()
                rttm_timestamps, _ = extract_frame_info_from_rttm(0, samples[-1]['duration'], rttm_lines)
                rttm_mat = get_frame_targets_from_rttm(
                    rttm_timestamps=rttm_timestamps,
                    offset=0,
                    duration=samples[-1]['duration'],
                    round_digits=3,
                    feat_per_sec=round(float(1 / feat_per_sec), 2),
                    max_spks=max_spks,
                )
                rttms_mask_mats.append(rttm_mat)
            samples[-1]['duration'] = None
            if 'offset' not in item:
                samples[-1]['offset'] = 0

    if len(rttms_mask_mats) > 0:
        rttms_mask_mats = collate_matrices(rttms_mask_mats)
    else:
        rttms_mask_mats = None
    return samples, rttms_mask_mats


def setup_diarization_model(cfg: DictConfig, map_location: Optional[str] = None) -> SortformerEncLabelModel:
    """Setup model from cfg and return diarization model and model name for next step"""
    if cfg.diar_model_path.endswith(".ckpt"):
        diar_model = SortformerEncLabelModel.load_from_checkpoint(
            checkpoint_path=cfg.diar_model_path, map_location=map_location, strict=False
        )
        model_name = os.path.splitext(os.path.basename(cfg.diar_model_path))[0]
    elif cfg.diar_model_path.endswith(".nemo"):
        diar_model = SortformerEncLabelModel.restore_from(restore_path=cfg.diar_model_path, map_location=map_location)
        model_name = os.path.splitext(os.path.basename(cfg.diar_model_path))[0]
    elif cfg.diar_pretrained_name.startswith("nvidia/"):
        diar_model = SortformerEncLabelModel.from_pretrained(cfg.diar_pretrained_name)
        model_name = os.path.splitext(os.path.basename(cfg.diar_pretrained_name))[0]
    else:
        raise ValueError("cfg.diar_model_path must end with.ckpt or.nemo!")
    return diar_model, model_name


def write_seglst(output_filepath: str, seglst_list: list) -> None:
    """
    Write the segmentation list to a file.

    Args:
        output_filepath (str): The path to the output file.
        seglst_list (list): The list of segmentation lists.
    """
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(json.dumps(seglst_list, indent=2) + "\n")


def get_new_sentence_dict(
    speaker: str,
    start_time: float,
    end_time: float,
    text: str,
    session_id: Optional[str] = None,
) -> dict:
    """
    Get a new SegLST style sentence dictionary variable.

    Args:
        speaker (str): The speaker of the sentence.
        start_time (float): The start time of the sentence.
        end_time (float): The end time of the sentence.
        text (str): The text of the sentence.
        session_id (Optional[str]): The session id of the sentence.

    Returns:
        Dict[str, Any]: A new SegLST style sentence dictionary variable.
    """
    return {
        'speaker': speaker,
        'start_time': start_time,
        'end_time': end_time,
        'words': text.lstrip(),
        'session_id': session_id,
    }


def fix_frame_time_step(cfg: Any, new_tokens: List[str], new_words: List[str], frame_inds_seq: List[int]) -> List[int]:
    """
    Adjust the frame indices sequence to match the length of new tokens.

    This function handles mismatches between the number of tokens and the frame indices sequence.
    It adjusts the frame_inds_seq to ensure it has the same length as new_tokens.

    Args:
        cfg (Any): Configuration object containing logging settings.
        new_tokens (List[str]): List of new tokens.
        new_words (List[str]): List of new words.
        frame_inds_seq (List[int]): List of frame indices.

    Returns:
        List[int]: Adjusted frame indices sequence.
    """
    if len(new_tokens) != len(frame_inds_seq):
        # Sometimes there is a mismatch in the number of tokens between the new tokens and the frame indices sequence.
        if len(frame_inds_seq) > len(new_words):
            # Get unique frame indices sequence
            frame_inds_seq = list(OrderedDict.fromkeys(frame_inds_seq))
            if len(frame_inds_seq) < len(new_tokens):
                deficit = len(new_tokens) - len(frame_inds_seq)
                frame_inds_seq = [frame_inds_seq[0]] * deficit + frame_inds_seq
            elif len(frame_inds_seq) > len(new_tokens):
                deficit = len(frame_inds_seq) - len(new_tokens)
                frame_inds_seq = frame_inds_seq[deficit:]

        elif len(frame_inds_seq) < len(new_tokens):
            deficit = len(new_tokens) - len(frame_inds_seq)
            frame_inds_seq = [frame_inds_seq[0]] * deficit + frame_inds_seq
        if cfg.log:
            logging.warning(
                f"Length of new token sequence ({len(new_tokens)}) does not match"
                f"the length of frame indices sequence ({len(frame_inds_seq)}). Skipping this chunk."
            )
    return frame_inds_seq


def get_simulated_softmax(cfg, speaker_sigmoid: torch.Tensor) -> torch.Tensor:
    """
    Simulate the softmax operation for speaker diarization.

    Args:
        cfg (Any): Configuration object containing diarization settings.
        speaker_sigmoid (torch.Tensor): Speaker sigmoid values.

    Returns:
        speaker_softmax (torch.Tensor): Speaker softmax values.
    """
    if speaker_sigmoid.ndim != 1:
        raise ValueError(f"Expected 1D tensor for speaker_sigmoid, got shape {speaker_sigmoid.shape}")
    if speaker_sigmoid.shape[0] < cfg.max_num_of_spks:
        raise ValueError(f"speaker_sigmoid size {speaker_sigmoid.shape[0]} < max_num_of_spks {cfg.max_num_of_spks}")

    speaker_sigmoid = torch.clamp(speaker_sigmoid, min=cfg.min_sigmoid_val, max=1)
    sigmoid_sum = speaker_sigmoid.sum()
    if sigmoid_sum == 0:
        logging.warning("speaker_sigmoid sum is zero, returning uniform distribution")
        speaker_softmax = torch.ones_like(speaker_sigmoid) / speaker_sigmoid.shape[0]
    else:
        speaker_softmax = speaker_sigmoid / sigmoid_sum
    speaker_softmax = speaker_softmax.cpu()
    speaker_softmax[cfg.max_num_of_spks :] = 0.0
    return speaker_softmax


def get_word_dict_content_offline(
    cfg: Any,
    word: str,
    word_index: int,
    diar_pred_out: torch.Tensor,
    time_stt_end_tuple: Tuple[int],
    frame_len: float = 0.08,
) -> Dict[str, Any]:
    """
    Generate a dictionary containing word information and speaker diarization results.

    This function processes a single word and its associated tokens to determine
    the start and end frames, speaker, and other relevant information.

    Args:
        cfg (Any): Configuration object containing diarization settings.
        word (str): The word being processed.
        word_index (int): Index of the word in the sequence.
        diar_pred_out (torch.Tensor): Diarization prediction output stream.
        time_stt_end_tuple (int): Local time step offset.

        frame_len (float, optional): Length of each frame in seconds. Defaults to 0.08.

    Returns:
        Dict[str, Any]: A dictionary containing word information and diarization results.
    """
    frame_stt, frame_end = time_stt_end_tuple

    # Edge Cases: Sometimes, repeated token indexs can lead to incorrect frame and speaker assignment.
    if frame_stt == frame_end:
        if frame_stt >= diar_pred_out.shape[0] - 1:
            frame_stt, frame_end = (diar_pred_out.shape[1] - 1, diar_pred_out.shape[0])
        else:
            frame_end = frame_stt + 1

    # Get the speaker based on the frame-wise softmax probabilities.
    stt_p, end_p = max((frame_stt + cfg.left_frame_shift), 0), (frame_end + cfg.right_frame_shift)
    speaker_sigmoid = diar_pred_out[stt_p:end_p, :].mean(dim=0)
    speaker_softmax = get_simulated_softmax(cfg, speaker_sigmoid)

    speaker_softmax[cfg.max_num_of_spks :] = 0.0
    spk_id = speaker_softmax.argmax().item()
    stt_sec, end_sec = frame_stt * frame_len, frame_end * frame_len
    word_dict = {
        "word": word,
        "word_index": word_index,
        'frame_stt': frame_stt,
        'frame_end': frame_end,
        'start_time': round(stt_sec, 3),
        'end_time': round(end_sec, 3),
        'speaker': f"speaker_{spk_id}",
        'speaker_softmax': speaker_softmax,
    }
    return word_dict


def get_word_dict_content_online(
    cfg: Any,
    word: str,
    word_index: int,
    diar_pred_out_stream: torch.Tensor,
    token_group: List[str],
    frame_inds_seq: List[int],
    time_step_local_offset: int,
    frame_len: float = 0.08,
) -> Dict[str, Any]:
    """
    Generate a dictionary containing word information and speaker diarization results.

    This function processes a single word and its associated tokens to determine
    the start and end frames, speaker, and other relevant information.

    Args:
        cfg (Any): Configuration object containing diarization settings.
        word (str): The word being processed.
        word_index (int): Index of the word in the sequence.
        diar_pred_out_stream (torch.Tensor): Diarization prediction output stream.
            Dimensions: (num_frames, max_num_of_spks)
        token_group (List[str]): Group of tokens associated with the word.
        frame_inds_seq (List[int]): Sequence of frame indices.
        time_step_local_offset (int): Local time step offset.
        frame_len (float, optional): Length of each frame in seconds. Defaults to 0.08.

    Returns:
        Dict[str, Any]: A dictionary containing word information and diarization results.
    """
    _stt, _end = time_step_local_offset, time_step_local_offset + len(token_group) - 1
    if len(token_group) == 1:
        frame_stt, frame_end = frame_inds_seq[_stt], frame_inds_seq[_stt] + 1
    else:
        try:
            frame_stt, frame_end = frame_inds_seq[_stt], frame_inds_seq[_end]
        except IndexError:
            frame_stt, frame_end = frame_inds_seq[_stt], frame_inds_seq[_stt] + 1

    # Edge Cases: Sometimes, repeated token indexs can lead to incorrect frame and speaker assignment.
    if frame_stt == frame_end:
        if frame_stt >= diar_pred_out_stream.shape[0] - 1:
            frame_stt, frame_end = (diar_pred_out_stream.shape[0] - 1, diar_pred_out_stream.shape[0])
        else:
            frame_end = frame_stt + 1

    # Get the speaker based on the frame-wise softmax probabilities.
    stt_p, end_p = max((frame_stt + cfg.left_frame_shift), 0), (frame_end + cfg.right_frame_shift)
    speaker_sigmoid = diar_pred_out_stream[stt_p:end_p, :].mean(dim=0)
    speaker_softmax = get_simulated_softmax(cfg, speaker_sigmoid)

    speaker_softmax[cfg.max_num_of_spks :] = 0.0
    spk_id = speaker_softmax.argmax().item()
    stt_sec, end_sec = frame_stt * frame_len, frame_end * frame_len
    word_dict = {
        "word": word,
        "word_index": word_index,
        'frame_stt': frame_stt,
        'frame_end': frame_end,
        'start_time': round(stt_sec, 3),
        'end_time': round(end_sec, 3),
        'speaker': f"speaker_{spk_id}",
        'speaker_softmax': speaker_softmax,
    }
    return word_dict


def get_multitoken_words(
    cfg, word_and_ts_seq: Dict[str, List], word_seq: List[str], new_words: List[str], fix_prev_words_count: int = 5
) -> Dict[str, List]:
    """
    Fix multi-token words that were not fully captured by the previous chunk window.

    This function compares the words in the current sequence with the previously processed words,
    and updates any multi-token words that may have been truncated in earlier processing.

    Args:
        cfg (DiarizationConfig): Configuration object containing verbose setting.
        word_and_ts_seq (Dict[str, List]): Dictionary containing word sequences and timestamps.
        word_seq (List[str]): List of all words processed so far.
        new_words (List[str]): List of new words in the current chunk.
        fix_prev_words_count (int, optional): Number of previous words to check. Defaults to 5.

    Returns:
        Dict[str, List]: Updated word_and_ts_seq with fixed multi-token words.
    """
    prev_start = max(0, len(word_seq) - fix_prev_words_count - len(new_words))
    prev_end = max(0, len(word_seq) - len(new_words))
    for ct, prev_word in enumerate(word_seq[prev_start:prev_end]):
        if len(word_and_ts_seq["words"]) > fix_prev_words_count - ct:
            saved_word = word_and_ts_seq["words"][-fix_prev_words_count + ct]["word"]
            if len(prev_word) > len(saved_word):
                if cfg.verbose:
                    logging.info(f"[Replacing Multi-token Word]: {saved_word} with {prev_word}")
                word_and_ts_seq["words"][-fix_prev_words_count + ct]["word"] = prev_word
    return word_and_ts_seq


def append_word_and_ts_seq(
    cfg: Any, word_idx_offset: int, word_and_ts_seq: Dict[str, Any], word_dict: Dict[str, Any]
) -> tuple[int, Dict[str, Any]]:
    """
    Append the word dictionary to the word and time-stamp sequence.

    This function updates the word_and_ts_seq dictionary by appending new word information
    and managing the buffered words and speaker count.

    Args:
        cfg (Any): Configuration object containing parameters like word_window.
        word_idx_offset (int): The current word index offset.
        word_and_ts_seq (Dict[str, Any]): Dictionary containing word sequences and related information.
        word_dict (Dict[str, Any]): Dictionary containing information about the current word.

    Returns:
        tuple[int, Dict[str, Any]]: A tuple containing the updated word_idx_offset and word_and_ts_seq.
    """
    word_and_ts_seq["words"].append(word_dict)
    word_and_ts_seq["buffered_words"].append(word_dict)
    word_and_ts_seq["speaker_count_buffer"].append(word_dict["speaker"])
    word_and_ts_seq["word_window_seq"].append(word_dict['word'])

    if len(word_and_ts_seq["words"]) >= cfg.word_window + 1:
        word_and_ts_seq["buffered_words"].pop(0)
        word_and_ts_seq["word_window_seq"].pop(0)
        word_idx_offset = 0

    word_and_ts_seq["speaker_count"] = len(set(word_and_ts_seq["speaker_count_buffer"]))
    return word_idx_offset, word_and_ts_seq


class SpeakerTaggedASR:
    def __init__(
        self,
        cfg,
        asr_model,
        diar_model,
    ):
        # Required configs, models and datasets for inference
        self.cfg = cfg
        if self.cfg.manifest_file:
            self.test_manifest_dict = get_audio_rttm_map(self.cfg.manifest_file)
        elif self.cfg.audio_file is not None:
            uniq_id = get_uniqname_from_filepath(filepath=self.cfg.audio_file)
            self.test_manifest_dict = {
                uniq_id: {'audio_filepath': self.cfg.audio_file, 'seglst_filepath': None, 'rttm_filepath': None}
            }
        else:
            raise ValueError("One of the audio_file and manifest_file should be non-empty!")

        self.asr_model = asr_model
        self.diar_model = diar_model

        # ASR speaker tagging configs
        self._fix_prev_words_count = cfg.fix_prev_words_count
        self._sentence_render_length = int(self._fix_prev_words_count + cfg.update_prev_words_sentence)
        self._frame_len_sec = 0.08
        self._initial_steps = cfg.ignored_initial_frame_steps
        self._stt_words = []
        self._init_evaluator()
        self._frame_hop_length = self.asr_model.encoder.streaming_cfg.valid_out_len

        # Multi-instance configs
        self._max_num_of_spks = cfg.get("max_num_of_spks", 4)
        self._offset_chunk_start_time = 0.0
        self._sent_break_sec = cfg.get("sent_break_sec", 5.0)

        self._att_context_size = cfg.att_context_size
        self._nframes_per_chunk = self._att_context_size[1] + 1
        self._cache_gating = cfg.get("cache_gating", False)
        self._cache_gating_buffer_size = cfg.get("cache_gating_buffer_size", 2)
        self._binary_diar_preds = cfg.binary_diar_preds

        self._masked_asr = cfg.get("masked_asr", True)
        self._use_mask_preencode = cfg.get("mask_preencode", False)
        self._single_speaker_mode = cfg.get("single_speaker_mode", False)

        self.instance_manager = MultiTalkerInstanceManager(
            asr_model=self.asr_model,
            diar_model=self.diar_model,
            max_num_of_spks=self.diar_model._cfg.max_num_of_spks,
            batch_size=cfg.batch_size,
            sent_break_sec=self._sent_break_sec,
        )
        self.n_active_speakers_per_stream = self.cfg.max_num_of_spks

    def _init_evaluator(self):
        """
        Initialize the evaluator for the offline STT and speaker diarization.
        """
        self.online_evaluators, self._word_and_ts_seq = [], {}
        for _, (uniq_id, data_dict) in enumerate(self.test_manifest_dict.items()):
            uniq_id = uniq_id.split(".")[0]  # Make sure there is no "." in the uniq_id
            self._word_and_ts_seq[uniq_id] = {
                "words": [],
                "buffered_words": [],
                "token_frame_index": [],
                "offset_count": 0,
                "status": "success",
                "sentences": None,
                "last_word_index": 0,
                "speaker_count": None,
                "transcription": None,
                "max_spk_probs": [],
                "word_window_seq": [],
                "speaker_count_buffer": [],
                "sentence_memory": {},
            }

            if 'seglst_filepath' in data_dict and data_dict['seglst_filepath'] is not None:
                ref_seglst = read_seglst(data_dict['seglst_filepath'])
            else:
                ref_seglst = None

            if 'rttm_filepath' in data_dict and data_dict['rttm_filepath'] is not None:
                ref_rttm_labels = rttm_to_labels(data_dict['rttm_filepath'])
            else:
                ref_rttm_labels = None

            eval_instance = OnlineEvaluation(
                ref_seglst=ref_seglst,
                ref_rttm_labels=ref_rttm_labels,
                hyp_seglst=None,
                collar=0.25,
                ignore_overlap=False,
                verbose=True,
            )
            self.online_evaluators.append(eval_instance)

    def _get_offset_sentence(self, session_trans_dict: Dict[str, Any], offset: int) -> Dict[str, Any]:
        """
        For the very first word in a session, get the offset sentence.

        Args:
            session_trans_dict (dict): Dictionary containing session-related information.
            offset (int): Index of the word for which the offset sentence is needed.

        Returns:
            (Dict): Dictionary containing offset sentence information.
        """
        word_dict = session_trans_dict['words'][offset]
        return {
            'session_id': session_trans_dict['uniq_id'],
            'speaker': word_dict['speaker'],
            'start_time': word_dict['start_time'],
            'end_time': word_dict['end_time'],
            'words': f"{word_dict['word']} ",
        }

    def _get_sentence(self, word_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the sentence for a given word.

        Args:
            word_dict (Dict[str, Any]): Dictionary containing word-related information.
        """
        return {
            'speaker': word_dict['speaker'],
            'start_time': word_dict['start_time'],
            'end_time': word_dict['end_time'],
            'words': '',
        }

    def get_sentences_values(self, session_trans_dict: dict, sentence_render_length: int):
        """
        Get sentences (speaker-turn-level text) for a given session and sentence render length.

        Args:
            session_trans_dict (Dict[str, Any]): Dictionary containing session-related information.
            sentence_render_length (int): Length of the sentences to be generated.

        Returns:
            sentences (List[Dict[str, Any]]): List of sentences in the session.
        """
        stt_word_index = max(0, session_trans_dict['last_word_index'] - sentence_render_length)
        if session_trans_dict['sentences'] is None:
            sentence = self._get_offset_sentence(session_trans_dict=session_trans_dict, offset=0)
            sentences = []
            session_trans_dict['last_word_index'] = stt_word_index
            session_trans_dict['sentence_memory'].update(
                {stt_word_index: (deepcopy(sentences), deepcopy(sentence), sentence['speaker'])}
            )
            prev_speaker = session_trans_dict['words'][stt_word_index]['speaker']
        else:
            (_sentences, _sentence, prev_speaker) = session_trans_dict['sentence_memory'][stt_word_index]
            sentences, sentence = deepcopy(_sentences), deepcopy(_sentence)

        for word_idx in range(stt_word_index + 1, len(session_trans_dict['words'])):
            word_dict = session_trans_dict['words'][word_idx]
            word, end_point = word_dict['word'], word_dict['end_time']
            if word_dict['speaker'] != prev_speaker:
                sentence['words'] = sentence['words'].strip()
                sentences.append(sentence)
                sentence = self._get_sentence(word_dict=session_trans_dict['words'][word_idx])
            else:
                sentence['end_time'] = end_point
            sentence['words'] += word.strip() + ' '
            sentence['words'] = sentence['words']
            sentence['session_id'] = session_trans_dict['uniq_id']
            session_trans_dict['last_word_index'] = word_idx
            prev_speaker = word_dict['speaker']
            session_trans_dict['sentence_memory'][word_idx] = (deepcopy(sentences), deepcopy(sentence), prev_speaker)
        sentence['words'] = sentence['words'].strip()
        sentences.append(sentence)
        session_trans_dict['sentences'] = sentences
        return session_trans_dict

    def merge_transcript_and_speakers(
        self, test_manifest_dict: dict, asr_hypotheses: List[Hypothesis], diar_pred_out: torch.Tensor
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Merge the transcript and speakers and generate real-time scripts if the config is set.

        Args:
            test_manifest_dict (Dict): Dictionary containing test manifest data.
            asr_hypotheses (List[Hypothesis]): List of ASR hypotheses.
            diar_pred_out (torch.Tensor): Diarization prediction output stream.

        Returns:
            transcribed_speaker_texts (List[str]): List of transcribed speaker texts.
            self._word_and_ts_seq (Dict[str, Dict[str, Any]]): Dictionary of word-level dictionaries with uniq_id as key.
        """
        transcribed_speaker_texts = [None] * len(test_manifest_dict)

        for idx, (uniq_id, _) in enumerate(test_manifest_dict.items()):
            uniq_id = uniq_id.split(".")[0]  # Make sure there is no "." in the uniq_id
            if not len(asr_hypotheses[idx].text) == 0:
                # Get the word-level dictionaries for each word in the chunk
                self._word_and_ts_seq[uniq_id] = self.get_frame_and_words_offline(
                    uniq_id=uniq_id,
                    diar_pred_out=diar_pred_out[idx].squeeze(0),
                    asr_hypothesis=asr_hypotheses[idx],
                    word_and_ts_seq=self._word_and_ts_seq[uniq_id],
                )
                if len(self._word_and_ts_seq[uniq_id]["words"]) > 0:
                    self._word_and_ts_seq[uniq_id] = self.get_sentences_values(
                        session_trans_dict=self._word_and_ts_seq[uniq_id],
                        sentence_render_length=self._sentence_render_length,
                    )
                    if self.cfg.generate_realtime_scripts:
                        transcribed_speaker_texts[idx] = print_sentences(
                            sentences=self._word_and_ts_seq[uniq_id]["sentences"],
                            color_palette=get_color_palette(),
                            params=self.cfg,
                        )
                        write_txt(
                            f'{self.cfg.print_path}'.replace(".sh", f"_{idx}.sh"),
                            transcribed_speaker_texts[idx].strip(),
                        )
        return transcribed_speaker_texts, self._word_and_ts_seq

    def get_frame_and_words_offline(
        self,
        uniq_id: str,
        diar_pred_out: torch.Tensor,
        asr_hypothesis: Hypothesis,
        word_and_ts_seq: Dict[str, Any],
    ):
        """
        Get the frame and words for each word in the chunk.

        Args:
            uniq_id (str): The unique id of the chunk.
            diar_pred_out (torch.Tensor): Diarization prediction output stream.
            asr_hypothesis (Hypothesis): ASR hypothesis.
            word_and_ts_seq (Dict[str, Any]): Pre-existing word-level dictionaries.

        Returns:
            word_and_ts_seq (Dict[str, Any]): The updated word-level dictionaries with new words.
        """
        word_and_ts_seq['uniq_id'] = uniq_id

        for word_index, hyp_word_dict in enumerate(asr_hypothesis.timestamp['word']):
            time_stt_end_tuple = (hyp_word_dict['start_offset'], hyp_word_dict['end_offset'])
            word_dict = get_word_dict_content_offline(
                cfg=self.cfg,
                word=hyp_word_dict['word'],
                word_index=word_index,
                diar_pred_out=diar_pred_out,
                time_stt_end_tuple=time_stt_end_tuple,
                frame_len=self._frame_len_sec,
            )
            word_and_ts_seq["words"].append(word_dict)
            word_and_ts_seq["speaker_count_buffer"].append(word_dict["speaker"])
            word_and_ts_seq["word_window_seq"].append(word_dict['word'])

        word_and_ts_seq["buffered_words"] = word_and_ts_seq["words"]
        word_and_ts_seq["speaker_count"] = len(set(word_and_ts_seq["speaker_count_buffer"]))
        return word_and_ts_seq

    def get_frame_and_words_online(
        self,
        uniq_id: str,
        step_num: int,
        diar_pred_out_stream: torch.Tensor,
        previous_hypothesis: Hypothesis,
        word_and_ts_seq: Dict[str, Any],
    ):
        """
        Get the frame and words for each word object in the chunk during streaming inference.

        Args:
            uniq_id (str): The unique id of the chunk.
            step_num (int): The step number of the chunk.
            diar_pred_out_stream (torch.Tensor): The diarization prediction output stream.
            previous_hypothesis (Hypothesis): The previous hypothesis.
            word_and_ts_seq (Dict[str, Any]): The word and timestamp sequence.

        Returns:
            word_and_ts_seq (Dict[str, Any]): The word and timestamp sequence.
        """
        offset = step_num * self._frame_hop_length
        word_seq = previous_hypothesis.text.split()
        new_words = word_seq[word_and_ts_seq["offset_count"] :]
        new_token_group = self.asr_model.tokenizer.text_to_tokens(new_words)
        new_tokens = list(itertools.chain(*new_token_group))
        frame_inds_seq = (torch.tensor(previous_hypothesis.timestamp) + offset).tolist()
        frame_inds_seq = fix_frame_time_step(self.cfg, new_tokens, new_words, frame_inds_seq)
        word_and_ts_seq['uniq_id'] = uniq_id

        min_len = min(len(new_words), len(frame_inds_seq))
        for idx in range(min_len):
            word_and_ts_seq["token_frame_index"].append((new_tokens[idx], frame_inds_seq[idx]))
            word_and_ts_seq["offset_count"] += 1

        time_step_local_offset, word_idx_offset = 0, 0
        word_and_ts_seq = get_multitoken_words(
            cfg=self.cfg,
            word_and_ts_seq=word_and_ts_seq,
            word_seq=word_seq,
            new_words=new_words,
            fix_prev_words_count=self._fix_prev_words_count,
        )

        # Get the FIFO queue preds to word_and_ts_seq
        for local_idx, (token_group, word) in enumerate(zip(new_token_group, new_words)):
            word_dict = get_word_dict_content_online(
                cfg=self.cfg,
                word=word,
                word_index=(len(word_and_ts_seq["words"]) + local_idx),
                diar_pred_out_stream=diar_pred_out_stream,
                token_group=token_group,
                frame_inds_seq=frame_inds_seq,
                time_step_local_offset=time_step_local_offset,
                frame_len=self._frame_len_sec,
            )
            # Count the number of speakers in the word window
            time_step_local_offset += len(token_group)
            word_idx_offset, word_and_ts_seq = append_word_and_ts_seq(
                cfg=self.cfg, word_idx_offset=word_idx_offset, word_and_ts_seq=word_and_ts_seq, word_dict=word_dict
            )
        return word_and_ts_seq

    def _add_speaker_transcriptions(
        self,
        transcriptions: list,
        speaker_transcriptions: List[str],
        word_and_ts_seq: Dict[str, Dict[str, Any]],
        test_manifest_dict: dict,
    ) -> Tuple[List[Hypothesis], List[Hypothesis]]:
        """
        Add speaker tagging into the transcriptions generated from an ASR model.

        Args:
            transcriptions (Tuple[List[Hypothesis], List[Hypothesis]]):
                Tuple containing the transcriptions and n-best transcriptions.
            speaker_transcriptions (List[str]):
                List of speaker transcriptions.
            word_and_ts_seq (Dict[str, Dict[str, Any]]):
                Dictionary of word-level dictionaries with uniq_id as key.
            test_manifest_dict (dict):
                Dictionary containing test manifest data.

        Returns:
            Tuple[List[Hypothesis], List[Hypothesis]]: Tuple containing the updated transcriptions with speaker tags.
        """
        trans_hyp, _ = transcriptions
        for sess_idx, (uniq_id, _) in enumerate(test_manifest_dict.items()):
            uniq_id = uniq_id.split(".")[0]  # Make sure there is no "." in the uniq_id
            if speaker_transcriptions[sess_idx] is not None:
                trans_hyp[sess_idx].text = speaker_transcriptions[sess_idx]
            speaker_added_word_dicts = []
            for word_idx, trans_wdict in enumerate(trans_hyp[0].timestamp['word']):
                trans_wdict_copy = deepcopy(trans_wdict)
                trans_wdict_copy['speaker'] = word_and_ts_seq[uniq_id]['words'][word_idx]['speaker']
                speaker_added_word_dicts.append(trans_wdict_copy)
            trans_hyp[sess_idx].timestamp['word'] = speaker_added_word_dicts
            w_count, segment_list = 0, []
            for word_idx, trans_segdict in enumerate(trans_hyp[0].timestamp['segment']):
                words = trans_segdict['segment'].split()
                spk_vote_pool = []
                for word in words:
                    if word.lower() != word_and_ts_seq[uniq_id]['words'][w_count]['word'].lower():
                        raise ValueError(
                            f"Word mismatch: '{word.lower()}' != '{word_and_ts_seq[uniq_id]['words'][w_count]['word'].lower()}' "
                            f"at session {sess_idx}, word count {w_count}."
                        )
                    spk_int = int(word_and_ts_seq[uniq_id]['words'][w_count]['speaker'].split('_')[-1])
                    spk_vote_pool.append(spk_int)
                    w_count += 1
                trans_segdict['speaker'] = f"speaker_{torch.mode(torch.tensor(spk_vote_pool), dim=0).values.item()}"
                segment_list.append(trans_segdict)
            trans_hyp[sess_idx].timestamp['segment'] = segment_list
        transcriptions = (trans_hyp, trans_hyp)
        return transcriptions

    def perform_offline_stt_spk(self, override_cfg: Dict[str, Any]):
        """
        Perform offline STT and speaker diarization on the provided manifest file.

        Args:
            override_cfg (dict): Override configuration parameters.

        Returns:
            transcriptions (Tuple): Tuple containing the speaker-tagged transcripts.
        """
        transcriptions = self.asr_model.transcribe(
            audio=self.cfg.dataset_manifest,
            override_config=override_cfg,
        )
        best_hyp, _ = transcriptions
        _, pred_tensors = self.diar_model.diarize(audio=self.cfg.manifest_file, include_tensor_outputs=True)
        speaker_transcriptions, word_and_ts_seq = self.merge_transcript_and_speakers(
            test_manifest_dict=self.diar_model._diarize_audio_rttm_map,
            asr_hypotheses=best_hyp,
            diar_pred_out=pred_tensors,
        )
        transcriptions = self._add_speaker_transcriptions(
            transcriptions=transcriptions,
            speaker_transcriptions=speaker_transcriptions,
            word_and_ts_seq=word_and_ts_seq,
            test_manifest_dict=self.diar_model._diarize_audio_rttm_map,
        )
        return transcriptions

    def generate_seglst_dicts_from_serial_streaming(self, samples: List[Dict[str, Any]]):
        """
        Generate the seglst dictionary for SegLST format from serial streaming.
        For SegLST format, the session_id is the name of the audio file
        should not contain "." in the name.

        Args:
            samples (List[Dict[str, Any]]): List of samples.
        """
        # for _, word_ts_and_seq in enumerate(self._word_and_ts_seq):
        for sample in samples:
            uniq_id = get_uniqname_from_filepath(sample['audio_filepath']).split('.')[0]
            word_ts_and_seq_dict = self._word_and_ts_seq[uniq_id]
            for sentence_dict in word_ts_and_seq_dict['sentences']:
                session_id = word_ts_and_seq_dict['uniq_id'].split('.')[0]
                seglst_dict = get_new_sentence_dict(
                    speaker=sentence_dict['speaker'],
                    start_time=float(sentence_dict['start_time']),
                    end_time=float(sentence_dict['end_time']),
                    text=sentence_dict["words"],
                    session_id=session_id,
                )
                self.instance_manager.seglst_dict_list.append(seglst_dict)

    def generate_seglst_dicts_from_parallel_streaming(self, samples: List[Dict[str, Any]]):
        """
        Generate the seglst dictionary for SegLST format from parallel streaming.
        For SegLST format, the session_id is the name of the audio file
        should not contain "." in the name.

        Args:
            samples (List[Dict[str, Any]]): List of samples.
        """
        self.instance_manager.previous_asr_states.extend(self.instance_manager.batch_asr_states)
        for sample, asr_state in zip(samples, self.instance_manager.previous_asr_states):
            audio_filepath = sample["audio_filepath"]
            uniq_id = os.path.basename(audio_filepath).split('.')[0]
            seglsts = [
                get_new_sentence_dict(
                    speaker=seg['speaker'],
                    start_time=seg['start_time'],
                    end_time=seg['end_time'],
                    text=seg['words'],
                    session_id=uniq_id,
                )
                for seg in asr_state.seglsts
            ]
            seglsts = sorted(seglsts, key=lambda x: x['start_time'])
            self.instance_manager.seglst_dict_list.extend(seglsts)

    def _find_active_speakers(self, diar_preds: torch.Tensor, n_active_speakers_per_stream: int) -> List[List[int]]:
        """
        Find the active speakers from the diar prediction output.

        Args:
            diar_preds (torch.Tensor): The diar prediction output.
            n_active_speakers_per_stream (int): The number of active speakers per stream.

        Returns:
            speaker_ids_list (List[List[int]]): The list of active speakers for each stream.
        """
        if diar_preds.ndim != 3:
            raise ValueError(f"diar_preds must be 3D (B, T, N), got shape {diar_preds.shape}")
        if n_active_speakers_per_stream > diar_preds.shape[2]:
            raise ValueError(
                f"n_active_speakers_per_stream ({n_active_speakers_per_stream}) "
                f"> available speakers ({diar_preds.shape[2]})"
            )
        max_probs = torch.max(diar_preds, dim=1).values  # (B, T, N) --> (B, N)
        top_values, top_indices = torch.topk(max_probs, k=n_active_speakers_per_stream, dim=1)
        masks = top_values > 0.5

        speaker_ids_list = []
        for speaker_ids, mask in zip(top_indices, masks):
            speaker_ids_list.append(sorted(speaker_ids[mask].tolist()))
        return speaker_ids_list

    def forward_pre_encoded(
        self, audio_signal: torch.Tensor, length: torch.Tensor, drop_extra_pre_encoded: int = 0
    ) -> None:
        """
        Forward the pre-encoded features through the ASR model.

        Args:
            audio_signal (torch.Tensor): The audio signal.
            length (torch.Tensor): The length of the audio signal.
            drop_extra_pre_encoded (int): The number of extra pre-encoded tokens to drop.

        Returns:
            audio_signal (torch.Tensor): The pre-encoded audio signal.
            length (torch.Tensor): The length of the pre-encoded audio signal.
        """
        audio_signal = torch.transpose(audio_signal, 1, 2)  # (B, T, D) -> (B, D, T)

        audio_signal, length = self.asr_model.encoder.pre_encode(x=audio_signal, lengths=length)
        length = length.to(torch.int64)
        # `self.streaming_cfg` is set by setup_streaming_cfg(), called in the init
        if drop_extra_pre_encoded:
            audio_signal = audio_signal[:, drop_extra_pre_encoded:, :]
            length = (length - drop_extra_pre_encoded).clamp(min=0)
        return audio_signal, length

    def mask_features(
        self, chunk_audio: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5, mask_value: float = -16.6355
    ) -> torch.Tensor:
        """
        Mask the features of the chunk audio.

        Args:
            chunk_audio (torch.Tensor): The chunk audio.
            mask (torch.Tensor): The mask.
            threshold (float): The threshold for the mask.
            mask_value (float): The value for the masked audio.

        Returns:
            masked_chunk_audio (torch.Tensor): The masked chunk audio.
        """
        if chunk_audio.ndim != 3:
            raise ValueError(
                f"chunk_audio must be 3D (B, C, T), got {chunk_audio.ndim}D with shape {chunk_audio.shape}"
            )
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D (B, T), got {mask.ndim}D with shape {mask.shape}")
        if chunk_audio.shape[0] != mask.shape[0]:
            raise ValueError(f"Batch size mismatch: chunk_audio={chunk_audio.shape[0]}, mask={mask.shape[0]}")
        mask = (mask > threshold).float()
        mask = mask.unsqueeze(-1).repeat(1, 1, 8).flatten(1, 2)

        if mask.shape[1] > chunk_audio.shape[2]:
            logging.warning(f"Mask shape {mask.shape} is greater than chunk_audio shape {chunk_audio.shape}")
            mask = mask[:, : chunk_audio.shape[2]]
        elif mask.shape[1] < chunk_audio.shape[2]:
            logging.warning(f"Mask shape {mask.shape} is less than chunk_audio shape {chunk_audio.shape}")
            mask = torch.nn.functional.pad(mask, (chunk_audio.shape[2] - mask.shape[1], 0), mode='constant', value=0)

        masked_chunk_audio = chunk_audio * mask.unsqueeze(1)
        masked_chunk_audio[torch.where(chunk_audio == 0)] = mask_value

        return masked_chunk_audio

    def mask_preencode(self, chunk_audio: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Mask the pre-encoded features of the chunk audio.

        Args:
            chunk_audio (torch.Tensor): The chunk audio.
            mask (torch.Tensor): The mask.
            threshold (float): The threshold for the mask.

        Returns:
            masked_chunk_audio (torch.Tensor): The masked chunk audio.
        """
        mask = (mask > threshold).float()

        if mask.shape[1] > chunk_audio.shape[1]:
            logging.warning(f"Mask shape {mask.shape} is greater than chunk_audio shape {chunk_audio.shape}")
            mask = mask[:, : chunk_audio.shape[1]]
        elif mask.shape[1] < chunk_audio.shape[1]:
            logging.warning(f"Mask shape {mask.shape} is less than chunk_audio shape {chunk_audio.shape}")
            mask = torch.nn.functional.pad(mask, (chunk_audio.shape[1] - mask.shape[1], 0), mode='constant', value=0)

        masked_chunk_audio = chunk_audio * mask.unsqueeze(-1)

        return masked_chunk_audio

    def get_diar_pred_out_stream(self, step_num):
        """
        Get the diar prediction output stream for the given step number.

        Args:
            step_num (int): the step number

        Returns:
            new_diar_pred_out_stream (torch.Tensor): the diar prediction output stream for the given step number
            new_chunk_preds (torch.Tensor): the diar prediction output stream for the given step number
        """
        start_frame_idx = step_num * self._nframes_per_chunk
        end_frame_idx = start_frame_idx + self._nframes_per_chunk
        new_diar_pred_out_stream = self.diar_model.rttms_mask_mats[:, :end_frame_idx]
        new_chunk_preds = new_diar_pred_out_stream[:, start_frame_idx:end_frame_idx]
        return new_diar_pred_out_stream, new_chunk_preds

    @measure_eta
    def perform_serial_streaming_stt_spk(
        self,
        step_num: int,
        chunk_audio: torch.Tensor,
        chunk_lengths: torch.Tensor,
        is_buffer_empty: bool,
        drop_extra_pre_encoded: int,
    ):
        """
        Perform the serial streaming inference.
        Serial streaming inference deploys a single ASR model instance to transcribe multiple speakers in a chunk.
        All the updates are done to the instance manager in a `SpeakerTaggedASR` class instance.

        Args:
            step_num (int): The step number of the chunk.
            chunk_audio (torch.Tensor): The chunk audio.
            chunk_lengths (torch.Tensor): The length of the chunk audio.
            is_buffer_empty (bool): Whether the buffer is empty.
            drop_extra_pre_encoded (int): The number of extra pre-encoded tokens to drop.
        """
        # Initialize the instance manager with the batch size of the chunk audio.
        if step_num == 0:
            self.instance_manager.reset(batch_size=chunk_audio.shape[0])
            self.instance_manager.to(chunk_audio.device)

        # This part exists for compatibility with the parallel streaming inference.
        self.instance_manager.get_active_speakers_info(
            active_speakers=[[0] for _ in range(chunk_audio.shape[0])],
            chunk_audio=chunk_audio,
            chunk_lengths=chunk_lengths,
        )

        (
            asr_pred_out_stream,
            _,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = self.asr_model.conformer_stream_step(
            processed_signal=chunk_audio,
            processed_signal_length=chunk_lengths,
            cache_last_channel=self.instance_manager.active_cache_last_channel,
            cache_last_time=self.instance_manager.active_cache_last_time,
            cache_last_channel_len=self.instance_manager.active_cache_last_channel_len,
            previous_hypotheses=self.instance_manager.active_previous_hypotheses,
            previous_pred_out=self.instance_manager.active_asr_pred_out_stream,
            keep_all_outputs=is_buffer_empty,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            return_transcription=True,
        )

        if self.diar_model.rttms_mask_mats is None:

            new_streaming_state, diar_pred_out_stream = self.diar_model.forward_streaming_step(
                processed_signal=chunk_audio.transpose(1, 2),
                processed_signal_length=chunk_lengths,
                streaming_state=self.instance_manager.diar_states.streaming_state,
                total_preds=self.instance_manager.diar_states.diar_pred_out_stream,
                drop_extra_pre_encoded=drop_extra_pre_encoded,
            )
            self.instance_manager.update_diar_state(
                diar_pred_out_stream=diar_pred_out_stream,
                previous_chunk_preds=diar_pred_out_stream[:, -self._nframes_per_chunk :],
                diar_streaming_state=new_streaming_state,
            )
        else:
            _, new_chunk_preds = self.get_diar_pred_out_stream(step_num)
            diar_pred_out_stream = new_chunk_preds

        transcribed_speaker_texts = [None] * len(self.test_manifest_dict)
        for idx, (uniq_id, _) in enumerate(self.test_manifest_dict.items()):
            if not (len(previous_hypotheses[idx].text) == 0 and step_num <= self._initial_steps):
                # Get the word-level dictionaries for each word in the chunk
                self._word_and_ts_seq[uniq_id] = self.get_frame_and_words_online(
                    uniq_id=uniq_id,
                    step_num=step_num,
                    diar_pred_out_stream=diar_pred_out_stream[idx, :, :],
                    previous_hypothesis=previous_hypotheses[idx],
                    word_and_ts_seq=self._word_and_ts_seq[uniq_id],
                )
                if len(self._word_and_ts_seq[uniq_id]["words"]) > 0:
                    self._word_and_ts_seq[uniq_id] = self.get_sentences_values(
                        session_trans_dict=self._word_and_ts_seq[uniq_id],
                        sentence_render_length=self._sentence_render_length,
                    )
                    if self.cfg.generate_realtime_scripts:
                        transcribed_speaker_texts[idx] = print_sentences(
                            sentences=self._word_and_ts_seq[uniq_id]["sentences"],
                            color_palette=get_color_palette(),
                            params=self.cfg,
                        )
                        write_txt(
                            f'{self.cfg.print_path}'.replace(".sh", f"_{idx}.sh"),
                            transcribed_speaker_texts[idx].strip(),
                        )

        for batch_idx in range(chunk_audio.shape[0]):
            self.instance_manager.update_asr_state(
                batch_idx,
                speaker_id=0,
                cache_last_channel=cache_last_channel[:, batch_idx],
                cache_last_time=cache_last_time[:, batch_idx],
                cache_last_channel_len=cache_last_channel_len[batch_idx],
                previous_hypotheses=previous_hypotheses[batch_idx],
                previous_pred_out=asr_pred_out_stream[batch_idx],
            )

    @measure_eta
    def perform_parallel_streaming_stt_spk(
        self,
        step_num,
        chunk_audio,
        chunk_lengths,
        is_buffer_empty,
        drop_extra_pre_encoded,
    ):
        """
        Perform the parallel streaming inference.
        Parallel streaming inference deploys multiple ASR model instances to transcribe multiple speakers in a chunk.
        All the updates are done to the instance manager in a `SpeakerTaggedASR` class instance.

        Args:
            step_num (int): The step number of the chunk.
            chunk_audio (torch.Tensor): The chunk audio.
            chunk_lengths (torch.Tensor): The length of the chunk audio.
            is_buffer_empty (bool): Whether the buffer is empty.
            drop_extra_pre_encoded (int): The number of extra pre-encoded tokens to drop.
        """
        # Initialize the instance manager with the batch size of the chunk audio.
        if step_num == 0:
            self._offset_chunk_start_time = 0
            self.instance_manager.reset(batch_size=chunk_audio.shape[0])
            self.instance_manager.to(chunk_audio.device)

        # Step 2: diarize or get GT rttms
        if self.diar_model.rttms_mask_mats is None:
            new_streaming_state, new_diar_pred_out_stream = self.diar_model.forward_streaming_step(
                processed_signal=chunk_audio.transpose(1, 2),
                processed_signal_length=chunk_lengths,
                streaming_state=self.instance_manager.diar_states.streaming_state,
                total_preds=self.instance_manager.diar_states.diar_pred_out_stream,
                drop_extra_pre_encoded=drop_extra_pre_encoded,
            )
            new_chunk_preds = new_diar_pred_out_stream[:, -self._nframes_per_chunk :]

        else:
            new_diar_pred_out_stream, new_chunk_preds = self.get_diar_pred_out_stream(step_num)
            new_streaming_state = self.instance_manager.diar_states.streaming_state

        # Step 3: update diar states
        self.instance_manager.update_diar_state(
            diar_pred_out_stream=new_diar_pred_out_stream,
            previous_chunk_preds=new_chunk_preds,
            diar_streaming_state=new_streaming_state,
        )

        # For a session, if no second speaker is detected,
        # the spk_targets will be set to all ones in the single speaker mode
        if self._single_speaker_mode:
            if self._max_num_of_spks == 1:
                is_single_speaker = [True] * chunk_audio.shape[0]
            else:
                is_single_speaker = (new_diar_pred_out_stream > 0.5).any(1).sum(-1) <= 1.0
            for i in range(chunk_audio.shape[0]):
                if is_single_speaker[i]:
                    new_diar_pred_out_stream[i, :, 0] = 1.0
                    new_diar_pred_out_stream[i, :, 1:] = 0.0

        # Step 4: find active speakers
        diar_chunk_preds = new_diar_pred_out_stream[:, -self._nframes_per_chunk * self._cache_gating_buffer_size :]
        if self._cache_gating:
            active_speakers = self._find_active_speakers(
                diar_chunk_preds, n_active_speakers_per_stream=self.n_active_speakers_per_stream
            )
        else:
            active_speakers = [list(range(self.n_active_speakers_per_stream)) for _ in range(chunk_audio.shape[0])]

        if (self._masked_asr and self._use_mask_preencode) or not self._masked_asr:
            chunk_audio, chunk_lengths = self.forward_pre_encoded(chunk_audio, chunk_lengths, drop_extra_pre_encoded)
            bypass_pre_encode = True
        else:
            bypass_pre_encode = False

        # Step 5: generate instance for active speakers
        (
            active_chunk_audio,
            active_chunk_lengths,
            active_speaker_targets,
            inactive_speaker_targets,
        ) = self.instance_manager.get_active_speakers_info(
            active_speakers=active_speakers,
            chunk_audio=chunk_audio,
            chunk_lengths=chunk_lengths,
        )

        # skip current chunk if no active speakers are found
        if active_chunk_audio is None:
            return

        # Step 6:
        # 1) mask the non-active speakers for masked ASR; or
        # 2) set speaker targets for multitalker ASR
        if self._masked_asr:
            if self._use_mask_preencode:
                active_chunk_audio = self.mask_preencode(chunk_audio=active_chunk_audio, mask=active_speaker_targets)
            else:
                active_chunk_audio = self.mask_features(chunk_audio=active_chunk_audio, mask=active_speaker_targets)
        else:
            if self._binary_diar_preds:
                active_speaker_targets = (active_speaker_targets > 0.5).float()
                inactive_speaker_targets = (inactive_speaker_targets > 0.5).float()
            self.asr_model.set_speaker_targets(active_speaker_targets, inactive_speaker_targets)

        # Step 7: ASR forward pass for active speakers
        (
            pred_out_stream,
            _,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = self.asr_model.conformer_stream_step(
            processed_signal=active_chunk_audio,
            processed_signal_length=active_chunk_lengths,
            cache_last_channel=self.instance_manager.active_cache_last_channel,
            cache_last_time=self.instance_manager.active_cache_last_time,
            cache_last_channel_len=self.instance_manager.active_cache_last_channel_len,
            keep_all_outputs=is_buffer_empty,
            previous_hypotheses=self.instance_manager.active_previous_hypotheses,
            previous_pred_out=self.instance_manager.active_asr_pred_out_stream,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            return_transcription=True,
            bypass_pre_encode=bypass_pre_encode,
        )

        # Step 8: update ASR states
        active_id = 0
        for batch_idx, speaker_ids in enumerate(active_speakers):
            for speaker_id in speaker_ids:
                self.instance_manager.update_asr_state(
                    batch_idx,
                    speaker_id,
                    cache_last_channel[:, active_id],
                    cache_last_time[:, active_id],
                    cache_last_channel_len[active_id],
                    previous_hypotheses[active_id],
                    pred_out_stream[active_id],
                )
                active_id += 1

        # Step 9: update seglsts with timestamps
        self.instance_manager.update_seglsts(offset=self._offset_chunk_start_time)
        self._offset_chunk_start_time += self._nframes_per_chunk * self._frame_len_sec

        if self.cfg.generate_realtime_scripts:
            for session_idx in self.cfg.print_sample_indices:
                asr_state = self.instance_manager.batch_asr_states[session_idx]
                transcribed_speaker_texts = print_sentences(
                    sentences=asr_state.seglsts, color_palette=get_color_palette(), params=self.cfg
                )
                write_txt(
                    f'{self.cfg.print_path.replace(".sh", f"_{session_idx}.sh")}', transcribed_speaker_texts.strip()
                )


class MultiTalkerInstanceManager:
    """
    For multi-talker inference, we need to manage the information per speaker.
    Each sample in a batch can be considered as a multi-talker instance,
    and each instance may contain multiple speakers, which is the real
    batch size for inference. If there are at most N speakers and the batch
    size is B, then the real batch size for inference is at most B * N.
    """

    class ASRState:
        """
        ASR state for each instance.
        1. In parallel mode, each instance handles each potential speaker.
        2. In serial mode, each instance handles one session.

        The goal of ASR-State class is to handle the ASR cache state between streaming steps.
        The ASR-states required to perform streaming inference are all included in this class.
        """

        def __init__(self, max_num_of_spks: int = 4, frame_len_sec: float = 0.08, sent_break_sec: float = 5.0):
            """
            Initialize the ASR-State class with the initial parameters.

            Args:
                max_num_of_spks (int): The maximum number of speakers.
                frame_len_sec (float): The length of the frame in seconds.
                sent_break_sec (float): The minimum time gap between two sentences in seconds.
            """
            # Initialize the ASR state with the initial parameters.
            self.speakers: Optional[List[str]] = None
            self.cache_last_channel = None
            self.cache_last_time = None
            self.cache_last_channel_len = None
            self.previous_hypothesis = None
            self.previous_pred_out = None

            self.max_num_of_spks = max_num_of_spks

            self._frame_len_sec = frame_len_sec
            self._sent_break_sec = sent_break_sec
            self._speaker_wise_sentences = {}
            self._prev_history_speaker_texts = ["" for _ in range(self.max_num_of_spks)]

            self.seglsts = []

        def _reset_speaker_wise_sentences(self):
            """
            Reset the speaker-wise sentences which will be used to generate the SegLST transcription outputs.
            """
            self._speaker_wise_sentences = {}
            self._prev_history_speaker_texts = ["" for _ in range(self.max_num_of_spks)]

        def reset(self, asr_cache_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            """
            Reset the ASR state.

            Args:
                asr_cache_state (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The ASR cache state.
                    - cache_last_channel (torch.Tensor): The cache last channel.
                    - cache_last_time (torch.Tensor): The cache last time.
                    - cache_last_channel_len (torch.Tensor): The cache last channel length.
            """
            self.speakers = [0]
            self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = asr_cache_state
            self.previous_hypothesis = [None]
            self.previous_pred_out = [None]
            self.seglsts = []
            self._speaker_wise_sentences = {}
            self._prev_history_speaker_texts = ["" for _ in range(self.max_num_of_spks)]

        def update_asr_state(
            self,
            speaker_id,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypothesis,
            previous_pred_out,
        ):
            """
            Update the ASR state with the new ASR cache state.
            This function should be called at every streaming step to update the ASR cache state.

            Args:
                speaker_id (int): The speaker id.
                cache_last_channel (torch.Tensor): The cache last channel.
                cache_last_time (torch.Tensor): The cache last time.
                cache_last_channel_len (torch.Tensor): The cache last channel length.
                previous_hypothesis (Hypothesis): The previous hypothesis.
                previous_pred_out (torch.Tensor): The previous prediction output.
            """
            self.cache_last_channel[:, speaker_id] = cache_last_channel
            self.cache_last_time[:, speaker_id] = cache_last_time
            self.cache_last_channel_len[speaker_id] = cache_last_channel_len
            self.previous_hypothesis[speaker_id] = previous_hypothesis
            self.previous_pred_out[speaker_id] = previous_pred_out

        def to(self, device):
            """
            Override the to method to move the ASR state to the device.

            Args:
                device (torch.device): The device to move the ASR state to.
            """
            self.cache_last_channel = self.cache_last_channel.to(device)
            self.cache_last_time = self.cache_last_time.to(device)
            self.cache_last_channel_len = self.cache_last_channel_len.to(device)

        def get_speakers(self):
            """
            Get the speaker ids (int) for each instance.
            This function is used for serial streaming mode.
            """
            return self.speakers

        def add_speaker(self, speaker_id: int, asr_cache_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            """
            Add a speaker index and its initial cache state to the ASR state.

            Args:
                speaker_id (int): The speaker id.
                asr_cache_state (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The ASR cache state.
            """
            self.speakers.append(speaker_id)
            cache_last_channel, cache_last_time, cache_last_channel_len = asr_cache_state
            self.cache_last_channel = torch.cat([self.cache_last_channel, cache_last_channel], dim=1)
            self.cache_last_time = torch.cat([self.cache_last_time, cache_last_time], dim=1)
            self.cache_last_channel_len = torch.cat([self.cache_last_channel_len, cache_last_channel_len], dim=0)
            self.previous_hypothesis.append(None)
            self.previous_pred_out.append(None)

        def _update_last_sentence(self, spk_idx: int, end_time: float, diff_text: str):
            """
            Update the end time of the last sentence for a speaker.

            Args:
                spk_idx (int): The speaker id.
                end_time (float): The end time of the last sentence.
                diff_text (str): The difference text.
            """
            if end_time is not None:
                self._speaker_wise_sentences[spk_idx][-1]['end_time'] = end_time
            new_words = self._speaker_wise_sentences[spk_idx][-1]['words'] + diff_text
            self._speaker_wise_sentences[spk_idx][-1]['words'] = new_words.strip()

        def _is_new_text(self, spk_idx: int, text: str):
            """
            Check if the text is new for a speaker.

            Args:
                spk_idx (int): The speaker id.
                text (str): The text.
            """
            if text is None or text == self._prev_history_speaker_texts[spk_idx]:
                return None
            else:
                # Get the difference between the current text and the previous text
                if self._prev_history_speaker_texts[spk_idx] in text:
                    return text.replace(self._prev_history_speaker_texts[spk_idx], "")
                else:
                    return text.strip()

        def _compute_hypothesis_timestamps(self, hypothesis: Hypothesis, offset: float) -> Tuple[float, float, bool]:
            """
            Compute start and end timestamps for a hypothesis based on available timing information.

            This method calculates the temporal boundaries of a speech hypothesis, prioritizing
            frame-level timestamps when available. When timestamps are not available, it falls
            back to computing timing based on the hypothesis length.

            Args:
                hypothesis (Hypothesis): The ASR hypothesis object containing either frame-level
                offset (float): The time offset (in seconds) to add to the computed timestamps,
                    typically representing the start time of the current audio chunk.

            Returns:
                Tuple[float, float, bool]: A tuple containing:
                    - start_time (float): The absolute start time of the hypothesis in seconds
                    - end_time (float): The absolute end time of the hypothesis in seconds
                    - sep_flag (bool): A flag indicating whether timing was computed from length
                        rather than timestamps.

            Note:
                The end_time calculation from timestamps adds 1 to the last timestamp to account
                for the full duration of the final frame.
            """
            sep_flag = False
            if len(hypothesis.timestamp) > 0:
                start_time = offset + (hypothesis.timestamp[0]) * self._frame_len_sec
                end_time = offset + (hypothesis.timestamp[-1] + 1) * self._frame_len_sec
            else:
                start_time = offset
                end_time = offset + hypothesis.length.item() * self._frame_len_sec
                sep_flag = True

            return start_time, end_time, sep_flag

        def update_sessionwise_seglsts_for_parallel(self, offset: float):
            """
            Update the seglsts for the parallel mode streaming.
            Note that this function is NOT used for serial mode streaming.

            Args:
                offset (float): The offset in seconds.
                                This is usally the start time of the current audio chunk.
            """
            valid_speakers = set()
            for spk_idx in self.get_speakers():
                hypothesis = self.previous_hypothesis[spk_idx]
                if hypothesis is None:
                    continue
                valid_speakers.add(spk_idx)

                if spk_idx not in self._speaker_wise_sentences:
                    self._speaker_wise_sentences[spk_idx] = []

                diff_text = self._is_new_text(spk_idx=spk_idx, text=hypothesis.text)
                if diff_text is not None:

                    start_time, end_time, sep_flag = self._compute_hypothesis_timestamps(
                        hypothesis=hypothesis, offset=offset
                    )

                    # Get the last end time of the previous sentence or None if no sentences are present
                    if len(self._speaker_wise_sentences[spk_idx]) > 0:
                        last_end_time = self._speaker_wise_sentences[spk_idx][-1]['end_time']
                    else:
                        last_end_time = 0.0

                    # Case 1 - If start_tiime is greater than end_time + sent_break_sec, then we need to add the sentence
                    if sep_flag or (last_end_time == 0.0 or start_time > last_end_time + self._sent_break_sec):
                        stripped_text = diff_text.strip()
                        if len(stripped_text) > 0 and stripped_text[0] in ['.', ',', '?', '!']:
                            # This handles the case where the first character should be assigned to the previous sentence.
                            the_first_char, diff_text = stripped_text[0], stripped_text[1:]
                            self._update_last_sentence(spk_idx=spk_idx, end_time=None, diff_text=the_first_char)
                        self._speaker_wise_sentences[spk_idx].append(
                            get_new_sentence_dict(
                                speaker=f"speaker_{spk_idx}", start_time=start_time, end_time=end_time, text=diff_text
                            )
                        )
                    # Case 2 - If start_time is less than end_time + sent_break_sec, then we need to update the end_time
                    else:
                        self._update_last_sentence(spk_idx=spk_idx, end_time=end_time, diff_text=diff_text)

                # Update the previous history of the speaker text
                if hypothesis.text is not None:
                    self._prev_history_speaker_texts[spk_idx] = hypothesis.text

            self.seglsts = []

            # Merge all sentences for each speaker but sort by start_time
            for spk_idx in valid_speakers:
                self.seglsts.extend(self._speaker_wise_sentences[spk_idx])

            # Finally, sort the seglsts by start_time
            self.seglsts = sorted(self.seglsts, key=lambda x: x['start_time'])

    class DiarState:
        """
        Diar state for each diarization instance.
        There is no difference between serial and parallel mode for the diarization state.
        The goal of Diar-State class is to handle the diarization cache state between streaming steps.
        """

        def __init__(self, batch_size: int = 1, max_num_of_spks: int = 4):
            """
            Initialize the Diar-State class with the initial parameters.

            Args:
                batch_size (int): The batch size.
                max_num_of_spks (int): The maximum number of speakers.
            """
            self.batch_size = batch_size
            self.max_num_of_spks = max_num_of_spks
            self.diar_pred_out_stream = None
            self.previous_chunk_preds = None
            self.streaming_state = None

        def reset(self, diar_streaming_state: StreamingSortformerState):
            self.diar_pred_out_stream = torch.zeros((self.batch_size, 0, self.max_num_of_spks))
            self.previous_chunk_preds = torch.zeros((self.batch_size, 0, self.max_num_of_spks))
            self.streaming_state = diar_streaming_state

        def to(self, device):
            self.diar_pred_out_stream = self.diar_pred_out_stream.to(device)
            self.previous_chunk_preds = self.previous_chunk_preds.to(device)
            self.streaming_state.to(device)

    def __init__(
        self,
        asr_model=None,
        diar_model=None,
        batch_size: int = 1,
        max_num_of_spks: int = 4,
        sent_break_sec: float = 5.0,
    ):
        """
        Initialize the MultiTalkerInstanceManager class with the initial parameters.

        Args:
            asr_model: The ASR model.
            diar_model: The diarization model.
            batch_size (int): The batch size for ASR.
                1. For parallel mode, this is the number of potential speakers
                multiplied by the session counts.
                2. For serial mode, this is the number of sessions.
            max_num_of_spks (int): The maximum number of speakers.
        """
        self.asr_model = asr_model
        self.diar_model = diar_model

        self.batch_size = batch_size
        self.max_num_of_spks = max_num_of_spks
        self._sent_break_sec = sent_break_sec

        # ASR state bank
        self.batch_asr_states = []
        self.previous_asr_states = []

        # Diar states
        self.diar_states = None

        # SegLST output list
        self.seglst_dict_list = []

        # Active speaker buffer lists
        self._active_chunk_audio: List[torch.Tensor] = []
        self._active_chunk_lengths: List[torch.Tensor] = []
        self._active_speaker_targets: List[torch.Tensor] = []
        self._inactive_speaker_targets: List[torch.Tensor] = []
        self._active_previous_hypotheses: List[Hypothesis] = []
        self._active_asr_pred_out_stream: List[torch.Tensor] = []
        self._active_cache_last_channel: List[torch.Tensor] = []
        self._active_cache_last_time: List[torch.Tensor] = []
        self._active_cache_last_channel_len: List[torch.Tensor] = []

        # Active speaker attributes
        self.active_previous_hypotheses: Optional[List[Hypothesis]] = None
        self.active_asr_pred_out_stream: Optional[List[torch.Tensor]] = None
        self.active_cache_last_channel: Optional[torch.Tensor] = None
        self.active_cache_last_time: Optional[torch.Tensor] = None
        self.active_cache_last_channel_len: Optional[torch.Tensor] = None

    def _reset_active_speaker_buffers(self):
        """
        Reset the active speaker buffers need to update the active speaker information.
        """
        self._active_chunk_audio = []
        self._active_chunk_lengths = []
        self._active_speaker_targets = []
        self._inactive_speaker_targets = []
        self._active_previous_hypotheses = []

        self._active_asr_pred_out_stream = []
        self._active_cache_last_channel = []
        self._active_cache_last_time = []
        self._active_cache_last_channel_len = []

    def reset(self, batch_size: Optional[int] = None, max_num_of_spks: Optional[int] = None):
        """
        Reset the active speaker buffers need to update the active speaker information.

        Args:
            batch_size (Optional[int]): The batch size.
            max_num_of_spks (Optional[int]): The maximum number of speakers.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if max_num_of_spks is not None:
            self.max_num_of_spks = max_num_of_spks

        if len(self.batch_asr_states) > 0:
            self.previous_asr_states.extend(deepcopy(self.batch_asr_states))
        self.batch_asr_states = [
            self.ASRState(self.max_num_of_spks, sent_break_sec=self._sent_break_sec) for _ in range(self.batch_size)
        ]

        for i in range(self.batch_size):
            self.batch_asr_states[i].reset(self.asr_model.encoder.get_initial_cache_state(batch_size=1))

        self.diar_states = self.DiarState(batch_size=self.batch_size, max_num_of_spks=self.max_num_of_spks)
        self.diar_states.reset(self.diar_model.sortformer_modules.init_streaming_state(batch_size=self.batch_size))

        self.seglst_dict_list = []

    def add_speaker(self, batch_idx: int, speaker_id: int):
        """
        Add a speaker index and its initial cache state to the ASR state.

        Args:
            batch_idx (int): The batch index.
            speaker_id (int): The speaker id.
        """
        speakers = self.batch_asr_states[batch_idx].get_speakers()
        for speaker_index in range(0, speaker_id + 1):
            if speaker_index not in speakers:
                self.batch_asr_states[batch_idx].add_speaker(
                    speaker_id=speaker_index,
                    asr_cache_state=self.asr_model.encoder.get_initial_cache_state(batch_size=1),
                )

    def get_speakers(self, batch_idx: int):
        """
        Get the speaker ids (int) for each instance.

        Args:
            batch_idx (int): The batch index.
        """
        return self.batch_asr_states[batch_idx].get_speakers()

    def to(self, device: torch.device):
        """
        Override the to method to move the ASR and Diar states to the device.

        Args:
            device (torch.device): The device to move the ASR and Diar states to.
        """
        for batch_idx in range(self.batch_size):
            self.batch_asr_states[batch_idx].to(device)
        self.diar_states.to(device)

    def update_diar_state(
        self,
        diar_pred_out_stream: torch.Tensor,
        previous_chunk_preds: torch.Tensor,
        diar_streaming_state: StreamingSortformerState,
    ):
        """
        Update the diarization state from the diarization step.
        The diarization results are updated as a form of torch.Tensor.

        Args:
            diar_pred_out_stream (torch.Tensor): The diarization prediction output stream.
            previous_chunk_preds (torch.Tensor): The previous chunk prediction output.
            diar_streaming_state (StreamingSortformerState): The diarization streaming state.
        """
        self.diar_states.diar_pred_out_stream = diar_pred_out_stream
        self.diar_states.previous_chunk_preds = previous_chunk_preds
        self.diar_states.streaming_state = diar_streaming_state

    def update_asr_state(
        self,
        batch_idx,
        speaker_id,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
        previous_hypotheses,
        previous_pred_out,
    ):
        """
        A function to update the ASR state with the new ASR cache state.
        This function should be called at every streaming step to update the ASR cache state.

        Args:
            batch_idx (int): The batch index.
                If parallel mode, this is the index of the potential speaker.
                If serial mode, this is the index of the session.
            speaker_id (int): The speaker id in the given session.
            -- Cache aware ASR related parameters --
            cache_last_channel (torch.Tensor)
            cache_last_time (torch.Tensor)
            cache_last_channel_len (torch.Tensor)
            previous_hypotheses (Hypothesis)
            previous_pred_out (torch.Tensor) The previous prediction output.
        """
        self.batch_asr_states[batch_idx].update_asr_state(
            speaker_id,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
            previous_pred_out,
        )

    def get_active_speakers_info(self, active_speakers, chunk_audio, chunk_lengths):
        """
        Collect the active speaker information for the next streaming step and
        update the active speaker buffers.

        Args:
            active_speakers (List[List[int]]): The active speakers for each chunk.
            chunk_audio (torch.Tensor): The chunk audio.
            chunk_lengths (torch.Tensor): The chunk lengths.
        """
        # Reset the active speaker buffers
        self._reset_active_speaker_buffers()

        # Loop through the active speakers and update the active speaker buffers
        for batch_idx, speaker_ids in enumerate(active_speakers):
            for speaker_id in speaker_ids:
                self._active_chunk_audio.append(chunk_audio[batch_idx, :])
                self._active_chunk_lengths.append(chunk_lengths[batch_idx])
                self._active_speaker_targets.append(self.diar_states.previous_chunk_preds[batch_idx, :, speaker_id])
                inactive_speaker_ids = [i for i in range(len(speaker_ids)) if i != speaker_id]
                self._inactive_speaker_targets.append(
                    (self.diar_states.previous_chunk_preds[batch_idx, :, inactive_speaker_ids] > 0.5).sum(dim=-1) > 0
                )
                if speaker_id not in self.batch_asr_states[batch_idx].get_speakers():
                    self.add_speaker(batch_idx, speaker_id)

                self._active_previous_hypotheses.append(
                    self.batch_asr_states[batch_idx].previous_hypothesis[speaker_id]
                )
                self._active_asr_pred_out_stream.append(self.batch_asr_states[batch_idx].previous_pred_out[speaker_id])
                self._active_cache_last_channel.append(
                    self.batch_asr_states[batch_idx].cache_last_channel[:, speaker_id]
                )
                self._active_cache_last_time.append(self.batch_asr_states[batch_idx].cache_last_time[:, speaker_id])
                self._active_cache_last_channel_len.append(
                    self.batch_asr_states[batch_idx].cache_last_channel_len[speaker_id]
                )
        if len(self._active_chunk_audio) == 0:
            return None, None, None, None

        # Convert chunk audio and target info to tensors
        active_chunk_audio = torch.stack(self._active_chunk_audio)
        active_chunk_lengths = torch.stack(self._active_chunk_lengths)
        active_speaker_targets = torch.stack(self._active_speaker_targets)
        inactive_speaker_targets = torch.stack(self._inactive_speaker_targets)

        # Update active speaker attributes
        self.active_previous_hypotheses = deepcopy(self._active_previous_hypotheses)
        self.active_asr_pred_out_stream = deepcopy(self._active_asr_pred_out_stream)
        self.active_cache_last_channel = torch.stack(self._active_cache_last_channel).transpose(0, 1)
        self.active_cache_last_time = torch.stack(self._active_cache_last_time).transpose(0, 1)
        self.active_cache_last_channel_len = torch.stack(self._active_cache_last_channel_len)
        return active_chunk_audio, active_chunk_lengths, active_speaker_targets, inactive_speaker_targets

    def update_seglsts(self, offset: int):
        """
        Take the ASR states and update the seglsts.

        Args:
            offset (int): The offset of the chunk.
        """
        for asr_state in self.batch_asr_states:
            asr_state.update_sessionwise_seglsts_for_parallel(offset=offset)
