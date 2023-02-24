# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Tuple

import librosa
import numpy as np
import torch

from nemo.collections.asr.models import EncDecClassificationModel
from nemo.collections.tts.parts.utils.tts_dataset_utils import normalize_volume
from nemo.utils import logging


class AudioTrimmer(ABC):
    """Interface for silence trimming implementations
    """

    @abstractmethod
    def trim_audio(self, audio: np.array, sample_rate: int, audio_id: str) -> Tuple[np.array, int, int]:
        """Trim starting and trailing silence from the input audio.
           Args:
               audio: Numpy array containing audio samples. Float [-1.0, 1.0] format.
               sample_rate: Sample rate of input audio.
               audio_id: String identifier (eg. file name) used for logging.

           Returns numpy array with trimmed audio, and integer sample indices representing the start and end
           of speech within the original audio array.
        """
        raise NotImplementedError


class EnergyAudioTrimmer(AudioTrimmer):
    def __init__(
        self,
        db_threshold: int = 50,
        ref_amplitude: float = 1.0,
        speech_frame_threshold: int = 1,
        trim_win_length: int = 2048,
        trim_hop_length: int = 512,
        pad_seconds: float = 0.1,
        volume_norm: bool = True,
    ) -> None:
        """Energy/power based silence trimming using Librosa backend.
           Args:
               db_threshold: Audio frames at least db_threshold decibels below ref_amplitude will be
                 considered silence.
               ref_amplitude: Amplitude threshold for classifying speech versus silence.
               speech_frame_threshold: Start and end of speech will be detected where there are at least
                 speech_frame_threshold consecutive audio frames classified as speech. Setting this value higher
                 is more robust to false-positives (silence detected as speech), but setting it too high may result
                 in very short speech segments being cut out from the audio.
               trim_win_length: Length of audio frames to use when doing speech detection. This does not need to match
                 the win_length used any other part of the code or model.
               trim_hop_length: Stride of audio frames to use when doing speech detection. This does not need to match
                 the hop_length used any other part of the code or model.
               pad_seconds: Audio duration in seconds to keep before and after each speech segment.
                 Set this to at least 0.1 to avoid cutting off any speech audio, with larger values
                 being safer but increasing the average silence duration left afterwards.
               volume_norm: Whether to normalize the volume of audio before doing speech detection.
        """
        assert db_threshold >= 0
        assert ref_amplitude >= 0
        assert speech_frame_threshold > 0
        assert trim_win_length > 0
        assert trim_hop_length > 0

        self.db_threshold = db_threshold
        self.ref_amplitude = ref_amplitude
        self.speech_frame_threshold = speech_frame_threshold
        self.trim_win_length = trim_win_length
        self.trim_hop_length = trim_hop_length
        self.pad_seconds = pad_seconds
        self.volume_norm = volume_norm

    def trim_audio(self, audio: np.array, sample_rate: int, audio_id: str = "") -> Tuple[np.array, int, int]:
        if self.volume_norm:
            # Normalize volume so we have a fixed scale relative to the reference amplitude
            audio = normalize_volume(audio=audio, volume_level=1.0)

        speech_frames = librosa.effects._signal_to_frame_nonsilent(
            audio,
            ref=self.ref_amplitude,
            frame_length=self.trim_win_length,
            hop_length=self.trim_hop_length,
            top_db=self.db_threshold,
        )

        start_frame, end_frame = get_start_and_end_of_speech_frames(
            is_speech=speech_frames, speech_frame_threshold=self.speech_frame_threshold, audio_id=audio_id,
        )

        start_sample = librosa.core.frames_to_samples(start_frame, hop_length=self.trim_hop_length)
        end_sample = librosa.core.frames_to_samples(end_frame, hop_length=self.trim_hop_length)

        start_sample, end_sample = pad_sample_indices(
            start_sample=start_sample,
            end_sample=end_sample,
            max_sample=audio.shape[0],
            sample_rate=sample_rate,
            pad_seconds=self.pad_seconds,
        )

        trimmed_audio = audio[start_sample:end_sample]

        return trimmed_audio, start_sample, end_sample


class VadAudioTrimmer(AudioTrimmer):
    def __init__(
        self,
        model_name: str = "vad_multilingual_marblenet",
        vad_sample_rate: int = 16000,
        vad_threshold: float = 0.5,
        device: str = "cpu",
        speech_frame_threshold: int = 1,
        trim_win_length: int = 4096,
        trim_hop_length: int = 1024,
        pad_seconds: float = 0.1,
        volume_norm: bool = True,
    ) -> None:
        """Voice activity detection (VAD) based silence trimming.

           Args:
               model_name: NeMo VAD model to load. Valid configurations can be found with
                 EncDecClassificationModel.list_available_models()
               vad_sample_rate: Sample rate used for pretrained VAD model.
               vad_threshold: Softmax probability [0, 1] of VAD output, above which audio frames will be classified
                 as speech.
               device: Device "cpu" or "cuda" to use for running the VAD model.
               trim_win_length: Length of audio frames to use when doing speech detection. This does not need to match
                 the win_length used any other part of the code or model.
               trim_hop_length: Stride of audio frames to use when doing speech detection. This does not need to match
                 the hop_length used any other part of the code or model.
               pad_seconds: Audio duration in seconds to keep before and after each speech segment.
                 Set this to at least 0.1 to avoid cutting off any speech audio, with larger values
                 being safer but increasing the average silence duration left afterwards.
               volume_norm: Whether to normalize the volume of audio before doing speech detection.
        """
        assert vad_sample_rate > 0
        assert vad_threshold >= 0
        assert speech_frame_threshold > 0
        assert trim_win_length > 0
        assert trim_hop_length > 0

        self.device = device
        self.vad_model = EncDecClassificationModel.from_pretrained(model_name=model_name).eval().to(self.device)
        self.vad_sample_rate = vad_sample_rate
        self.vad_threshold = vad_threshold

        self.speech_frame_threshold = speech_frame_threshold
        self.trim_win_length = trim_win_length
        self.trim_hop_length = trim_hop_length
        # Window shift neeeded in order to center frames
        self.trim_shift = self.trim_win_length // 2

        self.pad_seconds = pad_seconds
        self.volume_norm = volume_norm

    def _detect_speech(self, audio: np.array) -> np.array:
        # [num_frames, win_length]
        audio_frames = librosa.util.frame(
            audio, frame_length=self.trim_win_length, hop_length=self.trim_hop_length
        ).transpose()
        audio_frame_lengths = audio_frames.shape[0] * [self.trim_win_length]

        # [num_frames, win_length]
        audio_signal = torch.tensor(audio_frames, dtype=torch.float32, device=self.device)
        # [1]
        audio_signal_len = torch.tensor(audio_frame_lengths, dtype=torch.int32, device=self.device)
        # VAD outputs 2 values for each audio frame with logits indicating the likelihood that
        # each frame is non-speech or speech, respectively.
        # [num_frames, 2]
        log_probs = self.vad_model(input_signal=audio_signal, input_signal_length=audio_signal_len)
        probs = torch.softmax(log_probs, dim=-1)
        probs = probs.detach().cpu().numpy()
        # [num_frames]
        speech_probs = probs[:, 1]
        speech_frames = speech_probs >= self.vad_threshold

        return speech_frames

    def _scale_sample_indices(self, start_sample: int, end_sample: int, sample_rate: int) -> Tuple[int, int]:
        sample_rate_ratio = sample_rate / self.vad_sample_rate
        start_sample = int(sample_rate_ratio * start_sample)
        end_sample = int(sample_rate_ratio * end_sample)
        return start_sample, end_sample

    def trim_audio(self, audio: np.array, sample_rate: int, audio_id: str = "") -> Tuple[np.array, int, int]:
        if sample_rate == self.vad_sample_rate:
            vad_audio = audio
        else:
            # Resample audio to match sample rate of VAD model
            vad_audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.vad_sample_rate)

        if self.volume_norm:
            # Normalize volume so we have a fixed scale relative to the reference amplitude
            vad_audio = normalize_volume(audio=vad_audio, volume_level=1.0)

        speech_frames = self._detect_speech(audio=vad_audio)

        start_frame, end_frame = get_start_and_end_of_speech_frames(
            is_speech=speech_frames, speech_frame_threshold=self.speech_frame_threshold, audio_id=audio_id,
        )

        if start_frame == 0:
            start_sample = 0
        else:
            start_sample = librosa.core.frames_to_samples(start_frame, hop_length=self.trim_hop_length)
            start_sample += self.trim_shift

        # Avoid trimming off the end because VAD model is not trained to classify partial end frames.
        if end_frame == speech_frames.shape[0]:
            end_sample = vad_audio.shape[0]
        else:
            end_sample = librosa.core.frames_to_samples(end_frame, hop_length=self.trim_hop_length)
            end_sample += self.trim_shift

        if sample_rate != self.vad_sample_rate:
            # Convert sample indices back to input sample rate
            start_sample, end_sample = self._scale_sample_indices(
                start_sample=start_sample, end_sample=end_sample, sample_rate=sample_rate
            )

        start_sample, end_sample = pad_sample_indices(
            start_sample=start_sample,
            end_sample=end_sample,
            max_sample=audio.shape[0],
            sample_rate=sample_rate,
            pad_seconds=self.pad_seconds,
        )

        trimmed_audio = audio[start_sample:end_sample]

        return trimmed_audio, start_sample, end_sample


def get_start_and_end_of_speech_frames(
    is_speech: np.array, speech_frame_threshold: int, audio_id: str = ""
) -> Tuple[int, int]:
    """Finds the speech frames corresponding to the start and end of speech for an utterance.
       Args:
           is_speech: [num_frames] boolean array with true entries labeling speech frames.
           speech_frame_threshold: The number of consecutive speech frames required to classify the speech boundaries.
           audio_id: String identifier (eg. file name) used for logging.

       Returns integers representing the frame indices of the start (inclusive) and end (exclusive) of speech.
    """
    num_frames = is_speech.shape[0]

    # Iterate forwards over the utterance until we find the first speech_frame_threshold consecutive speech frames.
    start_frame = None
    for i in range(0, num_frames - speech_frame_threshold + 1):
        high_i = i + speech_frame_threshold
        if all(is_speech[i:high_i]):
            start_frame = i
            break

    # Iterate backwards over the utterance until we find the last speech_frame_threshold consecutive speech frames.
    end_frame = None
    for i in range(num_frames, speech_frame_threshold - 1, -1):
        low_i = i - speech_frame_threshold
        if all(is_speech[low_i:i]):
            end_frame = i
            break

    if start_frame is None:
        logging.warning(f"Could not find start of speech for '{audio_id}'")
        start_frame = 0

    if end_frame is None:
        logging.warning(f"Could not find end of speech for '{audio_id}'")
        end_frame = num_frames

    return start_frame, end_frame


def pad_sample_indices(
    start_sample: int, end_sample: int, max_sample: int, sample_rate: int, pad_seconds: float
) -> Tuple[int, int]:
    """Shift the input sample indices by pad_seconds in front and back within [0, max_sample]
       Args:
           start_sample: Start sample index
           end_sample: End sample index
           max_sample: Maximum sample index
           sample_rate: Sample rate of audio
           pad_seconds: Amount to pad/shift the indices by.

       Returns the sample indices after padding by the input amount.
    """
    pad_samples = int(pad_seconds * sample_rate)
    start_sample = start_sample - pad_samples
    end_sample = end_sample + pad_samples

    start_sample = max(0, start_sample)
    end_sample = min(max_sample, end_sample)

    return start_sample, end_sample
