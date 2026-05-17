# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import random
import re
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import soundfile as sf
import torch
from lhotse import CutSet

from nemo.utils import logging

_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_DATASET_IN_SPEAKER = re.compile(r"(?:^|[\s|])Dataset:([^\s|]+)")


class AudioCodecLhotseDataset(torch.utils.data.Dataset):
    """
    A Lhotse-based dataset for audio codec model training.

    It is a simple dataset that mostly just loads the audio samples.
    In addition, it performs the following operations:
    * Resampling to the target sample rate
    * Random truncation of each cut's `target_audio` to a fixed duration
    * Sanity checks on the audio

    The operations below are handled directly by Lhotse according to the configuration
    applied in `AudioCodecModel._get_lhotse_dataloader()`:
    * Duration filtering
    * Any additional transformations configured in Lhotse during its construction are
      applied to the audio as it is loaded in `load_audio()`.
    """

    def __init__(
        self,
        sample_rate: int,
        truncate_duration: float,
        sanity_check_audio: bool = False,
        min_samples_for_sanity: Optional[int] = None,
        log_audio: bool = False,
        log_audio_dir: Union[str, Path] = "logged_audio",
        log_audio_num_batches: int = 3,
    ):
        """
        Args:
            sample_rate: The sample rate to resample the audio to.
            truncate_duration: Length of each training window in seconds. A random
                window of this length is taken from each cut's `target_audio` field
                (not from the parent `recording`, which may span a much longer file).
            sanity_check_audio: If True, perform sanity checks on the loaded audio.
            min_samples_for_sanity: cuts should have at least this many samples or an
                                    error will be raised. Only used when
                                    `sanity_check_audio` is True.
            log_audio: If True, save the original `target_audio` waveforms from
                       the first few batches before any dataset resampling.
            log_audio_dir: Directory where debug wav files will be written.
            log_audio_num_batches: Number of initial batches to log per dataset
                                   instance or dataloader worker.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.truncate_duration = truncate_duration
        self.truncate_samples = int(truncate_duration * sample_rate)
        self.sanity_check_audio = sanity_check_audio
        self.min_samples_for_sanity = min_samples_for_sanity
        self.log_audio = log_audio
        self.log_audio_dir = Path(log_audio_dir)
        self.log_audio_num_batches = log_audio_num_batches
        self._logged_audio_batches = 0

    def _maybe_log_audio(self, cuts: CutSet):
        """
        Save original target_audio waveforms for the first few batches.
        """
        if not self.log_audio or self._logged_audio_batches >= self.log_audio_num_batches:
            return

        self._log_target_audio_without_resampling(cuts)
        self._logged_audio_batches += 1

    def _log_target_audio_without_resampling(self, cuts: CutSet):
        """
        Save each cut's `target_audio` before `target_audio.resample()` is applied.

        This intentionally uses the custom `target_audio` recording and its own
        sampling rate, not `self.sample_rate`. To keep the debug files trustworthy,
        fail fast if the recording already has Lhotse audio transforms attached.
        """
        self.log_audio_dir.mkdir(parents=True, exist_ok=True)

        for cut in cuts:
            recording = cut.target_audio
            transform_names = self._recording_transform_names(recording)
            if transform_names:
                raise RuntimeError(
                    "Cannot log untransformed target_audio because the recording "
                    f"already has Lhotse audio transforms attached: {transform_names}. "
                    f"cut_id={cut.id}, recording_id={recording.id}"
                )

            audio = cut.load_custom("target_audio")
            speaker = getattr(cut.supervisions[0], "speaker", None) if cut.supervisions else None
            filename = self._recording_id_to_wav_name(recording.id, recording.sampling_rate, speaker=speaker)
            path = self.log_audio_dir / filename
            sf.write(str(path), self._audio_for_soundfile(audio), samplerate=recording.sampling_rate)
            logging.info(
                f"Saved original target_audio for cut_id={cut.id}, recording_id={recording.id}, "
                f"sampling_rate={recording.sampling_rate}, shape={audio.shape}, path={path}"
            )

    @staticmethod
    def _recording_transform_names(recording) -> list[str]:
        transform_names = []
        for transform in recording.transforms or []:
            if isinstance(transform, dict):
                transform_names.append(transform.get("name", str(transform)))
            else:
                transform_names.append(type(transform).__name__)
        return transform_names

    @staticmethod
    def _recording_id_to_wav_name(recording_id: str, sampling_rate: int, speaker: Optional[str] = None) -> str:
        safe_id = _SAFE_FILENAME_CHARS.sub("_", str(recording_id)).strip("._")
        if not safe_id:
            safe_id = "recording"
        dataset = AudioCodecLhotseDataset._dataset_from_speaker(speaker)
        if dataset is not None:
            safe_id = f"{dataset}_{safe_id}"
        return f"{safe_id}_{sampling_rate}Hz.wav"

    @staticmethod
    def _dataset_from_speaker(speaker: Optional[str]) -> Optional[str]:
        if speaker is None:
            return None
        match = _DATASET_IN_SPEAKER.search(speaker)
        if match is None:
            return None
        safe_dataset = _SAFE_FILENAME_CHARS.sub("_", match.group(1)).strip("._")
        return safe_dataset or None

    @staticmethod
    def _audio_for_soundfile(audio):
        if audio.ndim == 1:
            return audio
        if audio.shape[0] == 1:
            return audio[0]
        return audio.T

    def _load_and_truncate_target_audio(self, cut) -> torch.Tensor:
        """
        Load `target_audio`, resample, and return a random segmentof length `truncate_duration`.
        """
        if not cut.has_custom("target_audio"):
            raise ValueError(f"Cut {cut.id} is missing custom field 'target_audio'")

        target_audio_recording = cut.target_audio.resample(self.sample_rate)
        audio = target_audio_recording.load_audio()
        if audio.ndim > 1:
            audio = audio.squeeze(0)

        num_samples = audio.shape[-1]
        if num_samples < self.truncate_samples:
            raise ValueError(
                f"target_audio is shorter than truncate_duration: "
                f"cut_id={cut.id}, target_audio_id={target_audio_recording.id}, "
                f"num_samples={num_samples}, required={self.truncate_samples}, "
                f"truncate_duration={self.truncate_duration}s"
            )

        start = random.randint(0, num_samples - self.truncate_samples)
        window = audio[start : start + self.truncate_samples]
        return torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Loads the specified cuts and performs the operations listed above.

        Args:
            cuts: A Lhotse CutSet object.
        Returns:
            A dictionary with the `audio` and `audio_lens` tensors.
        """
        self._maybe_log_audio(cuts)
        # Load, resample and truncate the audio
        audio_list = [self._load_and_truncate_target_audio(cut) for cut in cuts]
        batch_audio = torch.stack(audio_list, dim=0)
        batch_audio_len = torch.full(
            (len(audio_list),),
            self.truncate_samples,
            dtype=torch.int32,
        )

        if self.sanity_check_audio:
            self._sanity_check_audio(batch_audio, batch_audio_len, cuts)

        return {
            "audio": batch_audio,
            "audio_lens": batch_audio_len,
        }

    def _sanity_check_audio(self, audio: torch.Tensor, audio_len: torch.Tensor, cuts: CutSet = None):
        """
        Performs sanity checks on the audio.
        * Errors out on clearly invalid data.
        * Warns if suspicious data is encountered.
        """
        # --- Error cases ---

        # Audio length is unexpectedly short
        if self.min_samples_for_sanity is not None and audio_len.min() < self.min_samples_for_sanity:
            raise ValueError(
                f"Audio length is less than {self.min_samples_for_sanity} samples (min: {audio_len.min()})"
            )
        # Audio contains NaN or Inf values
        if audio.isnan().any():
            raise ValueError("Audio contains NaN values")
        if audio.isinf().any():
            raise ValueError("Audio contains Inf values")

        # --- Warning cases ---

        # Detect audio samples way outside the expected [-1.0, 1.0) range.
        max_permitted_abs_val = (
            1.5  # Far enough outside the expected range that it would likely incidate corrupted data
        )
        per_item_max = audio.abs().max(dim=1).values
        offending_incides = (per_item_max > max_permitted_abs_val).nonzero(as_tuple=True)[0].tolist()
        if len(offending_incides) > 0:
            # Cuts with invalid samples were found. Log the offending cuts.
            cut_list = list(cuts)
            for i in offending_incides:
                cut = cut_list[i]
                cut_meta = (
                    f"id={cut.id}, "
                    f"recording_id={cut.target_audio.id}, "
                    f"start={cut.start}, "
                    f"duration={cut.duration}, "
                    f"num_samples={int(audio_len[i].item())}"
                )
                logging.warning(
                    f"WARNING: Audio contains a sample with an absolute value greater than {max_permitted_abs_val}: {per_item_max[i].item()} (cut: {cut_meta})"
                )
