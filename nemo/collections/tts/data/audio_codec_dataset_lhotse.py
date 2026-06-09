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
from typing import Dict

import numpy as np
import torch
from lhotse import CutSet

from nemo.utils import logging


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
    * Minimum duration filtering
    * Any additional transformations configured in Lhotse during its construction are
      applied to the audio as it is loaded in `load_audio()`.
    """

    def __init__(
        self,
        sample_rate: int,
        segment_duration: float,
        sanity_check_audio: bool = False,
    ):
        """
        Args:
            sample_rate: The sample rate to resample the audio to.
            segment_duration: Length of each training segment in seconds. A random
                segment of this length is taken from each cut's `target_audio` field
                (not from the parent `recording`, which may span a much longer duration).
            sanity_check_audio: If True, perform sanity checks on the loaded audio.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * sample_rate)
        self.sanity_check_audio = sanity_check_audio
        # Error out if audio is suspiciously short (leaving some slack for resampling).
        self.min_samples_for_sanity = max(1, self.segment_samples - 5)

    def _load_and_truncate_target_audio(self, cut) -> torch.Tensor:
        """
        Load `target_audio`, resample, and return a random segment of length `segment_duration`.
        """
        if not cut.has_custom("target_audio"):
            raise ValueError(f"Cut {cut.id} is missing custom field 'target_audio'")

        target_audio_recording = cut.target_audio.resample(self.sample_rate)
        # Load the target audio, resampling and applying and Lhotse transformation in the process
        audio = target_audio_recording.load_audio()
        if audio.ndim > 1:
            audio = audio.squeeze(0)

        num_samples = audio.shape[-1]
        if num_samples < self.segment_samples:
            raise ValueError(
                f"target_audio is shorter than segment_duration: "
                f"cut_id={cut.id}, target_audio_id={target_audio_recording.id}, "
                f"num_samples={num_samples}, required={self.segment_samples}, "
                f"segment_duration={self.segment_duration}s"
            )

        # Randomly select a segment of the audio
        start = random.randint(0, num_samples - self.segment_samples)
        segment = audio[start : start + self.segment_samples]
        return torch.from_numpy(np.ascontiguousarray(segment, dtype=np.float32))

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Loads the specified cuts and performs the operations listed above.

        Args:
            cuts: A Lhotse CutSet object.
        Returns:
            A dictionary with the `audio` and `audio_lens` tensors.
        """
        # Load, resample and truncate the audio
        audio_list = [self._load_and_truncate_target_audio(cut) for cut in cuts]
        batch_audio = torch.stack(audio_list, dim=0)
        batch_audio_len = torch.full(
            (len(audio_list),),
            self.segment_samples,
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
        if audio_len.min() < self.min_samples_for_sanity:
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
