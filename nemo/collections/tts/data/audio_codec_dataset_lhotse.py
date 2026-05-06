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

from typing import Dict, Optional

import torch
from lhotse import CutSet
from lhotse.dataset.collation import collate_audio

from nemo.utils import logging


class AudioCodecLhotseDataset(torch.utils.data.Dataset):
    """
    A Lhotse-based dataset for audio codec model training.

    It is a simple dataset that mostly just loads the audio samples.
    In addition, it performs the following operations:
    * Resampling to the target sample rate
    * Sanity checks on the audio

    The operations below are handled directly by Lhotse according to the configuration
    applied in `AudioCodecModel._get_lhotse_dataloader()`:
    * Duration filtering
    * Any additional transformations configured in Lhotse during its construction are
      applied to the audio as it is loaded in `collate_audio()`.
    """

    def __init__(
        self,
        sample_rate: int,
        sanity_check_audio: bool = False,
        min_samples_for_sanity: Optional[int] = None,
    ):
        """
        Args:
            sample_rate: The sample rate to resample the audio to.
            sanity_check_audio: If True, perform sanity checks on the loaded audio.
            min_samples_for_sanity: cuts should have at least this many samples or an
                                    error will be raised. Only used when
                                    `sanity_check_audio` is True.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.sanity_check_audio = sanity_check_audio
        self.min_samples_for_sanity = min_samples_for_sanity

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Loads the specified cuts and performs the operations listed above.

        Args:
            cuts: A Lhotse CutSet object.
        Returns:
            A dictionary with the `audio` and `audio_lens` tensors.
        """
        # Resample the audio to the target sample rate. We need to do this manually
        # because Lhotse only resamples its standard `recording` field automatically,
        # not custom fields like `target_audio`.
        for cut in cuts:
            cut.target_audio = cut.target_audio.resample(self.sample_rate)

        # Load and collate the audio, applying any transformations that were
        # configured in Lhotse in the process.
        # Note: fault_tolerant=False for now to avoid masking errors until we are more
        # confident in the new loader.
        batch_audio, batch_audio_len = collate_audio(cuts, recording_field="target_audio", fault_tolerant=False)

        # Sanity checks on the audio and its length
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
