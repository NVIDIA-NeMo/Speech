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
"""Dataset for StreamingSALM training."""

import logging

import torch
import torch.utils.data
from lhotse import CutSet
from lhotse.dataset.collation import collate_audio

from nemo.collections.audio.parts.utils.transforms import Resample


class StreamingSALMDataset(torch.utils.data.Dataset):
    """
    Dataset for StreamingSALM. Returns raw audio and transcript text.
    All sequence construction (Mimi encoding, forced alignment, interleaving)
    happens in the model's prepare_inputs method.

    Operates directly on Lhotse Cuts (no NeMoMultimodalConversation wrapper).

    Produces both 24 kHz audio (for Mimi) and 16 kHz numpy arrays
    (for QwenForcedAligner) so that resampling runs in dataloader workers
    and overlaps with GPU training.
    """

    EXPECTED_SAMPLE_RATE = 24000  # Mimi native sample rate
    ALIGNER_SAMPLE_RATE = 16000  # QwenForcedAligner expects 16 kHz

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._resample_to_16k = Resample(
            orig_freq=self.EXPECTED_SAMPLE_RATE,
            new_freq=self.ALIGNER_SAMPLE_RATE,
        )

    def __getitem__(self, cuts: CutSet) -> dict | None:
        try:
            audios, audio_lens, cuts = collate_audio(cuts, fault_tolerant=True)
        except Exception as e:
            logging.warning(f"Error collating audio from cuts: {e}")
            return None
        if len(cuts) == 0:
            return None

        # Resample to 16 kHz for forced aligner (CPU, in dataloader workers).
        # Store as list of per-utterance numpy arrays to avoid GPU→CPU transfer later.
        # Uses cached Resample transform with precompiled kernels for speed.
        audios_16k_np = []
        for i in range(audios.shape[0]):
            utt = audios[i, : audio_lens[i]]
            utt_16k = self._resample_to_16k(utt)
            audios_16k_np.append(utt_16k.numpy())

        transcripts = [cut.supervisions[0].text for cut in cuts]
        return {
            "audios": audios,
            "audio_lens": audio_lens,
            "audios_16k": audios_16k_np,
            "transcripts": transcripts,
            "cuts": cuts.drop_in_memory_data(),
            "sample_rate": self.EXPECTED_SAMPLE_RATE,
        }
