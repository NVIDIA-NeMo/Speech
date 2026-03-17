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
from collections import defaultdict

import torch

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.speechlm2.parts.precision import fp32_precision


class SECS:
    """
    Computes Speacker encoder cossine similarity (SECS) on generated audio with pretrained speaker encoder model.
    """

    def __init__(self, pretrained_se_name: str) -> None:
        self.speaker_encoder = None  # load into memory on reset()
        self.pretrained_se_name = pretrained_se_name
        self._secs = defaultdict(list)

    def reset(self):
        # Cleaning up GPU memory before we load ASRModel, because it may already
        # be quite fragmented and close to the limit after observing many
        # dynamic shapes during the training epoch.
        torch.cuda.memory.empty_cache()
        with fp32_precision():
            self.speaker_encoder = EncDecSpeakerLabelModel.from_pretrained(model_name=self.pretrained_se_name).eval()

        return self

    def update(
        self,
        name: str,
        target_audio: torch.Tensor,
        target_audio_lens: torch.Tensor,
        pred_audio: torch.Tensor,
        pred_audio_lens: torch.Tensor,
    ) -> None:
        if self.speaker_encoder is None:
            self.reset()

        with fp32_precision():
            with torch.no_grad():
                _, t_g = self.speaker_encoder(input_signal=target_audio, input_signal_length=target_audio_lens.long())
                _, s_g = self.speaker_encoder(input_signal=pred_audio, input_signal_length=pred_audio_lens.long())
            secs = torch.nn.functional.cosine_similarity(t_g, s_g, dim=-1).mean()

        self._secs[name].append(secs)

    def compute(self) -> dict[str, torch.Tensor]:
        """Computes the final score and deallocates ASR and partial results."""
        corpus_metric = {}
        avg_secs = []
        for name in self._secs.keys():
            secs = torch.stack(self._secs[name]).mean()
            corpus_metric[f"secs_{name}"] = secs
            avg_secs.append(secs)

        corpus_metric["secs"] = torch.stack(avg_secs).mean()
        self._secs.clear()
        self.speaker_encoder = None  # free up GPU memory
        torch.cuda.memory.empty_cache()
        return corpus_metric
