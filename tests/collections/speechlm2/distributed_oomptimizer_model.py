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

import lightning.pytorch as pl
import torch

from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class TinyDistributedOOMptimizerModel(pl.LightningModule):
    """Tiny CUDA model used by the distributed OOMptimizer functional test."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.vocab_size = int(cfg.get("vocab_size", 32))
        self.scratch_elements_per_sample = int(cfg.get("scratch_mb_per_sample", 96) * 1024 * 1024 // 4)
        self.scale = torch.nn.Parameter(torch.ones(()))

    @property
    def oomptimizer_schema(self) -> dict:
        return {
            "cls": dict,
            "inputs": [
                {"name": "audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.vocab_size,
                },
            ],
        }

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        audio = batch["audio"].float()
        tokens = batch["tokens"].float()
        batch_size = int(audio.shape[0])
        if self.scratch_elements_per_sample > 0:
            torch.empty((batch_size, self.scratch_elements_per_sample), device=audio.device, dtype=torch.float32)
        prediction = self.scale * audio.mean()
        target = tokens.mean() / max(1, self.vocab_size)
        return {"loss": (prediction - target).square()}

    def configure_optimizers(self) -> dict:
        return {"optimizer": torch.optim.SGD(self.parameters(), lr=1e-3)}
