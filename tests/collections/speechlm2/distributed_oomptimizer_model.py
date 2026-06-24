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
import torch.nn.functional as F

from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class SingleBlockDistributedOOMptimizerModel(pl.LightningModule):
    """Single-transformer-block model used by the distributed OOMptimizer functional test."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.vocab_size = int(cfg.get("vocab_size", 64))
        self.sample_rate = int(cfg.get("sample_rate", 32))
        self.frame_stride = int(cfg.get("frame_stride", 4))
        hidden_size = int(cfg.get("hidden_size", 128))
        num_heads = int(cfg.get("num_heads", 4))
        ffn_hidden_size = int(cfg.get("ffn_hidden_size", hidden_size * 4))
        dropout = float(cfg.get("dropout", 0.0))
        self.activation_reserve_elements_per_frame = int(
            float(cfg.get("activation_reserve_mb_per_frame", 0.0)) * 1024 * 1024 // 4
        )
        self.max_activation_reserve_frames = int(cfg.get("max_activation_reserve_frames", 160))

        self.input_projection = torch.nn.Linear(self.frame_stride, hidden_size)
        self.encoder = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.classifier = torch.nn.Linear(hidden_size, self.vocab_size)

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
        tokens = batch["tokens"].long()
        pad = (-audio.shape[1]) % self.frame_stride
        if pad:
            audio = F.pad(audio, (0, pad))

        frames = audio.reshape(audio.shape[0], -1, self.frame_stride)
        hidden = self.input_projection(frames)
        hidden = self.encoder(hidden)
        logits = self.classifier(hidden.mean(dim=1))
        target = tokens[:, 0].remainder(self.vocab_size)
        loss = F.cross_entropy(logits, target)

        self._reserve_peak_memory(hidden)
        return {"loss": loss}

    def _reserve_peak_memory(self, hidden: torch.Tensor) -> None:
        if self.activation_reserve_elements_per_frame <= 0:
            return
        reserve_frames = min(int(hidden.shape[1]), self.max_activation_reserve_frames)
        if reserve_frames <= 0:
            return

        # Keep transformer compute small while making memory pressure scale with sequence length.
        reserve = hidden.new_empty(
            (int(hidden.shape[0]), reserve_frames, self.activation_reserve_elements_per_frame), dtype=torch.float32
        )
        reserve[:, :, :1].zero_()

    def configure_optimizers(self) -> dict:
        return {"optimizer": torch.optim.SGD(self.parameters(), lr=1e-3)}
