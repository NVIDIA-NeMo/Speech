# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Waveform-to-token inference with TensorRT preprocessing, encoder and TDT networks."""

import json
import re
from pathlib import Path

import sentencepiece
import torch

from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import BatchedTDTOutput
from nemo.collections.asr.parts.submodules.tensorrt_encoder import TensorRTEncoder
from nemo.collections.asr.parts.submodules.tensorrt_plan import TensorRTPlan
from nemo.collections.asr.parts.submodules.tensorrt_tdt_decoder import load_tensorrt_tdt_decoder


class TensorRTASR:
    """Run a packaged offline SentencePiece TDT model on one CUDA stream.

    Inputs are mono waveforms at the bundle's sample rate.
    TensorRT executes every neural component and log-mel preprocessing; PyTorch manages tensors and the cached autoregressive loop.
    Instances are not safe for concurrent calls.
    """

    def __init__(
        self, directory: str | Path, device: str | torch.device = "cuda", use_cuda_graphs: bool = True
    ) -> None:
        directory = Path(directory)
        self.config = json.loads((directory / "config.json").read_text())
        if self.config["format_version"] != 1:
            raise ValueError("Unsupported waveform ASR bundle version.")
        self.preprocessor = TensorRTPlan(directory / "preprocessor.plan", device)
        self.encoder = TensorRTEncoder(directory / "encoder.plan", device)
        self.decoder = load_tensorrt_tdt_decoder(directory / "decoder", device, use_cuda_graphs)
        self.device = self.encoder.device
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(directory / "tokenizer.model"))
        self._punctuation = (
            re.compile(self.config["punctuation_pattern"]) if self.config["punctuation_pattern"] else None
        )
        self._language = re.compile(self.config["language_pattern"]) if self.config["strip_lang_tags"] else None

    def validate_inputs(self, audio: torch.Tensor, lengths: torch.Tensor) -> None:
        if audio.ndim != 2 or not 1 <= audio.shape[0] <= self.config["max_batch"]:
            raise ValueError("Expected waveforms [batch, samples] within the engine batch profile.")
        if not 1 <= audio.shape[1] <= self.config["max_samples"]:
            raise ValueError("Padded waveform width is outside the engine profile.")
        if audio.dtype != torch.float32 or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected float32 audio and integer sample lengths.")
        if lengths.ndim != 1 or lengths.shape[0] != audio.shape[0]:
            raise ValueError("Expected one sample length per waveform.")
        if bool(((lengths < 0) | (lengths > audio.shape[1])).any()):
            raise ValueError("Audio lengths must be between zero and the padded sample dimension.")
        minimum = self.config["min_valid_samples"]
        if bool(((lengths > 0) & (lengths < minimum)).any()):
            raise ValueError(f"Nonempty waveforms require at least {minimum} samples for normalization.")
        if not bool(torch.isfinite(audio).all()):
            raise ValueError("Waveforms must contain finite values.")

    @torch.inference_mode()
    def __call__(self, audio: torch.Tensor, lengths: torch.Tensor) -> BatchedTDTOutput:
        self.validate_inputs(audio, lengths)
        audio = audio.to(self.device)
        lengths = lengths.to(device=self.device, dtype=torch.int64)
        mask = torch.arange(audio.shape[1], device=self.device)[None, :] < lengths[:, None]
        audio = audio.masked_fill(~mask, 0)
        minimum = self.config["min_samples"]
        if audio.shape[1] < minimum:
            audio = torch.nn.functional.pad(audio, (0, minimum - audio.shape[1]))
        features = self.preprocessor(input_signal=audio, length=lengths)
        x = features["processed_signal"]
        if x.shape[2] < self.config["min_feature_frames"]:
            x = torch.nn.functional.pad(x, (0, self.config["min_feature_frames"] - x.shape[2]))
        encoded, encoded_lengths = self.encoder(x, features["processed_length"])
        return self.decoder.decode_tensors(encoded, encoded_lengths)

    def texts(self, result: BatchedTDTOutput) -> list[str]:
        """Decode tokens using the source checkpoint's punctuation and language-tag policy."""
        tokens, lengths = result.tokens.cpu().tolist(), result.lengths.cpu().tolist()
        texts = []
        for row, count in zip(tokens, lengths):
            text = self.tokenizer.decode(row[:count])
            if self._punctuation:
                text = self._punctuation.sub(r"\2", text)
            if self._language:
                text = self._language.sub("", text).strip()
            texts.append(text)
        return texts
