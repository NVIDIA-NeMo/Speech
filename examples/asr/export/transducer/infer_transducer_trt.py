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

"""Transcribe mono audio with a TensorRT encoder and a matching NeMo RNNT/TDT checkpoint."""

import json
from argparse import ArgumentParser
from pathlib import Path

import soundfile as sf
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel, EncDecRNNTModel
from nemo.collections.asr.parts.mixins import PromptStreamingMixin
from nemo.collections.asr.parts.submodules.tensorrt_encoder import TensorRTEncoder
from nemo.utils import logging


def main() -> None:
    """Keep preprocessing and greedy transducer decoding in PyTorch."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="The .nemo checkpoint used to export this encoder.")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True, help="Mono files at the model's sample rate.")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file for predictions.")
    parser.add_argument("--batch-size", type=int, default=1, help="Must fit the engine's optimization profile.")
    parser.add_argument("--compare-native", action="store_true", help="Also record native encoder predictions.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive.")
    if args.output.exists():
        raise FileExistsError(args.output)
    model = ASRModel.restore_from(str(args.model), map_location=torch.device("cpu")).eval().float()
    if not isinstance(model, EncDecRNNTModel):
        raise TypeError("This example requires an RNNT or TDT model.")
    if isinstance(model, PromptStreamingMixin):
        raise TypeError(
            f"{type(model).__name__} conditions the encoder output on a language prompt; this example passes "
            "the encoder output to the decoder without the prompt projection and does not support prompt models."
        )
    if getattr(model.encoder, "att_context_style", "regular") != "regular":
        logging.warning(
            "The encoder is a cache-aware streaming encoder. The engine has no caches and runs it offline, "
            "over whole utterances, so the predictions differ from streaming inference."
        )
    model = model.cuda()
    decoding = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding))
    with open_dict(decoding):
        decoding.strategy = "greedy_batch"
        # Decoder CUDA graphs are a throughput feature.
        # This accuracy example decodes eagerly, so that the TensorRT and native transcripts come from the same
        # plain decoder run on every supported PyTorch version.
        decoding.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(decoding)
    model.preprocessor.featurizer.dither = 0.0
    encoder = TensorRTEncoder(args.engine)
    sample_rate = model.cfg.preprocessor.sample_rate
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output, torch.inference_mode():
        for offset in range(0, len(args.audio), args.batch_size):
            paths = args.audio[offset : offset + args.batch_size]
            waves = []
            for path in paths:
                waveform, rate = sf.read(path, dtype="float32")
                if rate != sample_rate or waveform.ndim != 1 or len(waveform) == 0:
                    raise ValueError(f"Expected nonempty mono audio at {sample_rate} Hz: {path}")
                waves.append(torch.from_numpy(waveform))
            lengths = torch.tensor([wave.numel() for wave in waves], dtype=torch.int64, device="cuda")
            audio = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True).cuda()
            features, feature_lengths = model.preprocessor(input_signal=audio, length=lengths)
            encoded, encoded_lengths = encoder(features, feature_lengths)
            hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
                encoded.float(), encoded_lengths, return_hypotheses=True
            )
            native = None
            if args.compare_native:
                native_encoded, native_lengths = model.encoder(audio_signal=features, length=feature_lengths)
                native = model.decoding.rnnt_decoder_predictions_tensor(
                    native_encoded, native_lengths, return_hypotheses=True
                )
            for i, (path, hypothesis) in enumerate(zip(paths, hypotheses)):
                row = {"audio_filepath": str(path), "pred_text": hypothesis.text}
                if native is not None:
                    row["native_pred_text"] = native[i].text
                    row["text_equal"] = native[i].text == hypothesis.text
                output.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
