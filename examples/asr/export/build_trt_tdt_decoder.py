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

"""Build FP32 TensorRT plans that preserve TDT prediction and projection caching."""

import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules.rnnt import RNNTDecoder
from nemo.utils.trt_utils import build_engine


class Prediction(torch.nn.Module):
    def __init__(self, decoder: RNNTDecoder, joint: torch.nn.Module) -> None:
        super().__init__()
        self.decoder, self.projection = decoder, joint.pred

    def forward(
        self, labels: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, state = self.decoder.predict(labels, (hidden, cell), add_sos=False)
        return self.projection(output), state[0], state[1]


class Joint(torch.nn.Module):
    def __init__(self, joint: torch.nn.Module) -> None:
        super().__init__()
        self.joint = joint

    def forward(self, encoder: torch.Tensor, predictor: torch.Tensor) -> torch.Tensor:
        return self.joint.joint_after_projection(encoder, predictor)


def build(
    model: ASRModel, output: Path, max_batch: int = 16, max_frames: int = 512, max_symbols: int | None = None
) -> None:
    """Build three engines: predictor with projection, encoder projection, and joint."""
    if max_symbols is None:
        max_symbols = model.cfg.decoding.get("greedy", {}).get("max_symbols")
    if not isinstance(max_symbols, int) or max_symbols < 1:
        raise ValueError("Set a positive --max-symbols or decoding.greedy.max_symbols in the checkpoint.")
    if max_batch < 1 or max_frames < 1:
        raise ValueError("Profile bounds and max_symbols must be positive.")
    output.mkdir(parents=True, exist_ok=False)
    decoder, joint = model.decoder.float().eval(), model.joint.float().eval()
    if not isinstance(decoder, RNNTDecoder) or not decoder.blank_as_pad:
        raise ValueError("A blank-padded LSTM prediction network is required.")
    layers, hidden = decoder.pred_rnn_layers, decoder.pred_hidden
    enc, projected = joint.enc.in_features, joint.enc.out_features
    durations = list(model.cfg.decoding.durations)
    config = {
        "format_version": 1,
        "blank_id": decoder.blank_idx,
        "durations": durations,
        "pred_rnn_layers": layers,
        "pred_hidden": hidden,
        "encoder_hidden": enc,
        "num_classes_with_blank": joint.num_classes_with_blank,
        "max_symbols_per_step": max_symbols,
        "max_batch": max_batch,
        "max_frames": max_frames,
    }
    cases = [
        (
            "predictor",
            Prediction(decoder, joint),
            (
                torch.full((1, 1), decoder.blank_idx, dtype=torch.long),
                torch.zeros(layers, 1, hidden),
                torch.zeros(layers, 1, hidden),
            ),
            ["labels", "hidden", "cell"],
            ["projected", "hidden_out", "cell_out"],
            {
                "labels": {0: "batch"},
                "hidden": {1: "batch"},
                "cell": {1: "batch"},
                "projected": {0: "batch"},
                "hidden_out": {1: "batch"},
                "cell_out": {1: "batch"},
            },
            {
                "labels": [(1, 1), (max_batch, 1), (max_batch, 1)],
                "hidden": [(layers, 1, hidden), (layers, max_batch, hidden), (layers, max_batch, hidden)],
                "cell": [(layers, 1, hidden), (layers, max_batch, hidden), (layers, max_batch, hidden)],
            },
        ),
        (
            "encoder_projection",
            joint.enc,
            (torch.zeros(1, 2, enc),),
            ["encoded"],
            ["projected"],
            {"encoded": {0: "batch", 1: "time"}, "projected": {0: "batch", 1: "time"}},
            {"encoded": [(1, 1, enc), (max_batch, min(128, max_frames), enc), (max_batch, max_frames, enc)]},
        ),
        (
            "joint",
            Joint(joint),
            (torch.zeros(1, 1, projected), torch.zeros(1, 1, projected)),
            ["encoder", "predictor"],
            ["logits"],
            {"encoder": {0: "batch"}, "predictor": {0: "batch"}, "logits": {0: "batch"}},
            {
                name: [(1, 1, projected), (max_batch, 1, projected), (max_batch, 1, projected)]
                for name in ("encoder", "predictor")
            },
        ),
    ]
    for name, module, args, inputs, outputs, dynamic, profile in cases:
        onnx = output / (name + ".onnx")
        torch.onnx.export(
            module.eval(),
            args,
            str(onnx),
            input_names=inputs,
            output_names=outputs,
            dynamic_axes=dynamic,
            opset_version=17,
            dynamo=False,
        )
        build_engine(
            onnx_path=str(onnx),
            output_path=str(output / (name + ".plan")),
            fp16=False,
            tf32=False,
            input_profile=profile,
            workspace_size=2 * 1024**3,
        )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--max-symbols", type=int, help="Override the checkpoint decoding.greedy.max_symbols limit.")
    args = parser.parse_args()
    model = ASRModel.restore_from(str(args.model), map_location="cpu").float().eval()
    build(model, args.output, args.max_batch, args.max_frames, args.max_symbols)


if __name__ == "__main__":
    main()
