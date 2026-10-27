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

"""Export a checkpoint's waveform preprocessor and build an FP32 TensorRT engine."""

from argparse import ArgumentParser
from pathlib import Path

import torch

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.utils.trt_utils import build_engine


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Source .nemo checkpoint.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=800)
    parser.add_argument("--opt-samples", type=int, default=160000)
    parser.add_argument("--max-samples", type=int, default=640000)
    parser.add_argument("--opt-batch", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    args = parser.parse_args()
    if not 0 < args.min_samples <= args.opt_samples <= args.max_samples:
        parser.error("Require 0 < min-samples <= opt-samples <= max-samples.")
    if not 1 <= args.opt_batch <= args.max_batch or args.workspace_gib <= 0:
        parser.error("Require 1 <= opt-batch <= max-batch and positive workspace.")
    for path in (args.onnx, args.engine):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    model = ASRModel.restore_from(str(args.model), map_location=torch.device("cpu"))
    preprocessor = model.preprocessor.float().eval()
    if not isinstance(preprocessor, AudioToMelSpectrogramPreprocessor):
        raise TypeError("Only AudioToMelSpectrogramPreprocessor is supported.")
    if preprocessor.featurizer.normalize not in (None, False, "per_feature"):
        raise ValueError("Dynamic export currently supports per_feature or no normalization.")
    if preprocessor.featurizer.frame_splicing != 1:
        raise ValueError("Dynamic export currently requires frame_splicing=1.")
    preprocessor.export(
        str(args.onnx),
        input_example=(torch.zeros(1, args.opt_samples), torch.tensor([args.opt_samples])),
        onnx_opset_version=17,
        dynamic_axes={
            "input_signal": {0: "batch", 1: "samples"},
            "length": {0: "batch"},
        },
    )
    del model, preprocessor
    torch.cuda.init()
    build_engine(
        onnx_path=str(args.onnx),
        output_path=str(args.engine),
        fp16=False,
        tf32=False,
        input_profile={
            "input_signal": [
                (1, args.min_samples),
                (args.opt_batch, args.opt_samples),
                (args.max_batch, args.max_samples),
            ],
            "length": [(1,), (args.opt_batch,), (args.max_batch,)],
        },
        workspace_size=int(args.workspace_gib * 1024**3),
    )


if __name__ == "__main__":
    main()
