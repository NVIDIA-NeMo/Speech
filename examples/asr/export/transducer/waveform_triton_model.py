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

"""Triton backend that executes complete waveform-to-text TensorRT TDT inference."""

import json
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack

from nemo.collections.asr.parts.submodules.tensorrt_asr import TensorRTASR


class TritonPythonModel:
    def initialize(self, args):
        if args["model_instance_kind"] != "GPU":
            raise ValueError("Waveform ASR requires a GPU model instance.")
        device = f"cuda:{args['model_instance_device_id']}"
        torch.cuda.set_device(device)
        config = json.loads(args["model_config"])
        graphs = config.get("parameters", {}).get("use_cuda_graphs", {}).get("string_value", "true")
        if graphs not in ("true", "false"):
            raise ValueError("use_cuda_graphs must be true or false.")
        self.model = TensorRTASR(Path(__file__).parent, device, graphs == "true")

    @torch.inference_mode()
    def execute(self, requests):
        responses = [None] * len(requests)
        valid = []
        for index, request in enumerate(requests):
            try:
                audio = from_dlpack(pb_utils.get_input_tensor_by_name(request, "AUDIO").to_dlpack())
                lengths = from_dlpack(pb_utils.get_input_tensor_by_name(request, "AUDIO_LENGTH").to_dlpack())
                if lengths.ndim != 2 or lengths.shape[1] != 1:
                    raise ValueError("AUDIO_LENGTH must have shape [batch, 1].")
                lengths = lengths[:, 0]
                self.model.validate_inputs(audio, lengths)
                valid.append((index, audio, lengths))
            except (ValueError, AttributeError) as exc:
                responses[index] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(exc), pb_utils.TritonError.INVALID_ARG)
                )
            except RuntimeError as exc:
                responses[index] = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
        if not valid:
            return responses
        try:
            # Triton combines requests with equal padded widths into one model batch.
            audio = torch.cat([x.to(self.model.device) for _, x, _ in valid])
            lengths = torch.cat([x.to(self.model.device) for _, _, x in valid])
            result = self.model(audio, lengths)
            texts = self.model.texts(result)
            offset = 0
            for index, request_audio, _ in valid:
                size = request_audio.shape[0]
                outputs = [pb_utils.Tensor("TEXT", np.array(texts[offset : offset + size], dtype=object)[:, None])]
                for name, tensor in zip(("TOKENS", "TOKEN_LENGTH", "TIMESTAMPS", "TOKEN_DURATIONS"), result):
                    tensor = tensor[offset : offset + size]
                    if name == "TOKEN_LENGTH":
                        tensor = tensor[:, None]
                    outputs.append(pb_utils.Tensor.from_dlpack(name, tensor.contiguous()))
                responses[index] = pb_utils.InferenceResponse(output_tensors=outputs)
                offset += size
        except (ValueError, RuntimeError) as exc:
            for index, _, _ in valid:
                responses[index] = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
        return responses
