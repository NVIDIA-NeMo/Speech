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

"""Triton Python backend: one invocation executes the complete TDT decoding loop."""

import json
from pathlib import Path

import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack

from nemo.collections.asr.parts.submodules.tdt_decoder import TDTDecoder


class TritonPythonModel:
    def initialize(self, args: dict[str, str]) -> None:
        if args["model_instance_kind"] != "GPU":
            raise ValueError("The TDT serving example requires a GPU model instance.")
        device = f"cuda:{args['model_instance_device_id']}"
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        config = json.loads(args["model_config"])
        use_graphs = config.get("parameters", {}).get("use_cuda_graphs", {}).get("string_value", "true")
        if use_graphs not in ("true", "false"):
            raise ValueError("use_cuda_graphs must be true or false.")
        self.decoder = TDTDecoder.load(Path(__file__).with_name("decoder.pt"), device, use_graphs == "true")

    @torch.inference_mode()
    def execute(self, requests: list) -> list:
        responses = [None] * len(requests)
        valid = []
        for index, request in enumerate(requests):
            try:
                encoded = from_dlpack(pb_utils.get_input_tensor_by_name(request, "ENCODER_OUTPUT").to_dlpack())
                lengths = from_dlpack(pb_utils.get_input_tensor_by_name(request, "ENCODED_LENGTH").to_dlpack())
                if lengths.ndim != 2 or lengths.shape[1] != 1:
                    raise ValueError("ENCODED_LENGTH must have shape [batch, 1].")
                lengths = lengths[:, 0]
                self.decoder.validate_inputs(encoded, lengths)
                valid.append((index, encoded, lengths))
            except (ValueError, AttributeError) as exc:
                responses[index] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(exc), pb_utils.TritonError.INVALID_ARG)
                )
            except RuntimeError as exc:
                responses[index] = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
        if not valid:
            return responses
        try:
            # Triton groups equal non-batch shapes; combine requests into one decoder invocation.
            encoded = torch.cat([x.to(self.decoder.device) for _, x, _ in valid])
            lengths = torch.cat([x.to(self.decoder.device) for _, _, x in valid])
            result = self.decoder(encoded, lengths)
            offset = 0
            for index, request_encoded, _ in valid:
                size = request_encoded.shape[0]
                outputs = []
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
