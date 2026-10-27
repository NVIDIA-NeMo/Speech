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

"""TensorRT networks with NeMo's complete cached TDT decoding loop."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from nemo.collections.asr.modules.rnnt import RNNTDecoder
from nemo.collections.asr.parts.submodules.tdt_decoder import TDTDecoder
from nemo.collections.asr.parts.submodules.tensorrt_plan import TensorRTPlan


class TensorRTTDTDecoder(TDTDecoder):
    """Load cached predictor and joint plans built by build_trt_tdt_decoder.py.

    Network calculations use TensorRT; state updates, output assembly and the decoding loop use PyTorch, including the native CUDA graph implementation.
    The plans contain all network weights; no native model checkpoint is loaded.
    """

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda", use_cuda_graphs: bool = True) -> "TensorRTTDTDecoder":
        directory = Path(path)
        config = json.loads((directory / "config.json").read_text())
        if config["format_version"] != 1:
            raise ValueError("Unsupported TensorRT TDT artifact version.")
        predictor = _Prediction(directory, config, device)
        joint = _Joint(directory, config, device)
        return cls(
            predictor, joint, config["blank_id"], config["durations"], config["max_symbols_per_step"], use_cuda_graphs
        )

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Build a TensorRT decoder bundle from the source checkpoint.")


class _PlanModule(torch.nn.Module):
    def __init__(self, device: str | torch.device) -> None:
        super().__init__()
        # The native loop reads parameter device/dtype; plans own the actual weights.
        self.register_parameter(
            "_device_dtype",
            torch.nn.Parameter(torch.empty(0, device=device, dtype=torch.float32), requires_grad=False),
        )


class _Prediction(_PlanModule):
    batch_replace_states_mask = RNNTDecoder.batch_replace_states_mask
    batch_replace_states_all = RNNTDecoder.batch_replace_states_all
    clone_state = RNNTDecoder.clone_state
    batch_split_states = RNNTDecoder.batch_split_states
    batch_unsplit_states = RNNTDecoder.batch_unsplit_states
    initialize_state = RNNTDecoder.initialize_state

    def __init__(self, directory: Path, config: dict, device: str | torch.device) -> None:
        super().__init__(device)
        self.engine = TensorRTPlan(directory / "predictor.plan", device, cache_contexts=True)
        self.blank_as_pad = True
        self.blank_idx = config["blank_id"]
        self.pred_rnn_layers = config["pred_rnn_layers"]
        self.pred_hidden = config["pred_hidden"]
        self.random_state_sampling = False

    def predict(
        self,
        y: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        add_sos: bool = False,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if add_sos or y is None or state is None:
            raise ValueError("TensorRT prediction requires explicit labels and recurrent states, without added SOS.")
        out = self.engine(labels=y, hidden=state[0], cell=state[1])
        return out["projected"], (out["hidden_out"], out["cell_out"])


class _Joint(_PlanModule):
    def __init__(self, directory: Path, config: dict, device: str | torch.device) -> None:
        super().__init__(device)
        self.encoder = TensorRTPlan(directory / "encoder_projection.plan", device)
        self.joint = TensorRTPlan(directory / "joint.plan", device, cache_contexts=True)
        self.enc = SimpleNamespace(in_features=config["encoder_hidden"])
        self.num_classes_with_blank = config["num_classes_with_blank"]

    def project_encoder(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.encoder(encoded=encoder_output)["projected"]

    def project_prednet(self, prednet_output: torch.Tensor) -> torch.Tensor:
        return prednet_output

    def joint_after_projection(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.joint(encoder=f, predictor=g)["logits"]
