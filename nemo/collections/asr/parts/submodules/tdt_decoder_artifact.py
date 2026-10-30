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

"""Package the prediction and joint networks of a greedy TDT decoder, without the encoder.

The decoder itself is :class:`~nemo.collections.asr.parts.submodules.rnnt_greedy_decoding.GreedyBatchedTDTInfer`;
its ``decode_tensors`` method is the tensor interface that the serving examples call.
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedTDTInfer
from nemo.core.classes.common import Serialization


def build_tdt_decoder(
    decoder: torch.nn.Module,
    joint: torch.nn.Module,
    blank_id: int,
    durations: list[int],
    max_symbols_per_step: int = 10,
    use_cuda_graphs: bool = True,
) -> GreedyBatchedTDTInfer:
    """Create the greedy TDT decoder for the tensor interface, checking that the networks and parameters agree.

    The prediction network must use ``blank_as_pad``, and both networks must share one device and dtype.
    They are switched to evaluation mode; language-model fusion, confidence and alignments stay off.
    """
    if not decoder.blank_as_pad:
        raise ValueError("Complete TDT decoding requires blank_as_pad=True.")
    if not durations or any(type(d) is not int or d < 0 for d in durations):
        raise ValueError("TDT durations must be nonnegative integers.")
    if max_symbols_per_step < 1:
        raise ValueError("max_symbols_per_step must be positive.")
    if blank_id != decoder.blank_idx or joint.num_classes_with_blank != blank_id + 1 + len(durations):
        raise ValueError("Prediction vocabulary, blank ID and TDT joint dimensions disagree.")
    decoder_parameter, joint_parameter = next(decoder.parameters()), next(joint.parameters())
    if decoder_parameter.device != joint_parameter.device or decoder_parameter.dtype != joint_parameter.dtype:
        raise ValueError("Prediction and joint networks must use the same device and dtype.")
    return GreedyBatchedTDTInfer(
        decoder_model=decoder.eval(),
        joint_model=joint.eval(),
        blank_index=blank_id,
        durations=list(durations),
        max_symbols_per_step=max_symbols_per_step,
        include_duration=True,
        use_cuda_graph_decoder=use_cuda_graphs,
    )


def save_tdt_decoder(infer: GreedyBatchedTDTInfer, path: str | Path) -> None:
    """Save the prediction and joint parameters with their configuration and the decoding limits, without the encoder."""
    payload = {
        "format_version": 1,
        "blank_id": infer.decoder.blank_idx,
        "durations": list(infer.durations),
        "max_symbols_per_step": infer.max_symbols,
    }
    for name in ("decoder", "joint"):
        module = getattr(infer, name)
        payload[f"{name}_config"] = OmegaConf.to_container(module.to_config_dict(), resolve=True)
        payload[f"{name}_state"] = {key: value.detach().cpu() for key, value in module.state_dict().items()}
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)


def load_tdt_decoder(path: str | Path, device: str = "cuda", use_cuda_graphs: bool = True) -> GreedyBatchedTDTInfer:
    """Load a trusted decoder artifact of :func:`save_tdt_decoder`, with FP32 network parameters."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1:
        raise ValueError("Unsupported TDT decoder artifact version.")
    modules = {}
    for name in ("decoder", "joint"):
        module = Serialization.from_config_dict(OmegaConf.create(payload[f"{name}_config"]))
        module.load_state_dict(payload[f"{name}_state"], strict=True)
        modules[name] = module.float().eval().to(device)
    return build_tdt_decoder(
        **modules,
        blank_id=payload["blank_id"],
        durations=payload["durations"],
        max_symbols_per_step=payload["max_symbols_per_step"],
        use_cuda_graphs=use_cuda_graphs,
    )
