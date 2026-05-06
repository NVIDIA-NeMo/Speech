# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Single registered model class for the NeMo Speech LM (SALM) vLLM plugin.

Architecture: NeMo speech encoder (e.g. FastConformer) + projection + LLM.

A single ``NeMoSpeechLMForConditionalGeneration`` covers every supported
backbone family. Backbone-specific behavior (architecture choice, weight
rename rules, optional LoRA merge, mamba state passthroughs) lives in
``backends.py`` and is selected once at ``__init__`` time via
``make_backend(config)``. The class declares ``IsHybrid`` /
``SupportsMambaPrefixCaching`` so vLLM's hybrid KV-cache allocator picks up
NemotronH backbones; for transformer backbones the runtime
``ModelConfig.is_hybrid`` property returns False because ``config.py``
populates ``text_config.layer_types`` with all-attention markers (vLLM's
granite-4.0-micro escape hatch).

Requires NeMo toolkit for the audio encoder:
    pip install nemo_toolkit[asr]
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMambaPrefixCaching,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from nemo.collections.speechlm2.vllm.salm.audio import (
    NeMoSpeechLMAudioInputs,
    NeMoSpeechLMDummyInputsBuilder,
    NeMoSpeechLMMultiModalProcessor,
    NeMoSpeechLMProcessingInfo,
    _load_nemo_perception,
)
from nemo.collections.speechlm2.vllm.salm.backends import HybridBackend, make_backend
from nemo.collections.speechlm2.vllm.salm.config import _AUDIO_PLACEHOLDER


@MULTIMODAL_REGISTRY.register_processor(
    NeMoSpeechLMMultiModalProcessor,
    info=NeMoSpeechLMProcessingInfo,
    dummy_inputs=NeMoSpeechLMDummyInputsBuilder,
)
class NeMoSpeechLMForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    IsHybrid,
    SupportsMambaPrefixCaching,
):
    """Backbone-agnostic NeMo SpeechLM. Composition with a backend handles per-backbone details."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return _AUDIO_PLACEHOLDER
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        backend = make_backend(config)
        self._backend = backend

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=backend.architectures(),
            )

        with self._mark_tower_model(vllm_config, {"audio"}):
            self.perception = _load_nemo_perception(config.perception)
            self.perception = self.perception.to(torch.float32)

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    # ── audio processing ──

    def _parse_audio_input(
        self,
        audio_signal: torch.Tensor | list[torch.Tensor] | None = None,
        audio_signal_length: torch.Tensor | None = None,
        **kwargs,
    ) -> NeMoSpeechLMAudioInputs | None:
        if audio_signal is None:
            return None

        if not isinstance(audio_signal_length, torch.Tensor):
            raise ValueError(
                "audio_signal_length must be a torch.Tensor; got " f"{type(audio_signal_length).__name__}."
            )

        if isinstance(audio_signal, list):
            max_len = max(a.shape[-1] for a in audio_signal)
            padded = [torch.nn.functional.pad(a, (0, max_len - a.shape[-1])) for a in audio_signal]
            audio_signal = torch.stack(padded, dim=0)

        return NeMoSpeechLMAudioInputs(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )

    def _process_audio(self, audio_input: NeMoSpeechLMAudioInputs) -> tuple[torch.Tensor, ...]:
        # Real device placement happens at init via _mark_tower_model +
        # get_mm_mapping; this .to() is a no-op guard kept for paranoia.
        device = next(self.perception.parameters()).device
        self.perception = self.perception.to(device)

        audio_signal = audio_input.audio_signal
        if isinstance(audio_signal, list):
            audio_signal = torch.stack(audio_signal, dim=0)
        audio_signal = audio_signal.to(device=device, dtype=torch.float32)
        audio_lengths = audio_input.audio_signal_length.to(device=device)

        with torch.no_grad():
            audio_embeds, audio_embed_lens = self.perception(
                input_signal=audio_signal,
                input_signal_length=audio_lengths,
            )

        audio_embeds = audio_embeds.to(torch.bfloat16)

        return tuple(audio_embeds[i, : audio_embed_lens[i]] for i in range(audio_embeds.shape[0]))

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        audio_input = self._parse_audio_input(**kwargs)
        if audio_input is None:
            return []
        return self._process_audio(audio_input)

    # ── forward / logits ──

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="perception.proj",
            tower_model="perception.encoder",
        )

    # ── weight loading ──

    def _load_perception_weights(self, perception_weights: dict[str, torch.Tensor]) -> set[str]:
        float32_weights = {k: v.float() for k, v in perception_weights.items()}
        self.perception.load_state_dict(float32_weights, strict=False)
        self.perception = self.perception.to(torch.float32)
        return {"perception." + k for k in perception_weights}

    @staticmethod
    def _split_perception_llm(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], list[tuple[str, torch.Tensor]]]:
        perception: dict[str, torch.Tensor] = {}
        llm: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if "._extra_state" in name:
                continue
            if name.startswith("perception."):
                perception[name[len("perception.") :]] = tensor
            else:
                llm.append((name, tensor))
        return perception, llm

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        perception_weights, llm_raw = self._split_perception_llm(weights)
        loaded_perception = self._load_perception_weights(perception_weights)

        preprocessed = self._backend.preprocess_llm_weights(llm_raw)
        hf_weights = self._backend.nemo_to_hf_llm_weights(preprocessed)
        combined = (("language_model." + n, t) for n, t in hf_weights)

        loader = AutoWeightsLoader(self)
        loaded_llm = loader.load_weights(combined)

        return loaded_llm | loaded_perception

    # ── vLLM IsHybrid mamba state classmethods ──
    #
    # Reached only when vLLM's ``ModelConfig.is_hybrid`` returns True at
    # runtime (NemotronH backbones). For transformer backbones the
    # ``text_config.layer_types`` shim in ``config.py`` flips ``is_hybrid``
    # off at runtime so vLLM never calls into these.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> Any:
        return HybridBackend.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> Any:
        return HybridBackend.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls) -> Any:
        return HybridBackend.get_mamba_state_copy_func()
