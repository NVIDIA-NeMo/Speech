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

"""MTP speculative-decoding draft model for NeMo SpeechLM checkpoints.

``NeMoSpeechLMMTP`` wraps vLLM's ``NemotronHMTP`` and adapts the
weight-loading step for the NeMo checkpoint layout where all LLM
weights (including MTP layers) carry an ``llm.`` prefix:

    NeMo checkpoint            NemotronHMTP expects
    ──────────────────────     ────────────────────
    llm.mtp.layers.0.*    →    mtp.layers.0.*
    llm.lm_head.weight    →    lm_head.weight
    llm.model.embed_*     →    backbone.embeddings.*

The embedding alias is required by vLLM's Nemotron-H MTP loader even though
the proposer subsequently shares the table with the target model.
"""

from collections.abc import Iterable

import torch
from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP
from vllm.model_executor.models.utils import _merge_multimodal_embeddings

from nemo.collections.speechlm2.vllm.salm.audio import _pad_to_vocab_size


def _remap_nemo_mtp_weights(
    items: Iterable[tuple[str, torch.Tensor]],
    target_vocab: int | None = None,
    expected_layer_modules: int | None = None,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Select and map exported SpeechLM draft weights to vLLM aliases.

    ``expected_layer_modules`` is the number of sublayers instantiated for one
    EAGLE prediction step (the hybrid pattern length), not the logical draft K.
    Rejecting higher checkpoint indices prevents vLLM from silently dropping
    weights from a distinct multi-head checkpoint routed as a repeated head.
    Non-``llm.`` tensors are omitted because the target model loads them.
    """
    for name, tensor in items:
        if "._extra_state" in name:
            continue

        if not name.startswith("llm."):
            continue
        name = name[len("llm.") :]

        parts = name.split(".")
        if expected_layer_modules is not None and len(parts) > 2 and parts[:2] == ["mtp", "layers"]:
            try:
                layer_index = int(parts[2])
            except ValueError:
                layer_index = None
            if layer_index is not None and layer_index >= expected_layer_modules:
                raise ValueError(
                    f"Checkpoint contains {name!r}, but the configured repeated MTP step instantiates "
                    f"only {expected_layer_modules} hybrid-pattern layer module(s). This looks like a "
                    f"distinct multi-head checkpoint or stale hybrid-pattern metadata. Verify that the "
                    f"exported mtp.hybrid_override_pattern describes the built head; vLLM's NemotronH "
                    f"MTP proposer can reuse only one EAGLE-style prediction step."
                )

        # NeMo stores every expert in one rank-3 tensor, while vLLM's
        # NemotronHMTP loader expects one transposed matrix per expert.
        if name.endswith(".experts.down_projs"):
            prefix = name.removesuffix(".experts.down_projs")
            for expert_idx, expert_tensor in enumerate(tensor):
                yield f"{prefix}.experts.{expert_idx}.down_proj.weight", expert_tensor.t()
            continue
        if name.endswith(".experts.gate_and_up_projs"):
            prefix = name.removesuffix(".experts.gate_and_up_projs")
            for expert_idx, expert_tensor in enumerate(tensor):
                yield f"{prefix}.experts.{expert_idx}.up_proj.weight", expert_tensor.t()
            continue

        # NemotronHMTP.load_weights only admits embedding names containing
        # ``embeddings`` and then maps this backbone alias to
        # ``model.embed_tokens``. Passing model.embed_tokens through directly
        # is silently skipped and DefaultModelLoader reports it uninitialized.
        if name == "model.embed_tokens.weight":
            name = "backbone.embeddings.weight"

        if name in {"backbone.embeddings.weight", "lm_head.weight"} and target_vocab is not None:
            tensor = _pad_to_vocab_size(tensor, target_vocab)
        yield name, tensor


class NeMoSpeechLMMTP(NemotronHMTP):
    """NemotronH MTP draft model for NeMo SpeechLM checkpoints.

    Extends NemotronHMTP in two ways:

    * ``load_weights`` strips the ``llm.`` prefix from checkpoint names.
    * ``embed_input_ids`` fuses audio-feature embeddings into the token
      embeddings at placeholder positions, exactly like the target model.
      The MTP heads were trained on the same mixed text+audio embedding
      stream as the backbone, so the draft must see it too. vLLM probes
      ``draft_model.embed_input_ids(ids, multimodal_embeddings=None)`` at
      load time (``llm_base_proposer.load_model``); without this method
      the probe raises AttributeError and speculative decoding silently
      falls back to text-only draft inputs, which collapses acceptance
      rates on audio prompts.
    """

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed token IDs and merge audio embeddings at placeholder positions.

        The target model inherits this fusion from ``SupportsMultiModal``; the
        draft must implement it itself because ``NemotronHMTP`` is not
        multimodal. The embedding table is shared from the target by vLLM's
        MTP framework, so text-token rows stay identical.
        """
        inputs_embeds = self.model.get_input_embeddings(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        if is_multimodal is None:
            raise ValueError("is_multimodal is required when multimodal_embeddings are provided.")

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load only the SALM-prefixed weights required by the reusable draft head."""
        # NeMoSpeechLMConfig delegates this to the already padded text-config
        # vocabulary used to construct NemotronHMTP. Reading config directly
        # avoids coupling padding to vLLM's internal module names.
        target_vocab = int(self.config.vocab_size)
        if target_vocab <= 0:
            raise ValueError(f"Draft model vocabulary size must be positive, got {target_vocab}.")

        # vLLM instantiates one physical module per character in the configured
        # hybrid pattern. Derive the checkpoint bound from serialized metadata
        # instead of depending on a predictor-internal implementation detail.
        pattern = self.config.mtp_hybrid_override_pattern
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"mtp_hybrid_override_pattern must be a non-empty string, got {pattern!r}.")
        return super().load_weights(
            _remap_nemo_mtp_weights(
                weights,
                target_vocab,
                expected_layer_modules=len(pattern),
            )
        )
