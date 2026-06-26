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
from omegaconf import DictConfig, open_dict

from nemo.core.classes.common import safe_instantiate


def build_speaker_tokens(speaker_cfg: DictConfig | dict | None, tokenizer) -> list[int]:
    """Resolve native ``<spk:N>`` speaker-token ids from the LLM tokenizer.

    The tokenizer is expected to already contain ``template.format(i=0)..template.format(i=max_speakers-1)``
    as fixed entries. This helper validates that lookup does not grow the tokenizer and that ids match the
    configured contiguous range.
    """
    if speaker_cfg is None or not bool(speaker_cfg.get("enable", True)):
        return []
    template = speaker_cfg.get("template", "<spk:{i}>")
    max_speakers = int(speaker_cfg.get("max_speakers", 10))
    base_token_id = int(speaker_cfg.get("base_token_id", 100))

    before = tokenizer.vocab_size
    speaker_token_ids: list[int] = []
    for i in range(max_speakers):
        token = template.format(i=i)
        tid = tokenizer.token_to_id(token)
        expected = base_token_id + i
        if tid is None:
            raise ValueError(
                f"Could not resolve speaker token {token!r} in the LLM tokenizer. "
                "Ensure pretrained_llm points at the patched tokenizer dir "
                "(e.g. '...-spk/') produced by patch_nano_v3_speaker_tokens.py."
            )
        if tid != expected:
            raise ValueError(
                f"Speaker token {token!r} resolved to id {tid}, expected "
                f"{expected} (= base_token_id={base_token_id} + i={i}). The "
                "tokenizer does not match the configured speaker_tokens layout."
            )
        speaker_token_ids.append(tid)
    after = tokenizer.vocab_size
    if before != after:
        raise ValueError(
            f"Resolving speaker tokens grew the tokenizer ({before} -> {after}); "
            "speaker_tokens requires the tokens to already exist in the patched "
            "tokenizer (no resize_token_embeddings on this path)."
        )
    return speaker_token_ids


def maybe_init_lss_loss(loss_cfg: DictConfig | None, speaker_token_ids: list[int] | None = None):
    """Optionally build the auxiliary Latent Speaker Supervision (LSS) loss.

    The loss is instantiated from ``cfg.lss_loss`` via Hydra ``_target_``. SALM computes CE
    separately with ``loss_parallel()``, so any LSS-provided CE term is disabled here.
    """
    if loss_cfg is None:
        return None
    if loss_cfg.get("include_ce_loss", False):
        raise ValueError(
            "model.lss_loss.include_ce_loss must be False (or omitted) on the SALM "
            "automodel path: SALM already computes CE inside loss_parallel(), so a "
            "second CE term inside LSS would be double-counted."
        )
    with open_dict(loss_cfg):
        loss_cfg.setdefault("pad_id", -100)
        loss_cfg.setdefault("include_ce_loss", False)
        if loss_cfg.get("speaker_token_ids", None) is None:
            if not speaker_token_ids:
                raise ValueError(
                    "model.lss_loss is configured but no speaker_token_ids are available. "
                    "Either set model.speaker_tokens (so ids are derived from the patched "
                    "tokenizer's native <spk:N> entries) or pass an explicit "
                    "model.lss_loss.speaker_token_ids list."
                )
            loss_cfg.speaker_token_ids = list(speaker_token_ids)
    return safe_instantiate(loss_cfg)
