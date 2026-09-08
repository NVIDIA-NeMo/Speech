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
from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig, open_dict

from nemo.core.classes.common import safe_instantiate

__all__ = ["MultiSpeakerConfig", "build_speaker_tokens", "maybe_init_lss_loss"]


@dataclass(frozen=True)
class MultiSpeakerConfig:
    """Settings for RTTM-derived SOT speaker targets, shared by SALM and the streaming SpeechLM.

    Field names follow the SALM/phPEE reference so recipes port across unchanged. Note that
    ``from_dict`` collapses the YAML-facing ``window_stride`` / ``sample_rate`` /
    ``subsampling_factor`` knobs into the frame-rate fields the helpers actually take.
    """

    num_speakers: int = 4
    no_rttm_to_ones: bool = True
    num_sample_per_mel_frame: int = 160
    num_mel_frame_per_target_frame: int = 8
    # Bound on the brute-force permutation search in `fix_speaker_activity`. Uncapped, an
    # 8-active-speaker cut enumerates 8! = 40320 permutations and materialises a
    # (P, n_tokens, n_frames) array -- measured at 48.7 s per cut in a dataloader worker.
    max_alignment_permutations: Optional[int] = 720
    # Value written for a cut with no resolvable RTTM. ParallelExpertEncoder detects rows that are
    # entirely <= this and substitutes its embedded diarizer for them. Leaving `no_rttm_to_ones`
    # output as literal ones instead would train the speaker kernel on "every speaker active at
    # every frame" -- confidently wrong supervision rather than an abstention.
    missing_rttm_target: float = -1.0
    # --- streaming SpeechLM only (ignored by SALM) ---
    enable: bool = True
    speaker_token_template: str = "<spk:{i}>"

    @property
    def max_permutable(self) -> Optional[int]:
        """Largest speaker count whose factorial fits the permutation budget (720 -> 6).

        ``fix_speaker_activity`` takes a *speaker count*, while the reference config expresses the
        limit as a *permutation budget*; this bridges the two.
        """
        if self.max_alignment_permutations is None:
            return None
        count, factorial = 1, 1
        while factorial * (count + 1) <= int(self.max_alignment_permutations):
            count += 1
            factorial *= count
        return count

    @staticmethod
    def from_dict(cfg: "DictConfig | dict | None") -> "Optional[MultiSpeakerConfig]":
        """Build a config from a raw settings dict, or ``None`` when no SOT settings are given."""
        if cfg is None:
            return None
        return MultiSpeakerConfig(
            num_speakers=int(cfg.get('num_speakers', 4)),
            no_rttm_to_ones=cfg.get('no_rttm_to_ones', True),
            num_sample_per_mel_frame=int(cfg.get('window_stride', 0.01) * cfg.get('sample_rate', 16000)),
            num_mel_frame_per_target_frame=int(cfg.get('subsampling_factor', 8)),
            missing_rttm_target=float(cfg.get('missing_rttm_target', -1.0)),
            max_alignment_permutations=(
                None
                if cfg.get('max_alignment_permutations', 720) is None
                else int(cfg.get('max_alignment_permutations', 720))
            ),
            enable=bool(cfg.get('enable', True)),
            speaker_token_template=cfg.get('speaker_token_template', "<spk:{i}>"),
        )


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
