# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import logging
import os
from copy import copy
from dataclasses import dataclass
from itertools import groupby
from typing import Iterable, Union

import numpy as np
import torch
import torch.utils.data
from lhotse import CutSet, fastcopy
from lhotse.cut import MixedCut, MultiCut
from lhotse.dataset import AudioSamples
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.asr.parts.utils.sot_speaker_alignment import (
    collate_speaker_activity_targets,
    ensure_single_speaker_sot,
    fix_speaker_activity,
    speaker_activity_from_cut,
)
from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import (
    AudioTurn,
    Formattable,
    TextTurn,
    collate_conversation_audio_fault_tolerant,
    collate_conversation_audio_packed_fault_tolerant,
)
from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts import Llama2PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id


class SALMDataset(torch.utils.data.Dataset):
    """
    A dataset for Speech-Augmented Language Models (SALM) that processes multimodal conversations
    containing both text and audio turns.

    This dataset handles NeMoMultimodalConversation objects which combine text messages
    and audio segments in a conversational format. It uses audio_locator_tag in the text,
    where each such placeholder corresponds to an entire audio segment.

    Args:
        tokenizer (AutoTokenizer):
            Tokenizer for converting text to token IDs and vice versa. Must have a special
            audio_locator_tag token that will be replaced with audio embeddings during model's
            training step.
        multispeaker_cfg (dict | None):
            Optional Serialized Output Training (SOT) speaker-activity settings.
            ``num_speakers`` is required when this mapping is provided.
            When provided, each batch additionally includes RTTM-derived
            ``spk_targets`` / ``spk_target_length``. Rows without an explicit
            RTTM path contain the reserved value ``-1`` so the perception encoder
            can replace them with inferred speaker activity.
        pack_audio (bool):
            Return valid waveform samples contiguously as `packed_audio_samples`
            plus `audio_cu_seqlens`, instead of materializing `audios[B, T_max]`.
            Defaults to `False` for complete batch-API compatibility.
        pack_sequences (bool):
            Return every variable-length sequence tensor without batch padding.
            Text IDs and masks are concatenated to shape ``[T_total]`` and
            described by ``text_cu_seqlens``; audio is returned in the same
            packed form as ``pack_audio=True``; optional speaker targets are
            concatenated to shape ``[T_spk_total, N_spk]`` and described by
            ``spk_target_cu_seqlens``. This option implies ``pack_audio=True``.
        batch_tokens (int | None):
            Token budget used by the Lhotse sampler. When provided, and the
            sampler attached an exact ``num_tokens`` measurement to every
            retained conversation, the batch contains a scalar
            ``packing_efficiency`` equal to the measured token sum divided by
            this budget.
        strict_audio_loading (bool):
            Re-raises audio collation errors and
            rejects conversations or audio items dropped or reordered by the
            fault-tolerant collator. Defaults to ``False`` so audio I/O and
            decoder failures remain fault tolerant. The datamodule controls
            this through ``fault_tolerant_audio_loading``.

            [ SOT Example for overlapping speakers ]
            Speaker-parallel transcription as a timeline:
                <spk:0>: Well, we should focus on the most important issues first.
                <spk:1>:                     Let me finish. John, let me finish.
            Serialized Output Training (SOT) transcription:
                <spk:0> Well, we should focus on <spk:1> Let me <spk:0> the most
                <spk:1> finish, John, <spk:0> important issues <spk:1> let me finish.
                <spk:0> first.

    Returns:
        A dictionary with the following keys:
            - audios: Tensor of audio waveform samples [B_audio, T_samples] (default mode)
            - packed_audio_samples: Tensor of contiguous waveform samples [T_total] (packed mode)
            - audio_cu_seqlens: Tensor of cumulative waveform offsets [B_audio + 1] (packed mode)
            - audio_lens: Tensor of audio lengths [B_audio]
            - input_ids: Tensor of text token IDs [B, T_tokens] (padded mode) or
                [T_total] (packed mode), including audio_locator_tag tokens
            - loss_mask: Boolean tensor with the same shape as input_ids indicating which
                tokens are part of the assistant's responses (True) and should be used for loss
            - text_cu_seqlens: Tensor of cumulative text offsets [B + 1] (packed mode)
            - packing_efficiency: Optional scalar measuring sampled tokens divided by
                ``batch_tokens``

    Notes:
        - Each audio_locator_tag token in input_ids corresponds to an audio segment in audios
        - The SALM model later replaces these audio_locator_tag tokens with encoded audio embeddings
        - The loss_mask identifies which tokens are part of the target sequences (assistant responses)
          and which are part of the source sequences (user prompts)
        - The input_ids and loss_mask will be expanded during model forward pass to account for
          the variable-length audio segments that replace each audio_locator_tag token
        - Serialized Output Training (SOT) speaker tags ``<spk:N>`` stay regular text tokens here;
          normalization and aliasing happen upstream. Auxiliary SOT mode (off by default) is opt-in via
          ``multispeaker_cfg`` and does not affect the default single-speaker behavior.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        multispeaker_cfg: dict | None = None,
        pack_audio: bool = False,
        pack_sequences: bool = False,
        batch_tokens: int | None = None,
        strict_audio_loading: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_id = get_pad_id(tokenizer)
        self.pack_sequences = bool(pack_sequences)
        self.pack_audio = bool(pack_audio) or self.pack_sequences
        self.batch_tokens = int(batch_tokens) if batch_tokens is not None else None
        self.strict_audio_loading = bool(strict_audio_loading)
        if self.batch_tokens is not None and self.batch_tokens <= 0:
            raise ValueError(f"batch_tokens must be positive, got {self.batch_tokens}")
        # Setting USE_AIS_GET_BATCH=true makes the loader issue a single AIStore GetBatch
        # call per minibatch, paired with URL-backed cuts produced by the multimodal
        # conversation adapters (NeMoMultimodalConversation{Jsonl,ShareGPTJsonl}Adapter).
        # USE_AIS_INDIVIDUAL_GETS=true (only meaningful when USE_AIS_GET_BATCH=true) forces
        # the underlying AISBatchLoader to skip MOSS GetBatch and issue one
        # ``Object.get_reader().read_all()`` per object — useful when the deployment
        # doesn't support GetBatch or its performance is degraded.
        self.load_audio = AudioSamples(
            fault_tolerant=True,
            use_batch_loader=os.environ.get("USE_AIS_GET_BATCH", "False").lower() == "true",
            ais_force_individual=os.environ.get("USE_AIS_INDIVIDUAL_GETS", "False").lower() == "true",
            mono_downmix=True,
        )
        self.multispeaker_cfg = MultiSpeakerConfig.from_dict(multispeaker_cfg)
        self.multispeaker_processor = (
            SALMMultiSpeakerProcessor(self.multispeaker_cfg, pack_targets=self.pack_sequences)
            if self.multispeaker_cfg is not None
            else None
        )

    def with_fault_tolerant_audio_loading(self, enabled: bool) -> "SALMDataset":
        """Return a per-loader view with the requested audio I/O failure policy.

        ``DataModule`` shares one dataset factory between train/validation/test,
        while each loader may have a different audio-loading policy. A shallow
        copy keeps the tokenizer and model-independent processors shared, and a
        copied ``AudioSamples`` instance avoids mutating another loader's
        strictness state.
        """
        enabled = bool(enabled)
        dataset = copy(self)
        dataset.strict_audio_loading = not enabled
        dataset.load_audio = copy(self.load_audio)
        dataset.load_audio.fault_tolerant = enabled
        if self.load_audio.ais_batch_loader is not None:
            dataset.load_audio.ais_batch_loader = copy(self.load_audio.ais_batch_loader)
            dataset.load_audio.ais_batch_loader.skip_failed_fetches = enabled
        return dataset

    def __getitem__(self, conversations: CutSet) -> dict | None:
        # The collator retains its fault-tolerant 3-tuple API, but strict mode
        # verifies exact conversation/audio identity and raises on any drop.
        # DataModule gates FallbackDataset behind the same audio-loading policy.
        if self.strict_audio_loading:
            requested_conversation_ids = tuple(id(conversation) for conversation in conversations)
            requested_audio_cut_ids = _audio_cut_ids(conversations)
        else:
            requested_conversation_ids = requested_audio_cut_ids = ()

        try:
            if self.pack_audio:
                packed_audio_samples, audio_cu_seqlens, audio_lens, conversations = (
                    collate_conversation_audio_packed_fault_tolerant(conversations, self.load_audio)
                )
                audio_inputs = {
                    "packed_audio_samples": packed_audio_samples,
                    "audio_cu_seqlens": audio_cu_seqlens,
                }
            else:
                audios, audio_lens, conversations = collate_conversation_audio_fault_tolerant(
                    conversations, self.load_audio
                )
                audio_inputs = {"audios": audios}
        except Exception as e:
            if self.strict_audio_loading:
                raise
            logging.warning(f"Error collating conversations: {e}")
            return None
        if self.strict_audio_loading:
            materialized_conversation_ids = tuple(id(conversation) for conversation in conversations)
            if materialized_conversation_ids != requested_conversation_ids:
                raise RuntimeError(
                    "Strict SALM validation dropped or reordered conversations: "
                    f"requested={len(requested_conversation_ids)} "
                    f"materialized={len(materialized_conversation_ids)}"
                )
            materialized_audio_cut_ids = _audio_cut_ids(conversations)
            if (
                materialized_audio_cut_ids != requested_audio_cut_ids
                or len(audio_lens) != len(requested_audio_cut_ids)
            ):
                raise RuntimeError(
                    "Strict SALM validation dropped or reordered audio items: "
                    f"requested={len(requested_audio_cut_ids)} "
                    f"materialized={len(audio_lens)}"
                )
        if not conversations:
            if self.strict_audio_loading:
                raise RuntimeError("Strict SALM loading produced an empty conversation batch.")
            return None
        input_ids = [c.input_ids for c in conversations]
        loss_masks = [getattr(c, "mask", torch.empty(0)) for c in conversations]
        if self.pack_sequences:
            packed_input_ids, text_cu_seqlens = pack_vectors(input_ids)
            packed_loss_mask, loss_mask_cu_seqlens = pack_vectors(loss_masks)
            if not torch.equal(text_cu_seqlens, loss_mask_cu_seqlens):
                raise ValueError(
                    "Each SALM loss mask must have the same length as its input IDs; "
                    f"got offsets {text_cu_seqlens.tolist()} and {loss_mask_cu_seqlens.tolist()}."
                )
            text_inputs = {
                "input_ids": packed_input_ids,
                "loss_mask": packed_loss_mask.to(torch.bool),
                "text_cu_seqlens": text_cu_seqlens,
            }
        else:
            text_inputs = {
                "input_ids": left_collate_vectors(input_ids, padding_value=self.pad_id),
                "loss_mask": left_collate_vectors(loss_masks, padding_value=0).to(torch.bool),
            }

        batch = {
            **audio_inputs,
            **text_inputs,
            "audio_lens": audio_lens,
            # Keep decoded in-memory audio available until auxiliary targets
            # are materialized. Native ShareGPT WDS cuts intentionally use
            # memory-backed recordings; dropping them first replaces their
            # sources with unresolved Shar placeholders, and multichannel
            # downmixing in SALMMultiSpeakerProcessor then cannot load audio.
            "conversations": conversations,
        }
        if self.batch_tokens is not None:
            sampled_lengths = [getattr(conversation, "num_tokens", None) for conversation in conversations]
            if all(length is not None for length in sampled_lengths):
                batch["packing_efficiency"] = torch.tensor(
                    sum(sampled_lengths) / self.batch_tokens,
                    dtype=torch.float32,
                )
        if self.multispeaker_processor is not None:
            self.multispeaker_processor(batch)
        batch["conversations"] = drop_in_memory_data(conversations)
        return batch


def left_collate_vectors(
    tensors: Iterable[Union[torch.Tensor, np.ndarray]],
    padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
) -> torch.Tensor:
    tensors = [torch.as_tensor(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="left")


def pack_vectors(tensors: Iterable[Union[torch.Tensor, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate 1-D rows and return their cumulative offsets without padding."""
    tensors = [torch.as_tensor(t) for t in tensors]
    if not tensors:
        raise ValueError("Cannot pack an empty sequence collection.")
    if not all(t.ndim == 1 for t in tensors):
        raise ValueError(f"Expected only 1-D input tensors, got shapes {[tuple(t.shape) for t in tensors]}.")
    values = torch.cat(tensors, dim=0)
    lengths = torch.as_tensor([t.shape[0] for t in tensors], dtype=torch.long, device=values.device)
    cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    return values, cu_seqlens


def drop_in_memory_data(conversations: CutSet) -> CutSet:
    def _drop(conversation: Formattable) -> Formattable:
        if not isinstance(conversation, NeMoMultimodalConversation):
            return conversation
        turns = []
        for t in conversation.turns:
            if isinstance(t, AudioTurn):
                t = fastcopy(t, cut=t.cut.drop_in_memory_data())
            turns.append(t)
        return fastcopy(conversation, turns=turns)

    return conversations.map(_drop, apply_fn=None)


def _audio_cut_ids(conversations: Iterable[Formattable]) -> tuple[str, ...]:
    cut_ids = []
    for conversation in conversations:
        if isinstance(conversation, NeMoMultimodalConversation):
            cut_ids.extend(cut.id for cut in conversation.list_cuts())
        elif not isinstance(conversation, Formattable):
            raise TypeError(
                "SALMDataset expected a prompt-formatted example, "
                f"got {type(conversation).__name__}."
            )
    return tuple(cut_ids)


@registered_prompt_format_fn(NeMoMultimodalConversation, Llama2PromptFormatter)
def default_multimodal_conversation_prompt_format_fn(
    example: NeMoMultimodalConversation, prompt: Llama2PromptFormatter, **prompt_kwargs
):
    # Collapse consecutive same-role turns into single turn for proper prompt formatting.
    turns = groupby(
        [
            {
                "role": turn.role,
                "slots": {"message": turn.value if isinstance(turn, TextTurn) else turn.audio_locator_tag},
            }
            for turn in example.turns
        ],
        key=lambda turn: turn["role"],
    )
    turns = [
        {"role": role, "slots": {"message": " ".join(t["slots"]["message"] for t in turn_grp)}}
        for role, turn_grp in turns
    ]
    if hasattr(example, "system_prompt"):
        turns[0]["role"] = "system_and_user"
        turns[0]["slots"]["system"] = example.system_prompt
    return prompt.encode_dialog(turns, **prompt_kwargs)


@dataclass(frozen=True)
class MultiSpeakerConfig:
    """Configuration for auxiliary multi-speaker SOT targets."""

    num_speakers: int
    num_sample_per_mel_frame: int = 160
    num_mel_frame_per_target_frame: int = 8
    max_alignment_permutations: int | None = 720

    @staticmethod
    def from_dict(cfg: dict | None) -> "MultiSpeakerConfig | None":
        """Build a config from a raw settings dict, or return ``None`` when no SOT settings are given."""
        if cfg is None:
            return None
        if 'num_speakers' not in cfg:
            raise ValueError(
                "multispeaker_cfg.num_speakers must be set explicitly; there is no implicit 4-speaker cap."
            )
        num_speakers = int(cfg['num_speakers'])
        if num_speakers <= 0:
            raise ValueError(f"multispeaker_cfg.num_speakers must be positive, got {num_speakers}.")
        max_alignment_permutations = cfg.get('max_alignment_permutations', 720)
        return MultiSpeakerConfig(
            num_speakers=num_speakers,
            num_sample_per_mel_frame=int(cfg.get('window_stride', 0.01) * cfg.get('sample_rate', 16000)),
            num_mel_frame_per_target_frame=int(cfg.get('subsampling_factor', 8)),
            max_alignment_permutations=(
                None if max_alignment_permutations is None else int(max_alignment_permutations)
            ),
        )


class SALMMultiSpeakerProcessor:
    """Add SOT activity targets, using ``-1`` rows to request inferred diarization."""

    def __init__(self, cfg: MultiSpeakerConfig, *, pack_targets: bool = False) -> None:
        self.cfg = cfg
        self.pack_targets = bool(pack_targets)

    def __call__(self, batch: dict) -> None:
        """Attach RTTM targets or missing-RTTM sentinels to ``batch`` in place."""
        cfg = self.cfg
        speaker_activities = self._build_speaker_activities(batch["conversations"])
        if not speaker_activities:
            return
        # The shared collator pads variable-length rows with zeros. Preserve
        # the all--1 sentinel across that padding: PEE detects missing-RTTM
        # rows over the whole padded tensor, so a single padded zero would
        # otherwise make a short no-RTTM row look like an explicit
        # (all-silent) RTTM target and bypass its embedded Sortformer.
        missing_rttm_rows = torch.tensor(
            [bool(torch.all(activity == -1.0)) for activity in speaker_activities],
            dtype=torch.bool,
        )
        dtype = (batch["audios"] if "audios" in batch else batch["packed_audio_samples"]).dtype
        if self.pack_targets:
            normalized = []
            for activity, missing_rttm in zip(speaker_activities, missing_rttm_rows):
                n_spk = activity.shape[1]
                if n_spk > cfg.num_speakers:
                    activity = activity[:, : cfg.num_speakers]
                elif n_spk < cfg.num_speakers:
                    activity = torch.nn.functional.pad(
                        activity,
                        (0, cfg.num_speakers - n_spk),
                        mode="constant",
                        value=0.0,
                    )
                activity = activity.to(dtype=dtype)
                if missing_rttm:
                    activity = torch.full_like(activity, -1.0)
                normalized.append(activity)
            target_length = torch.as_tensor([target.shape[0] for target in normalized], dtype=torch.long)
            targets = torch.cat(normalized, dim=0)
            target_cu_seqlens = torch.cat([target_length.new_zeros(1), target_length.cumsum(0)])
            batch["spk_targets"] = targets
            batch["spk_target_length"] = target_length
            batch["spk_target_cu_seqlens"] = target_cu_seqlens
        else:
            targets, target_length = collate_speaker_activity_targets(
                speaker_activities,
                batch["audio_lens"],
                num_speakers=cfg.num_speakers,
                num_sample_per_mel_frame=cfg.num_sample_per_mel_frame,
                num_mel_frame_per_target_frame=cfg.num_mel_frame_per_target_frame,
                dtype=dtype,
            )
            targets[missing_rttm_rows] = -1.0
            batch["spk_targets"] = targets
            batch["spk_target_length"] = target_length

    def _build_speaker_activities(self, conversations: CutSet) -> list[torch.Tensor]:
        cfg = self.cfg
        speaker_activities = []
        for conversation in conversations:
            # Generic prompt-formatted text-only examples intentionally do not
            # expose multimodal turns and have no speaker targets to materialize.
            for turn in getattr(conversation, "turns", ()):
                if not isinstance(turn, AudioTurn):
                    continue

                has_rttm = self._has_rttm_filepath(turn.cut)
                cut = self._prepare_audio_turn_cut(turn)
                if not has_rttm:
                    speaker_activity = speaker_activity_from_cut(
                        cut,
                        num_speakers=cfg.num_speakers,
                        num_sample_per_mel_frame=cfg.num_sample_per_mel_frame,
                        num_mel_frame_per_target_frame=cfg.num_mel_frame_per_target_frame,
                    )
                    # Request inferred activity instead of the synthetic
                    # single-speaker fallback.  Skip SOT/RTTM column alignment:
                    # the sentinel replaces the whole target, and factorial
                    # alignment work here would be both wasted and misleading.
                    speaker_activity = torch.full_like(speaker_activity, -1.0)
                else:
                    text = self._audio_turn_text(turn, cut)
                    new_text, _, _ = ensure_single_speaker_sot(text)
                    speaker_activity, is_permutation_resolved = speaker_activity_from_cut(
                        cut,
                        num_speakers=cfg.num_speakers,
                        num_sample_per_mel_frame=cfg.num_sample_per_mel_frame,
                        num_mel_frame_per_target_frame=cfg.num_mel_frame_per_target_frame,
                        text=new_text,
                        return_permutation_resolved=True,
                    )
                    if not is_permutation_resolved:
                        speaker_activity = fix_speaker_activity(
                            new_text,
                            speaker_activity,
                            cfg.num_speakers,
                            max_alignment_permutations=cfg.max_alignment_permutations,
                        )
                speaker_activities.append(speaker_activity)
        return speaker_activities

    @staticmethod
    def _has_rttm_filepath(cut) -> bool:
        """Return whether a cut or any constituent mixed track has an explicit RTTM file."""
        custom = getattr(cut, "custom", None) or {}
        if custom.get("rttm_filepath", None):
            return True
        if isinstance(cut, MixedCut):
            return any(SALMMultiSpeakerProcessor._has_rttm_filepath(track.cut) for track in cut.tracks)
        return False

    @staticmethod
    def _prepare_audio_turn_cut(turn: AudioTurn):
        cut = turn.cut
        if isinstance(cut, MultiCut):
            cut = cut.to_mono(mono_downmix=True)
        elif isinstance(cut, MixedCut):
            pass
        elif cut.num_channels is not None and cut.num_channels > 1:
            logging.warning(
                "Multiple channels detected in cut '%s' (%d channels). "
                "Only the first channel will be used for speaker targets; remaining channels are ignored.",
                cut.id,
                cut.num_channels,
            )
            cut = cut.with_channels(channels=[0])

        if getattr(cut, "custom", None) is None:
            cut = fastcopy(cut, custom={})

        return cut

    @staticmethod
    def _audio_turn_text(turn: AudioTurn, cut) -> str:
        text = turn.text or getattr(cut, "text", None)
        if text:
            return text
        return " ".join(s.text for s in getattr(cut, "supervisions", []) if s.text)
