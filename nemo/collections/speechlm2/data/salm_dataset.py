# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
    TextTurn,
    collate_conversation_audio_fault_tolerant,
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
            When provided, each batch additionally includes RTTM-derived
            ``spk_targets`` / ``spk_target_length``.

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
            - audios: Tensor of audio waveform samples [B_audio, T_samples]
            - audio_lens: Tensor of audio lengths [B_audio]
            - input_ids: Tensor of text token IDs [B, T_tokens], including audio_locator_tag tokens
            - loss_mask: Boolean tensor [B, T_tokens] indicating which tokens are part of the
                assistant's responses (True) and should be used for computing loss

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

    def __init__(self, tokenizer: AutoTokenizer, multispeaker_cfg: dict | None = None) -> None:
        self.tokenizer = tokenizer
        self.pad_id = get_pad_id(tokenizer)
        # Setting USE_AIS_GET_BATCH=true makes the loader issue a single AIStore GetBatch
        # call per minibatch, paired with URL-backed cuts produced by the multimodal
        # conversation adapters (NeMoMultimodalConversation{Jsonl,ShareGPTJsonl}Adapter).
        self.load_audio = AudioSamples(
            fault_tolerant=True,
            use_batch_loader=os.environ.get("USE_AIS_GET_BATCH", "False").lower() == "true",
            mono_downmix=True,
        )
        self.multispeaker_cfg = MultiSpeakerConfig.from_dict(multispeaker_cfg)
        self.multispeaker_processor = (
            SALMMultiSpeakerProcessor(self.multispeaker_cfg) if self.multispeaker_cfg is not None else None
        )

    def __getitem__(self, conversations: CutSet) -> dict | None:
        # Note: the function call below may filter out some or all conversations due to audio loading issues.
        # If all conversations are filtered out, we'll return None, and expect users to wrap this dataset
        # in ``nemo.collections.common.data.fallback.FallbackDataset`` to use the previous mini-batch instead.
        try:
            audios, audio_lens, conversations = collate_conversation_audio_fault_tolerant(
                conversations, self.load_audio
            )
        except Exception as e:
            logging.warning(f"Error collating conversations: {e}")
            return None
        if not conversations:
            return None
        batch = {
            "audios": audios,
            "audio_lens": audio_lens,
            "input_ids": left_collate_vectors([c.input_ids for c in conversations], padding_value=self.pad_id),
            "loss_mask": left_collate_vectors(
                [getattr(c, "mask", torch.empty(0)) for c in conversations], padding_value=0
            ).to(torch.bool),
            "conversations": drop_in_memory_data(conversations),
        }
        if self.multispeaker_processor is not None:
            self.multispeaker_processor(batch)
        return batch


def left_collate_vectors(
    tensors: Iterable[Union[torch.Tensor, np.ndarray]],
    padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
) -> torch.Tensor:
    tensors = [torch.as_tensor(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="left")


def drop_in_memory_data(conversations: CutSet) -> CutSet:
    def _drop(conversation: NeMoMultimodalConversation) -> NeMoMultimodalConversation:
        turns = []
        for t in conversation.turns:
            if isinstance(t, AudioTurn):
                t = fastcopy(t, cut=t.cut.drop_in_memory_data())
            turns.append(t)
        return fastcopy(conversation, turns=turns)

    return conversations.map(_drop, apply_fn=None)


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

    num_speakers: int = 4
    no_rttm_to_ones: bool = True
    num_sample_per_mel_frame: int = 160
    num_mel_frame_per_target_frame: int = 8

    @staticmethod
    def from_dict(cfg: dict | None) -> "MultiSpeakerConfig | None":
        """Build a config from a raw settings dict, or return ``None`` when no SOT settings are given."""
        if cfg is None:
            return None
        return MultiSpeakerConfig(
            num_speakers=int(cfg.get('num_speakers', 4)),
            no_rttm_to_ones=cfg.get('no_rttm_to_ones', True),
            num_sample_per_mel_frame=int(cfg.get('window_stride', 0.01) * cfg.get('sample_rate', 16000)),
            num_mel_frame_per_target_frame=int(cfg.get('subsampling_factor', 8)),
        )


class SALMMultiSpeakerProcessor:
    """Adds auxiliary SOT speaker-activity targets to an otherwise prepared SALM batch."""

    def __init__(self, cfg: MultiSpeakerConfig) -> None:
        self.cfg = cfg

    def __call__(self, batch: dict) -> None:
        """Attach RTTM-derived ``spk_targets`` / ``spk_target_length`` to ``batch`` in place."""
        cfg = self.cfg
        speaker_activities = self._build_speaker_activities(batch["conversations"])
        if not speaker_activities:
            return
        targets, target_length = collate_speaker_activity_targets(
            speaker_activities,
            batch["audio_lens"],
            num_speakers=cfg.num_speakers,
            num_sample_per_mel_frame=cfg.num_sample_per_mel_frame,
            num_mel_frame_per_target_frame=cfg.num_mel_frame_per_target_frame,
            dtype=batch["audios"].dtype,
        )
        batch["spk_targets"] = targets
        batch["spk_target_length"] = target_length

    def _build_speaker_activities(self, conversations: CutSet) -> list[torch.Tensor]:
        cfg = self.cfg
        speaker_activities = []
        for conversation in conversations:
            for turn in conversation.turns:
                if not isinstance(turn, AudioTurn):
                    continue

                cut = self._prepare_audio_turn_cut(turn)
                speaker_activity = speaker_activity_from_cut(
                    cut,
                    num_speakers=cfg.num_speakers,
                    num_sample_per_mel_frame=cfg.num_sample_per_mel_frame,
                    num_mel_frame_per_target_frame=cfg.num_mel_frame_per_target_frame,
                    no_rttm_to_ones=cfg.no_rttm_to_ones,
                )

                text = self._audio_turn_text(turn, cut)
                new_text, _, _ = ensure_single_speaker_sot(text)

                speaker_activity = fix_speaker_activity(new_text, speaker_activity, cfg.num_speakers)
                speaker_activities.append(speaker_activity)
        return speaker_activities

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
