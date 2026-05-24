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

"""
SALMDataset with support for two pretraining patterns (see Figure 2).
The relationship between audio and text is encoded purely by position:

  Repetition pattern:
      … <audio_loc> <transcript_n> …
      loss_mask=True on transcript tokens only.
      The model learns ASR: given Aₙ predict Tₙ.

  Continuation pattern:
      <prompt_text> <audio_loc_1> <text_2> <audio_loc_2> <text_3> …
      loss_mask=True on assistant text tokens only.
      The model learns cross-modal continuation: given Aₙ predict Tₙ₊₁.

Manifest entry schema (continuation / mixed):
─────────────────────────────────────────────
{
  "id": "<sample-id>",
  "conversations": [
    {"from": "user",      "value": "<prompt>",    "type": "text"},
    {"from": "user",      "value": "/utt1.wav",   "duration": 12.34, "type": "audio"},
    {"from": "assistant", "value": "<text2>",     "type": "text"},
    {"from": "user",      "value": "/utt2.wav",   "duration": 13.01, "type": "audio"},
    {"from": "assistant", "value": "<text3>",     "type": "text"}
  ]
}
"""

import logging
from itertools import groupby
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data
from lhotse import CutSet, fastcopy
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _encode_text(tokenizer: AutoTokenizer, text: str) -> torch.Tensor:
    """Encode *text* and return a 1-D LongTensor of token ids (no BOS/EOS)."""
    ids = tokenizer.text_to_ids(text)
    return torch.tensor(ids, dtype=torch.long)


def _token_id(tokenizer: AutoTokenizer, token: str) -> int:
    """Return the single integer id for a special token string."""
    ids = tokenizer.text_to_ids(token)
    if len(ids) != 1:
        raise ValueError(
            f"Special token '{token}' should map to exactly one id, got {ids}. "
            "Make sure it is added to the tokenizer vocabulary."
        )
    return ids[0]


def _cat(parts: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate a list of 1-D tensors; return empty LongTensor when empty."""
    if not parts:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(parts, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Core dataset
# ──────────────────────────────────────────────────────────────────────────────

class SALMDataset(torch.utils.data.Dataset):
    """
    Dataset for Speech-Augmented Language Models (SALM).

    Supports two pretraining patterns
    The audio–text relationship is encoded purely by sequence position.

    Pattern "repetition":
        <audio_loc> <transcript>   (loss on transcript only)

    Pattern "continuation":
        <prompt> <audio_loc_1> <text_2> <audio_loc_2> <text_3> …
        (loss on assistant text turns only)

    Pattern "mixed":
        Per-sample "pattern" key in the manifest selects the builder;
        falls back to "continuation" when the key is absent.

    Args:
        tokenizer:         NeMo tokenizer. Must contain ``audio_locator_tag``
                           as a single registered special token.
        audio_locator_tag: Placeholder string for audio turns.
        pattern:           Default pattern: "continuation" | "repetition" | "mixed".
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        audio_locator_tag: str = "<|audioplaceholder|>",
        pattern: str = "continuation",
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_id = get_pad_id(tokenizer)
        self.audio_locator_tag = audio_locator_tag
        self.pattern = pattern

        # Only the audio-locator placeholder must be a single registered token.
        self._audio_loc_id = _token_id(tokenizer, audio_locator_tag)

        # EOS must be present and supervised at the end of each target sequence;
        # otherwise autoregressive generation has no learned stop signal.
        self._eos_id = getattr(tokenizer, "eos_id", None)
        if self._eos_id is None:
            self._eos_id = getattr(tokenizer, "eos_token_id", None)
        if self._eos_id is None:
            raise ValueError("SALMDataset: tokenizer has neither eos_id nor eos_token_id")
    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def __getitem__(self, conversations: CutSet) -> Optional[dict]:
        """
        Process a mini-batch of NeMoMultimodalConversation cuts.

        Returns a dict with keys:
            audios        – FloatTensor [B_audio, T_samples]
            audio_lens    – LongTensor  [B_audio]
            input_ids     – LongTensor  [B, T_tokens]  (left-padded)
            loss_mask     – BoolTensor  [B, T_tokens]  (True = compute loss)
            conversations – CutSet (in-memory data dropped)
        """
        try:
            audios, audio_lens, conversations = collate_conversation_audio_fault_tolerant(
                conversations
            )
        except Exception as exc:
            logging.warning(f"Error collating conversations: {exc}")
            return None

        if not conversations:
            return None

        all_input_ids: List[torch.Tensor] = []
        all_loss_masks: List[torch.Tensor] = []

        for conv in conversations:
            sample_pattern = getattr(conv, "pattern", self.pattern)
            if sample_pattern == "repetition":
                ids, mask = self._build_repetition(conv)
            else:
                ids, mask = self._build_continuation(conv)

            all_input_ids.append(ids)
            all_loss_masks.append(mask)

        return {
            "audios": audios,
            "audio_lens": audio_lens,
            "input_ids": left_collate_vectors(all_input_ids, padding_value=self.pad_id),
            "loss_mask": left_collate_vectors(all_loss_masks, padding_value=0).to(torch.bool),
            "conversations": drop_in_memory_data(conversations),
        }

    # ------------------------------------------------------------------ #
    # Pattern builders                                                     #
    # ------------------------------------------------------------------ #

    def _build_repetition(
        self, conv: NeMoMultimodalConversation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Repetition pattern – no signal token, purely positional:

            [context …]  <audio_loc>  <transcript_n>

        loss_mask = True only over <transcript_n> tokens.
        """
        ids_parts: List[torch.Tensor] = []
        mask_parts: List[torch.Tensor] = []

        turns = list(conv.turns)
        i = 0
        while i < len(turns):
            turn = turns[i]

            if isinstance(turn, AudioTurn):
                # audio placeholder – no loss
                ids_parts.append(torch.tensor([self._audio_loc_id], dtype=torch.long))
                mask_parts.append(torch.zeros(1, dtype=torch.long))
                # immediately following TextTurn is the transcript target
                if i + 1 < len(turns) and isinstance(turns[i + 1], TextTurn):
                    i += 1
                    transcript_ids = _encode_text(self.tokenizer, turns[i].value)
                    ids_parts.append(transcript_ids)
                    mask_parts.append(torch.ones(len(transcript_ids), dtype=torch.long))

            elif isinstance(turn, TextTurn):
                # context / prompt – no loss
                text_ids = _encode_text(self.tokenizer, turn.value)
                ids_parts.append(text_ids)
                mask_parts.append(torch.zeros(len(text_ids), dtype=torch.long))
            i += 1
          
        ids_parts.append(torch.tensor([self._eos_id], dtype=torch.long))
        mask_parts.append(torch.ones(1, dtype=torch.long))

        return _cat(ids_parts), _cat(mask_parts)

    def _build_continuation(
        self, conv: NeMoMultimodalConversation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Continuation pattern – no signal token, purely positional:

            <prompt_text>  <audio_loc_1>  <text_2>
                           <audio_loc_2>  <text_3>  …

        Turn roles:
            user  TextTurn  → context,        loss_mask = False
            AudioTurn       → <audio_loc>,    loss_mask = False
            assistant TextTurn → target text, loss_mask = True
        """
        ids_parts: List[torch.Tensor] = []
        mask_parts: List[torch.Tensor] = []

        turns = list(conv.turns)
        i = 0
        while i < len(turns):
            turn = turns[i]

            if isinstance(turn, AudioTurn):
                # audio placeholder – no loss
                ids_parts.append(torch.tensor([self._audio_loc_id], dtype=torch.long))
                mask_parts.append(torch.zeros(1, dtype=torch.long))

                # immediately following assistant TextTurn is the prediction target
                if (
                    i + 1 < len(turns)
                    and isinstance(turns[i + 1], TextTurn)
                    and getattr(turns[i + 1], "role", "assistant") == "assistant"
                ):
                    i += 1
                    target_ids = _encode_text(self.tokenizer, turns[i].value)
                    ids_parts.append(target_ids)
                    mask_parts.append(torch.ones(len(target_ids), dtype=torch.long))

            elif isinstance(turn, TextTurn):
                # user prompt / context – no loss
                text_ids = _encode_text(self.tokenizer, turn.value)
                ids_parts.append(text_ids)
                mask_parts.append(torch.zeros(len(text_ids), dtype=torch.long))
            i += 1
        ids_parts.append(torch.tensor([self._eos_id], dtype=torch.long))
        mask_parts.append(torch.ones(1, dtype=torch.long))
        return _cat(ids_parts), _cat(mask_parts)


# ──────────────────────────────────────────────────────────────────────────────
# Collation / utility functions  (public API unchanged)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Prompt format fn
# ──────────────────────────────────────────────────────────────────────────────

@registered_prompt_format_fn(NeMoMultimodalConversation, Llama2PromptFormatter)
def default_multimodal_conversation_prompt_format_fn(
    example: NeMoMultimodalConversation,
    prompt: Llama2PromptFormatter,
):
    """Build dialog turns for the prompt formatter (unchanged semantics)."""
    raw_turns = [
        {
            "role": turn.role,
            "slots": {
                "message": (
                    turn.value if isinstance(turn, TextTurn) else turn.audio_locator_tag
                )
            },
        }
        for turn in example.turns
    ]

    collapsed = [
        {
            "role": role,
            "slots": {"message": " ".join(t["slots"]["message"] for t in grp)},
        }
        for role, grp in groupby(raw_turns, key=lambda t: t["role"])
    ]

    if hasattr(example, "system_prompt"):
        collapsed[0]["role"] = "system_and_user"
        collapsed[0]["slots"]["system"] = example.system_prompt

    return prompt.encode_dialog(collapsed)
