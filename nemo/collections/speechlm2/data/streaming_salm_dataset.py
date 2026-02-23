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
"""Dataset for StreamingSALM training."""

import logging

import torch
import torch.utils.data
from lhotse import CutSet

from nemo.collections.common.data.lhotse.text_adapters import (
    TextTurn,
    collate_conversation_audio_fault_tolerant,
)
from nemo.collections.speechlm2.data.salm_dataset import drop_in_memory_data


class StreamingSALMDataset(torch.utils.data.Dataset):
    """
    Dataset for StreamingSALM. Returns raw audio and transcript text.
    All sequence construction (Mimi encoding, forced alignment, interleaving)
    happens in the model's prepare_inputs method.
    """

    EXPECTED_SAMPLE_RATE = 24000  # Mimi native sample rate

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getitem__(self, conversations: CutSet) -> dict | None:
        try:
            audios, audio_lens, conversations = collate_conversation_audio_fault_tolerant(conversations)
        except Exception as e:
            logging.warning(f"Error collating conversations: {e}")
            return None
        if not conversations:
            return None

        transcripts = extract_transcripts(conversations)
        return {
            "audios": audios,
            "audio_lens": audio_lens,
            "transcripts": transcripts,
            "conversations": drop_in_memory_data(conversations),
            "sample_rate": self.EXPECTED_SAMPLE_RATE,
        }


def extract_transcripts(conversations: CutSet) -> list[str]:
    """Extract the assistant/target text from each conversation."""
    transcripts = []
    for conv in conversations:
        text_parts = []
        for turn in conv.turns:
            if isinstance(turn, TextTurn) and turn.role == "assistant":
                text_parts.append(turn.value)
        transcripts.append(" ".join(text_parts))
    return transcripts
