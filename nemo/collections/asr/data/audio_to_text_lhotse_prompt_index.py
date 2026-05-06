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
Simplified Lhotse dataset that returns language ID indices instead of full prompt tensors.
The model creates the prompt tensor using the actual encoded length.
"""

import random
from typing import Dict, Optional, Tuple

import torch
import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


class LhotseSpeechToTextBpeDatasetWithPromptIndex(torch.utils.data.Dataset):
    """
    Simplified dataset class for speech-to-text with prompt support.
    
    Instead of computing full prompt tensors, this dataset returns just the
    language ID index per sample. The model creates the prompt tensor using
    the actual encoder output length, guaranteeing no size mismatch.
    
    Returns:
        audio_signal: Audio waveform [B, T]
        audio_signal_length: Audio lengths [B]
        transcripts: Token IDs [B, T]
        transcript_length: Token lengths [B]
        prompt_indices: Language ID indices [B] (NOT full tensors)
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'audio_signal_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'prompt_indices': NeuralType(tuple('B'), LabelsType()),  # Just indices, not full tensors
        }

    def __init__(self, tokenizer, cfg):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg

        # Load prompt dictionary from config
        self.prompt_dict = cfg.get('prompt_dictionary')
        if not self.prompt_dict:
            raise ValueError("prompt_dictionary is required in config")
        
        self.num_prompts = cfg.get('num_prompts', 128)

        # Field to use for prompt key (default to 'target_lang')
        self.prompt_field = cfg.get('prompt_field', 'target_lang')

        # Training mode flag: when True, randomly use auto (101) 50% of the time
        self.training_mode = cfg.get('training_mode', True)

        logging.info(f"LhotseSpeechToTextBpeDatasetWithPromptIndex: Returns indices only, model creates prompt tensor")

    def _get_prompt_index(self, prompt_key: str) -> int:
        """Maps prompt keys to indices using the prompt dictionary."""
        if prompt_key not in self.prompt_dict:
            available_keys = list(self.prompt_dict.keys())
            raise ValueError(
                f"Unknown prompt key: '{prompt_key}'. Available: {available_keys[:10]}{'...' if len(available_keys) > 10 else ''}"
            )
        return self.prompt_dict[prompt_key]

    def _get_prompt_index_for_cut(self, cut) -> int:
        """
        Get prompt index for a cut, with training mode randomization.
        During training: 50% chance to use auto (101), 50% actual language ID
        During inference: always use the actual language ID
        """
        if self.training_mode and random.random() < 0.5:
            return 101  # Auto/language-agnostic
        else:
            return self._get_prompt_index(cut.supervisions[0].language)

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)
        tokens = [torch.as_tensor(self.tokenizer(c.supervisions[0].text, c.supervisions[0].language)) for c in cuts]

        # Get prompt indices (just the language ID per sample, NOT full tensors)
        prompt_indices = torch.tensor(
            [self._get_prompt_index_for_cut(c) for c in cuts],
            dtype=torch.long
        )

        # Create final tensors
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)

        return (
            audio,          # Audio signal [B, T]
            audio_lens,     # Audio lengths [B]
            tokens,         # Text tokens [B, T]
            token_lens,     # Token lengths [B]
            prompt_indices, # Language ID indices [B] - model creates full tensor
        )


class TokenizerWrapper:
    """Provide a unified interface for NeMo Tokenizer, AggregateTokenizer, and (char) Parser."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        if isinstance(tokenizer, AggregateTokenizer):
            self._impl = self._call_agg_tokenizer
        elif isinstance(tokenizer, TokenizerSpec):
            self._impl = self._call_tokenizer
        else:
            self._impl = self._call_parser

    def __call__(self, text: str, lang: Optional[str] = None):
        return self._impl(text, lang)

    def _call_agg_tokenizer(self, text: str, lang: Optional[str] = None):
        assert lang is not None, "Expected 'lang' to be set for AggregateTokenizer."
        return self._tokenizer.text_to_ids(text, lang)

    def _call_tokenizer(self, text: str, lang: Optional[str] = None):
        return self._tokenizer.text_to_ids(text)

    def _call_parser(self, text: str, lang: Optional[str] = None):
        return self._tokenizer(text)
