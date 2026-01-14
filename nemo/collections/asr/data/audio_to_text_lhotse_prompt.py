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
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_matrices, collate_vectors

from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseSpeechToTextBpeDatasetWithPrompt(torch.utils.data.Dataset):
    """
    Dataset class for speech-to-text with prompt vectors.
    Supports both language ID and custom prompts.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'audio_signal_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'prompt': NeuralType(('B', 'T', 'D'), LabelsType()),
        }

    def __init__(self, tokenizer, cfg):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg

        # Calculate num_sample_per_mel_frame from config
        sample_rate = cfg.get('sample_rate', 16000)
        window_stride = cfg.get('window_stride', 0.01)
        self.num_sample_per_mel_frame = int(sample_rate * window_stride)

        self.subsampling_factor = cfg.get('subsampling_factor', 8)

        # Load prompt dictionary from config if provided
        self.prompt_dict = cfg.get('prompt_dictionary')
        if self.prompt_dict:
            # Set num_prompts based on the length of prompt_dictionary or a minimum value
            # This ensures we have enough dimensions in our embedding space to add scale up without changing the model
            self.num_prompts = cfg.get('num_prompts', 128)

        # Field to use for prompt key (default to 'language')
        self.prompt_field = cfg.get('prompt_field', 'language')

    def _get_prompt_index(self, prompt_key: str) -> int:
        """
        Maps prompt keys to indices using the prompt dictionary.
        """
        if not self.prompt_dict:
            raise ValueError("Prompt dictionary is empty. Please provide a valid prompt_dictionary in the config.")

        if prompt_key not in self.prompt_dict:
            available_keys = list(self.prompt_dict.keys())
            raise ValueError(
                f"Unknown prompt key: '{prompt_key}'. Available prompts: {available_keys[:10]}{'...' if len(available_keys) > 10 else ''}"
            )

        return self.prompt_dict[prompt_key]

    def prompt_to_target(self, cut, num_prompts: int, window_stride: int, subsampling_factor: int):
        """
        Create prompt target tensor for the sequence.
        """
        # Calculate encoder output length based on subsampling factor
        encoder_hidden_len = self.get_hidden_length_from_sample_length(cut.num_samples)

        # Initialize prompt target matrix
        mask = np.zeros((num_prompts, encoder_hidden_len))

        # Get prompt index - default to language if prompt not specified
        # revise supervisions to include prompt key
        # prompt_key = getattr(cut.supervisions[0].custom_fields, cut.supervisions[0].language)cut.supervisions[0].custom_fields,
        prompt_id = self._get_prompt_index(cut.supervisions[0].language)

        # Set the corresponding prompt ID to 1 for all time steps
        mask[prompt_id, :] = 1

        return mask

    def get_hidden_length_from_sample_length(self, num_samples: int) -> int:
        """
        Calculate the hidden length from the given number of samples.

        Parameters:
            num_samples (int): The total number of audio samples.

        Returns:
            hidden_length (int): The calculated hidden length in terms of the number of frames.
        """
        mel_frame_count = math.ceil((num_samples + 1) / self.num_sample_per_mel_frame)
        hidden_length = math.ceil(mel_frame_count / self.subsampling_factor)
        return int(hidden_length)

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)
        tokens = [torch.as_tensor(self.tokenizer(c.supervisions[0].text, c.supervisions[0].language)) for c in cuts]

        # Create prompt targets
        prompt_targets = [
            torch.transpose(
                torch.as_tensor(
                    self.prompt_to_target(
                        c,
                        self.num_prompts,
                        self.num_sample_per_mel_frame,
                        self.subsampling_factor,
                    ),
                    dtype=torch.float32,
                ),
                0,
                1,
            )
            for c in cuts
        ]

        # Create final tensors
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        prompt_targets = collate_matrices(prompt_targets)

        return (
            audio,  # Audio signal
            audio_lens,  # Audio lengths
            tokens,  # Text tokens
            token_lens,  # Token lengths
            prompt_targets,  # Prompt targets
        )


class TokenizerWrapper:
    """
    Provide a unified interface for NeMo Tokenizer, AggregateTokenizer, and (char) Parser.
    """

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
