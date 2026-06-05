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

from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
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

    def __init__(self, tokenizer: TokenizerSpec, cfg: Dict) -> None:
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg

        # Load prompt dictionary from config
        self.prompt_dict = cfg.get('prompt_dictionary')
        if not self.prompt_dict:
            raise ValueError("prompt_dictionary is required in config")

        self.num_prompts = cfg.get('num_prompts', 128)

        # Per-dataset prompt mode is read from cut.custom["prompt_mode"] at runtime.
        # Supported values:
        #   "langID"  — always pass the real language ID
        #   "auto"    — always pass auto (101)
        #   "unified" — randomize: auto with probability unified_auto_ratio, else lang ID
        # Set via lhotse input_cfg tags, e.g.  tags: { prompt_mode: langID }
        self.prompt_mode_field = cfg.get('prompt_mode_field', 'prompt_mode')
        self.default_prompt_mode = cfg.get('default_prompt_mode', 'unified')
        self.unified_auto_ratio = cfg.get('unified_auto_ratio', 0.5)

        # Index used for the language-agnostic / auto prompt
        self.auto_index = self.prompt_dict.get('auto', 101)

        logging.info(
            f"LhotseSpeechToTextBpeDatasetWithPromptIndex: "
            f"default_prompt_mode={self.default_prompt_mode}, "
            f"unified_auto_ratio={self.unified_auto_ratio}"
        )

    def _get_prompt_index(self, prompt_key: str) -> int:
        """Maps prompt keys to indices using the prompt dictionary."""
        if prompt_key not in self.prompt_dict:
            available_keys = list(self.prompt_dict.keys())
            raise ValueError(
                f"Unknown prompt key: '{prompt_key}'. Available: {available_keys[:10]}{'...' if len(available_keys) > 10 else ''}"
            )
        return self.prompt_dict[prompt_key]

    def _get_prompt_mode(self, cut) -> str:
        """Resolve the prompt_mode for a cut from its custom tags."""
        if cut.custom is not None:
            mode = cut.custom.get(self.prompt_mode_field)
            if mode is not None:
                return mode
        return self.default_prompt_mode

    def _get_prompt_index_for_cut(self, cut) -> int:
        """
        Determine the prompt index for a cut based on its prompt_mode tag.

        Behaviour depends on prompt_mode (set per-dataset via lhotse
        input_cfg tags, falling back to ``default_prompt_mode``):
            "langID"  — always return the real language ID
                        (use for inference / language-forced tasks)
            "auto"    — always return auto index (language-agnostic)
            "unified" — return auto with probability unified_auto_ratio,
                        otherwise the real language ID
        """
        mode = self._get_prompt_mode(cut)

        if mode == 'langID':
            return self._get_prompt_index(cut.supervisions[0].language)
        elif mode == 'auto':
            return self.auto_index
        elif mode == 'unified':
            if random.random() < self.unified_auto_ratio:
                return self.auto_index
            return self._get_prompt_index(cut.supervisions[0].language)
        else:
            logging.warning(f"Unknown prompt_mode '{mode}', falling back to unified")
            if random.random() < self.unified_auto_ratio:
                return self.auto_index
            return self._get_prompt_index(cut.supervisions[0].language)

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)
        tokens = [torch.as_tensor(self.tokenizer(c.supervisions[0].text, c.supervisions[0].language)) for c in cuts]

        # Get prompt indices (just the language ID per sample, NOT full tensors)
        prompt_indices = torch.tensor([self._get_prompt_index_for_cut(c) for c in cuts], dtype=torch.long)

        # Create final tensors
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)

        return (
            audio,  # Audio signal [B, T]
            audio_lens,  # Audio lengths [B]
            tokens,  # Text tokens [B, T]
            token_lens,  # Token lengths [B]
            prompt_indices,  # Language ID indices [B] - model creates full tensor
        )
