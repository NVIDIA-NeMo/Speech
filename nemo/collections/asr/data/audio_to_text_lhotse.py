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

import os
from typing import Dict, Optional, Tuple

import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseSpeechToTextBpeDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text.py.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.

    NOTE:
    If the environment variable ``USE_AIS_GET_BATCH`` is set to ``true`` (case-insensitive),
    then batch audio loading from AIStore will be enabled for this dataset. This will use the
    AISBatchLoader to load the audio from AIStore. This can improve data loading efficiency in some setups.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, tokenizer: TokenizerSpec, return_cuts: bool = False):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.use_ais_get_batch = os.environ.get("USE_AIS_GET_BATCH", "False").lower() == "true"

        # Try to use use_batch_loader if available (Lhotse >= 1.32.0)
        try:
            self.load_audio = AudioSamples(fault_tolerant=True, use_batch_loader=self.use_ais_get_batch)
        except TypeError:
            # Lhotse < 1.32.0 doesn't support use_batch_loader
            if self.use_ais_get_batch:
                import logging

                logging.warning(
                    "AIS batch loading requested but not supported by this Lhotse version. "
                    "Please upgrade to Lhotse >= 1.32.0"
                )
            self.load_audio = AudioSamples(fault_tolerant=True)

        self.return_cuts = return_cuts

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)
        tokens = [
            torch.cat(
                [
                    torch.as_tensor(s.tokens if hasattr(s, "tokens") else self.tokenizer(s.text or "", s.language))
                    for s in c.supervisions
                ],
                dim=0,
            )
            for c in cuts
        ]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        if self.return_cuts:
            return audio, audio_lens, tokens, token_lens, cuts.drop_in_memory_data()
        return audio, audio_lens, tokens, token_lens
