# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import random
from typing import Dict, Optional, Tuple

import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import speaker_to_target
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseSpeechToTextSpkBpeDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text.py. It has the same functionality of LhotseSpeechToTextBpeDataset but also yield speaker target tensor.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'spk_targets': NeuralType(('B', 'T'), LabelsType()),
            'bg_spk_targets': NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(self, cfg, tokenizer: TokenizerSpec):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg
        self.num_speakers = self.cfg.get('num_speakers', 4)
        self.num_sample_per_mel_frame = self.cfg.get('num_sample_per_mel_frame', 160)
        self.num_mel_frame_per_asr_frame = self.cfg.get('num_mel_frame_per_asr_frame', 8)
        self.fixed_spk_id = self.cfg.get('fixed_spk_id', None)
        self.inference_mode = self.cfg.get('inference_mode', False)

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:

        audio, audio_lens, cuts = self.load_audio(cuts)

        tokens = []
        spk_targets = []
        bg_spk_targets = []

        if self.inference_mode:
            return audio, audio_lens, None, None, None, None

        for idx, cut in enumerate(cuts):

            speaker_targets, texts = speaker_to_target(
                a_cut=cut,
                num_speakers=self.num_speakers,
                num_sample_per_mel_frame=self.num_sample_per_mel_frame,
                num_mel_frame_per_asr_frame=self.num_mel_frame_per_asr_frame,
                return_text=True,
            )
            speaker_targets = speaker_targets.transpose(0, 1)[: len(texts)]

            target_speaker_id = random.choice(range(len(texts)))
            non_target_speaker_ids = [i for i in range(len(texts)) if i != target_speaker_id]
            text = texts[target_speaker_id]
            speaker_target = speaker_targets[target_speaker_id]
            bg_speaker_target = speaker_targets[non_target_speaker_ids].sum(dim=0) > 0

            tokens.append(torch.as_tensor(self.tokenizer(text, cut.supervisions[0].language)))
            spk_targets.append(speaker_target)
            bg_spk_targets.append(bg_speaker_target)

        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        spk_targets = collate_vectors(spk_targets, padding_value=0)
        bg_spk_targets = collate_vectors(bg_spk_targets, padding_value=0)

        return audio, audio_lens, tokens, token_lens, spk_targets, bg_spk_targets
