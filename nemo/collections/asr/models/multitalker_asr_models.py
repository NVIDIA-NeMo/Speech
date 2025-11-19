# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#
import os
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data.audio_to_text_lhotse_speaker import LhotseSpeechToTextSpkBpeDataset
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.mixins import TranscribeConfig
from nemo.collections.asr.parts.mixins.multitalker_asr_mixins import SpeakerKernelMixin
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo


class EncDecMultiTalkerRNNTBPEModel(EncDecRNNTBPEModel, SpeakerKernelMixin):
    """Base class for encoder decoder RNNT-based models with subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        # Initialize speaker kernel functionality from mixin
        self._init_speaker_kernel_config(cfg)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        return results

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            # Use open_dict to allow dynamic key addition
            with open_dict(config):
                config.global_rank = self.global_rank
                config.world_size = self.world_size

            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextSpkBpeDataset(
                    cfg=config,
                    tokenizer=self.tokenizer,
                ),
            )

    def training_step(self, batch, batch_nb):
        """Training step with speaker targets."""
        signal, signal_len, transcript, transcript_len, *additional_args = batch
        spk_targets, bg_spk_targets = additional_args

        self.set_speaker_targets(spk_targets, bg_spk_targets)

        batch = (signal, signal_len, transcript, transcript_len)

        return super().training_step(batch, batch_nb)

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        """Validation pass with speaker targets."""
        signal, signal_len, transcript, transcript_len, *additional_args = batch
        spk_targets, bg_spk_targets = additional_args

        self.set_speaker_targets(spk_targets, bg_spk_targets)

        batch = (signal, signal_len, transcript, transcript_len)

        return super().validation_pass(batch, batch_idx, dataloader_idx)

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        """Transcribe forward with speaker targets."""
        signal, signal_len, transcript, transcript_len, *additional_args = batch
        spk_targets, bg_spk_targets = additional_args

        self.set_speaker_targets(spk_targets, bg_spk_targets)

        batch = (signal, signal_len, transcript, transcript_len)

        return super()._transcribe_forward(batch, trcfg)

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'dataset_manifest' in config:
            manifest_filepath = config['dataset_manifest']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'use_lhotse': config.get('use_lhotse', True),
            'use_bucketing': False,
            'channel_selector': config.get('channel_selector', None),
            'inference_mode': self.cfg.test_ds.get('inference_mode', True),
            'fixed_spk_id': config.get('fixed_spk_id', None),
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))

        return temporary_datalayer
