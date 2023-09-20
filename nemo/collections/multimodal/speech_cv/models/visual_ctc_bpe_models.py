# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import os
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer_bpe import WERBPE, CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.multimodal.speech_cv.data import video_to_text_dataset
from nemo.collections.multimodal.speech_cv.models.visual_ctc_models import VisualEncDecCTCModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils

__all__ = ['VisualEncDecCTCModelBPE']


class VisualEncDecCTCModelBPE(VisualEncDecCTCModel, ASRBPEMixin):
    """Encoder decoder CTC-based models with Byte Pair Encoding."""

    def __init__(self, cfg: DictConfig, trainer=None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        with open_dict(cfg):
            # sidestepping the potential overlapping tokens issue in aggregate tokenizers
            if self.tokenizer_type == "agg":
                cfg.decoder.vocabulary = ListConfig(vocabulary)
            else:
                cfg.decoder.vocabulary = ListConfig(list(vocabulary.keys()))

        # Override number of classes if placeholder provided
        num_classes = cfg.decoder["num_classes"]

        if num_classes < 1:
            logging.info(
                "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                    num_classes, len(vocabulary)
                )
            )
            cfg.decoder["num_classes"] = len(vocabulary)

        super().__init__(cfg=cfg, trainer=trainer)

        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)

        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg

        self.decoding = CTCBPEDecoding(self.cfg.decoding, tokenizer=self.tokenizer)

        # Setup metric with decoding strategy
        self._wer = WERBPE(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        dataset = video_to_text_dataset.get_video_to_text_bpe_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
            preprocessor_cfg=self.cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        shuffle = config['shuffle']
        if config.get('is_tarred', False):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
            prefetch_factor=config.get('prefetch_factor', 2),
        )

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided video file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2video_files: (a list) of paths to video files.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the video manifest is temporarily
                stored.
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

        Returns:
            A pytorch DataLoader for the given video file(s).
        """

        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2video_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'channel_selector': config.get('channel_selector', None),
            'use_start_end_token': self.cfg.validation_ds.get('use_start_end_token', False),
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def change_vocabulary(
        self,
        new_tokenizer_dir: Union[str, DictConfig],
        new_tokenizer_type: str,
        decoding_cfg: Optional[DictConfig] = None,
    ):
        """
        Changes vocabulary of the tokenizer used during CTC decoding process.
        Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_tokenizer_dir: Directory path to tokenizer or a config for a new tokenizer (if the tokenizer type is `agg`)
            new_tokenizer_type: Either `agg`, `bpe` or `wpe`. `bpe` is used for SentencePiece tokenizers,
                whereas `wpe` is used for `BertTokenizer`.
            new_tokenizer_cfg: A config for the new tokenizer. if provided, pre-empts the dir and type

        Returns: None

        """
        if isinstance(new_tokenizer_dir, DictConfig):
            if new_tokenizer_type == 'agg':
                new_tokenizer_cfg = new_tokenizer_dir
            else:
                raise ValueError(
                    f'New tokenizer dir should be a string unless the tokenizer is `agg`, but this tokenizer type is: {new_tokenizer_type}'
                )
        else:
            new_tokenizer_cfg = None

        if new_tokenizer_cfg is not None:
            tokenizer_cfg = new_tokenizer_cfg
        else:
            if not os.path.isdir(new_tokenizer_dir):
                raise NotADirectoryError(
                    f'New tokenizer dir must be non-empty path to a directory. But I got: {new_tokenizer_dir}'
                    f"New tokenizer dir must be non-empty path to a directory. But I got: {new_tokenizer_dir}"
                )

            if new_tokenizer_type.lower() not in ('bpe', 'wpe'):
                raise ValueError(f'New tokenizer type must be either `bpe` or `wpe`')

            tokenizer_cfg = OmegaConf.create({'dir': new_tokenizer_dir, 'type': new_tokenizer_type})

        # Setup the tokenizer
        self._setup_tokenizer(tokenizer_cfg)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        decoder_config = copy.deepcopy(self.decoder.to_config_dict())
        # sidestepping the potential overlapping tokens issue in aggregate tokenizers
        if self.tokenizer_type == "agg":
            decoder_config.vocabulary = ListConfig(vocabulary)
        else:
            decoder_config.vocabulary = ListConfig(list(vocabulary.keys()))

        decoder_num_classes = decoder_config['num_classes']

        # Override number of classes if placeholder provided
        logging.info(
            "\nReplacing old number of classes ({}) with new number of classes - {}".format(
                decoder_num_classes, len(vocabulary)
            )
        )

        decoder_config['num_classes'] = len(vocabulary)

        del self.decoder
        self.decoder = VisualEncDecCTCModelBPE.from_config_dict(decoder_config)
        del self.loss
        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

        if decoding_cfg is None:
            # Assume same decoding config as before
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = CTCBPEDecoding(decoding_cfg=decoding_cfg, tokenizer=self.tokenizer)

        self._wer = WERBPE(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get("log_prediction", False),
            dist_sync_on_step=True,
        )

        # Update config
        with open_dict(self.cfg.decoder):
            self._cfg.decoder = decoder_config

        with open_dict(self.cfg.decoding):
            self._cfg.decoding = decoding_cfg

        logging.info(f"Changed tokenizer to {self.decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig):
        """
        Changes decoding strategy used during CTC decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = CTCBPEDecoding(decoding_cfg=decoding_cfg, tokenizer=self.tokenizer,)

        self._wer = WERBPE(
            decoding=self.decoding,
            use_cer=self._wer.use_cer,
            log_prediction=self._wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        return results
