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

# pylint: disable=E1101
import math
import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechE2ESpkDiarDataset
from nemo.collections.asr.data.audio_to_diar_label_lhotse import LhotseAudioToSpeechE2ESpkDiarDataset
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig, SpkDiarizationMixin
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures, WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    get_ats_targets_hungarian,
    get_pil_targets_hungarian,
)
from nemo.collections.asr.parts.utils.speaker_utils import generate_diarization_output_lines
from nemo.collections.asr.parts.utils.vad_utils import predlist_to_timestamps
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, safe_instantiate
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging

__all__ = ['SortformerEncLabelModel']


class SortformerEncLabelModel(ModelPT, ExportableEncDecModel, SpkDiarizationMixin):
    """
    Encoder class for Sortformer diarization model.
    Model class creates training, validation methods for setting up data performing model forward pass.

    This model class expects config dict for:
        * preprocessor
        * Transformer Encoder
        * FastConformer Encoder
        * Sortformer Modules
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly
        from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []

        model = PretrainedModelInfo(
            pretrained_model_name="diar_sortformer_4spk-v1",
            description="For details about this model, please visit https://huggingface.co/nvidia/diar_sortformer_4spk-v1",
            location="https://huggingface.co/nvidia/diar_sortformer_4spk-v1",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="diar_streaming_sortformer_4spk-v2",
            description="For details about this model, please visit https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2",
            location="https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="diar_streaming_sortformer_4spk-v2.1",
            description="For details about this model, please visit https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1",
            location="https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1",
        )
        result.append(model)

        return result

    def _resolve_output_resolution(self):
        """
        Resolve and validate the high-resolution and returned-output configuration.

        Returns:
            high_resolution (bool): Whether the model emits predictions at the pre-subsampling frame rate.
            output_subsampling_factor (int): Validated number of 10 ms feature frames per returned prediction.
        """
        high_resolution = self._cfg.get("high_resolution", False)
        if not isinstance(high_resolution, bool):
            raise TypeError(f"high_resolution must be a boolean, got {type(high_resolution).__name__}")

        encoder_factor = self._cfg.encoder.get("subsampling_factor", 8)
        if type(encoder_factor) is not int or encoder_factor < 1:
            raise ValueError(f"encoder.subsampling_factor must be a positive integer, got {encoder_factor}")

        output_factor = self._cfg.get("output_subsampling_factor", encoder_factor)
        if type(output_factor) is not int or output_factor < 1:
            raise ValueError(f"output_subsampling_factor must be a positive integer, got {output_factor}")
        native_output_factor = 1 if high_resolution else encoder_factor
        if output_factor % native_output_factor != 0:
            logging.warning(
                f"output_subsampling_factor={output_factor} must be an integer multiple of the model's native "
                f"subsampling factor ({native_output_factor}). Using {native_output_factor} instead."
            )
            output_factor = native_output_factor
        return high_resolution, output_factor

    def _check_streaming_parameters(self):
        """Validate module-internal and output-resolution streaming parameters."""
        self.sortformer_modules._check_streaming_parameters()
        native_output_factor = 1 if self.high_resolution else self.encoder.subsampling_factor
        downsample_factor = self.output_subsampling_factor // native_output_factor
        native_chunk_prediction_length = self.sortformer_modules.chunk_len * self.upsample_factor
        if native_chunk_prediction_length % downsample_factor != 0:
            raise ValueError(
                "Streaming output downsampling requires complete pooling windows at chunk boundaries, but the "
                f"native chunk prediction length ({native_chunk_prediction_length}) is not divisible by the "
                f"downsample factor ({downsample_factor}). Got chunk_len={self.sortformer_modules.chunk_len}, "
                f"upsample_factor={self.upsample_factor}, and "
                f"output_subsampling_factor={self.output_subsampling_factor}. Choose a compatible chunk length "
                "or output subsampling factor."
            )

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initialize an Sortformer Diarizer model and a pretrained NEST encoder.
        In this init function, training and validation datasets are prepared.
        """
        random.seed(42)
        self._trainer = trainer if trainer else None
        self._cfg = cfg
        self.high_resolution, self.output_subsampling_factor = self._resolve_output_resolution()

        if self._trainer:
            self.world_size = trainer.num_nodes * trainer.num_devices
        else:
            self.world_size = 1

        if self._trainer is not None and self._cfg.get('augmentor', None) is not None:
            self.augmentor = process_augmentations(self._cfg.augmentor)
        else:
            self.augmentor = None
        super().__init__(cfg=self._cfg, trainer=trainer)
        self.preprocessor = SortformerEncLabelModel.from_config_dict(self._cfg.preprocessor)

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = SortformerEncLabelModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.encoder = SortformerEncLabelModel.from_config_dict(self._cfg.encoder).to(self.device)
        self.upsample_factor = self.encoder.subsampling_factor if self.high_resolution else 1
        sortformer_modules_cfg = OmegaConf.create(OmegaConf.to_container(self._cfg.sortformer_modules, resolve=True))
        sortformer_modules_cfg.subsampling_factor = self.encoder.subsampling_factor
        sortformer_modules_cfg.upsample_factor = self.upsample_factor
        self.sortformer_modules = SortformerEncLabelModel.from_config_dict(sortformer_modules_cfg).to(self.device)
        transformer_encoder_cfg = self._cfg.get('transformer_encoder')
        self.transformer_encoder = (
            SortformerEncLabelModel.from_config_dict(transformer_encoder_cfg).to(self.device)
            if transformer_encoder_cfg is not None
            else None
        )
        if self._cfg.encoder.d_model != self._cfg.model_defaults.tf_d_model:
            self.sortformer_modules.encoder_proj = self.sortformer_modules.encoder_proj.to(self.device)
        else:
            self.sortformer_modules.encoder_proj = None
        self._init_loss_weights()

        self.eps = self._cfg.get("eps", 1e-3)
        self.negative_init_val = self._cfg.get("negative_init_val", -99)
        self.loss = safe_instantiate(self._cfg.loss)

        self.async_streaming = self._cfg.get("async_streaming", False)
        # Async rows are ragged; padding to full state capacity keeps the encoder time dimension fixed.
        self.async_pad_to_max = self._cfg.get("async_pad_to_max", False)
        self.streaming_mode = self._cfg.get("streaming_mode", False)
        if self.streaming_mode:
            # Validate streaming parameters once at initialization for streaming models
            self._check_streaming_parameters()
        self.save_hyperparameters("cfg")
        self._init_eval_metrics()

        self.max_batch_dur = self._cfg.get("max_batch_dur", 20000)
        self.rttms_mask_mats: List[torch.Tensor] = None  # Used when GT diarization needs to be tested.

    def add_rttms_mask_mats(self, rttms_mask_mats, device: torch.device):
        """
        Check if the rttms_mask_mats is empty then add it to the list

        Args:
            rttms_mask_mats (List[torch.Tensor]): List of PyTorch tensors containing the rttms mask matrices.
        """
        if self.rttms_mask_mats is None:
            self.rttms_mask_mats = rttms_mask_mats.to(device)
        else:
            raise ValueError(
                f"{self.rttms_mask_mats.shape}: rttms_mask_mats already exist but new one is being added."
            )

    def _init_loss_weights(self):
        pil_weight = self._cfg.get("pil_weight", 0.0)
        ats_weight = self._cfg.get("ats_weight", 1.0)
        if pil_weight + ats_weight == 0:
            raise ValueError(f"weights for PIL {pil_weight} and ATS {ats_weight} cannot sum to 0")
        self.pil_weight = pil_weight / (pil_weight + ats_weight)
        self.ats_weight = ats_weight / (pil_weight + ats_weight)
        self.ats_tolerance = float(self._cfg.get("ats_tolerance", 0))
        if self.ats_tolerance < 0:
            raise ValueError(f"ats_tolerance must be non-negative, got {self.ats_tolerance}")

    def _init_eval_metrics(self):
        """
        If there is no label, then the evaluation metrics will be based on Permutation Invariant Loss (PIL).
        """
        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()

        self._accuracy_test_ats = MultiBinaryAccuracy()
        self._accuracy_train_ats = MultiBinaryAccuracy()
        self._accuracy_valid_ats = MultiBinaryAccuracy()

    def _reset_train_metrics(self):
        self._accuracy_train.reset()
        self._accuracy_train_ats.reset()

    def _reset_valid_metrics(self):
        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()

    def __setup_dataloader_from_config(self, config):
        config_values = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
        config = OmegaConf.create(config_values)
        config.subsampling_factor = self.output_subsampling_factor
        # Switch to lhotse dataloader if specified in the config
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseAudioToSpeechE2ESpkDiarDataset(cfg=config),
            )

        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=self.augmentor
        )
        fb_featurizer = FilterbankFeatures(
            sample_rate=self._cfg.preprocessor.sample_rate,
            normalize=self._cfg.preprocessor.normalize,
            n_window_size=int(self._cfg.preprocessor.window_size * config['sample_rate']),
            n_window_stride=int(self._cfg.preprocessor.window_stride * config['sample_rate']),
            window=self._cfg.preprocessor.window,
            nfilt=self._cfg.preprocessor.features,
            n_fft=self._cfg.preprocessor.n_fft,
            frame_splicing=self._cfg.preprocessor.frame_splicing,
            dither=self._cfg.preprocessor.dither,
        )

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        logging.info(f"Loading dataset from {config.manifest_filepath}")

        if self._trainer is not None:
            global_rank = self._trainer.global_rank
        else:
            global_rank = 0

        dataset = AudioToSpeechE2ESpkDiarDataset(
            manifest_filepath=config.manifest_filepath,
            soft_label_thres=config.soft_label_thres,
            session_len_sec=config.session_len_sec,
            num_spks=config.num_spks,
            featurizer=featurizer,
            fb_featurizer=fb_featurizer,
            window_stride=self._cfg.preprocessor.window_stride,
            global_rank=global_rank,
            soft_targets=config.soft_targets if 'soft_targets' in config else False,
            subsampling_factor=self.output_subsampling_factor,
            device=self.device,
        )

        self.data_collection = dataset.collection
        self.collate_ds = dataset

        dataloader_instance = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=self.collate_ds.eesd_train_collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 1),
            pin_memory=config.get('pin_memory', False),
        )
        return dataloader_instance

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(
            config=train_data_config,
        )

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self._validation_dl = self.__setup_dataloader_from_config(
            config=val_data_layer_config,
        )

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        self._test_dl = self.__setup_dataloader_from_config(
            config=test_data_config,
        )

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl
        return None

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "audio_signal": NeuralType(('B', 'T'), audio_eltype),
            "audio_signal_length": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return OrderedDict(
            {
                "preds": NeuralType(('B', 'T', 'C'), ProbsType()),
            }
        )

    def _call_pre_encode(self, features, lengths):
        """
        Invoke heterogeneous encoder pre-encode modules through one streaming interface.

        Streaming and export callers provide time-major features and require a ``(features, lengths)`` result,
        while supported pre-encode modules expose different interfaces. ``torch.nn.Linear`` returns only projected
        features, ``FeatureStacking`` expects channel-first input, and other subsamplers accept time-major input with
        keyword arguments. This helper adapts those cases without changing the caller-facing layout.

        Args:
            features (torch.Tensor): Input features with shape ``(B, T, D)``.
            lengths (torch.Tensor): Valid feature lengths with shape ``(B,)``.

        Returns:
            pre_encoded_features (torch.Tensor): Pre-encoded features with shape ``(B, T', D')``.
            pre_encoded_lengths (torch.Tensor): Valid pre-encoded lengths with shape ``(B,)``.
        """
        from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking

        if isinstance(self.encoder.pre_encode, torch.nn.Linear):
            return self.encoder.pre_encode(features), lengths
        if isinstance(self.encoder.pre_encode, FeatureStacking):
            return self.encoder.pre_encode(features.transpose(1, 2), lengths)
        return self.encoder.pre_encode(x=features, lengths=lengths)

    def frontend_encoder(self, processed_signal, processed_signal_length, bypass_pre_encode: bool = False):
        """
        Generate encoder outputs from frontend encoder.

        Args:
            processed_signal (torch.Tensor): tensor containing audio-feature
                (mel spectrogram, mfcc, etc.).
            processed_signal_length (torch.Tensor): tensor containing lengths
                of audio signal in integers.

        Returns:
            emb_seq (torch.Tensor): tensor containing encoder outputs.
            emb_seq_length (torch.Tensor): tensor containing lengths of encoder outputs.
        """
        emb_seq, emb_seq_length = self.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length,
            bypass_pre_encode=bypass_pre_encode,
        )
        emb_seq = emb_seq.transpose(1, 2)
        if self.sortformer_modules.encoder_proj is not None:
            emb_seq = self.sortformer_modules.encoder_proj(emb_seq)
        return emb_seq, emb_seq_length

    def forward_infer(self, emb_seq, emb_seq_length):
        """
        The main forward pass for diarization for offline diarization inference.

        Args:
            emb_seq (torch.Tensor): Tensor containing FastConformer encoder states (embedding vectors).
                Shape: (batch_size, diar_frame_count, emb_dim)
            emb_seq_length (torch.Tensor): Tensor containing lengths of FastConformer encoder states.
                Shape: (batch_size,)

        Returns:
            preds (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
        """
        encoder_mask = self.sortformer_modules.length_to_mask(emb_seq_length, emb_seq.shape[1])
        trans_emb_seq = (
            self.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
            if self.transformer_encoder is not None
            else emb_seq
        )
        trans_emb_seq = self.sortformer_modules.upsample_hidden(trans_emb_seq)
        if self.high_resolution:
            output_mask = encoder_mask.repeat_interleave(self.upsample_factor, dim=1)
        else:
            output_mask = encoder_mask
        _preds = self.sortformer_modules.forward_speaker_sigmoids(trans_emb_seq)
        preds = _preds * output_mask.unsqueeze(-1)
        return preds

    def _diarize_forward(self, batch: Any):
        """
        A counterpart of `_transcribe_forward` function in ASR.
        This function is a wrapper for forward pass functions for compataibility
        with the existing classes.

        Args:
            batch (Any): The input batch containing audio signal and audio signal length.

        Returns:
            preds (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
        """
        with torch.no_grad():
            preds = self.forward(audio_signal=batch[0], audio_signal_length=batch[1])
            preds = preds.to('cpu')
            torch.cuda.empty_cache()
        return preds

    def _diarize_output_processing(
        self, outputs, uniq_ids, diarcfg: DiarizeConfig
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[torch.Tensor]]]:
        """
        Processes the diarization outputs and generates RTTM (Real-time Text Markup) files.

        Args:
            outputs (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            uniq_ids (List[str]): List of unique identifiers for each audio file.
            diarcfg (DiarizeConfig): Configuration object for diarization.

        Returns:
            diar_output_lines_list (List[List[str]]): A list of lists, where each inner list contains
                                                      the RTTM lines for a single audio file.
            preds_list (List[torch.Tensor]): A list of tensors containing the diarization outputs
                                             for each audio file.
        """
        preds_list, diar_output_lines_list = [], []
        if outputs.shape[0] == 1:  # batch size = 1
            preds_list.append(outputs)
        else:
            preds_list.extend(torch.split(outputs, [1] * outputs.shape[0]))

        batch_audio_rttm_map = {uniq_id: self._diarize_audio_rttm_map[uniq_id] for uniq_id in uniq_ids}
        total_speaker_timestamps = predlist_to_timestamps(
            batch_preds_list=preds_list,
            audio_rttm_map_dict=batch_audio_rttm_map,
            cfg_vad_params=diarcfg.postprocessing_params,
            unit_10ms_frame_count=self.output_subsampling_factor,
            bypass_postprocessing=False,
        )
        for speaker_timestamps in total_speaker_timestamps:
            diar_output_lines = generate_diarization_output_lines(
                speaker_timestamps=speaker_timestamps, model_spk_num=len(speaker_timestamps)
            )
            diar_output_lines_list.append(diar_output_lines)
        if diarcfg.include_tensor_outputs:
            return (diar_output_lines_list, preds_list)
        else:
            return diar_output_lines_list

    def _setup_diarize_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            - manifest_filepath: Path to the manifest file containing audio file paths
              and corresponding speaker labels.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'num_spks': config.get('num_spks', self._cfg.max_num_of_spks),
            'batch_size': batch_size,
            'shuffle': False,
            'soft_label_thres': 0.5,
            'session_len_sec': config['session_len_sec'],
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'use_lhotse': config.get('use_lhotse', False),
        }
        temporary_datalayer = self.__setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def oom_safe_feature_extraction(self, input_signal, input_signal_length):
        """
        This function divides the input signal into smaller sub-batches and processes them sequentially
        to prevent out-of-memory errors during feature extraction.

        Args:
            input_signal (torch.Tensor): The input audio signal.
            input_signal_length (torch.Tensor): The lengths of the input audio signals.

        Returns:
            A tuple of ``(processed_signal, processed_signal_length)`` where
            ``processed_signal`` is the aggregated audio signal tensor
            (length matches original batch size) and
            ``processed_signal_length`` contains the lengths of the processed signals.
        """
        input_signal = input_signal.cpu()
        processed_signal_list, processed_signal_length_list = [], []
        max_batch_sec = input_signal.shape[1] / self.preprocessor._cfg.sample_rate
        org_batch_size = input_signal.shape[0]
        div_batch_count = min(int(max_batch_sec * org_batch_size // self.max_batch_dur + 1), org_batch_size)
        div_size = math.ceil(org_batch_size / div_batch_count)

        for div_count in range(div_batch_count):
            start_idx = int(div_count * div_size)
            end_idx = int((div_count + 1) * div_size)
            if start_idx >= org_batch_size:
                break
            input_signal_div = input_signal[start_idx:end_idx, :].to(self.device)
            input_signal_length_div = input_signal_length[start_idx:end_idx]
            processed_signal_div, processed_signal_length_div = self.preprocessor(
                input_signal=input_signal_div, length=input_signal_length_div
            )
            processed_signal_div = processed_signal_div.detach().cpu()
            processed_signal_length_div = processed_signal_length_div.detach().cpu()
            processed_signal_list.append(processed_signal_div)
            processed_signal_length_list.append(processed_signal_length_div)

        processed_signal = torch.cat(processed_signal_list, 0)
        processed_signal_length = torch.cat(processed_signal_length_list, 0)
        assert processed_signal.shape[0] == org_batch_size, (
            f"The resulting batch size of processed signal - {processed_signal.shape[0]} "
            f"is not equal to original batch size: {org_batch_size}"
        )
        processed_signal = processed_signal.to(self.device)
        processed_signal_length = processed_signal_length.to(self.device)
        return processed_signal, processed_signal_length

    def process_signal(self, audio_signal, audio_signal_length):
        """
        Extract audio features from time-series signal for further processing in the model.

        This function performs the following steps:
        1. Moves the audio signal to the correct device.
        2. Normalizes the time-series audio signal.
        3. Extrac audio feature from from the time-series audio signal using the model's preprocessor.

        Args:
            audio_signal (torch.Tensor): The input audio signal.
                Shape: (batch_size, num_samples)
            audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                Shape: (batch_size,)

        Returns:
            processed_signal (torch.Tensor): The preprocessed audio signal.
                Shape: (batch_size, num_features, num_frames)
            processed_signal_length (torch.Tensor): The length of each processed signal.
                Shape: (batch_size,)
        """
        audio_signal, audio_signal_length = audio_signal.to(self.device), audio_signal_length.to(self.device)
        if not self.streaming_mode:
            audio_signal = (1 / (audio_signal.max() + self.eps)) * audio_signal

        batch_total_dur = audio_signal.shape[0] * audio_signal.shape[1] / self.preprocessor._cfg.sample_rate
        if self.max_batch_dur > 0 and self.max_batch_dur < batch_total_dur:
            processed_signal, processed_signal_length = self.oom_safe_feature_extraction(
                input_signal=audio_signal, input_signal_length=audio_signal_length
            )
        else:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=audio_signal, length=audio_signal_length
            )
        # This cache clearning can significantly slow down the training speed.
        # Only perform `empty_cache()` when the input file is extremely large for streaming mode.
        if not self.training and self.streaming_mode:
            del audio_signal, audio_signal_length
            torch.cuda.empty_cache()
        return processed_signal, processed_signal_length

    def forward(
        self,
        audio_signal,
        audio_signal_length,
    ):
        """
        Forward pass for training and inference.

        Args:
            audio_signal (torch.Tensor): Tensor containing audio waveform
                Shape: (batch_size, num_samples)
            audio_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)

        Returns:
            preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Shape: (batch_size, max. diar frame count, num_speakers)
        """
        processed_signal, processed_signal_length = self.process_signal(
            audio_signal=audio_signal, audio_signal_length=audio_signal_length
        )
        processed_signal = processed_signal[:, :, : processed_signal_length.max()]
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
        if self.streaming_mode:
            preds = self.forward_streaming(processed_signal, processed_signal_length)
            preds_frame_factor = self.output_subsampling_factor
        else:
            emb_seq, emb_seq_length = self.frontend_encoder(
                processed_signal=processed_signal, processed_signal_length=processed_signal_length
            )
            preds = self.forward_infer(emb_seq, emb_seq_length)
            preds_frame_factor = 1 if self.high_resolution else self.encoder.subsampling_factor

        max_output_length = min(preds.shape[1], math.ceil(processed_signal.shape[2] / preds_frame_factor))
        output_lengths = torch.div(
            processed_signal_length + preds_frame_factor - 1,
            preds_frame_factor,
            rounding_mode='floor',
        ).clamp(max=max_output_length)
        preds = preds[:, :max_output_length]
        output_mask = self.sortformer_modules.length_to_mask(output_lengths, max_output_length)
        preds = preds * output_mask.unsqueeze(-1)

        downsample_factor = self.output_subsampling_factor // preds_frame_factor
        if downsample_factor > 1:
            preds = self.sortformer_modules.downsample_preds(
                preds,
                downsample_factor,
                lengths=output_lengths,
            )
        return preds

    @property
    def input_names(self):
        return ["chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"]

    @property
    def output_names(self):
        return ["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs", "chunk_pre_encode_lengths"]

    def streaming_input_examples(self, batch_size: int = 1):
        """
        Create model-sized input examples for exporting the streaming graph.

        Args:
            batch_size (int): Number of examples in each generated tensor. Defaults to 1.

        Returns:
            chunk (torch.Tensor): Input feature chunk with shape ``(B, T_chunk, D_feature)``.
            chunk_lengths (torch.Tensor): Valid chunk lengths with shape ``(B,)``.
            spkcache (torch.Tensor): Speaker-cache embeddings with shape ``(B, T_cache, D_embedding)``.
            spkcache_lengths (torch.Tensor): Valid speaker-cache lengths with shape ``(B,)``.
            fifo (torch.Tensor): FIFO embeddings with shape ``(B, T_fifo, D_embedding)``.
            fifo_lengths (torch.Tensor): Valid FIFO lengths with shape ``(B,)``.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        chunk_frames = (
            self.sortformer_modules.chunk_left_context
            + self.sortformer_modules.chunk_len
            + self.sortformer_modules.chunk_right_context
        ) * self.encoder.subsampling_factor
        spkcache_capacity = self.sortformer_modules.spkcache_len
        fifo_capacity = self.sortformer_modules.fifo_len
        embedding_dim = self.sortformer_modules.fc_d_model

        chunk = torch.rand((batch_size, chunk_frames, self.encoder._feat_in), device=self.device)
        chunk_lengths = torch.full((batch_size,), chunk_frames, dtype=torch.long, device=self.device)
        spkcache = torch.randn((batch_size, spkcache_capacity, embedding_dim), device=self.device)
        spkcache_lengths = torch.full((batch_size,), spkcache_capacity // 2, dtype=torch.long, device=self.device)
        fifo = torch.randn((batch_size, fifo_capacity, embedding_dim), device=self.device)
        fifo_lengths = torch.full((batch_size,), fifo_capacity // 2, dtype=torch.long, device=self.device)
        return chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths

    def streaming_export(self, output: str, input_example=None, batch_size: int = 1):
        """
        Export the streaming graph with explicit inputs or model-sized defaults.

        Args:
            output (str): Output filename whose extension selects the export format.
            input_example (Optional[Tuple[torch.Tensor, ...]]): Six streaming input tensors used to trace the model.
                If ``None``, model-sized examples are generated.
            batch_size (int): Batch size for generated input examples. Ignored when ``input_example`` is provided.

        Returns:
            exported_outputs (List[str]): Output paths returned by ``Exportable.export`` for the exported subnets.
            export_descriptions (List[str]): Descriptions returned by ``Exportable.export`` for the exported subnets.
        """
        if input_example is None:
            input_example = self.streaming_input_examples(batch_size=batch_size)
        return self.export(output, input_example=input_example)

    def forward_for_export(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
        """
        This forward pass is for ONNX model export.

        Args:
            chunk (torch.Tensor): Tensor containing audio waveform.
                The term "chunk" refers to the "input buffer" in the speech processing pipeline.
                The size of chunk (input buffer) determines the latency introduced by buffering.
                Shape: (batch_size, feature frame count, dimension)
            chunk_lengths (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)
            spkcache (torch.Tensor): Tensor containing speaker cache embeddings from start
                Shape: (batch_size, spkcache_len, emb_dim)
            spkcache_lengths (torch.Tensor): Tensor containing lengths of speaker cache
                Shape: (batch_size,)
            fifo (torch.Tensor): Tensor containing embeddings from latest chunks
                Shape: (batch_size, fifo_len, emb_dim)
            fifo_lengths (torch.Tensor): Tensor containing lengths of FIFO queue embeddings
                Shape: (batch_size,)

        Returns:
            spkcache_fifo_chunk_preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Shape: (batch_size, max. diar frame count, num_speakers)
            chunk_pre_encode_embs (torch.Tensor): Tensor containing pre-encoded embeddings from the chunk
                Shape: (batch_size, num_frames, emb_dim)
            chunk_pre_encode_lengths (torch.Tensor): Tensor containing lengths of pre-encoded embeddings
                from the chunk (=input buffer).
                Shape: (batch_size,)
        """
        # pre-encode the chunk
        chunk_pre_encode_embs, chunk_pre_encode_lengths = self._call_pre_encode(chunk, chunk_lengths)
        chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)

        # concat the embeddings from speaker cache, FIFO queue and the chunk
        output_length = spkcache.shape[1] + fifo.shape[1] + chunk_pre_encode_embs.shape[1]
        spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths = (
            self.sortformer_modules.concat_and_pad(
                [spkcache, fifo, chunk_pre_encode_embs],
                [spkcache_lengths, fifo_lengths, chunk_pre_encode_lengths],
                output_length=output_length,
            )
        )

        # encode the concatenated embeddings
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.frontend_encoder(
            processed_signal=spkcache_fifo_chunk_pre_encode_embs,
            processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
            bypass_pre_encode=True,
        )

        # forward pass for inference
        spkcache_fifo_chunk_preds = self.forward_infer(
            spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths
        )
        if self.high_resolution:
            spkcache_fifo_chunk_preds = self.sortformer_modules.downsample_preds(
                spkcache_fifo_chunk_preds, self.upsample_factor
            )
        return spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths

    def forward_streaming(
        self,
        processed_signal,
        processed_signal_length,
    ):
        """
        The main forward pass for diarization inference in streaming mode.

        Args:
            processed_signal (torch.Tensor): Tensor containing audio waveform
                Shape: (batch_size, num_samples)
            processed_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)

        Returns:
            total_preds (torch.Tensor): Tensor containing predicted speaker labels for the current chunk
                and all previous chunks
                Shape: (batch_size, pred_len, num_speakers)
        """
        streaming_state = self.sortformer_modules.init_streaming_state(
            batch_size=processed_signal.shape[0], async_streaming=self.async_streaming, device=self.device
        )

        batch_size, ch, sig_length = processed_signal.shape
        processed_signal_offset = torch.zeros((batch_size,), dtype=torch.long, device=self.device)

        if dist.is_available() and dist.is_initialized():
            local_tensor = torch.tensor([sig_length], device=processed_signal.device)
            dist.all_reduce(
                local_tensor, op=dist.ReduceOp.MAX, async_op=False
            )  # get max feature length across all GPUs
            max_n_frames = local_tensor.item()
            if dist.get_rank() == 0:
                logging.info(f"Maximum feature length across all GPUs: {max_n_frames}")
        else:
            max_n_frames = sig_length

        if sig_length < max_n_frames:  # need padding to have the same feature length for all GPUs
            pad_tensor = torch.full(
                (batch_size, ch, max_n_frames - sig_length),
                self.negative_init_val,
                dtype=processed_signal.dtype,
                device=processed_signal.device,
            )
            processed_signal = torch.cat([processed_signal, pad_tensor], dim=2)

        att_mod = False
        if self.training:
            rand_num = random.random()
            if rand_num < self.sortformer_modules.causal_attn_rate:
                self.encoder.att_context_size = [-1, self.sortformer_modules.causal_attn_rc]
                if self.transformer_encoder is not None:
                    self.transformer_encoder.diag = self.sortformer_modules.causal_attn_rc
                att_mod = True

        total_preds = torch.zeros((batch_size, 0, self.sortformer_modules.n_spk), device=self.device)

        feat_len = processed_signal.shape[2]
        num_chunks = math.ceil(
            feat_len / (self.sortformer_modules.chunk_len * self.sortformer_modules.subsampling_factor)
        )
        streaming_loader = self.sortformer_modules.streaming_feat_loader(
            feat_seq=processed_signal,
            feat_seq_length=processed_signal_length,
            feat_seq_offset=processed_signal_offset,
        )
        for _, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in tqdm(
            streaming_loader,
            total=num_chunks,
            desc="Streaming Steps",
            disable=self.training,
        ):
            streaming_state, total_preds = self.forward_streaming_step(
                processed_signal=chunk_feat_seq_t,
                processed_signal_length=feat_lengths,
                streaming_state=streaming_state,
                total_preds=total_preds,
                left_offset=left_offset,
                right_offset=right_offset,
            )

        if att_mod:
            self.encoder.att_context_size = [-1, -1]
            if self.transformer_encoder is not None:
                self.transformer_encoder.diag = None

        del processed_signal, processed_signal_length

        output_frames = math.ceil(sig_length / self.output_subsampling_factor)
        total_preds = total_preds[:, :output_frames]
        return total_preds

    def _extract_async_high_resolution_chunk_preds(
        self,
        high_resolution_preds,
        spkcache_lengths,
        fifo_lengths,
        chunk_lengths,
        max_chunk_len,
        lc_enc,
    ):
        """
        Gather ragged high-resolution chunk predictions without synchronizing lengths to Python.

        Args:
            high_resolution_preds (torch.Tensor): Predictions over packed speaker-cache, FIFO, and chunk frames with
                shape ``(B, T_packed * upsample_factor, S)``.
            spkcache_lengths (torch.Tensor): Per-row valid speaker-cache lengths in encoder frames, with shape
                ``(B,)``.
            fifo_lengths (torch.Tensor): Per-row valid FIFO lengths in encoder frames, with shape ``(B,)``.
            chunk_lengths (torch.Tensor): Per-row valid central-chunk lengths in encoder frames, with shape ``(B,)``.
            max_chunk_len (int): Physical central-chunk capacity in encoder frames after excluding both contexts.
            lc_enc (int): Number of encoded left-context frames preceding the central chunk in packed time.

        Returns:
            chunk_preds (torch.Tensor): Gathered predictions with shape
                ``(B, max_chunk_len * upsample_factor, S)`` and zero padding after each valid row.
        """
        batch_size, _, num_speakers = high_resolution_preds.shape
        output_length = max_chunk_len * self.upsample_factor
        output_positions = torch.arange(output_length, device=high_resolution_preds.device).unsqueeze(0)
        output_lengths = chunk_lengths.unsqueeze(1) * self.upsample_factor
        start_indices = (spkcache_lengths + fifo_lengths + lc_enc).unsqueeze(1) * self.upsample_factor
        source_indices = start_indices + output_positions
        valid_mask = output_positions < output_lengths
        padding_index = high_resolution_preds.shape[1]
        source_indices = torch.where(valid_mask, source_indices, padding_index)
        padded_preds = torch.cat(
            [
                high_resolution_preds,
                high_resolution_preds.new_zeros((batch_size, 1, num_speakers)),
            ],
            dim=1,
        )
        return torch.gather(
            padded_preds,
            1,
            source_indices.unsqueeze(-1).expand(-1, -1, num_speakers),
        )

    def forward_streaming_step(
        self,
        processed_signal,
        processed_signal_length,
        streaming_state,
        total_preds,
        drop_extra_pre_encoded=0,
        left_offset=0,
        right_offset=0,
    ):
        """
        One-step forward pass for diarization inference in streaming mode.

        Args:
            processed_signal (torch.Tensor): Tensor containing audio waveform
                Shape: (batch_size, num_samples)
            processed_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)
            streaming_state (SortformerStreamingState):
                    Tensor variables that contain the streaming state of the model.
                    Find more details in the `SortformerStreamingState` class in `sortformer_modules.py`.

                Attributes:
                    spkcache (torch.Tensor): Speaker cache to store embeddings from start
                    spkcache_lengths (torch.Tensor): Lengths of the speaker cache
                    spkcache_preds (torch.Tensor): The speaker predictions for the speaker cache parts
                    spkcache_compressed (bool or torch.Tensor): Speaker-cache compression status
                    fifo (torch.Tensor): FIFO queue to save the embedding from the latest chunks
                    fifo_lengths (torch.Tensor): Lengths of the FIFO queue
                    fifo_preds (torch.Tensor): The speaker predictions for the FIFO queue parts
                    spk_perm (torch.Tensor): Speaker permutation information for the speaker cache

            total_preds (torch.Tensor): Tensor containing total predicted speaker activity probabilities
                Shape: (batch_size, cumulative pred length, num_speakers)
            drop_extra_pre_encoded (int): Number of leading pre-encoded frames to discard before streaming updates.
            left_offset (int): left offset for the current chunk
            right_offset (int): right offset for the current chunk

        Returns:
            streaming_state (SortformerStreamingState):
                    Tensor variables that contain the updated streaming state of the model from
                    this function call.
            total_preds (torch.Tensor):
                Tensor containing the updated total predicted speaker activity probabilities.
                Shape: (batch_size, cumulative pred length, num_speakers)
        """
        chunk_pre_encode_embs, chunk_pre_encode_lengths = self._call_pre_encode(
            processed_signal, processed_signal_length
        )
        # To match the output of the ASR model, we need to drop the extra pre-encoded embeddings
        if drop_extra_pre_encoded > 0:
            chunk_pre_encode_embs = chunk_pre_encode_embs[:, drop_extra_pre_encoded:, :]
            chunk_pre_encode_lengths = chunk_pre_encode_lengths - drop_extra_pre_encoded

        if self.async_streaming:
            output_length = None
            if self.async_pad_to_max:
                output_length = (
                    streaming_state.spkcache.shape[1]
                    + streaming_state.fifo.shape[1]
                    + self.sortformer_modules.chunk_left_context
                    + self.sortformer_modules.chunk_len
                    + self.sortformer_modules.chunk_right_context
                )
            spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths = (
                self.sortformer_modules.concat_and_pad(
                    [streaming_state.spkcache, streaming_state.fifo, chunk_pre_encode_embs],
                    [streaming_state.spkcache_lengths, streaming_state.fifo_lengths, chunk_pre_encode_lengths],
                    output_length=output_length,
                )
            )
        else:
            spkcache_fifo_chunk_pre_encode_embs = self.sortformer_modules.concat_embs(
                [streaming_state.spkcache, streaming_state.fifo, chunk_pre_encode_embs], dim=1, device=self.device
            )
            spkcache_fifo_chunk_pre_encode_lengths = (
                streaming_state.spkcache.shape[1] + streaming_state.fifo.shape[1] + chunk_pre_encode_lengths
            )
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.frontend_encoder(
            processed_signal=spkcache_fifo_chunk_pre_encode_embs,
            processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
            bypass_pre_encode=True,
        )
        spkcache_fifo_chunk_preds = self.forward_infer(
            emb_seq=spkcache_fifo_chunk_fc_encoder_embs, emb_seq_length=spkcache_fifo_chunk_fc_encoder_lengths
        )

        lc_enc = round(left_offset / self.encoder.subsampling_factor)
        rc_enc = math.ceil(right_offset / self.encoder.subsampling_factor)
        high_resolution_preds = None
        if self.high_resolution:
            high_resolution_preds = spkcache_fifo_chunk_preds
            spkcache_fifo_chunk_preds = self.sortformer_modules.downsample_preds(
                high_resolution_preds, self.upsample_factor
            ).detach()
            if not self.async_streaming and streaming_state.spk_perm is not None:
                inv_spk_perm = torch.stack(
                    [
                        torch.argsort(streaming_state.spk_perm[batch_index])
                        for batch_index in range(high_resolution_preds.shape[0])
                    ]
                )
                high_resolution_preds = torch.stack(
                    [
                        high_resolution_preds[batch_index, :, inv_spk_perm[batch_index]]
                        for batch_index in range(high_resolution_preds.shape[0])
                    ]
                )

        spkcache_fifo_chunk_preds = self.sortformer_modules.apply_mask_to_preds(
            spkcache_fifo_chunk_preds, spkcache_fifo_chunk_fc_encoder_lengths
        )
        if self.async_streaming:
            saved_spkcache_lengths = streaming_state.spkcache_lengths.clone()
            saved_fifo_lengths = streaming_state.fifo_lengths.clone()
            streaming_state, chunk_preds = self.sortformer_modules.streaming_update_async(
                streaming_state=streaming_state,
                chunk=chunk_pre_encode_embs,
                chunk_lengths=chunk_pre_encode_lengths,
                preds=spkcache_fifo_chunk_preds,
                lc=lc_enc,
                rc=rc_enc,
            )
            if self.high_resolution:
                max_chunk_len = chunk_pre_encode_embs.shape[1] - lc_enc - rc_enc
                chunk_lengths = (chunk_pre_encode_lengths - lc_enc).clamp(min=0, max=max_chunk_len)
                chunk_preds = self._extract_async_high_resolution_chunk_preds(
                    high_resolution_preds=high_resolution_preds,
                    spkcache_lengths=saved_spkcache_lengths,
                    fifo_lengths=saved_fifo_lengths,
                    chunk_lengths=chunk_lengths,
                    max_chunk_len=max_chunk_len,
                    lc_enc=lc_enc,
                )
        else:
            saved_spkcache_len = streaming_state.spkcache.shape[1]
            saved_fifo_len = streaming_state.fifo.shape[1]
            streaming_state, chunk_preds = self.sortformer_modules.streaming_update(
                streaming_state=streaming_state,
                chunk=chunk_pre_encode_embs,
                preds=spkcache_fifo_chunk_preds,
                lc=lc_enc,
                rc=rc_enc,
            )
            if self.high_resolution:
                chunk_len = chunk_pre_encode_embs.shape[1] - lc_enc - rc_enc
                start = (saved_spkcache_len + saved_fifo_len + lc_enc) * self.upsample_factor
                chunk_preds = high_resolution_preds[:, start : start + chunk_len * self.upsample_factor]
        native_output_factor = 1 if self.high_resolution else self.encoder.subsampling_factor
        downsample_factor = self.output_subsampling_factor // native_output_factor
        if downsample_factor > 1:
            chunk_preds = self.sortformer_modules.downsample_preds(chunk_preds, downsample_factor)
        total_preds = torch.cat([total_preds, chunk_preds], dim=1)

        return streaming_state, total_preds

    @staticmethod
    def _align_predictions_and_targets(preds, targets, target_lens):
        """
        Align prediction and target time dimensions and preserve their shared dtype.

        Args:
            preds (torch.Tensor): Speaker predictions with shape ``(B, T_pred, S)``.
            targets (torch.Tensor): Speaker targets with shape ``(B, T_target, S)``.
            target_lens (torch.Tensor): Valid target lengths with shape ``(B,)``.

        Returns:
            aligned_preds (torch.Tensor): Predictions truncated to the shared time dimension.
            aligned_targets (torch.Tensor): Targets cast to the prediction dtype and truncated to the shared time
                dimension.
            clamped_target_lens (torch.Tensor): Target lengths clamped to the shared time dimension.
        """
        targets = targets.to(preds.dtype)
        common_num_frames = min(preds.shape[1], targets.shape[1])
        if preds.shape[1] != targets.shape[1]:
            logging.info(
                f"Prediction and target frame counts differ ({preds.shape[1]} vs {targets.shape[1]}). "
                f"Truncating both to {common_num_frames} frames."
            )
        preds = preds[:, :common_num_frames]
        targets = targets[:, :common_num_frames]
        target_lens = target_lens.clamp(max=common_num_frames)
        return preds, targets, target_lens

    def _get_aux_train_evaluations(self, preds, targets, target_lens) -> dict:
        """
        Compute auxiliary training evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        Args:
            preds (torch.Tensor): Predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)

        Returns:
            (dict): A dictionary containing the following training metrics.
        """
        preds, targets, target_lens = self._align_predictions_and_targets(preds, targets, target_lens)
        targets_ats, _ = get_ats_targets_hungarian(targets, preds, tolerance=self.ats_tolerance)
        targets_pil, _ = get_pil_targets_hungarian(targets, preds)
        ats_loss = self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
        pil_loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
        loss = self.ats_weight * ats_loss + self.pil_weight * pil_loss

        self._accuracy_train(preds, targets_pil, target_lens)
        train_f1_acc, train_precision, train_recall = self._accuracy_train.compute()

        self._accuracy_train_ats(preds, targets_ats, target_lens)
        train_f1_acc_ats, _, _ = self._accuracy_train_ats.compute()

        train_metrics = {
            'loss': loss,
            'ats_loss': ats_loss,
            'pil_loss': pil_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'train_f1_acc': train_f1_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1_acc_ats': train_f1_acc_ats,
        }
        return train_metrics

    def training_step(self, batch: list, batch_idx: int) -> dict:
        """
        Performs a single training step.

        Args:
            batch (list): A list containing the following elements:
                - audio_signal (torch.Tensor): The input audio signal in time-series format.
                - audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                - targets (torch.Tensor): The target labels for the batch.
                - target_lens (torch.Tensor): The length of each target sequence in the batch.
            batch_idx (int): The index of the current batch.

        Returns:
            (dict): A dictionary containing the 'loss' key with the calculated loss value.
        """
        audio_signal, audio_signal_length, targets, target_lens = batch
        preds = self.forward(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        train_metrics = self._get_aux_train_evaluations(preds, targets, target_lens)
        self._reset_train_metrics()
        self.log_dict(train_metrics, sync_dist=True, on_step=True, on_epoch=False, logger=True)
        return {'loss': train_metrics['loss']}

    def _get_aux_validation_evaluations(self, preds, targets, target_lens) -> dict:
        """
        Compute auxiliary validation evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        Args:
            preds (torch.Tensor): Predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)

        Returns:
            val_metrics (dict): A dictionary containing the following validation metrics
        """
        preds, targets, target_lens = self._align_predictions_and_targets(preds, targets, target_lens)
        targets_ats, _ = get_ats_targets_hungarian(targets, preds, tolerance=self.ats_tolerance)
        targets_pil, _ = get_pil_targets_hungarian(targets, preds)

        val_ats_loss = self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
        val_pil_loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
        val_loss = self.ats_weight * val_ats_loss + self.pil_weight * val_pil_loss

        self._accuracy_valid(preds, targets_pil, target_lens)
        val_f1_acc, val_precision, val_recall = self._accuracy_valid.compute()

        self._accuracy_valid_ats(preds, targets_ats, target_lens)
        valid_f1_acc_ats, _, _ = self._accuracy_valid_ats.compute()

        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()

        val_metrics = {
            'val_loss': val_loss,
            'val_ats_loss': val_ats_loss,
            'val_pil_loss': val_pil_loss,
            'val_f1_acc': val_f1_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1_acc_ats': valid_f1_acc_ats,
        }
        return val_metrics

    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        """
        Performs a single validation step.

        This method processes a batch of data during the validation phase. It forward passes
        the audio signal through the model, computes various validation metrics, and stores
        these metrics for later aggregation.

        Args:
            batch (list): A list containing the following elements:
                - audio_signal (torch.Tensor): The input audio signal.
                - audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                - targets (torch.Tensor): The target labels for the batch.
                - target_lens (torch.Tensor): The length of each target sequence in the batch.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int, optional): The index of the dataloader in case of multiple
                                            validation dataloaders. Defaults to 0.

        Returns:
            dict: A dictionary containing various validation metrics for this batch.
        """
        audio_signal, audio_signal_length, targets, target_lens = batch
        preds = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )
        val_metrics = self._get_aux_validation_evaluations(preds, targets, target_lens)
        if isinstance(self.trainer.val_dataloaders, list) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(val_metrics)
        else:
            self.validation_step_outputs.append(val_metrics)
        return val_metrics

    def test_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        """
        Performs a single validation step.

        This method processes a batch of data during the validation phase. It forward passes
        the audio signal through the model, computes various validation metrics, and stores
        these metrics for later aggregation.

        Args:
            batch (list): A list containing the following elements:
                - audio_signal (torch.Tensor): The input audio signal.
                - audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                - targets (torch.Tensor): The target labels for the batch.
                - target_lens (torch.Tensor): The length of each target sequence in the batch.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int, optional): The index of the dataloader in case of multiple
                                            validation dataloaders. Defaults to 0.

        Returns:
            dict: A dictionary containing various validation metrics for this batch.
        """
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        if not outputs:
            logging.warning(f"`outputs` is None; empty outputs for dataloader={dataloader_idx}")
            return None
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_ats_loss_mean = torch.stack([x['val_ats_loss'] for x in outputs]).mean()
        val_pil_loss_mean = torch.stack([x['val_pil_loss'] for x in outputs]).mean()
        val_f1_acc_mean = torch.stack([x['val_f1_acc'] for x in outputs]).mean()
        val_precision_mean = torch.stack([x['val_precision'] for x in outputs]).mean()
        val_recall_mean = torch.stack([x['val_recall'] for x in outputs]).mean()
        val_f1_acc_ats_mean = torch.stack([x['val_f1_acc_ats'] for x in outputs]).mean()

        self._reset_valid_metrics()

        multi_val_metrics = {
            'val_loss': val_loss_mean,
            'val_ats_loss': val_ats_loss_mean,
            'val_pil_loss': val_pil_loss_mean,
            'val_f1_acc': val_f1_acc_mean,
            'val_precision': val_precision_mean,
            'val_recall': val_recall_mean,
            'val_f1_acc_ats': val_f1_acc_ats_mean,
        }
        return {'log': multi_val_metrics}

    def _get_aux_test_batch_evaluations(self, batch_idx: int, preds, targets, target_lens):
        """
        Compute auxiliary validation evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        Args:
            preds (torch.Tensor): Predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)
        """
        preds, targets, target_lens = self._align_predictions_and_targets(preds, targets, target_lens)
        if targets.shape[2] > preds.shape[2]:
            metric_preds = torch.nn.functional.pad(preds, (0, targets.shape[2] - preds.shape[2]))
        else:
            metric_preds = preds
        targets_ats, _ = get_ats_targets_hungarian(targets, metric_preds, tolerance=self.ats_tolerance)
        targets_pil, _ = get_pil_targets_hungarian(targets, metric_preds)
        self._accuracy_test(metric_preds, targets_pil, target_lens)
        f1_acc, precision, recall = self._accuracy_test.compute()
        self.batch_f1_accs_list.append(f1_acc)
        self.batch_precision_list.append(precision)
        self.batch_recall_list.append(recall)
        logging.info(f"batch {batch_idx}: f1_acc={f1_acc}, precision={precision}, recall={recall}")

        self._accuracy_test_ats(metric_preds, targets_ats, target_lens)
        f1_acc_ats, precision_ats, recall_ats = self._accuracy_test_ats.compute()
        self.batch_f1_accs_ats_list.append(f1_acc_ats)
        logging.info(
            f"batch {batch_idx}: f1_acc_ats={f1_acc_ats}, precision_ats={precision_ats}, recall_ats={recall_ats}"
        )

        self._accuracy_test.reset()
        self._accuracy_test_ats.reset()

    def test_batch(
        self,
    ):
        """
        Perform batch testing on the model.

        This method iterates through the test data loader, making predictions for each batch,
        and calculates various evaluation metrics. It handles both single and multi-sample batches.
        """
        (
            self.preds_total_list,
            self.batch_f1_accs_list,
            self.batch_precision_list,
            self.batch_recall_list,
            self.batch_f1_accs_ats_list,
        ) = ([], [], [], [], [])

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self._test_dl)):
                audio_signal, audio_signal_length, targets, target_lens = batch
                audio_signal = audio_signal.to(self.device)
                audio_signal_length = audio_signal_length.to(self.device)
                targets = targets.to(self.device)
                preds = self.forward(
                    audio_signal=audio_signal,
                    audio_signal_length=audio_signal_length,
                )
                self._get_aux_test_batch_evaluations(batch_idx, preds, targets, target_lens)
                preds = preds.detach().to('cpu')
                if preds.shape[0] == 1:  # batch size = 1
                    self.preds_total_list.append(preds)
                else:
                    self.preds_total_list.extend(torch.split(preds, [1] * preds.shape[0]))
                torch.cuda.empty_cache()

        logging.info(f"Batch F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_list))}")
        logging.info(f"Batch Precision MEAN: {torch.mean(torch.tensor(self.batch_precision_list))}")
        logging.info(f"Batch Recall MEAN: {torch.mean(torch.tensor(self.batch_recall_list))}")
        logging.info(f"Batch ATS F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_ats_list))}")

    def on_validation_epoch_end(self) -> Optional[dict[str, dict[str, torch.Tensor]]]:
        """Run validation with sync_dist=True."""
        return super().on_validation_epoch_end(sync_metrics=True)

    @torch.no_grad()
    def diarize(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        sample_rate: Optional[int] = None,
        batch_size: int = 1,
        include_tensor_outputs: bool = False,
        postprocessing_yaml: Optional[str] = None,
        num_workers: int = 0,
        verbose: bool = True,
        override_config: Optional[DiarizeConfig] = None,
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[torch.Tensor]]]:
        """One-click runner function for diarization.

        Args:
            audio: (a single or list) of paths to audio files or path to a manifest file.
            batch_size: (int) Batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            include_tensor_outputs: (bool) Include raw speaker activity probabilities to the output.
                See Returns: for more details.
            postprocessing_yaml: Optional(str) Path to .yaml file with postprocessing parameters.
            num_workers: (int) Number of workers for DataLoader.
            verbose: (bool) Whether to display tqdm progress bar.
            override_config: (Optional[DiarizeConfig]) A config to override the default config.

        Returns:
            If include_tensor_outputs is False: A list of lists of speech segments with a corresponding speaker index,
            in format "[begin_seconds, end_seconds, speaker_index]".
            If include_tensor_outputs is True: A tuple of the above list
            and list of tensors of raw speaker activity probabilities.
        """
        return super().diarize(
            audio=audio,
            sample_rate=sample_rate,
            batch_size=batch_size,
            include_tensor_outputs=include_tensor_outputs,
            postprocessing_yaml=postprocessing_yaml,
            num_workers=num_workers,
            verbose=verbose,
            override_config=override_config,
        )
