# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
import pickle as pkl
import tempfile
from collections import OrderedDict
from pathlib import Path
from statistics import mode
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pyannote.core import Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechMSDDInferDataset, AudioToSpeechMSDDTrainDataset
from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.models.clustering_diarizer import (
    _MODEL_CONFIG_YAML,
    _SPEAKER_MODEL,
    _VAD_MODEL,
    get_available_model_names,
)
from nemo.collections.asr.models.configs.diarizer_config import NeuralDiarizerInferenceConfig
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_embs_and_timestamps,
    get_id_tup_dict,
    get_scale_mapping_argmat,
    get_uniq_id_list_from_manifest,
    labels_to_pyannote_object,
    make_rttm_with_overlap,
    parse_scale_configs,
    rttm_to_labels,
)
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


__all__ = ['EncDecDiarLabelModel', 'ClusterEmbedding', 'NeuralDiarizer']


class EncDecDiarLabelModel(ModelPT, ExportableEncDecModel):
    """
    Encoder decoder class for multiscale diarization decoder (MSDD). Model class creates training, validation methods for setting
    up data performing model forward pass.

    This model class expects config dict for:
        * preprocessor
        * msdd_model
        * speaker_model
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []

        model = PretrainedModelInfo(
            pretrained_model_name="diar_msdd_telephonic",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/diar_msdd_telephonic/versions/1.0.1/files/diar_msdd_telephonic.nemo",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:diar_msdd_telephonic",
        )
        result.append(model)
        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initialize an MSDD model and the specified speaker embedding model. In this init function, training and validation datasets are prepared.
        """
        self._trainer = trainer if trainer else None
        self.cfg_msdd_model = cfg

        if self._trainer:
            self._init_segmentation_info()
            self.world_size = trainer.num_nodes * trainer.num_devices
            self.emb_batch_size = self.cfg_msdd_model.emb_batch_size
            self.pairwise_infer = False
        else:
            self.world_size = 1
            self.pairwise_infer = True
        super().__init__(cfg=self.cfg_msdd_model, trainer=trainer)

        window_length_in_sec = self.cfg_msdd_model.diarizer.speaker_embeddings.parameters.window_length_in_sec
        if isinstance(window_length_in_sec, int) or len(window_length_in_sec) <= 1:
            raise ValueError("window_length_in_sec should be a list containing multiple segment (window) lengths")
        else:
            self.cfg_msdd_model.scale_n = len(window_length_in_sec)
            self.cfg_msdd_model.msdd_module.scale_n = self.cfg_msdd_model.scale_n
            self.scale_n = self.cfg_msdd_model.scale_n

        self.preprocessor = EncDecSpeakerLabelModel.from_config_dict(self.cfg_msdd_model.preprocessor)
        self.frame_per_sec = int(1 / self.preprocessor._cfg.window_stride)
        self.msdd = EncDecDiarLabelModel.from_config_dict(self.cfg_msdd_model.msdd_module)

        if trainer is not None:
            self._init_speaker_model()
            self.add_speaker_model_config(cfg)
        else:
            self.msdd._speaker_model = EncDecSpeakerLabelModel.from_config_dict(cfg.speaker_model_cfg)

        # Call `self.save_hyperparameters` in modelPT.py again since cfg should contain speaker model's config.
        self.save_hyperparameters("cfg")

        self.loss = instantiate(self.cfg_msdd_model.loss)
        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()

    def add_speaker_model_config(self, cfg):
        """
        Add config dictionary of the speaker model to the model's config dictionary. This is required to
        save and load speaker model with MSDD model.

        Args:
            cfg (DictConfig): DictConfig type variable that conatains hyperparameters of MSDD model.
        """
        with open_dict(cfg):
            cfg_cp = copy.copy(self.msdd._speaker_model.cfg)
            cfg.speaker_model_cfg = cfg_cp
            del cfg.speaker_model_cfg.train_ds
            del cfg.speaker_model_cfg.validation_ds

    def _init_segmentation_info(self):
        """Initialize segmentation settings: window, shift and multiscale weights.
        """
        self._diarizer_params = self.cfg_msdd_model.diarizer
        self.multiscale_args_dict = parse_scale_configs(
            self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
        )

    def _init_speaker_model(self):
        """
        Initialize speaker embedding model with model name or path passed through config. Note that speaker embedding model is loaded to
        `self.msdd` to enable multi-gpu and multi-node training. In addition, speaker embedding model is also saved with msdd model when
        `.ckpt` files are saved.
        """
        model_path = self.cfg_msdd_model.diarizer.speaker_embeddings.model_path
        self._diarizer_params = self.cfg_msdd_model.diarizer

        if not torch.cuda.is_available():
            rank_id = torch.device('cpu')
        elif self._trainer:
            rank_id = torch.device(self._trainer.global_rank)
        else:
            rank_id = None

        if model_path is not None and model_path.endswith('.nemo'):
            self.msdd._speaker_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=rank_id)
            logging.info("Speaker Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            self._speaker_model = EncDecSpeakerLabelModel.load_from_checkpoint(model_path, map_location=rank_id)
            logging.info("Speaker Model restored locally from {}".format(model_path))
        else:
            if model_path not in get_available_model_names(EncDecSpeakerLabelModel):
                logging.warning(
                    "requested {} model name not available in pretrained models, instead".format(model_path)
                )
                model_path = "titanet_large"
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self.msdd._speaker_model = EncDecSpeakerLabelModel.from_pretrained(
                model_name=model_path, map_location=rank_id
            )
        self._speaker_params = self.cfg_msdd_model.diarizer.speaker_embeddings.parameters

    def __setup_dataloader_from_config(self, config):
        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=None
        )

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
        dataset = AudioToSpeechMSDDTrainDataset(
            manifest_filepath=config.manifest_filepath,
            emb_dir=config.emb_dir,
            multiscale_args_dict=self.multiscale_args_dict,
            soft_label_thres=config.soft_label_thres,
            featurizer=featurizer,
            window_stride=self.cfg_msdd_model.preprocessor.window_stride,
            emb_batch_size=config.emb_batch_size,
            pairwise_infer=False,
            global_rank=self._trainer.global_rank,
        )

        self.data_collection = dataset.collection
        collate_ds = dataset
        collate_fn = collate_ds.msdd_train_collate_fn
        batch_size = config['batch_size']
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def __setup_dataloader_from_config_infer(
        self, config: DictConfig, emb_dict: dict, emb_seq: dict, clus_label_dict: dict, pairwise_infer=False
    ):
        shuffle = config.get('shuffle', False)

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        dataset = AudioToSpeechMSDDInferDataset(
            manifest_filepath=config['manifest_filepath'],
            emb_dict=emb_dict,
            clus_label_dict=clus_label_dict,
            emb_seq=emb_seq,
            soft_label_thres=config.soft_label_thres,
            seq_eval_mode=config.seq_eval_mode,
            window_stride=self._cfg.preprocessor.window_stride,
            use_single_scale_clus=False,
            pairwise_infer=pairwise_infer,
        )
        self.data_collection = dataset.collection
        collate_ds = dataset
        collate_fn = collate_ds.msdd_infer_collate_fn
        batch_size = config['batch_size']
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(config=train_data_config,)

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self._validation_dl = self.__setup_dataloader_from_config(config=val_data_layer_config,)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if self.pairwise_infer:
            self._test_dl = self.__setup_dataloader_from_config_infer(
                config=test_data_config,
                emb_dict=self.emb_sess_test_dict,
                emb_seq=self.emb_seq_test,
                clus_label_dict=self.clus_test_label_dict,
                pairwise_infer=self.pairwise_infer,
            )

    def setup_multiple_test_data(self, test_data_config):
        """
        MSDD does not use multiple_test_data template. This function is a placeholder for preventing error.
        """
        return None

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "features": NeuralType(('B', 'T'), audio_eltype),
            "feature_length": NeuralType(('B',), LengthsType()),
            "ms_seg_timestamps": NeuralType(('B', 'C', 'T', 'D'), LengthsType()),
            "ms_seg_counts": NeuralType(('B', 'C'), LengthsType()),
            "clus_label_index": NeuralType(('B', 'T'), LengthsType()),
            "scale_mapping": NeuralType(('B', 'C', 'T'), LengthsType()),
            "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return OrderedDict(
            {
                "probs": NeuralType(('B', 'T', 'C'), ProbsType()),
                "scale_weights": NeuralType(('B', 'T', 'C', 'D'), ProbsType()),
            }
        )

    def get_ms_emb_seq(
        self, embs: torch.Tensor, scale_mapping: torch.Tensor, ms_seg_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Reshape the given tensor and organize the embedding sequence based on the original sequence counts.
        Repeat the embeddings according to the scale_mapping information so that the final embedding sequence has
        the identical length for all scales.

        Args:
            embs (Tensor):
                Merged embeddings without zero-padding in the batch. See `ms_seg_counts` for details.
                Shape: (Total number of segments in the batch, emb_dim)
            scale_mapping (Tensor):
		The element at the m-th row and the n-th column of the scale mapping matrix indicates the (m+1)-th scale
		segment index which has the closest center distance with (n+1)-th segment in the base scale.
		Example:
		    scale_mapping_argmat[2][101] = 85
		In the above example, it means that 86-th segment in the 3rd scale (python index is 2) is mapped with
		102-th segment in the base scale. Thus, the longer segments bound to have more repeating numbers since
		multiple base scale segments (since the base scale has the shortest length) fall into the range of the
		longer segments. At the same time, each row contains N numbers of indices where N is number of
		segments in the base-scale (i.e., the finest scale).
                Shape: (batch_size, scale_n, self.diar_window_length)
            ms_seg_counts (Tensor):
                Cumulative sum of the number of segments in each scale. This information is needed to reconstruct
                the multi-scale input matrix during forward propagating.

		Example: `batch_size=3, scale_n=6, emb_dim=192`
                    ms_seg_counts =  
                     [[8,  9, 12, 16, 25, 51],  
                      [11, 13, 14, 17, 25, 51],  
                      [ 9,  9, 11, 16, 23, 50]]  

		In this function, `ms_seg_counts` is used to get the actual length of each embedding sequence without
		zero-padding.

        Returns:
            ms_emb_seq (Tensor):
	        Multi-scale embedding sequence that is mapped, matched and repeated. The longer scales are less repeated,
                while shorter scales are more frequently repeated following the scale mapping tensor.
        """
        scale_n, batch_size = scale_mapping[0].shape[0], scale_mapping.shape[0]
        split_emb_tup = torch.split(embs, ms_seg_counts.view(-1).tolist(), dim=0)
        batch_emb_list = [split_emb_tup[i : i + scale_n] for i in range(0, len(split_emb_tup), scale_n)]
        ms_emb_seq_list = []
        for batch_idx in range(batch_size):
            feats_list = []
            for scale_index in range(scale_n):
                repeat_mat = scale_mapping[batch_idx][scale_index]
                feats_list.append(batch_emb_list[batch_idx][scale_index][repeat_mat, :])
            repp = torch.stack(feats_list).permute(1, 0, 2)
            ms_emb_seq_list.append(repp)
        ms_emb_seq = torch.stack(ms_emb_seq_list)
        return ms_emb_seq

    @torch.no_grad()
    def get_cluster_avg_embs_model(
        self, embs: torch.Tensor, clus_label_index: torch.Tensor, ms_seg_counts: torch.Tensor, scale_mapping
    ) -> torch.Tensor:
        """
        Calculate the cluster-average speaker embedding based on the ground-truth speaker labels (i.e., cluster labels).

        Args:
            embs (Tensor):
                Merged embeddings without zero-padding in the batch. See `ms_seg_counts` for details.
                Shape: (Total number of segments in the batch, emb_dim)
            clus_label_index (Tensor):
                Merged ground-truth cluster labels from all scales with zero-padding. Each scale's index can be
                retrieved by using segment index in `ms_seg_counts`.
                Shape: (batch_size, maximum total segment count among the samples in the batch)
            ms_seg_counts (Tensor):
                Cumulative sum of the number of segments in each scale. This information is needed to reconstruct
                multi-scale input tensors during forward propagating.

                Example: `batch_size=3, scale_n=6, emb_dim=192`
                    ms_seg_counts =  
                     [[8,  9, 12, 16, 25, 51],  
                      [11, 13, 14, 17, 25, 51],  
                      [ 9,  9, 11, 16, 23, 50]]  
                    Counts of merged segments: (121, 131, 118)  
                    embs has shape of (370, 192)  
                    clus_label_index has shape of (3, 131)  

                Shape: (batch_size, scale_n)

        Returns:
            ms_avg_embs (Tensor):
                Multi-scale cluster-average speaker embedding vectors. These embedding vectors are used as reference for
                each speaker to predict the speaker label for the given multi-scale embedding sequences.
                Shape: (batch_size, scale_n, emb_dim, self.num_spks_per_model)
        """
        scale_n, batch_size = scale_mapping[0].shape[0], scale_mapping.shape[0]
        split_emb_tup = torch.split(embs, ms_seg_counts.view(-1).tolist(), dim=0)
        batch_emb_list = [split_emb_tup[i : i + scale_n] for i in range(0, len(split_emb_tup), scale_n)]
        ms_avg_embs_list = []
        for batch_idx in range(batch_size):
            oracle_clus_idx = clus_label_index[batch_idx]
            max_seq_len = sum(ms_seg_counts[batch_idx])
            clus_label_index_batch = torch.split(oracle_clus_idx[:max_seq_len], ms_seg_counts[batch_idx].tolist())
            session_avg_emb_set_list = []
            for scale_index in range(scale_n):
                spk_set_list = []
                for idx in range(self.cfg_msdd_model.max_num_of_spks):
                    _where = (clus_label_index_batch[scale_index] == idx).clone().detach()
                    if not torch.any(_where):
                        avg_emb = torch.zeros(self.msdd._speaker_model._cfg.decoder.emb_sizes).to(embs.device)
                    else:
                        avg_emb = torch.mean(batch_emb_list[batch_idx][scale_index][_where], dim=0)
                    spk_set_list.append(avg_emb)
                session_avg_emb_set_list.append(torch.stack(spk_set_list))
            session_avg_emb_set = torch.stack(session_avg_emb_set_list)
            ms_avg_embs_list.append(session_avg_emb_set)

        ms_avg_embs = torch.stack(ms_avg_embs_list).permute(0, 1, 3, 2)
        ms_avg_embs = ms_avg_embs.float().detach().to(embs.device)
        assert (
            not ms_avg_embs.requires_grad
        ), "ms_avg_embs.requires_grad = True. ms_avg_embs should be detached from the torch graph."
        return ms_avg_embs

    @torch.no_grad()
    def get_ms_mel_feat(
        self,
        processed_signal: torch.Tensor,
        processed_signal_len: torch.Tensor,
        ms_seg_timestamps: torch.Tensor,
        ms_seg_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load acoustic feature from audio segments for each scale and save it into a torch.tensor matrix.
        In addition, create variables containing the information of the multiscale subsegmentation information.

        Note: `self.emb_batch_size` determines the number of embedding tensors attached to the computational graph.
        If `self.emb_batch_size` is greater than 0, speaker embedding models are simultaneosly trained. Due to the
        constrant of GPU memory size, only a subset of embedding tensors can be attached to the computational graph.
        By default, the graph-attached embeddings are selected randomly by `torch.randperm`. Default value of
        `self.emb_batch_size` is 0.

        Args:
            processed_signal (Tensor):
                Zero-padded Feature input.
                Shape: (batch_size, feat_dim, the longest feature sequence length)
            processed_signal_len (Tensor):
                The actual legnth of feature input without zero-padding.
                Shape: (batch_size,)
            ms_seg_timestamps (Tensor):
                Timestamps of the base-scale segments.
                Shape: (batch_size, scale_n, number of base-scale segments, self.num_spks_per_model)
            ms_seg_counts (Tensor):
                Cumulative sum of the number of segments in each scale. This information is needed to reconstruct
                the multi-scale input matrix during forward propagating.
                Shape: (batch_size, scale_n)

        Returns:
            ms_mel_feat (Tensor):
                Feature input stream split into the same length.
                Shape: (total number of segments, feat_dim, self.frame_per_sec * the-longest-scale-length)
            ms_mel_feat_len (Tensor):
                The actual length of feature without zero-padding.
                Shape: (total number of segments,)
            seq_len (Tensor):
                The length of the input embedding sequences.
                Shape: (total number of segments,)
            detach_ids (tuple):
                Tuple containing both detached embeding indices and attached embedding indices
        """
        device = processed_signal.device
        _emb_batch_size = min(self.emb_batch_size, ms_seg_counts.sum().item())
        feat_dim = self.preprocessor._cfg.features
        max_sample_count = int(self.multiscale_args_dict["scale_dict"][0][0] * self.frame_per_sec)
        ms_mel_feat_len_list, sequence_lengths_list, ms_mel_feat_list = [], [], []
        total_seg_count = torch.sum(ms_seg_counts)

        batch_size = processed_signal.shape[0]
        for batch_idx in range(batch_size):
            for scale_idx in range(self.scale_n):
                scale_seg_num = ms_seg_counts[batch_idx][scale_idx]
                for k, (stt, end) in enumerate(ms_seg_timestamps[batch_idx][scale_idx][:scale_seg_num]):
                    stt, end = int(stt.detach().item()), int(end.detach().item())
                    end = min(end, stt + max_sample_count)
                    _features = torch.zeros(feat_dim, max_sample_count).to(torch.float32).to(device)
                    _features[:, : (end - stt)] = processed_signal[batch_idx][:, stt:end]
                    ms_mel_feat_list.append(_features)
                    ms_mel_feat_len_list.append(end - stt)
            sequence_lengths_list.append(ms_seg_counts[batch_idx][-1])
        ms_mel_feat = torch.stack(ms_mel_feat_list).to(device)
        ms_mel_feat_len = torch.tensor(ms_mel_feat_len_list).to(device)
        seq_len = torch.tensor(sequence_lengths_list).to(device)

        if _emb_batch_size == 0:
            attached, _emb_batch_size = torch.tensor([]), 0
            detached = torch.arange(total_seg_count)
        else:
            torch.manual_seed(self._trainer.current_epoch)
            attached = torch.randperm(total_seg_count)[:_emb_batch_size]
            detached = torch.randperm(total_seg_count)[_emb_batch_size:]
        detach_ids = (attached, detached)
        return ms_mel_feat, ms_mel_feat_len, seq_len, detach_ids

    def forward_infer(self, input_signal, input_signal_length, emb_vectors, targets):
        """
        Wrapper function for inference case.
        """
        preds, scale_weights = self.msdd(
            ms_emb_seq=input_signal, length=input_signal_length, ms_avg_embs=emb_vectors, targets=targets
        )
        return preds, scale_weights

    @typecheck()
    def forward(
        self, features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets
    ):
        processed_signal, processed_signal_len = self.msdd._speaker_model.preprocessor(
            input_signal=features, length=feature_length
        )
        audio_signal, audio_signal_len, sequence_lengths, detach_ids = self.get_ms_mel_feat(
            processed_signal, processed_signal_len, ms_seg_timestamps, ms_seg_counts
        )

        # For detached embeddings
        with torch.no_grad():
            self.msdd._speaker_model.eval()
            logits, embs_d = self.msdd._speaker_model.forward_for_export(
                processed_signal=audio_signal[detach_ids[1]], processed_signal_len=audio_signal_len[detach_ids[1]]
            )
            embs = torch.zeros(audio_signal.shape[0], embs_d.shape[1]).to(embs_d.device)
            embs[detach_ids[1], :] = embs_d.detach()

        # For attached embeddings
        self.msdd._speaker_model.train()
        if len(detach_ids[0]) > 1:
            logits, embs_a = self.msdd._speaker_model.forward_for_export(
                processed_signal=audio_signal[detach_ids[0]], processed_signal_len=audio_signal_len[detach_ids[0]]
            )
            embs[detach_ids[0], :] = embs_a

        ms_emb_seq = self.get_ms_emb_seq(embs, scale_mapping, ms_seg_counts)
        ms_avg_embs = self.get_cluster_avg_embs_model(embs, clus_label_index, ms_seg_counts, scale_mapping)
        preds, scale_weights = self.msdd(
            ms_emb_seq=ms_emb_seq, length=sequence_lengths, ms_avg_embs=ms_avg_embs, targets=targets
        )
        return preds, scale_weights

    def training_step(self, batch: list, batch_idx: int):
        features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets = batch
        sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts.detach()])
        preds, _ = self.forward(
            features=features,
            feature_length=feature_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            clus_label_index=clus_label_index,
            scale_mapping=scale_mapping,
            targets=targets,
        )
        loss = self.loss(probs=preds, labels=targets, signal_lengths=sequence_lengths)
        self._accuracy_train(preds, targets, sequence_lengths)
        torch.cuda.empty_cache()
        f1_acc = self._accuracy_train.compute()
        self.log('loss', loss, sync_dist=True)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'], sync_dist=True)
        self.log('train_f1_acc', f1_acc, sync_dist=True)
        self._accuracy_train.reset()
        return {'loss': loss}

    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets = batch
        sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts])
        preds, _ = self.forward(
            features=features,
            feature_length=feature_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            clus_label_index=clus_label_index,
            scale_mapping=scale_mapping,
            targets=targets,
        )
        loss = self.loss(probs=preds, labels=targets, signal_lengths=sequence_lengths)
        self._accuracy_valid(preds, targets, sequence_lengths)
        f1_acc = self._accuracy_valid.compute()
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_f1_acc', f1_acc, sync_dist=True)
        return {
            'val_loss': loss,
            'val_f1_acc': f1_acc,
        }

    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        f1_acc = self._accuracy_valid.compute()
        self._accuracy_valid.reset()

        self.log('val_loss', val_loss_mean, sync_dist=True)
        self.log('val_f1_acc', f1_acc, sync_dist=True)
        return {
            'val_loss': val_loss_mean,
            'val_f1_acc': f1_acc,
        }

    def multi_test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        f1_acc = self._accuracy_test.compute()
        self._accuracy_test.reset()
        self.log('test_f1_acc', f1_acc, sync_dist=True)
        return {
            'test_loss': test_loss_mean,
            'test_f1_acc': f1_acc,
        }

    def compute_accuracies(self):
        """
        Calculate F1 score and accuracy of the predicted sigmoid values.

        Returns:
            f1_score (float):
                F1 score of the estimated diarized speaker label sequences.
            simple_acc (float):
                Accuracy of predicted speaker labels: (total # of correct labels)/(total # of sigmoid values)
        """
        f1_score = self._accuracy_test.compute()
        num_correct = torch.sum(self._accuracy_test.true.bool())
        total_count = torch.prod(torch.tensor(self._accuracy_test.targets.shape))
        simple_acc = num_correct / total_count
        return f1_score, simple_acc


class ClusterEmbedding(torch.nn.Module):
    """
    This class is built for calculating cluster-average embeddings, segmentation and load/save of the estimated cluster labels.
    The methods in this class is used for the inference of MSDD models.

    Args:
        cfg_diar_infer (DictConfig):
            Config dictionary from diarization inference YAML file
        cfg_msdd_model (DictConfig):
            Config dictionary from MSDD model checkpoint file

    Class Variables:
        self.cfg_diar_infer (DictConfig):
            Config dictionary from diarization inference YAML file
        cfg_msdd_model (DictConfig):
            Config dictionary from MSDD model checkpoint file
        self._speaker_model (class `EncDecSpeakerLabelModel`):
            This is a placeholder for class instance of `EncDecSpeakerLabelModel`
        self.scale_window_length_list (list):
            List containing the window lengths (i.e., scale length) of each scale.
        self.scale_n (int):
            Number of scales for multi-scale clustering diarizer
        self.base_scale_index (int):
            The index of the base-scale which is the shortest scale among the given multiple scales
    """

    def __init__(
        self, cfg_diar_infer: DictConfig, cfg_msdd_model: DictConfig, speaker_model: Optional[EncDecSpeakerLabelModel]
    ):
        super().__init__()
        self.cfg_diar_infer = cfg_diar_infer
        self._cfg_msdd = cfg_msdd_model
        self._speaker_model = speaker_model
        self.scale_window_length_list = list(
            self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.window_length_in_sec
        )
        self.scale_n = len(self.scale_window_length_list)
        self.base_scale_index = len(self.scale_window_length_list) - 1
        self.clus_diar_model = ClusteringDiarizer(cfg=self.cfg_diar_infer, speaker_model=self._speaker_model)

    def prepare_cluster_embs_infer(self):
        """
        Launch clustering diarizer to prepare embedding vectors and clustering results.
        """
        self.max_num_speakers = self.cfg_diar_infer.diarizer.clustering.parameters.max_num_speakers
        self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict, _ = self.run_clustering_diarizer(
            self._cfg_msdd.test_ds.manifest_filepath, self._cfg_msdd.test_ds.emb_dir
        )

    def assign_labels_to_longer_segs(self, base_clus_label_dict: Dict, session_scale_mapping_dict: Dict):
        """
        In multi-scale speaker diarization system, clustering result is solely based on the base-scale (the shortest scale).
        To calculate cluster-average speaker embeddings for each scale that are longer than the base-scale, this function assigns
        clustering results for the base-scale to the longer scales by measuring the distance between subsegment timestamps in the
        base-scale and non-base-scales.

        Args:
            base_clus_label_dict (dict):
                Dictionary containing clustering results for base-scale segments. Indexed by `uniq_id` string.
            session_scale_mapping_dict (dict):
                Dictionary containing multiscale mapping information for each session. Indexed by `uniq_id` string.

        Returns:
            all_scale_clus_label_dict (dict):
                Dictionary containing clustering labels of all scales. Indexed by scale_index in integer format.

        """
        all_scale_clus_label_dict = {scale_index: {} for scale_index in range(self.scale_n)}
        for uniq_id, uniq_scale_mapping_dict in session_scale_mapping_dict.items():
            base_scale_clus_label = np.array([x[-1] for x in base_clus_label_dict[uniq_id]])
            all_scale_clus_label_dict[self.base_scale_index][uniq_id] = base_scale_clus_label
            for scale_index in range(self.scale_n - 1):
                new_clus_label = []
                assert (
                    uniq_scale_mapping_dict[scale_index].shape[0] == base_scale_clus_label.shape[0]
                ), "The number of base scale labels does not match the segment numbers in uniq_scale_mapping_dict"
                max_index = max(uniq_scale_mapping_dict[scale_index])
                for seg_idx in range(max_index + 1):
                    if seg_idx in uniq_scale_mapping_dict[scale_index]:
                        seg_clus_label = mode(base_scale_clus_label[uniq_scale_mapping_dict[scale_index] == seg_idx])
                    else:
                        seg_clus_label = 0 if len(new_clus_label) == 0 else new_clus_label[-1]
                    new_clus_label.append(seg_clus_label)
                all_scale_clus_label_dict[scale_index][uniq_id] = new_clus_label
        return all_scale_clus_label_dict

    def get_base_clus_label_dict(self, clus_labels: List[str], emb_scale_seq_dict: Dict[int, dict]):
        """
        Retrieve base scale clustering labels from `emb_scale_seq_dict`.

        Args:
            clus_labels (list):
                List containing cluster results generated by clustering diarizer.
            emb_scale_seq_dict (dict):
                Dictionary containing multiscale embedding input sequences.
        Returns:
            base_clus_label_dict (dict):
                Dictionary containing start and end of base scale segments and its cluster label. Indexed by `uniq_id`.
            emb_dim (int):
                Embedding dimension in integer.
        """
        base_clus_label_dict = {key: [] for key in emb_scale_seq_dict[self.base_scale_index].keys()}
        for line in clus_labels:
            uniq_id = line.split()[0]
            label = int(line.split()[-1].split('_')[-1])
            stt, end = [round(float(x), 2) for x in line.split()[1:3]]
            base_clus_label_dict[uniq_id].append([stt, end, label])
        emb_dim = emb_scale_seq_dict[0][uniq_id][0].shape[0]
        return base_clus_label_dict, emb_dim

    def get_cluster_avg_embs(
        self, emb_scale_seq_dict: Dict, clus_labels: List, speaker_mapping_dict: Dict, session_scale_mapping_dict: Dict
    ):
        """
        MSDD requires cluster-average speaker embedding vectors for each scale. This function calculates an average embedding vector for each cluster (speaker)
        and each scale.

        Args:
            emb_scale_seq_dict (dict):
                Dictionary containing embedding sequence for each scale. Keys are scale index in integer.
            clus_labels (list):
                Clustering results from clustering diarizer including all the sessions provided in input manifest files.
            speaker_mapping_dict (dict):
                Speaker mapping dictionary in case RTTM files are provided. This is mapping between integer based speaker index and
                speaker ID tokens in RTTM files.
                Example:
                    {'en_0638': {'speaker_0': 'en_0638_A', 'speaker_1': 'en_0638_B'},
                     'en_4065': {'speaker_0': 'en_4065_B', 'speaker_1': 'en_4065_A'}, ...,}
            session_scale_mapping_dict (dict):
                Dictionary containing multiscale mapping information for each session. Indexed by `uniq_id` string.

        Returns:
            emb_sess_avg_dict (dict):
                Dictionary containing speaker mapping information and cluster-average speaker embedding vector.
                Each session-level dictionary is indexed by scale index in integer.
            output_clus_label_dict (dict):
                Subegmentation timestamps in float type and Clustering result in integer type. Indexed by `uniq_id` keys.
        """
        self.scale_n = len(emb_scale_seq_dict.keys())
        emb_sess_avg_dict = {
            scale_index: {key: [] for key in emb_scale_seq_dict[self.scale_n - 1].keys()}
            for scale_index in emb_scale_seq_dict.keys()
        }
        output_clus_label_dict, emb_dim = self.get_base_clus_label_dict(clus_labels, emb_scale_seq_dict)
        all_scale_clus_label_dict = self.assign_labels_to_longer_segs(
            output_clus_label_dict, session_scale_mapping_dict
        )
        for scale_index in emb_scale_seq_dict.keys():
            for uniq_id, _emb_tensor in emb_scale_seq_dict[scale_index].items():
                if type(_emb_tensor) == list:
                    emb_tensor = torch.tensor(np.array(_emb_tensor))
                else:
                    emb_tensor = _emb_tensor
                clus_label_list = all_scale_clus_label_dict[scale_index][uniq_id]
                spk_set = set(clus_label_list)

                # Create a label array which identifies clustering result for each segment.
                label_array = torch.Tensor(clus_label_list)
                avg_embs = torch.zeros(emb_dim, self.max_num_speakers)
                for spk_idx in spk_set:
                    selected_embs = emb_tensor[label_array == spk_idx]
                    avg_embs[:, spk_idx] = torch.mean(selected_embs, dim=0)

                if speaker_mapping_dict is not None:
                    inv_map = {clus_key: rttm_key for rttm_key, clus_key in speaker_mapping_dict[uniq_id].items()}
                else:
                    inv_map = None

                emb_sess_avg_dict[scale_index][uniq_id] = {'mapping': inv_map, 'avg_embs': avg_embs}
        return emb_sess_avg_dict, output_clus_label_dict

    def run_clustering_diarizer(self, manifest_filepath: str, emb_dir: str):
        """
        If no pre-existing data is provided, run clustering diarizer from scratch. This will create scale-wise speaker embedding
        sequence, cluster-average embeddings, scale mapping and base scale clustering labels. Note that speaker embedding `state_dict`
        is loaded from the `state_dict` in the provided MSDD checkpoint.

        Args:
            manifest_filepath (str):
                Input manifest file for creating audio-to-RTTM mapping.
            emb_dir (str):
                Output directory where embedding files and timestamp files are saved.

        Returns:
            emb_sess_avg_dict (dict):
                Dictionary containing cluster-average embeddings for each session.
            emb_scale_seq_dict (dict):
                Dictionary containing embedding tensors which are indexed by scale numbers.
            base_clus_label_dict (dict):
                Dictionary containing clustering results. Clustering results are cluster labels for the base scale segments.
        """
        self.cfg_diar_infer.diarizer.manifest_filepath = manifest_filepath
        self.cfg_diar_infer.diarizer.out_dir = emb_dir

        # Run ClusteringDiarizer which includes system VAD or oracle VAD.
        self._out_dir = self.clus_diar_model._diarizer_params.out_dir
        self.out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(self.out_rttm_dir, exist_ok=True)

        self.clus_diar_model._cluster_params = self.cfg_diar_infer.diarizer.clustering.parameters
        self.clus_diar_model.multiscale_args_dict[
            "multiscale_weights"
        ] = self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.multiscale_weights
        self.clus_diar_model._diarizer_params.speaker_embeddings.parameters = (
            self.cfg_diar_infer.diarizer.speaker_embeddings.parameters
        )
        cluster_params = self.clus_diar_model._cluster_params
        cluster_params = dict(cluster_params) if isinstance(cluster_params, DictConfig) else cluster_params.dict()
        clustering_params_str = json.dumps(cluster_params, indent=4)

        logging.info(f"Multiscale Weights: {self.clus_diar_model.multiscale_args_dict['multiscale_weights']}")
        logging.info(f"Clustering Parameters: {clustering_params_str}")
        scores = self.clus_diar_model.diarize(batch_size=self.cfg_diar_infer.batch_size)

        # If RTTM (ground-truth diarization annotation) files do not exist, scores is None.
        if scores is not None:
            metric, speaker_mapping_dict, _ = scores
        else:
            metric, speaker_mapping_dict = None, None

        # Get the mapping between segments in different scales.
        self._embs_and_timestamps = get_embs_and_timestamps(
            self.clus_diar_model.multiscale_embeddings_and_timestamps, self.clus_diar_model.multiscale_args_dict
        )
        session_scale_mapping_dict = self.get_scale_map(self._embs_and_timestamps)
        emb_scale_seq_dict = self.load_emb_scale_seq_dict(emb_dir)
        clus_labels = self.load_clustering_labels(emb_dir)
        emb_sess_avg_dict, base_clus_label_dict = self.get_cluster_avg_embs(
            emb_scale_seq_dict, clus_labels, speaker_mapping_dict, session_scale_mapping_dict
        )
        emb_scale_seq_dict['session_scale_mapping'] = session_scale_mapping_dict
        return emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict, metric

    def get_scale_map(self, embs_and_timestamps):
        """
        Save multiscale mapping data into dictionary format.

        Args:
            embs_and_timestamps (dict):
                Dictionary containing embedding tensors and timestamp tensors. Indexed by `uniq_id` string.
        Returns:
            session_scale_mapping_dict (dict):
                Dictionary containing multiscale mapping information for each session. Indexed by `uniq_id` string.
        """
        session_scale_mapping_dict = {}
        for uniq_id, uniq_embs_and_timestamps in embs_and_timestamps.items():
            scale_mapping_dict = get_scale_mapping_argmat(uniq_embs_and_timestamps)
            session_scale_mapping_dict[uniq_id] = scale_mapping_dict
        return session_scale_mapping_dict

    def check_clustering_labels(self, out_dir):
        """
        Check whether the laoded clustering label file is including clustering results for all sessions.
        This function is used for inference mode of MSDD.

        Args:
            out_dir (str):
                Path to the directory where clustering result files are saved.
        Returns:
            file_exists (bool):
                Boolean that indicates whether clustering result file exists.
            clus_label_path (str):
                Path to the clustering label output file.
        """
        clus_label_path = os.path.join(
            out_dir, 'speaker_outputs', f'subsegments_scale{self.base_scale_index}_cluster.label'
        )
        file_exists = os.path.exists(clus_label_path)
        if not file_exists:
            logging.info(f"Clustering label file {clus_label_path} does not exist.")
        return file_exists, clus_label_path

    def load_clustering_labels(self, out_dir):
        """
        Load clustering labels generated by clustering diarizer. This function is used for inference mode of MSDD.

        Args:
            out_dir (str):
                Path to the directory where clustering result files are saved.
        Returns:
            emb_scale_seq_dict (dict):
                List containing clustering results in string format.
        """
        file_exists, clus_label_path = self.check_clustering_labels(out_dir)
        logging.info(f"Loading cluster label file from {clus_label_path}")
        with open(clus_label_path) as f:
            clus_labels = f.readlines()
        return clus_labels

    def load_emb_scale_seq_dict(self, out_dir):
        """
        Load saved embeddings generated by clustering diarizer. This function is used for inference mode of MSDD.

        Args:
            out_dir (str):
                Path to the directory where embedding pickle files are saved.
        Returns:
            emb_scale_seq_dict (dict):
                Dictionary containing embedding tensors which are indexed by scale numbers.
        """
        window_len_list = list(self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.window_length_in_sec)
        emb_scale_seq_dict = {scale_index: None for scale_index in range(len(window_len_list))}
        for scale_index in range(len(window_len_list)):
            pickle_path = os.path.join(
                out_dir, 'speaker_outputs', 'embeddings', f'subsegments_scale{scale_index}_embeddings.pkl'
            )
            logging.info(f"Loading embedding pickle file of scale:{scale_index} at {pickle_path}")
            with open(pickle_path, "rb") as input_file:
                emb_dict = pkl.load(input_file)
            for key, val in emb_dict.items():
                emb_dict[key] = val
            emb_scale_seq_dict[scale_index] = emb_dict
        return emb_scale_seq_dict


class NeuralDiarizer(LightningModule):
    """
    Class for inference based on multiscale diarization decoder (MSDD). MSDD requires initializing clustering results from
    clustering diarizer. Overlap-aware diarizer requires separate RTTM generation and evaluation modules to check the effect of
    overlap detection in speaker diarization.
    """

    def __init__(self, cfg: Union[DictConfig, NeuralDiarizerInferenceConfig]):
        super().__init__()
        self._cfg = cfg

        # Parameter settings for MSDD model
        self.use_speaker_model_from_ckpt = cfg.diarizer.msdd_model.parameters.get('use_speaker_model_from_ckpt', True)
        self.use_clus_as_main = cfg.diarizer.msdd_model.parameters.get('use_clus_as_main', False)
        self.max_overlap_spks = cfg.diarizer.msdd_model.parameters.get('max_overlap_spks', 2)
        self.num_spks_per_model = cfg.diarizer.msdd_model.parameters.get('num_spks_per_model', 2)
        self.use_adaptive_thres = cfg.diarizer.msdd_model.parameters.get('use_adaptive_thres', True)
        self.max_pred_length = cfg.diarizer.msdd_model.parameters.get('max_pred_length', 0)
        self.diar_eval_settings = cfg.diarizer.msdd_model.parameters.get(
            'diar_eval_settings', [(0.25, True), (0.25, False), (0.0, False)]
        )

        self._init_msdd_model(cfg)
        self.diar_window_length = cfg.diarizer.msdd_model.parameters.diar_window_length
        self.msdd_model.cfg = self.transfer_diar_params_to_model_params(self.msdd_model, cfg)

        # Initialize clustering and embedding preparation instance (as a diarization encoder).
        self.clustering_embedding = ClusterEmbedding(
            cfg_diar_infer=cfg, cfg_msdd_model=self.msdd_model.cfg, speaker_model=self._speaker_model
        )

        # Parameters for creating diarization results from MSDD outputs.
        self.clustering_max_spks = self.msdd_model._cfg.max_num_of_spks
        self.overlap_infer_spk_limit = cfg.diarizer.msdd_model.parameters.get(
            'overlap_infer_spk_limit', self.clustering_max_spks
        )

    def transfer_diar_params_to_model_params(self, msdd_model, cfg):
        """
        Transfer the parameters that are needed for MSDD inference from the diarization inference config files
        to MSDD model config `msdd_model.cfg`.
        """
        msdd_model.cfg.diarizer.out_dir = cfg.diarizer.out_dir
        msdd_model.cfg.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        msdd_model.cfg.test_ds.emb_dir = cfg.diarizer.out_dir
        msdd_model.cfg.test_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size
        msdd_model.cfg.test_ds.seq_eval_mode = cfg.diarizer.msdd_model.parameters.seq_eval_mode
        msdd_model._cfg.max_num_of_spks = cfg.diarizer.clustering.parameters.max_num_speakers
        return msdd_model.cfg

    @rank_zero_only
    def save_to(self, save_path: str):
        """
        Saves model instances (weights and configuration) into EFF archive.
        You can use "restore_from" method to fully restore instance from .nemo file.

        .nemo file is an archive (tar.gz) with the following:
            model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
            model_wights.chpt - model checkpoint

        Args:
            save_path: Path to .nemo file where model instance should be saved
        """
        self.clus_diar = self.clustering_embedding.clus_diar_model
        _NEURAL_DIAR_MODEL = "msdd_model.nemo"

        with tempfile.TemporaryDirectory() as tmpdir:
            config_yaml = os.path.join(tmpdir, _MODEL_CONFIG_YAML)
            spkr_model = os.path.join(tmpdir, _SPEAKER_MODEL)
            neural_diar_model = os.path.join(tmpdir, _NEURAL_DIAR_MODEL)

            self.clus_diar.to_config_file(path2yaml_file=config_yaml)
            if self.clus_diar.has_vad_model:
                vad_model = os.path.join(tmpdir, _VAD_MODEL)
                self.clus_diar._vad_model.save_to(vad_model)
            self.clus_diar._speaker_model.save_to(spkr_model)
            self.msdd_model.save_to(neural_diar_model)
            self.clus_diar.__make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    def extract_standalone_speaker_model(self, prefix: str = 'msdd._speaker_model.') -> EncDecSpeakerLabelModel:
        """
        MSDD model file contains speaker embedding model and MSDD model. This function extracts standalone speaker model and save it to
        `self.spk_emb_state_dict` to be loaded separately for clustering diarizer.

        Args:
            ext (str):
                File-name extension of the provided model path.
        Returns:
            standalone_model_path (str):
                Path to the extracted standalone model without speaker embedding extractor model.
        """
        model_state_dict = self.msdd_model.state_dict()
        spk_emb_module_names = []
        for name in model_state_dict.keys():
            if prefix in name:
                spk_emb_module_names.append(name)

        spk_emb_state_dict = {}
        for name in spk_emb_module_names:
            org_name = name.replace(prefix, '')
            spk_emb_state_dict[org_name] = model_state_dict[name]

        _speaker_model = EncDecSpeakerLabelModel.from_config_dict(self.msdd_model.cfg.speaker_model_cfg)
        _speaker_model.load_state_dict(spk_emb_state_dict)
        return _speaker_model

    def _init_msdd_model(self, cfg: Union[DictConfig, NeuralDiarizerInferenceConfig]):

        """
        Initialized MSDD model with the provided config. Load either from `.nemo` file or `.ckpt` checkpoint files.
        """
        model_path = cfg.diarizer.msdd_model.model_path
        if model_path.endswith('.nemo'):
            logging.info(f"Using local nemo file from {model_path}")
            self.msdd_model = EncDecDiarLabelModel.restore_from(restore_path=model_path, map_location=cfg.device)
        elif model_path.endswith('.ckpt'):
            logging.info(f"Using local checkpoint from {model_path}")
            self.msdd_model = EncDecDiarLabelModel.load_from_checkpoint(
                checkpoint_path=model_path, map_location=cfg.device
            )
        else:
            if model_path not in get_available_model_names(EncDecDiarLabelModel):
                logging.warning(f"requested {model_path} model name not available in pretrained models, instead")
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self.msdd_model = EncDecDiarLabelModel.from_pretrained(model_name=model_path, map_location=cfg.device)
        # Load speaker embedding model state_dict which is loaded from the MSDD checkpoint.
        if self.use_speaker_model_from_ckpt:
            self._speaker_model = self.extract_standalone_speaker_model()
        else:
            self._speaker_model = None

    def get_pred_mat(self, data_list: List[Union[Tuple[int], List[torch.Tensor]]]) -> torch.Tensor:
        """
        This module puts together the pairwise, two-speaker, predicted results to form a finalized matrix that has dimension of
        `(total_len, n_est_spks)`. The pairwise results are evenutally averaged. For example, in 4 speaker case (speaker 1, 2, 3, 4),
        the sum of the pairwise results (1, 2), (1, 3), (1, 4) are then divided by 3 to take average of the sigmoid values.

        Args:
            data_list (list):
                List containing data points from `test_data_collection` variable. `data_list` has sublists `data` as follows:
                data[0]: `target_spks` tuple
                    Examples: (0, 1, 2)
                data[1]: Tensor containing estimaged sigmoid values.
                   [[0.0264, 0.9995],
                    [0.0112, 1.0000],
                    ...,
                    [1.0000, 0.0512]]

        Returns:
            sum_pred (Tensor):
                Tensor containing the averaged sigmoid values for each speaker.
        """
        all_tups = tuple()
        for data in data_list:
            all_tups += data[0]
        n_est_spks = len(set(all_tups))
        digit_map = dict(zip(sorted(set(all_tups)), range(n_est_spks)))
        total_len = max([sess[1].shape[1] for sess in data_list])
        sum_pred = torch.zeros(total_len, n_est_spks)
        for (_dim_tup, pred_mat) in data_list:
            dim_tup = [digit_map[x] for x in _dim_tup]
            if len(pred_mat.shape) == 3:
                pred_mat = pred_mat.squeeze(0)
            if n_est_spks <= self.num_spks_per_model:
                sum_pred = pred_mat
            else:
                _end = pred_mat.shape[0]
                sum_pred[:_end, dim_tup] += pred_mat.cpu().float()
        sum_pred = sum_pred / (n_est_spks - 1)
        return sum_pred

    def get_integrated_preds_list(
        self, uniq_id_list: List[str], test_data_collection: List[Any], preds_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Merge multiple sequence inference outputs into a session level result.

        Args:
            uniq_id_list (list):
                List containing `uniq_id` values.
            test_data_collection (collections.DiarizationLabelEntity):
                Class instance that is containing session information such as targeted speaker indices, audio filepaths and RTTM filepaths.
            preds_list (list):
                List containing tensors filled with sigmoid values.

        Returns:
            output_list (list):
                List containing session-level estimated prediction matrix.
        """
        session_dict = get_id_tup_dict(uniq_id_list, test_data_collection, preds_list)
        output_dict = {uniq_id: [] for uniq_id in uniq_id_list}
        for uniq_id, data_list in session_dict.items():
            sum_pred = self.get_pred_mat(data_list)
            output_dict[uniq_id] = sum_pred.unsqueeze(0)
        output_list = [output_dict[uniq_id] for uniq_id in uniq_id_list]
        return output_list

    def get_emb_clus_infer(self, cluster_embeddings):
        """Assign dictionaries containing the clustering results from the class instance `cluster_embeddings`.
        """
        self.msdd_model.emb_sess_test_dict = cluster_embeddings.emb_sess_test_dict
        self.msdd_model.clus_test_label_dict = cluster_embeddings.clus_test_label_dict
        self.msdd_model.emb_seq_test = cluster_embeddings.emb_seq_test

    @torch.no_grad()
    def diarize(self) -> Optional[List[Optional[List[Tuple[DiarizationErrorRate, Dict]]]]]:
        """
        Launch diarization pipeline which starts from VAD (or a oracle VAD stamp generation), initialization clustering and multiscale diarization decoder (MSDD).
        Note that the result of MSDD can include multiple speakers at the same time. Therefore, RTTM output of MSDD needs to be based on `make_rttm_with_overlap()`
        function that can generate overlapping timestamps. `self.run_overlap_aware_eval()` function performs DER evaluation.
        """
        self.clustering_embedding.prepare_cluster_embs_infer()
        self.msdd_model.pairwise_infer = True
        self.get_emb_clus_infer(self.clustering_embedding)
        preds_list, targets_list, signal_lengths_list = self.run_pairwise_diarization()
        thresholds = list(self._cfg.diarizer.msdd_model.parameters.sigmoid_threshold)
        return [self.run_overlap_aware_eval(preds_list, threshold) for threshold in thresholds]

    def get_range_average(
        self, signals: torch.Tensor, emb_vectors: torch.Tensor, diar_window_index: int, test_data_collection: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        This function is only used when `split_infer=True`. This module calculates cluster-average embeddings for the given short range.
        The range length is set by `self.diar_window_length`, and each cluster-average is only calculated for the specified range.
        Note that if the specified range does not contain some speakers (e.g. the range contains speaker 1, 3) compared to the global speaker sets
        (e.g. speaker 1, 2, 3, 4) then the missing speakers (e.g. speakers 2, 4) are assigned with zero-filled cluster-average speaker embedding.

        Args:
            signals (Tensor):
                Zero-padded Input multi-scale embedding sequences.
                Shape: (length, scale_n, emb_vectors, emb_dim)
            emb_vectors (Tensor):
                Cluster-average multi-scale embedding vectors.
                Shape: (length, scale_n, emb_vectors, emb_dim)
            diar_window_index (int):
                Index of split diarization wondows.
            test_data_collection (collections.DiarizationLabelEntity)
                Class instance that is containing session information such as targeted speaker indices, audio filepath and RTTM filepath.

        Returns:
            return emb_vectors_split (Tensor):
                Cluster-average speaker embedding vectors for each scale.
            emb_seq (Tensor):
                Zero-padded multi-scale embedding sequences.
            seq_len (int):
                Length of the sequence determined by `self.diar_window_length` variable.
        """
        emb_vectors_split = torch.zeros_like(emb_vectors)
        uniq_id = os.path.splitext(os.path.basename(test_data_collection.audio_file))[0]
        clus_label_tensor = torch.tensor([x[-1] for x in self.msdd_model.clus_test_label_dict[uniq_id]])
        for spk_idx in range(len(test_data_collection.target_spks)):
            stt, end = (
                diar_window_index * self.diar_window_length,
                min((diar_window_index + 1) * self.diar_window_length, clus_label_tensor.shape[0]),
            )
            seq_len = end - stt
            if stt < clus_label_tensor.shape[0]:
                target_clus_label_tensor = clus_label_tensor[stt:end]
                emb_seq, seg_length = (
                    signals[stt:end, :, :],
                    min(
                        self.diar_window_length,
                        clus_label_tensor.shape[0] - diar_window_index * self.diar_window_length,
                    ),
                )
                target_clus_label_bool = target_clus_label_tensor == test_data_collection.target_spks[spk_idx]

                # There are cases where there is no corresponding speaker in split range, so any(target_clus_label_bool) could be False.
                if any(target_clus_label_bool):
                    emb_vectors_split[:, :, spk_idx] = torch.mean(emb_seq[target_clus_label_bool], dim=0)

                # In case when the loop reaches the end of the sequence
                if seq_len < self.diar_window_length:
                    emb_seq = torch.cat(
                        [
                            emb_seq,
                            torch.zeros(self.diar_window_length - seq_len, emb_seq.shape[1], emb_seq.shape[2]).to(
                                signals.device
                            ),
                        ],
                        dim=0,
                    )
            else:
                emb_seq = torch.zeros(self.diar_window_length, emb_vectors.shape[0], emb_vectors.shape[1]).to(
                    signals.device
                )
                seq_len = 0
        return emb_vectors_split, emb_seq, seq_len

    def get_range_clus_avg_emb(
        self, test_batch: List[torch.Tensor], _test_data_collection: List[Any], device: torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This function is only used when `get_range_average` function is called. This module calculates cluster-average embeddings for
        the given short range. The range length is set by `self.diar_window_length`, and each cluster-average is only calculated for the specified range.

        Args:
            test_batch: (list)
                List containing embedding sequences, length of embedding sequences, ground truth labels (if exists) and initializing embedding vectors.
            test_data_collection: (list)
                List containing test-set dataloader contents. test_data_collection includes wav file path, RTTM file path, clustered speaker indices.

        Returns:
            sess_emb_vectors (Tensor):
                Tensor of cluster-average speaker embedding vectors.
                Shape: (batch_size, scale_n, emb_dim, 2*num_of_spks)
            sess_emb_seq (Tensor):
                Tensor of input multi-scale embedding sequences.
                Shape: (batch_size, length, scale_n, emb_dim)
            sess_sig_lengths (Tensor):
                Tensor of the actucal sequence length without zero-padding.
                Shape: (batch_size)
        """
        _signals, signal_lengths, _targets, _emb_vectors = test_batch
        sess_emb_vectors, sess_emb_seq, sess_sig_lengths = [], [], []
        split_count = torch.ceil(torch.tensor(_signals.shape[1] / self.diar_window_length)).int()
        self.max_pred_length = max(self.max_pred_length, self.diar_window_length * split_count)
        for k in range(_signals.shape[0]):
            signals, emb_vectors, test_data_collection = _signals[k], _emb_vectors[k], _test_data_collection[k]
            for diar_window_index in range(split_count):
                emb_vectors_split, emb_seq, seq_len = self.get_range_average(
                    signals, emb_vectors, diar_window_index, test_data_collection
                )
                sess_emb_vectors.append(emb_vectors_split)
                sess_emb_seq.append(emb_seq)
                sess_sig_lengths.append(seq_len)
        sess_emb_vectors = torch.stack(sess_emb_vectors).to(device)
        sess_emb_seq = torch.stack(sess_emb_seq).to(device)
        sess_sig_lengths = torch.tensor(sess_sig_lengths).to(device)
        return sess_emb_vectors, sess_emb_seq, sess_sig_lengths

    def diar_infer(
        self, test_batch: List[torch.Tensor], test_data_collection: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Launch forward_infer() function by feeding the session-wise embedding sequences to get pairwise speaker prediction values.
        If split_infer is True, the input audio clips are broken into short sequences then cluster average embeddings are calculated
        for inference. Split-infer might result in an improved results if calculating clustering average on the shorter tim-espan can
        help speaker assignment.

        Args:
            test_batch: (list)
                List containing embedding sequences, length of embedding sequences, ground truth labels (if exists) and initializing embedding vectors.
            test_data_collection: (list)
                List containing test-set dataloader contents. test_data_collection includes wav file path, RTTM file path, clustered speaker indices.

        Returns:
            preds (Tensor):
                Tensor containing predicted values which are generated from MSDD model.
            targets (Tensor):
                Tensor containing binary ground-truth values.
            signal_lengths (Tensor):
                The actual Session length (number of steps = number of base-scale segments) without zero padding.
        """
        signals, signal_lengths, _targets, emb_vectors = test_batch
        if self._cfg.diarizer.msdd_model.parameters.split_infer:
            split_count = torch.ceil(torch.tensor(signals.shape[1] / self.diar_window_length)).int()
            sess_emb_vectors, sess_emb_seq, sess_sig_lengths = self.get_range_clus_avg_emb(
                test_batch, test_data_collection, device=self.msdd_model.device
            )
            with autocast():
                _preds, scale_weights = self.msdd_model.forward_infer(
                    input_signal=sess_emb_seq,
                    input_signal_length=sess_sig_lengths,
                    emb_vectors=sess_emb_vectors,
                    targets=None,
                )
            _preds = _preds.reshape(len(signal_lengths), split_count * self.diar_window_length, -1)
            _preds = _preds[:, : signals.shape[1], :]
        else:
            with autocast():
                _preds, scale_weights = self.msdd_model.forward_infer(
                    input_signal=signals, input_signal_length=signal_lengths, emb_vectors=emb_vectors, targets=None
                )
        self.max_pred_length = max(_preds.shape[1], self.max_pred_length)
        preds = torch.zeros(_preds.shape[0], self.max_pred_length, _preds.shape[2])
        targets = torch.zeros(_preds.shape[0], self.max_pred_length, _preds.shape[2])
        preds[:, : _preds.shape[1], :] = _preds
        return preds, targets, signal_lengths

    @torch.no_grad()
    def run_pairwise_diarization(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Setup the parameters needed for batch inference and run batch inference. Note that each sample is pairwise speaker input.
        The pairwise inference results are reconstructed to make session-wise prediction results.

        Returns:
            integrated_preds_list: (list)
                List containing the session-wise speaker predictions in torch.tensor format.
            targets_list: (list)
                List containing the ground-truth labels in matrix format filled with  0 or 1.
            signal_lengths_list: (list)
                List containing the actual length of each sequence in session.
        """
        self.out_rttm_dir = self.clustering_embedding.out_rttm_dir
        self.msdd_model.setup_test_data(self.msdd_model.cfg.test_ds)
        self.msdd_model.eval()
        cumul_sample_count = [0]
        preds_list, targets_list, signal_lengths_list = [], [], []
        uniq_id_list = get_uniq_id_list_from_manifest(self.msdd_model.cfg.test_ds.manifest_filepath)
        test_data_collection = [d for d in self.msdd_model.data_collection]
        for sidx, test_batch in enumerate(tqdm(self.msdd_model.test_dataloader())):
            signals, signal_lengths, _targets, emb_vectors = test_batch
            cumul_sample_count.append(cumul_sample_count[-1] + signal_lengths.shape[0])
            preds, targets, signal_lengths = self.diar_infer(
                test_batch, test_data_collection[cumul_sample_count[-2] : cumul_sample_count[-1]]
            )
            if self._cfg.diarizer.msdd_model.parameters.seq_eval_mode:
                self.msdd_model._accuracy_test(preds, targets, signal_lengths)

            preds_list.extend(list(torch.split(preds, 1)))
            targets_list.extend(list(torch.split(targets, 1)))
            signal_lengths_list.extend(list(torch.split(signal_lengths, 1)))

        if self._cfg.diarizer.msdd_model.parameters.seq_eval_mode:
            f1_score, simple_acc = self.msdd_model.compute_accuracies()
            logging.info(f"Test Inference F1 score. {f1_score:.4f}, simple Acc. {simple_acc:.4f}")
        integrated_preds_list = self.get_integrated_preds_list(uniq_id_list, test_data_collection, preds_list)
        return integrated_preds_list, targets_list, signal_lengths_list

    def run_overlap_aware_eval(
        self, preds_list: List[torch.Tensor], threshold: float
    ) -> List[Optional[Tuple[DiarizationErrorRate, Dict]]]:
        """
        Based on the predicted sigmoid values, render RTTM files then evaluate the overlap-aware diarization results.

        Args:
            preds_list: (list)
                List containing predicted pairwise speaker labels.
            threshold: (float)
                A floating-point threshold value that determines overlapped speech detection.
                    - If threshold is 1.0, no overlap speech is detected and only detect major speaker.
                    - If threshold is 0.0, all speakers are considered active at any time step.
        """
        logging.info(
            f"     [Threshold: {threshold:.4f}] [use_clus_as_main={self.use_clus_as_main}] [diar_window={self.diar_window_length}]"
        )
        outputs = []
        manifest_filepath = self.msdd_model.cfg.test_ds.manifest_filepath
        rttm_map = audio_rttm_map(manifest_filepath)
        for k, (collar, ignore_overlap) in enumerate(self.diar_eval_settings):
            all_reference, all_hypothesis = make_rttm_with_overlap(
                manifest_filepath,
                self.msdd_model.clus_test_label_dict,
                preds_list,
                threshold=threshold,
                infer_overlap=True,
                use_clus_as_main=self.use_clus_as_main,
                overlap_infer_spk_limit=self.overlap_infer_spk_limit,
                use_adaptive_thres=self.use_adaptive_thres,
                max_overlap_spks=self.max_overlap_spks,
                out_rttm_dir=self.out_rttm_dir,
            )
            output = score_labels(
                rttm_map,
                all_reference,
                all_hypothesis,
                collar=collar,
                ignore_overlap=ignore_overlap,
                verbose=self._cfg.verbose,
            )
            outputs.append(output)
        logging.info(f"  \n")
        return outputs

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        vad_model_name: str = 'vad_multilingual_marblenet',
        map_location: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Instantiate a `NeuralDiarizer` to run Speaker Diarization.

        Args:
            model_name (str): Path/Name of the neural diarization model to load.
            vad_model_name (str): Path/Name of the voice activity detection (VAD) model to load.
            map_location (str): Optional str to map the instantiated model to a device (cpu, cuda).
                By default, (None), it will select a GPU if available, falling back to CPU otherwise.
            verbose (bool): Enable verbose logging when loading models/running diarization.
        Returns:
            `NeuralDiarizer`
        """
        logging.setLevel(logging.INFO if verbose else logging.WARNING)
        cfg = NeuralDiarizerInferenceConfig.init_config(
            diar_model_path=model_name, vad_model_path=vad_model_name, map_location=map_location, verbose=verbose,
        )
        return cls(cfg)

    def __call__(
        self,
        audio_filepath: str,
        batch_size: int = 64,
        num_workers: int = 1,
        max_speakers: Optional[int] = None,
        num_speakers: Optional[int] = None,
        out_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> Union[Annotation, List[Annotation]]:
        """
        Run the `NeuralDiarizer` inference pipeline.

        Args:
            audio_filepath (str, list): Audio path to run speaker diarization on.
            max_speakers (int): If known, the max number of speakers in the file(s).
            num_speakers (int): If known, the exact number of speakers in the file(s).
            batch_size (int): Batch size when running inference.
            num_workers (int): Number of workers to use in data-loading.
            out_dir (str): Path to store intermediate files during inference (default temp directory).
        Returns:
            `pyannote.Annotation` for each audio path, containing speaker labels and segment timestamps.
        """
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
            manifest_path = os.path.join(tmpdir, 'manifest.json')
            meta = [
                {
                    'audio_filepath': audio_filepath,
                    'offset': 0,
                    'duration': None,
                    'label': 'infer',
                    'text': '-',
                    'num_speakers': num_speakers,
                    'rttm_filepath': None,
                    'uem_filepath': None,
                }
            ]

            with open(manifest_path, 'w') as f:
                f.write('\n'.join(json.dumps(x) for x in meta))

            self._initialize_configs(
                manifest_path=manifest_path,
                max_speakers=max_speakers,
                num_speakers=num_speakers,
                tmpdir=tmpdir,
                batch_size=batch_size,
                num_workers=num_workers,
                verbose=verbose,
            )

            self.msdd_model.cfg.test_ds.manifest_filepath = manifest_path
            self.diarize()

            pred_labels_clus = rttm_to_labels(f'{tmpdir}/pred_rttms/{Path(audio_filepath).stem}.rttm')
        return labels_to_pyannote_object(pred_labels_clus)

    def _initialize_configs(
        self,
        manifest_path: str,
        max_speakers: Optional[int],
        num_speakers: Optional[int],
        tmpdir: tempfile.TemporaryDirectory,
        batch_size: int,
        num_workers: int,
        verbose: bool,
    ) -> None:
        self._cfg.batch_size = batch_size
        self._cfg.num_workers = num_workers
        self._cfg.diarizer.manifest_filepath = manifest_path
        self._cfg.diarizer.out_dir = tmpdir
        self._cfg.verbose = verbose
        self._cfg.diarizer.clustering.parameters.oracle_num_speakers = num_speakers is not None
        if max_speakers:
            self._cfg.diarizer.clustering.parameters.max_num_speakers = max_speakers
        self.transfer_diar_params_to_model_params(self.msdd_model, self._cfg)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        return EncDecDiarLabelModel.list_available_models()
