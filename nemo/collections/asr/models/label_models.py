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
import copy
import itertools
from collections import Counter
from math import ceil
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torchmetrics import Accuracy
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_label import AudioToSpeechLabelDataset, cache_datastore_manifests
from nemo.collections.asr.data.audio_to_label_dataset import (
    get_concat_tarred_speech_label_dataset,
    get_tarred_speech_label_dataset,
)
from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.metrics import TopKClassificationAccuracy
from nemo.collections.common.parts.preprocessing.collections import ASRSpeechLabel
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import *
from nemo.utils import logging

__all__ = ['EncDecSpeakerLabelModel']


class EncDecSpeakerLabelModel(ModelPT, ExportableEncDecModel):
    """
    Encoder decoder class for speaker label models.
    Model class creates training, validation methods for setting up data
    performing model forward pass.
    Expects config dict for

        * preprocessor

        * Jasper/Quartznet Encoder

        * Speaker Decoder
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
            pretrained_model_name="speakerverification_speakernet",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/speakerverification_speakernet/versions/1.16.0/files/speakerverification_speakernet.nemo",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:speakerverification_speakernet",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="ecapa_tdnn",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/ecapa_tdnn/versions/1.16.0/files/ecapa_tdnn.nemo",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:ecapa_tdnn",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="titanet_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/titanet_large/versions/v1/files/titanet-l.nemo",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="langid_ambernet",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/langid_ambernet/versions/1.12.0/files/ambernet.nemo",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/langid_ambernet",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="titanet_small",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:titanet_small",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/titanet_small/versions/1.19.0/files/titanet-s.nemo",
        )
        result.append(model)

        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        self.cal_labels_occurrence_train = False
        self.labels_occurrence = None
        self.labels = None

        num_classes = cfg.decoder.num_classes

        if 'loss' in cfg:
            if 'weight' in cfg.loss:
                if cfg.loss.weight == 'auto':
                    weight = num_classes * [1]
                    self.cal_labels_occurrence_train = True
                else:
                    weight = cfg.loss.weight
            else:
                weight = None  # weight is None for angular loss and CE loss if it's not specified.

        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        super().__init__(cfg=cfg, trainer=trainer)

        if self.labels_occurrence:
            # Goal is to give more weight to the classes with less samples so as to match the ones with the higher frequencies
            weight = [sum(self.labels_occurrence) / (len(self.labels_occurrence) * i) for i in self.labels_occurrence]

        if 'loss' in cfg:
            cfg_eval_loss = copy.deepcopy(cfg.loss)

            if '_target_' in cfg.loss and 'angular' in cfg.loss._target_:
                OmegaConf.set_struct(cfg, True)
                with open_dict(cfg):
                    cfg.decoder.angular = True

            if 'weight' in cfg.loss:
                cfg.loss.weight = weight
                cfg_eval_loss.weight = None

            # May need a general check for arguments of loss
            self.loss = instantiate(cfg.loss)
            self.eval_loss = instantiate(cfg_eval_loss)

        else:
            tmp_loss_cfg = OmegaConf.create(
                {"_target_": "nemo.collections.common.losses.cross_entropy.CrossEntropyLoss"}
            )

            self.loss = instantiate(tmp_loss_cfg)
            self.eval_loss = instantiate(tmp_loss_cfg)

        self._accuracy = TopKClassificationAccuracy(top_k=[1])

        self.preprocessor = EncDecSpeakerLabelModel.from_config_dict(cfg.preprocessor)
        self.encoder = EncDecSpeakerLabelModel.from_config_dict(cfg.encoder)
        self.decoder = EncDecSpeakerLabelModel.from_config_dict(cfg.decoder)

        self._macro_accuracy = Accuracy(num_classes=num_classes, top_k=1, average='macro', task='multiclass')

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecSpeakerLabelModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

    @staticmethod
    def extract_labels(data_layer_config):
        labels = set()
        manifest_filepath = data_layer_config.get('manifest_filepath', None)
        if manifest_filepath is None:
            logging.warning("No manifest_filepath was provided, no labels got extracted!")
            return None
        manifest_filepaths = convert_to_config_list(data_layer_config['manifest_filepath'])

        for manifest_filepath in itertools.chain.from_iterable(manifest_filepaths):
            cache_datastore_manifests(manifest_filepaths=manifest_filepath)
            collection = ASRSpeechLabel(
                manifests_files=manifest_filepath,
                min_duration=data_layer_config.get("min_duration", None),
                max_duration=data_layer_config.get("max_duration", None),
                index_by_file_id=True,
            )
            labels.update(collection.uniq_labels)
        labels = list(sorted(labels))
        logging.warning(f"Total number of {len(labels)} found in all the manifest files.")
        return labels

    def __setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=augmentor
        )
        shuffle = config.get('shuffle', False)
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            if config.get("is_concat", False):
                dataset = get_concat_tarred_speech_label_dataset(
                    featurizer=featurizer,
                    config=config,
                    shuffle_n=shuffle_n,
                    global_rank=self.global_rank,
                    world_size=self.world_size,
                )
            else:
                dataset = get_tarred_speech_label_dataset(
                    featurizer=featurizer,
                    config=config,
                    shuffle_n=shuffle_n,
                    global_rank=self.global_rank,
                    world_size=self.world_size,
                )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = AudioToSpeechLabelDataset(
                manifest_filepath=config['manifest_filepath'],
                labels=config['labels'],
                featurizer=featurizer,
                max_duration=config.get('max_duration', None),
                min_duration=config.get('min_duration', None),
                trim=config.get('trim_silence', False),
                normalize_audio=config.get('normalize_audio', False),
                cal_labels_occurrence=config.get('cal_labels_occurrence', False),
            )
            if dataset.labels_occurrence:
                self.labels_occurrence = dataset.labels_occurrence

        if hasattr(dataset, 'fixed_seq_collate_fn'):
            collate_fn = dataset.fixed_seq_collate_fn
        else:
            collate_fn = dataset.datasets[0].fixed_seq_collate_fn

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

    def setup_training_data(self, train_data_layer_config: Optional[Union[DictConfig, Dict]]):
        if self.cal_labels_occurrence_train:
            # Calculate labels occurence for weighed CE loss for train set if weight equals 'auto'
            # Note in this case, the cal_labels_occurrence in val_data_layer_config and test_data_layer_params need to be stay as False
            OmegaConf.set_struct(train_data_layer_config, True)
            with open_dict(train_data_layer_config):
                train_data_layer_config['cal_labels_occurrence'] = True

        self.labels = self.extract_labels(train_data_layer_config)
        train_data_layer_config['labels'] = self.labels
        if 'shuffle' not in train_data_layer_config:
            train_data_layer_config['shuffle'] = True
        self._train_dl = self.__setup_dataloader_from_config(config=train_data_layer_config)
        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if (
            self._train_dl is not None
            and hasattr(self._train_dl, 'dataset')
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_layer_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        val_data_layer_config['labels'] = self.labels
        self._validation_dl = self.__setup_dataloader_from_config(config=val_data_layer_config)

    def setup_test_data(self, test_data_layer_params: Optional[Union[DictConfig, Dict]]):
        if hasattr(self, 'dataset'):
            test_data_layer_params['labels'] = self.labels

        self.embedding_dir = test_data_layer_params.get('embedding_dir', './')
        self._test_dl = self.__setup_dataloader_from_config(config=test_data_layer_params)
        self.test_manifest = test_data_layer_params.get('manifest_filepath', None)

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
            "input_signal": NeuralType(('B', 'T'), audio_eltype),
            "input_signal_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "logits": NeuralType(('B', 'D'), LogitsType()),
            "embs": NeuralType(('B', 'D'), AcousticEncodedRepresentation()),
        }

    def forward_for_export(self, processed_signal, processed_signal_len):
        encoded, length = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        logits, embs = self.decoder(encoder_output=encoded, length=length)
        return logits, embs

    @typecheck()
    def forward(self, input_signal, input_signal_length):
        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=input_signal,
            length=input_signal_length,
        )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_len)

        encoded, length = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        logits, embs = self.decoder(encoder_output=encoded, length=length)
        return logits, embs

    # PTL-specific methods
    def training_step(self, batch, batch_idx):
        audio_signal, audio_signal_len, labels, _ = batch
        logits, _ = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss = self.loss(logits=logits, labels=labels)

        self.log('loss', loss)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', self.trainer.global_step)

        self._accuracy(logits=logits, labels=labels)
        top_k = self._accuracy.compute()
        self._accuracy.reset()
        for i, top_i in enumerate(top_k):
            self.log(f'training_batch_accuracy_top_{i}', top_i)

        return {'loss': loss}

    def evaluation_step(self, batch, batch_idx, dataloader_idx: int = 0, tag: str = 'val'):
        audio_signal, audio_signal_len, labels, _ = batch
        logits, _ = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.eval_loss(logits=logits, labels=labels)
        acc_top_k = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k
        self._macro_accuracy.update(preds=logits, target=labels)
        stats = self._macro_accuracy._final_state()

        output = {
            f'{tag}_loss': loss_value,
            f'{tag}_correct_counts': correct_counts,
            f'{tag}_total_counts': total_counts,
            f'{tag}_acc_micro_top_k': acc_top_k,
            f'{tag}_acc_macro_stats': stats,
        }
        if tag == 'val':
            if isinstance(self.trainer.val_dataloaders, (list, tuple)) and len(self.trainer.val_dataloaders) > 1:
                self.validation_step_outputs[dataloader_idx].append(output)
            else:
                self.validation_step_outputs.append(output)
        else:
            if isinstance(self.trainer.test_dataloaders, (list, tuple)) and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx].append(output)
            else:
                self.test_step_outputs.append(output)

        return output

    def multi_evaluation_epoch_end(self, outputs, dataloader_idx: int = 0, tag: str = 'val'):
        loss_mean = torch.stack([x[f'{tag}_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x[f'{tag}_correct_counts'] for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x[f'{tag}_total_counts'] for x in outputs]).sum(axis=0)

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()

        self._macro_accuracy.tp = torch.stack([x[f'{tag}_acc_macro_stats'][0] for x in outputs]).sum(axis=0)
        self._macro_accuracy.fp = torch.stack([x[f'{tag}_acc_macro_stats'][1] for x in outputs]).sum(axis=0)
        self._macro_accuracy.tn = torch.stack([x[f'{tag}_acc_macro_stats'][2] for x in outputs]).sum(axis=0)
        self._macro_accuracy.fn = torch.stack([x[f'{tag}_acc_macro_stats'][3] for x in outputs]).sum(axis=0)
        macro_accuracy_score = self._macro_accuracy.compute()

        self._accuracy.reset()
        self._macro_accuracy.reset()

        self.log(f'{tag}_loss', loss_mean, sync_dist=True)
        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            self.log(f'{tag}_acc_micro_top_{top_k}', score, sync_dist=True)
        self.log(f'{tag}_acc_macro', macro_accuracy_score, sync_dist=True)

        return {
            f'{tag}_loss': loss_mean,
            f'{tag}_acc_micro_top_k': topk_scores,
            f'{tag}_acc_macro': macro_accuracy_score,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, 'val')

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_evaluation_epoch_end(outputs, dataloader_idx, 'val')

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, 'test')

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_evaluation_epoch_end(outputs, dataloader_idx, 'test')

    @torch.no_grad()
    def infer_file(self, path2audio_file):
        """
        Args:
            path2audio_file: path to an audio wav file

        Returns:
            emb: speaker embeddings (Audio representations)
            logits: logits corresponding of final layer
        """
        audio, sr = sf.read(path2audio_file)
        target_sr = self._cfg.train_ds.get('sample_rate', 16000)
        if sr != target_sr:
            audio = librosa.core.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio_length = audio.shape[0]
        device = self.device
        audio = np.array([audio])
        audio_signal, audio_signal_len = (
            torch.tensor(audio, device=device, dtype=torch.float32),
            torch.tensor([audio_length], device=device),
        )
        mode = self.training
        self.freeze()

        logits, emb = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)

        self.train(mode=mode)
        if mode is True:
            self.unfreeze()
        del audio_signal, audio_signal_len
        return emb, logits

    @torch.no_grad()
    def infer_segment(self, segment):
        """
        Args:
            segment: segment of audio file

        Returns:
            emb: speaker embeddings (Audio representations)
            logits: logits corresponding of final layer
        """
        segment_length = segment.shape[0]

        device = self.device
        audio = np.array([segment])
        audio_signal, audio_signal_len = (
            torch.tensor(audio, device=device, dtype=torch.float32),
            torch.tensor([segment_length], device=device),
        )
        mode = self.training
        self.freeze()

        logits, emb = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)

        self.train(mode=mode)
        if mode is True:
            self.unfreeze()
        del audio_signal, audio_signal_len
        return emb, logits

    def get_label(
        self, path2audio_file: str, segment_duration: float = np.inf, num_segments: int = 1, random_seed: int = None
    ):
        """
        Returns label of path2audio_file from classes the model was trained on.
        Args:
            path2audio_file (str): Path to audio wav file.
            segment_duration (float): Random sample duration in seconds.
            num_segments (int): Number of segments of file to use for majority vote.
            random_seed (int): Seed for generating the starting position of the segment.

        Returns:
            label: label corresponding to the trained model
        """
        audio, sr = sf.read(path2audio_file)
        target_sr = self._cfg.train_ds.get('sample_rate', 16000)
        if sr != target_sr:
            audio = librosa.core.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio_length = audio.shape[0]

        duration = target_sr * segment_duration
        if duration > audio_length:
            duration = audio_length

        label_id_list = []
        np.random.seed(random_seed)
        starts = np.random.randint(0, audio_length - duration + 1, size=num_segments)
        for start in starts:
            audio = audio[start : start + duration]

            _, logits = self.infer_segment(audio)
            label_id = logits.argmax(axis=1)
            label_id_list.append(int(label_id[0]))

        m_label_id = Counter(label_id_list).most_common(1)[0][0]

        trained_labels = self._cfg['train_ds'].get('labels', None)
        if trained_labels is not None:
            trained_labels = list(trained_labels)
            label = trained_labels[m_label_id]
        else:
            logging.info("labels are not saved to model, hence only outputting the label id index")
            label = m_label_id

        return label

    def get_embedding(self, path2audio_file):
        """
        Returns the speaker embeddings for a provided audio file.

        Args:
            path2audio_file: path to an audio wav file

        Returns:
            emb: speaker embeddings (Audio representations)
        """

        emb, _ = self.infer_file(path2audio_file=path2audio_file)

        return emb

    @torch.no_grad()
    def verify_speakers(self, path2audio_file1, path2audio_file2, threshold=0.7):
        """
        Verify if two audio files are from the same speaker or not.

        Args:
            path2audio_file1: path to audio wav file of speaker 1
            path2audio_file2: path to audio wav file of speaker 2
            threshold: cosine similarity score used as a threshold to distinguish two embeddings (default = 0.7)

        Returns:
            True if both audio files are from same speaker, False otherwise
        """
        embs1 = self.get_embedding(path2audio_file1).squeeze()
        embs2 = self.get_embedding(path2audio_file2).squeeze()
        # Length Normalize
        X = embs1 / torch.linalg.norm(embs1)
        Y = embs2 / torch.linalg.norm(embs2)
        # Score
        similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
        similarity_score = (similarity_score + 1) / 2
        # Decision
        if similarity_score >= threshold:
            logging.info(" two audio files are from same speaker")
            return True
        else:
            logging.info(" two audio files are from different speakers")
            return False

    @torch.no_grad()
    def batch_inference(self, manifest_filepath, batch_size=32, sample_rate=16000, device='cuda'):
        """
        Perform batch inference on EncDecSpeakerLabelModel.
        To perform inference on single audio file, once can use infer_model, get_label or get_embedding

        To map predicted labels, one can do
            `arg_values = logits.argmax(axis=1)`
            `pred_labels = list(map(lambda t : trained_labels[t], arg_values))`

        Args:
            manifest_filepath: Path to manifest file
            batch_size: batch size to perform batch inference
            sample_rate: sample rate of audio files in manifest file
            device: compute device to perform operations.

        Returns:
            The variables below all follow the audio file order in the manifest file.
            embs: embeddings of files provided in manifest file
            logits: logits of final layer of EncDecSpeakerLabel Model
            gt_labels: labels from manifest file (needed for speaker enrollment and testing)
            trained_labels: Classification labels sorted in the order that they are mapped by the trained model

        """
        mode = self.training
        self.freeze()
        self.eval()
        self.to(device)
        trained_labels = self._cfg['train_ds']['labels']
        if trained_labels is not None:
            trained_labels = list(trained_labels)

        featurizer = WaveformFeaturizer(sample_rate=sample_rate)

        dataset = AudioToSpeechLabelDataset(manifest_filepath=manifest_filepath, labels=None, featurizer=featurizer)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=dataset.fixed_seq_collate_fn,
        )

        logits = []
        embs = []
        gt_labels = []

        for test_batch in tqdm(dataloader):
            if device == 'cuda':
                test_batch = [x.to(device) for x in test_batch]
            audio_signal, audio_signal_len, labels, _ = test_batch
            logit, emb = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)

            logits.extend(logit.cpu().numpy())
            gt_labels.extend(labels.cpu().numpy())
            embs.extend(emb.cpu().numpy())

        gt_labels = list(map(lambda t: dataset.id2label[t], gt_labels))

        self.train(mode=mode)
        if mode is True:
            self.unfreeze()

        logits, embs, gt_labels = np.asarray(logits), np.asarray(embs), np.asarray(gt_labels)

        return embs, logits, gt_labels, trained_labels
