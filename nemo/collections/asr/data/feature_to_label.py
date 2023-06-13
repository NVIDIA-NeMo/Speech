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
from typing import Dict, List, Optional

import torch

from nemo.collections.asr.parts.preprocessing.feature_loader import ExternalFeatureLoader
from nemo.collections.common.parts.preprocessing import collections
from nemo.core.classes import Dataset
from nemo.core.neural_types import AcousticEncodedRepresentation, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


def _feature_collate_fn(batch):
    """collate batch of feat sig, feat len, labels, labels len, assuming all features have the same shape.
    Args:
        batch (FloatTensor, LongTensor, LongTensor, LongTensor):  A tuple of tuples of feature, feature lengths,
               encoded labels, and encoded labels length. 
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 5:
        _, feat_lengths, _, labels_lengths, sample_ids = packed_batch
    elif len(packed_batch) == 4:
        sample_ids = None
        _, feat_lengths, _, labels_lengths = packed_batch
    else:
        raise ValueError("Expects 4 or 5 tensors in the batch!")

    features, labels = [], []
    for b in batch:
        feat_i, labels_i = b[0], b[2]
        features.append(feat_i)
        labels.append(labels_i)

    features = torch.stack(features)
    feat_lengths = torch.stack(feat_lengths)

    labels = torch.stack(labels)
    labels_lengths = torch.stack(labels_lengths)

    if sample_ids is None:
        return features, feat_lengths, labels, labels_lengths
    else:
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
        return features, feat_lengths, labels, labels_lengths, sample_ids


def _audio_feature_collate_fn(batch, feat_pad_val, label_pad_id):
    """collate batch of audio feature, audio len, labels, labels len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of feature, feature lengths,
               labels, and label lengths.  This collate func assumes the 
               features are torch tensors of Log-Melspectrogram (i.e. [N_MEL, T]).
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 5:
        _, feat_lengths, _, labels_lengths, sample_ids = packed_batch
    elif len(packed_batch) == 4:
        sample_ids = None
        _, feat_lengths, _, labels_lengths = packed_batch
    else:
        raise ValueError("Expects 4 or 5 tensors in the batch!")
    max_feat_len = 0
    has_feat = feat_lengths[0] is not None
    if has_feat:
        max_feat_len = max(feat_lengths).item()
    max_labels_len = max(labels_lengths).item()

    features, labels = [], []
    for b in batch:
        feat_i, feat_i_len, label_i, label_i_len = b[0], b[1], b[2], b[3]

        if has_feat:
            feat_i_len = feat_i_len.item()
            if feat_i_len < max_feat_len:
                pad = (0, max_feat_len - feat_i_len)
                feat_i = torch.nn.functional.pad(feat_i, pad, value=feat_pad_val)
            features.append(feat_i)

        label_i_len = label_i_len.item()
        if label_i_len < max_labels_len:
            pad = (0, max_labels_len - label_i_len)
            label_i = torch.nn.functional.pad(label_i, pad, value=label_pad_id)
        labels.append(label_i)

    if has_feat:
        features = torch.stack(features)
        feature_lengths = torch.stack(feat_lengths)
    else:
        features, feat_lengths = None, None
    labels = torch.stack(labels)
    labels_lengths = torch.stack(labels_lengths)

    if sample_ids is None:
        return features, feature_lengths, labels, labels_lengths
    else:
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
        return features, feature_lengths, labels, labels_lengths, sample_ids


def _vad_feature_segment_collate_fn(batch, window_length_in_sec, shift_length_in_sec, frame_unit_in_sec):
    """collate batch of audio features, features len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
            LongTensor):  A tuple of tuples of signal, signal lengths,
            encoded tokens, and encoded tokens length.  This collate func
            assumes the signals are 1d torch tensors (i.e. mono audio).
            batch size equals to 1.
    """
    slice_length = int(window_length_in_sec / frame_unit_in_sec)
    audio_features, feat_lengths, _, tokens_lengths = zip(*batch)

    slice_length = int(min(slice_length, max(feat_lengths)))
    shift = int(shift_length_in_sec / frame_unit_in_sec)
    has_audio = feat_lengths[0] is not None

    f_dim = audio_features[0].shape[0]
    audio_features, num_slices, tokens, feat_lengths = [], [], [], []
    append_len_start = torch.div(slice_length, 2, rounding_mode='trunc')
    append_len_end = slice_length - torch.div(slice_length, 2, rounding_mode='trunc')
    for feat_i, feat_i_len, tokens_i, _ in batch:
        start = torch.zeros(f_dim, append_len_start)
        end = torch.zeros(f_dim, append_len_end)
        feat_i = torch.cat((start, feat_i, end), dim=1)
        feat_i_len += slice_length

        if has_audio:
            slices = max(1, torch.div(feat_i_len - slice_length, shift, rounding_mode='trunc'))

            for slice_id in range(slices):
                start_idx = slice_id * shift
                end_idx = start_idx + slice_length
                feat_slice = feat_i[:, start_idx:end_idx]
                audio_features.append(feat_slice)

            num_slices.append(slices)
            tokens.extend([tokens_i] * slices)
            feat_lengths.extend([slice_length] * slices)

    if has_audio:
        audio_features = torch.stack(audio_features)
        feat_lengths = torch.tensor(feat_lengths)
    else:
        audio_features, feat_lengths = None, None

    tokens = torch.stack(tokens)
    tokens_lengths = torch.tensor(num_slices)
    return audio_features, feat_lengths, tokens, tokens_lengths


class _FeatureSeqSpeakerLabelDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to feature files, sequences of labels. 
    Each new line is a different sample. Example below:
    and their target labels. JSON files should be of the following format:
        {"feature_filepath": "/path/to/feature_0.p", "seq_label": speakerA speakerB SpeakerA ....} \
        ...
        {"feature_filepath": "/path/to/feature_n.p", "seq_label": target_seq_label_n} 
    target_seq_label_n is the string of sequence of speaker label, separated by space.

    Args:
        manifest_filepath (str): Dataset parameter. Path to JSON containing data.
        labels (Optional[list]): Dataset parameter. List of unique labels collected from all samples.
        feature_loader : Dataset parameter. Feature loader to load (external) feature.       
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """
        # TODO output type for external features
        output_types = {
            'external_feat': NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            'feat_length': NeuralType(tuple('B'), LengthsType()),
        }

        if self.is_speaker_emb:
            output_types.update(
                {
                    'embs': NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                    'embs_length': NeuralType(tuple('B'), LengthsType()),
                    'label': NeuralType(('B', 'T'), LabelsType()),
                    'label_length': NeuralType(tuple('B'), LengthsType()),
                }
            )
        else:
            output_types.update(
                {'label': NeuralType(('B', 'T'), LabelsType()), 'label_length': NeuralType(tuple('B'), LengthsType()),}
            )

        return output_types

    def __init__(
        self, *, manifest_filepath: str, labels: List[str], feature_loader, is_speaker_emb: bool = False,
    ):
        super().__init__()
        self.collection = collections.ASRFeatureSequenceLabel(manifests_files=manifest_filepath.split(','),)

        self.feature_loader = feature_loader
        self.labels = labels if labels else self.collection.uniq_labels
        self.is_speaker_emb = is_speaker_emb

        self.label2id, self.id2label = {}, {}
        for label_id, label in enumerate(self.labels):
            self.label2id[label] = label_id
            self.id2label[label_id] = label

        for idx in range(len(self.labels[:5])):
            logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        sample = self.collection[index]

        features = self.feature_loader.process(sample.feature_file)
        f, fl = features, torch.tensor(features.shape[0]).long()

        t = torch.tensor(sample.seq_label).float()
        tl = torch.tensor(len(sample.seq_label)).long()

        return f, fl, t, tl


class FeatureToSeqSpeakerLabelDataset(_FeatureSeqSpeakerLabelDataset):
    """
    Dataset that loads tensors via a json file containing paths to feature
    files and sequence of speakers. Each new line is a
    different sample. Example below:
    {"feature_filepath": "/path/to/feature_0.p", "seq_label": speakerA speakerB SpeakerA ....} \
    ...
    {"feature_filepath": "/path/to/feature_n.p", "seq_label": target_seq_label_n} 
    target_seq_label_n is the string of sequence of speaker label, separated by space.

    Args:
        manifest_filepath (str): Path to manifest json as described above. Canbe comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRFeatureSequenceLabel collection.
        feature_loader, Feature load to loader (external) feature.
    
    """

    def _collate_fn(self, batch):
        return _feature_collate_fn(batch)


class FeatureToLabelDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to feature files and their labels. 
    Each new line is a different sample. Example below:
    and their target labels. JSON files should be of the following format:
        {"feature_filepath": "/path/to/audio_feature.pt", "label": "1"}
        ...
        {"feature_filepath": "/path/to/audio_feature.pt", "label": "0"} 
    Args:
        manifest_filepath (str): Path to JSON containing data.
        labels (Optional[list]): List of unique labels collected from all samples.
        augmentor (Optional): feature augmentation
        window_length_in_sec (float): Window length in seconds.
        shift_length_in_sec (float): Shift length in seconds.
        is_regression_task (bool): if True, the labels are treated as for a regression task.
        cal_labels_occurrence (bool): if True, the labels occurrence will be calculated.
        zero_spec_db_val (float): Value to replace non-speech signals in log-melspectrogram.
        min_duration (float): Minimum duration of the audio file in seconds.
        max_duration (float): Maximum duration of the audio file in seconds.
    """

    ZERO_LEVEL_SPEC_DB_VAL = -16.635  # Log-Melspectrogram value for zero signal
    FRAME_UNIT_TIME_SECS = 0.01

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """
        output_types = {
            'audio_feat': NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            'feat_length': NeuralType(tuple('B'), LengthsType()),
            'labels': NeuralType(('B'), LabelsType()),
            'labels_length': NeuralType(tuple('B'), LengthsType()),
        }

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        labels: List[str] = None,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        window_length_in_sec: float = 0.63,
        shift_length_in_sec: float = 0.01,
        is_regression_task: bool = False,
        cal_labels_occurrence: Optional[bool] = False,
        zero_spec_db_val: float = -16.635,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ):
        super().__init__()
        self.window_length_in_sec = window_length_in_sec
        self.shift_length_in_sec = shift_length_in_sec
        self.zero_spec_db_val = zero_spec_db_val

        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(',')

        self.collection = collections.ASRFeatureLabel(
            manifests_files=manifest_filepath,
            is_regression_task=is_regression_task,
            cal_labels_occurrence=cal_labels_occurrence,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        self.feature_loader = ExternalFeatureLoader(augmentor=augmentor)
        self.labels = labels if labels else self.collection.uniq_labels

        self.is_regression_task = is_regression_task

        if not is_regression_task:
            self.labels = labels if labels else self.collection.uniq_labels
            self.num_classes = len(self.labels) if self.labels is not None else 1
            self.label2id, self.id2label = {}, {}
            self.id2occurrence, self.labels_occurrence = {}, []

            for label_id, label in enumerate(self.labels):
                self.label2id[label] = label_id
                self.id2label[label_id] = label
                if cal_labels_occurrence:
                    self.id2occurrence[label_id] = self.collection.labels_occurrence[label]

            if cal_labels_occurrence:
                self.labels_occurrence = [self.id2occurrence[k] for k in sorted(self.id2occurrence)]

            for idx in range(len(self.labels[:5])):
                logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))
        else:
            self.labels = []
            self.num_classes = 1

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        sample = self.collection[index]

        features = self.feature_loader.process(sample.feature_file)
        f, fl = features, torch.tensor(features.shape[1]).long()

        t = torch.tensor(self.label2id[sample.label])
        tl = torch.tensor(1).long()

        return f, fl, t, tl

    def _collate_fn(self, batch):
        return _audio_feature_collate_fn(batch, self.zero_spec_db_val, 0)

    def _vad_segment_collate_fn(self, batch):
        return _vad_feature_segment_collate_fn(
            batch, self.window_length_in_sec, self.shift_length_in_sec, self.FRAME_UNIT_TIME_SECS
        )


class FeatureToMultiLabelDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to feature files and their labels. 
    Each new line is a different sample. Example below:
    and their target labels. JSON files should be of the following format:
        {"feature_filepath": "/path/to/audio_feature.pt", "label": "1 1 0 0 1"}
        ...
        {"feature_filepath": "/path/to/audio_feature.pt", "label": "0 1 0 0"} 
    Args:
        manifest_filepath (str): Path to JSON containing data.
        labels (Optional[list]): List of unique labels collected from all samples.
        augmentor (Optional): feature augmentation
        delimiter (str): delimiter to split the labels.
        is_regression_task (bool): if True, the labels are treated as for a regression task.
        cal_labels_occurrence (bool): if True, the labels occurrence will be calculated.
        zero_spec_db_val (float): Value to replace non-speech signals in log-melspectrogram.
        min_duration (float): Minimum duration of the audio file in seconds.
        max_duration (float): Maximum duration of the audio file in seconds.
    """

    ZERO_LEVEL_SPEC_DB_VAL = -16.635  # Log-Melspectrogram value for zero signal

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """
        output_types = {
            'audio_feat': NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            'feat_length': NeuralType(tuple('B'), LengthsType()),
            'labels': NeuralType(('B', 'T'), LabelsType()),
            'labels_length': NeuralType(tuple('B'), LengthsType()),
        }

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        labels: List[str] = None,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        delimiter: Optional[str] = None,
        is_regression_task: bool = False,
        cal_labels_occurrence: Optional[bool] = False,
        zero_spec_db_val: float = -16.635,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ):
        super().__init__()
        self.delimiter = delimiter
        self.zero_spec_db_val = zero_spec_db_val

        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(',')

        self.collection = collections.ASRFeatureLabel(
            manifests_files=manifest_filepath,
            is_regression_task=is_regression_task,
            cal_labels_occurrence=cal_labels_occurrence,
            delimiter=delimiter,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        self.is_regression_task = is_regression_task
        self.feature_loader = ExternalFeatureLoader(augmentor=augmentor)
        self.labels = labels if labels else self.collection.uniq_labels

        self.label2id, self.id2label = {}, {}
        if not is_regression_task:
            self.labels = labels if labels else self._get_label_set()
            self.num_classes = len(self.labels) if self.labels is not None else 1
            self.label2id, self.id2label = {}, {}
            for label_id, label in enumerate(self.labels):
                self.label2id[label] = label_id
                self.id2label[label_id] = label
                if cal_labels_occurrence:
                    self.id2occurrence[label_id] = self.collection.labels_occurrence[label]
                    self.labels_occurrence.append(self.id2occurrence[label_id])

            for idx in range(len(self.labels[:5])):
                logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))
        else:
            self.labels = []
            self.num_classes = 1

    def _get_label_set(self):
        labels = []
        for sample in self.collection:
            label_str = sample.label
            if label_str:
                label_str_list = label_str.split(self.delimiter) if self.delimiter else label_str.split()
                labels.extend(label_str_list)
        return sorted(set(labels))

    def _label_str_to_tensor(self, label_str: str):
        labels = label_str.split(self.delimiter) if self.delimiter else label_str.split()

        if self.is_regression_task:
            labels = [float(s) for s in labels]
            labels = torch.tensor(labels).float()
        else:
            labels = [self.label2id[s] for s in labels]
            labels = torch.tensor(labels).long()
        return labels

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        sample = self.collection[index]

        features = self.feature_loader.process(sample.feature_file)
        f, fl = features, torch.tensor(features.shape[1]).long()

        t = self._label_str_to_tensor(sample.label)
        tl = torch.tensor(t.size(0)).long()

        return f, fl, t, tl

    def _collate_fn(self, batch):
        return _audio_feature_collate_fn(batch, self.zero_spec_db_val, 0)
