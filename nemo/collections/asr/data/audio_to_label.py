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
import io
import os
from typing import Dict, List, Optional, Union

import torch
import webdataset as wds

from nemo.collections.asr.data.audio_to_text import cache_datastore_manifests, expand_sharded_filepaths
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import available_formats as valid_sf_formats
from nemo.collections.common.parts.preprocessing import collections
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType, RegressionValuesType
from nemo.utils import logging
from nemo.utils.distributed import webdataset_split_by_workers

# List of valid file formats (prioritized by order of importance)
VALID_FILE_FORMATS = ';'.join(['wav', 'mp3', 'flac', 'opus'] + [fmt.lower() for fmt in valid_sf_formats.keys()])


def repeat_signal(signal: torch.Tensor, sig_len: int, required_length: int) -> torch.Tensor:
    """repeat signal to make short signal to have required_length
    Args:
        signal (Tensor): input signal
        sig_len (int): length of input signal
        required_length (int): length of generated signal
    Returns:
        signal (Tensor): generated signal of required_length by repeating itself.
    """
    sub: torch.Tensor = torch.tensor([])
    repeat = int(required_length // sig_len)
    rem = int(required_length % sig_len)
    sub: torch.Tensor = torch.tensor([])
    rep_sig: torch.Tensor = torch.cat(repeat * [signal])
    if rem > 0:
        sub = signal[-rem:]
        signal = torch.cat((rep_sig, sub))
    else:
        signal = rep_sig
    return signal


def normalize(signal):
    """normalize signal
    Args:
        signal(FloatTensor): signal to be normalized.
    """
    signal_minusmean = signal - signal.mean()
    return signal_minusmean / signal_minusmean.abs().max()


def count_occurence(manifest_file_id):
    """Count number of wav files in Dict manifest_file_id. Use for _TarredAudioToLabelDataset.
    Args:
        manifest_file_id (Dict): Dict of files and their corresponding id. {'A-sub0' : 1, ..., 'S-sub10':100}
    Returns:
        count (Dict): Dict of wav files {'A' : 2, ..., 'S':10}
    """
    count = dict()
    for i in manifest_file_id:
        audio_filename = i.split("-sub")[0]
        count[audio_filename] = count.get(audio_filename, 0) + 1
    return count


def _speech_collate_fn(batch, pad_id):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    _, audio_lengths, _, tokens_lengths = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens = [], []
    for sig, sig_len, tokens_i, tokens_i_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


def _fixed_seq_collate_fn(self, batch):
    """collate batch of audio sig, audio len, tokens, tokens len
        Args:
            batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
                LongTensor):  A tuple of tuples of signal, signal lengths,
                encoded tokens, and encoded tokens length.  This collate func
                assumes the signals are 1d torch tensors (i.e. mono audio).
        """
    _, audio_lengths, _, tokens_lengths = zip(*batch)

    has_audio = audio_lengths[0] is not None
    fixed_length = int(max(audio_lengths))

    audio_signal, tokens, new_audio_lengths = [], [], []
    for sig, sig_len, tokens_i, _ in batch:
        if has_audio:
            sig_len = sig_len.item()
            chunck_len = sig_len - fixed_length

            if chunck_len < 0:
                repeat = fixed_length // sig_len
                rem = fixed_length % sig_len
                sub = sig[-rem:] if rem > 0 else torch.tensor([])
                rep_sig = torch.cat(repeat * [sig])
                sig = torch.cat((rep_sig, sub))
            new_audio_lengths.append(torch.tensor(fixed_length))

            audio_signal.append(sig)

        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(new_audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


def _vad_frame_seq_collate_fn(self, batch):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
            LongTensor):  A tuple of tuples of signal, signal lengths,
            encoded tokens, and encoded tokens length.  This collate func
            assumes the signals are 1d torch tensors (i.e. mono audio).
            batch size equals to 1.
    """
    slice_length = int(self.featurizer.sample_rate * self.window_length_in_sec)
    _, audio_lengths, _, tokens_lengths = zip(*batch)
    slice_length = int(min(slice_length, max(audio_lengths)))
    shift = int(self.featurizer.sample_rate * self.shift_length_in_sec)
    has_audio = audio_lengths[0] is not None

    audio_signal, num_slices, tokens, audio_lengths = [], [], [], []

    append_len_start = slice_length // 2
    append_len_end = slice_length - slice_length // 2
    for sig, sig_len, tokens_i, _ in batch:
        if self.normalize_audio:
            sig = normalize(sig)
        start = torch.zeros(append_len_start)
        end = torch.zeros(append_len_end)
        sig = torch.cat((start, sig, end))
        sig_len += slice_length

        if has_audio:
            slices = torch.div(sig_len - slice_length, shift, rounding_mode='trunc')
            for slice_id in range(slices):
                start_idx = slice_id * shift
                end_idx = start_idx + slice_length
                signal = sig[start_idx:end_idx]
                audio_signal.append(signal)

            num_slices.append(slices)
            tokens.extend([tokens_i] * slices)
            audio_lengths.extend([slice_length] * slices)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.tensor(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    tokens = torch.stack(tokens)
    tokens_lengths = torch.tensor(num_slices)
    return audio_signal, audio_lengths, tokens, tokens_lengths


class _AudioLabelDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files,
    labels, and durations and offsets(in seconds). Each new line is a
    different sample. Example below:
    and their target labels. JSON files should be of the following format::
        {"audio_filepath": "/path/to/audio_wav_0.wav", "duration": time_in_sec_0, "label": \
target_label_0, "offset": offset_in_sec_0}
        ...
        {"audio_filepath": "/path/to/audio_wav_n.wav", "duration": time_in_sec_n, "label": \
target_label_n, "offset": offset_in_sec_n}
    Args:
        manifest_filepath (Union[str, List[str]]): Dataset parameter. Path to JSON containing data.
        labels (list): Dataset parameter. List of target classes that can be output by the speaker recognition model.
        featurizer
        min_duration (float): Dataset parameter. All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim (bool): Whether to use trim silence from beginning and end of audio signal using librosa.effects.trim().
            Defaults to False.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """

        output_types = {
            'audio_signal': NeuralType(
                ('B', 'T'),
                AudioSignal(freq=self._sample_rate)
                if self is not None and hasattr(self, '_sample_rate')
                else AudioSignal(),
            ),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

        if self.is_regression_task:
            output_types.update(
                {
                    'targets': NeuralType(tuple('B'), RegressionValuesType()),
                    'targets_length': NeuralType(tuple('B'), LengthsType()),
                }
            )
        else:

            output_types.update(
                {'label': NeuralType(tuple('B'), LabelsType()), 'label_length': NeuralType(tuple('B'), LengthsType()),}
            )

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: Union[str, List[str]],
        labels: List[str],
        featurizer,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        is_regression_task: bool = False,
        cal_labels_occurrence: Optional[bool] = False,
    ):
        super().__init__()
        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(',')
        cache_datastore_manifests(manifest_filepaths=manifest_filepath, cache_audio=True)
        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            is_regression_task=is_regression_task,
            cal_labels_occurrence=cal_labels_occurrence,
        )

        self.featurizer = featurizer
        self.trim = trim
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

        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim)
        f, fl = features, torch.tensor(features.shape[0]).long()

        if not self.is_regression_task:
            t = torch.tensor(self.label2id[sample.label]).long()
        else:
            t = torch.tensor(sample.label).float()

        tl = torch.tensor(1).long()  # For compatibility with collate_fn used later

        return f, fl, t, tl


# Ported from https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech_commands.py
class AudioToClassificationLabelDataset(_AudioLabelDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, command class, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio_wav_0.wav", "duration": time_in_sec_0, "label": \
        target_label_0, "offset": offset_in_sec_0}
    ...
    {"audio_filepath": "/path/to/audio_wav_n.wav", "duration": time_in_sec_n, "label": \
        target_label_n, "offset": offset_in_sec_n}
    Args:
        manifest_filepath (Union[str, List[str]]): Path to manifest json as described above. Can
            be comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRSpeechLabel collection.
        featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        trim: Boolean flag whether to trim the audio
    """

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=0)


class AudioToSpeechLabelDataset(_AudioLabelDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, command class, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio_wav_0.wav", "duration": time_in_sec_0, "label": \
        target_label_0, "offset": offset_in_sec_0}
    ...
    {"audio_filepath": "/path/to/audio_wav_n.wav", "duration": time_in_sec_n, "label": \
        target_label_n, "offset": offset_in_sec_n}
    Args:
        manifest_filepath (Union[str, List[str]]): Path to manifest json as described above. Can
            be comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRSpeechLabel collection.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        window_length_in_sec (float): length of window/slice (in seconds)
            Use this for speaker recognition and VAD tasks.
        shift_length_in_sec (float): amount of shift of window for generating the frame for VAD task in a batch
            Use this for VAD task during inference.
        normalize_audio (bool): Whether to normalize audio signal.
            Defaults to False.
        is_regression_task (bool): Whether the dataset is for a regression task instead of classification.
            Defaults to False.
        cal_labels_occurrence (bool): Whether to calculate occurrence of labels
            Defaults to False.
    """

    def __init__(
        self,
        *,
        manifest_filepath: Union[str, List[str]],
        labels: List[str],
        featurizer,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        window_length_in_sec: Optional[float] = 8,
        shift_length_in_sec: Optional[float] = 1,
        normalize_audio: bool = False,
        is_regression_task: bool = False,
        cal_labels_occurrence: Optional[bool] = False,
    ):
        self.window_length_in_sec = window_length_in_sec
        self.shift_length_in_sec = shift_length_in_sec
        self.normalize_audio = normalize_audio

        logging.debug("Window/slice length considered for collate func is {}".format(self.window_length_in_sec))
        logging.debug("Shift length considered for collate func is {}".format(self.shift_length_in_sec))

        super().__init__(
            manifest_filepath=manifest_filepath,
            labels=labels,
            featurizer=featurizer,
            min_duration=min_duration,
            max_duration=max_duration,
            trim=trim,
            is_regression_task=is_regression_task,
            cal_labels_occurrence=cal_labels_occurrence,
        )

    def fixed_seq_collate_fn(self, batch):
        return _fixed_seq_collate_fn(self, batch)

    def vad_frame_seq_collate_fn(self, batch):
        return _vad_frame_seq_collate_fn(self, batch)


class _TarredAudioLabelDataset(IterableDataset):
    """
    A similar Dataset to the AudioLabelDataSet, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToSpeechLabelDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the label and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
    This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
    Supported opening braces - { <=> (, [, < and the special tag _OP_.
    Supported closing braces - } <=> ), ], > and the special tag _CL_.
    For SLURM based tasks, we suggest the use of the special tags for ease of use.

    See the documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioLabelDataSet; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speaker recognition model.
        featurizer
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim(bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        window_length_in_sec (float): length of slice/window (in seconds) # Pass this only for speaker recognition and VAD task
        shift_length_in_sec (float): amount of shift of window for generating the frame for VAD task. in a batch # Pass this only for VAD task during inference.
        normalize_audio (bool): Whether to normalize audio signal. Defaults to False.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on the value of `shuffle_n`.

                .. warning::
                    Replicated strategy allows every node to sample the entire set of available tarfiles,
                    and therefore more than one node may sample the same tarfile, and even sample the same
                    data points! As such, there is no assured guarantee that all samples in the dataset will be
                    sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                    occasions (when the number of shards is not divisible with ``world_size``), will not sample
                    the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                    or test datasets.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        is_regression_task (bool): Whether it is a regression task. Defualts to False.
    """

    def __init__(
        self,
        *,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: Union[str, List[str]],
        labels: List[str],
        featurizer,
        shuffle_n: int = 0,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
        is_regression_task: bool = False,
    ):
        cache_datastore_manifests(manifest_filepaths=manifest_filepath)
        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            index_by_file_id=True,  # Must set this so the manifest lines can be indexed by file ID
        )

        self.file_occurence = count_occurence(self.collection.mapping)

        self.featurizer = featurizer
        self.trim = trim

        self.labels = labels if labels else self.collection.uniq_labels
        self.num_classes = len(self.labels)

        self.label2id, self.id2label = {}, {}
        for label_id, label in enumerate(self.labels):
            self.label2id[label] = label_id
            self.id2label[label_id] = label

        for idx in range(len(self.labels[:5])):
            logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))

        audio_tar_filepaths = expand_sharded_filepaths(
            sharded_filepaths=audio_tar_filepaths,
            shard_strategy=shard_strategy,
            world_size=world_size,
            global_rank=global_rank,
        )
        # Put together WebDataset
        self._dataset = wds.DataPipeline(
            wds.SimpleShardList(urls=audio_tar_filepaths),
            webdataset_split_by_workers,
            wds.shuffle(shuffle_n),
            wds.tarfile_to_samples(),
            wds.rename(audio=VALID_FILE_FORMATS, key='__key__'),
            wds.to_tuple('audio', 'key'),
            self._filter,
            wds.map(self._build_sample),
        )

    def _filter(self, iterator):
        """This function is used to remove samples that have been filtered out by ASRSpeechLabel already.
        Otherwise, we would get a KeyError as _build_sample attempts to find the manifest entry for a sample
        that was filtered out (e.g. for duration).
        Note that if using multi-GPU training, filtering may lead to an imbalance in samples in each shard,
        which may make your code hang as one process will finish before the other.
        """

        class TarredAudioFilter:
            def __init__(self, collection, file_occurence):
                self.iterator = iterator
                self.collection = collection
                self.file_occurence = file_occurence
                self._iterable = self._internal_generator()

            def __iter__(self):
                self._iterable = self._internal_generator()
                return self

            def __next__(self):
                try:
                    values = next(self._iterable)
                except StopIteration:
                    # reset generator
                    self._iterable = self._internal_generator()
                    values = next(self._iterable)

                return values

            def _internal_generator(self):
                """
                WebDataset requires an Iterator, but we require an iterable that yields 1-or-more
                values per value inside self.iterator.

                Therefore wrap the iterator with a generator function that will yield 1-or-more
                values per sample in the iterator.
                """
                for _, tup in enumerate(self.iterator):
                    audio_bytes, audio_filename = tup

                    file_id, _ = os.path.splitext(os.path.basename(audio_filename))
                    if audio_filename in self.file_occurence:
                        for j in range(0, self.file_occurence[file_id]):
                            if j == 0:
                                audio_filename = file_id
                            else:
                                audio_filename = file_id + "-sub" + str(j)
                            yield audio_bytes, audio_filename

        return TarredAudioFilter(self.collection, self.file_occurence)

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info.
        """
        audio_bytes, audio_filename = tup
        # Grab manifest entry from self.collection
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))

        manifest_idx = self.collection.mapping[file_id]
        manifest_entry = self.collection[manifest_idx]

        offset = manifest_entry.offset
        if offset is None:
            offset = 0

        # Convert audio bytes to IO stream for processing (for SoundFile to read)
        audio_filestream = io.BytesIO(audio_bytes)
        features = self.featurizer.process(
            audio_filestream, offset=offset, duration=manifest_entry.duration, trim=self.trim,
        )

        audio_filestream.close()

        # Audio features
        f, fl = features, torch.tensor(features.shape[0]).long()

        t = self.label2id[manifest_entry.label]
        tl = 1  # For compatibility with collate_fn used later

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return len(self.collection)


class TarredAudioToClassificationLabelDataset(_TarredAudioLabelDataset):
    """
    A similar Dataset to the AudioToClassificationLabelDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToClassificationLabelDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToBPEDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speaker recognition model.
        featurizer
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim(bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                .. warning::
                    Replicated strategy allows every node to sample the entire set of available tarfiles,
                    and therefore more than one node may sample the same tarfile, and even sample the same
                    data points! As such, there is no assured guarantee that all samples in the dataset will be
                    sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                    occasions (when the number of shards is not divisible with ``world_size``), will not sample
                    the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                    or test datasets.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        is_regression_task (bool): Whether it is a regression task. Defualts to False.
    """

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=0)


class TarredAudioToSpeechLabelDataset(_TarredAudioLabelDataset):
    """
    A similar Dataset to the AudioToSpeechLabelDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToSpeechLabelDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToBPEDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speaker recognition model.
        featurizer
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim(bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        window_length_in_sec (float): time length of window/slice (in seconds) # Pass this only for speaker recognition and VAD task
        shift_length_in_sec (float): amount of shift of window for generating the frame for VAD task. in a batch # Pass this only for VAD task during inference.
        normalize_audio (bool): Whether to normalize audio signal. Defaults to False.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                .. warning::
                    Replicated strategy allows every node to sample the entire set of available tarfiles,
                    and therefore more than one node may sample the same tarfile, and even sample the same
                    data points! As such, there is no assured guarantee that all samples in the dataset will be
                    sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                    occasions (when the number of shards is not divisible with ``world_size``), will not sample
                    the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                    or test datasets.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
    """

    def __init__(
        self,
        *,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: Union[str, List[str]],
        labels: List[str],
        featurizer,
        shuffle_n: int = 0,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        window_length_in_sec: Optional[float] = 8,
        shift_length_in_sec: Optional[float] = 1,
        normalize_audio: bool = False,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
    ):
        logging.info("Window/slice length considered for collate func is {}".format(window_length_in_sec))
        logging.info("Shift length considered for collate func is {}".format(shift_length_in_sec))
        self.window_length_in_sec = window_length_in_sec
        self.shift_length_in_sec = shift_length_in_sec
        self.normalize_audio = normalize_audio

        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            labels=labels,
            featurizer=featurizer,
            shuffle_n=shuffle_n,
            min_duration=min_duration,
            max_duration=max_duration,
            trim=trim,
            shard_strategy=shard_strategy,
            global_rank=global_rank,
            world_size=world_size,
        )

    def fixed_seq_collate_fn(self, batch):
        return _fixed_seq_collate_fn(self, batch)

    def sliced_seq_collate_fn(self, batch):
        raise NotImplementedError

    def vad_frame_seq_collate_fn(self, batch):
        return _vad_frame_seq_collate_fn(self, batch)


class AudioToMultiLabelDataset(Dataset):
    """
    Dataset that loads a json file containing paths to audio files, durations (in seconds), and a sequence of labels.
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio_wav_0.wav", "duration": time_in_sec_0, "label": \
        "0 1 1 0 1", "offset": offset_in_sec_0}
    ...
    {"audio_filepath": "/path/to/audio_wav_n.wav", "duration": time_in_sec_n, "label": \
        "0 1 0 0 1", "offset": offset_in_sec_n}
    Args:
        manifest_filepath (Union[str, List[str]]): Path to manifest json as described above. Can
            be comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRSpeechLabel collection.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        window_length_in_sec (float): length of window/slice (in seconds)
            Use this for speaker recognition and VAD tasks.
        shift_length_in_sec (float): amount of shift of window for generating the frame for VAD task in a batch
            Use this for VAD task during inference.
        normalize_audio (bool): Whether to normalize audio signal.
            Defaults to False.
        is_regression_task (bool): Whether the dataset is for a regression task instead of classification.
            Defaults to False.
        cal_labels_occurrence (bool): Whether to calculate occurrence of labels
            Defaults to False.
        delimiter (Optional[str]): Delimiter to use when splitting the label string, default to None.
        normalize_audio_db (Optional[float]):  normalize audio signal to a target db, default to None.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """

        output_types = {
            'audio_signal': NeuralType(
                ('B', 'T'),
                AudioSignal(freq=self._sample_rate)
                if self is not None and hasattr(self, '_sample_rate')
                else AudioSignal(),
            ),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

        if self.is_regression_task:
            output_types.update(
                {
                    'targets': NeuralType(tuple('B, T'), RegressionValuesType()),
                    'targets_length': NeuralType(tuple('B'), LengthsType()),
                }
            )
        else:
            output_types.update(
                {'label': NeuralType(('B', 'T'), LabelsType()), 'label_length': NeuralType(tuple('B'), LengthsType()),}
            )

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: Union[str, List[str]],
        sample_rate: int,
        labels: Optional[List[str]] = None,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim_silence: bool = False,
        is_regression_task: bool = False,
        cal_labels_occurrence: Optional[bool] = False,
        delimiter: Optional[str] = None,
        normalize_audio_db: Optional[float] = None,
    ):
        super().__init__()
        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(',')

        self.delimiter = delimiter
        self.normalize_audio_db = normalize_audio_db

        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            is_regression_task=is_regression_task,
            cal_labels_occurrence=cal_labels_occurrence,
            delimiter=delimiter,
        )

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim_silence
        self.is_regression_task = is_regression_task
        self.id2occurrence = {}
        self.labels_occurrence = None

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

        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(
            sample.audio_file,
            offset=offset,
            duration=sample.duration,
            trim=self.trim,
            normalize_db=self.normalize_audio_db,
        )

        f, fl = features, torch.tensor(features.size(0)).long()

        t = self._label_str_to_tensor(sample.label)

        tl = torch.tensor(t.size(0)).long()

        return f, fl, t, tl

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=0)


class TarredAudioToMultiLabelDataset(IterableDataset):
    """
    A similar Dataset to the AudioToMultiLabelDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToSpeechLabelDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToBPEDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speaker recognition model.
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim(bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        window_length_in_sec (float): time length of window/slice (in seconds) # Pass this only for speaker recognition and VAD task
        shift_length_in_sec (float): amount of shift of window for generating the frame for VAD task. in a batch # Pass this only for VAD task during inference.
        normalize_audio (bool): Whether to normalize audio signal. Defaults to False.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                .. warning::
                    Replicated strategy allows every node to sample the entire set of available tarfiles,
                    and therefore more than one node may sample the same tarfile, and even sample the same
                    data points! As such, there is no assured guarantee that all samples in the dataset will be
                    sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                    occasions (when the number of shards is not divisible with ``world_size``), will not sample
                    the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                    or test datasets.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        delimiter (Optional[str]): Delimiter to use when splitting the label string, default to None.
        normalize_audio_db (Optional[float]):  normalize audio signal to a target db, default to None.
    """

    def __init__(
        self,
        *,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: Union[str, List[str]],
        sample_rate: int,
        labels: Optional[List[str]] = None,
        shuffle_n: int = 0,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim_silence: bool = False,
        is_regression_task: bool = False,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
        delimiter: Optional[str] = None,
        normalize_audio_db: Optional[float] = None,
    ):
        super().__init__()
        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(',')

        self.trim = trim_silence
        self.is_regression_task = is_regression_task
        self.delimiter = delimiter
        self.normalize_audio_db = normalize_audio_db

        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            is_regression_task=is_regression_task,
            index_by_file_id=True,
        )
        self.file_occurence = count_occurence(self.collection.mapping)

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)

        if not is_regression_task:
            self.labels = labels if labels else self._get_label_set()
            self.num_classes = len(self.labels) if self.labels is not None else 1
            self.label2id, self.id2label = {}, {}
            for label_id, label in enumerate(self.labels):
                self.label2id[label] = label_id
                self.id2label[label_id] = label
            for idx in range(len(self.labels[:5])):
                logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))
        else:
            self.labels = []
            self.num_classes = 1

        audio_tar_filepaths = expand_sharded_filepaths(
            sharded_filepaths=audio_tar_filepaths,
            shard_strategy=shard_strategy,
            world_size=world_size,
            global_rank=global_rank,
        )
        # Put together WebDataset
        self._dataset = wds.DataPipeline(
            wds.SimpleShardList(urls=audio_tar_filepaths),
            webdataset_split_by_workers,
            wds.shuffle(shuffle_n),
            wds.tarfile_to_samples(),
            wds.rename(audio=VALID_FILE_FORMATS, key='__key__'),
            wds.to_tuple('audio', 'key'),
            self._filter,
            wds.map(self._build_sample),
        )

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

    def _filter(self, iterator):
        """This function is used to remove samples that have been filtered out by ASRSpeechLabel already.
        Otherwise, we would get a KeyError as _build_sample attempts to find the manifest entry for a sample
        that was filtered out (e.g. for duration).
        Note that if using multi-GPU training, filtering may lead to an imbalance in samples in each shard,
        which may make your code hang as one process will finish before the other.
        """

        class TarredAudioFilter:
            def __init__(self, collection, file_occurence):
                self.iterator = iterator
                self.collection = collection
                self.file_occurence = file_occurence
                self._iterable = self._internal_generator()

            def __iter__(self):
                self._iterable = self._internal_generator()
                return self

            def __next__(self):
                try:
                    values = next(self._iterable)
                except StopIteration:
                    # reset generator
                    self._iterable = self._internal_generator()
                    values = next(self._iterable)

                return values

            def _internal_generator(self):
                """
                WebDataset requires an Iterator, but we require an iterable that yields 1-or-more
                values per value inside self.iterator.

                Therefore wrap the iterator with a generator function that will yield 1-or-more
                values per sample in the iterator.
                """
                for _, tup in enumerate(self.iterator):
                    audio_bytes, audio_filename = tup

                    file_id, _ = os.path.splitext(os.path.basename(audio_filename))
                    if audio_filename in self.file_occurence:
                        for j in range(0, self.file_occurence[file_id]):
                            if j == 0:
                                audio_filename = file_id
                            else:
                                audio_filename = file_id + "-sub" + str(j)
                            yield audio_bytes, audio_filename

        return TarredAudioFilter(self.collection, self.file_occurence)

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info.
        """
        audio_bytes, audio_filename = tup
        # Grab manifest entry from self.collection
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))

        manifest_idx = self.collection.mapping[file_id]
        manifest_entry = self.collection[manifest_idx]

        offset = manifest_entry.offset
        if offset is None:
            offset = 0

        # Convert audio bytes to IO stream for processing (for SoundFile to read)
        audio_filestream = io.BytesIO(audio_bytes)
        features = self.featurizer.process(
            audio_filestream,
            offset=offset,
            duration=manifest_entry.duration,
            trim=self.trim,
            normalize_db=self.normalize_audio_db,
        )

        audio_filestream.close()

        # Audio features
        f, fl = features, torch.tensor(features.shape[0]).long()

        t = self._label_str_to_tensor(manifest_entry.label)

        tl = torch.tensor(t.size(0)).long()

        return f, fl, t, tl

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return len(self.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=0)
