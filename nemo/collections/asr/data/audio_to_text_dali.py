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

import math
import operator
import os.path
import time
from collections.abc import Iterator
from typing import Callable, List, Optional, Union

import torch
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text import ASRManifestProcessor, expand_sharded_filepaths
from nemo.collections.common.parts.preprocessing import parsers
from nemo.utils import logging, model_utils

try:
    import nvidia.dali as dali
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as DALIPytorchIterator
    from nvidia.dali.plugin.pytorch import LastBatchPolicy as LastBatchPolicy

    HAVE_DALI = True
except (ImportError, ModuleNotFoundError):
    HAVE_DALI = False

__all__ = [
    'AudioToCharDALIDataset',
    'AudioToBPEDALIDataset',
]

"""
Below minimum version is required to access the "read_idxs" argument in
dali.fn.readers.nemo_asr
"""
__DALI_MINIMUM_VERSION__ = "1.11"

DALI_INSTALLATION_MESSAGE = (
    "Could not import `nvidia.dali`.\n"
    "Please install DALI by following the steps provided here - \n"
    "https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
)


def is_dali_supported(min_version: str, verbose: bool = False) -> bool:
    """
    Checks if DALI in installed, and version is >= min_verion.

    Args:
        min_version: A semver str that is the minimum requirement.
        verbose: Whether to log the installation instructions if DALI is not found.

    Returns:
        bool - whether DALI could be imported or not.
    """
    module_available, _ = model_utils.check_lib_version(
        'nvidia.dali', checked_version=min_version, operator=operator.ge
    )

    # If DALI is not installed
    if module_available is None:
        if verbose:
            logging.info(DALI_INSTALLATION_MESSAGE)

        return False

    return module_available


class DALIOutputs(object):
    def __init__(self, out_dict):
        self._has_processed_signal = 'processed_signal' in out_dict and 'processed_signal_len' in out_dict
        if not self._has_processed_signal:
            assert 'audio' in out_dict and 'audio_len' in out_dict
        assert 'transcript' in out_dict and 'transcript_len' in out_dict
        if self._has_processed_signal:
            self._outs = (
                out_dict['processed_signal'],
                out_dict['processed_signal_len'].reshape(-1),
                out_dict['transcript'],
                out_dict['transcript_len'].reshape(-1),
            )
        else:
            self._outs = (
                out_dict['audio'],
                out_dict['audio_len'].reshape(-1),
                out_dict['transcript'],
                out_dict['transcript_len'].reshape(-1),
            )

    @property
    def has_processed_signal(self):
        return self._has_processed_signal

    def __getitem__(self, key):
        return self._outs[key]

    def __len__(self):
        return len(self._outs)


class _AudioTextDALIDataset(Iterator):
    """
    NVIDIA DALI pipeline that loads tensors via one or more manifest files where each line containing a sample descriptor in JSON,
    including audio files, transcripts, and durations (in seconds).
    Here's an example:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    Args:
        manifest_filepath: Path to manifest file with the format described above. Can be comma-separated paths.
        device (str): Determines the device type to be used for preprocessing. Allowed values are: 'cpu', 'gpu'.
        batch_size (int): Number of samples in a batch.
        parser (str, callable): A str for an inbuilt parser, or a callable with signature f(str) -> List[int].
        sample_rate (int): Sample rate to resample loaded audio to.
        num_threads (int): Number of CPU processing threads to be created by the DALI pipeline.
        max_duration (float): Determines the maximum allowed duration, in seconds, of the loaded audio files.
        min_duration (float): Determines the minimum allowed duration, in seconds, of the loaded audio files.
        bos_id (int): Id of beginning of sequence symbol to append if not None
        eos_id (int): Id of end of sequence symbol to append if not None
        pad_id (int): Id used to pad the input. Defaults to 0 if not provided.
        trim (bool): If True, it will extract the nonsilent region of the loaded audio signal.
        shuffle (bool): If set to True, the dataset will shuffled after loading.
        drop_last (bool): If set to True, the last batch will be dropped if incomplete. This will be the case when the shard size is not divisible by the batch size.
                          If set to False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
        device_id (int): Index of the GPU to be used (local_rank). Only applicable when device == 'gpu'. Defaults to 0.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 1.
        preprocessor_cfg (DictConfig): Preprocessor configuration. Supports AudioToMelSpectrogramPreprocessor and AudioToMFCCPreprocessor.
        return_sample_id (bool): whether to return the sample_id as a part of each sample (not supported yet).
    """

    def __init__(
        self,
        manifest_filepath: str,
        device: str,
        batch_size: int,
        parser: Union[str, Callable],
        audio_tar_filepaths: Optional[Union[str, List[str]]] = None,
        audio_tar_index_filepaths: Optional[Union[str, List[str]]] = None,
        sample_rate: int = 16000,
        num_threads: int = 4,
        max_duration: float = 0.0,
        min_duration: float = 0.0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        trim: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        shard_strategy: str = "scatter",
        device_id: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
        preprocessor_cfg: DictConfig = None,
        return_sample_id: bool = False,
    ):
        self.drop_last = drop_last  # used by lr_scheduler
        if return_sample_id:
            raise ValueError(
                "Currently DALI data layers don't support returning the sample_id and return_sample_id can not be enabled."
            )
        self.return_sample_id = return_sample_id

        if not HAVE_DALI:
            raise ModuleNotFoundError(
                f"{self} requires NVIDIA DALI to be installed. "
                f"See: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html#id1"
            )

        if device not in ('cpu', 'gpu'):
            raise ValueError(
                f"{self} received an unexpected device argument {device}. Supported values are: 'cpu', 'gpu'"
            )

        device_id = device_id if device == 'gpu' else None

        self.batch_size = batch_size  # Used by NeMo

        self.device = device
        self.device_id = device_id

        if world_size > 1:
            self.shard_id = global_rank
            self.num_shards = world_size
        else:
            self.shard_id = None
            self.num_shards = None

        self.eos_id = eos_id
        self.bos_id = bos_id
        self.sample_rate = sample_rate

        self.pipe = Pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=self.device_id,
            exec_async=True,
            exec_pipelined=True,
        )

        has_preprocessor = preprocessor_cfg is not None
        if has_preprocessor:
            if preprocessor_cfg._target_ == "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor":
                feature_type = "mel_spectrogram"
            elif preprocessor_cfg._target_ == "nemo.collections.asr.modules.AudioToMFCCPreprocessor":
                feature_type = "mfcc"
            else:
                raise ValueError(
                    f"{self} received an unexpected preprocessor configuration: {preprocessor_cfg._target_}."
                    f" Supported preprocessors are: AudioToMelSpectrogramPreprocessor, AudioToMFCCPreprocessor"
                )

            # Default values taken from AudioToMelSpectrogramPreprocessor
            params = preprocessor_cfg
            self.dither = params['dither'] if 'dither' in params else 0.0
            self.preemph = params['preemph'] if 'preemph' in params else 0.97
            self.window_size_sec = params['window_size'] if 'window_size' in params else 0.02
            self.window_stride_sec = params['window_stride'] if 'window_stride' in params else 0.01
            self.sample_rate = params['sample_rate'] if 'sample_rate' in params else sample_rate
            self.window_size = int(self.window_size_sec * self.sample_rate)
            self.window_stride = int(self.window_stride_sec * self.sample_rate)

            normalize = params['normalize'] if 'normalize' in params else 'per_feature'
            if normalize == 'per_feature':  # Each freq channel independently
                self.normalization_axes = (1,)
            elif normalize == 'all_features':
                self.normalization_axes = (0, 1)
            else:
                raise ValueError(
                    f"{self} received {normalize} for the normalize parameter."
                    f" It must be either 'per_feature' or 'all_features'."
                )

            self.window = None
            window_name = params['window'] if 'window' in params else 'hann'
            torch_windows = {
                'hann': torch.hann_window,
                'hamming': torch.hamming_window,
                'blackman': torch.blackman_window,
                'bartlett': torch.bartlett_window,
                'none': None,
            }

            if window_name == 'ones':
                window_tensor = torch.ones(self.window_size)
            else:
                try:
                    window_fn = torch_windows.get(window_name, None)
                except:
                    raise ValueError(
                        f"{self} received '{window_name}' for the window parameter."
                        f" It must be one of: ('hann', 'ones', 'hamming', 'blackman', 'bartlett', None)."
                        f" None is equivalent to 'hann'."
                    )
                window_tensor = window_fn(self.window_size, periodic=False) if window_fn else None
            self.window = window_tensor.numpy().tolist() if window_tensor is not None else None

            self.n_fft = params['n_fft'] if 'n_fft' in params else 2 ** math.ceil(math.log2(self.window_size))
            self.n_mels = params['n_mels'] if 'n_mels' in params else 64
            self.n_mfcc = params['n_mfcc'] if 'n_mfcc' in params else 64

            features = params['features'] if 'features' in params else 0
            if features > 0:
                if feature_type == 'mel_spectrogram':
                    self.n_mels = features
                elif feature_type == 'mfcc':
                    self.n_mfcc = features

            # TODO Implement frame splicing
            if 'frame_splicing' in params:
                assert params['frame_splicing'] == 1, "Frame splicing is not implemented"

            self.freq_low = params['lowfreq'] if 'lowfreq' in params else 0.0
            self.freq_high = params['highfreq'] if 'highfreq' in params else self.sample_rate / 2.0
            self.log_features = params['log'] if 'log' in params else True

            # We want to avoid taking the log of zero
            # There are two options: either adding or clamping to a small value

            self.log_zero_guard_type = params['log_zero_guard_type'] if 'log_zero_guard_type' in params else 'add'
            if self.log_zero_guard_type not in ["add", "clamp"]:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_type} for the "
                    f"log_zero_guard_type parameter. It must be either 'add' or "
                    f"'clamp'."
                )

            self.log_zero_guard_value = (
                params['log_zero_guard_value'] if 'log_zero_guard_value' in params else 2 ** -24
            )
            if isinstance(self.log_zero_guard_value, str):
                if self.log_zero_guard_value == "tiny":
                    self.log_zero_guard_value = torch.finfo(torch.float32).tiny
                elif self.log_zero_guard_value == "eps":
                    self.log_zero_guard_value = torch.finfo(torch.float32).eps
                else:
                    raise ValueError(
                        f"{self} received {self.log_zero_guard_value} for the log_zero_guard_type parameter."
                        f"It must be either a number, 'tiny', or 'eps'"
                    )

            self.mag_power = params['mag_power'] if 'mag_power' in params else 2
            if self.mag_power != 1.0 and self.mag_power != 2.0:
                raise ValueError(
                    f"{self} received {self.mag_power} for the mag_power parameter." f" It must be either 1.0 or 2.0."
                )

            self.pad_to = max(params['pad_to'], 1) if 'pad_to' in params else 16
            self.pad_value = params['pad_value'] if 'pad_value' in params else 0.0

        with self.pipe:
            if audio_tar_filepaths is None and audio_tar_index_filepaths is None:
                audio, indices = dali.fn.readers.nemo_asr(
                    name="Reader",
                    manifest_filepaths=manifest_filepath.split(','),
                    dtype=dali.types.FLOAT,
                    downmix=True,
                    sample_rate=float(self.sample_rate),
                    min_duration=min_duration,
                    max_duration=max_duration,
                    read_sample_rate=False,
                    read_text=False,
                    read_idxs=True,
                    random_shuffle=shuffle,
                    shard_id=self.shard_id,
                    num_shards=self.num_shards,
                    pad_last_batch=True,
                )

                self.is_tarred_dataset = False

            elif audio_tar_filepaths is not None and audio_tar_index_filepaths is not None:
                audio_tar_filepaths = expand_sharded_filepaths(
                    audio_tar_filepaths, shard_strategy=shard_strategy, world_size=world_size, global_rank=global_rank
                )
                audio_tar_index_filepaths = expand_sharded_filepaths(
                    audio_tar_index_filepaths,
                    shard_strategy=shard_strategy,
                    world_size=world_size,
                    global_rank=global_rank,
                )

                if len(audio_tar_filepaths) != len(audio_tar_index_filepaths) and len(audio_tar_index_filepaths) != 0:
                    raise ValueError(
                        f"Number of filepaths provided for `audio_tar_filepaths` must match "
                        f"`audio_tar_index_filepaths`. Got {len(audio_tar_filepaths)} audio_tar_filepaths and "
                        f"{len(audio_tar_index_filepaths)} audio_tar_index_filepaths."
                    )

                tar_file = dali.fn.readers.webdataset(
                    paths=audio_tar_filepaths,
                    index_paths=audio_tar_index_filepaths,
                    name="Reader",
                    ext=["wav"],
                    missing_component_behavior="error",
                    random_shuffle=shuffle,
                    shard_id=self.shard_id,
                    num_shards=self.num_shards,
                    pad_last_batch=True,
                )
                audio, _ = dali.fn.decoders.audio(
                    tar_file, dtype=dali.types.FLOAT, downmix=True, sample_rate=float(self.sample_rate),
                )
                indices = dali.fn.get_property(tar_file, key="source_info")
                indices = dali.fn.pad(indices)

                self.is_tarred_dataset = True

            else:
                raise RuntimeError(
                    "When using DALI datasets, either `audio_tar_filepaths` "
                    "and `audio_tar_index_filepaths` should either both be None (sequential dataset)"
                    "or provided (tarred dataset)."
                )

            # Extract nonsilent region, if necessary
            if trim:
                # Need to extract non-silent region before moving to the GPU
                roi_start, roi_len = dali.fn.nonsilent_region(audio, cutoff_db=-60)
                audio = audio.gpu() if self.device == 'gpu' else audio
                audio = dali.fn.slice(
                    audio, roi_start, roi_len, normalized_anchor=False, normalized_shape=False, axes=[0]
                )
            else:
                audio = audio.gpu() if self.device == 'gpu' else audio

            if not has_preprocessor:
                # No preprocessing, the output is the audio signal
                audio_len = dali.fn.shapes(dali.fn.reshape(audio, shape=[-1]))
                audio = dali.fn.pad(audio)
                self.pipe.set_outputs(audio, audio_len, indices)
            else:
                # Additive gaussian noise (dither)
                if self.dither > 0.0:
                    gaussian_noise = dali.fn.random.normal(audio)
                    audio = audio + self.dither * gaussian_noise

                # Preemphasis filter
                if self.preemph > 0.0:
                    audio = dali.fn.preemphasis_filter(audio, preemph_coeff=self.preemph, border='zero')

                # Power spectrogram
                spec = dali.fn.spectrogram(
                    audio,
                    nfft=self.n_fft,
                    window_length=self.window_size,
                    window_step=self.window_stride,
                    window_fn=self.window,
                )

                if feature_type == 'mel_spectrogram' or feature_type == 'mfcc':
                    # Spectrogram to Mel Spectrogram
                    spec = dali.fn.mel_filter_bank(
                        spec,
                        sample_rate=self.sample_rate,
                        nfilter=self.n_mels,
                        normalize=True,
                        freq_low=self.freq_low,
                        freq_high=self.freq_high,
                    )
                    # Mel Spectrogram to MFCC
                    if feature_type == 'mfcc':
                        spec = dali.fn.mfcc(spec, n_mfcc=self.n_mfcc)

                # Logarithm
                if self.log_zero_guard_type == 'add':
                    spec = spec + self.log_zero_guard_value

                spec = dali.fn.to_decibels(
                    spec, multiplier=math.log(10), reference=1.0, cutoff_db=math.log(self.log_zero_guard_value)
                )

                # Normalization
                spec = dali.fn.normalize(spec, axes=self.normalization_axes, epsilon=1e-5 ** 2, ddof=1)

                # Extracting the length of the spectrogram
                spec_len = dali.fn.slice(dali.fn.shapes(spec), 1, 1, axes=(0,))

                # Pads feature dimension to be a multiple of `pad_to` and the temporal dimension to be as big as the largest sample (shape -1)
                spec = dali.fn.pad(spec, fill_value=self.pad_value, axes=(0, 1), align=(self.pad_to, 1), shape=(1, -1))
                self.pipe.set_outputs(spec, spec_len, indices)

        x = time.time()
        # Building DALI pipeline
        self.pipe.build()
        y = time.time()

        logging.info(f"Time for pipe.build() : {(y - x)} seconds")

        if has_preprocessor:
            output_names = ['processed_signal', 'processed_signal_len', 'manifest_indices']
        else:
            output_names = ['audio', 'audio_len', 'manifest_indices']

        x = time.time()
        last_batch_policy = LastBatchPolicy.DROP if drop_last else LastBatchPolicy.PARTIAL
        self._iter = DALIPytorchIterator(
            [self.pipe],
            output_map=output_names,
            reader_name="Reader",
            last_batch_policy=last_batch_policy,
            dynamic_shape=True,
            auto_reset=True,
        )
        y = time.time()
        logging.info(f"Time for DALIPytorchIterator to initialize : {(y - x)} seconds")

        # TODO come up with a better solution
        class DummyDataset:
            def __init__(self, parent):
                self.parent = parent

            def __len__(self):
                return self.parent.size

        self.dataset = DummyDataset(self)  # Used by NeMo

        x = time.time()
        self.manifest_processor = ASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=0,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            index_by_file_id=self.is_tarred_dataset,
        )
        y = time.time()
        logging.info(f"Time to build nemo manifest processor - {(y - x)} seconds")

    def reset(self):
        self._iter.reset()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    @property
    def size(self):
        return self._iter.size

    def __len__(self):
        return len(self._iter)

    def __next__(self):
        outputs = self._iter.next()
        assert len(outputs) == 1
        dali_out = outputs[0]
        manifest_indices = dali_out['manifest_indices'].numpy()

        out = {}
        out_names = ['processed_signal', 'processed_signal_len', 'audio', 'audio_len']
        for out_name in out_names:
            if out_name in dali_out:
                out[out_name] = dali_out[out_name].detach().clone()

        text_tokens = []
        text_tokens_len = []
        max_len = 0
        batch_size = manifest_indices.shape[0]
        for i, manifest_index in enumerate(manifest_indices):

            if not self.is_tarred_dataset:
                # Loose-file dataset. Index is integer based.
                manifest_index = manifest_index[0]
                text, text_length = self.manifest_processor.process_text_by_id(manifest_index)
            else:
                # Tarred-file dataset. Index is filename based.
                resolved_manifest_indices = manifest_index.tobytes().decode().split(":")
                resolved_manifest_index = resolved_manifest_indices[2]  # we require just the filename segment
                resolved_manifest_index = os.path.splitext(resolved_manifest_index)[0]  # we dont need file extension
                text, text_length = self.manifest_processor.process_text_by_file_id(resolved_manifest_index)

            text_tokens_len.append(text_length)
            text_tokens.append(text)
            if text_length > max_len:
                max_len = text_length

        transcript_out = torch.full([batch_size, max_len], fill_value=self.manifest_processor.pad_id, dtype=torch.long)
        for i, n in enumerate(text_tokens_len):
            transcript_out[i, :n] = torch.tensor(text_tokens[i], dtype=torch.long)
        transcript_len_out = torch.tensor(text_tokens_len, dtype=torch.long)

        out['transcript'] = transcript_out
        out['transcript_len'] = transcript_len_out
        return DALIOutputs(out)


class AudioToCharDALIDataset(_AudioTextDALIDataset):
    """
    Character based NVIDIA DALI pipeline that loads tensors via one or more manifest files where each line containing a
    sample descriptor in JSON, including audio files, transcripts, and durations (in seconds).
    Here's an example:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    Args:
        manifest_filepath: Path to manifest file with the format described above. Can be comma-separated paths.
        device (str): Determines the device type to be used for preprocessing. Allowed values are: 'cpu', 'gpu'.
        batch_size (int): Number of samples in a batch.
        labels (List[str]): String containing all the possible characters to map to.
        sample_rate (int): Sample rate to resample loaded audio to.
        num_threads (int): Number of CPU processing threads to be created by the DALI pipeline.
        max_duration (float): Determines the maximum allowed duration, in seconds, of the loaded audio files.
        min_duration (float): Determines the minimum allowed duration, in seconds, of the loaded audio files.
        blank_index (int): blank character index, default = -1
        unk_index (int): unk_character index, default = -1
        normalize (bool): whether to normalize transcript text (default): True
        bos_id (int): Id of beginning of sequence symbol to append if not None
        eos_id (int): Id of end of sequence symbol to append if not None
        pad_id (int): Id used to pad the input. Defaults to 0 if not provided.
        trim (bool): If True, it will extract the nonsilent region of the loaded audio signal.
        shuffle (bool): If set to True, the dataset will shuffled after loading.
        drop_last (bool): If set to True, the last batch will be dropped if incomplete. This will be the case when the shard size is not divisible by the batch size.
                          If set to False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
        parser (str, callable): A str for an inbuilt parser, or a callable with signature f(str) -> List[int].
        device_id (int): Index of the GPU to be used (local_rank). Only applicable when device == 'gpu'. Defaults to 0.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 1.
        preprocessor_cfg (DictConfig): Preprocessor configuration. Supports AudioToMelSpectrogramPreprocessor and AudioToMFCCPreprocessor.
        return_sample_id (bool): whether to return the sample_id as a part of each sample (not supported yet).
    """

    def __init__(
        self,
        manifest_filepath: str,
        device: str,
        batch_size: int,
        labels: Union[str, List[str]],
        sample_rate: int = 16000,
        audio_tar_filepaths: Optional[Union[str, List[str]]] = None,
        audio_tar_index_filepaths: Optional[Union[str, List[str]]] = None,
        num_threads: int = 4,
        max_duration: float = 0.0,
        min_duration: float = 0.0,
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        trim: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        parser: Union[str, Callable] = 'en',
        shard_strategy: str = "scatter",
        device_id: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
        preprocessor_cfg: DictConfig = None,
        return_sample_id: bool = False,
    ):
        self.labels = labels

        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )

        super().__init__(
            manifest_filepath=manifest_filepath,
            device=device,
            batch_size=batch_size,
            audio_tar_filepaths=audio_tar_filepaths,
            audio_tar_index_filepaths=audio_tar_index_filepaths,
            sample_rate=sample_rate,
            num_threads=num_threads,
            max_duration=max_duration,
            min_duration=min_duration,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            trim=trim,
            shuffle=shuffle,
            drop_last=drop_last,
            parser=parser,
            shard_strategy=shard_strategy,
            device_id=device_id,
            global_rank=global_rank,
            world_size=world_size,
            preprocessor_cfg=preprocessor_cfg,
            return_sample_id=return_sample_id,
        )


class AudioToBPEDALIDataset(_AudioTextDALIDataset):
    """
    Subword based NVIDIA DALI pipeline that loads tensors via one or more manifest files where each line containing a
    sample descriptor in JSON, including audio files, transcripts, and durations (in seconds).
    Here's an example:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    Args:
        manifest_filepath: Path to manifest file with the format described above. Can be comma-separated paths.
        tokenizer (TokenizerSpec): A TokenizerSpec implementation that wraps a tokenization implementation.
        device (str): Determines the device type to be used for preprocessing. Allowed values are: 'cpu', 'gpu'.
        batch_size (int): Number of samples in a batch.
        sample_rate (int): Sample rate to resample loaded audio to.
        num_threads (int): Number of CPU processing threads to be created by the DALI pipeline.
        max_duration (float): Determines the maximum allowed duration, in seconds, of the loaded audio files.
        min_duration (float): Determines the minimum allowed duration, in seconds, of the loaded audio files.
        bos_id (int): Id of beginning of sequence symbol to append if not None. Injected from the tokenizer.
        eos_id (int): Id of end of sequence symbol to append if not None. Injected from the tokenizer.
        pad_id (int): Id used to pad the input. Defaults to 0 if not provided. Injected from the tokenizer.
        trim (bool): If True, it will extract the nonsilent region of the loaded audio signal.
        shuffle (bool): If set to True, the dataset will shuffled after loading.
        drop_last (bool): If set to True, the last batch will be dropped if incomplete. This will be the case when the shard size is not divisible by the batch size.
                          If set to False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.

        device_id (int): Index of the GPU to be used (local_rank). Only applicable when device == 'gpu'. Defaults to 0.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 1.
        preprocessor_cfg (DictConfig): Preprocessor configuration. Supports AudioToMelSpectrogramPreprocessor and AudioToMFCCPreprocessor.
        use_start_end_token (bool): Boolean which dictates whether to add [BOS] and [EOS] tokens to beginning and
            ending of speech respectively.
        return_sample_id (bool): whether to return the sample_id as a part of each sample (not supported yet).
    """

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        device: str,
        batch_size: int,
        sample_rate: int = 16000,
        audio_tar_filepaths: Optional[Union[str, List[str]]] = None,
        audio_tar_index_filepaths: Optional[Union[str, List[str]]] = None,
        num_threads: int = 4,
        max_duration: float = 0.0,
        min_duration: float = 0.0,
        trim: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        shard_strategy: str = "scatter",
        device_id: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
        preprocessor_cfg: DictConfig = None,
        use_start_end_token: bool = True,
        return_sample_id: bool = False,
    ):

        if use_start_end_token and hasattr(tokenizer, 'bos_token'):
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, 'eos_token'):
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, 'pad_token'):
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                self._tokenizer = tokenizer

            def __call__(self, text):
                t = self._tokenizer.text_to_ids(text)
                return t

        super().__init__(
            manifest_filepath=manifest_filepath,
            device=device,
            batch_size=batch_size,
            sample_rate=sample_rate,
            audio_tar_filepaths=audio_tar_filepaths,
            audio_tar_index_filepaths=audio_tar_index_filepaths,
            num_threads=num_threads,
            max_duration=max_duration,
            min_duration=min_duration,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            trim=trim,
            shuffle=shuffle,
            drop_last=drop_last,
            parser=TokenizerWrapper(tokenizer),
            shard_strategy=shard_strategy,
            device_id=device_id,
            global_rank=global_rank,
            world_size=world_size,
            preprocessor_cfg=preprocessor_cfg,
            return_sample_id=return_sample_id,
        )
