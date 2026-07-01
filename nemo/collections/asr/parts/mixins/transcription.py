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

import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment, ChannelSelectorType
from nemo.collections.asr.parts.utils import manifest_utils
from nemo.collections.asr.parts.utils.chunking_utils import merge_flat_chunk_hypotheses
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.data.utils import move_data_to_device
from nemo.utils import logging, logging_mode

TranscriptionReturnType = Union[List[str], List[Hypothesis], Tuple[List[str]], Tuple[List[Hypothesis]]]
GenericTranscriptionType = Union[List[Any], List[List[Any]], Tuple[Any], Tuple[List[Any]], Dict[str, List[Any]]]


@dataclass
class InternalTranscribeConfig:
    # Internal values
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    training_mode: bool = False
    logging_level: Optional[Any] = None

    # Preprocessor values
    dither_value: float = 0.0
    pad_to_value: int = 0

    # Scratch space
    temp_dir: Optional[str] = None
    manifest_filepath: Optional[str] = None


@dataclass
class TranscribeConfig:
    use_lhotse: bool = True
    batch_size: int = 4
    return_hypotheses: bool = False
    num_workers: Optional[int] = None
    channel_selector: ChannelSelectorType = None
    augmentor: Optional[DictConfig] = None
    timestamps: Optional[bool] = None  # returns timestamps for each word and segments if model supports punctuations
    verbose: bool = True
    # Utility
    partial_hypothesis: Optional[List[Any]] = None
    enable_chunking: Optional[bool] = False
    _internal: Optional[InternalTranscribeConfig] = None
    chunk_range: Optional[List[int]] = field(default_factory=lambda: [240, 300])


def get_value_from_transcription_config(trcfg, key, default):
    """
    Utility function to get a value from the transcription config.
    If the value is not present in the transcription config, the default value is returned.

    Args:
        trcfg: A dataclass that represents the transcription config.
        key: The name of the arg to retrieve.
        default: The default value to return if the key is not present in the transcription config.

    Returns:
        The value of the key in the transcription config or the default value.
    """
    if hasattr(trcfg, key):
        return getattr(trcfg, key)
    else:
        logging.debug(
            f"Using default value of {default} for {key} because it is not present \
                in the transcription config {trcfg}."
        )
        return default


def resolve_chunking(
    audio: Union[str, List[str], np.ndarray, torch.Tensor, DataLoader],
    enable_chunking: bool,
) -> bool:
    """Returns True when chunking is requested and audio is not an external DataLoader."""
    return enable_chunking and not isinstance(audio, DataLoader)


class TranscriptionTensorDataset(Dataset):
    """
    Dataset for transcribing audio tensors.

    Chunking:
    If `enable_chunking` is True, each audio sample is pre-split into optimally sized overlapping
    chunks at construction time. The dataset then has one entry per chunk and each item is a 4-tuple
    ``(chunk, chunk_len, sample_idx, chunk_start_samples)`` where the last two scalars carry the
    provenance metadata needed by the post-inference merge logic.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.audio_tensors = config['audio_tensors']
        self.channel_selector = config['channel_selector']
        self.augmentor_cfg = config.get('augmentor', None)
        self.sample_rate = config['sample_rate']
        self.pad_min_duration = config.get('pad_min_duration', 1.0)
        self.pad_direction = config.get('pad_direction', 'both')
        self.pad_min_samples = int(self.pad_min_duration * self.sample_rate)
        self.enable_chunking = config.get('enable_chunking', False)
        self.chunk_range = config.get('chunk_range', [240, 300])

        if self.augmentor_cfg is not None:
            self.augmentor = process_augmentations(self.augmentor_cfg, global_rank=0, world_size=1)
        else:
            self.augmentor = None

        if self.enable_chunking:
            # Pre-expand every audio tensor into its chunks so the dataloader can use
            # any batch size and each item already carries (sample_idx, chunk_start_samples).
            from nemo.collections.asr.parts.utils.chunking_utils import chunk_waveform

            self._chunks: List[Tuple[torch.Tensor, int, int, int]] = []
            for sample_idx, tensor in enumerate(self.audio_tensors):
                tensor = self._load_and_pad(tensor)
                chunks, lens, starts = chunk_waveform(
                    waveform=tensor,
                    chunk_range=self.chunk_range,
                    overlap_sec=1.0,
                    sample_rate=self.sample_rate,
                )
                for chunk, length, start in zip(chunks, lens, starts):
                    self._chunks.append((chunk, length, sample_idx, start))
            self.length = len(self._chunks)
        else:
            self.length = len(self.audio_tensors)

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError(f"Index {index} out of range for dataset of size {self.length}")

        return self.get_item(index)

    def __len__(self):
        return self.length

    def _pad_audio(self, samples: torch.Tensor) -> torch.Tensor:
        """Pad audio to minimum duration, matching Lhotse dataloader behavior."""
        current_len = samples.shape[0]
        if current_len >= self.pad_min_samples:
            return samples

        pad_total = self.pad_min_samples - current_len
        if self.pad_direction == 'both':
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
        elif self.pad_direction == 'left':
            pad_left = pad_total
            pad_right = 0
        else:  # right (default)
            pad_left = 0
            pad_right = pad_total
        samples = torch.nn.functional.pad(samples, (pad_left, pad_right), mode='constant', value=0.0)
        return samples

    def _load_and_pad(self, samples: torch.Tensor) -> torch.Tensor:
        """Apply augmentation (if any) and minimum-duration padding."""
        if self.augmentor is not None:
            logging.warning(
                "Audio Augmentations are being applied during inference by moving the tensor onto CPU. "
                "This is highly inefficient and therefore not recommended.",
                mode=logging_mode.ONCE,
            )

            original_dtype = samples.dtype
            samples = samples.to(device='cpu', dtype=torch.float32).numpy()
            segment = AudioSegment(
                samples, self.sample_rate, target_sr=self.sample_rate, channel_selector=self.channel_selector
            )
            samples = self.augmentor.perturb(segment)
            samples = torch.tensor(samples.samples, dtype=original_dtype)

        return self._pad_audio(samples)

    def get_item(self, index):
        if self.enable_chunking:
            chunk, length, sample_idx, start_samples = self._chunks[index]
            seq_len = torch.tensor(length, dtype=torch.long)
            # Return (audio, audio_len, sample_idx, chunk_start_samples) so that
            # _transcribe_output_processing can assign stable IDs and correct offsets.
            return (
                chunk,
                seq_len,
                torch.tensor(sample_idx, dtype=torch.long),
                torch.tensor(start_samples, dtype=torch.long),
            )

        samples = self._load_and_pad(self.audio_tensors[index])
        seq_len = torch.tensor(samples.shape[0], dtype=torch.long)
        # Typically NeMo ASR models expect the mini-batch to be a 4-tuple of (audio, audio_len, text, text_len).
        # For inference, we set text and text_len to None to not disrupt the shape of the tuple.
        return samples, seq_len, None, None


def _pre_chunked_tensor_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for pre-chunked tensor datasets (``TranscriptionTensorDataset`` with
    ``enable_chunking=True``).

    Each item is ``(chunk, chunk_len, sample_idx, chunk_start_samples)`` — all tensors.
    The function pads chunks to a uniform length and stacks everything.

    Returns:
        audio: ``(B, T)`` padded chunk audio.
        audio_lens: ``(B,)`` true (unpadded) length of each chunk in samples.
        sample_ids: ``(B,)`` original sample index for each chunk (stable ID for merging).
        chunk_starts: ``(B,)`` start offset of each chunk within its original audio, in samples.
    """
    chunks, lens, sample_ids, starts = zip(*batch)
    max_len = max(c.shape[0] for c in chunks)
    padded = []
    for chunk in chunks:
        pad_len = max_len - chunk.shape[0]
        if pad_len > 0:
            chunk = torch.nn.functional.pad(chunk, (0, pad_len))
        padded.append(chunk)
    return (
        torch.stack(padded),
        torch.stack(lens),
        torch.stack(sample_ids),
        torch.stack(starts),
    )


class TranscriptionMixin(ABC):
    """
    An abstract class for transcribe-able models.

    Creates a template function `transcribe()` that provides an interface to perform transcription of audio tensors or
    filepaths.

    The following abstract classes must be implemented by the subclass:

        - `_transcribe_input_manifest_processing()`:
            Process the provided input arguments (filepaths only) and return a
            config dict for the dataloader. The data loader is should generally operate on NeMo manifests.

        - `_setup_transcribe_dataloader()`:
            Setup the dataloader for transcription. Receives the output from
            `_transcribe_input_manifest_processing()`.

        - `_transcribe_forward()`:
            Implements the model's custom forward pass to return outputs that are processed by
            `_transcribe_output_processing()`.

        - `_transcribe_output_processing()`:
            Implements the post processing of the model's outputs to return the results to
            the user. The result can be a list of objects, list of list of objects, tuple of objects, tuple of list of
            objects, or a dict of list of objects.

    """

    def _is_chunking_compatible_decoding(self) -> bool:
        """
        Check if the current decoding strategy is compatible with chunking.

        Beam search with `return_best_hypothesis=False` returns multiple hypotheses per chunk,
        which is incompatible with the current chunking merge implementation that expects
        a single hypothesis per chunk.

        Returns:
            bool: True if decoding is compatible with chunking, False otherwise.
        """
        # Check if the model has a decoding attribute (RNNT/TDT models)
        if not hasattr(self, 'decoding'):
            return True

        # Check if decoding has an inner decoder (AbstractRNNTDecoding stores decoder as self.decoding)
        inner_decoder = getattr(self.decoding, 'decoding', None)
        if inner_decoder is None:
            return True

        # Check if the inner decoder has return_best_hypothesis attribute (beam decoders have this)
        return_best_hypothesis = getattr(inner_decoder, 'return_best_hypothesis', True)

        if not return_best_hypothesis:
            logging.warning(
                "Chunking is not compatible with beam search when `return_best_hypothesis=False`. "
                "The decoding returns multiple hypotheses per chunk, but chunking merge expects "
                "a single hypothesis per chunk. Disabling chunking. "
                "Set `decoding.beam.return_best_hypothesis=True` to enable chunking with beam search."
            )
            return False

        return True

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        use_lhotse: bool = True,
        batch_size: int = 4,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        timestamps: Optional[bool] = None,
        enable_chunking: bool = False,
        override_config: Optional[TranscribeConfig] = None,
        **config_kwargs,
    ) -> GenericTranscriptionType:
        """
        Template function that defines the execution strategy for transcribing audio.

        Args:
            audio: (a single or list) of paths to audio files or a np.ndarray audio array.
                Can also be a dataloader object that provides values that can be consumed by the model.
                Recommended length per file is between 5 and 25 seconds.
                But it is possible to pass a few hours long file if enough GPU memory is available.
            use_lhotse: (bool) If audio is not a dataloder, defines whether to create a lhotse dataloader or a
                non-lhotse dataloader.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring.
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from
                multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set
                to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            timestamps: Optional(Bool): timestamps will be returned if set to True as part of hypothesis object
                (output.timestep['segment']/output.timestep['word']). Refer to `Hypothesis` class for more details.
                Default is None and would retain the previous state set by using self.change_decoding_strategy().
            override_config: (Optional[TranscribeConfig]) override transcription config pre-defined by the user.
                **Note**: All other arguments in the function will be ignored if override_config is passed.
                You should call this argument as `model.transcribe(audio, override_config=TranscribeConfig(...))`.
            **config_kwargs: (Optional[Dict]) additional arguments to override the default TranscribeConfig.
                Note: If override_config is passed, these arguments will be ignored.

        Returns:
            Output is defined by the subclass implementation of `TranscriptionMixin._transcribe_output_processing()`.
            It can be:

                - List[str/Hypothesis]

                - List[List[str/Hypothesis]]

                - Tuple[str/Hypothesis]

                - Tuple[List[str/Hypothesis]]

                - Dict[str, List[str/Hypothesis]]
        """
        if override_config is None:
            transcribe_cfg = TranscribeConfig(
                use_lhotse=use_lhotse,
                batch_size=batch_size,
                return_hypotheses=return_hypotheses,
                num_workers=num_workers,
                channel_selector=channel_selector,
                augmentor=augmentor,
                verbose=verbose,
                timestamps=timestamps,
                enable_chunking=enable_chunking,
                **config_kwargs,
            )
        else:
            if not hasattr(override_config, '_internal'):
                raise ValueError(
                    "`transcribe_cfg must have an `_internal` argument, which must be of an object of type "
                    "InternalTranscribeConfig or its subclass."
                )

            if override_config._internal is None:
                override_config._internal = InternalTranscribeConfig()

            transcribe_cfg = override_config
        # Add new internal config
        if transcribe_cfg._internal is None:
            transcribe_cfg._internal = InternalTranscribeConfig()
        else:
            # Check if internal config is valid
            if not isinstance(transcribe_cfg._internal, InternalTranscribeConfig):
                raise ValueError(
                    "`transcribe_cfg._internal` must be of an object of type InternalTranscribeConfig or "
                    "its subclass"
                )
        # Classification and regression models never use chunking
        if type(transcribe_cfg).__name__ in ('ClassificationInferConfig', 'RegressionInferConfig'):
            transcribe_cfg.enable_chunking = False
        else:
            transcribe_cfg.enable_chunking = (
                resolve_chunking(
                    audio=audio,
                    enable_chunking=transcribe_cfg.enable_chunking,
                )
                and self._is_chunking_compatible_decoding()
            )
        # Hold the results here
        results = None  # type: GenericTranscriptionType
        try:
            generator = self.transcribe_generator(audio, override_config=transcribe_cfg)

            for processed_outputs in generator:
                # Store results
                if isinstance(processed_outputs, list):
                    # Create a results of the same type as each element in processed_outputs
                    if results is None:
                        results = []

                    results.extend(processed_outputs)

                elif isinstance(processed_outputs, dict):
                    # Create a results of the same type as each element in processed_outputs
                    if results is None:
                        results = processed_outputs
                    else:
                        for k, v in processed_outputs.items():
                            results[k].extend(v)

                elif isinstance(processed_outputs, tuple):
                    # Create a results of the same type as each element in processed_outputs
                    if results is None:
                        results = tuple([[] for _ in processed_outputs])

                    # If nested list structure
                    if isinstance(processed_outputs[0], list):
                        for i, processed_output in enumerate(processed_outputs):
                            results[i].extend(processed_output)
                    else:
                        # If flat list structure
                        if len(processed_outputs) != len(results):
                            raise RuntimeError(
                                f"The number of elements in the result ({len(results)}) does not "
                                f"match the results of the current batch ({len(processed_outputs)})."
                            )

                        for i, processed_output in enumerate(processed_outputs):
                            results[i].append(processed_output)

                else:
                    raise NotImplementedError(
                        "Given output result for transcription is not supported. "
                        "Please return a list of results, list of list of results, "
                        "a dict of list of results, or "
                        "a tuple of list of results."
                    )
        except StopIteration:
            pass
        if transcribe_cfg.enable_chunking and isinstance(results, list):
            _timestamps = override_config.timestamps if override_config is not None else timestamps
            _subsampling = self.encoder.subsampling_factor if hasattr(self.encoder, 'subsampling_factor') else 8
            _window_stride = (
                self.cfg['preprocessor']['window_stride']
                if hasattr(self, 'cfg') and 'preprocessor' in self.cfg
                else None
            )
            # Vocabulary for char-based models (RNNT: joint.vocabulary, CTC: decoder.vocabulary)
            _vocabulary = getattr(getattr(self, 'joint', None), 'vocabulary', None) or getattr(
                getattr(self, 'decoder', None), 'vocabulary', None
            )
            results = merge_flat_chunk_hypotheses(
                hypotheses_list=results,
                timestamps=_timestamps,
                tokenizer=getattr(self, 'tokenizer', None),
                subsampling_factor=_subsampling,
                window_stride=_window_stride,
                vocabulary=_vocabulary,
                return_hypotheses=transcribe_cfg.return_hypotheses,
            )
        return results

    def transcribe_generator(self, audio, override_config: Optional[TranscribeConfig]):
        """
        A generator version of `transcribe` function.
        """
        if override_config is None:
            override_config = TranscribeConfig()

        if not hasattr(override_config, '_internal'):
            raise ValueError(
                "`transcribe_cfg must have an `_internal` argument, which must be of an object of type "
                "InternalTranscribeConfig or its subclass."
            )

        # Add new internal config
        if override_config._internal is None:
            override_config._internal = InternalTranscribeConfig()
        else:
            # Check if internal config is valid
            if not isinstance(override_config._internal, InternalTranscribeConfig):
                raise ValueError(
                    "`transcribe_cfg._internal` must be of an object of type InternalTranscribeConfig or "
                    "its subclass"
                )

        transcribe_cfg = override_config

        try:
            # Initialize and assert the transcription environment
            self._transcribe_on_begin(audio, transcribe_cfg)

            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                transcribe_cfg._internal.temp_dir = tmpdir

                # Create a DataLoader if not already present
                if not isinstance(audio, DataLoader):
                    dataloader = self._transcribe_input_processing(audio, transcribe_cfg)
                else:
                    dataloader = audio

                if hasattr(transcribe_cfg, 'verbose'):
                    verbose = transcribe_cfg.verbose
                else:
                    verbose = True

                for test_batch in tqdm(dataloader, desc="Transcribing", disable=not verbose):
                    # Move batch to device
                    test_batch = move_data_to_device(test_batch, transcribe_cfg._internal.device)
                    # Run forward pass
                    model_outputs = self._transcribe_forward(test_batch, transcribe_cfg)
                    processed_outputs = self._transcribe_output_processing(model_outputs, transcribe_cfg)

                    # clear up memory
                    del test_batch, model_outputs

                    # Yield results if generator
                    yield processed_outputs

                    del processed_outputs

        finally:
            # set mode back to its original value
            self._transcribe_on_end(transcribe_cfg)

    """
    Transcribe Execution Flow
    """

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        """
        Internal function to setup the model for transcription. Perform all setup and pre-checks here.

        Args:
            audio: Of type `GenericTranscriptionType`
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        if audio is None:
            return {}

        if isinstance(audio, (str, np.ndarray, torch.Tensor)):
            audio = [audio]

        if isinstance(audio, list) and len(audio) == 0:
            return {}

        _params = next(self.parameters())
        if trcfg._internal.device is None:
            trcfg._internal.device = _params.device

        if trcfg._internal.dtype is None:
            trcfg._internal.dtype = _params.dtype

        # Set num_workers
        num_workers = get_value_from_transcription_config(trcfg, 'num_workers', default=0)

        if num_workers is None:
            _batch_size = get_value_from_transcription_config(trcfg, 'batch_size', default=4)
            num_workers = min(_batch_size, os.cpu_count() - 1)

        # Assign num_workers if available as key in trcfg
        if hasattr(trcfg, 'num_workers'):
            trcfg.num_workers = num_workers

        # Model's mode and device
        trcfg._internal.training_mode = self.training

        # Switch model to evaluation mode
        if hasattr(self, 'preprocessor'):
            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'dither'):
                trcfg._internal.dither_value = self.preprocessor.featurizer.dither
                self.preprocessor.featurizer.dither = 0.0

            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'pad_to'):
                trcfg._internal.pad_to_value = self.preprocessor.featurizer.pad_to
                self.preprocessor.featurizer.pad_to = 0

        # Switch model to evaluation mode
        self.eval()

        # Disable logging
        trcfg._internal.logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

    def _transcribe_input_processing(self, audio, trcfg: TranscribeConfig):
        """
        Internal function to process the input audio data and return a DataLoader. This function is called by
        `transcribe()` and `transcribe_generator()` to setup the input data for transcription.

        Args:
            audio: Of type `GenericTranscriptionType`
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A DataLoader object that is used to iterate over the input audio data.
        """
        # Resolve chunking based on input constraints
        if isinstance(audio, (list, tuple)):
            if len(audio) == 0:
                raise ValueError("Input `audio` is empty")
        else:
            audio = [audio]

        # Check if audio is a list of strings (filepaths or manifests)
        if isinstance(audio[0], str):
            if len(audio) == 1 and audio[0].endswith('.json') or audio[0].endswith('.jsonl'):
                # Assume it is a path to a manifest file
                trcfg._internal.manifest_filepath = audio[0]
                audio = manifest_utils.read_manifest(audio[0])

            audio_files = list(audio)

            tmp_dir = trcfg._internal.temp_dir
            ds_config = self._transcribe_input_manifest_processing(audio_files, tmp_dir, trcfg)
            temp_dataloader = self._setup_transcribe_dataloader(ds_config)
            return temp_dataloader

        # Check if audio is a list of numpy or torch tensors
        elif isinstance(audio[0], (np.ndarray, torch.Tensor)):
            audio_tensors = list(audio)

            # Convert numpy tensors to torch tensors
            if any([isinstance(_tensor, np.ndarray) for _tensor in audio_tensors]):
                audio_tensors = [
                    torch.as_tensor(audio_tensor) if isinstance(audio_tensor, np.ndarray) else audio_tensor
                    for audio_tensor in audio_tensors
                ]

            tmp_dir = trcfg._internal.temp_dir
            ds_config = self._transcribe_input_tensor_processing(audio_tensors, tmp_dir, trcfg)
            temp_dataloader = self._setup_transcribe_tensor_dataloader(ds_config, trcfg)
            return temp_dataloader

        else:
            raise ValueError(
                f"Input `audio` is of type {type(audio[0])}. "
                "Only `str` (path to audio file), `np.ndarray`, and `torch.Tensor` "
                "are supported as input."
            )

    def _transcribe_input_tensor_processing(
        self, audio_tensors: List[Union[np.ndarray, torch.Tensor]], temp_dir: str, trcfg: TranscribeConfig
    ):
        """
        Internal function to process the input audio tensors and return a config dict for the dataloader.

        Args:
            audio_tensors: A list of numpy or torch tensors. The user must ensure that they satisfy the correct
                sample rate and channel format.
            temp_dir: A temporary directory to store intermediate files.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A config dict that is used to setup the dataloader for transcription.
        """
        # Check if sample rate is set
        sample_rate = None
        if hasattr(self, 'cfg') and 'sample_rate' in self.cfg:
            sample_rate = self.cfg.sample_rate
        elif hasattr(self, 'sample_rate'):
            sample_rate = self.sample_rate

        if sample_rate is None:
            raise RuntimeError(
                "Provided `audio` data contains numpy or torch tensors, however the class "
                "does not have `sample_rate` attribute. Please set `sample_rate` attribute to the model explicitly."
            )

        # Determine model-specific default chunk range
        # Canary models (have transf_decoder) use shorter chunks (30-40s)
        # Parakeet models use longer chunks (240-300s)
        ds_config = {
            'use_lhotse': get_value_from_transcription_config(trcfg, 'use_lhotse', True),
            'audio_tensors': audio_tensors,
            'batch_size': get_value_from_transcription_config(trcfg, 'batch_size', 4),
            'temp_dir': temp_dir,
            'num_workers': get_value_from_transcription_config(trcfg, 'num_workers', 0),
            'channel_selector': get_value_from_transcription_config(trcfg, 'channel_selector', None),
            'sample_rate': sample_rate,
            'pad_min_duration': get_value_from_transcription_config(trcfg, 'pad_min_duration', 1.0),
            'pad_direction': get_value_from_transcription_config(trcfg, 'pad_direction', 'both'),
            'enable_chunking': get_value_from_transcription_config(trcfg, 'enable_chunking', False),
            'chunk_range': get_value_from_transcription_config(trcfg, 'chunk_range', [240, 300]),
        }

        augmentor = get_value_from_transcription_config(trcfg, 'augmentor', None)
        if augmentor:
            ds_config['augmentor'] = augmentor

        return ds_config

    @abstractmethod
    def _transcribe_input_manifest_processing(
        self, audio_files: List[str], temp_dir: str, trcfg: TranscribeConfig
    ) -> Dict[str, Any]:
        """
        Internal function to process the input audio filepaths and return a config dict for the dataloader.

        Args:
            audio_files: A list of string filepaths for audio files, or a single string filepath for a manifest file.
            temp_dir: A temporary directory to store intermediate files.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A config dict that is used to setup the dataloader for transcription.
        """
        pass

    @abstractmethod
    def _setup_transcribe_dataloader(self, config: Dict) -> DataLoader:
        """
        Internal function to setup the dataloader for transcription. This function is called by
        `transcribe()` and `transcribe_generator()` to setup the input data for transcription.

        Args:
            config: A config dict that is used to setup the dataloader for transcription. It can be generated either
                by `_transcribe_input_manifest_processing()` or `_transcribe_input_tensor_processing()`.

        Returns:
            A DataLoader object that is used to iterate over the input audio data.
        """
        pass

    @abstractmethod
    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        """
        Internal function to perform the model's custom forward pass to return outputs that are processed by
        `_transcribe_output_processing()`.
        This function is called by `transcribe()` and `transcribe_generator()` to perform the model's forward pass.

        Args:
            batch: A batch of input data from the data loader that is used to perform the model's forward pass.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            The model's outputs that are processed by `_transcribe_output_processing()`.
        """
        pass

    @abstractmethod
    def _transcribe_output_processing(self, outputs, trcfg: TranscribeConfig) -> GenericTranscriptionType:
        """
        Internal function to process the model's outputs to return the results to the user. This function is called by
        `transcribe()` and `transcribe_generator()` to process the model's outputs.

        Args:
            outputs: The model's outputs that are processed by `_transcribe_forward()`.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            The output can be a list of
            objects, list of list of objects, tuple of objects, tuple of list of objects, or a dict of list of objects.
            Its type is defined in `TranscriptionReturnType`.
        """
        pass

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        """
        Internal function to teardown the model after transcription. Perform all teardown and post-checks here.

        Args:
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        # set mode back to its original value
        self.train(mode=trcfg._internal.training_mode)

        if hasattr(self, 'preprocessor'):
            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'dither'):
                self.preprocessor.featurizer.dither = trcfg._internal.dither_value

            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'pad_to'):
                self.preprocessor.featurizer.pad_to = trcfg._internal.pad_to_value

        if trcfg._internal.logging_level is not None:
            logging.set_verbosity(trcfg._internal.logging_level)

    def _setup_transcribe_tensor_dataloader(self, config: Dict, trcfg: TranscribeConfig) -> DataLoader:
        """
        Internal function to setup the dataloader for transcription. This function is called by
        `transcribe()` and `transcribe_generator()` to setup the input data for transcription.

        Args:
            config: A config dict that is used to setup the dataloader for transcription. It can be generated either
                by `_transcribe_input_manifest_processing()` or `_transcribe_input_tensor_processing()`.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A DataLoader object that is used to iterate over the input audio data.
        """
        dataset = TranscriptionTensorDataset(config)

        # Import collate function here to avoid circular imports
        from nemo.collections.asr.data.audio_to_text import _speech_collate_fn

        # Calculate pad id
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'pad_id'):
            pad_id = self.tokenizer.pad_id
        elif hasattr(self, 'transcribe_pad_id'):
            logging.info("Pad id is explicitly set to `model.transcribe_pad_id` = {}".format(self.transcribe_pad_id))
            pad_id = self.transcribe_pad_id
        else:
            logging.info(
                "Pad id is being set to 0 because it could not be resolved from the tokenizer. "
                "This can happen for various reasons, especially for character based models. "
                "If pad id is incorrect, please provide the pad id explicitly by setting "
                "`model.transcribe_pad_id`."
            )
            pad_id = 0

        # Choose collate function based on chunking setting.
        # When chunking is enabled, TranscriptionTensorDataset pre-expands each audio into
        # chunks at construction time, so each dataset item is already one chunk with metadata.
        # Use the dedicated collate that handles (chunk, len, sample_idx, chunk_start) tuples.
        enable_chunking = config.get('enable_chunking', False)
        if enable_chunking:
            collate_fn = _pre_chunked_tensor_collate_fn
        else:
            collate_fn = partial(_speech_collate_fn, pad_id=pad_id)

        return DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )


class ASRTranscriptionMixin(TranscriptionMixin):
    """
    An abstract class for ASR models that can transcribe audio. This class is a subclass of `TranscriptionMixin` that
    implements the default implementation of common abstract methods among the speech recognition model classes.

    The following abstract classes must be implemented by the subclass:

        - _transcribe_forward():
            Implements the model's custom forward pass to return outputs that are processed by
            `_transcribe_output_processing()`.

        - _transcribe_output_processing():
            Implements the post processing of the model's outputs to return the results to
            the user. The result can be a list of objects, list of list of objects, tuple of objects, tuple of list of
    """

    def _transcribe_input_manifest_processing(
        self, audio_files: List[str], temp_dir: str, trcfg: TranscribeConfig
    ) -> Dict[str, Any]:
        """
        Internal function to process the input audio filepaths and return a config dict for the dataloader.
        Specializes to ASR models which can have Encoder-Decoder-Joint architectures.

        Args:
            audio_files: A list of string filepaths for audio files.
            temp_dir: A temporary directory to store intermediate files.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A config dict that is used to setup the dataloader for transcription.
        """
        with open(os.path.join(temp_dir, 'manifest.json'), 'w', encoding='utf-8') as fp:
            for audio_file in audio_files:
                if isinstance(audio_file, str):
                    entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                    fp.write(json.dumps(entry) + '\n')
                elif isinstance(audio_file, dict):
                    fp.write(json.dumps(audio_file) + '\n')
                else:
                    raise ValueError(
                        f"Input `audio` is of type {type(audio_file)}. "
                        "Only `str` (path to audio file) or `dict` are supported as input."
                    )

        ds_config = {
            'use_lhotse': get_value_from_transcription_config(trcfg, 'use_lhotse', True),
            'paths2audio_files': audio_files,
            'batch_size': get_value_from_transcription_config(trcfg, 'batch_size', 4),
            'temp_dir': temp_dir,
            'num_workers': get_value_from_transcription_config(trcfg, 'num_workers', 0),
            'channel_selector': get_value_from_transcription_config(trcfg, 'channel_selector', None),
            'text_field': get_value_from_transcription_config(trcfg, 'text_field', 'text'),
            'lang_field': get_value_from_transcription_config(trcfg, 'lang_field', 'lang'),
            'enable_chunking': get_value_from_transcription_config(trcfg, 'enable_chunking', False),
            'chunk_range': get_value_from_transcription_config(trcfg, 'chunk_range', [240, 300]),
        }

        augmentor = get_value_from_transcription_config(trcfg, 'augmentor', None)
        if augmentor:
            ds_config['augmentor'] = augmentor

        return ds_config

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        """
        Internal function to setup the model for transcription. Perform all setup and pre-checks here.

        Args:
            audio: Of type `GenericTranscriptionType`
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        super()._transcribe_on_begin(audio, trcfg)

        # Freeze the encoder and decoder modules
        if hasattr(self, 'encoder'):
            self.encoder.freeze()

        if hasattr(self, 'decoder'):
            self.decoder.freeze()

        if hasattr(self, 'joint'):
            self.joint.freeze()

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        """
        Internal function to teardown the model after transcription. Perform all teardown and post-checks here.

        Args:
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        super()._transcribe_on_end(trcfg)

        # Unfreeze the encoder and decoder modules
        if hasattr(self, 'encoder'):
            self.encoder.unfreeze(partial=True)

        if hasattr(self, 'decoder'):
            self.decoder.unfreeze(partial=True)

        if hasattr(self, 'joint'):
            self.joint.unfreeze(partial=True)

    @classmethod
    def get_transcribe_config(cls) -> TranscribeConfig:
        """
        Utility method that returns the default config for transcribe() function.

        Returns:
            A dataclass
        """
        return TranscribeConfig()
