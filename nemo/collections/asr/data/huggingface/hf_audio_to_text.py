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

from typing import Callable, Dict, List, Optional, Tuple, Union

import datasets as hf_datasets
import torch
from datasets import concatenate_datasets
from datasets.distributed import split_dataset_by_node
from omegaconf import DictConfig, ListConfig, open_dict

from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common import tokenizers
from nemo.collections.common.parts.preprocessing import parsers
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


class HFTextProcessor:
    """
    Text processor for huggingface datasets, mimicing the behavior of 
    `nemo.collections.asr.data.audio_to_text.ASRManifestProcessor`. 
    Basic text cleaning is also supported.
    Args:
        parser: Str for a language specific preprocessor or a callable.
        bos_id: BOS token id to add to the beginning of the transcript.
        eos_id: EOS token id to add to the end of the transcript.
        pad_id: PAD token id to pad transcripts to the same length.
        normalize_text: If true, normalizes text in HFTextProcessor
        symbols_to_keep: If not None, only keeps symbols in this list when normalizing text
    """

    def __init__(
        self,
        parser: Union[str, Callable],
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        normalize_text: bool = False,
        symbols_to_keep: Optional[str | List[str]] = None,
    ):
        self.parser = parser
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.normalize_text = normalize_text
        self.symbols_to_keep = [x for x in symbols_to_keep] if symbols_to_keep is not None else []

    def process_text(self, text: str, lang: Optional[str] = None) -> List[int]:

        if self.normalize_text:
            text = text.lower()
            # only keep alphanumeric characters, spaces and symbols defined in self.symbols_to_keep
            text = ''.join([c for c in text if c.isalnum() or c.isspace() or c in self.symbols_to_keep])

        if hasattr(self.parser, "is_aggregate") and self.parser.is_aggregate and isinstance(text, str):
            if lang is not None:
                text_tokens = self.parser(text, lang)
            # for future use if want to add language bypass to audio_to_text classes
            # elif hasattr(parser, "lang") and parser.lang is not None:
            #    text_tokens = parser(text, parser.lang)
            else:
                raise ValueError("lang required in manifest when using aggregate tokenizers")
        else:
            text_tokens = self.parser(text)
        text_tokens_length = len(text_tokens)
        if self.bos_id is not None:
            text_tokens = [self.bos_id] + text_tokens
            text_tokens_length += 1
        if self.eos_id is not None:
            text_tokens = text_tokens + [self.eos_id]
            text_tokens_length += 1
        return text_tokens, text_tokens_length


def get_nested_dict_value(dictionary: dict, key: str):
    """
    the key should be a string of nested keys separated by `.`, e.g. `key1.key2.key3`,
    then the returned value will be `dictionary[key1][key2][key3]`
    """
    nested_keys = key.split(".")
    result = dictionary
    for k in nested_keys:
        if k not in result:
            raise KeyError(
                f"Key `{key}` not found in [{result.keys()}], target is {nested_keys}, input is {dictionary}"
            )
        result = result[k]
    return result


class _HFAudioTextDataset(Dataset):
    """
    A Dataset wrapper that loads from HuggingFace datasets and converts to NeMo compatible format.
    Args:
        audio_key: key to access audio data from the dataset
        text_key: key to access text data from the dataset
        sample_rate_key: key to access sample rate data from the dataset
        hf_data_cfg: HuggingFace dataset config, all params in this config will be passed to `hf_datasets.load_dataset`
        parser: Str for a language specific preprocessor or a callable.
        augmentor: An instance of `nemo.collections.asr.parts.perturb.AudioAugmentor` to apply on audio.
        trim: If true, trims silence using `nemo.collections.asr.parts.preprocessing.segment.AudioSegment`
        bos_id: BOS token id to add to the beginning of the transcript.
        eos_id: EOS token id to add to the end of the transcript.
        pad_id: PAD token id to pad transcripts to the same length.
        return_sample_id: If true, returns sample id from the dataset.
        channel_selector: ChannelSelectorType, which channel(s) to use for audio.
        normalize_db: Target RMS value for audio normalization.
        ref_channel: Reference channel for normalization.
        id_key: key to access sample id from the dataset
        normalize_text: If true, normalizes text in HFTextProcessor
        symbols_to_keep: If not None, only keeps symbols in this list when normalizing text 
    """

    def __init__(
        self,
        audio_key: str,
        text_key: str,
        sample_rate_key: str,
        hf_data_cfg: Union[DictConfig, ListConfig],
        parser: Union[str, Callable],
        sample_rate: int,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        normalize_db: Optional[float] = None,
        ref_channel: Optional[int] = None,
        id_key: Optional[str] = None,
        normalize_text: bool = False,
        symbols_to_keep: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.audio_key = audio_key
        self.text_key = text_key
        self.sample_rate_key = sample_rate_key
        self.id_key = id_key
        self.sample_rate = sample_rate
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.trim = trim
        self.return_sample_id = return_sample_id
        self.channel_selector = channel_selector
        self.normalize_db = normalize_db
        self.ref_channel = ref_channel

        self.text_processor = HFTextProcessor(parser, bos_id, eos_id, pad_id, normalize_text, symbols_to_keep)

        data_config_list = [hf_data_cfg] if isinstance(hf_data_cfg, DictConfig) else hf_data_cfg
        dataset_list = []
        for data_cfg in data_config_list:
            with open_dict(data_cfg):
                if "streaming" in data_cfg and data_cfg.streaming:
                    logging.warning(
                        "streaming must be False for random access dataset, but you use streaming=True. Forcing streaming=False"
                    )
                data_cfg.streaming = False
            logging.info(f"Loading HuggingFace Dataset with cfg: {data_cfg}")
            dataset_list.append(hf_datasets.load_dataset(**data_cfg))
            logging.info(f"Dataset loaded with {len(dataset_list[-1])} samples")
        self.dataset = concatenate_datasets(dataset_list)

        logging.info(f"Total number of samples loaded: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple:
        item = self.dataset[index]

        audio_array = get_nested_dict_value(item, self.audio_key)
        origin_sr = get_nested_dict_value(item, self.sample_rate_key)
        audio_segment = AudioSegment(
            samples=audio_array,
            sample_rate=origin_sr,
            target_sr=self.sample_rate,
            trim=self.trim,
            channel_selector=self.channel_selector,
            normalize_db=self.normalize_db,
            ref_channel=self.ref_channel,
        )
        self.augmentor.perturb(audio_segment)
        f = torch.tensor(audio_segment.samples, dtype=torch.float)
        fl = torch.tensor(f.shape[0], dtype=torch.long)

        text = get_nested_dict_value(item, self.text_key)
        t, tl = self.text_processor.process_text(text)

        index = get_nested_dict_value(item, self.id_key) if self.id_key else index
        if self.return_sample_id:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), index
        else:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

        return output

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=self.text_processor.pad_id)


class HFAudioToCharDataset(_HFAudioTextDataset):
    """
    Wrapper class for loading HuggingFace dataset for a char-based ASR model
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        audio_key: str,
        text_key: str,
        sample_rate_key: str,
        hf_data_cfg: DictConfig,
        labels: List[str],
        sample_rate: int,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        normalize_db: Optional[float] = None,
        ref_channel: Optional[int] = None,
        parser: Union[str, Callable] = 'en',
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        id_key: Optional[str] = None,
        normalize_text: bool = False,
        symbols_to_keep: Optional[str] = None,
    ):
        self.labels = labels

        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )

        super().__init__(
            audio_key=audio_key,
            text_key=text_key,
            sample_rate_key=sample_rate_key,
            hf_data_cfg=hf_data_cfg,
            parser=parser,
            sample_rate=sample_rate,
            augmentor=augmentor,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            return_sample_id=return_sample_id,
            channel_selector=channel_selector,
            normalize_db=normalize_db,
            ref_channel=ref_channel,
            id_key=id_key,
            normalize_text=normalize_text,
            symbols_to_keep=symbols_to_keep,
        )


class HFAudioToBPEDataset(_HFAudioTextDataset):
    """
    Wrapper class for loading a HuggingFace dataset for a BPE-based ASR model
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        audio_key: str,
        text_key: str,
        sample_rate_key: str,
        hf_data_cfg: DictConfig,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        trim: bool = False,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        normalize_db: Optional[float] = None,
        ref_channel: Optional[int] = None,
        use_start_end_token: bool = True,
        id_key: Optional[str] = None,
        normalize_text: bool = False,
        symbols_to_keep: Optional[str] = None,
    ):
        if use_start_end_token and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer

            def __call__(self, *args):
                if isinstance(args[0], List) and self.is_aggregate:
                    t = []
                    for span in args[0]:
                        t.extend(self._tokenizer.text_to_ids(span['str'], span['lang']))
                    return t

                t = self._tokenizer.text_to_ids(*args)
                return t

        super().__init__(
            audio_key=audio_key,
            text_key=text_key,
            sample_rate_key=sample_rate_key,
            hf_data_cfg=hf_data_cfg,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            augmentor=augmentor,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            return_sample_id=return_sample_id,
            channel_selector=channel_selector,
            normalize_db=normalize_db,
            ref_channel=ref_channel,
            id_key=id_key,
            normalize_text=normalize_text,
            symbols_to_keep=symbols_to_keep,
        )


class _HFIterableAudioTextDataset(IterableDataset):
    """
    Wrapper class for loading HuggingFace IterableDataset and converts to NeMo compatible format. 
    Args:
        audio_key: key to access audio data from the dataset
        text_key: key to access text data from the dataset
        sample_rate_key: key to access sample rate data from the dataset
        hf_data_cfg: HuggingFace dataset config, all params in this config will be passed to `hf_datasets.load_dataset`
        parser: Str for a language specific preprocessor or a callable.
        augmentor: An instance of `nemo.collections.asr.parts.perturb.AudioAugmentor` to apply on audio.
        trim: If true, trims silence using `nemo.collections.asr.parts.preprocessing.segment.AudioSegment`
        bos_id: BOS token id to add to the beginning of the transcript.
        eos_id: EOS token id to add to the end of the transcript.
        pad_id: PAD token id to pad transcripts to the same length.
        return_sample_id: If true, returns sample id from the dataset.
        channel_selector: ChannelSelectorType, which channel(s) to use for audio.
        normalize_db: Target RMS value for audio normalization.
        ref_channel: Reference channel for normalization.
        id_key: key to access sample id from the dataset
        global_rank: global rank of the current worker
        world_size: total number of workers
        shuffle_n: buffer size for shuffling
        shuffle_seed: seed for shuffling
        normalize_text: If true, normalizes text in HFTextProcessor
        symbols_to_keep: If not None, only keeps symbols in this list when normalizing text
    """

    def __init__(
        self,
        audio_key: str,
        text_key: str,
        sample_rate_key: str,
        hf_data_cfg: Union[DictConfig, ListConfig],
        parser: Union[str, Callable],
        sample_rate: int,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        normalize_db: Optional[float] = None,
        ref_channel: Optional[int] = None,
        id_key: Optional[str] = None,
        global_rank: int = 0,
        world_size: int = 0,
        shuffle_n: int = 0,
        shuffle_seed: Optional[int] = None,
        normalize_text: bool = False,
        symbols_to_keep: Optional[str] = None,
    ) -> None:
        super().__init__()

        if return_sample_id and id_key is None:
            raise ValueError("return_sample_id is True, but id_key is None")

        self.audio_key = audio_key
        self.text_key = text_key
        self.sample_rate_key = sample_rate_key
        self.id_key = id_key
        self.sample_rate = sample_rate
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.trim = trim
        self.return_sample_id = return_sample_id
        self.channel_selector = channel_selector
        self.normalize_db = normalize_db
        self.ref_channel = ref_channel

        self.text_processor = HFTextProcessor(parser, bos_id, eos_id, pad_id, normalize_text, symbols_to_keep)

        data_config_list = [hf_data_cfg] if isinstance(hf_data_cfg, DictConfig) else hf_data_cfg
        dataset_list = []
        for data_cfg in data_config_list:
            with open_dict(data_cfg):
                if "streaming" in data_cfg and not data_cfg.streaming:
                    logging.warning(
                        "streaming must be True for streaming dataset, but you use streaming=False. Forcing streaming=True"
                    )
                # streaming must be True for iterable dataset
                data_cfg.streaming = True
            logging.info(f"Streaming HuggingFace IterableDataset with cfg: {data_cfg}")
            dataset_list.append(hf_datasets.load_dataset(**data_cfg))

        self.dataset = concatenate_datasets(dataset_list)
        logging.info(f"Total number of samples cannot be extracted from HF streaming dataset")

        if shuffle_n > 0:
            self.dataset = self.dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_n)

        self.dataset = split_dataset_by_node(self.dataset, global_rank, world_size)
        self.dataset = self.dataset.map(self._build_sample)

    def __len__(self):
        raise NotImplementedError(
            f"len() is not supported for {self.__class__.__name__}. Please set `trainer.max_steps` to explicitly set the number of steps to train for."
        )

    def __iter__(self):
        return self.dataset.__iter__()

    def _collate_fn(self, batch):
        a_signal = [b['audio_signal'] for b in batch]
        a_sig_length = [b['a_sig_length'] for b in batch]
        transcripts = [b['transcripts'] for b in batch]
        transcript_length = [b['transcript_length'] for b in batch]
        if self.return_sample_id:
            sample_id = [b['sample_id'] for b in batch]
            batch_list = list(zip(a_signal, a_sig_length, transcripts, transcript_length, sample_id))
        else:
            batch_list = list(zip(a_signal, a_sig_length, transcripts, transcript_length))

        return _speech_collate_fn(batch_list, pad_id=self.text_processor.pad_id)

    def _build_sample(self, sample):
        audio_array = get_nested_dict_value(sample, self.audio_key)
        origin_sr = get_nested_dict_value(sample, self.sample_rate_key)
        audio_segment = AudioSegment(
            samples=audio_array,
            sample_rate=origin_sr,
            target_sr=self.sample_rate,
            trim=self.trim,
            channel_selector=self.channel_selector,
            normalize_db=self.normalize_db,
            ref_channel=self.ref_channel,
        )
        self.augmentor.perturb(audio_segment)
        f = torch.tensor(audio_segment.samples, dtype=torch.float)
        fl = torch.tensor(f.shape[0], dtype=torch.long)

        text = get_nested_dict_value(sample, self.text_key)
        t, tl = self.text_processor.process_text(text)

        output = {
            'audio_signal': f,
            'a_sig_length': fl,
            'transcripts': torch.tensor(t).long(),
            'transcript_length': torch.tensor(tl).long(),
        }

        if self.return_sample_id:
            output['sample_id'] = get_nested_dict_value(sample, self.id_key)
        return output


class HFIterableAudioToCharDataset(_HFIterableAudioTextDataset):
    """
    Wrapper class for loading HuggingFace IterableDataset for a char-based ASR model
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        labels: List[str],
        audio_key: str,
        text_key: str,
        sample_rate_key: str,
        hf_data_cfg: DictConfig,
        sample_rate: int,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        trim: bool = False,
        bos_id: int | None = None,
        eos_id: int | None = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
        id_key: str | None = None,
        channel_selector: ChannelSelectorType | None = None,
        normalize_db: float | None = None,
        ref_channel: int | None = None,
        global_rank: int = 0,
        world_size: int = 0,
        shuffle_n: int = 0,
        shuffle_seed: Optional[int] = None,
        parser: Union[str, Callable] = 'en',
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        normalize_text: bool = False,
        symbols_to_keep: Optional[str] = None,
    ) -> None:
        self.labels = labels

        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )

        super().__init__(
            audio_key=audio_key,
            text_key=text_key,
            sample_rate_key=sample_rate_key,
            hf_data_cfg=hf_data_cfg,
            parser=parser,
            sample_rate=sample_rate,
            augmentor=augmentor,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            return_sample_id=return_sample_id,
            id_key=id_key,
            channel_selector=channel_selector,
            normalize_db=normalize_db,
            ref_channel=ref_channel,
            global_rank=global_rank,
            world_size=world_size,
            shuffle_n=shuffle_n,
            shuffle_seed=shuffle_seed,
            normalize_text=normalize_text,
            symbols_to_keep=symbols_to_keep,
        )


class HFIterableAudioToBPEDataset(_HFIterableAudioTextDataset):
    """
    Wrapper class for loading HuggingFace IterableDataset for a BPE-based ASR model
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        audio_key: str,
        text_key: str,
        sample_rate_key: str,
        hf_data_cfg: DictConfig,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        trim: bool = False,
        return_sample_id: bool = False,
        id_key: str | None = None,
        channel_selector: ChannelSelectorType | None = None,
        normalize_db: float | None = None,
        ref_channel: int | None = None,
        global_rank: int = 0,
        world_size: int = 0,
        shuffle_n: int = 0,
        shuffle_seed: Optional[int] = None,
        use_start_end_token: bool = True,
        normalize_text: bool = False,
        symbols_to_keep: Optional[str] = None,
    ) -> None:

        if use_start_end_token and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer

            def __call__(self, *args):
                if isinstance(args[0], List) and self.is_aggregate:
                    t = []
                    for span in args[0]:
                        t.extend(self._tokenizer.text_to_ids(span['str'], span['lang']))
                    return t

                t = self._tokenizer.text_to_ids(*args)
                return t

        super().__init__(
            audio_key=audio_key,
            text_key=text_key,
            sample_rate_key=sample_rate_key,
            hf_data_cfg=hf_data_cfg,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            augmentor=augmentor,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            return_sample_id=return_sample_id,
            id_key=id_key,
            channel_selector=channel_selector,
            normalize_db=normalize_db,
            ref_channel=ref_channel,
            global_rank=global_rank,
            world_size=world_size,
            shuffle_n=shuffle_n,
            shuffle_seed=shuffle_seed,
            normalize_text=normalize_text,
            symbols_to_keep=symbols_to_keep,
        )
