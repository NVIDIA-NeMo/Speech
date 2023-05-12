# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from __future__ import annotations

import concurrent.futures
import copy
import gc
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Union

import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.core.classes import Dataset, IterableDataset
from nemo.utils import logging

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
except Exception as e:
    pass  # Normalizer imported only for annotation purposes, error can be ignored

AnyPath = Union[Path, str]


class TextToTextItem(NamedTuple):
    tts_text: torch.Tensor  # normalized and tokenized text for TTS
    transcript: torch.Tensor  # tokenized text for ASR
    speaker: int  # speaker id for multi-speaker TTS


class TextToTextBatch(NamedTuple):
    tts_texts: torch.Tensor  # tokenized texts for tts
    tts_text_lengths: torch.Tensor
    transcripts: torch.Tensor  # tokenized texts for ASR
    transcript_lengths: torch.Tensor
    speakers: torch.Tensor  # speaker ids for multi-speaker TTS

    @staticmethod
    def collate_fn(batch: List[TextToTextItem], asr_pad_id: int, tts_text_pad_id: int) -> TextToTextBatch:
        return TextToTextBatch(
            tts_texts=pad_sequence([item.tts_text for item in batch], batch_first=True, padding_value=tts_text_pad_id),
            tts_text_lengths=torch.tensor([item.tts_text.shape[0] for item in batch]).long(),
            transcripts=pad_sequence([item.transcript for item in batch], batch_first=True, padding_value=asr_pad_id),
            transcript_lengths=torch.tensor([item.transcript.shape[0] for item in batch]).long(),
            speakers=torch.tensor([item.speaker for item in batch]).long(),
        )


class TextOrAudioToTextBatch(NamedTuple):
    audio_signals: torch.Tensor
    audio_signal_lengths: torch.Tensor
    tts_texts: torch.Tensor
    tts_text_lengths: torch.Tensor
    speakers: torch.Tensor
    transcripts: torch.Tensor
    transcript_lengths: torch.Tensor

    @staticmethod
    def collate_fn(
        batch: List[Union[TextToTextItem, tuple]], tts_text_pad_id: int, asr_pad_id: int
    ) -> Union[TextToTextBatch, TextOrAudioToTextBatch, tuple]:
        """
        Collate function for dataloader
        Can accept mixed batch of text-to-text items and audio-text items (typical for ASR)
        """
        text_items: List[TextToTextItem] = [item for item in batch if isinstance(item, TextToTextItem)]
        if not text_items:
            # pure audio-text batch
            return _speech_collate_fn(batch=batch, pad_id=asr_pad_id)

        asr_items = [item for item in batch if not isinstance(item, TextToTextItem)]

        if not asr_items:
            # pure text-to-text batch
            return TextToTextBatch.collate_fn(batch=text_items, asr_pad_id=asr_pad_id, tts_text_pad_id=tts_text_pad_id)

        # mixed batch

        # each asr item is a tuple:
        # audio_signal (0), audio_length (1), transcript (2), transcript_length (3), sample_id (4, optional)
        audio_signals = pad_sequence([item[0] for item in asr_items], batch_first=True, padding_value=0.0)
        audio_signal_lengths = torch.tensor([item[1] for item in asr_items]).long()

        tts_texts = pad_sequence(
            [item.tts_text for item in text_items], batch_first=True, padding_value=tts_text_pad_id
        )
        tts_text_lengths = torch.tensor([item.tts_text.shape[0] for item in text_items]).long()
        speakers = torch.tensor([item.speaker for item in text_items]).long()

        transcripts = pad_sequence(
            [item.transcript for item in text_items] + [item[2] for item in asr_items],
            batch_first=True,
            padding_value=asr_pad_id,
        )
        transcript_lengths = torch.tensor(
            [item.transcript.shape[0] for item in text_items] + [item[3] for item in asr_items]
        ).long()

        return TextOrAudioToTextBatch(
            audio_signals=audio_signals,
            audio_signal_lengths=audio_signal_lengths,
            tts_texts=tts_texts,
            tts_text_lengths=tts_text_lengths,
            speakers=speakers,
            transcripts=transcripts,
            transcript_lengths=transcript_lengths,
        )


def _asr_text_to_tokens(text: str) -> np.ndarray:
    """
    Helper function for asr tokenization with multiprocessing pool only.
    Must be defined on the top level.
    Expects asr_tokenizer_global, asr_bos_id_global, asr_eos_id_global to exist in the current pool process
    """
    ids = asr_tokenizer_global.text_to_ids(text)
    if asr_bos_id_global is not None:
        ids = [asr_bos_id_global] + ids
    if asr_eos_id_global is not None:
        ids.append(asr_eos_id_global)
    return np.asarray(ids)


def _tts_text_to_tokens(text: str) -> np.ndarray:
    """
    Helper function for asr tokenization with multiprocessing pool only.
    Must be defined on the top level.
    Expects tts_tokenizer_global to exist in the current pool process
    """
    return np.asarray(tts_tokenizer_global(text))


def _iterate_manifest(filepath: AnyPath) -> Iterable[Dict[str, Any]]:
    """
    Helper function to iterate manifest
    """
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            yield record


class TextToTextDatasetBase:
    """
    Base class for loading text-to-text manifests
    Map-style and Iterable datasets should inherit this class
    """

    asr_pad_id: int
    tts_text_pad_id: int
    asr_bos_id: Optional[int] = None
    asr_eos_id: Optional[int] = None
    data: List[Dict[str, Any]]

    def __init__(
        self,
        manifest_filepath: Union[AnyPath, List[AnyPath]],
        speakers_filepath: Union[AnyPath, List[AnyPath]],
        asr_tokenizer: TokenizerSpec,
        asr_use_start_end_token: bool,
        tts_parser: Callable,
        tts_text_pad_id: int,
        tts_text_normalizer: "Normalizer",
        tts_text_normalizer_call_kwargs: Dict,
        min_words: int = 1,
        max_words: int = 1_000_000,
        tokenizer_workers: int = 1,
        num_parts: int = 1,
        current_part_index: int = 0,
    ):
        super().__init__()
        # ASR tokenizer setup
        if asr_use_start_end_token and hasattr(asr_tokenizer, 'bos_token'):
            self.asr_bos_id = asr_tokenizer.bos_id

        if asr_use_start_end_token and hasattr(asr_tokenizer, 'eos_token'):
            self.asr_eos_id = asr_tokenizer.eos_id

        if hasattr(asr_tokenizer, 'pad_token'):
            self.asr_pad_id = asr_tokenizer.pad_id
        else:
            self.asr_pad_id = 0

        self.asr_tokenizer = asr_tokenizer

        # TTS tokenizer setup
        self.tts_parser = tts_parser
        self.tts_normalizer = tts_text_normalizer
        self.tts_normalizer_kwargs = tts_text_normalizer_call_kwargs
        self.tts_text_pad_id = tts_text_pad_id

        # Load speakers
        if isinstance(speakers_filepath, str):
            speakers_filepath = speakers_filepath.split(",")
        elif isinstance(speakers_filepath, Path):
            speakers_filepath = [speakers_filepath]
        speakers: Set[int] = set()
        for filepath in speakers_filepath:
            with open(Path(filepath).expanduser(), "r") as f:
                speakers.update(map(int, f.read().split()))
        self.speakers = np.asarray(sorted(speakers))
        logging.info(f"Loaded {len(self.speakers)} speakers")

        # Load manifest
        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(",")
        elif isinstance(manifest_filepath, Path):
            manifest_filepath = [manifest_filepath]
        self.manifest_paths = [Path(filepath) for filepath in manifest_filepath]

        num_skipped_words = 0
        num_skipped_utterances = 0
        asr_texts = []
        tts_texts = []
        need_normalization = False

        for manifest_path in self.manifest_paths:
            for tmp_item in tqdm(_iterate_manifest(manifest_path)):
                text = tmp_item["text"]
                num_words = len(text.split())
                # skip if number of works not in desired range
                # TODO: maybe it would be valuable to sample sub-utterances from long utterances
                if not (min_words <= num_words <= max_words):
                    num_skipped_words += num_words
                    num_skipped_utterances += 1
                    continue
                asr_texts.append(tmp_item["text"])
                if "tts_text_normalized" in tmp_item:
                    tts_texts.append(tmp_item["tts_text_normalized"])
                else:
                    tts_texts.append(tmp_item["tts_text"])
                    need_normalization = True

        if need_normalization:
            logging.warning("TTS normalization is extremely slow! It is recommended to normalize TTS text")

        if num_skipped_utterances:
            logging.warning(f"Skipped {num_skipped_utterances} utterances " f"with {num_skipped_words}")

        num_utterances = len(asr_texts)
        # preprocessing is very costly, if we need only part - remove unnecessary utterances
        if num_parts > 1:
            # NB: floor division, full dataset can contain fewer utterances than original, like in tarred dataset
            num_utterances_part = num_utterances // num_parts
            start = num_utterances_part * current_part_index
            end = start + num_utterances_part
            logging.info(
                f"Taking part of the dataset: {current_part_index} index, total {num_parts} from {start} to {end}"
            )
            asr_texts = asr_texts[start:end]
            tts_texts = tts_texts[start:end]
            num_utterances = num_utterances_part

        self.data = [dict() for _ in range(num_utterances)]

        if len(asr_texts) == 0:
            # no data was loaded
            logging.warning("Text-to-text dataset is empty")
            return

        if tokenizer_workers == 1:
            logging.warning(
                "Preprocessing large text with tokenizer_workers=1 may be slow with TTS tokenizer. "
                "Prefer tokenizer_workers=(num_cpu_cores/num_gpus_per_node)"
            )
            for i, tokenized_text in enumerate(
                tqdm((self._asr_text_to_tokens(text) for text in asr_texts), total=len(asr_texts))
            ):
                self.data[i]["asr_text_tokens"] = tokenized_text
        else:
            # Multiprocessing hack: use global variables for every process (not really global in program context)
            def _init_asr_tokenize_process(tokenizer, bos_id, eos_id):
                global asr_tokenizer_global, asr_bos_id_global, asr_eos_id_global  # process-global
                # deepcopy to avoid serialization of parent models
                asr_tokenizer_global = copy.deepcopy(tokenizer)
                asr_bos_id_global = copy.deepcopy(bos_id)
                asr_eos_id_global = copy.deepcopy(eos_id)

            with concurrent.futures.ProcessPoolExecutor(
                initializer=_init_asr_tokenize_process,
                initargs=(asr_tokenizer, self.asr_bos_id, self.asr_eos_id),
                max_workers=tokenizer_workers,
            ) as pool:
                # chunk size for pool map is empirically chosen as a trade-off between speed and responsiveness
                for i, tokenized_text in enumerate(
                    tqdm(pool.map(_asr_text_to_tokens, asr_texts, chunksize=1000), total=len(asr_texts))
                ):
                    self.data[i]["asr_text_tokens"] = tokenized_text
        # force free memory
        del asr_texts
        gc.collect()

        if tokenizer_workers == 1:
            logging.warning(
                "Preprocessing large text with tokenizer_workers=1 may be slow with TTS tokenizer. "
                "Prefer tokenizer_workers=(num_cpu_cores/num_gpus_per_node)"
            )
            for i, tokenized_text in enumerate(
                tqdm(
                    (self._tts_text_to_tokens(text, normalize=need_normalization) for text in tts_texts),
                    total=len(tts_texts),
                )
            ):
                self.data[i]["tts_text_tokens"] = tokenized_text
        else:
            if need_normalization:
                # TODO: implement, if we really need normalization inplace
                raise NotImplementedError(
                    "Normalization with tokenizer_workers > 1 is not implemented. "
                    "It is not recommended to use normalization on the fly at all, since it's extremely slow"
                )

            def _init_tts_tokenize_process(tokenizer):
                global tts_tokenizer_global  # process-global
                tts_tokenizer_global = copy.deepcopy(tokenizer)

            with concurrent.futures.ProcessPoolExecutor(
                initializer=_init_tts_tokenize_process, initargs=(tts_parser,), max_workers=tokenizer_workers,
            ) as pool:
                # chunk size for pool map is empirically chosen as a trade-off between speed and responsiveness
                for i, tokenized_text in enumerate(
                    tqdm(pool.map(_tts_text_to_tokens, tts_texts, chunksize=1000), total=len(tts_texts))
                ):
                    self.data[i]["tts_text_tokens"] = tokenized_text
        # force free memory
        del tts_texts
        gc.collect()

    def _asr_text_to_tokens(self, text: str) -> np.ndarray:
        ids = self.asr_tokenizer.text_to_ids(text)
        if self.asr_bos_id is not None:
            ids = [self.asr_bos_id] + ids
        if self.asr_eos_id is not None:
            ids.append(self.asr_eos_id)
        return np.asarray(ids)

    def _tts_text_to_tokens(self, text: str, normalize=True) -> np.ndarray:
        if normalize:
            text = self.tts_normalizer.normalize(text, **self.tts_normalizer_kwargs)
        tokens = self.tts_parser(text)
        return np.asarray(tokens)

    def __getitem__(self, index):
        item = self.data[index]
        return TextToTextItem(
            transcript=torch.from_numpy(item["asr_text_tokens"]).long(),
            tts_text=torch.from_numpy(item["tts_text_tokens"]).long(),
            speaker=random.choice(self.speakers),
        )

    def __len__(self):
        return len(self.data)


class TextToTextDataset(TextToTextDatasetBase, Dataset):
    """Text-to-Text Map-style Dataset for hybrid ASR-TTS models"""

    def __init__(
        self,
        manifest_filepath: Union[AnyPath, List[AnyPath]],
        speakers_filepath: Union[AnyPath, List[AnyPath]],
        asr_tokenizer: TokenizerSpec,
        asr_use_start_end_token: bool,
        tts_parser: Callable,
        tts_text_pad_id: int,
        tts_text_normalizer: "Normalizer",
        tts_text_normalizer_call_kwargs: Dict,
        min_words: int = 1,
        max_words: int = 1_000_000,
        tokenizer_workers: int = 1,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            speakers_filepath=speakers_filepath,
            asr_tokenizer=asr_tokenizer,
            asr_use_start_end_token=asr_use_start_end_token,
            tts_parser=tts_parser,
            tts_text_pad_id=tts_text_pad_id,
            tts_text_normalizer=tts_text_normalizer,
            tts_text_normalizer_call_kwargs=tts_text_normalizer_call_kwargs,
            min_words=min_words,
            max_words=max_words,
            tokenizer_workers=tokenizer_workers,
            num_parts=1,
        )

    def collate_fn(
        self, batch: List[Union[TextToTextItem, tuple]]
    ) -> Union[TextToTextBatch, TextOrAudioToTextBatch, tuple]:
        """
        Collate function for dataloader
        Can accept mixed batch of text-to-text items and audio-text items (typical for ASR)
        """
        return TextOrAudioToTextBatch.collate_fn(
            batch=batch, asr_pad_id=self.asr_pad_id, tts_text_pad_id=self.tts_text_pad_id
        )


class TextToTextIterableDataset(TextToTextDatasetBase, IterableDataset):
    """
    Text-to-Text Iterable Dataset for hybrid ASR-TTS models
    Only part necessary for current process should be loaded and stored
    """

    def __init__(
        self,
        manifest_filepath: Union[AnyPath, List[AnyPath]],
        speakers_filepath: Union[AnyPath, List[AnyPath]],
        asr_tokenizer: TokenizerSpec,
        asr_use_start_end_token: bool,
        tts_parser: Callable,
        tts_text_pad_id: int,
        tts_text_normalizer: "Normalizer",
        tts_text_normalizer_call_kwargs: Dict,
        min_words: int = 1,
        max_words: int = 1_000_000,
        tokenizer_workers: int = 1,
        num_parts: int = 1,
        current_part_index: int = 0,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            speakers_filepath=speakers_filepath,
            asr_tokenizer=asr_tokenizer,
            asr_use_start_end_token=asr_use_start_end_token,
            tts_parser=tts_parser,
            tts_text_pad_id=tts_text_pad_id,
            tts_text_normalizer=tts_text_normalizer,
            tts_text_normalizer_call_kwargs=tts_text_normalizer_call_kwargs,
            min_words=min_words,
            max_words=max_words,
            tokenizer_workers=tokenizer_workers,
            num_parts=num_parts,
            current_part_index=current_part_index,
        )

    def __iter__(self):
        # Implementation based on docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            start = 0
            end = len(self)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self))
        indices = np.arange(start, end)
        np.random.shuffle(indices)
        return map(self.__getitem__, indices)

    def collate_fn(
        self, batch: List[Union[TextToTextItem, tuple]]
    ) -> Union[TextToTextBatch, TextOrAudioToTextBatch, tuple]:
        """
        Collate function for dataloader
        Can accept mixed batch of text-to-text items and audio-text items (typical for ASR)
        """
        return TextOrAudioToTextBatch.collate_fn(
            batch=batch, asr_pad_id=self.asr_pad_id, tts_text_pad_id=self.tts_text_pad_id
        )
