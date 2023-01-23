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

__all__ = [
    'BertPunctuationCapitalizationDataset',
    'LABEL_ID_DIR_FOR_NEMO_CHECKPOINT',
    'Progress',
    'PunctuationCapitalizationEvalDataConfig',
    'PunctuationCapitalizationTrainDataConfig',
    'create_label_ids',
    'create_masks_and_segment_ids',
    'is_legacy_data_config',
    'legacy_data_config_to_new_data_config',
    'load_label_ids',
    'raise_not_equal_labels_error',
    'save_label_ids',
]

import itertools
import multiprocessing as mp
import os
import pickle
import tempfile
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from queue import Empty
from time import sleep
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from omegaconf import MISSING, DictConfig, OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils.data_preprocessing import get_label_stats, get_stats
from nemo.core.classes import Dataset
from nemo.core.neural_types import AudioSignal, ChannelType, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

try:
    from nemo.collections.asr.parts.preprocessing import AudioSegment

    ASR_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ASR_AVAILABLE = False


MAX_NUM_QUERIES_IN_SPLIT = 10 ** 4
TOKENIZATION_PROGRESS_REPORT_PERIOD = 10 ** 3
BATCH_MARK_UP_PROGRESS_REPORT_PERIOD = 10 ** 4
BATCH_BUILDING_PROGRESS_REPORT_PERIOD = 10 ** 4

LABEL_ID_DIR_FOR_NEMO_CHECKPOINT = "label_id_files_for_nemo_checkpoint"


@dataclass
class PunctuationCapitalizationDataConfigBase:
    """A base class for punctuation and capitalization data configs. This class does not define ``ds_item``
    attribute which works differently for train and evaluation data."""

    ###################################################
    # AUDIO DATASET PARAMETERS
    ###################################################
    use_audio: bool = False
    """
    Whether to use audio or not. If set to True you should provide ``audio_file``.  
    """

    audio_file: Optional[str] = None
    """
    Path to the file with audio paths one per row.
    """

    sample_rate: Optional[int] = 16000
    """
    Sample rate of audios to use.
    """

    use_bucketing: Optional[bool] = True
    """
    Whether to pack samples into ``tokens_in_batch`` or not. Increases GPU utilization but may cause significant RAM consumption if used together with ``use_audio``. 
    """

    batch_size: Optional[int] = 32
    """
    Batch size used if ``use_bucketing`` set to False.
    """

    preload_audios: Optional[bool] = True
    """
    If set to True audios will be loaded during ``__init__`` call of dataset. Otherwise it will be loaded during ``collate_fn ``call
    """

    ###################################################
    # PARAMETERS COMMON FOR REGULAR AND TARRED DATASETS
    ###################################################
    use_tarred_dataset: bool = MISSING
    """Whether to use tarred dataset. If True, then you should provide ``tar_metadata_file``. Otherwise, you should
    provide ``text_file``, ``labels_file``, ``tokens_in_batch``."""

    label_info_save_dir: Optional[str] = None
    """A path to a directory where files created during dataset processing are stored. These files include label id
    files and label stats files. By default, it is a directory containing ``text_file`` or ``tar_metadata_file``.
    You may need this parameter if dataset directory is read-only and thus does not allow saving anything near dataset
    files"""

    #################################################
    # REGULAR DATASET PARAMETERS
    #################################################
    text_file: Optional[str] = None
    """A path to a file with source text data without punctuation and capitalization."""

    labels_file: Optional[str] = None
    """A path to a file with punctuation and capitalization labels in NeMo format. NeMo format is described in
    `documentation
    <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#nemo-data-format>`_
    """

    tokens_in_batch: Optional[int] = None
    """Number of tokens in a batch including paddings and special tokens ([CLS], [SEP], [UNK]). This config does
    not have ``batch_size`` parameter."""

    max_seq_length: int = 512
    """Max number of tokens in a source sequence. ``max_seq_length`` includes [CLS] and [SEP] tokens. Sequences
    which are too long will be clipped by removal of tokens from the end of a sequence."""

    num_samples: int = -1
    """A number of samples loaded from ``text_file`` and ``labels_file`` which are used in the dataset. If this
    parameter equals ``-1``, then all samples are used."""

    use_cache: bool = True
    """Whether to use pickled features. If pickled features file does not exist or ``use_cache=False``, then features
    are pickled in ``cache_dir``. Pickled features include input ids, subtokens mask (mask of first tokens in words),
    encoded punctuation and capitalization labels, label ids. Features creation consumes considerable time and this
    ``use_cache=True`` significantly speeds up training starting. Pickled features are also used for sharing features
    between processes if data parallel training is used."""

    cache_dir: Optional[str] = None
    """A path to a directory containing cache or directory where newly created cache is saved. By default, it is
    a directory containing ``text_file``. You may need this parameter if cache for a dataset is going to be created
    and the dataset directory is read-only.

    ``cache_dir`` and ``label_info_save_dir`` are separate parameters for the case when a cache is ready and this cache
    is stored in a read only directory. In this case you will separate ``label_info_save_dir``."""

    get_label_frequences: bool = False
    """Whether to show and save label frequencies. Frequencies are showed if ``verbose`` parameter is ``True``. If
    ``get_label_frequencies=True``, then frequencies are saved into ``label_info_save_dir``"""

    verbose: bool = True
    """If ``True`` dataset instance will print progress messages and examples of acquired features."""

    n_jobs: Optional[int] = 0
    """Number of workers used for features creation (tokenization, label encoding, and clipping). If 0, then
    multiprocessing is not used; if ``None``, then n_jobs is equal to the number of CPU cores.
    There can be weird deadlocking errors with some tokenizers (e.g. SentencePiece) if ``n_jobs`` is greater than zero.
    """

    #################################################
    # TARRED DATASET PARAMETERS
    #################################################
    tar_metadata_file: Optional[str] = None
    """A path to tarred dataset metadata file. Tarred metadata file and other parts of tarred dataset are usually
    created by the script
    `examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py
    <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py>`_
    """

    tar_shuffle_n: int = 1
    """The size of shuffle buffer of `webdataset`. The number of batches which are permuted."""

    shard_strategy: Optional[str] = 'scatter'
    """Tarred dataset shard distribution strategy chosen as a str value during ddp. Accepted values are `scatter` and `replicate`.
    `scatter`: The default shard strategy applied by WebDataset, where each node gets a unique set of shards, which are permanently
    pre-allocated and never changed at runtime. `replicate` is an optional shard strategy, where each node gets the entire set of shards
    available in the tarred dataset, which are permanently pre-allocated and never changed at runtime. The benefit of replication is that
    it allows each node to sample data points from the entire dataset independently of other nodes, and reduces dependence on value of
    ``tar_shuffle_n``.

    .. warning::
        Replicated strategy allows every node to sample the entire set of available tar files, and therefore more than one node may sample
        the same tarfile, and even sample the same data points! As such, there is no assured guarantee that all samples in the dataset
        will be sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific occasions (when the number of
        shards is not divisible with ``world_size``), will not sample the entire dataset. For these reasons it is not advisable to use
        tarred datasets as validation or test datasets.
    """

    #################################################
    # PYTORCH DATALOADER PARAMETERS
    #################################################
    shuffle: bool = True
    """Shuffle batches every epoch. For regular training datasets, the parameter also activates batch repacking every
    epoch. For tarred dataset, it would be only batches permutation."""

    drop_last: bool = False
    """In cases when data parallelism is used, ``drop_last`` defines the way data pipeline behaves when some replicas
    are out of data and some are not. If ``drop_last`` is ``True``, then epoch ends in the moment when any replica runs
    out of data. If ``drop_last`` is ``False``, then the replica will replace missing batch with a batch from a pool of
    batches that the replica has already processed. If data parallelism is not used, then parameter ``drop_last`` does
    not do anything. For more information see ``torch.utils.data.distributed.DistributedSampler``"""

    pin_memory: bool = True
    """See ``torch.utils.data.DataLoader`` documentation."""

    num_workers: int = 8
    """See ``torch.utils.data.DataLoader`` documentation."""

    persistent_workers: bool = True
    """See ``torch.utils.data.DataLoader`` documentation."""


@dataclass
class PunctuationCapitalizationTrainDataConfig(PunctuationCapitalizationDataConfigBase):
    ds_item: Optional[str] = MISSING
    """Path to a directory where `tar_metadata_file` or `text_file` and `labels_file` lay."""


@dataclass
class PunctuationCapitalizationEvalDataConfig(PunctuationCapitalizationDataConfigBase):
    ds_item: Optional[Any] = MISSING
    """Path to a directory where `tar_metadata_file` or `text_file` and `labels_file` lay. ``Any`` = ``str`` or
    ``List[str]``. If a ``List[str]``, then the model is tested or validated on several datasets."""


def is_legacy_data_config(ds_section: DictConfig) -> bool:
    return 'use_tarred_dataset' not in ds_section


def legacy_data_config_to_new_data_config(
    ds_section: DictConfig, legacy_dataset_section: DictConfig, train: bool
) -> DictConfig:
    """
    Transform old style dataset to new format dataset.
    Args:
        ds_section: a ds section (``train_ds``, or ``validation_ds``, or ``test_ds``) from old style config. Such
            section contain ``batch_size`` parameter.
        legacy_dataset_section: a ``model.dataset`` section. ``model.dataset`` section contains ``data_dir`` parameter
        train: ``True`` if ``train_ds`` is transformed and ``False`` otherwise

    Returns:
        New format dataset based on either ``PunctuationCapitalizationTrainDataConfig`` (``train=True``) or
            ``PunctuationCapitalizationEvalDataConfig`` (``train=False``)
    """
    if train:
        cls = PunctuationCapitalizationTrainDataConfig
        ds_item = legacy_dataset_section.get('data_dir')
    else:
        cls = PunctuationCapitalizationEvalDataConfig
        ds_item = ds_section.get('ds_item')
        ds_item = legacy_dataset_section.get('data_dir') if ds_item is None else ds_item
    if ds_item is None:
        raise ValueError(
            f"Data directory was not found in legacy config.\nspecific dataset configuration:\n"
            f"{OmegaConf.to_yaml(ds_section)}\nmodel.dataset:\n{OmegaConf.to_yaml(legacy_dataset_section)}"
        )
    new_config = OmegaConf.structured(
        cls(
            use_tarred_dataset=False,
            text_file=ds_section.text_file,
            labels_file=ds_section.labels_file,
            ds_item=ds_item,
            max_seq_length=legacy_dataset_section.get(
                'max_seq_length', PunctuationCapitalizationDataConfigBase.max_seq_length
            ),
        )
    )
    return new_config


def _check_number_of_labels(
    words: List[str],
    query: str,
    qi: int,
    split_i: int,
    punctuation_labels: List[str],
    capitalization_labels: List[str],
) -> None:
    if len(words) != len(punctuation_labels):
        raise ValueError(
            f"Number of punctuation labels for a query number {qi} in a split number {split_i} is not equal to "
            f"number of words. Number of words: {len(words)}, number of punctuation labels: "
            f"{len(punctuation_labels)}. First 100 characters of the query: '{query[:100]}', punctuation labels: "
            f"'{punctuation_labels}'"
        )
    if len(words) != len(capitalization_labels):
        raise ValueError(
            f"Number of capitalization labels for a query number {qi} in a split number {split_i} is not equal to "
            f"number of words. Number of words: {len(words)}, number of capitalization labels: "
            f"{len(capitalization_labels)}. First 100 characters of the query: '{query[:100]}', "
            f"capitalization labels: '{capitalization_labels}'"
        )


def _show_prog(queues: Tuple[mp.Queue, ...], totals: List[int], descriptions: List[str], units: List[str]) -> None:
    """
    Show several ``tqdm`` progress bars.
    Args:
        queues: a list of queues by which progress is delivered into this function. Each queue is responsible for one
            progress bar. ``show_prog`` function extracts integers from ``queues`` elements and adds them to progress
            bars. If value extracted from a queue equals ``-1``, then corresponding progress bar is closed. When all
            progress bars are closed, this function returns.
        totals: list of values 100% of progress bars. See more in a description of ``total`` parameter of
            ``tqdm.tqdm`` function
        descriptions: list of descriptions of progress bars. See more in a description of ``desc`` parameter of
            ``tqdm.tqdm`` function
        units: list of progress bar units. See more in a description of ``unit`` parameter of ``tqdm.tqdm`` function
    """
    if not all([len(queues) == len(v) for v in [totals, descriptions, units]]):
        raise ValueError(
            f"All of parameters `queues`, `total_num_lines`, `descriptions`, `units` have to have equal lengths. "
            f"len(queues)={len(queues)}, len(total_num_lines)={len(totals)}, "
            f"len(descriptions)={len(descriptions)}, len(units)={len(units)}."
        )
    prog = [
        tqdm(total=tt, desc=dd, unit=uu, unit_scale=True, position=i)
        for i, (tt, dd, uu) in enumerate(zip(totals, descriptions, units))
    ]
    finished = [False] * len(queues)
    while True:
        for i, queue in enumerate(queues):
            stop = False
            to_add = 0
            try:
                v = queue.get(block=False)
                while v != -1:
                    to_add += v
                    v = queue.get(block=False)
                stop = True
            except Empty:
                if to_add == 0 and not stop:
                    continue
            prog[i].n += to_add
            prog[i].update(0)
            if prog[i].n >= totals[i]:
                finished[i] = True
                prog[i].close()
            if stop:
                if prog[i].n < totals[i]:
                    logging.warning(
                        f"Progress with description '{descriptions[i]}' terminated before progress bar "
                        f"reached 100%. prog.n={prog[i].n}, total_num_lines={totals[i]}"
                    )
                finished[i] = True
                prog[i].close()
        if all(finished):
            break
        sleep(0.1)


class Progress:
    """
    Manages several ``tqdm`` progress bars for multiprocess tasks. This class can be used as context manager.

    The class starts separate process which creates and updates progress bars. Information to progress process is
    passed via multiprocessing queues. There is a separate queue for every progress bar.

    You can use it as context manager:

    .. code-block:: python
        with Progress([10, 20], ["progress bar 1", "progress bar 2"], ["parrot", "frog"]) as progress_queues:
            num_processes = 10
            with multiprocessing.Pool(num_processes) as pool:
                data = list(zip(my_data, [progress_queues[0]] * num_processes, [progress_queues[1]] * num_processes))
                pool.starmap(worker_func, data)

    Or without context manager:

    .. code-block:: python
        progress = Progress([10, 20], ["progress bar 1", "progress bar 2"], ["parrot", "frog"])
        progress_queues = progress.get_queue()
        num_processes = 10
        with multiprocessing.Pool(num_processes) as pool:
            data = list(zip(my_data, [progress_queues[0]] * num_processes, [progress_queues[1]] * num_processes))
            pool.starmap(worker_func, data)
        progress.finish()

    In a worker function you will have to put number of processed items into the progress queues. For example:

    .. code-block:: python
        def worker_func(my_datum, parrot_progress_queue, frog_progress_queue):
            ...
            for i in range(10):
                parrot_progress_queue.put(1)
                frog_progress_queue.put(2)

    Progress bars and progress process are closed when ``finish`` or ``__exit__`` methods are called.
    """

    def __init__(self, total: Union[int, List[int]], desc: Union[str, List[str]], unit: Union[str, List[str]]) -> None:
        """
        Starts progress process and creates queues for passing information to the progress process. Number of progress
        bars is equal to the max length of lists ``total``, ``desc``, ``unit``. If none of these parameters is a list,
        then 1 progress bar is created.

        Args:
            total: a list of ``int`` which length is equal to the number of progress bars OR an ``int`` OR a list of
                one ``int``. Number which comprises 100% of progress bar. When sum of values passed through the
                corresponding queue equals ``total`` corresponding progress bar reaches 100%. If ``total`` is an
                ``int`` or a list of one element, then all progress bars have equal ``total`` parameter.
            desc: a list of ``str`` which length is equal to the number of progress bars OR a ``str`` OR a list of one
                ``str``. Description of a progress bar which is showed as a prefix. See more in description of
                parameter ``desc`` of function ``tqdm.tqdm``.
            unit: a list of ``str`` which length is equal to the number of progress bars OR a ``str`` OR a list of one
                ``str``. A unit of a progress bar. See more in description of parameter ``unit`` of function
                ``tqdm.tqdm``.
        """
        if not isinstance(total, list):
            total = [total]
        if not isinstance(desc, list):
            desc = [desc]
        if not isinstance(unit, list):
            unit = [unit]
        num_processes = max([len(total), len(desc), len(unit)])
        for param in [total, desc, unit]:
            if len(param) not in [num_processes, 1]:
                raise ValueError(
                    f"If parameter of `Progress.__init__` method is a list, then it has to be the same length as other "
                    f"parameters which are lists"
                )
            if len(param) == 1:
                param *= num_processes
        manager = mp.Manager()
        self.progress_queues = tuple(manager.Queue() for _ in range(num_processes))
        self.progress_process = mp.Process(target=_show_prog, args=(self.progress_queues, total, desc, unit))
        self.progress_process.start()

    def __enter__(self) -> Tuple[mp.Queue, ...]:
        return self.get_queues()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()

    def get_queues(self) -> Tuple[mp.Queue, ...]:
        return self.progress_queues

    def finish(self) -> None:
        for q in self.progress_queues:
            q.put(-1)
        self.progress_process.join()


class TokenizeCreateMasksClipWorker:
    """A worker for tokenization, encoding labels, creating masks for first token in a word, sequence clipping"""

    def __init__(
        self,
        max_seq_length: int,
        tokenizer: TokenizerSpec,
        punct_label_ids: Optional[Dict[str, int]],
        capit_label_ids: Optional[Dict[str, int]],
        pad_label: str,
        verbose: bool,
        progress_queue: mp.Queue,
    ) -> None:
        """
        Args:
            max_seq_length: max number of tokens in an input sequence including [CLS] and [SEP] tokens. If number of
                tokens in a sequence exceeds ``max_seq_length``, then excess tokens in the end of the sequence
                are removed
            tokenizer: a tokenizer instance which has properties ``cls_id``, ``pad_id``, ``sep_id``, ``unk_id``
            punct_label_ids: dict to map punctuation labels to label ids. Starts with pad_label->0.
            capit_label_ids: dict to map capitalization labels to label ids. Starts with pad_label->0.
            pad_label: pad value use for labels. By default, it's the neutral label for punctuation and capitalization.
                Its id in ``punct_label_ids`` and ``capit_label_ids`` has to be ``0``
            verbose: whether to report when the worker finishes its job
            progress_queue: a multiprocessing queue used for reporting progress. Useful for creating tarred dataset
        """
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.punct_label_ids = punct_label_ids
        self.capit_label_ids = capit_label_ids
        self.pad_label = pad_label
        self.verbose = verbose
        self.progress_queue = progress_queue

    def _maybe_clip(self, values: List[int], append_value: int) -> List[int]:
        if len(values) > self.max_seq_length:
            return values[: self.max_seq_length - 1] + [append_value]
        return values

    def __call__(
        self,
        queries: List[str],
        punct_label_lines: Optional[Union[List[str], Tuple[str, ...]]],
        capit_label_lines: Optional[Union[List[str], Tuple[str, ...]]],
        split_i: int,
        audio_queries: Optional[List[str]] = None,
        sample_rate: Optional[int] = None,
        preload_audios: Optional[bool] = True,
    ) -> Tuple[
        List[ndarray],
        List[ndarray],
        List[ndarray],
        List[ndarray],
        Union[List[Any], List[None]],
        Union[List[Any], List[None]],
        Union[List[Any], List[None]],
    ]:
        """
        Tokenize, clip, encode labels, and create masks of first tokens in words.

        Args:
            queries: text sequences
            punct_label_lines: a list or a tuple of labels for every word in a sequence (str)
            capit_label_lines: a list of a tuple labels for every word in a sequence (str)
            split_i: number of a split which is processed. Used for logging
            audio_queries: a list of audio filepaths
            sample_rate: target sample rate of audios
            preload_audios: whether to preload audios or not

        Returns:
            input_ids: a list of 1D int32 arrays. Each array contains token ids of the corresponding query
            subtokens_mask: a list of 1D boolean arrays. An array element is ``True`` if corresponding token is the
                first token in a word
            punct_labels: a list of 1D int32 arrays. Encoded punctuation labels for every token in a query. Tokens in
                one word have identical labels
            capit_labels: a list of 1D int32 arrays. Encoded capitalization labels for every token in a query. Tokens
                in one word have identical labels
        """
        all_input_ids, all_subtokens_mask, punct_all_labels, capit_all_labels = [], [], [], []
        dummy = [None] * len(queries)  # Needed to avoid code duplication with different values of `self.use_audio`
        all_audio_waveforms = [] if preload_audios else dummy
        audio_lengths = [] if preload_audios else dummy
        audio_filepaths = [] if not preload_audios else dummy
        progress_made = 0
        queries = zip(queries, audio_queries) if audio_queries else zip(queries, dummy)
        for i, (query, audio_query) in enumerate(queries):
            words = query.split()
            input_ids, subtokens_mask = [self.tokenizer.cls_id], [0]
            _check_number_of_labels(words, query, i, split_i, punct_label_lines[i], capit_label_lines[i])
            pad_id = self.punct_label_ids[self.pad_label]
            punct_labels = [pad_id]
            punct_query_labels = [self.punct_label_ids[lab] for lab in punct_label_lines[i]]
            capit_labels = [pad_id]
            capit_query_labels = [self.capit_label_ids[lab] for lab in capit_label_lines[i]]
            for j, word in enumerate(words):
                word_ids = self.tokenizer.text_to_ids(word)
                if not word_ids and len(word):
                    word_ids = [self.tokenizer.unk_id]
                input_ids.extend(word_ids)

                subtokens_mask.append(1)
                subtokens_mask.extend([0] * (len(word_ids) - 1))

                punct_labels.extend([punct_query_labels[j]] * len(word_ids))
                capit_labels.extend([capit_query_labels[j]] * len(word_ids))

            # add eos token
            input_ids.append(self.tokenizer.sep_id)
            subtokens_mask.append(0)

            all_input_ids.append(np.array(self._maybe_clip(input_ids, self.tokenizer.sep_id), dtype=np.int32))
            all_subtokens_mask.append(np.array(self._maybe_clip(subtokens_mask, 0), dtype=bool))

            punct_labels.append(pad_id)
            punct_all_labels.append(np.array(self._maybe_clip(punct_labels, pad_id), dtype=np.int32))
            capit_labels.append(pad_id)
            capit_all_labels.append(np.array(self._maybe_clip(capit_labels, pad_id), dtype=np.int32))
            if preload_audios and audio_query:
                if ASR_AVAILABLE:
                    segment = AudioSegment.from_file(audio_query.strip(), target_sr=sample_rate)
                    all_audio_waveforms.append(segment.samples)
                    audio_lengths.append(segment.num_samples)
                else:
                    raise ModuleNotFoundError(
                        'Nemo ASR was not installed, see https://github.com/NVIDIA/NeMo#installation for installation instructions'
                    )

            elif audio_query:
                audio_filepaths.append(audio_query.strip())

            progress_made += 1
            if progress_made >= TOKENIZATION_PROGRESS_REPORT_PERIOD:
                self.progress_queue.put(progress_made)
                progress_made = 0

        self.progress_queue.put(progress_made)
        if self.verbose:
            logging.info(f"Finished processing data split number {split_i}")

        return (
            all_input_ids,
            all_subtokens_mask,
            punct_all_labels,
            capit_all_labels,
            all_audio_waveforms,
            audio_lengths,
            audio_filepaths,
        )


def _get_features(
    queries: Union[List[str], Tuple[str, ...]],
    punct_label_lines: Union[List[str], Tuple[str, ...]],
    capit_label_lines: Union[List[str], Tuple[str, ...]],
    max_seq_length: int,
    tokenizer: TokenizerSpec,
    punct_label_ids: Dict[str, int] = None,
    capit_label_ids: Dict[str, int] = None,
    pad_label: str = 'O',
    verbose: bool = True,
    n_jobs: Optional[int] = 0,
    progress_queue: Optional[mp.Queue] = None,
    audio_queries: Optional[List[str]] = None,
    sample_rate: Optional[int] = None,
    preload_audios: Optional[bool] = True,
) -> Tuple[List[Any], List[Any], List[Any], List[Any], List[Any], List[Any], List[Any]]:
    """
    Tokenizes data, encodes labels, creates masks of first tokens in words, clips sequences by number of tokens.

    Args:
        queries: text sequences
        max_seq_length: max number of tokens in an input sequence including [CLS] and [SEP] tokens. If number of tokens
            in a sequence exceeds ``max_seq_length``, then excess tokens in the end of the sequence are removed
        tokenizer: a tokenizer instance which has properties ``cls_id``, ``pad_id``, ``sep_id``, ``unk_id``
        punct_label_ids: dict to map punctuation labels to label ids. Starts with pad_label->0.
        capit_label_ids: dict to map capitalization labels to label ids. Starts with pad_label->0.
        pad_label: pad value use for labels. By default, it's the neutral label for punctuation and capitalization.
            Its id in ``punct_label_ids`` and ``capit_label_ids`` has to be ``0``
        punct_label_lines: a list of a tuple of labels for every word in a sequence (str)
        capit_label_lines: a list or a tuple of labels for every word in a sequence (str)
        verbose: whether to show examples of tokenized data and various progress information
        n_jobs: a number of workers used for preparing features. If ``n_jobs <= 0``, then do not use multiprocessing
            and run features creation in this process. If not set, number of workers will be equal to the number of
            CPUs.

            !!WARNING!!
            There can be deadlocking problems with some tokenizers (e.g. SentencePiece, HuggingFace AlBERT)
            if ``n_jobs > 0``.

        progress_queue: a multiprocessing queue used for reporting progress. Useful for creating tarred dataset
        audio_queries: a list of audio filepaths
        sample_rate: target sample rate of audios
        preload_audios: whether to preload audios or not

    Returns:
        input_ids: a list of 1D int32 arrays. Each array contains token ids of corresponding query
        subtokens_mask: a list of 1D boolean arrays. An array element is ``True`` if corresponding token is the
            first token in a word
        punct_labels: a list of 1D int32 arrays. Encoded punctuation labels for every token in a query. Tokens in one
            word have identical labels.
        capit_labels: a list of 1D int32 arrays. Encoded capitalization labels for every token in a query. Tokens in
            one word have identical labels
    """
    if verbose:
        logging.info("Start initial tokenization.")
    create_progress_process = progress_queue is None
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), len(queries))

    if verbose:
        logging.info(f"Running tokenization with {n_jobs} jobs.")

    # Number of queries in split
    split_size = min(len(queries) // max(n_jobs, 1), MAX_NUM_QUERIES_IN_SPLIT)
    n_split = len(queries) // split_size
    split_queries = [queries[split_size * i : split_size * (i + 1)] for i in range(n_split - 1)] + [
        queries[split_size * (n_split - 1) :]
    ]
    split_punct_labels_lines = [
        punct_label_lines[split_size * i : split_size * (i + 1)] for i in range(n_split - 1)
    ] + [punct_label_lines[split_size * (n_split - 1) :]]
    split_capit_labels_lines = [
        capit_label_lines[split_size * i : split_size * (i + 1)] for i in range(n_split - 1)
    ] + [capit_label_lines[split_size * (n_split - 1) :]]

    args = list(zip(split_queries, split_punct_labels_lines, split_capit_labels_lines, range(n_split)))
    if audio_queries:
        split_audio_queries = [audio_queries[split_size * i : split_size * (i + 1)] for i in range(n_split - 1)] + [
            audio_queries[split_size * (n_split - 1) :]
        ]

        args = list(
            zip(
                split_queries,
                split_punct_labels_lines,
                split_capit_labels_lines,
                range(n_split),
                split_audio_queries,
                [sample_rate for _ in range(n_split)],
                [preload_audios for _ in range(n_split)],
            )
        )
    if create_progress_process:
        progress = Progress(len(queries), "Tokenization", "query")
        progress_queue = progress.get_queues()[0]
    if n_jobs > 0:
        with mp.Pool(n_jobs) as pool:
            result = pool.starmap(
                TokenizeCreateMasksClipWorker(
                    max_seq_length, tokenizer, punct_label_ids, capit_label_ids, pad_label, verbose, progress_queue,
                ),
                args,
            )
    else:
        result = []
        for x in args:
            result.append(
                TokenizeCreateMasksClipWorker(
                    max_seq_length, tokenizer, punct_label_ids, capit_label_ids, pad_label, verbose, progress_queue,
                )(*x)
            )
    if create_progress_process:
        progress.finish()

    input_ids, subtokens_mask, punct_labels, capit_labels, waveforms, audio_lengths, audio_filepaths = tuple(
        list(itertools.chain(*e)) for e in zip(*result)
    )
    if verbose:
        logging.info("Finished initial tokenization.")
        get_stats([len(inp) for inp in input_ids])
        logging.info(f"Finished clipping and padding.")
        for i in range(min(len(input_ids), 5)):
            logging.info("*** Example ***")
            logging.info("i: %s" % i)
            logging.info("subtokens: %s" % " ".join(list(map(str, input_ids[i]))))
            logging.info("subtokens_mask: %s" % " ".join(list(map(str, subtokens_mask[i]))))
            logging.info("punct_labels: %s" % " ".join(list(map(str, punct_labels[i]))))
            logging.info("capit_labels: %s" % " ".join(list(map(str, capit_labels[i]))))

    return (
        input_ids,
        subtokens_mask,
        waveforms,
        audio_lengths,
        audio_filepaths,
        punct_labels,
        capit_labels,
    )


def create_masks_and_segment_ids(
    input_ids: np.ndarray,
    subtokens_mask: np.ndarray,
    pad_id: int,
    cls_id: int,
    sep_id: int,
    ignore_start_end: bool,
    ignore_extra_tokens: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates segment ids array, input mask, loss mask.

    Segment ids array is BERT token type ids in HuggingFace terminology. It is a zeros array for punctuation
    and capitalization task.

    Input mask element is ``True`` if an element of ``input_ids`` is not padding and ``False`` otherwise.

    Loss mask element is ``True`` for the first token in a word. If ``ignore_start_end=False``, then loss mask
    element is ``True`` for [CLS] and [SEP] tokens. If ``ignore_extra_tokens=False``, then loss mask element is ``True``
    for all word tokens. In all other cases loss mask elements are ``False``.

    Args:
        input_ids: an integer array of shape ``[Batch, Time]`` containing ids of source token ids
        subtokens_mask: a boolean array of shape ``[Batch, Time]`` which elements are ``True`` if they correspond to
            the first token of some word
        pad_id: an id of padding token
        cls_id: an id of [CLS] token
        sep_id: an id of [SEP] token
        ignore_start_end: whether to compute loss for [CLS] and [SEP] tokens
        ignore_extra_tokens: whether to compute loss for not first tokens in words

    Returns:
        segment_ids: int8 array of shape [Batch, Time]
        input_mask: boolean array of shape [Batch, Time]
        loss_mask: boolean array of shape [Batch, Time]
    """
    segment_ids = np.zeros_like(input_ids, dtype=np.int8)
    input_mask = np.not_equal(input_ids, pad_id)
    special_mask = np.equal(input_ids, cls_id) & np.equal(input_ids, sep_id)
    if ignore_start_end:
        if ignore_extra_tokens:
            loss_mask = subtokens_mask
        else:
            loss_mask = input_mask & ~special_mask
    else:
        if ignore_extra_tokens:
            loss_mask = subtokens_mask | special_mask
        else:
            loss_mask = input_mask
    return segment_ids, input_mask, loss_mask


def create_label_ids(unique_labels: Set[str], pad_label: str) -> Dict[str, int]:
    """
    Returns label ids dictionary. ``pad_label`` always has id ``0``. Other labels are sorted in alphabetical order.
    Args:
        unique_labels: a set of labels from which label ids dictionary is created. May or may not contain ``pad_label``
        pad_label: label used for padding. It is also a neutral label

    Returns:
        label ids dictionary
    """
    label_ids = {pad_label: 0}
    if pad_label in unique_labels:
        unique_labels.remove(pad_label)
    for label in sorted(unique_labels):
        label_ids[label] = len(label_ids)
    return label_ids


def load_label_ids(file_path: Union[str, os.PathLike]) -> Dict[str, int]:
    ids = {}
    with open(file_path, encoding='utf_8') as f:
        for i, line in enumerate(f):
            ids[line.strip()] = i
    return ids


def save_label_ids(label_ids: Dict[str, int], file_path: Path) -> None:
    """
    Saves label ids map to a file. In each line of a file one label is saved. Labels are saved in the order of
    increasing of their ids.

    Args:
        label_ids: label id dictionary. Pad label has to have id ``0``
        file_path: path to a file where labels will be saved
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('w', encoding='utf_8', newline='\n') as out:
        labels, _ = zip(*sorted(label_ids.items(), key=lambda x: x[1]))
        out.write('\n'.join(labels))


def raise_not_equal_labels_error(
    first_labels: Dict[str, int], second_labels: Dict[str, int], first_labels_desc: str, second_labels_desc: str
) -> None:
    """
    A helper function for raising comprehensible error if labels from 2 sources are different.
    Such sources may include:
      - labels stored in .nemo checkpoint
      - labels stored in tarred dataset
      - labels passed in config parameters ``model.common_dataset_parameters.{punct_label_ids,capit_label_ids}``
      - labels from files passed in config parameters ``model.class_labels.{punct_labels_file,capit_labels_file}``
      - labels in attributes ``PunctuationCapitalizationModel.{punct_label_ids,capit_label_ids}``
      - any other source
    This function helps to detect configuration early and give error messages that are easy to interpret.
    Call this function if ``first_labels != second_labels``.

    Args:
        first_labels: first dictionary with labels
        second_labels: second dictionary with labels
        first_labels_desc: a description of first labels
        second_labels_desc: a description of second labels
    """
    missing_in_first = {k: second_labels[k] for k in set(second_labels) - set(first_labels)}
    missing_in_second = {k: first_labels[k] for k in set(first_labels) - set(second_labels)}
    not_equal = {
        k: {'FIRST LABELS': first_labels[k], 'SECOND LABELS': second_labels[k]}
        for k in set(first_labels) & set(second_labels)
        if first_labels[k] != second_labels[k]
    }
    msg = f"{first_labels_desc} (FIRST LABELS) are not equal to {second_labels_desc} (SECOND LABELS)."
    if len(missing_in_first) > 0:
        msg += f" Number of SECOND LABELS missing in the FIRST LABELS: {len(missing_in_first)}."
    if len(missing_in_second) > 0:
        msg += f" Number of FIRST LABELS missing in the SECOND LABELS: {len(missing_in_second)}."
    if len(not_equal) > 0:
        msg += f" Number of labels which are not equal: {len(not_equal)}."
    if len(missing_in_first) > 0:
        msg += (
            f" Several examples of missing SECONDS LABELS in the FIRST LABELS: "
            f"{dict(list(missing_in_first.items())[:3])}."
        )
    if len(missing_in_second) > 0:
        msg += (
            f" Several examples of missing FIRST LABELS in the SECOND LABELS: "
            f"{dict(list(missing_in_second.items())[:3])}."
        )
    if len(not_equal) > 0:
        msg += f" Several examples of labels which are not equal: {dict(list(not_equal.items())[:3])}"
    raise ValueError(msg)


def pad(vectors: List[np.ndarray], length: int, value: Union[int, float, bool]) -> np.ndarray:
    """
    Pad vectors to length ``length`` and then stack.
    Args:
        vectors: a list of 1D arrays. Arrays to pad and stack
        length: a length of padded sequence. Has to be greater or equal to the maximum length of an element of
            ``vectors``.
        value: a value used for padding

    Returns:
        an array of padded vectors
    """
    result = []
    for v in vectors:
        result.append(np.concatenate([v, np.full([length - v.shape[0]], value, dtype=v.dtype)]))
    return np.stack(result)


class BertPunctuationCapitalizationDataset(Dataset):
    """
    A dataset to use during training for punctuation and capitalization tasks.
    For inference, you will need
    :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_infer_dataset.BertPunctuationCapitalizationInferDataset`.
    For huge datasets which cannot be loaded into memory simultaneously use
    :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset.BertPunctuationCapitalizationTarredDataset`.

    Args:
        text_file (:obj:`Union[str, os.PathLike]`): a path to a file with sequences, each line should contain a text
            without punctuation and capitalization
        labels_file (:obj:`Union[str, os.PathLike]`): a path to a file with labels, each line corresponds to word
            labels for a sentence in the ``text_file``. Labels have to follow format described in this section of
            documentation :ref:`NeMo Data Format<nemo-data-format-label>`.
        max_seq_length (:obj:`int`): max number of tokens in a source sequence. ``max_seq_length`` includes for [CLS]
            and [SEP] tokens. Sequences which are too long will be clipped by removal of tokens from the end of the
            sequence.
        tokenizer (:obj:`TokenizerSpec`): a tokenizer instance which has properties ``unk_id``, ``sep_id``, ``bos_id``,
            ``eos_id``.
        num_samples (:obj:`int`, `optional`, defaults to :obj:`-1`): a number of samples you want to use for the
            dataset. If ``-1``, use all dataset. Useful for testing.
        tokens_in_batch (:obj:`int`, `optional`, defaults to :obj:`5000`): number of tokens in a batch including
            paddings and special tokens ([CLS], [SEP], [UNK]). This class :meth:`__getitem__` method returns not
            samples but ready batches. Number of samples in a batch is adjusted for input sequences lengths. If input
            sequences are short, then a batch will contain more samples. Before packing into batches, samples are
            sorted by number of tokens they contain. Sorting allows to reduce number of pad tokens in a batch
            significantly. Regular PyTorch data loader shuffling will only permute batches with changing their content.
            Proper shuffling is achieved via calling method :meth:`repack_batches_with_shuffle` every epoch. If
            parameter ``number_of_batches_is_multiple_of`` is greater than 1, some batches may be split into smaller
            pieces.
        pad_label (:obj:`str`, `optional`, defaults to :obj:`'O'`): pad value to use for labels. It's also the neutral
            label both for punctuation and capitalization.
        punct_label_ids (:obj:`Dict[str, int]`, `optional`): dict to map punctuation labels to label ids. For dev set,
            use label ids generated during training to support cases when not all labels are present in the dev set.
            For training, it is recommended to set ``punct_label_ids`` to ``None`` or load from cache.
        capit_label_ids (:obj:`Dict[str, int]`, `optional`): same ``punct_label_ids`` for capitalization labels.
        ignore_extra_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`): whether to compute loss on
            tokens which are not first tokens in a word. For example, assume that word ``'tokenization'`` is tokenized
            into ``['token', 'ization']``. If ``ignore_extra_tokens=True``, loss mask for the word is
            ``[True, False]``, and if ``ignore_extra_tokens=False``, then loss mask is ``[True, True]``.
        ignore_start_end (:obj:`bool`, `optional`, defaults to :obj:`True`): whether to ignore [CLS] and [SEP] tokens
            in the loss_mask.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`): whether to use pickled features already present
            in ``cache_dir`` or not. If pickled features file does not exist or ``use_cache=False``, then features are
            pickled in ``cache_dir``. Pickled features include input ids, subtokens mask (mask of first tokens in
            words), encoded punctuation and capitalization labels, label ids. Features creation consumes considerable
            time and this ``use_cache=True`` significantly speeds up training starting. Pickled features are also
            used for sharing features between processes if data parallel training is used.
        cache_dir (:obj:`Union[str, os.PathLike]`, `optional`): a path to a directory where cache (pickled features)
            is stored. By default, ``text_file`` parent directory is used. This parameter is useful if dataset
            directory is read-only, and you wish to pickle features. In such a case specify a path to directory which
            allows writing in ``cache_dir`` parameter.
        get_label_frequencies (:obj:`bool`, `optional`, defaults to :obj:`False`): whether to print and save label
            frequencies. Frequencies are showed if ``verbose`` parameter is ``True``. If
            ``get_label_frequencies=True``, then frequencies are saved into ``label_info_save_dir`` directory.
        label_info_save_dir (:obj:`Union[str, os.PathLike]`, `optional`): a path to a directory where label frequencies
            are saved. By default, a ``text_file`` parent directory is used. When method
            :meth:`save_labels_and_get_file_paths` is called label ids are saved into ``label_info_save_dir``
            directory. This parameter is useful if directory containing ``text_file`` is read-only.
        punct_label_vocab_file (:obj:`Union[str, os.PathLike]`, `optional`): a path to a .csv file containing
            punctuation label vocabulary. Each line in such a vocabulary file contains exactly one label. The first
            line has to contain `pad_label`, otherwise error will be raised.
        capit_label_vocab_file (:obj:`Union[str, os.PathLike]`, `optional`): same as ``punct_label_vocab_file`` for
            capitalization labels.
        add_masks_and_segment_ids_to_batch (:obj:`bool`, `optional`, defaults to :obj:`True`): whether to add
            ``'loss_mask'``, ``'input_mask'``, ``'segment_ids'`` items to a batch. Useful for creation of tarred
            dataset and can NOT be used during model training and inference.
        verbose (:obj:`bool`, `optional`, defaults to :obj:`True`): whether to show data examples, label stats and
            other useful information.
        n_jobs (:obj:`int`, `optional`, defaults to :obj:`0`): number of workers used for tokenization, encoding
            labels, creating "first token in word" mask, and clipping. If ``n_jobs <= 0`` data preparation is performed
            without multiprocessing. By default, ``n_jobs`` is ``0``.

            .. warning::
                There can be deadlocking problems with some tokenizers (e.g. SentencePiece, HuggingFace AlBERT)
                if ``n_jobs > 0``.

        number_of_batches_is_multiple_of (:obj:`int`, `optional`, defaults to :obj:`1`): number of batches in the
            dataset is made divisible by ``number_of_batches_is_multiple_of``. If ``number_of_batches_is_multiple_of``
            is greater than 1, then several batches are split in parts until number of batches
            is divisible by ``number_of_batches_is_multiple_of``. If there is no enough queries in the dataset to
            create enough batches, then a warning is printed. This parameter is useful for dev and validation datasets
            if multiple GPUs are used. The problem is that if number of batches is not evenly divisible by number of
            GPUs, then some queries may be processed several times and metrics will be distorted.
        batch_shuffling_random_seed (:obj:`int`, defaults to :obj:`int`): a random seed used for batches repacking and
            shuffling.
        tokenization_progress_queue (:obj:`multiprocessing.Queue`, `optional`): a queue for reporting tokenization
            progress. Useful for creation of tarred dataset
        batch_mark_up_progress_queue (:obj:`multiprocessing.Queue`, `optional`): a queue for reporting progress in
            deciding which samples batches will contain. Useful for creation of tarred dataset
        batch_building_progress_queue (:obj:`multiprocessing.Queue`, `optional`): a queue for reporting progress in
            batch creation (stacking and padding). Useful for creation of tarred dataset
        use_audio (:obj:`bool`, `optional`, defaults to :obj: `False`): If set to True dataset will return audio as well as text.
        audio_file (:obj:`Union[str, os.PathLike]`, `optional`): a path to file with audio paths.
        sample_rate (:obj:`int`, `optional`, defaults to :obj:`None`): sample rate of audios. Can be used for up sampling or down sampling of audio.
        use_bucketing (:obj:`bool`, `optional`, defaults to :obj: `True`): If set to False dataset will return ``batch_size`` batches instead of ``number_of_tokens`` tokens.
        preload_audios (:obj:`bool`, `optional`, defaults to :obj: `True`): If set to True batches will include waveforms, if set to False will store audio_filepaths instead and load audios during ``collate_fn`` call
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports. """
        if self.use_audio:
            return {
                'input_ids': NeuralType(('B', 'T'), ChannelType()),
                'segment_ids': NeuralType(('B', 'T'), ChannelType()),
                'input_mask': NeuralType(('B', 'T'), MaskType()),
                'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
                'loss_mask': NeuralType(('B', 'T'), MaskType()),
                'punct_labels': NeuralType(('B', 'T'), LabelsType()),
                'capit_labels': NeuralType(('B', 'T'), LabelsType()),
                'features': NeuralType(('B', 'T'), AudioSignal()),
                'features_length': NeuralType(('B', 'T'), LengthsType()),
            }
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'punct_labels': NeuralType(('B', 'T'), LabelsType()),
            'capit_labels': NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(
        self,
        text_file: Union[str, os.PathLike],
        labels_file: Union[str, os.PathLike],
        max_seq_length: int,
        tokenizer: TokenizerSpec,
        num_samples: int = -1,
        tokens_in_batch: int = 5000,
        pad_label: str = 'O',
        punct_label_ids: Optional[Union[Dict[str, int], DictConfig]] = None,
        capit_label_ids: Optional[Union[Dict[str, int], DictConfig]] = None,
        ignore_extra_tokens: bool = False,
        ignore_start_end: bool = True,
        use_cache: bool = True,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        get_label_frequencies: bool = False,
        label_info_save_dir: Optional[Union[str, os.PathLike]] = None,
        punct_label_vocab_file: Optional[Union[str, os.PathLike]] = None,
        capit_label_vocab_file: Optional[Union[str, os.PathLike]] = None,
        add_masks_and_segment_ids_to_batch: bool = True,
        verbose: bool = True,
        n_jobs: Optional[int] = 0,
        number_of_batches_is_multiple_of: int = 1,
        batch_shuffling_random_seed: int = 42,
        tokenization_progress_queue: Optional[mp.Queue] = None,
        batch_mark_up_progress_queue: Optional[mp.Queue] = None,
        batch_building_progress_queue: Optional[mp.Queue] = None,
        use_audio: Optional[bool] = False,
        audio_file: Optional[Union[str, os.PathLike]] = None,
        sample_rate: Optional[int] = None,
        use_bucketing: Optional[bool] = True,
        preload_audios: Optional[bool] = True,
    ) -> None:
        """ Initializes BertPunctuationCapitalizationDataset. """
        if isinstance(punct_label_ids, DictConfig):
            punct_label_ids = OmegaConf.to_container(punct_label_ids)
        if isinstance(capit_label_ids, DictConfig):
            capit_label_ids = OmegaConf.to_container(capit_label_ids)

        self._check_constructor_parameters(
            text_file,
            labels_file,
            punct_label_ids,
            capit_label_ids,
            punct_label_vocab_file,
            capit_label_vocab_file,
            num_samples,
            use_cache,
            number_of_batches_is_multiple_of,
            use_audio,
            audio_file,
            sample_rate,
        )

        if punct_label_vocab_file is not None:
            punct_label_vocab_file = Path(punct_label_vocab_file).expanduser()
            punct_label_ids = load_label_ids(punct_label_vocab_file)
        if capit_label_vocab_file is not None:
            capit_label_vocab_file = Path(capit_label_vocab_file).expanduser()
            capit_label_ids = load_label_ids(capit_label_vocab_file)
        self.text_file, self.labels_file = Path(text_file).expanduser(), Path(labels_file).expanduser()
        if label_info_save_dir is None:
            self.label_info_save_dir = self.text_file.parent
        else:
            self.label_info_save_dir = Path(label_info_save_dir).expanduser()

        self.tokens_in_batch = tokens_in_batch
        self.tokenizer = tokenizer
        self.pad_label = pad_label
        self.ignore_extra_tokens = ignore_extra_tokens
        self.ignore_start_end = ignore_start_end
        self.add_masks_and_segment_ids_to_batch = add_masks_and_segment_ids_to_batch
        self.verbose = verbose
        self.batch_mark_up_progress_queue = batch_mark_up_progress_queue
        self.batch_building_progress_queue = batch_building_progress_queue
        self.use_audio = use_audio
        self.audio_file = audio_file
        self.sample_rate = sample_rate
        self.use_bucketing = use_bucketing
        self.preload_audios = preload_audios

        master_device = is_global_rank_zero()
        self.features_pkl = self._get_path_to_pkl_features(
            self.text_file, self.labels_file, cache_dir, max_seq_length, num_samples
        )
        features = None
        if master_device and not (self.features_pkl.is_file() and use_cache):
            if verbose:
                logging.info(
                    f'Processing {self.text_file}' + f' {self.audio_file if self.audio_file else ""} '.rstrip()
                )

            (
                text_lines,
                punct_label_lines,
                capit_label_lines,
                punct_unique_labels,
                capit_unique_labels,
                audio_lines,
            ) = self._read_dataset(self.text_file, self.labels_file, num_samples, self.audio_file)

            if punct_label_ids:
                self._check_label_ids_vs_unique_labels(
                    punct_label_ids, punct_unique_labels, 'punct', 'punctuation', self.labels_file
                )
            else:
                punct_label_ids = create_label_ids(punct_unique_labels, self.pad_label)
            if capit_label_ids:
                self._check_label_ids_vs_unique_labels(
                    capit_label_ids, capit_unique_labels, 'capit', 'capitalization', self.labels_file
                )
            else:
                capit_label_ids = create_label_ids(capit_unique_labels, self.pad_label)
            features = _get_features(
                text_lines,
                punct_label_lines,
                capit_label_lines,
                max_seq_length,
                self.tokenizer,
                pad_label=self.pad_label,
                punct_label_ids=punct_label_ids,
                capit_label_ids=capit_label_ids,
                verbose=self.verbose,
                progress_queue=tokenization_progress_queue,
                n_jobs=n_jobs,
                audio_queries=audio_lines if self.use_audio else None,
                sample_rate=self.sample_rate,
                preload_audios=self.preload_audios,
            )
            self.features_pkl.parent.mkdir(parents=True, exist_ok=True)

            # save features to a temp file first to make sure that non-master processes don't start reading the file
            # until the master process is done with writing
            ofd, tmp_features_pkl = tempfile.mkstemp(
                suffix='.pkl', prefix=os.path.basename(self.features_pkl), dir=os.path.dirname(self.features_pkl)
            )
            with os.fdopen(ofd, 'wb') as temp_f:
                pickle.dump(tuple(list(features) + [punct_label_ids, capit_label_ids]), temp_f)

            os.rename(tmp_features_pkl, self.features_pkl)

            if self.verbose:
                logging.info(f'Features saved to {self.features_pkl}')

        # wait until the master process writes to the processed data files
        if not master_device:
            while features is None and not os.path.exists(self.features_pkl):
                sleep(10)

        if features is None:
            features = pickle.load(self.features_pkl.open('rb'))
            li = features[-2:]
            self._check_label_ids_loaded_from_pkl(
                punct_label_ids, capit_label_ids, *li, punct_label_vocab_file, capit_label_vocab_file
            )
            punct_label_ids, capit_label_ids = li[-2], li[-1]
            if tokenization_progress_queue is not None:
                tokenization_progress_queue.put(len(features[0]))
            if self.verbose:
                logging.info(f'Features restored from {self.features_pkl}')
            features = features[:-2]

        (
            self.input_ids,
            self.subtokens_mask,
            self.waveforms,
            self.waveforms_length,
            self.audio_filepaths,
            self.punct_labels,
            self.capit_labels,
        ) = features
        self.punct_label_ids, self.capit_label_ids = punct_label_ids, capit_label_ids
        self.number_of_batches_is_multiple_of = number_of_batches_is_multiple_of
        self.batch_shuffling_random_state = np.random.RandomState(batch_shuffling_random_seed)
        if get_label_frequencies:
            self.punct_label_frequencies = self._calculate_and_save_label_frequencies(self.punct_labels, 'punct')
            self.capit_label_frequencies = self._calculate_and_save_label_frequencies(self.capit_labels, 'capit')
        if self.use_bucketing:
            self.batches = self._pack_into_batches(
                input_ids=self.input_ids,
                subtokens_mask=self.subtokens_mask,
                punct_labels=self.punct_labels,
                capit_labels=self.capit_labels,
                waveforms=self.waveforms,
                audio_lengths=self.waveforms_length,
                audio_filepaths=self.audio_filepaths,
            )
        else:
            self.batches = self._form_batches(
                input_ids=self.input_ids,
                subtokens_mask=self.subtokens_mask,
                punct_labels=self.punct_labels,
                capit_labels=self.capit_labels,
                waveforms=self.waveforms,
                audio_lengths=self.waveforms_length,
                audio_filepaths=self.audio_filepaths,
            )

    def _get_path_to_pkl_features(
        self,
        text_file: Path,
        labels_file: Path,
        cache_dir: Optional[Union[str, os.PathLike]],
        max_seq_length: int,
        num_samples: int,
    ) -> Path:
        if cache_dir is None:
            cache_dir = text_file.parent
        else:
            cache_dir = Path(cache_dir).expanduser()
        vocab_size = getattr(self.tokenizer, "vocab_size", 0)
        features_pkl = cache_dir / "cached.{}.{}.max_seq_length{}.vocab{}.{}.punctuation_capitalization.pkl".format(
            '__' + text_file.name + '__' + labels_file.name + '__',
            self.tokenizer.name,
            max_seq_length,
            vocab_size,
            f'num_samples{num_samples}' if num_samples > 0 else 'all_samples',
        )
        return features_pkl

    @staticmethod
    def _check_constructor_parameters(
        text_file: Union[str, os.PathLike],
        labels_file: Union[str, os.PathLike],
        punct_label_ids: Optional[Dict[str, int]],
        capit_label_ids: Optional[Dict[str, int]],
        punct_label_vocab_file: Union[str, os.PathLike],
        capit_label_vocab_file: Union[str, os.PathLike],
        num_samples: int,
        use_cache: bool,
        number_of_batches_is_multiple_of: int,
        use_audio: bool = False,
        audio_file: Optional[Union[str, os.PathLike]] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1 and not use_cache:
            raise ValueError(
                f"If you already created process group and the world size is greater than 1, then `use_cache` "
                f"parameter has to be `True`. Only master process prepares features and if `use_cache=False`, then "
                f"other processes will not be able to obtain features. Alternatively, you may set `use_cache=False` "
                f"and set up data before spawning processes. Use `cache_dir` dataset directory with "
                f"`text_file` and `labels_file` is read-only."
            )
        if not (os.path.exists(text_file) and os.path.exists(labels_file)):
            raise FileNotFoundError(
                f'{text_file} or {labels_file} not found. The data should be split into 2 files: text.txt and '
                f'labels.txt. Each line of the text.txt file contains text sequences, where words are separated with '
                f'spaces. The labels.txt file contains corresponding labels for each word in text.txt, the labels are '
                f'separated with spaces. Each line of the files should follow the format:\n'
                f'   [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and '
                f'   [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
            )
        if not use_audio and audio_file:
            raise ValueError(f"Audio file {audio_file} was passed but use_audio was set to False")
        if use_audio and audio_file and not os.path.exists(audio_file):
            raise FileNotFoundError(
                f'use_audio was set to True but {audio_file} not found. Audio data should be listed in .txt file with one path per line'
            )
        if punct_label_ids is not None and punct_label_vocab_file is not None:
            punct_label_vocab_file = Path(punct_label_vocab_file).expanduser()
            file_punct_label_ids = load_label_ids(punct_label_vocab_file)
            if file_punct_label_ids != punct_label_ids:
                raise_not_equal_labels_error(
                    first_labels=punct_label_ids,
                    second_labels=file_punct_label_ids,
                    first_labels_desc='Punctuation labels passed to the `PunctuationCapitalizationDataset` '
                    'constructor in parameter `punct_label_ids`',
                    second_labels_desc=f'Punctuation labels loaded from file {punct_label_vocab_file} path to which '
                    f'is passed in parameter `punct_label_vocab_file`',
                )
        if capit_label_ids is not None and capit_label_vocab_file is not None:
            capit_vocab_file = Path(capit_label_vocab_file).expanduser()
            file_capit_label_ids = load_label_ids(capit_vocab_file)
            if file_capit_label_ids != capit_label_ids:
                raise_not_equal_labels_error(
                    first_labels=capit_label_ids,
                    second_labels=file_capit_label_ids,
                    first_labels_desc='Capitalization labels passed to the `PunctuationCapitalizationDataset` '
                    'constructor in parameter `capit_label_ids`',
                    second_labels_desc=f'Capitalization labels loaded from file {capit_label_vocab_file} path to '
                    f'which is passed in parameter `capit_label_vocab_file`',
                )
        if num_samples == 0:
            raise ValueError(
                f"Parameter `num_samples` has to be positive or negative whereas `num_samples={num_samples}`. "
                f"Negative `num_samples` is for using all samples in a dataset."
            )
        if number_of_batches_is_multiple_of < 1 or not isinstance(number_of_batches_is_multiple_of, int):
            raise ValueError(
                f"Parameter `number_of_batches_is_multiple_of` has to be positive integer whereas "
                f"{number_of_batches_is_multiple_of} is given."
            )

        if use_audio and not isinstance(sample_rate, int):
            raise TypeError(f'use_audio was set to True but sample_rate was not set')

        if use_audio and sample_rate < 1:
            raise ValueError(f'sample_rate set to {sample_rate} but it cannot be less than 1')

    def _check_label_ids_loaded_from_pkl(
        self,
        parameter_punct_label_ids: Dict[str, int],
        parameter_capit_label_ids: Dict[str, int],
        pkl_punct_label_ids: Any,
        pkl_capit_label_ids: Any,
        punct_label_vocab_file: Optional[Path],
        capit_label_vocab_file: Optional[Path],
    ) -> None:
        if not isinstance(pkl_punct_label_ids, dict):
            raise ValueError(
                f"Punctuation label ids loaded from features file {self.features_pkl} have wrong type "
                f"{type(pkl_punct_label_ids)}"
            )
        if parameter_punct_label_ids is not None:
            if parameter_punct_label_ids != pkl_punct_label_ids:
                raise_not_equal_labels_error(
                    first_labels=parameter_punct_label_ids,
                    second_labels=pkl_punct_label_ids,
                    first_labels_desc="Punctuation labels passed in parameter `punct_label_ids`"
                    if punct_label_vocab_file is None
                    else f"Punctuation labels loaded from file {punct_label_vocab_file}",
                    second_labels_desc=f"Punctuation label ids loaded from features file {self.features_pkl}",
                )
        if not isinstance(pkl_capit_label_ids, dict):
            raise ValueError(
                f"Capitalization label ids loaded from features file {self.features_pkl} has wrong type "
                f"{type(pkl_capit_label_ids)}"
            )
        if parameter_capit_label_ids is not None:
            if parameter_capit_label_ids != pkl_capit_label_ids:
                raise_not_equal_labels_error(
                    first_labels=parameter_capit_label_ids,
                    second_labels=pkl_capit_label_ids,
                    first_labels_desc="Capitalization labels passed in parameter `capit_label_ids`"
                    if capit_label_vocab_file is None
                    else f"Capitalization labels loaded from file {capit_label_vocab_file}",
                    second_labels_desc=f"Capitalization label ids loaded from features file {self.features_pkl}",
                )

    @staticmethod
    def _check_label_ids_vs_unique_labels(
        label_ids: Dict[str, int], unique_labels: Set[str], label_type: str, task: str, label_file: Path
    ) -> None:
        if unique_labels - set(label_ids):
            not_present_labels = list(unique_labels - set(label_ids))
            raise ValueError(
                f"{len(not_present_labels)} {task} labels found in {label_file} are not present in "
                f"`{label_type}_label_ids`. Examples of unexpected labels from {label_file}: {not_present_labels[:3]}"
            )

    @staticmethod
    def _read_dataset(
        text_file: Path, labels_file: Path, num_samples: int, audio_file: Optional[Path] = None
    ) -> Union[Tuple[Any, Any, Any, Set[Any], Set[Any], Any], Tuple[Any, Any, Any, Set[Any], Set[Any]]]:
        with open(text_file, 'r', encoding='utf_8') as f:
            text_lines = f.readlines()
        punct_unique_labels, capit_unique_labels = set(), set()
        punct_labels_lines, capit_labels_lines = [], []
        with labels_file.open(encoding='utf_8') as f:
            for i, line in enumerate(f):
                pairs = line.split()
                if not all([len(p) == 2 for p in pairs]):
                    raise ValueError(
                        f"Some label pairs are not pairs but have wrong length (!= 2) in line {i} in label file "
                        f"{labels_file}"
                    )
                words = text_lines[i].split()
                if len(pairs) != len(words):
                    raise ValueError(
                        f"In line {i} in text file {text_file} number of words {len(words)} is not equal to the "
                        f"number of labels {len(pairs)} in labels file {labels_file}."
                    )
                punct_line, capit_line = zip(*pairs)
                punct_labels_lines.append(punct_line)
                capit_labels_lines.append(capit_line)
                punct_unique_labels.update(punct_line)
                capit_unique_labels.update(capit_line)
        if len(punct_labels_lines) != len(text_lines):
            raise ValueError(
                f"Number of text lines {len(text_lines)} in text file {text_file} is not equal to the number of lines "
                f"{len(punct_labels_lines)} in labels file {labels_file}."
            )

        if audio_file:
            with open(audio_file, 'r') as f:
                audio_lines = f.readlines()
            if len(audio_lines) != len(text_lines):
                raise ValueError(
                    f'Number of lines in {audio_file} equals {len(audio_lines)} which is not equal to '
                    f'number of lines in {text_file} which is {len(text_lines)}'
                )
            dataset = list(zip(text_lines, punct_labels_lines, capit_labels_lines, audio_lines))
        else:
            dataset = list(zip(text_lines, punct_labels_lines, capit_labels_lines))
        if len(dataset) == 0:
            raise ValueError(f"Dataset loaded from files {text_file} and {labels_file} is empty.")
        if num_samples > 0:
            dataset = dataset[:num_samples]
        if audio_file:
            text_lines, punct_labels_lines, capit_labels_lines, audio_lines = zip(*dataset)
            return (
                text_lines,
                punct_labels_lines,
                capit_labels_lines,
                punct_unique_labels,
                capit_unique_labels,
                audio_lines,
            )
        else:
            text_lines, punct_labels_lines, capit_labels_lines = zip(*dataset)
            return text_lines, punct_labels_lines, capit_labels_lines, punct_unique_labels, capit_unique_labels, None

    @staticmethod
    def calc_batch_seq_length(queries: List[np.ndarray], length_is_multiple_of: int) -> int:
        return ceil(max([len(elem) for elem in queries]) / length_is_multiple_of) * length_is_multiple_of

    def _adjust_number_of_batches(
        self,
        input_ids: List[np.ndarray],
        batch_beginnings: List[int],
        batch_sizes: List[int],
        batch_seq_lengths: List[int],
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        If length of ``batch_sizes`` list is not divisible by ``self.number_of_batches_is_multiple_of``, then
        one or several batches are split into parts until number of batches is divisible by
        ``self.number_of_batches_is_multiple_of``.

        The method selects a batch and tries to slice smaller batches with 8 elements each from the batch. If
        the batch cannot be sliced any further and there are still not enough batches, then the next batch from dataset
        is selected.

        If slicing batches of size 8 is not enough, then batches of size 1 are created.

        If dataset is too small to create enough batches, then a warning is shown.

        Args:
            input_ids: tokenized queries of the dataset. `input_ids` are expected to be sorted by length in ascending
                order.
            batch_beginnings: indices of first elements of batches created inside :meth:`_mark_up_batches` method.
                Expected to be sorted in ascending order.
            batch_sizes: sizes of batches created inside :meth:`_mark_up_batches` method.
            batch_seq_lengths: lengths of elements in batch after padding created inside :meth:`_mark_up_batches`
                method.

        Returns:
            batch_beginnings: a list of indices in ``input_ids`` of first samples of every batch
            batch_sizes: a list of numbers of samples in batches
            batch_seq_lengths: a list of sequence lengths after padding for every batch
        """
        batch_beginnings, batch_sizes = batch_beginnings.copy(), batch_sizes.copy()
        batch_seq_lengths = batch_seq_lengths.copy()
        num_missing_batches = (
            self.number_of_batches_is_multiple_of - len(batch_sizes) % self.number_of_batches_is_multiple_of
        )
        if num_missing_batches == 0:
            return batch_beginnings, batch_sizes, batch_seq_lengths
        if sum(batch_sizes) - len(batch_sizes) < num_missing_batches:
            logging.warning(
                f"Unable to achieve number of batches multiple of {self.number_of_batches_is_multiple_of} because "
                f"dataset in files '{self.text_file}' and '{self.labels_file}' contains not enough queries "
                f"({sum(batch_sizes)}) or queries in the dataset are too long. Dataset will have "
                f"{len(batch_sizes)} batches instead. For validation or test dataset if multiple GPUs are used "
                f"this will lead to distorted metrics because some batches will be processed several times. "
                f"To fix this problem you may try to tweak (increase) parameter `tokens_in_batch`, though result is "
                f"not guaranteed."
            )
            return batch_beginnings, batch_sizes, batch_seq_lengths
        num_cut = 0
        for ss in [8, 1]:  # ss - split_size
            old_num_batches = len(batch_sizes)
            # Starting from the last batch because its size is likely to be not multiple of 8. Thus number of
            # batches which size is not multiple of 8 can be reduced by 1.
            original_batch_index = old_num_batches - 1
            while original_batch_index >= 0 and num_cut < num_missing_batches:
                bs, bb = batch_sizes[original_batch_index], batch_beginnings[original_batch_index]
                rb = 0  # an index of sliced first element of sliced batch in original batch (relative beginning)
                if rb < bs - ss:
                    while rb < bs - ss and num_cut < num_missing_batches:
                        batch_sizes.append(ss)
                        batch_beginnings.append(bb + rb)
                        batch_seq_lengths.append(
                            self.calc_batch_seq_length(input_ids[bb + rb : bb + rb + ss], length_is_multiple_of=8)
                        )
                        rb += ss
                        num_cut += 1
                    assert len(input_ids[bb + rb : bb + bs]) > 0
                    batch_sizes[original_batch_index] = bs - rb
                    batch_beginnings[original_batch_index] = bb + rb
                    batch_seq_lengths[original_batch_index] = self.calc_batch_seq_length(
                        input_ids[bb + rb : bb + bs], length_is_multiple_of=8
                    )
                original_batch_index -= 1
            # Keeping order of batches.
            batch_beginnings, batch_sizes, batch_seq_lengths = map(
                list, zip(*sorted(zip(batch_beginnings, batch_sizes, batch_seq_lengths), key=lambda x: x[0]))
            )
        assert len(batch_beginnings) % self.number_of_batches_is_multiple_of == 0
        assert len(batch_sizes) % self.number_of_batches_is_multiple_of == 0
        assert len(batch_seq_lengths) % self.number_of_batches_is_multiple_of == 0
        return batch_beginnings, batch_sizes, batch_seq_lengths

    def _mark_up_batches(self, input_ids: List[np.ndarray]) -> Tuple[List[int], List[int], List[int]]:
        """
        Computes indices of first samples in batch, batch sizes, seq lengths for batches. ``input_ids`` has to be
        sorted by number of tokens in ascending order.

        Batches are marked up with respect to following conditions:
            - total number of tokens in batch including paddings is less or equal to ``self.tokens_in_batch``
            - batch size is evenly divisible by 8 (except for the last batch)
            - seq length (elements of the third returned object) is evenly divisible by 8

        If ``self.batch_mark_up_progress_queue`` is not None, then the progress in mark up is reported via
        ``self.batch_mark_up_progress_queue``. Otherwise, ``tqdm`` instance is created in this function.

        Args:
            input_ids: a list of 1D int32 arrays. Elements of ``input_ids`` have to be sorted by length in ascending
                order

        Returns:
            batch_beginnings: a list of indices in ``input_ids`` of first samples of every batch
            batch_sizes: a list of numbers of samples in batches
            batch_seq_lengths: a list of sequence lengths after padding for every batch
        """
        batch_beginnings, batch_sizes, batch_seq_lengths = [], [], []
        current_max_length = 0
        start = 0
        if self.batch_mark_up_progress_queue is None:
            inp_iterator = tqdm(enumerate(input_ids), total=len(input_ids), desc="Batch mark up", unit="query")
        else:
            inp_iterator = enumerate(input_ids)
            progress_made = 0
        for i, inp in inp_iterator:
            current_max_length = max(current_max_length, ceil(len(inp) / 8) * 8)
            if current_max_length * (i + 1 - start) > self.tokens_in_batch:
                batch_size = (i - start) // 8 * 8
                if batch_size == 0:
                    if i > start:
                        batch_size = i - start
                        logging.warning(
                            f"Could not create batch with multiple of 8 size. Probably, there is a too long sequence "
                            f"in the dataset or parameter `tokens_in_batch` is too small. Current length of sequences "
                            f"in batch is {current_max_length}. Batch size will be reduced to {batch_size}. "
                            f"tokens_in_batch={self.tokens_in_batch}. The batch includes sequences from "
                            f"{start} to {i - 1}."
                        )
                    else:
                        logging.warning(
                            f"Input sequence number {i - 1} is too long. Could not fit it into batch with "
                            f"{self.tokens_in_batch} tokens. Sequence number {i - 1} will not be added to batches."
                        )
                        start = i
                        current_max_length = ceil(len(inp) / 8) * 8
                        continue
                seq_length = self.calc_batch_seq_length(input_ids[start : start + batch_size], length_is_multiple_of=8)
                batch_beginnings.append(start)
                batch_sizes.append(batch_size)
                batch_seq_lengths.append(seq_length)
                start += batch_size
                current_max_length = self.calc_batch_seq_length(input_ids[start : i + 1], length_is_multiple_of=8)
            if self.batch_mark_up_progress_queue is not None:
                progress_made += 1
                if progress_made >= BATCH_MARK_UP_PROGRESS_REPORT_PERIOD:
                    self.batch_mark_up_progress_queue.put(progress_made)
                    progress_made = 0
        if start < len(input_ids):
            seq_length = self.calc_batch_seq_length(input_ids[start:], length_is_multiple_of=8)
            batch_beginnings.append(start)
            batch_sizes.append(len(input_ids) - start)
            batch_seq_lengths.append(seq_length)
            if self.batch_mark_up_progress_queue is not None:
                self.batch_mark_up_progress_queue.put(progress_made)
        if len(batch_beginnings) % self.number_of_batches_is_multiple_of:
            batch_beginnings, batch_sizes, batch_seq_lengths = self._adjust_number_of_batches(
                input_ids, batch_beginnings, batch_sizes, batch_seq_lengths
            )
        assert sum(batch_sizes) == len(input_ids)
        for i in range(len(batch_beginnings) - 1):
            assert batch_beginnings[i] + batch_sizes[i] == batch_beginnings[i + 1]
            assert batch_seq_lengths[i] >= max(
                [len(inp) for inp in input_ids[batch_beginnings[i] : batch_beginnings[i] + batch_sizes[i]]]
            )
        return batch_beginnings, batch_sizes, batch_seq_lengths

    def _form_batches(
        self,
        input_ids: List[np.ndarray],
        subtokens_mask: List[np.ndarray],
        punct_labels: List[np.ndarray],
        capit_labels: List[np.ndarray],
        waveforms: Optional[List[np.ndarray]] = None,
        audio_lengths: Optional[List[np.ndarray]] = None,
        audio_filepaths: Optional[List[str]] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """

        Args:
            input_ids: a list of 1D int32 arrays which contain token ids of dataset source
            subtokens_mask: a list of 1D boolean arrays which elements are ``True`` if corresponding token is the
                first token in some word
            punct_labels: a list of 1D int32 arrays which contain encoded punctuation labels
            capit_labels: a list of 1D int32 arrays which contain encoded capitalization labels
            waveforms:  a list of 1D float arrays which contain raw waveforms of audios.
            audio_lengths: a list of 1D int32 arrays which contain length of corresponding audio from `waveforms`
            audio_filepaths: a list of strings which contain paths to audio

        Returns:
            a list of batches. Each batch is a dictionary with items:
              - ``'input_ids'``: a ``np.int32`` numpy array;
              - ``'subtokens_mask'``: a boolean numpy array;
              - ``'punct_labels'``: a ``np.int32`` numpy array;
              - ``'capit_labels'``: a ``np.int32`` numpy array.
            If ``self.add_masks_and_segment_ids_to_batch`` is ``True``, then a batch also contain items
              - ``'segment_ids'``: a ``np.int8`` numpy array;
              - ``'input_mask'``: a boolean numpy array;
              - ``'loss_mask'``: a boolean numpy array.
            If ``waveforms`` is not ``None``, then a batch also contain items
              - ``features``: a ``np.float`` numpy array.
              - ``features_length`` a ``np.int32`` numpy array.
            If ``audio_filepaths`` is not ``None``, then a natch also contain items
              - ``audio_filepaths`` a list of strings.

            The values of a batch dictionary are numpy arrays of identical shape.
        """
        batches = []
        dummy = [None] * len(input_ids)

        zipped = list(
            zip(
                input_ids,
                subtokens_mask,
                punct_labels,
                capit_labels,
                waveforms if waveforms else dummy,
                audio_lengths if audio_lengths else dummy,
                audio_filepaths if audio_filepaths else dummy,
            )
        )

        for item in zipped:
            batch = {
                "input_ids": item[0],
                "subtokens_mask": item[1],
                "punct_labels": item[2].astype(np.int64),
                "capit_labels": item[3].astype(np.int64),
            }
            if self.use_audio and self.preload_audios:
                batch['features'] = item[4].astype(np.float)
                batch['features_length'] = item[5]
            elif self.use_audio and not self.preload_audios:
                batch['audio_filepaths'] = item[6]
            batches.append(batch)
        return batches

    def _pack_into_batches(
        self,
        input_ids: List[np.ndarray],
        subtokens_mask: List[np.ndarray],
        punct_labels: List[np.ndarray],
        capit_labels: List[np.ndarray],
        waveforms: Optional[List[np.ndarray]] = None,
        audio_lengths: Optional[List[np.ndarray]] = None,
        audio_filepaths: Optional[List[str]] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Shuffle input sequences, sort them by number of tokens, pad, and pack into batches which satisfy following
        conditions:
            - total number of tokens in batch including paddings is less or equal to ``self.tokens_in_batch``
            - batch size is evenly divisible by 8 (except for the last batch)
            - seq length (elements of the third returned object) is evenly divisible by 8
        Created batches are shuffled before returning.

        If ``self.add_masks_and_segment_ids_to_batch`` is ``True``, then ``'segment_ids'``, ``'loss_mask'``, and
        ``'input_mask'`` are added to the batch.

        If ``self.batch_building_progress_queue`` is not ``None``, then padding progress is reported to
        ``self.batch_building_progress_queue``. Otherwise, a new ``tqdm`` instance is created in ``pack_into_batches``
        method.

        Args:
            input_ids: a list of 1D int32 arrays which contain token ids of dataset source
            subtokens_mask: a list of 1D boolean arrays which elements are ``True`` if corresponding token is the
                first token in some word
            punct_labels: a list of 1D int32 arrays which contain encoded punctuation labels
            capit_labels: a list of 1D int32 arrays which contain encoded capitalization labels
            waveforms:  a list of 1D float arrays which contain raw waveforms of audios.
            audio_lengths: a list of 1D int32 arrays which contain length of corresponding audio from `waveforms`
            audio_filepaths: a list of strings which contain paths to audio

        Returns:
            a list of batches. Each batch is a dictionary with items:
              - ``'input_ids'``: a ``np.int32`` numpy array;
              - ``'subtokens_mask'``: a boolean numpy array;
              - ``'punct_labels'``: a ``np.int32`` numpy array;
              - ``'capit_labels'``: a ``np.int32`` numpy array.
            If ``self.add_masks_and_segment_ids_to_batch`` is ``True``, then a batch also contain items
              - ``'segment_ids'``: a ``np.int8`` numpy array;
              - ``'input_mask'``: a boolean numpy array;
              - ``'loss_mask'``: a boolean numpy array.
            If ``waveforms`` is not ``None``, then a batch also contain items
              - ``features``: a ``np.float`` numpy array.
              - ``features_length`` a ``np.int32`` numpy array.
            If ``audio_filepaths`` is not ``None``, then a natch also contain items
              - ``audio_filepaths`` a list of strings.

            The values of a batch dictionary are numpy arrays of identical shape.
        """
        dummy = [None] * len(input_ids)
        zipped = list(
            zip(
                input_ids,
                subtokens_mask,
                punct_labels,
                capit_labels,
                waveforms if waveforms else dummy,
                audio_lengths if audio_lengths else dummy,
                audio_filepaths if audio_filepaths else dummy,
            )
        )
        self.batch_shuffling_random_state.shuffle(zipped)

        dim_sort = 4 if self.use_audio and self.preload_audios else 0

        input_ids, subtokens_mask, punct_labels, capit_labels, waveforms, audio_lengths, audio_filepaths = zip(
            *sorted(zipped, key=lambda x: x[dim_sort].shape[0])
        )
        batch_beginnings, batch_sizes, batch_seq_lengths = self._mark_up_batches(input_ids)
        batches = []
        if self.batch_building_progress_queue is None:
            inp_iterator = tqdm(
                zip(batch_beginnings, batch_sizes, batch_seq_lengths),
                total=len(batch_beginnings),
                desc="Batch building",
                unit="batch",
            )
        else:
            # In this case we report number of queries not number of batches
            inp_iterator = zip(batch_beginnings, batch_sizes, batch_seq_lengths)
            progress_made = 0
        for start, size, length in inp_iterator:
            batch_input_ids = pad(input_ids[start : start + size], length, self.tokenizer.pad_id)
            batch_subtokens_mask = pad(subtokens_mask[start : start + size], length, False)
            batch = {
                "input_ids": batch_input_ids,
                "subtokens_mask": batch_subtokens_mask,
                "punct_labels": pad(
                    punct_labels[start : start + size], length, self.punct_label_ids[self.pad_label]
                ).astype(np.int64),
                "capit_labels": pad(
                    capit_labels[start : start + size], length, self.capit_label_ids[self.pad_label]
                ).astype(np.int64),
            }
            if self.use_audio and self.preload_audios:
                batch['features'] = pad(
                    waveforms[start : start + size], max(audio_lengths[start : start + size]), 0.0
                ).astype(np.float)
                batch['features_length'] = audio_lengths[start : start + size]
            elif self.use_audio and not self.preload_audios:
                batch['audio_filepaths'] = audio_filepaths[start : start + size]

            if self.add_masks_and_segment_ids_to_batch:
                batch_segment_ids, batch_input_mask, batch_loss_mask = create_masks_and_segment_ids(
                    batch_input_ids,
                    batch_subtokens_mask,
                    self.tokenizer.pad_id,
                    self.tokenizer.cls_id,
                    self.tokenizer.sep_id,
                    self.ignore_start_end,
                    self.ignore_extra_tokens,
                )
                batch['segment_ids'] = batch_segment_ids
                batch['input_mask'] = batch_input_mask
                batch['loss_mask'] = batch_loss_mask
            batches.append(batch)
            if self.batch_building_progress_queue is not None:
                progress_made += size
                if progress_made >= BATCH_BUILDING_PROGRESS_REPORT_PERIOD:
                    self.batch_building_progress_queue.put(progress_made)
                    progress_made = 0
        if self.batch_building_progress_queue is not None:
            self.batch_building_progress_queue.put(progress_made)
        self.batch_shuffling_random_state.shuffle(batches)
        return batches

    def repack_batches_with_shuffle(self) -> None:
        """A function for proper shuffling of a dataset. Pytorch data loader shuffling will only permute batches."""
        if not self.use_bucketing:
            return
        logging.info("Shuffling training dataset")
        self.batches = self._pack_into_batches(
            self.input_ids,
            self.subtokens_mask,
            self.punct_labels,
            self.capit_labels,
            self.waveforms,
            self.waveforms_length,
            self.audio_filepaths,
        )

    def _calculate_and_save_label_frequencies(self, all_labels: List[np.ndarray], name: str) -> Dict[str, float]:
        """Calculates and saves labels frequencies in :attr:`label_info_save_dir`."""
        merged_labels = itertools.chain.from_iterable(all_labels)
        if self.verbose:
            logging.info('Three most popular labels')
        self.label_info_save_dir.mkdir(parents=True, exist_ok=True)
        _, label_frequencies, _ = get_label_stats(
            merged_labels, str(self.label_info_save_dir / f'label_count_{name}.tsv')
        )
        return label_frequencies

    def save_labels_and_get_file_paths(
        self, punct_labels_file_name: str, capit_labels_file_name: str
    ) -> Tuple[Path, Path]:
        """
        Saves label ids into files located in ``self.label_info_save_dir``. Saved label ids are usually used for
        ``.nemo`` checkpoint creation.

        The signatures of this method and the signature of the method
        :meth:`~nemo.collections.nlp.data.token_classification.BertPunctuationCapitalizationTarredDataset.save_labels_and_get_file_paths`
        must be identical.

        Args:
            punct_labels_file_name (:obj:`str`): a name of a punctuation labels file
            capit_labels_file_name (:obj:`str`): a name of a capitalization labels file

        Returns:
            :obj:`Tuple[pathlib.Path, pathlib.Path]`: a tuple containing:

                - :obj:`pathlib.Path`: a path to the saved punctuation labels file
                - :obj:`pathlib.Path`: a path to the saved capitalization labels file
        """
        nemo_dir = self.label_info_save_dir / LABEL_ID_DIR_FOR_NEMO_CHECKPOINT
        punct_labels_file = nemo_dir / punct_labels_file_name
        capit_labels_file = nemo_dir / capit_labels_file_name
        save_label_ids(self.punct_label_ids, punct_labels_file)
        save_label_ids(self.capit_label_ids, capit_labels_file)
        return punct_labels_file, capit_labels_file

    def __len__(self) -> int:
        return len(self.batches)

    def collate_fn(self, batches: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        If ``self.use_bucketing`` set to ``True`` returns zeroth batch from ``batches`` list passed for collating and casts ``'segment_ids'``, ``'punct_labels'``,
        ``'capit_labels'`` to types supported by
        :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_model.PunctuationCapitalizationModel`
        or :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_model.PunctuationCapitalizationLexicalAudioModel` if ``self.use_audio`` set to ``True``
        All output tensors have shape ``[Batch, Time]``.

        .. warning::
            A ``batch_size`` parameter of a PyTorch data loader and sampler has to be ``1`` if ``self.use_bucketing`` set to ``True``

        Args:
            batches (:obj:`List[Dict[str, np.ndarray]]`): a list containing 1 batch passed for collating

        Returns:
            :obj:`Dict[str, torch.Tensor]`: a batch dictionary with following items (for detailed description of batch
            items see method :meth:`__getitem__`):

              - ``'input_ids'`` (:obj:`torch.Tensor`): :obj:`torch.int32` tensor,
              - ``'subtokens_mask'`` (:obj:`torch.Tensor`): :obj:`torch.bool` tensor,
              - ``'punct_labels'`` (:obj:`torch.Tensor`): :obj:`torch.int64` tensor,
              - ``'capit_labels'`` (:obj:`torch.Tensor`): :obj:`torch.int64` tensor,
              - ``'segment_ids'`` (:obj:`torch.Tensor`): :obj:`torch.int32` tensor,
              - ``'input_mask'`` (:obj:`torch.Tensor`): :obj:`torch.bool` tensor,
              - ``'loss_mask'`` (:obj:`torch.Tensor`): :obj:`torch.bool` tensor.
              - ``'features'`` (:obj:`torch.Tensor`): :obj:`torch.float` tensor.
              - ``'features_length'`` (:obj:`torch.Tensor`): :obj:`torch.long` tensor.
        """
        if self.use_bucketing:
            batch = {k: torch.as_tensor(v) for k, v in batches[0].items() if k != 'audio_filepaths'}
            batch['segment_ids'] = batch['segment_ids'].int()
            batch['punct_labels'] = batch['punct_labels'].long()
            batch['capit_labels'] = batch['capit_labels'].long()
            if self.use_audio and self.preload_audios:
                batch['features'] = batch['features'].to(torch.float32)
            return batch
        else:
            for batch in batches:
                batch_segment_ids, batch_input_mask, batch_loss_mask = create_masks_and_segment_ids(
                    batch['input_ids'],
                    batch['subtokens_mask'],
                    self.tokenizer.pad_id,
                    self.tokenizer.cls_id,
                    self.tokenizer.sep_id,
                    self.ignore_start_end,
                    self.ignore_extra_tokens,
                )
                batch['segment_ids'] = torch.as_tensor(batch_segment_ids, dtype=torch.int)
                batch['input_mask'] = torch.as_tensor(batch_input_mask)
                batch['loss_mask'] = torch.as_tensor(batch_loss_mask)
                batch['input_ids'] = torch.as_tensor(batch['input_ids'], dtype=torch.int)
                batch['subtokens_mask'] = torch.as_tensor(batch['subtokens_mask'])
                batch['punct_labels'] = torch.as_tensor(batch['punct_labels'], dtype=torch.long)
                batch['capit_labels'] = torch.as_tensor(batch['capit_labels'], dtype=torch.long)
                if 'features' in batch:
                    batch['features'] = torch.as_tensor(batch['features'], dtype=torch.float)
                    batch['features_length'] = torch.as_tensor(batch['features_length'], dtype=torch.long)
                elif self.use_audio:
                    if ASR_AVAILABLE:
                        waveform = AudioSegment.from_file(batch['audio_filepaths'], target_sr=self.sample_rate)
                        batch['features'] = torch.as_tensor(waveform.samples, dtype=torch.float)
                        batch['features_length'] = torch.as_tensor(waveform.num_samples, dtype=torch.long)
                    else:
                        raise ModuleNotFoundError(
                            'Nemo ASR was not installed, see https://github.com/NVIDIA/NeMo#installation for installation instructions'
                        )

            segment_ids = pad_sequence([batch['segment_ids'] for batch in batches])
            input_mask = pad_sequence([batch['input_mask'] for batch in batches])
            loss_mask = pad_sequence([batch['loss_mask'] for batch in batches])
            input_ids = pad_sequence([batch['input_ids'] for batch in batches], padding_value=self.tokenizer.pad_id)
            subtokens_mask = pad_sequence([batch['subtokens_mask'] for batch in batches], padding_value=False)
            punct_labels = pad_sequence([batch['punct_labels'] for batch in batches], padding_value=0)
            capit_labels = pad_sequence([batch['capit_labels'] for batch in batches], padding_value=0)
            features = pad_sequence([batch['features'] for batch in batches], padding_value=0.0)
            features_length = torch.tensor([batch['features_length'] for batch in batches])
            return {
                'input_ids': input_ids.T,
                'subtokens_mask': subtokens_mask.T,
                'punct_labels': punct_labels.T,
                'capit_labels': capit_labels.T,
                'features': features.T,
                'features_length': features_length,
                'segment_ids': segment_ids.T,
                'input_mask': input_mask.T,
                'loss_mask': loss_mask.T,
            }

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Return a batch with index ``idx``. The values of a batch dictionary are numpy arrays of identical shapes
        ``[Batch, Time]``. Labels are identical for all tokens in a word. For example, if

          - word ``'Tokenization'`` is tokenized into tokens ``['token', 'ization']``,
          - it is followed by comma,

        then punctuation labels are ``[',', ',']`` and capitalization labels are ``['U', 'U']`` (``'U'`` is a label
        for words which start with upper case character).

        Args:
            idx: an index of returned batch

        Returns:
            :obj:`Dict[str, np.ndarray]`: a dictionary with items:

              - ``'input_ids'`` (:obj:`numpy.ndarray`): :obj:`numpy.int32` array containing encoded tokens,
              - ``'subtokens_mask'`` (:obj:`numpy.ndarray`): :obj:`bool` array which elements are ``True`` if they
                correspond to first token in a word,
              - ``'punct_labels'`` (:obj:`numpy.ndarray`): :obj:`numpy.int32` array containing encoded punctuation
                labels,
              - ``'capit_labels'`` (:obj:`numpy.ndarray`): :obj:`numpy.int32` array containing encoded capitalization
                labels.
              - ``'segment_ids'`` (:obj:`numpy.ndarray`): :obj:`numpy.int8` array filled with zeros (BERT token types
                in HuggingFace terminology) (if ``self.add_masks_and_segment_ids_to_batch`` is ``False``, then these
                items is missing),
              - ``'input_mask'`` (:obj:`numpy.ndarray`): :obj:`bool` array which elements are ``True`` if corresponding
                token is not a padding token (if ``self.add_masks_and_segment_ids_to_batch`` is ``False``, then these
                items is missing),
              - ``'loss_mask'`` (:obj:`numpy.ndarray`): :obj:`bool` array which elements are ``True`` if loss is
                computed for corresponding token. See more in description of constructor parameters
                ``ignore_start_end``, ``ignore_extra_tokens`` (if ``self.add_masks_and_segment_ids_to_batch`` is
                ``False``, then these items is missing).
              - ``'features'`` (:obj:`numpy.ndarray`) :obj:`np.float` array of waveforms of audio if ``self.preload_audio`` is set to ``True`` else empty.
              - ``'features_length'`` (:obj:`numpy.ndarray`) :obj:`np.long` array of number of samples per audio.
              - ``'audio_filepaths'`` (:obj:`List`) :obj:`str` contains paths of audio files if ``self.preload_audio`` set to ``False``
        """
        return self.batches[idx]
