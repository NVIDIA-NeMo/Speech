# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import inspect
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

import numpy as np
import torch
import webdataset as wds
from joblib import Parallel, delayed
from omegaconf import DictConfig
from torch.utils.data import IterableDataset

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import (
    LABEL_ID_DIR_FOR_NEMO_CHECKPOINT,
    BertPunctuationCapitalizationDataset,
    Progress,
    create_label_ids,
    create_masks_and_segment_ids,
    load_label_ids,
    raise_not_equal_labels_error,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.neural_types import AudioSignal, ChannelType, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging

NUMBER_RE = "(0|[1-9][0-9]*)"
TAR_FRAGMENT_TMPL_IN_PROGRESS = "fragment{fragment_idx}.{file_idx}.tar"
TAR_FRAGMENT_TMPL_FINISHED = "fragment{fragment_idx}.num_batches{num_batches}.{file_idx}.tar"
TAR_FRAGMENT_TMPL_TO_REPACK = "fragment{fragment_idx}.num_batches{num_batches}.{file_idx}.tar.to_repack"
TAR_FRAGMENT_PATTERN_IN_PROGRESS = re.compile(f"fragment{NUMBER_RE}.{NUMBER_RE}.tar$")
TAR_FRAGMENT_PATTERN_FINISHED = re.compile(f"fragment{NUMBER_RE}.num_batches{NUMBER_RE}.{NUMBER_RE}.tar$")
TAR_FRAGMENT_PATTERN_TO_REPACK = re.compile(f"fragment{NUMBER_RE}.num_batches{NUMBER_RE}.{NUMBER_RE}.tar.to_repack$")
NOT_ALLOWED_CHARACTERS_IN_FILE_NAME = re.compile(f"[^a-zA-Z0-9_.-]")
REPLACE_NOT_ALLOWED_CHARACTERS_IN_FILE_NAME = re.compile(f"-*[^a-zA-Z0-9_.-]+-*")

DATASET_PARAMETERS_TMPL = "{prefix}.tokens{tokens_in_batch}.max_seq_length{max_seq_length}.{tokenizer}"
TAR_FINAL_TMPL = ".batches{num_batches}.{ctr}.tar"

PROGRESS_REPORT_PERIOD = 10 ** 4

METADATA_PUNCT_LABEL_VOCAB_KEY = 'punct_label_vocab_file'
METADATA_CAPIT_LABEL_VOCAB_KEY = 'capit_label_vocab_file'
DEFAULT_PUNCT_LABEL_VOCAB_FILE_NAME = 'punct_label_vocab.csv'
DEFAULT_CAPIT_LABEL_VOCAB_FILE_NAME = 'capit_label_vocab.csv'


def count_lines_and_get_fragment_starting_positions(
    file_name: Path, lines_per_dataset_fragment: int
) -> Tuple[int, List[int]]:
    """
    Returns number of lines in a file and indices of fragment starting bytes.

    Args:
        file_name: a path to a text or label file
        lines_per_dataset_fragment: number of lines in a dataset fragment. The last fragment can contain fewer lines

    Returns:
        num_lines: number of lines in a file
        start_bytes: indices of fragment starting bytes
    """
    pos = [0]
    with file_name.open() as f:
        i = 0
        line = f.readline()
        while line:
            i += 1
            if i % lines_per_dataset_fragment == 0:
                pos.append(f.tell())
            line = f.readline()
    return i, pos[:-1] if i % lines_per_dataset_fragment == 0 else pos


def get_fragment_start_bytes(
    text_file: Path, labels_file: Path, lines_per_dataset_fragment: int, audio_file: Path = None
) -> Union[Tuple[Any, Any, Any, Any], Tuple[Any, Any, Any]]:
    """
    A function for calculating borders of dataset fragments. The function is used to split ``text_file`` and
    ``labels_file`` for processing them in parallel.

    Args:
        text_file: a path to a dataset source file
        labels_file: a path to a dataset label file
        lines_per_dataset_fragment: a number of lines in one fragment
        audio_file: a path to a dataset audio file if one needed

    Returns:
        num_lines: total number of elements in the dataset (number of lines in ``text_file``` and ``labels_file``)
        text_start_bytes: indices of the first bytes of fragments in ``text_file``
        label_start_bytes: indices of the first bytes of fragments in ``labels_file``
    """
    logging.info(
        f"Counting lines in files {text_file} and {labels_file} and creating segment borders. This may take "
        f"considerable time. 86GB, 1.27b lines file was processed in 7 minutes."
    )
    if audio_file:
        result = Parallel(n_jobs=3)(
            delayed(count_lines_and_get_fragment_starting_positions)(file_name, lines_per_dataset_fragment)
            for file_name in [text_file, labels_file, audio_file]
        )
        num_lines = result[0][0]
        if result[0][0] != result[1][0]:
            raise ValueError(
                f"Text file {text_file} and label file {labels_file} contain different number of lines. Number of lines "
                f"in text file: {result[0][0]}, number of lines in label file: {result[1][0]}."
            )
        text_start_bytes, label_start_bytes, manifest_start_bytes = result[0][1], result[1][1], result[2][1]
        assert len(text_start_bytes) == len(label_start_bytes) == len(manifest_start_bytes)
        return num_lines, text_start_bytes, label_start_bytes, manifest_start_bytes
    else:
        result = Parallel(n_jobs=2)(
            delayed(count_lines_and_get_fragment_starting_positions)(file_name, lines_per_dataset_fragment)
            for file_name in [text_file, labels_file]
        )
        num_lines = result[0][0]
        if result[0][0] != result[1][0]:
            raise ValueError(
                f"Text file {text_file} and label file {labels_file} contain different number of lines. Number of lines "
                f"in text file: {result[0][0]}, number of lines in label file: {result[1][0]}."
            )
        text_start_bytes, label_start_bytes = result[0][1], result[1][1]
        assert len(text_start_bytes) == len(label_start_bytes)
        return num_lines, text_start_bytes, label_start_bytes


def process_fragment(
    text_file: Path,
    labels_file: Path,
    output_dir: Path,
    text_start_pos: int,
    label_start_pos: int,
    lines_per_dataset_fragment: int,
    max_seq_length: int,
    tokens_in_batch: int,
    num_batches_per_tarfile: int,
    tokenizer_name: str,
    tokenizer_model: Optional[Path],
    vocab_file: Optional[Path],
    merges_file: Optional[Path],
    special_tokens: Dict[str, str],
    use_fast_tokenizer: Optional[bool],
    pad_label: str,
    punct_label_ids: Dict[str, int],
    capit_label_ids: Dict[str, int],
    fragment_idx: int,
    tokenization_progress_queue: mp.Queue,
    batch_mark_up_progress_queue: mp.Queue,
    batch_building_progress_queue: mp.Queue,
    writing_to_tar_progress_queue: mp.Queue,
    audio_file: Path = None,
    sample_rate: int = None,
    audio_file_start_pos: int = None,
    use_audio: bool = False,
) -> None:
    tokenizer = get_tokenizer(
        tokenizer_name,
        tokenizer_model=None if tokenizer_model is None else str(tokenizer_model),
        vocab_file=None if vocab_file is None else str(vocab_file),
        merges_file=None if merges_file is None else str(merges_file),
        special_tokens=special_tokens,
        use_fast=use_fast_tokenizer,
    )
    tmp_text: Optional[str] = None
    tmp_labels: Optional[str] = None
    tmp_audio: Optional[str] = None
    try:
        otfd, tmp_text = tempfile.mkstemp(suffix='.txt', prefix=f'text_{fragment_idx}_', dir=output_dir, text=True)
        olfd, tmp_labels = tempfile.mkstemp(suffix='.txt', prefix=f'labels_{fragment_idx}_', dir=output_dir, text=True)
        if use_audio:
            oafd, tmp_audio = tempfile.mkstemp(
                suffix='.txt', prefix=f'audio_{fragment_idx}_', dir=output_dir, text=True
            )
        with text_file.open() as tf, labels_file.open() as lf, os.fdopen(otfd, 'w') as otf, os.fdopen(
            olfd, 'w'
        ) as olf:  # handle audio manifest
            if use_audio:
                mf = audio_file.open()
                mf.seek(audio_file_start_pos)
                oaf = os.fdopen(oafd, 'w')
            tf.seek(text_start_pos)
            lf.seek(label_start_pos)
            for _ in range(lines_per_dataset_fragment):
                text_line = tf.readline()
                if not text_line:
                    break
                otf.write(text_line)
                olf.write(lf.readline())
                if use_audio:
                    oaf.write(mf.readline())
        if use_audio:
            mf.close()
            oaf.close()
        dataset = BertPunctuationCapitalizationDataset(
            tmp_text,
            tmp_labels,
            max_seq_length,
            tokenizer,
            tokens_in_batch=tokens_in_batch,
            pad_label=pad_label,
            punct_label_ids=punct_label_ids,
            capit_label_ids=capit_label_ids,
            n_jobs=0,
            use_cache=False,
            add_masks_and_segment_ids_to_batch=False,
            verbose=False,
            tokenization_progress_queue=tokenization_progress_queue,
            batch_mark_up_progress_queue=batch_mark_up_progress_queue,
            batch_building_progress_queue=batch_building_progress_queue,
            audio_file=tmp_audio,
            sample_rate=sample_rate,
            use_audio=use_audio,
            use_bucketing=True,
            preload_audios=use_audio,
        )
    finally:
        if tmp_text is not None and os.path.exists(tmp_text):
            os.remove(tmp_text)
        if tmp_labels is not None and os.path.exists(tmp_labels):
            os.remove(tmp_labels)
        if tmp_audio is not None and os.path.exists(tmp_audio):
            os.remove(tmp_audio)
    dataset.features_pkl.unlink()
    tar_ctr = 0
    current_file_name = output_dir / TAR_FRAGMENT_TMPL_IN_PROGRESS.format(fragment_idx=fragment_idx, file_idx=tar_ctr)
    current_num_batches = 0
    sink = wds.TarWriter(str(current_file_name))
    progress_made = 0
    for batch_i, batch in enumerate(dataset):
        sink.write({"__key__": f"fragment-{fragment_idx}-batch-{batch_i}", "batch.pyd": batch})
        current_num_batches += 1
        progress_made += len(batch['input_ids'])
        if current_num_batches % num_batches_per_tarfile == 0:
            sink.close()
            current_file_name.rename(
                output_dir
                / TAR_FRAGMENT_TMPL_FINISHED.format(
                    fragment_idx=fragment_idx, num_batches=current_num_batches, file_idx=tar_ctr
                )
            )
            writing_to_tar_progress_queue.put(progress_made)
            progress_made = 0
            tar_ctr += 1
            current_file_name = output_dir / TAR_FRAGMENT_TMPL_IN_PROGRESS.format(
                fragment_idx=fragment_idx, file_idx=tar_ctr
            )
            current_num_batches = 0
            sink = wds.TarWriter(str(current_file_name))
    sink.close()
    writing_to_tar_progress_queue.put(progress_made)
    if progress_made > 0:
        new_file_name = output_dir / TAR_FRAGMENT_TMPL_TO_REPACK.format(
            fragment_idx=fragment_idx, num_batches=current_num_batches, file_idx=tar_ctr
        )
        current_file_name.rename(new_file_name)
    else:
        current_file_name.unlink()
    if fragment_idx == 0:
        punct_label_ids_file, capit_label_ids_file = dataset.save_labels_and_get_file_paths(
            DEFAULT_PUNCT_LABEL_VOCAB_FILE_NAME, DEFAULT_CAPIT_LABEL_VOCAB_FILE_NAME
        )
        punct_label_ids_file.rename(output_dir / DEFAULT_PUNCT_LABEL_VOCAB_FILE_NAME)
        capit_label_ids_file.rename(output_dir / DEFAULT_CAPIT_LABEL_VOCAB_FILE_NAME)
        shutil.rmtree(punct_label_ids_file.parent)


def remove_unexpected_files_and_dirs(output_dir: Path, output_file_tmpl: str, metadata_file_name: Path) -> None:
    """
    This function removes all files with names which may be used in the dataset creation.

    Args:
        output_dir: a path to directory where removal is performed
        output_file_tmpl: a format string for a name of final tar file. Must include fields ``ctr`` for number of the
            file and ``num_batches`` for number of batches in the file.
        metadata_file_name: a metadata file name
    """
    if not output_dir.is_dir():
        return
    tar_final_pattern = re.compile(output_file_tmpl.format(ctr=NUMBER_RE, num_batches=NUMBER_RE))
    unexpected_tar_files = [
        path
        for path in output_dir.iterdir()
        if any(
            [
                p.match(path.name) is not None
                for p in [
                    TAR_FRAGMENT_PATTERN_IN_PROGRESS,
                    TAR_FRAGMENT_PATTERN_FINISHED,
                    TAR_FRAGMENT_PATTERN_TO_REPACK,
                    tar_final_pattern,
                ]
            ]
        )
    ]
    if unexpected_tar_files:
        logging.warning(
            f"Found {len(unexpected_tar_files)} unexpected tar files in the output directory {output_dir}. "
            f"All of them are going to be removed. The files match one of 3 patterns: "
            f"'{TAR_FRAGMENT_PATTERN_IN_PROGRESS.pattern}', '{TAR_FRAGMENT_PATTERN_FINISHED.pattern}', "
            f"'{tar_final_pattern.pattern}'. The first unexpected files: "
            f"{', '.join([str(f) for f in unexpected_tar_files[:3]])}."
        )
        for fn in unexpected_tar_files:
            fn.unlink()
    if metadata_file_name.exists():
        logging.warning(f"Found metadata file {metadata_file_name}. It is going to be removed.")
        metadata_file_name.unlink()
    punct_label_ids = output_dir / DEFAULT_PUNCT_LABEL_VOCAB_FILE_NAME
    capit_label_ids = output_dir / DEFAULT_CAPIT_LABEL_VOCAB_FILE_NAME
    if punct_label_ids.exists():
        logging.warning(f"Found unexpected punctuation label file {punct_label_ids}. It is going to be removed.")
        punct_label_ids.unlink()
    if capit_label_ids.exists():
        logging.warning(f"Found unexpected capitalization label file {capit_label_ids}. It is going to be removed.")
        capit_label_ids.unlink()


def collect_unique_labels_from_fragment(
    labels_file: Path, start_pos: int, lines_per_dataset_fragment: int, progress_queue: mp.Queue, fragment_idx: int
) -> Tuple[Set[str], Set[str]]:
    """
    Returns a set of unique punctuation labels and a set of unique capitalization labels.

    Args:
        labels_file: a path to a file with labels
        start_pos: an index of the first byte of a fragment in ``labels_file``
        lines_per_dataset_fragment: number of lines in dataset fragment. In the last fragment there can be less lines.
        progress_queue: a queue for reporting number of processed lines
        fragment_idx: a processed fragment index

    Returns:
        unique_punct: a set of unique punctuation labels
        unique_capit: a set of unique capitalization labels
    """
    unique_punct, unique_capit = set(), set()
    with labels_file.open() as f:
        f.seek(start_pos)
        progress_report = 0
        for i in range(lines_per_dataset_fragment):
            line = f.readline()
            if not line:
                break
            pairs = line.split()
            if not all([len(p) == 2 for p in pairs]):
                broken_pairs = [i for i, p in enumerate(pairs) if len(p) != 2]
                raise ValueError(
                    f"Found broken labels line in number {fragment_idx * lines_per_dataset_fragment + i} in file "
                    f"{labels_file}. Indices of broken pairs of labels: {broken_pairs}"
                )
            punct, capit = zip(*pairs)
            unique_punct.update(punct)
            unique_capit.update(capit)
            progress_report += 1
            if progress_report >= PROGRESS_REPORT_PERIOD:
                progress_queue.put(progress_report)
                progress_report = 0
        progress_queue.put(progress_report)
    return unique_punct, unique_capit


def create_label_dictionaries(
    labels_file: Path,
    text_start_bytes: List[int],
    num_lines: int,
    lines_per_dataset_fragment: int,
    pad_label: str,
    n_jobs: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Creates punctuation and capitalization label ids dictionaries based on labels present in ``labels_file``.

    Args:
        labels_file: a path to file with labels
        text_start_bytes: indices of first bytes of fragments in ``labels_file``
        num_lines: total number of lines in ``labels_file``
        lines_per_dataset_fragment: number of lines in dataset fragments. The last fragment can have fewer lines
        pad_label: a label used for padding and for absence of punctuation and capitalization
        n_jobs: a number of fragments processed in parallel

    Returns:
        punct_label_ids: a dictionary with punctuation label ids
        capit_label_ids: a dictionary with capitalization label ids
    """
    with Progress(num_lines, "Creating label dictionary", "line") as progress_queues:
        result = Parallel(n_jobs=min(n_jobs, len(text_start_bytes)))(
            delayed(collect_unique_labels_from_fragment)(
                labels_file, start_pos, lines_per_dataset_fragment, *progress_queues, fragment_idx
            )
            for fragment_idx, start_pos in enumerate(text_start_bytes)
        )
    unique_punct, unique_capit = zip(*result)
    unique_punct = set().union(*unique_punct)
    unique_capit = set().union(*unique_capit)
    return create_label_ids(unique_punct, pad_label), create_label_ids(unique_capit, pad_label)


def check_label_ids(pad_label: str, punct_label_ids: Dict[str, int], capit_label_ids: Dict[str, int]) -> None:
    """
    A function for checking that pad label has zeroth id in ``punct_label_dis`` and ``capit_label_ids`` dictionaries.
    Args:
        pad_label: a pad label
        punct_label_ids: a dictionary with punctuation label ids
        capit_label_ids: a dictionary with capitalization label ids
    """
    msg = "Parameter `pad_label` has to have id 0 in dictionary `{param_name}` whereas it has id {id_}." + (
        '' if len(pad_label) > 10 else f" pad_label='{pad_label}'"
    )
    if punct_label_ids is not None:
        if punct_label_ids[pad_label] != 0:
            raise ValueError(msg.format(param_name='punct_label_ids', id_=punct_label_ids[pad_label]))
    if capit_label_ids is not None:
        if capit_label_ids[pad_label] != 0:
            raise ValueError(msg.format(param_name='capit_label_ids', id_=capit_label_ids[pad_label]))


def process_error(msg: str, error_class_or_function: Union[Type[Exception], Callable[[str], Any]]) -> None:
    if inspect.isclass(error_class_or_function) and issubclass(error_class_or_function, Exception):
        raise error_class_or_function(msg)
    if callable(error_class_or_function):
        error_class_or_function(msg)
    raise ValueError(
        f"Parameter `error_class_or_function` has to be a subclass of `Exception` or a function."
        f"Given {type(error_class_or_function)}"
    )


def check_labels_for_being_unique_before_building_label_ids(
    pad_label: str,
    other_labels: List[str],
    pad_label_name: str,
    other_labels_name: str,
    error_class_or_function: Union[Type[Exception], Callable[[str], Any]],
) -> None:
    """
    A function for checking that that all labels are unique.

    Args:
        pad_label: a pad label
        other_labels: a list of labels except for the pad label
        pad_label_name: a name of the pad label used in error message
        other_labels_name: a name of other labels used in error message
        error_class_or_function: a class of an exception which is raised if there is a problem with labels.
            Alternatively it can be a function for handling exceptions, for example ``argparse.ArgumentParser.error``.
            Such a function has to take one argument -- error message.
    """
    for i, lbl in enumerate(other_labels):
        if lbl == pad_label:
            msg = f"Label number {i} in parameter `{other_labels_name}` is equal to `{pad_label_name}`."
            process_error(msg, error_class_or_function)
    for i in range(len(other_labels) - 1):
        for lbl in other_labels[i + 1 :]:
            if lbl == other_labels[i]:
                msg = f"Label number {i} occurs at least 2 times in parameter `{other_labels_name}`."
                process_error(msg, error_class_or_function)


def build_label_ids_from_list_of_labels(pad_label: str, other_labels: List[str]) -> Dict[str, int]:
    """
    Builds label ids dictionary from pad label and list of other labels. Used for parsing command line arguments.
    Args:
        pad_label: a pad label
        other_labels: list of labels except for the pad label

    Returns:
        a dictionary with label ids
    """
    check_labels_for_being_unique_before_building_label_ids(
        pad_label, other_labels, 'pad_label', 'other_labels', ValueError
    )
    ids = {pad_label: 0}
    for lbl in other_labels:
        ids[lbl] = len(ids)
    return ids


def get_label_dictionaries(
    labels_file: Path,
    start_bytes: List[int],
    num_lines: int,
    lines_per_dataset_fragment: int,
    pad_label: str,
    punct_label_ids: Optional[Dict[str, int]],
    capit_label_ids: Optional[Dict[str, int]],
    punct_label_vocab_file: Optional[Path],
    capit_label_vocab_file: Optional[Path],
    n_jobs: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Return label ids if the label ids are present in parameters ``punct_label_ids``, ``capit_label_ids``,
    ``punct_label_vocab_file``, ``capit_label_vocab_file``. Otherwise, label ids are created using ``labels_file``.

    Args:
        labels_file: a path to file with labels. Labels have to be given in the format described in
            https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#nemo-data-format
        start_bytes: a list of positions in ``labels_file`` at which fragments start. Parameter ``start_bytes`` is used
            for creating labels for several fragments in parallel
        num_lines: total number of lines in ``labels_file``. Parameter ``num_lines`` is used for showing progress of
            label ids collection
        lines_per_dataset_fragment: number of lines in a dataset fragment
        pad_label: a label used for padding and also neutral label showing there is no punctuation and capitalization.
            Label ``pad_label`` has to have id ``0`` in parameters ``punct_label_ids``, ``capit_label_ids``,
            ``punct_label_vocab_file``, ``capit_label_vocab_file`` if these parameters are provided.
        punct_label_ids: a dictionary with punctuation label ids. Pad label has to have id ``0``. No more than 1 of
            parameters ``punct_label_ids`` and ``punct_label_vocab_file`` can be provided.
        capit_label_ids: a dictionary with capitalization label ids. Pad label has to have id ``0``. No more than 1 of
            parameters ``capit_label_ids`` and ``capit_label_vocab_file`` can be provided.
        punct_label_vocab_file: a text file with punctuation labels. Every line in the file contains 1 label. Pad label
            has to be in the first line. No more than 1 of parameters ``punct_label_ids`` and
            ``punct_label_vocab_file`` can be provided.
        capit_label_vocab_file: a text file with capitalization labels. Every line in the file contains 1 label. Pad
            label has to be in the first line. No more than 1 of parameters ``capit_label_ids`` and
            ``capit_label_vocab_file`` can be provided.
        n_jobs: a number of fragments processed in parallel

    Returns:
        punct_label_ids: a dictionary with punctuation label ids
        capit_label_ids: a dictionary with capitalization label ids
    """
    if punct_label_ids is not None and punct_label_vocab_file is not None:
        raise ValueError("You can provide at most one of parameters `punct_label_ids` and `punct_label_vocab_file`.")
    if capit_label_ids is not None and capit_label_vocab_file is not None:
        raise ValueError("You can provide at most one of parameters `capit_label_ids` and `capit_label_vocab_file`.")
    if punct_label_ids is None and punct_label_vocab_file is not None:
        punct_label_ids = load_label_ids(punct_label_vocab_file)
    if capit_label_ids is None and capit_label_vocab_file is not None:
        capit_label_ids = load_label_ids(capit_label_vocab_file)
    check_label_ids(pad_label, punct_label_ids, capit_label_ids)
    if punct_label_ids is None or capit_label_ids is None:
        _punct_label_ids, _capit_label_ids = create_label_dictionaries(
            labels_file, start_bytes, num_lines, lines_per_dataset_fragment, pad_label, n_jobs
        )
        if punct_label_ids is None:
            punct_label_ids = _punct_label_ids
        if capit_label_ids is None:
            capit_label_ids = _capit_label_ids
    return punct_label_ids, capit_label_ids


def decode_pyd(key: str, value: bytes) -> Any:
    """
    Used for decoding batch loaded by ``webdataset`` from tar files.
    Args:
        key: name of a batch
        value: pickled batch

    Returns:
        decoded batch
    """
    return pickle.loads(value)


def repack_tar_files_with_not_enough_batches(output_dir: Path, num_batches_per_tarfile: int) -> None:
    f"""
    It is possible that number of batches in a fragment is not evenly divisible by ``num_batches_per_tarfile``.
    In such a case excess batches are put in a tar file which matches a pattern
    ``fragment(0|[1-9][0-9]*).num_batches(0|[1-9][0-9]*).(0|[1-9][0-9]*).tar.to_repack``. Such files are repacked by
    ``repack_tar_files_with_not_enough_batches`` function into tar files with correct ``num_batches_per_tarfile``
    batches each. If there is no enough batches in repacked files, then up to ``num_batches_per_tarfile - 1``
    remaining batches may be discarded.

    Args:
        output_dir: a path to the output directory which contains files to repack and where new files are saved
        num_batches_per_tarfile: a number of batches in 1 tar file. If number of batches in files matching a pattern
            ``fragment(0|[1-9][0-9]*).num_batches(0|[1-9][0-9]*).(0|[1-9][0-9]*).tar.to_repack`` is not evenly
            divisible by ``num_batches_per_tarfile`` excess batches are discarded.
    """
    files_to_repack_with_matches = [
        (path, TAR_FRAGMENT_PATTERN_TO_REPACK.match(path.name))
        for path in output_dir.iterdir()
        if TAR_FRAGMENT_PATTERN_TO_REPACK.match(path.name) is not None
    ]
    files_to_repack_with_matches = sorted(files_to_repack_with_matches, key=lambda x: int(x[1].group(3)))
    logging.info(f"Found {len(files_to_repack_with_matches)} files for repacking.")
    files_to_repack_with_matches = deque(files_to_repack_with_matches)
    total_batches_in_repacked_files = 0
    initial_number_of_files_to_repack = len(files_to_repack_with_matches)
    pop_file_ds = None
    new_file_sink = None
    new_file_num_batches = 0
    while files_to_repack_with_matches:
        assert pop_file_ds is None or new_file_sink is None
        if new_file_sink is None:
            # `append_file` is a file which content will serve as a start for new tar file. `append_file` content is
            # copied into a `new_file` and then content of other files needing repacking is appended to content of
            # `new_file`.
            append_file, match = files_to_repack_with_matches.popleft()
            new_file = append_file.parent / TAR_FRAGMENT_TMPL_FINISHED.format(
                fragment_idx=match.group(1), num_batches=num_batches_per_tarfile, file_idx=match.group(3)
            )
            new_file_sink = wds.TarWriter(str(new_file))
            append_ds_to_rewrite = (
                wds.WebDataset(urls=[str(append_file)], nodesplitter=None)
                .decode(wds.handle_extension('.pyd', decode_pyd))
                .to_tuple('__key__', 'batch.pyd')
            )
            for key, batch in iter(append_ds_to_rewrite):
                new_file_sink.write({"__key__": key, "batch.pyd": batch})
                new_file_num_batches += 1
                total_batches_in_repacked_files += 1
            assert total_batches_in_repacked_files < initial_number_of_files_to_repack * num_batches_per_tarfile
            assert new_file_num_batches == int(match.group(2)), (
                f"Number of batches {new_file_num_batches} in {append_file} is different from number of batches "
                f"{match.group(2)} in repacked tar file with name {append_file}."
            )
            append_file.unlink()
        if files_to_repack_with_matches and pop_file_ds is None:
            pop_file, _ = files_to_repack_with_matches.pop()
            pop_file_ds = (
                wds.WebDataset(urls=[str(pop_file)], nodesplitter=None)
                .decode(wds.handle_extension('.pyd', decode_pyd))
                .to_tuple('__key__', 'batch.pyd')
            )
            pop_file_ds = iter(pop_file_ds)
        if pop_file_ds is not None and new_file_sink is not None:
            while new_file_num_batches < num_batches_per_tarfile:
                try:
                    key, batch = next(pop_file_ds)
                except StopIteration:
                    pop_file_ds = None
                    pop_file.unlink()
                    break
                new_file_sink.write({"__key__": key, "batch.pyd": batch})
                total_batches_in_repacked_files += 1
                assert total_batches_in_repacked_files < initial_number_of_files_to_repack * num_batches_per_tarfile
                new_file_num_batches += 1
            if new_file_num_batches >= num_batches_per_tarfile:
                assert new_file_num_batches == num_batches_per_tarfile
                new_file_sink.close()
                new_file_sink = None
                new_file_num_batches = 0
    if new_file_sink is not None:
        new_file_sink.close()
        new_file.unlink()
        logging.info(f"Discarded {new_file_num_batches} batches.")
    if pop_file_ds is not None:
        pop_file.unlink()
    logging.info(f"Repacked {total_batches_in_repacked_files} batches from short tar files")


def create_metadata_file(
    output_dir: Path, output_file_tmpl: str, metadata_file_name: Path, num_batches_per_tarfile: int
) -> None:
    """
    Rename tar files according to template ``output_file_tmpl`` and save metadata file.
    Args:
        output_dir: a path to directory which contains initial tar files and where renamed tar files are saved
        output_file_tmpl: a template of a new tar file name
        metadata_file_name: a path to a file into which metadata is going to be saved
        num_batches_per_tarfile: a required number of batches in tar files. Used for checking that present tar files
            have correct number of batches
    """
    metadata = {"num_batches": 0, "tar_files": []}
    for i, fn in enumerate([fn for fn in output_dir.iterdir() if TAR_FRAGMENT_PATTERN_FINISHED.match(fn.name)]):
        nb = int(TAR_FRAGMENT_PATTERN_FINISHED.match(fn.name).group(2))
        assert nb == num_batches_per_tarfile
        new_name = output_dir / output_file_tmpl.format(ctr=i, num_batches=nb)
        fn.rename(new_name)
        metadata['tar_files'].append(new_name.name)
        metadata["num_batches"] += nb
    metadata[METADATA_PUNCT_LABEL_VOCAB_KEY] = DEFAULT_PUNCT_LABEL_VOCAB_FILE_NAME
    metadata[METADATA_CAPIT_LABEL_VOCAB_KEY] = DEFAULT_CAPIT_LABEL_VOCAB_FILE_NAME
    logging.info(f"{metadata['num_batches']} batches are in tarred dataset with metadata file {metadata_file_name}")
    with metadata_file_name.open('w') as f:
        json.dump(metadata, f, indent=2)


def check_tar_file_prefix(
    tar_file_prefix: str, error_class_or_function: Union[Type[Exception], Callable[[str], Any]], var_name: str
) -> None:
    not_allowed_characters_in_prefix = NOT_ALLOWED_CHARACTERS_IN_FILE_NAME.findall(tar_file_prefix)
    if not_allowed_characters_in_prefix:
        not_allowed_characters_in_prefix = set(not_allowed_characters_in_prefix)
        msg = (
            f"Found {len(not_allowed_characters_in_prefix)} not allowed characters in `{var_name}`. Only 'A-Z', "
            f"'a-z', '0-9', '_', '-', '.' characters are allowed. Examples of not allowed characters: "
            f"{list(not_allowed_characters_in_prefix)[:10]}. `{var_name}`[:30]={repr(tar_file_prefix)[:30]}."
        )
        process_error(msg, error_class_or_function)


def create_tarred_dataset(
    text_file: Union[os.PathLike, str],
    labels_file: Union[os.PathLike, str],
    output_dir: Union[os.PathLike, str],
    max_seq_length: int,
    tokens_in_batch: int,
    lines_per_dataset_fragment: int,
    num_batches_per_tarfile: int,
    tokenizer_name: str,
    tokenizer_model: Optional[Union[os.PathLike, str]] = None,
    vocab_file: Optional[Union[os.PathLike, str]] = None,
    merges_file: Optional[Union[os.PathLike, str]] = None,
    special_tokens: Optional[Dict[str, str]] = None,
    use_fast_tokenizer: Optional[bool] = False,
    pad_label: str = 'O',
    punct_label_ids: Optional[Dict[str, int]] = None,
    capit_label_ids: Optional[Dict[str, int]] = None,
    punct_label_vocab_file: Optional[Union[os.PathLike, str]] = None,
    capit_label_vocab_file: Optional[Union[os.PathLike, str]] = None,
    tar_file_prefix: Optional[str] = 'punctuation_capitalization',
    n_jobs: Optional[int] = None,
    audio_file: Optional[Path] = None,
    use_audio: Optional[bool] = False,
    sample_rate: Optional[int] = 16000,
) -> None:
    """
    Creates tarred dataset from ``text_file`` and ``labels_file``. A tarred dataset allows to train on large amounts of
    data without storing it all into memory simultaneously. You may use these function directly or try script
    `examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py
    <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py>`_.

    Tarred dataset is a directory which contains metadata file, tar files with batches,
    ``punct_label_vocab.csv`` and ``capit_label_vocab.csv`` files.

    Metadata file is a JSON file with 4 items: ``'num_batches'``, ``'tar_files'``, ``'punct_label_vocab_file'``,
    ``'capit_label_vocab_file'``. The item ``'num_batches'`` (``int``) is a total number of batches in tarred dataset.
    ``'tar_files'`` is a list of paths to tar files relative to directory containing the metadata file. The items
    ``'punct_label_vocab_file'`` and ``'capit_label_vocab_file'`` are correspondingly paths to punctuation and
    capitalization label vocabulary files. These paths are relative to directory containing the metadata file.

    Every tar file contains objects written using ``webdataset.TarWriter``. Each object is a dictionary with two items:
    ``'__key__'`` and ``'batch.pyd'``. ``'__key__'`` is a name of a batch and ``'batch.pyd'`` is a pickled dictionary
    which contains ``'input_ids'``, ``'subtokens_mask'``, ``'punct_labels'``, ``'capit_labels'``. ``'input_ids'`` is an
    array containing ids of source tokens, ``'subtokens_mask'`` is a boolean array showing first tokens in words,
    ``'punct_labels'`` and ``'capit_labels'`` are arrays with ids of labels.

    Metadata file should be passed to constructor of :class:`BertPunctuationCapitalizationTarredDataset` and the
    instance of the class will handle iteration and constructing masks and token types for BERT model.

    Args:
        text_file (:obj:`Union[os.PathLike, str]`): a path to a file with dataset source. Dataset source is lowercase
            text without punctuation. Number of lines in ``text_file`` has to be equal to the number of lines in
            ``labels_file``.
        labels_file (:obj:`Union[os.PathLike, str]`): a path to a file with labels. Labels are given in the format
            described in :ref:`NeMo Data Format<nemo-data-format-label>`.
        output_dir (:obj:`Union[os.PathLike, str]`): a path to a directory where metadata file, tar files and
            ``'punct_label_ids.csv'`` and ``'capit_label_ids.csv'`` files are saved.
        max_seq_length (:obj:`int`): Maximum number of subtokens in an input sequence. A source sequence which contains
            too many subtokens is clipped to ``max_seq_length - 2`` subtokens and then [CLS] token is prepended to the
            clipped sequence and [SEP] token is appended to the clipped sequence. The clipping is performed via removal
            of subtokens in the end of a source sequence.
        tokens_in_batch (:obj:`int`): maximum number of tokens in a batch including [CLS], [SEP], [UNK], and [PAD]
            tokens. Before packing into batches source sequences are sorted by number of tokens in order to reduce
            number of pad tokens. So the number of samples in a batch may vary.
        lines_per_dataset_fragment (:obj:`int`): a number of lines processed by one worker during creation of tarred
            dataset. A worker tokenizes ``lines_per_dataset_fragment`` lines and keeps in RAM tokenized text labels
            before packing them into batches. Reducing ``lines_per_dataset_fragment`` leads to reducing of the amount
            of memory used by this function.
        num_batches_per_tarfile (:obj:`int`): a number of batches saved in a tar file. If you increase
            ``num_batches_per_tarfile``, then there will be less tar files in the dataset. There cannot be less than
            ``num_batches_per_tarfile`` batches in a tar file, and all excess batches are removed. Maximum number of
            discarded batches is ``num_batches_per_tarfile - 1``.
        tokenizer_name (:obj:`str`): a name of the tokenizer used for tokenization of source sequences. Possible
            options are ``'sentencepiece'``, ``'word'``, ``'char'``, HuggingFace tokenizers. For more options see
            function ``nemo.collections.nlp.modules.common.get_tokenizer``. The tokenizer must have properties
            ``cls_id``, ``pad_id``, ``sep_id``, ``unk_id``.
        tokenizer_model (:obj:`Union[os.PathLike, str]`, `optional`): a path to a tokenizer model required for
            ``'sentencepiece'`` tokenizer.
        vocab_file (:obj:`Union[os.PathLike, str]`, `optional`): a path to a vocabulary file which can be used in
            ``'word'``, ``'char'``, and HuggingFace tokenizers.
        merges_file (:obj:`Union[os.PathLike, str]`, `optional`): a path to merges file which can be used in
            HuggingFace tokenizers.
        special_tokens (:obj:`Dict[str, str]`, `optional`): a dictionary with special tokens passed to constructors of
            ``'char'``, ``'word'``, ``'sentencepiece'``, and various HuggingFace tokenizers.
        use_fast_tokenizer (:obj:`bool`, `optional`, defaults to :obj:`False`): whether to use fast HuggingFace
            tokenizer.
        pad_label (:obj:`str`, `optional`, defaults to :obj:`'O'`): a pad label both for punctuation and
            capitalization. This label is also a neutral label (used for marking words which do not need punctuation
            and capitalization).
        punct_label_ids (:obj:`Dict[str, int]`, `optional`): a dictionary which keys are punctuation labels and values
            are label ids. The pad label ``pad_label`` has to have id ``0``. You can provide at most one of parameters
            ``punct_label_ids`` and ``punct_label_vocab_file``. If none of parameters ``punct_label_ids`` and
            ``punct_label_vocab_file`` is provided, then punctuation label ids will be inferred from ``labels_file``
            file.
        capit_label_ids (:obj:`Dict[str, int]`, `optional`): same as ``punct_label_ids`` for capitalization labels.
        punct_label_vocab_file (:obj:`Union[os.PathLike, str]`, `optional`): a path to a file with punctuation labels.
            These labels include pad label. The pad label has to be the first label in the file. Each label is written
            on a separate line. Alternatively you can use ``punct_labels_ids`` parameter. If none of parameters
            ``punct_labels_ids`` and ``punct_label_vocab_file`` is provided, then punctuation label ids will be
            inferred from ``labels_file`` file.
        capit_label_vocab_file (:obj:`Union[os.PathLike, str]`, `optional`): same as ``punct_label_vocab_file`` for
            capitalization labels.
        tar_file_prefix (:obj:`str`, `optional`, defaults :obj:`'punctuation_capitalization'`): a string from which tar
            file names start. The string can contain only characters ``A-Z``, ``a-z``, ``0-9``, ``_``, ``-``, ``.``.
        n_jobs (:obj:`int`, `optional`): a number of workers for creating tarred dataset. If ``None``, then ``n_jobs``
            is equal to number of CPUs.
        audio_file (:obj:`Optional[Union[os.PathLike, str]]`, defaults to :obj:`None`): a path to a file with audio dataset file paths if dataset is lexical and audio. Must contain one path per line.
        use_audio (:obj:`bool`, `optional`, defaults to :obj:`False`): If set to ``True`` dataset becomes lexical and audio rather than only lexical.
        sample_rate (:obj:`int`, `optional`, defaults to :obj:`16000`) Targeted sample rate of audios If ``use_audio`` set to ``True``.
    """
    check_tar_file_prefix(tar_file_prefix, ValueError, 'tar_file_prefix')
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    text_file, labels_file = Path(text_file).expanduser(), Path(labels_file).expanduser()
    output_dir = Path(output_dir).expanduser()
    ds_params_str = DATASET_PARAMETERS_TMPL.format(
        prefix=tar_file_prefix,
        tokens_in_batch=tokens_in_batch,
        max_seq_length=max_seq_length,
        tokenizer=REPLACE_NOT_ALLOWED_CHARACTERS_IN_FILE_NAME.sub('-', tokenizer_name),
    )
    output_file_tmpl = ds_params_str + TAR_FINAL_TMPL
    metadata_file_name = output_dir / ('metadata.' + ds_params_str + '.json')
    remove_unexpected_files_and_dirs(output_dir, output_file_tmpl, metadata_file_name)
    audio_start_bytes = None
    if use_audio:
        num_lines, text_start_bytes, label_start_bytes, audio_start_bytes = get_fragment_start_bytes(
            text_file, labels_file, lines_per_dataset_fragment, audio_file
        )
    else:
        num_lines, text_start_bytes, label_start_bytes = get_fragment_start_bytes(
            text_file, labels_file, lines_per_dataset_fragment
        )
    if text_start_bytes:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"Both {labels_file} and {text_file} are empty. Tarred dataset cannot be created.")
    punct_label_ids, capit_label_ids = get_label_dictionaries(
        labels_file,
        label_start_bytes,
        num_lines,
        lines_per_dataset_fragment,
        pad_label,
        punct_label_ids,
        capit_label_ids,
        punct_label_vocab_file,
        capit_label_vocab_file,
        n_jobs,
    )

    with Progress(
        num_lines, ["Tokenization", "Batch mark up", "Batch building", "Writing tarred dataset"], "query"
    ) as progress_queues:
        Parallel(n_jobs=min(n_jobs, len(text_start_bytes)))(
            delayed(process_fragment)(
                text_file,
                labels_file,
                output_dir,
                text_start_pos,
                label_start_pos,
                lines_per_dataset_fragment,
                max_seq_length,
                tokens_in_batch,
                num_batches_per_tarfile,
                tokenizer_name,
                None if tokenizer_model is None else Path(tokenizer_model).expanduser(),
                None if vocab_file is None else Path(vocab_file).expanduser(),
                None if merges_file is None else Path(merges_file).expanduser(),
                special_tokens,
                use_fast_tokenizer,
                pad_label,
                punct_label_ids,
                capit_label_ids,
                fragment_idx,
                *progress_queues,
                audio_file,
                sample_rate,
                audio_file_start_pos,
                use_audio,
            )
            for fragment_idx, (text_start_pos, label_start_pos, audio_file_start_pos) in enumerate(
                zip(
                    text_start_bytes,
                    label_start_bytes,
                    audio_start_bytes if use_audio else [None for _ in range(len(text_start_bytes))],
                )
            )
        )
    repack_tar_files_with_not_enough_batches(output_dir, num_batches_per_tarfile)
    create_metadata_file(output_dir, output_file_tmpl, metadata_file_name, num_batches_per_tarfile)


class BertPunctuationCapitalizationTarredDataset(IterableDataset):
    """
    Punctuation capitalization dataset which allows not to load all data in memory simultaneously. A tarred dataset
    is created from text and label files using script
    `examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py
    <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py>`_
    or function
    :func:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset.create_tarred_dataset`.

    Args:
        metadata_file (:obj:`Union[os.PathLike, str]`): a path to tarred dataset metadata file. Metadata file and files
            referenced in metadata file are created by
            `examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py
            <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py>`_.
            Metadata file is a JSON file which contains ``'num_batches'``, ``'tar_files'``,
            ``'punct_label_vocab_file'``, ``'capit_label_vocab_file'`` items. The first item is total number of batches
            in a dataset, the second is a list of paths to tar files relative to directory containing
            ``metadata_file``. Items ``'punct_label_vocab_file'`` and ``'capit_label_vocab_file'`` are paths to
            ``.csv`` files which contain unique punctuation a capitalization label vocabularies. Vocabulary file paths
            are relative to directory containing the ``metadata_file``. Each line in ``'punct_label_vocab_file'`` and
            ``'capit_label_vocab_file'`` contains 1 label. The first lines in ``'punct_label_vocab_file'`` and
            ``'capit_label_vocab_file'`` files are neutral labels which also serve as pad labels. Neutral labels for
            punctuation and capitalization must be equal to the ``pad_label`` parameter.
        tokenizer (:obj:`TokenizerSpec`): a tokenizer instance used for tokenization of dataset source. A tokenizer
            instance is used for getting ids of [CLS], [PAD], and [SEP] tokens which are used for masks creation.
        pad_label (:obj:`str`): a label that is used for padding and for absence of punctuation or
            capitalization. Used for checking items ``'punct_label_vocab'`` and ``'capit_label_vocab'`` of dictionary
            in ``metadata_file``.
        label_info_save_dir (:obj:`Union[os.PathLike, str]`, `optional`): a path to a directory where label
            vocabularies are copied when method :meth:`save_labels_and_get_file_paths` is called. This parameter is
            useful if tarred dataset directory is read-only.
        ignore_extra_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`): whether to use only first token in a
            word for loss computation and training. If set to ``True``, then loss will be computed only for the first
            tokens of words.
        ignore_start_end (:obj:`bool`, `optional`, defaults to :obj:`True`): whether to compute loss for [CLS] and
            [SEP] tokens. If set to ``True``, then loss will not be computed for [CLS] and [SEP] tokens.
        world_size (:obj:`int`, `optional`, defaults to :obj:`1`): a number of processes used for model training. It is
            used together with a ``global_rank`` parameter to decide which tar files will be used in the current
            process.
        global_rank (:obj:`int`, `optional`, defaults to :obj:`0`): a number of current process in the pool of workers
            used for model training. It is used together with ``world_size`` parameter to decide which tar files will
            be used in the current process.
        shuffle_n (:obj:`int`, `optional`, defaults to :obj:`1`): a number of shuffled batches in a buffer.
            ``shuffle_n`` batches are loaded into memory, shuffled, and then yielded by a dataset instance.
        shard_strategy (:obj:`str`, defaults to :obj:``'scatter'``): Tarred dataset shard distribution strategy chosen as
            a str value during ddp.
            -   ``'scatter'``: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   ``'replicate'``: Optional shard strategy, where each node gets all the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of :param:`shuffle_n`.

                .. warning::
                    Replicated strategy allows every node to sample the entire set of available tar files,
                    and therefore more than one node may sample the same tarfile, and even sample the same
                    data points! As such, there is no assured guarantee that all samples in the dataset will be
                    sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                    occasions (when the number of shards is not divisible with ``world_size``), will not sample
                    the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                    or test datasets.
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
        metadata_file: Union[os.PathLike, str],
        tokenizer: TokenizerSpec,
        pad_label: str,
        label_info_save_dir: Optional[Union[os.PathLike, str]] = None,
        ignore_extra_tokens: bool = False,
        ignore_start_end: bool = True,
        world_size: int = 1,
        global_rank: int = 0,
        shuffle_n: int = 1,
        shard_strategy: str = "scatter",
        use_audio: bool = False,
    ) -> None:
        super().__init__()

        valid_shard_strategies = ['scatter', 'replicate']
        if shard_strategy not in valid_shard_strategies:
            raise ValueError(
                f"Invalid shard strategy of type {type(shard_strategy)} "
                f"{repr(shard_strategy) if len(repr(shard_strategy)) < 100 else repr(shard_strategy)[:100] + '...'}! "
                f"Allowed values are: {valid_shard_strategies}."
            )

        self.tokenizer = tokenizer
        self.metadata_file = Path(metadata_file).expanduser()
        if label_info_save_dir is None:
            self.for_nemo_ckpt = self.metadata_file.parent / LABEL_ID_DIR_FOR_NEMO_CHECKPOINT
        else:
            self.for_nemo_ckpt = Path(label_info_save_dir).expanduser() / LABEL_ID_DIR_FOR_NEMO_CHECKPOINT
        with open(self.metadata_file) as f:
            self.metadata = json.load(f)
        self.ignore_extra_tokens = ignore_extra_tokens
        self.ignore_start_end = ignore_start_end
        self.tar_files = []
        for file_path in self.metadata['tar_files']:
            file_path = Path(file_path).expanduser()
            if file_path.is_absolute():
                self.tar_files.append(str(file_path))
            else:
                self.tar_files.append(str(self.metadata_file.parent / file_path))
        self.punct_label_vocab_file = self.metadata_file.parent / self.metadata[METADATA_PUNCT_LABEL_VOCAB_KEY]
        self.capit_label_vocab_file = self.metadata_file.parent / self.metadata[METADATA_CAPIT_LABEL_VOCAB_KEY]
        self.punct_label_ids = load_label_ids(self.punct_label_vocab_file)
        self.capit_label_ids = load_label_ids(self.capit_label_vocab_file)
        self.pad_label = pad_label
        self._check_pad_label()

        if shard_strategy == 'scatter':
            logging.info("Tarred dataset shards will be scattered evenly across all nodes.")
            if len(self.tar_files) % world_size != 0:
                logging.warning(
                    f"Number of shards in tarred dataset ({len(self.tar_files)}) is not divisible "
                    f"by number of distributed workers ({world_size}). "
                    f"Some shards will not be used ({len(self.tar_files) % world_size})."
                )
            begin_idx = (len(self.tar_files) // world_size) * global_rank
            end_idx = begin_idx + (len(self.tar_files) // world_size)
            logging.info(
                "Partitioning tarred dataset: process (%d) taking shards [%d, %d)", global_rank, begin_idx, end_idx
            )
            batches_per_tar = self.metadata['num_batches'] // len(self.tar_files)
            self.tar_files = self.tar_files[begin_idx:end_idx]
            self.length = batches_per_tar * len(self.tar_files) * world_size

        elif shard_strategy == 'replicate':
            logging.info("All tarred dataset shards will be replicated across all nodes.")
            self.length = self.metadata['num_batches']

        else:
            raise ValueError(f"Invalid shard strategy! Allowed values are: {valid_shard_strategies}")

        self._dataset = wds.WebDataset(urls=self.tar_files, nodesplitter=None).decode(
            wds.handle_extension('.pyd', decode_pyd)
        )
        if shuffle_n > 0:
            self._dataset.shuffle(shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")
        self._dataset = self._dataset.to_tuple('__key__', 'batch.pyd').map(f=self._build_sample)

        self.use_audio = use_audio

    def _check_pad_label(self) -> None:
        """
        Checks the condition that ``pad_label`` passed to this class constructor has ``0`` id in
        ``self.punct_label_ids`` and ``self.capit_label_ids`` loaded from tarred dataset.
        """
        for label_ids, labels_file, task in [
            (self.punct_label_ids, self.metadata[METADATA_PUNCT_LABEL_VOCAB_KEY], "punctuation"),
            (self.capit_label_ids, self.metadata[METADATA_CAPIT_LABEL_VOCAB_KEY], "capitalization"),
        ]:
            if label_ids[self.pad_label] != 0:
                raise ValueError(
                    f"Pad label '{self.pad_label}' has non zero id {label_ids[self.pad_label]} in {task} "
                    f"ids dictionary loaded from {labels_file}."
                )

    def check_for_label_consistency_with_model_config(
        self,
        punct_label_ids: Optional[Dict[str, int]],
        capit_label_ids: Optional[Dict[str, int]],
        class_labels: DictConfig,
        common_dataset_parameters_config: DictConfig,
    ) -> None:
        """
        Checks that label ids loaded from tarred dataset are identical to those provided in
        ``model.common_dataset_parameters`` :ref:`config<common-dataset-parameters-config-label>` item. In addition,
        this method checks that label ids set in attributes ``punct_label_ids`` and ``capit_label_ids`` of an instance
        of
        :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_model.PunctuationCapitalizationModel`
        are identical to label ids loaded from tarred dataset.

        Args:
            punct_label_ids: a content of ``punct_label_ids`` attribute of an instance of
                :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_model.PunctuationCapitalizationModel`
                in which this tarred dataset is used.
            capit_label_ids: a content of ``capit_label_ids`` attribute of an instance of
                :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_model.PunctuationCapitalizationModel`
                in which this tarred dataset is used.
            class_labels: a config item ``model.class_labels``. See more in description of
                :ref:`class labels' config<class-labels-config-label>`.
            common_dataset_parameters_config: a config item ``model.common_dataset_parameters``. See more in
                of :ref:`common dataset parameters config<common-dataset-parameters-config-label>`.
        """
        tarred_dataset_label_desc_tmpl = (
            f'{{label_type}} labels loaded from tarred dataset with metadata file {self.metadata_file}'
        )
        if punct_label_ids is not None:
            if punct_label_ids != self.punct_label_ids:
                raise_not_equal_labels_error(
                    first_labels=self.punct_label_ids,
                    second_labels=punct_label_ids,
                    first_labels_desc=tarred_dataset_label_desc_tmpl.format(label_type='Punctuation'),
                    second_labels_desc="Punctuation labels stored in an attribute "
                    "`PunctuationCapitalizationModel.punct_label_ids`",
                )
        if capit_label_ids is not None:
            if capit_label_ids != self.capit_label_ids:
                raise_not_equal_labels_error(
                    first_labels=self.capit_label_ids,
                    second_labels=capit_label_ids,
                    first_labels_desc=tarred_dataset_label_desc_tmpl.format(label_type='Capitalization'),
                    second_labels_desc="Capitalization labels stored in an attribute"
                    "`PunctuationCapitalizationModel.capit_label_ids`",
                )
        if common_dataset_parameters_config.punct_label_ids is not None:
            cfg_punct_label_ids = dict(common_dataset_parameters_config.punct_label_ids)
            if cfg_punct_label_ids != self.punct_label_ids:
                raise_not_equal_labels_error(
                    first_labels=self.punct_label_ids,
                    second_labels=cfg_punct_label_ids,
                    first_labels_desc=tarred_dataset_label_desc_tmpl.format(label_type='Punctuation'),
                    second_labels_desc='Punctuation labels stored a config field '
                    '`model.common_dataset_parameters.punct_label_ids`',
                )
        if common_dataset_parameters_config.capit_label_ids is not None:
            cfg_capit_label_ids = dict(common_dataset_parameters_config.capit_label_ids)
            if cfg_capit_label_ids != self.capit_label_ids:
                raise_not_equal_labels_error(
                    first_labels=self.capit_label_ids,
                    second_labels=cfg_capit_label_ids,
                    first_labels_desc=tarred_dataset_label_desc_tmpl.format(label_type='Capitalization'),
                    second_labels_desc='Capitalization labels stored a config field '
                    '`model.common_dataset_parameters.capit_label_ids`',
                )
        if common_dataset_parameters_config.label_vocab_dir is not None:
            label_vocab_dir = Path(common_dataset_parameters_config.label_vocab_dir).expanduser()
            punct_label_vocab_file = label_vocab_dir / class_labels.punct_labels_file
            file_punct_vocab = load_label_ids(punct_label_vocab_file)
            if file_punct_vocab != self.punct_label_ids:
                raise_not_equal_labels_error(
                    first_labels=self.punct_label_ids,
                    second_labels=file_punct_vocab,
                    first_labels_desc=tarred_dataset_label_desc_tmpl.format(label_type='Punctuation'),
                    second_labels_desc=f'labels stored in file {punct_label_vocab_file} passed in '
                    f'`model.common_dataset_parameters.punct_label_vocab_file`',
                )
            capit_label_vocab_file = label_vocab_dir / class_labels.capit_labels_file
            file_capit_vocab = load_label_ids(capit_label_vocab_file)
            if file_capit_vocab != self.capit_label_ids:
                raise_not_equal_labels_error(
                    first_labels=self.capit_label_ids,
                    second_labels=file_capit_vocab,
                    first_labels_desc=tarred_dataset_label_desc_tmpl.format(label_type='Capitalization'),
                    second_labels_desc=f'labels stored in file {capit_label_vocab_file} passed in '
                    f'`model.common_dataset_parameters.capit_label_vocab_file`',
                )

    def save_labels_and_get_file_paths(
        self, punct_labels_file_name: str, capit_labels_file_name: str
    ) -> Tuple[Path, Path]:
        """
        Copies label vocabulary files for punctuation and capitalization into directory passed in the constructor
        parameter ``label_info_save_dir``. The names of new
        files are ``punct_labels_file_name`` and ``capit_labels_file_name``.

        The signatures of this method and the signature of the method
        :meth:`~nemo.collections.nlp.data.token_classification.BertPunctuationCapitalizationDataset.save_labels_and_get_file_paths`
        must be identical.

        Args:
            punct_labels_file_name (:obj:`str`): a name of punctuation labels file
            capit_labels_file_name (:obj:`str`): a name of capitalization labels file

        Returns:
            :obj:`Tuple[Path, Path]`: a tuple of 2 elements

                - :obj:`pathlib.Path`: a path to the new punctuation label ids file
                - :obj:`pathlib.Path`: a path to the new capitalization label ids file
        """
        self.for_nemo_ckpt.mkdir(parents=True, exist_ok=True)
        punct_label_ids_file = self.for_nemo_ckpt / punct_labels_file_name
        capit_label_ids_file = self.for_nemo_ckpt / capit_labels_file_name
        shutil.copy(str(self.punct_label_vocab_file), str(punct_label_ids_file))
        shutil.copy(str(self.capit_label_vocab_file), str(capit_label_ids_file))
        return punct_label_ids_file, capit_label_ids_file

    def _build_sample(self, batch: Tuple[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Takes batch loaded from tarred dataset and transforms it for passing to the model. Adds ``'segment_ids'``,
        ``'input_mask'``, ``'loss_mask'`` items to the batch.

        Args:
            batch: a tuple of 2 elements: batch name and a dictionary with ``'input_ids'``, ``'subtokens_mask'``,
                ``'punct_labels'``, ``'capit_labels'``. Batch name is not needed for training and inference and
                discarded.

        Returns:
            a batch in the form of a dictionary with items:
              - ``'input_ids'``: a ``np.int32`` numpy array of shape ``[Batch, Time]``;
              - ``'subtokens_mask'``: a boolean numpy array of shape ``[Batch, Time]``;
              - ``'punct_labels'``: a ``np.int32`` numpy array of shape ``[Batch, Time]``;
              - ``'capit_labels'``: a ``np.int32`` numpy array of shape ``[Batch, Time]``;
              - ``'segment_ids'``: a ``np.int8`` numpy array of shape ``[Batch, Time]``;
              - ``'input_mask'``: a boolean numpy array of shape ``[Batch, Time]``;
              - ``'loss_mask'``: a boolean numpy array of shape ``[Batch, Time]``.
        """
        _, batch = batch
        batch_segment_ids, batch_input_mask, batch_loss_mask = create_masks_and_segment_ids(
            batch['input_ids'],
            batch['subtokens_mask'],
            self.tokenizer.pad_id,
            self.tokenizer.cls_id,
            self.tokenizer.sep_id,
            self.ignore_start_end,
            self.ignore_extra_tokens,
        )
        batch['segment_ids'] = batch_segment_ids
        batch['input_mask'] = batch_input_mask
        batch['loss_mask'] = batch_loss_mask
        return batch

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """
        Constructs an iterator of batches. The values of one batch dictionary are numpy arrays of identical shapes
        ``[Batch, Time]``.

        Returns:
            :obj:`Iterator[Dict[str, np.ndarray]]`: an iterator of batches with items:

              - ``'input_ids'``: ``np.int32`` array containing encoded tokens,
              - ``'subtokens_mask'``: ``bool`` array which elements are ``True`` if they correspond to first token in
                a word,
              - ``'punct_labels'``: ``np.int32`` array with encoded punctuation labels,
              - ``'capit_labels'``: ``np.int32`` array with encoded capitalization labels,
              - ``'segment_ids'``: ``np.int8`` array filled with zeros (BERT token types in HuggingFace terminology),
              - ``'input_mask'``: ``bool`` array which elements are ``True`` if corresponding token is not a padding
                token,
              - ``'loss_mask'``: ``bool`` array which elements are ``True`` if loss is computed for corresponding
                token. See more in description of constructor parameters ``ignore_start_end``, ``ignore_extra_tokens``.
        """
        return self._dataset.__iter__()

    def __len__(self) -> int:
        return self.length

    def collate_fn(self, batches: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Return zeroth batch of ``batches`` list passed for collating and casts ``'segment_ids'``, ``'punct_labels'``,
        ``'capit_labels'`` to types supported by
        :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_model.PunctuationCapitalizationModel`.
        All output tensors have shape ``[Batch, Time]``.

        .. warning::
            ``batch size`` parameter of a PyTorch data loader and sampler has to be ``1``.

        Args:
            batches (:obj:`List[Dict[str, np.ndarray]]`): a list of batches passed for collating

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
        """
        batch = {k: torch.as_tensor(v) for k, v in batches[0].items()}
        batch['segment_ids'] = batch['segment_ids'].int()
        batch['punct_labels'] = batch['punct_labels'].long()
        batch['capit_labels'] = batch['capit_labels'].long()
        if self.use_audio:
            batch['features'] = batch['features'].to(torch.float32)
        return batch
