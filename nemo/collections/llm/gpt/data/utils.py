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

import copy
import datetime
import json
import logging
import multiprocessing as mp
import os
import pickle
import re
import signal
import time
from functools import lru_cache, partial
from queue import Empty
from typing import Any, Callable, List, Optional, Type, Union

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.classes import Dataset
from nemo.utils import AppState

logger = logging.getLogger(__name__)

PREFIX_STR = (
    "\x00"  # the prefix string used in the tokenizer to deal with the added empty token for some of the tokenizers
)

IGNORE_INDEX = -100
SYSTEM_TOKEN = "System"

TYPE_INSTRUCTION = {
    'TEXT_TO_VALUE': "",
    'VALUE_TO_TEXT': '',
}

GENERATION_REGEX = re.compile(r'\{%-?\s+generation\s+-?%\}')

__idx_version__ = "0.2"  # index file version
__idx_suffix__ = "idx"  # index file suffix


def build_index_from_memdata(fn, newline_int):
    """
    Build index of delimiter positions between samples in memmap.
    Can be provided externally.

    Returns a 1D array of ints.
    """
    # use memmap to read file
    mdata = np.memmap(fn, dtype=np.uint8, mode="r")
    # find newline positions
    midx = np.where(mdata == newline_int)[0]
    midx_dtype = midx.dtype
    # make sure to account for all data
    midx = midx.tolist()
    # add last item in case there is no new-line at the end of the file
    if (len(midx) == 0) or (midx[-1] + 1 != len(mdata)):
        midx = midx + [len(mdata) + 1]

    # remove empty lines from end of file
    while len(midx) > 1 and (midx[-1] - midx[-2]) < 2:
        midx.pop(-1)
    midx = np.asarray(midx, dtype=midx_dtype)

    # free memmap
    mdata._mmap.close()
    del mdata

    return midx


def safe_map(fn, iterable, workers=1, ctx="fork"):
    """
    Crash-resilient alternative to multiprocessing.Pool.map() that can handle
    worker process crashes gracefully without hanging the entire operation.
    """
    ctx = mp.get_context(ctx)
    input_queue = ctx.Queue()
    output_queue = ctx.Queue()
    indexed_inputs = list(enumerate(iterable))
    for job in indexed_inputs:
        input_queue.put(job)
    for _ in range(workers):
        input_queue.put(None)  # poison pill

    def worker_loop():
        while True:
            job = input_queue.get()
            if job is None:
                break
            i, item = job
            try:
                result = fn(item)
                output_queue.put((i, True, result, None))
            except Exception as e:
                output_queue.put((i, False, None, str(e)))

    processes = [ctx.Process(target=worker_loop) for _ in range(workers)]
    for p in processes:
        p.start()

    results = [None] * len(indexed_inputs)
    seen_indices = set()
    expected = len(indexed_inputs)
    received = 0

    # Collect whatever gets returned from live workers
    while received < expected:
        try:
            i, success, result, err = output_queue.get(timeout=0.5)
            seen_indices.add(i)
            results[i] = result if success else None
            if not success:
                logger.warning(f"Item {i}: {err}")
            received += 1
        except Empty:
            # Check if all workers are dead
            if all(not p.is_alive() for p in processes):
                logger.error("All workers exited before completing all tasks.")
                break
            continue

    # Join and check for crashes
    for p in processes:
        p.join()
        if p.exitcode is not None and p.exitcode < 0:
            sig = -p.exitcode
            try:
                sig_name = signal.Signals(sig).name
            except Exception:
                sig_name = f"signal {sig}"
            logger.warning(f"PID {p.pid} died from {sig_name}")

    # Patch any missing results from crashed workers
    for i in range(len(results)):
        if i not in seen_indices:
            logger.warning(f"No result for item {i}, likely crash")
            results[i] = None

    return results


class _TextMemMapDataset(Dataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    """

    def __init__(
        self,
        dataset_paths: List[str],
        newline_int: Optional[int] = 10,
        header_lines: Optional[int] = 0,
        workers: Optional[int] = None,
        tokenizer: Optional[Type["TokenizerSpec"]] = None,
        build_index_fn: Optional[Callable[[str, Optional[int]], bool]] = build_index_from_memdata,
        sort_dataset_paths: Optional[bool] = True,
        index_mapping_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_paths: list of JSONL file paths.
            newline_int: ASCII code to use to interpret newlines in file.
            header_lines: number of header lines in JSON files.
            workers: number of workers to use for creating index files.
            tokenizer: tokenizer to use to convert text to tokens.
            build_index_fn: a callable build_index_fn(fn, newline_int) -> midx [np.array]
                that returns the index of newlines in a file fn must be pickleable
                (to be used in multiprocessing.Pool.map).
            sort_dataset_paths: whether to sort datasets by paths.
            index_mapping_dir: directory to save the index mapping to.
                If None, will write to the same folder as the dataset.
        """
        super().__init__()
        self.mdata_midx_list = []

        # Make a single string into a list
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        if len(dataset_paths) < 1:
            raise ValueError("files_list must contain at leat one file name")

        self._newline_int = newline_int
        # skip first N lines
        self._header_lines = header_lines
        self._files_list = dataset_paths
        self._worker = workers
        self.tokenizer = tokenizer
        self._sort_dataset_paths = sort_dataset_paths

        if sort_dataset_paths:
            self._files_list = sorted(self._files_list)

        logger.info("Building data files")
        # load all files into memmap
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

        if not is_distributed or (is_distributed and torch.distributed.get_rank() == 0):
            # Create index files on global rank 0.
            build_index_files(
                dataset_paths,
                newline_int,
                workers=self._worker,
                build_index_fn=build_index_fn,
                index_mapping_dir=index_mapping_dir,
            )

        if is_distributed and not lightning_prepare_data():
            torch.distributed.barrier()

        if is_distributed and AppState().local_rank == 0:
            # If we are in a distributed multi-node set-up and index files are not stored on
            # a shared filesystem, then the index files created on global rank 0 are only
            # accessible to the workers on that node.
            #
            # Two cases may occur here:
            #
            # 1. case of a shared filesystem, or global_rank==0: the index files are present in
            #    the locally available filesystem, calling build_index_files() again is a no-op.
            # 2. case of a non-shared filesystem, and global_rank>0: the index files are not
            #    present in the locally available filesystem, calling build_index_files() again
            #    will create them.
            #
            # Outcome in all cases: all nodes have access to the index files in their filesystem.
            build_index_files(
                dataset_paths,
                newline_int,
                workers=self._worker,
                build_index_fn=build_index_fn,
                index_mapping_dir=index_mapping_dir,
            )

        if is_distributed and not lightning_prepare_data():
            torch.distributed.barrier()

        logger.info("Loading data files")
        start_time = time.time()
        mdata_midx_list = [self.load_file(fn, index_mapping_dir) for fn in self._files_list]
        logger.info(
            f"Time loading {len(mdata_midx_list)} "
            f"mem-mapped files: {datetime.timedelta(seconds=time.time() - start_time)}"
        )

        logger.info("Computing global indices")
        midx_bins = np.cumsum([(len(midx) - header_lines) for _, midx in mdata_midx_list])

        self.midx_bins = midx_bins
        self.mdata_midx_list = mdata_midx_list

        # figure out size of the dataset
        self._size = self.midx_bins[-1]

    def __del__(self):
        if self.mdata_midx_list:
            for mdata, midx in self.mdata_midx_list:
                mdata._mmap.close()

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        """
        Return a string from binary memmap
        """
        if (idx >= len(self)) or (idx < 0):
            raise IndexError(f"Index {idx} if out of dataset range with {len(self)} samples")

        # Identify the file containing the record
        file_id = np.digitize(idx, self.midx_bins, right=False)
        base_idx = self.midx_bins[file_id - 1] if file_id > 0 else 0
        file_idx = idx - base_idx + self._header_lines
        mdata, midx = self.mdata_midx_list[file_id]
        # load sample
        if file_idx == 0:
            i = 0
            j = midx[0]
        else:
            i = midx[file_idx - 1] + 1  # ignore newline
            j = midx[file_idx]

        # fetch sample from memmap

        try:
            sample = self._fetch_sample_from_memmap(mdata, i, j)
        except Exception as e:
            logger.error(f"Error while fetching sample from memmap: {e}")
            logger.error(f"file_id: {file_id}, file_idx: {file_idx}, i: {i}, j: {j}")
            raise e

        # parse raw text (e.g., tokenize)
        try:
            data = self._build_data_from_text(sample)
        except Exception as e:
            logger.error(f"Error while building data from text, possible issue with sample expected format: {e}")
            logger.error(f"sample: {sample}, file_id: {file_id}, file_idx: {file_idx}, i: {i}, j: {j}")
            raise e

        return data

    def _fetch_sample_from_memmap(self, mdata, i, j):
        """
        Fetchs the text sample.
        Can be overriden by child-classes to support loading of partial samples and alternative decode methods.
        """

        # load text sample by slicing memmap data[i:j]
        text = mdata[i:j].tobytes().decode("utf-8")

        return text

    def _build_data_from_text(self, text):
        """Allows child-classes to modify the parsing of raw text, prior to tokenization"""
        # tokenize text if tokenizer is given
        if self.tokenizer is not None:
            data = self.tokenizer.text_to_ids(text)
        else:
            data = text

        return data

    def load_file(self, fn, index_mapping_dir: Optional[str] = None):
        """
        Loads a text file as np.int8.

        Returns:
            mdata - memorymap of np.int8
            midx - indices pointing to the end-of-line (or end of file) position
            size - number of lines in file
        """
        logger.info(f"Loading {fn}")
        idx_fn = _index_fn(fn, index_mapping_dir)

        # create data map
        mdata = np.memmap(fn, dtype=np.uint8, mode="r")

        if _index_file_exists(idx_fn):
            # load index file into memory map
            midx = np.load(idx_fn + ".npy", allow_pickle=True, mmap_mode="r")
            # test for header
            if len(midx) < self._header_lines:
                raise RuntimeError(f"Missing header, expected {self._header_lines} header lines")

            # load meta info
            with open(idx_fn + ".info", "rb") as fp:
                idx_info_dict = pickle.load(fp)
            # test for mismatch in expected newline_int
            if "newline_int" in idx_info_dict:
                newline_int = idx_info_dict["newline_int"]
                if self._newline_int != newline_int:
                    logger.warning(f"Mismatch in newline_int, expected = {self._newline_int} but loaded {newline_int}")

            # test for version mismatch (useful to force recreation of index files)
            idx_version = idx_info_dict.get("version", "0.0")
            if __idx_version__ != idx_version:
                raise RuntimeError(
                    f"Version mismatch: Please delete existing '.{__idx_suffix__}' files. "
                    f"Expected version = {__idx_version__}, but file version = {idx_version}. File path = {idx_fn}"
                )
        else:
            raise ValueError(
                f"Memory Map for {fn} is not found, missing one or more of files: {idx_fn}.{{.npy,.info}}"
            )

        return (mdata, midx)


class _JSONLMemMapDataset(_TextMemMapDataset):
    """
    Memory-mapped iteration over a JSONL file.
    """

    def __init__(
        self,
        dataset_paths: List[str],
        newline_int: Optional[int] = 10,
        header_lines: Optional[int] = 0,
        workers: Optional[int] = None,
        tokenizer: Optional[Type["TokenizerSpec"]] = None,
        sort_dataset_paths: Optional[bool] = True,
        index_mapping_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_paths: list of JSONL file paths.
            newline_int: ASCII code to use to interpret newlines in file.
            header_lines: number of header lines in JSON files.
            workers: number of workers to use for creating index files.
            tokenizer: tokenizer to use to convert text to tokens.
            sort_dataset_paths: whether to sort datasets by paths.
            index_mapping_dir: directory to save the index mapping to.
                If None, will write to the same folder as the dataset.
        """
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=newline_int,
            header_lines=header_lines,
            workers=workers,
            tokenizer=tokenizer,
            sort_dataset_paths=sort_dataset_paths,
            index_mapping_dir=index_mapping_dir,
        )

    def _build_data_from_text(self, text):
        """Return a dictionary of data based on a single JSON line."""
        try:
            record = json.loads(text)
        except Exception as e:
            logger.error(f"Exception: {e}")
            logger.error(f"datapoint: {text}")
            raise e
        return record


class _OnlineSampleMapping:
    """
    This class replaces NeMo's get_samples_mapping function which pre-computes.
    It is used to create a sample mapping for certain number of samples, including
    pseudo-random shuffling.
    The sampler allows to down, or upsample a given dataset.
    Shuffling leads to pseudo-random shuffling, where blocks are shuffled,
    and each block is internally shuffled.
    """

    def __init__(
        self,
        dataset_size: int,
        num_samples: int,
        block_size: int = 1000000,
        cache_maxsize: int = 2,
        seed: int = 1,
        shuffle: bool = True,
        truncate_to_block_boundary: bool = False,
    ):
        """
        Args:
            dataset_size (int): Size of the dataset.
            num_samples (int): Number of samples the dataset should contain.
            block_size (int): Size of each sample block. This is used to shuffle the samples.
                              None will be replaced with dataset size.
            cache_maxsize (int): Maximum size of the blocks cache for the get_sample_block function.
            seed (int): Seed for the random number generator used for shuffling.
            shuffle (bool): Whether to shuffle the samples.
            truncate_to_block_boundary (bool): Whether to truncate the last block to the block boundary.
        """
        self.dataset_size = dataset_size
        self.num_samples = num_samples
        self.block_size = block_size if block_size is not None else self.dataset_size
        self.cache_maxsize = cache_maxsize
        self.seed = seed
        self.shuffle = shuffle
        self.truncate_to_block_boundary = truncate_to_block_boundary

        # we need at least num_samples (up-sampling) or dataset_size samples (correct down-sampling)
        self.required_samples = max(self.num_samples, self.dataset_size)
        # block size cannot be larger than dataset size
        self.block_size = min(self.block_size, self.dataset_size)
        # reduce the last block if needed, to match the required number of samples
        last_block_size = self.required_samples % self.block_size
        # store required blocks to cover num_samples samples and dataset_size samples
        self.num_blocks = int(np.ceil(self.required_samples / self.block_size))

        # if required, truncate the last block to the block boundary
        if self.truncate_to_block_boundary and last_block_size:
            # update num_samples to account for truncated last block only if needed
            if self.required_samples == self.num_samples:
                self.num_samples -= last_block_size

            # apdate num_blocks to account for truncated last block
            self.num_blocks -= 1
            self.required_samples -= last_block_size
            last_block_size = 0

        # create a list of blocks (should cover the entire dataset for correct down sampling)
        block_idx_list = np.arange(self.num_blocks)
        # compute the size of each block
        block_size_list = np.full(self.num_blocks, self.block_size)
        if last_block_size:
            block_size_list[-1] = last_block_size
            self.use_digitize = True
        else:
            self.use_digitize = False
        if shuffle:
            local_rng = np.random.RandomState(seed=self.seed)
            idx = local_rng.permutation(np.arange(self.num_blocks))
            block_idx_list = block_idx_list[idx]
            block_size_list = block_size_list[idx]

        # store only required number of blocks
        self.block_idx_list = block_idx_list
        self.block_size_list = block_size_list
        self.block_bins = np.cumsum(block_size_list)

        # NOTE: MAKE get_sample_block A CACHED FUNCTION!!!
        self.get_sample_block = lru_cache(maxsize=cache_maxsize, typed=False)(self.get_sample_block)

    def __str__(self):
        return (
            f"OnlineSampleMapping(dataset_size={self.dataset_size}, num_samples={self.num_samples}, "
            f"block_size={self.block_size}, cache_maxsize={self.cache_maxsize}, seed={self.seed}, "
            f"shuffle={self.shuffle}, truncate_to_block_boundary={self.truncate_to_block_boundary})"
        )

    def __getitem__(self, idx: int) -> int:
        # handle slices
        if isinstance(idx, slice):
            slc = idx
            start, stop, step = slc.start, slc.stop, slc.step

            # Handle None values
            start = handle_index(self, start if start is not None else 0)
            if start >= self.num_samples:
                start = self.num_samples
            stop = handle_index(self, stop if stop is not None else self.num_samples)
            if stop >= self.num_samples:
                stop = self.num_samples
            step = step if step is not None else 1
            sample_slice = [self[idx] for idx in range(start, stop, step)]
            return sample_slice
        # handle indices
        else:
            # If the index is out of range, raise IndexError
            if idx >= self.num_samples:
                raise IndexError("Index out of range")

            # support negative indices
            if idx < 0:
                idx += self.num_samples

                if idx < 0:
                    raise IndexError("Index out of range")

            # fetch the block sample index
            if self.use_digitize:
                block_idx = np.digitize(idx, self.block_bins)
            else:
                block_idx = idx // self.block_size
            sample_block = self.get_sample_block(block_idx)

            # use the local index to fetch the sample
            local_idx = idx - self.block_bins[block_idx]
            sample_idx = sample_block[local_idx]

            return sample_idx, None, None  # for comtability with NeMo's get_samples_mapping

    def __len__(self) -> int:
        return self.num_samples

    def __reduce__(self):
        """Add support for pickling. Needed due to functools.lru_cache."""
        # Return a tuple with a callable and arguments to recreate the object
        return (
            self.__class__,
            (
                self.dataset_size,
                self.num_samples,
                self.block_size,
                self.cache_maxsize,
                self.seed,
                self.shuffle,
                self.truncate_to_block_boundary,
            ),
        )

    def __reduce_ex__(self, protocol):
        # Optional method that defines the protocol version
        return self.__reduce__()

    def get_sample_block(self, block_idx: int) -> np.ndarray:
        """
        Returns a block of samples of size self.block_size, shuffled if needed.
        NOTE: This method will be cached using functools.lru_cache for efficiency during construction.
        """
        if block_idx >= self.num_blocks:
            raise IndexError(f"block_idx {block_idx} is out of range. Maximum block_idx is {self.num_blocks-1}")

        # recover index of original block (before shuffling)
        start_idx = self.block_idx_list[block_idx] * self.block_size
        end_idx = start_idx + self.block_size_list[block_idx]
        sample_block = np.arange(start_idx, end_idx)

        # shuffle if needed
        if self.shuffle:
            local_rng = np.random.RandomState(seed=self.seed + block_idx)
            sample_block = local_rng.permutation(sample_block)

        # project indices to the dataset size
        sample_block = sample_block % self.dataset_size

        return sample_block


def build_index_files(
    dataset_paths,
    newline_int,
    workers=None,
    build_index_fn=build_index_from_memdata,
    index_mapping_dir: str = None,
):
    """Auxiliary method to build multiple index files"""
    if len(dataset_paths) < 1:
        raise ValueError("files_list must contain at leat one file name")

    if workers is None:
        workers = max(1, os.cpu_count() // 2)

    logger.info(f"Processing {len(dataset_paths)} data files using {workers} workers")
    # load all files into memmap
    start_time = time.time()
    build_status = safe_map(
        partial(
            _build_memmap_index_files,
            newline_int,
            build_index_fn,
            index_mapping_dir=index_mapping_dir,
        ),
        dataset_paths,
        workers=workers,
    )

    logger.info(
        f"Time building {sum(build_status)} / {len(build_status)} "
        f"mem-mapped files: {datetime.timedelta(seconds=time.time() - start_time)}"
    )


def handle_index(dataset, idx):
    """
    Remaps negative indices and handles numpy int indices.

    Arguments:
        dataset (Dataset): dataset to index into
        idx (int): Index. Can include negative indices.
    Returns:
        int: Remapped and fully qualified index.

    Raises:
        IndexError: If a negative index is out of range.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from torch.utils.data import TensorDataset
        >>> from nemo_chem.data.fasta_dataset import handle_index
        >>> dataset = TensorDataset(torch.tensor(-np.arange(5)))
        >>> handle_index(dataset, 1)
        1
        >>> handle_index(dataset, -2)
        3

    """
    if idx < 0 and idx > -len(dataset) - 1:
        idx = len(dataset) + idx
    elif idx < 0:
        raise IndexError(f'Index out of range: {idx}')

    return idx


def lightning_prepare_data():
    """
    This function checks whether it is invoked in lightning's hook "prepare_data", which is run only on rank 0.
    TextMemMapDataset contains a torch.distributed.barrier operation, so when run inside the single-process hook
    prepare_data, the barrier operation would hang forever.
    """
    import inspect

    return any(
        [
            frame.function == 'prepare_data' and 'prepare_packed_sequence_data' in frame.code_context[0]
            for frame in inspect.stack()
        ]
    )


def _get_samples_mapping(
    indexed_dataset,
    data_prefix,
    num_epochs,
    max_num_samples,
    max_seq_length,
    short_seq_prob,
    seed,
    name,
    binary_head,
    index_mapping_dir: str = None,
    samples_mapping: Any = None,
    sanity_check_dist_workers: bool = True,
):
    """Get a list that maps a sample index to a starting sentence index, end sentence index, and length"""

    from megatron.core import parallel_state

    if not num_epochs:
        if not max_num_samples:
            raise ValueError("Need to specify either max_num_samples " "or num_epochs")
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    if index_mapping_dir is not None:
        indexmap_filename = os.path.join(index_mapping_dir, os.path.basename(data_prefix))
    else:
        indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += '_{}ep'.format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{:0.2f}ssp'.format(short_seq_prob)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'

    # Build the indexed mapping if not exist and not provided externally.
    if samples_mapping is None and torch.distributed.get_rank() == 0 and not os.path.isfile(indexmap_filename):
        # Fake index mapping if missing
        if (getattr(indexed_dataset, 'doc_idx', None) is None) and (getattr(indexed_dataset, 'sizes', None) is None):
            _make_indexed_dataset_compatibility(indexed_dataset)

        print(
            ' > WARNING: could not find index map file {}, building '
            'the indices on rank 0 ...'.format(indexmap_filename)
        )

        # Make sure the types match the helpers input types.
        assert indexed_dataset.doc_idx.dtype == np.int64
        assert indexed_dataset.sizes.dtype == np.int32

        # Build samples mapping
        verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        logger.info(' > building samples index mapping for {} ...'.format(name))
        # First compile and then import.
        try:
            from megatron.core.datasets import helpers_cpp
        except ImportError:
            raise ImportError(
                'Could not compile megatron dataset C++ helper functions '
                'and therefore cannot import helpers python file.'
            )
        samples_mapping = helpers_cpp.build_mapping(
            indexed_dataset.doc_idx,
            indexed_dataset.sizes,
            num_epochs,
            max_num_samples,
            max_seq_length,
            short_seq_prob,
            seed,
            verbose,
            2 if binary_head else 1,
        )
        logger.info(' > done building samples index maping')
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        logger.info(' > saved the index mapping in {}'.format(indexmap_filename))
        # Make sure all the ranks have built the mapping
        logger.info(
            ' > elasped time to build and save samples mapping ' '(seconds): {:4f}'.format(time.time() - start_time)
        )

    if sanity_check_dist_workers:
        torch.distributed.barrier()
        counts = torch.cuda.LongTensor([1])
        torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group(with_context_parallel=True))
        torch.distributed.all_reduce(counts, group=parallel_state.get_pipeline_model_parallel_group())
        assert counts[0].item() == (
            torch.distributed.get_world_size()
            // torch.distributed.get_world_size(group=parallel_state.get_tensor_model_parallel_group())
        )
    # Load indexed dataset if not given externally.
    if samples_mapping is None:
        logger.info(' > loading indexed mapping from {}'.format(indexmap_filename))
        start_time = time.time()
        samples_mapping = np.load(indexmap_filename, allow_pickle=True, mmap_mode='r')
        logger.info('    loaded indexed file in {:3.3f} seconds'.format(time.time() - start_time))
        logger.info('    total number of samples: {}'.format(samples_mapping.shape[0]))

    # Deallocate temporary numpy arrays that were created for `get_samples_mapping()` when needed
    if hasattr(indexed_dataset, 'doc_idx') and hasattr(indexed_dataset, 'sizes'):
        _deallocate_indexed_dataset_memory(indexed_dataset)

    return samples_mapping


def _make_indexed_dataset_compatibility(dataset):
    """Make any dataset compatible with IndexedDataset for Megatron samples mapping."""
    if (getattr(dataset, 'doc_idx', None) is not None) or (getattr(dataset, 'sizes', None) is not None):
        raise AttributeError("Dataset already has doc_idx or sizes attributes.")

    dataset.doc_idx = np.arange(len(dataset) + 1, dtype=np.int64)
    dataset.sizes = np.ones(len(dataset), dtype=np.int32)

    return dataset


def _preprocess(
    source: dict,
    tokenizer: TokenizerSpec,
    name_end_token_ids: int,
    label_start_ids: list,
    special_tokens: dict,
    num_turn_start_tokens: int,
):
    """
    Given a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    header, conversation, data_type, mask_role = _get_header_conversation_type_mask_role(source, special_tokens)
    # tokenize conversations
    input_ids = tokenizer.text_to_ids(conversation)
    target = copy.deepcopy(input_ids)
    header_tokens = tokenizer.text_to_ids(header)
    header_len = len(header_tokens)

    ids = []
    tokenized_lens = []
    if not torch.equal(torch.tensor(target[:header_len]), torch.tensor(header_tokens)):
        logger.warning(
            "First few tokens of the conversation are not the same as the header tokens. "
            f"{target[:header_len]=}\n {header_tokens=}"
        )
    for s in source['conversations']:
        # hack to remove the extra empty token in front
        id1 = tokenizer.text_to_ids(PREFIX_STR + s["value"])
        id2 = tokenizer.text_to_ids(PREFIX_STR)
        tokenized_sentence = id1[len(id2) :]
        ids.append(torch.tensor(tokenized_sentence))
        tokenized_lens.append(len(tokenized_sentence))
    speakers = [sentence["from"] for sentence in source['conversations']]
    # assert mask_role in speakers, "mask role not in the conversation"
    split_mask = mask_role.split(',')
    for s in split_mask:
        assert s in speakers, "mask role not in the conversation"

    target = torch.LongTensor(target)
    # not going to train on the header
    target[:header_len] = IGNORE_INDEX
    input_ids = torch.LongTensor(input_ids)
    _mask_targets(
        target,
        tokenized_lens,
        speakers,
        header_len,
        ids,
        tokenizer,
        mask_role,
        data_type,
        name_end_token_ids,
        special_tokens,
        label_start_ids,
        num_turn_start_tokens,
    )
    mask = (target != IGNORE_INDEX).bool()
    assert mask.sum().item() != 0, "mask is empty"
    # Choose the last conversation as answer other history are context
    last_ignore_index_pos = torch.nonzero(target == IGNORE_INDEX)[-1].item() + 1
    context_ids = input_ids[:last_ignore_index_pos]
    answer_ids = input_ids[last_ignore_index_pos:]

    return dict(input_ids=input_ids, loss_mask=mask, context_ids=context_ids, answer_ids=answer_ids)


def _convert_to_openai_messages(source: dict) -> List[dict]:
    """
    Input
        source - HuggingFace AutoTokenizer messages format
            {"messages": [
                {"role": "system","content":"<text>"},
                {"role": "user","content":"<text>"},
                {"role": "assistant","content":"<text>"}
            ]}
        source - can also be conversation format, these are converted to HF messages format
            Mask and type are ignored. Mask will apply to all non-assistant output tokens.
            {"conversations": [
                {"from": "User","value":"<text>"},
                {"from": "Assistant","value":"<text>", "mask": "User", "system": "<text>", "type": "TEXT_TO_VALUE"}
            ]}

    Output

    [
        {"role": "system","content":"<text>"},
        {"role": "user","content":"<text>"},
        {"role": "assistant","content":"<text>"}
    ]
    """
    if isinstance(source, dict):
        if source.get("conversations"):
            # Detect if Nemo {"conversations": [{"from": "User/Assistant", "value": ""}]}
            # Convert to HuggingFace chat template [{"role": "stystem/user/assistant", "content": ""}]
            chat = [{"role": convo["from"].lower(), "content": convo["value"]} for convo in source["conversations"]]
            if source.get("system"):
                chat.insert(0, {"role": "system", "content": source["system"]})

        elif source.get("messages"):
            # HuggingFace chat template {"messages": [{"role": "system/user/assistant", "content": ""}]}
            chat = source.get("messages")
    else:
        chat = source

    return chat


def _chat_preprocess(source: dict, tokenizer: TokenizerSpec, tool_schemas: Optional[List[Any]] = None) -> dict:
    """
    Preprocess messages to apply chat template and tokenize. Returns a dictionary of tokens
    Input:
        source - HuggingFace AutoTokenizer messages format
            {"messages": [
                {"role": "system","content":"<text>"},
                {"role": "user","content":"<text>"},
                {"role": "assistant","content":"<text>"}
            ]}
        source - can also be conversation format, these are converted to HF messages format
            Mask and type are ignored. Mask will apply to all non-assistant output tokens.
            {"conversations": [
                {"from": "User","value":"<text>"},
                {"from": "Assistant","value":"<text>", "mask": "User", "system": "<text>", "type": "TEXT_TO_VALUE"}
            ]}
        tokenizer - tokenizer to apply chat templates to
        tool_schemas - Optional tool_schemas to supply to apply_chat_template, these will be superseeded
           by tools supplied with the message

    Output
    {
        "input_ids": torch.LongTensor(),
        "mask": torch.BoolTensor(),
        "context_ids": torch.LongTensor(),
        "answer_ids": torch.LongTensor(),
    }

    * input_ids contain tokenized messages with chat template applied
    * mask corresponds to tokens of input_ids where 1 represents output tokens for the role `assistant` in both
    context and answer for multi-turn, and 0 to mask all other tokens, e.g. system, user, and tool calling.
    * context_ids contain tokenized messages with chat template applied for all messages except assistant's last
    * answer_ids contain tokenized messages with chat template applied for only the assistant's last generated
    output
    """
    if not hasattr(tokenizer.tokenizer, "apply_chat_template"):
        raise ValueError("Cannot apply chat template with tokenizer that is not a HuggingFace AutoTokenizer")

    chat = _convert_to_openai_messages(source)
    tools = None
    if isinstance(source, dict):
        tools = source.get("tools") or tool_schemas
    else:
        tools = tool_schemas

    # assistant mask only works if chat template has generation keyword
    template_has_generation_kwd = GENERATION_REGEX.search(tokenizer.tokenizer.chat_template) is not None

    tokenized_chat = tokenizer.tokenizer.apply_chat_template(
        chat,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=template_has_generation_kwd,
    )

    # Choose the last conversation as answer other history are context by finding the last masked token
    # which indicates end of context and beginning of answer
    input_ids = tokenized_chat.get("input_ids")
    if template_has_generation_kwd:
        mask = tokenized_chat['assistant_masks']
    else:
        mask = [1] * len(input_ids)

    if tokenizer.eos_id and input_ids[-1] != tokenizer.eos_id:
        input_ids += [tokenizer.eos_id]
        mask += [1]

    if 0 in mask:
        # traverse the list backward for first occurrence of masked token
        context_end_idx = len(mask) - mask[::-1].index(0)
    else:
        context_end_idx = len(mask)

    context_ids = input_ids[:context_end_idx]
    answer_ids = input_ids[context_end_idx:]

    return dict(
        input_ids=torch.LongTensor(input_ids),
        loss_mask=torch.BoolTensor(mask),
        context_ids=torch.LongTensor(context_ids),
        answer_ids=torch.LongTensor(answer_ids),
    )


def _mask_targets(
    target,
    tokenized_lens,
    speakers,
    header_len,
    s_ids,
    tokenizer,
    mask_role,
    gtype,
    name_end_token_ids,
    special_tokens,
    label_start_ids,
    num_turn_start_tokens,
):
    """This function masks the tokens so the loss is computed only on the non-masked role's responses.
    For 'TEXT_TO_VALUE' type, the loss is computed on the value attributes.

    Args:
        target (Tensor): input ids
        tokenized_lens (List[int]): array of lengths of each turns
        speakers (List[str]): array of speakers of each turns
        header_len (int): the system prompt length
        s_ids (List[Tensor]): array of tokenized ids of each turns
        tokenizer (TokenizerSpec): tokenizer object
        mask_role (str): the speaker id to be masked from loss computation.
        gtype (str): either 'TEXT_TO_VALUE' or 'VALUE_TO_TEXT'
        name_end_token_ids (int): end of name token ids
        special_tokens (dict): special tokens used for the chat prompt.
        label_start_ids (list): list of label start token ids,
        num_turn_start_tokens (int): number of tokens of the turn_start str
    """
    TURN_TOKEN = special_tokens['turn_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']
    label_start_ids = torch.tensor(label_start_ids)
    name_end_token_ids = torch.tensor(name_end_token_ids)

    cur_idx = header_len
    tgt_len = target.shape[0]
    for i, (tokenized_len, speaker, s_id) in enumerate(zip(tokenized_lens, speakers, s_ids)):
        # note, sentence piece will add extra empty token in front. has to compute the diff
        id1 = tokenizer.text_to_ids(PREFIX_STR)
        id2 = tokenizer.text_to_ids(PREFIX_STR + TURN_TOKEN + speaker + END_NAME_SIGNAL)
        skip_name_len = len(id2) - len(
            id1
        )  # s_ids[:skip_name_len] is the name part of the prompt 'TURN_TOKEN + speaker + END_NAME_SIGNAL'
        # get the position of the label start string in this turn
        location = _identify_start_index_of_subsequence(label_start_ids, s_id)

        if location >= 0:
            # if it contains the label start tokens
            if gtype == 'VALUE_TO_TEXT':
                # handles the case that condition on labels to generate respone
                # the next token after the name part of the prompt is the beginning of the label start tokens
                assert skip_name_len == location
                # find the first new line token after the label part, which indicates the end of the whole label string
                # newline_loc = torch.where((s_id[skip_name_len:] == name_end_token_ids))[0]
                newline_loc = _identify_start_index_of_subsequence(name_end_token_ids, s_id[skip_name_len:])
                if newline_loc < 0:
                    # cannot find new line token, which means the the whole turn
                    # is just a partial label string. Mask the whole turn
                    target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
                    continue
                # skip the label part and the new line token
                more_skip_len = newline_loc + len(name_end_token_ids)
                # skip the name part and the label part
                skip_name_len += more_skip_len
            elif gtype == 'TEXT_TO_VALUE':
                # handles the case that condition on response to generate label
                # skip the name part, response and the label start tokens part,
                # the remainder is the label string without label start, e.g. 'quality:9,toxicity:8...'
                skip_name_len = location + len(label_start_ids)
        if cur_idx >= tgt_len:
            break
        elif cur_idx + tokenized_len < tgt_len:
            # Check whether the mask is applied to the correct position, the first token is turn start tokens
            if not torch.equal(target[cur_idx + 1 : cur_idx + tokenized_len], s_id[1:]):
                logger.warning("a sentence mismatches the corresponding piece " "in the conversation")
        if i == 0 and (gtype == 'VALUE_TO_TEXT' or gtype is None):
            # mask the first turn completely to provide at least one turn as context for the rest
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker in mask_role and i == 1 and gtype == 'TEXT_TO_VALUE':
            # leave the first turn start tag unmasked, servers severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker in mask_role and (i > 1):
            # leave the first turn start tag unmasked, which severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker in mask_role and (i <= 1):
            # mask out everything in the second turn
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        else:
            # mask up to name part, label part for VALUE_TO_TEXT, or name part,
            # response and label start tokens for TEXT_TO_VALUE, or just the name part if gtype is None
            target[cur_idx : cur_idx + skip_name_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _get_header_conversation_type_mask_role(source, special_tokens):
    END_SIGNAL = special_tokens['end_of_turn']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    data_type = None
    if 'type' in source:
        data_type = source['type']
        if data_type is not None:
            assert data_type in TYPE_INSTRUCTION, f"source type {data_type} not supported"
    # add end signal and concatenate together
    conversation = source['system']
    if data_type is not None:
        if TYPE_INSTRUCTION[data_type] != '':
            conversation = conversation + '\n' + TYPE_INSTRUCTION[data_type]
    mask_role = source.get('mask', 'User')
    header = f"{special_tokens['system_turn_start']}{SYSTEM_TOKEN}{END_NAME_SIGNAL}{conversation}{END_SIGNAL}"
    conversation = _add_speaker_and_signal(header, source['conversations'], mask_role, data_type, special_tokens)

    return header, conversation, data_type, mask_role


def _add_speaker_and_signal(header, source, mask_role, gtype, special_tokens):
    TURN_TOKEN = special_tokens['turn_start']
    END_SIGNAL = special_tokens['end_of_turn']
    LABEL_START = special_tokens['label_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = ""
    conversation = header
    for i, sentence in enumerate(source):
        sentence_from = sentence["from"]
        role_token = TURN_TOKEN
        if gtype is None:
            sentence["value"] = (
                BEGIN_SIGNAL + role_token + sentence_from + END_NAME_SIGNAL + sentence["value"] + END_SIGNAL
            )
        elif gtype == "VALUE_TO_TEXT":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + (
                    _response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
                + sentence["value"]
                + END_SIGNAL
            )
        elif gtype == "TEXT_TO_VALUE":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + sentence["value"]
                + END_SIGNAL
                + (
                    _response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
            )
        else:
            raise ValueError(
                f"source type {gtype} not supported, only 'VALUE_TO_TEXT' and 'TEXT_TO_VALUE' are supported"
            )
        conversation += sentence["value"]
        # if the last turn is not masked, add next token start token to the end,
        # which will be included for loss calculation
        if sentence_from not in mask_role and i == len(source) - 1:
            conversation += TURN_TOKEN

    return conversation


def _response_value_formater(label, label_start, end_signal):
    if isinstance(label, str):
        return label_start + label + end_signal
    elif label is None:
        return ''
    else:
        raise ValueError(f'Unknown label type {type(label)}, only str type is supported')


def _identify_start_index_of_subsequence(subsequence, sequence):
    """find the location of the small tensor in the large tensor.
        e.g.  small = [1,3], large = [2,3,1,3], returns 2
              small = [3,2], large = [2,3,1,3], returns -1
    Args:
        small (tensor): small tensor
        large (tensor): large tensor
    """
    for i in range(sequence.size(0) - subsequence.size(0) + 1):
        if torch.equal(sequence[i : i + subsequence.size(0)], subsequence):
            return i
    return -1


def _build_memmap_index_files(newline_int, build_index_fn, fn, index_mapping_dir: str):
    """Helper function to build an index file"""
    idx_fn = _index_fn(fn, index_mapping_dir)

    # create data map
    if _index_file_exists(idx_fn):
        return False
    else:
        logger.info(f"Building indexing for fn = {fn}")
        # find all newline positions
        midx = build_index_fn(fn, newline_int)
        # validate midx
        midx = np.asarray(midx)
        if not np.issubdtype(midx.dtype, np.integer):
            raise TypeError(f"midx must be an integer array, but got type = {midx.dtype}")

        # create e metadata file
        data = dict(newline_int=newline_int, version=__idx_version__)

        # save index as numpy array to enable memmap reading
        logger.info(f"Saving idx file = {idx_fn}.npy")
        np.save(idx_fn + ".npy", midx, allow_pickle=True)
        logger.info(f"Saving metadata file = {idx_fn}.info")
        pickle.dump(data, open(idx_fn + ".info", "wb"))

        return True


def _index_fn(fn: str, index_mapping_dir: str) -> str:
    """Return base file name of index files.

    This returns the base file name associated with specified index
    files. This base name is the base on top of which suffixes
    like .npy or .info are added.

    The parent directory is created if it does not already exist.

    fn may be specified in multiple ways:
    1. file name: data.jsonl,
    2. relative path to a file: relative/path/to/data.jsonl,
    3. absolute path to a file: /absolute/path/to/data.jsonl.

    This function returns paths in the pattern of:
    1. /path/to/input_mapping_dir/data.jsonl.idx
    2. /path/to/input_mapping_dir/relative/path/to/data.jsonl.idx
    3. /path/to/input_mapping_dir/absolute/path/to/data.jsonl.idx

    Args:
        fn: filename to get base name for.
        index_mapping_dir: directory to save the index mapping to.
                If None, will write to the same folder as the dataset.
    """
    if index_mapping_dir:
        # Remove leading "/" and "..".
        while fn.startswith(("/", "..")):
            if fn.startswith(".."):
                fn = fn.lstrip("..")
            if fn.startswith("/"):
                fn = fn.lstrip("/")
        idx_fn = f"{os.path.join(index_mapping_dir, fn)}.{__idx_suffix__}"
        # Create parent directory if needed.
        os.makedirs(os.path.dirname(idx_fn), exist_ok=True)
    else:
        idx_fn = f"{fn}.{__idx_suffix__}"
    return idx_fn


def _index_file_exists(idx_fn):
    """Helper function to test if index file exists"""
    if os.path.exists(idx_fn + ".npy") and os.path.exists(idx_fn + ".info"):
        return True
    else:
        return False


def _deallocate_indexed_dataset_memory(indexed_dataset):
    """Deallocate memory of an IndexedDataset."""
    indexed_dataset.sizes = None
    indexed_dataset.doc_idx = None


def _reconfigure_limit_batches(limit_batches, dataloader) -> Union[int, float]:
    """
    Reconfigure trainer.limit_val_batches for pretraining
    """
    # Override limit_batches in terms of num microbatches and so there are limit_batches//num_micro_batches
    #   num of global batches
    try:
        from megatron.core.num_microbatches_calculator import get_num_microbatches

    except (ImportError, ModuleNotFoundError):
        logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
        from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    if isinstance(limit_batches, int):
        limit_batches *= get_num_microbatches()
    else:
        assert isinstance(limit_batches, float)
        # Don't reconfigure if limit_batches is 0.0 or if there's no dataloader
        if limit_batches == 0.0 or dataloader is None:
            return limit_batches
        # len(dataloader) returns len as num of microbatches
        dl_len_in_micro_batches = len(dataloader)
        if len(dataloader) != float("inf"):
            if limit_batches == 1.0:
                limit_batches = dl_len_in_micro_batches
            else:
                limit_micro_batches = int(dl_len_in_micro_batches * limit_batches)
                if limit_micro_batches == 0 and limit_batches > 0.0:
                    min_percentage = 1.0 / len(dataloader)
                    raise ValueError(
                        f"You requested to check {limit_batches} of the val_dataloader but"
                        f" {limit_batches} * {len(dataloader)} < 1. Please increase the"
                        f" `limit_val_batches` argument. Try at least"
                        f" `limit_val_batches={min_percentage}`"
                    )
                # Make sure trainer.limit_val_batches is a multiple of num of microbatches
                if limit_micro_batches < get_num_microbatches():
                    limit_batches = get_num_microbatches()
                else:
                    limit_batches = limit_batches - limit_batches % get_num_microbatches()

    return limit_batches
