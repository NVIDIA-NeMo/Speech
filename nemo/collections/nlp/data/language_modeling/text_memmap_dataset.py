# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import datetime
import json
import multiprocessing as mp
import os
import pickle
import time
from functools import partial
from typing import Callable, List, Optional, Type

import numpy as np
import torch

from nemo.core import Dataset
from nemo.utils import AppState, logging

__all__ = ["TextMemMapDataset", "CSVMemMapDataset", "build_index_files"]
__idx_version__ = "0.2"  # index file version
__idx_suffix__ = "idx"  # index file suffix


def _build_index_from_memdata(fn, newline_int):
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


class TextMemMapDataset(Dataset):
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
        build_index_fn: Optional[Callable[[str, Optional[int]], bool]] = _build_index_from_memdata,
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

        logging.info(f"Building data files")
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

        if is_distributed:
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

        if is_distributed:
            torch.distributed.barrier()

        logging.info(f"Loading data files")
        start_time = time.time()
        mdata_midx_list = [self.load_file(fn, index_mapping_dir) for fn in self._files_list]
        logging.info(
            f"Time loading {len(mdata_midx_list)} mem-mapped files: {datetime.timedelta(seconds=time.time() - start_time)}"
        )

        logging.info("Computing global indices")
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
            logging.error(f"Error while fetching sample from memmap: {e}")
            logging.error(f"file_id: {file_id}, file_idx: {file_idx}, i: {i}, j: {j}")
            raise e

        # parse raw text (e.g., tokenize)
        try:
            data = self._build_data_from_text(sample)
        except Exception as e:
            logging.error(
                f"Error while building data from text, possible issue with sample expected format (see offending sample below): {e}"
            )
            logging.error(f"sample: {sample}, file_id: {file_id}, file_idx: {file_idx}, i: {i}, j: {j}")
            raise e

        return data

    def _fetch_sample_from_memmap(self, mdata, i, j):
        """Fetchs the text sample. Can be overriden by child-classes to support loading of partial samples and alternative decode methods"""
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
        logging.info(f"Loading {fn}")
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
            idx_info_dict = pickle.load(open(idx_fn + ".info", "rb"))
            # test for mismatch in expected newline_int
            if "newline_int" in idx_info_dict:
                newline_int = idx_info_dict["newline_int"]
                if self._newline_int != newline_int:
                    logging.warning(
                        f"Mismatch in newline_int, expected = {self._newline_int} but loaded {newline_int}"
                    )

            # test for version mismatch (useful to force recreation of index files)
            idx_version = idx_info_dict.get("version", "0.0")
            if __idx_version__ != idx_version:
                raise RuntimeError(
                    f"Version mismatch: Please delete existing '.{__idx_suffix__}' files. Expected version = {__idx_version__}, but file version = {idx_version}. File path = {idx_fn}"
                )
        else:
            raise ValueError(
                f"Memory Map for {fn} is not found, missing one or more of files: {idx_fn}.{{.npy,.info}}"
            )

        return (mdata, midx)


class CSVMemMapDataset(TextMemMapDataset):
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
        sort_dataset_paths: Optional[bool] = True,
        data_col=1,
        data_sep=",",
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
            data_col: index of data column.
            data_sep: data separator.
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
        self._data_col = data_col
        self._data_sep = data_sep

    def _build_data_from_text(self, text):
        """Return a CSV field from text"""
        # get CSV field
        text = text.split(self._data_sep)[self._data_col]
        # tokenize
        return super()._build_data_from_text(text)


class CSVFieldsMemmapDataset(TextMemMapDataset):
    """
    Allow per-line lazy access to multiple csv files using numpy memmap.
    Returns a dictionary with multiple fields.
    """

    def __init__(
        self,
        dataset_paths,
        newline_int=10,
        header_lines=1,
        workers=None,
        tokenizer=None,
        sort_dataset_paths=True,
        data_sep=',',
        data_fields={"data": 0},
        index_mapping_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_paths: list of csv file paths to read data from
            newline_int: ASCII code to use to interpret newlines in file.
            header_lines: number of header lines in csv files.
            workers: number of workers to use for creating index files.
            tokenizer: tokenizer to use to convert text to tokens.
            sort_dataset_paths: whether to sort datasets by paths.
            data_sep: data separator.
            data_fields:  dict of field names and their corresponding column indices
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

        self._data_fields = data_fields
        self._data_sep = data_sep

    def _build_data_from_text(self, text: str):
        """

        """
        _build_data_from_text = super()._build_data_from_text
        data = {}
        text_fields = text.split(self._data_sep)
        for field_name, field_idx in self._data_fields.items():
            data[field_name] = _build_data_from_text(text_fields[field_idx])

        return data


class JSONLMemMapDataset(TextMemMapDataset):
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
            logging.error(f"Exception: {e}")
            logging.error(f"datapoint: {text}")
            raise e
        return record


def _index_file_exists(idx_fn):
    """Helper function to test if index file exists"""
    if os.path.exists(idx_fn + ".npy") and os.path.exists(idx_fn + ".info"):
        return True
    else:
        return False


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


def _build_memmap_index_files(newline_int, build_index_fn, fn, index_mapping_dir: str):
    """Helper function to build an index file"""
    idx_fn = _index_fn(fn, index_mapping_dir)

    # create data map
    if _index_file_exists(idx_fn):
        return False
    else:
        logging.info(f"Building indexing for fn = {fn}")
        # find all newline positions
        midx = build_index_fn(fn, newline_int)
        # validate midx
        midx = np.asarray(midx)
        if not np.issubdtype(midx.dtype, np.integer):
            raise TypeError(f"midx must be an integer array, but got type = {midx.dtype}")

        # create e metadata file
        data = dict(newline_int=newline_int, version=__idx_version__)

        # save index as numpy array to enable memmap reading
        logging.info(f"Saving idx file = {idx_fn}.npy")
        np.save(idx_fn + ".npy", midx, allow_pickle=True)
        logging.info(f"Saving metadata file = {idx_fn}.info")
        pickle.dump(data, open(idx_fn + ".info", "wb"))

        return True


def build_index_files(
    dataset_paths, newline_int, workers=None, build_index_fn=_build_index_from_memdata, index_mapping_dir: str = None,
):
    """Auxiliary method to build multiple index files"""
    if len(dataset_paths) < 1:
        raise ValueError("files_list must contain at leat one file name")

    if workers is None:
        workers = max(1, os.cpu_count() // 2)

    logging.info(f"Processing {len(dataset_paths)} data files using {workers} workers")
    # load all files into memmap
    start_time = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as p:
        build_status = p.map(
            partial(_build_memmap_index_files, newline_int, build_index_fn, index_mapping_dir=index_mapping_dir,),
            dataset_paths,
        )

    logging.info(
        f"Time building {sum(build_status)} / {len(build_status)} mem-mapped files: {datetime.timedelta(seconds=time.time() - start_time)}"
    )
