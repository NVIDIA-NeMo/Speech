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
import multiprocessing as mp
import os
import pickle
import time
from functools import partial

import numpy as np
import torch

from nemo.core import Dataset
from nemo.utils import logging

__all__ = ['TextMemMapDataset', 'CSVMemMapDataset', 'build_index_files']
__idx_version__ = '0.2'  # index file version
__idx_suffix__ = 'idx'  # index file suffix


class TextMemMapDataset(Dataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    """

    # FIXME: header_lines=0 by default
    def __init__(
        self, dataset_paths, newline_int=10, header_lines=0, workers=None, tokenizer=None, sort_dataset_paths=True,
    ):
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
        is_ditributed = torch.distributed.is_available() and torch.distributed.is_initialized()

        if not is_ditributed or (is_ditributed and torch.distributed.get_rank() == 0):
            build_index_files(dataset_paths, newline_int, workers=self._worker)

        if is_ditributed:
            torch.distributed.barrier()

        logging.info(f"Loading data files")
        start_time = time.time()
        mdata_midx_list = [self.load_file(fn) for fn in self._files_list]
        logging.info(
            f'Time loading {len(mdata_midx_list)} mem-mapped files: {datetime.timedelta(seconds=time.time() - start_time)}'
        )

        logging.info("Computing global indices")
        midx_bins = np.cumsum([(len(midx) - header_lines) for _, midx in mdata_midx_list])

        self.midx_bins = midx_bins
        self.mdata_midx_list = mdata_midx_list

    def __del__(self):
        if self.mdata_midx_list:
            for mdata, midx in self.mdata_midx_list:
                mdata._mmap.close()

    def __len__(self):
        return self.midx_bins[-1]

    def __getitem__(self, idx):
        """
        Return a string from binary memmap
        """
        if (idx >= self.midx_bins[-1]) or (idx < 0):
            raise IndexError(f"Index {idx} if out of dataset range with {self.midx_bins[-1]} samples")

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

        text = mdata[i:j].tobytes().decode("utf-8")

        # parse raw text (e.g., tokenize)
        data = self._build_data_from_text(text)

        return data

    def _build_data_from_text(self, text):
        """Allows child-classes to modify the parsing of raw text, prior to tokenization"""
        # tokenize text if tokenizer is given
        if self.tokenizer is not None:
            data = self.tokenizer.text_to_ids(text)
        else:
            data = text

        return data

    def load_file(self, fn):
        """
        Loads a text file as np.int8.

        Returns:
            mdata - memorymap of np.int8
            midx - indices pointing to the end-of-line (or end of file) position
            size - number of lines in file
        """
        logging.info(f"Loading {fn}")
        idx_fn = f"{fn}.{__idx_suffix__}"

        # create data map
        mdata = np.memmap(fn, dtype=np.uint8, mode='r')

        if os.path.exists(idx_fn + ".npy"):
            # load index file into memory map
            midx = np.load(idx_fn + ".npy", allow_pickle=True, mmap_mode='r')
            # test for header
            if len(midx) < self._header_lines:
                raise RuntimeError(f"Missing header, expected {self._header_lines} header lines")

            # load meta info
            idx_info_dict = pickle.load(open(idx_fn + ".info", 'rb'))
            # test for mismatch in expected newline_int
            if 'newline_int' in idx_info_dict:
                newline_int = idx_info_dict['newline_int']
                if self._newline_int != newline_int:
                    logging.warning(
                        f"Mismatch in newline_int, expected = {self._newline_int} but loaded {newline_int}"
                    )

            # test for version mismatch (useful to force recreation of index files)
            idx_version = idx_info_dict.get('version', '0.0')
            if __idx_version__ != idx_version:
                raise RuntimeError(
                    f"Version mismatch: Please delete existing '.{__idx_suffix__}' files. Expected version = {__idx_version__}, but file version = {idx_version}. File path = {idx_fn}"
                )
        else:
            raise ValueError(f'Memory Map for {fn} is not found')

        return (mdata, midx)


class CSVMemMapDataset(TextMemMapDataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    """

    def __init__(
        self,
        dataset_paths,
        newline_int=10,
        header_lines=1,
        workers=None,
        tokenizer=None,
        sort_dataset_paths=True,
        data_col=1,
        data_sep=',',
    ):
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=newline_int,
            header_lines=header_lines,
            workers=workers,
            tokenizer=tokenizer,
            sort_dataset_paths=sort_dataset_paths,
        )
        self._data_col = data_col
        self._data_sep = data_sep

    def _build_data_from_text(self, text):
        """Return a CSV field from text"""
        # get CSV field
        text = text.split(self._data_sep)[self._data_col]
        # tokenize
        return super()._build_data_from_text(text)


def _build_memmap_index_files(newline_int, fn):
    """Helper function to build an index file"""
    idx_fn = f"{fn}.{__idx_suffix__}"

    # create data map
    mdata = np.memmap(fn, dtype=np.uint8, mode='r')
    if os.path.exists(idx_fn + ".npy"):
        return False
    else:
        logging.info(f"Building idx file = {idx_fn}.npy")
        midx = np.where(mdata == newline_int)[0]
        midx_dtype = midx.dtype
        # add last item in case there is no new-line
        if (len(midx) == 0) or (midx[-1] + 1 != len(mdata)):
            midx = np.asarray(midx.tolist() + [len(midx) + 1], dtype=midx_dtype)

        # remove empty lines from end of file
        midx = midx.tolist()
        while len(midx) > 1 and (midx[-1] - midx[-2]) < 2:
            midx.pop(-1)
        midx = np.asarray(midx, dtype=midx_dtype)

        data = dict(newline_int=newline_int, version=__idx_version__)
        # save index as numpy array to enable memmap reading
        np.save(idx_fn + ".npy", midx, allow_pickle=True)
        pickle.dump(data, open(idx_fn + ".info", "wb"))
        mdata._mmap.close()
        del mdata

        return True


def build_index_files(dataset_paths, newline_int, workers=None):
    """Auxiliary method to build multiple index files"""
    if len(dataset_paths) < 1:
        raise ValueError("files_list must contain at leat one file name")

    if workers is None:
        workers = max(1, os.cpu_count() // 2)

    logging.info(f"Processing {len(dataset_paths)} data files using {workers} workers")
    # load all files into memmap
    start_time = time.time()
    with mp.Pool(workers) as p:
        build_status = p.map(partial(_build_memmap_index_files, newline_int), dataset_paths)

    logging.info(
        f'Time building {sum(build_status)} / {len(build_status)} mem-mapped files: {datetime.timedelta(seconds=time.time() - start_time)}'
    )
