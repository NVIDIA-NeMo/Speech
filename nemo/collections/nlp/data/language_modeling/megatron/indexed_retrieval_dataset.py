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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
from typing import List

import numpy as np
import torch

from nemo.utils import logging

__all__ = ["KNNIndex", "MMapRetrievalIndexedDataset", "MMapRetrievalIndexedDatasetBuilder"]


dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64, 7: np.double, 8: np.uint16}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class KNNIndex(object):
    """
    Index file for fast KNN mapping.
    It is built by `build_knn_map_index.py` script.
    It contains a big matrix of shape (chunk_id, K neighbors)
    where `chunk_id` are all the chunk ids in the RETRO training data.
    E.g. the KNN neighbor chunk ids in the retrieval data for ith chunk id in the training data
    is self.knn_map[i].
    This index can hold partial maps used for building sharding index.
    """

    _HDR_MAGIC = b'KNNRETM\x00\x00'

    @classmethod
    def writer(cls, path, K, offset=0):
        """
        path: file path of the index
        K: number of neighbors for a chunk
        offset: start chunk_id for shard index
        """

        class _Writer(object):
            def __enter__(self):
                self._file = open(path, 'wb')
                self._file.write(cls._HDR_MAGIC)
                self._file.write(struct.pack('<Q', 1))
                self._file.write(struct.pack('<Q', K))
                # reserve the space for total number of chunks
                self._file.write(struct.pack('<Q', 0))
                # chunk start
                self._file.write(struct.pack('<Q', offset))
                self.K = K
                self.count_chunks = 0
                self.path = path
                return self

            def write(self, chunk_knn: np.array):
                assert chunk_knn.dtype == np.int64
                assert chunk_knn.shape[1] == self.K
                self._file.write(chunk_knn.tobytes(order='C'))
                self.count_chunks += chunk_knn.shape[0]

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._file.close()
                # Update the chunk size, Since total number of chunks is determined in the end
                _bin_buffer_mmap = np.memmap(self.path, mode='r+', order='C', shape=(9 + 8 + 8 + 8),)
                buffer = memoryview(_bin_buffer_mmap)
                len_array = np.frombuffer(buffer, dtype=np.int64, count=1, offset=9 + 8 + 8)
                len_array[0] = self.count_chunks
                _bin_buffer_mmap.flush()
                _bin_buffer_mmap._mmap.close()

        return _Writer()

    def __init__(self, path, skip_warmup=True):
        with open(path, 'rb') as stream:
            magic_test = stream.read(9)
            assert self._HDR_MAGIC == magic_test, 'Index file doesn\'t match expected format. '
            version = struct.unpack('<Q', stream.read(8))
            assert (1,) == version

            self.K = struct.unpack('<Q', stream.read(8))[0]
            self.len = struct.unpack('<Q', stream.read(8))[0]
            self.chunk_start_id = struct.unpack('<Q', stream.read(8))[0]
            self.chunk_end_id = self.chunk_start_id + self.len
            offset = stream.tell()

        if not skip_warmup:
            logging.info("    warming up index mmap file...")
            _warmup_mmap_file(path)

        self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)
        logging.info("    reading KNN map")
        self.knn_map = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self.len * self.K, offset=offset).reshape(
            self.len, self.K
        )

    def get_KNN_chunk_ids(self, chunk_id):
        """ get the chunk address (in bytes) from chunk id
        """
        if not (self.chunk_start_id <= chunk_id < self.chunk_end_id):
            raise ValueError(f'chunk {chunk_id} is out side the range [{self.chunk_start_id}, {self.chunk_end_id})')
        return self.knn_map[chunk_id - self.chunk_start_id]

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap

    def __len__(self):
        """
        total number of chunks in the data
        """
        return self.len


def merge_knn_files(knn_files: List[KNNIndex], output_file: str):
    """
    Merge a list of knn sharding index files into one.
    """
    files = [KNNIndex(f) for f in knn_files]
    sorted_files = sorted(files, key=lambda x: x.chunk_start_id)
    # consistence check
    start_id = sorted_files[0].chunk_start_id
    previous_end = sorted_files[0].chunk_end_id
    K = sorted_files[0].K
    for i in sorted_files[1:]:
        assert previous_end == i.chunk_start_id
        assert K == i.K
        previous_end = i.chunk_end_id
    with KNNIndex.writer(output_file, K, offset=start_id) as w:
        for i in sorted_files:
            w.write(i.knn_map)
    f = KNNIndex(output_file)
    logging.info(f'{output_file} index starts at {f.chunk_start_id}')
    logging.info(f'{output_file} index ends at {f.chunk_end_id}')
    logging.info(f'total len {f.len}')
    assert f.len == (f.chunk_end_id - f.chunk_start_id)


class MMapRetrievalIndexedDataset(torch.utils.data.Dataset):
    """
    Memory Map Index and Binary file for RETRO DATA.
    It provides `chunks` to the original MMap data so data can be fetched at both document and chunk level.
    It can be used both for training data and Retrieval Data.
    Retrieval Dataset adds an extra `chunk_size` padded tokens at the end of each document. '
    `self._index.retrieval_db` is indicating whether it is retrieval dataset or not.

    It is built by `preprocess_data_for_megatron.py` script.

    """

    class Index(object):
        _HDR_MAGIC = b'MMIDRET\x00\x00'

        @classmethod
        def writer(cls, path, dtype, retrieval_db):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, 'wb')

                    self._file.write(cls._HDR_MAGIC)
                    # write index file version
                    self._file.write(struct.pack('<L', 1))
                    return self

                @staticmethod
                def _get_pointers(sizes, chunk_size):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size
                        if retrieval_db:
                            # if it is retrieval db, the the last chunk is reserved for padding
                            address += chunk_size * dtype_size
                    return pointers

                @staticmethod
                def _get_chunk_id_and_address(sizes, chunk_size, stride):
                    if chunk_size % stride != 0:
                        raise ValueError(f"the chunk size {chunk_size} should be the multiple of {stride}")
                    dtype_size = dtype().itemsize
                    chunk_ids = []
                    last_id = 0
                    address = 0
                    pointers = []
                    for size in sizes:
                        chunk_ids.append(last_id)
                        num_of_chunks = len(range(0, size - chunk_size + 1, stride))
                        if size % chunk_size != 0:
                            raise ValueError(f"the document size {size} should be the multiple of {chunk_size}")
                        for i in range(0, size - chunk_size + 1, stride):
                            pointers.append(address)
                            if i == size - chunk_size:
                                address += chunk_size * dtype_size
                            else:
                                address += stride * dtype_size
                        if retrieval_db:
                            # if it is retrieval db, the the last chunk is reserved for padding
                            address += chunk_size * dtype_size
                        last_id += num_of_chunks
                    return chunk_ids, pointers

                def write(self, sizes, chunk_size, stride=64):
                    pointers = self._get_pointers(sizes, chunk_size)
                    chunk_ids, chunk_address = self._get_chunk_id_and_address(sizes, chunk_size, stride)
                    # write index chunk stride step
                    self._file.write(struct.pack('<L', stride))
                    self._file.write(struct.pack('<B', code(dtype)))

                    self._file.write(struct.pack('<Q', len(sizes)))
                    self._file.write(struct.pack('<Q', chunk_size))
                    self._file.write(struct.pack('<Q', len(chunk_address)))
                    self._file.write(struct.pack('<B', int(retrieval_db)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order='C'))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers

                    chunk_ids = np.array(chunk_ids, dtype=np.int64)
                    self._file.write(chunk_ids.tobytes(order='C'))
                    del chunk_ids

                    chunk_address = np.array(chunk_address, dtype=np.int64)
                    self._file.write(chunk_address.tobytes(order='C'))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=True):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<L', stream.read(4))
                assert (1,) == version
                # load the stride size
                (self.stride,) = struct.unpack('<L', stream.read(4))
                # for legacy compatibility
                if self.stride == 0:
                    self.stride = 64

                (dtype_code,) = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self.chunk_size = struct.unpack('<Q', stream.read(8))[0]
                self.num_chunks = struct.unpack('<Q', stream.read(8))[0]
                self.retrieval_db = bool(struct.unpack('<B', stream.read(1))[0])
                # self.chunk_size = struct.unpack('<Q', stream.read(8))[0]
                # self.num_chunks = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                logging.info("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            logging.info("    reading document sizes...")
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            logging.info("    reading document pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes
            )
            logging.info("    reading document chunk offset...")
            self._chunk_id_start = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )
            logging.info("    reading chunk address...")
            self._chunk_address = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self.num_chunks,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes + self._chunk_id_start.nbytes,
            )

        def get_chunk_address(self, chunk_id):
            """ get the chunk address from chunk id
            """
            return self._chunk_address[chunk_id]

        def get_chunk_id(self, sentence_id, position):
            """ get the chunk id from sentence idx and offset position.
            """
            chunk_offset = position // self.stride
            size = self._sizes[sentence_id]
            if chunk_offset * self.stride >= size:
                raise ValueError('offset is too large')
            return (self._chunk_id_start[sentence_id] + chunk_offset).item()

        def from_chunk_id_to_doc_id(self, chunk_id):
            """ from chunk_id, calculate the document id
            """
            if chunk_id >= self.num_chunks:
                raise ValueError('chunk_id is out of bound')
            doc_id = np.searchsorted(self._chunk_id_start, chunk_id, side='right')
            return doc_id - 1

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            """
            Token id integer type
            """
            return self._dtype

        @property
        def sizes(self):
            """
            number of tokens for each of the documents
            """
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            """
            return a single document staring address (in bytes) and number of tokens
            """
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=True):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    # def __setstate__(self, state):
    #     self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            logging.info("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        logging.info("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        logging.info("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        """
        Total number of documents
        """
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        """
        return a single document or a slice of documents, excluding the paddings for the retrieval db
        """
        if isinstance(idx, int):
            # no need to handle retrieval_db since size exclude the paddings
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            if self._index.retrieval_db:
                # for retrieval db, need to add the padding of chunk_size at the end of each document
                sizes = self._index._sizes[idx] + self._index.chunk_size
            else:
                sizes = self._index._sizes[idx]
            # offsets get the number of tokens for each document including the paddings
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            if self._index.retrieval_db:
                # remove the paddings
                sents = [sent[: -self._index.chunk_size] for sent in sents]
            return sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        # no need to handle retrieval_db since size exclude the paddings
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)
        return np_array

    def get_chunk_id(self, idx, offset=0):
        """ get the chunk id from document idx and offset position.
        """
        # make sure offset is a multiple of chunk_size
        assert offset % self._index.chunk_size == 0
        return self._index.get_chunk_id(idx, offset)

    def from_chunk_id_to_doc_id(self, chunk_id):
        """ from chunk_id, calculate the document id
        """
        return self._index.from_chunk_id_to_doc_id(chunk_id)

    def get_chunk(self, chunk_id, force_no_cont_ids=False):
        """ Retrieves a single chunk item from the dataset.
        It will get chunk_size tokens for training data
        or 2*chunk_size tokens for retrieval data.
        If force_no_cont_ids=True, it will always get chunk_size tokens
        """
        if isinstance(chunk_id, (int, np.int64, np.int32)):
            ptr = self._index.get_chunk_address(chunk_id)
            if self._index.retrieval_db and (not force_no_cont_ids):
                size = self._index.chunk_size * 2
            else:
                size = self._index.chunk_size
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(chunk_id, slice):
            start, stop, step = chunk_id.indices(self.chunks)
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            if self._index.retrieval_db and (not force_no_cont_ids):
                chunk_size = self._index.chunk_size * 2
            else:
                chunk_size = self._index.chunk_size
            ptr = self._index.get_chunk_address(start)
            end_address = self._index.get_chunk_address(stop - 1) + chunk_size * self._index._dtype_size
            address = self._index._chunk_address[chunk_id]
            starting_pos = address // self._index._dtype_size
            total_size = (end_address - ptr) // self._index._dtype_size
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = [np_array[pos : pos + chunk_size] for pos in starting_pos - starting_pos[0]]
            return sents

    @property
    def sizes(self):
        """
        Number of tokens for each of the documents
        """
        return self._index.sizes

    @property
    def chunks(self):
        """
        Total number of chunks
        """
        return self._index.num_chunks

    @property
    def chunk_size(self):
        """
        Number of tokens per chunk
        """
        return self._index.chunk_size

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))


class MMapRetrievalIndexedDatasetBuilder(object):
    def __init__(self, out_file, chunk_size, pad_id, retrieval_db=False, dtype=np.int64, stride=64):
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self.chunk_size = chunk_size
        self._sizes = []
        self.retrieval_db = retrieval_db
        self.pad_id = pad_id
        self.stride = stride

    def add_item(self, tensor):
        """
        Add one document to the indexed dataset.
        It will pad the tokens to be the multiple of chunk_size.
        If it is retrieval dataset, it will pad extra chunk_size tokens at the end of the document.
        """
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        padded_size = self.chunk_size - (len(np_array) % self.chunk_size)
        data_size = np_array.size + padded_size
        if self.retrieval_db:
            # for retrieval database, added one more chunk in the end as padding
            padded_size += self.chunk_size
        np_array = np.pad(np_array, (0, padded_size), 'constant', constant_values=self.pad_id)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(data_size)

    def end_document(self):
        """
        Do nothing. Since each item is one document
        """
        pass

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapRetrievalIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        """
        Last step of creating the indexed dataset.
        Flush and close the binary file.
        Finalizing the index file by using the aggregated document size information.
        """
        self._data_file.close()

        with MMapRetrievalIndexedDataset.Index.writer(index_file, self._dtype, self.retrieval_db) as index:
            index.write(self._sizes, self.chunk_size, stride=self.stride)
