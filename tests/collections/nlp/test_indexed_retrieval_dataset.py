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


import os

import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal
from omegaconf import OmegaConf
from scripts.nlp_language_modeling.build_knn_map_index import build_map, dedup

from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import (
    KNNIndex,
    MMapRetrievalIndexedDataset,
    MMapRetrievalIndexedDatasetBuilder,
    merge_knn_files,
)
from nemo.collections.nlp.data.language_modeling.megatron.retro_dataset_legacy import RETRODataset

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not HAVE_MEGATRON_CORE, reason="megatron-core is not installed")
class TestRetrievalIndexFiles:
    @classmethod
    def setup_class(cls):
        init_method = 'tcp://'
        master_ip = 'localhost'
        master_port = '6000'
        init_method += master_ip + ':' + master_port
        torch.distributed.init_process_group(backend='gloo', world_size=1, rank=0, init_method=init_method)
        parallel_state.initialize_model_parallel(1, 1)

    @pytest.mark.unit
    def test_index(self):
        chunk_size = 64
        stride = 32
        sizes = np.array([128, 256], dtype=np.int32)
        dtype = np.int64
        itemsize = dtype().itemsize
        index_file = '/tmp/test.idx'
        try:
            with MMapRetrievalIndexedDataset.Index.writer(index_file, dtype, False) as index:
                index.write(sizes, chunk_size, stride=stride)

            index_load = MMapRetrievalIndexedDataset.Index(index_file)
            assert index_load.chunk_size == chunk_size
            assert not index_load.retrieval_db
            assert np.array_equal(index_load.sizes, sizes)
            assert np.array_equal(
                index_load._chunk_id_start,
                np.array([0, len(range(0, sizes[0] - chunk_size + 1, stride))], dtype=np.int64),
            )
            add1 = [i * itemsize for i in list(range(0, sizes[0] - chunk_size + 1, stride))]
            start = max(add1) + chunk_size * itemsize
            add2 = [i * itemsize + start for i in list(range(0, sizes[1] - chunk_size + 1, stride))]
            addr = add1 + add2
            assert np.array_equal(index_load._chunk_address, np.array(addr, dtype=np.int64))
            assert np.array_equal(index_load._pointers, np.array([0, sizes[0] * itemsize], dtype=np.int64))
            assert len(index_load._chunk_address) == index_load.num_chunks
        finally:
            os.remove(index_file)

        chunk_size = 64
        stride = 64
        sizes = np.array([128, 256], dtype=np.int32)
        dtype = np.int64
        itemsize = dtype().itemsize
        index_file = '/tmp/test.idx'
        try:
            with MMapRetrievalIndexedDataset.Index.writer(index_file, dtype, False) as index:
                index.write(sizes, chunk_size, stride=stride)

            index_load = MMapRetrievalIndexedDataset.Index(index_file)
            assert index_load.chunk_size == chunk_size
            assert not index_load.retrieval_db
            assert np.array_equal(index_load.sizes, sizes)
            assert np.array_equal(
                index_load._chunk_id_start,
                np.array([0, len(range(0, sizes[0] - chunk_size + 1, stride))], dtype=np.int64),
            )
            add1 = [i * itemsize for i in list(range(0, sizes[0] - chunk_size + 1, stride))]
            start = max(add1) + chunk_size * itemsize
            add2 = [i * itemsize + start for i in list(range(0, sizes[1] - chunk_size + 1, stride))]
            addr = add1 + add2
            assert np.array_equal(index_load._chunk_address, np.array(addr, dtype=np.int64))
            assert np.array_equal(index_load._pointers, np.array([0, sizes[0] * itemsize], dtype=np.int64))
            assert len(index_load._chunk_address) == index_load.num_chunks
        finally:
            os.remove(index_file)

    @pytest.mark.unit
    def test_create_data_index_stride32(self):
        chunk_size = 64
        pad_id = 0
        stride = 32
        sentence1 = torch.arange(0, 200, 2, dtype=torch.int64)
        padded_size = chunk_size - (len(sentence1) % chunk_size)
        gt1 = np.pad(sentence1, (0, padded_size), 'constant', constant_values=pad_id)

        sentence2 = torch.arange(1, 500, 2, dtype=torch.int64)
        padded_size = chunk_size - (len(sentence2) % chunk_size)
        gt2 = np.pad(sentence2, (0, padded_size), 'constant', constant_values=pad_id)

        data_file = '/tmp/test'
        index_file = data_file + '.idx'
        bin_file = data_file + '.bin'
        try:
            builder = MMapRetrievalIndexedDatasetBuilder(
                bin_file, chunk_size, pad_id, False, dtype=np.int64, stride=stride
            )
            builder.add_item(sentence1)
            builder.add_item(sentence2)
            builder.finalize(index_file)
            # load the data
            ds = MMapRetrievalIndexedDataset(data_file)
            assert np.array_equal(ds.get(0), gt1)
            assert np.array_equal(ds.get(1), gt2)
            fetch1, fetch2 = ds[0:2]
            assert np.array_equal(fetch1, gt1)
            assert np.array_equal(fetch2, gt2)
            chunk_id = ds.get_chunk_id(0, 64)
            assert chunk_id == 2
            assert ds.from_chunk_id_to_doc_id(0) == 0
            assert ds.from_chunk_id_to_doc_id(1) == 0
            assert ds.from_chunk_id_to_doc_id(2) == 0
            with pytest.raises(ValueError):
                ds.get_chunk_id(0, 128)
            assert np.array_equal(ds.get_chunk(chunk_id), gt1[64 : 64 + 64])
            chunk_id = ds.get_chunk_id(1, 0)
            assert chunk_id == 3
            assert ds.from_chunk_id_to_doc_id(3) == 1
            assert ds.from_chunk_id_to_doc_id(4) == 1
            assert ds.from_chunk_id_to_doc_id(5) == 1
            assert ds.from_chunk_id_to_doc_id(6) == 1
            assert ds.from_chunk_id_to_doc_id(7) == 1
            assert ds.from_chunk_id_to_doc_id(8) == 1
            assert ds.from_chunk_id_to_doc_id(9) == 1
            with pytest.raises(ValueError):
                ds.from_chunk_id_to_doc_id(10)
            assert np.array_equal(ds.get_chunk(chunk_id), gt2[0:64])
            assert np.array_equal(ds.get_chunk(chunk_id + 1), gt2[stride : stride + chunk_size])
            assert np.array_equal(ds.get_chunk(chunk_id + 2), gt2[stride * 2 : stride * 2 + chunk_size])
            assert np.array_equal(ds.get_chunk(chunk_id + 3), gt2[stride * 3 : stride * 3 + chunk_size])
            assert ds.get_chunk_id(1, 64) == 5
            assert ds.get_chunk_id(1, 128) == 7
            assert ds.get_chunk_id(1, 192) == 9
            with pytest.raises(ValueError):
                ds.get_chunk_id(0, 256)
        finally:
            os.remove(index_file)
            os.remove(bin_file)

    @pytest.mark.unit
    def test_create_data_index(self):
        chunk_size = 64
        pad_id = 0
        sentence1 = torch.arange(0, 200, 2, dtype=torch.int64)
        padded_size = chunk_size - (len(sentence1) % chunk_size)
        gt1 = np.pad(sentence1, (0, padded_size), 'constant', constant_values=pad_id)

        sentence2 = torch.arange(1, 500, 2, dtype=torch.int64)
        padded_size = chunk_size - (len(sentence2) % chunk_size)
        gt2 = np.pad(sentence2, (0, padded_size), 'constant', constant_values=pad_id)

        data_file = '/tmp/test'
        index_file = data_file + '.idx'
        bin_file = data_file + '.bin'
        try:
            builder = MMapRetrievalIndexedDatasetBuilder(bin_file, chunk_size, pad_id, False)
            builder.add_item(sentence1)
            builder.add_item(sentence2)
            builder.finalize(index_file)
            # load the data
            ds = MMapRetrievalIndexedDataset(data_file)
            assert np.array_equal(ds.get(0), gt1)
            assert np.array_equal(ds.get(1), gt2)
            fetch1, fetch2 = ds[0:2]
            assert np.array_equal(fetch1, gt1)
            assert np.array_equal(fetch2, gt2)
            chunk_id = ds.get_chunk_id(0, 64)
            assert chunk_id == 1
            assert ds.from_chunk_id_to_doc_id(0) == 0
            assert ds.from_chunk_id_to_doc_id(1) == 0
            with pytest.raises(ValueError):
                ds.get_chunk_id(0, 128)
            assert np.array_equal(ds.get_chunk(chunk_id), gt1[64 : 64 + 64])
            chunk_id = ds.get_chunk_id(1, 0)
            assert chunk_id == 2
            assert ds.from_chunk_id_to_doc_id(2) == 1
            assert ds.from_chunk_id_to_doc_id(3) == 1
            assert ds.from_chunk_id_to_doc_id(4) == 1
            assert ds.from_chunk_id_to_doc_id(5) == 1
            with pytest.raises(ValueError):
                ds.from_chunk_id_to_doc_id(6)
            assert np.array_equal(ds.get_chunk(chunk_id), gt2[0:64])
            assert np.array_equal(ds.get_chunk(chunk_id + 1), gt2[64:128])
            assert np.array_equal(ds.get_chunk(chunk_id + 2), gt2[128:192])
            assert np.array_equal(ds.get_chunk(chunk_id + 3), gt2[192:256])
            assert ds.get_chunk_id(1, 64) == 3
            assert ds.get_chunk_id(1, 128) == 4
            assert ds.get_chunk_id(1, 192) == 5
            with pytest.raises(ValueError):
                ds.get_chunk_id(0, 256)
        finally:
            os.remove(index_file)
            os.remove(bin_file)

    @pytest.mark.unit
    def test_create_retrieval_data_index_stride32(self):
        stride = 32
        chunk_size = 64
        pad_id = 0
        sentence1 = torch.arange(0, 200, 2, dtype=torch.int64)
        padded_size = chunk_size - (len(sentence1) % chunk_size)
        gt1 = np.pad(sentence1, (0, padded_size), 'constant', constant_values=pad_id)
        padded_gt1 = np.pad(sentence1, (0, padded_size + chunk_size), 'constant', constant_values=pad_id)

        sentence2 = torch.arange(1, 500, 2, dtype=torch.int64)
        padded_size = chunk_size - (len(sentence2) % chunk_size)
        gt2 = np.pad(sentence2, (0, padded_size), 'constant', constant_values=pad_id)
        padded_gt2 = np.pad(sentence2, (0, padded_size + chunk_size), 'constant', constant_values=pad_id)

        data_file = '/tmp/test'
        index_file = data_file + '.idx'
        bin_file = data_file + '.bin'
        try:
            builder = MMapRetrievalIndexedDatasetBuilder(bin_file, chunk_size, pad_id, True, stride=stride)
            builder.add_item(sentence1)
            builder.add_item(sentence2)
            builder.finalize(index_file)
            # load the data
            ds = MMapRetrievalIndexedDataset(data_file)
            assert np.array_equal(ds.get(0), gt1)
            assert np.array_equal(ds.get(1), gt2)
            fetch1, fetch2 = ds[0:2]
            assert np.array_equal(fetch1, gt1)
            assert np.array_equal(fetch2, gt2)
            chunk_id = ds.get_chunk_id(0, 64)
            assert chunk_id == 2
            assert ds.from_chunk_id_to_doc_id(0) == 0
            assert ds.from_chunk_id_to_doc_id(1) == 0
            assert ds.from_chunk_id_to_doc_id(2) == 0
            with pytest.raises(ValueError):
                ds.get_chunk_id(0, 128)
            assert np.array_equal(ds.get_chunk(chunk_id), padded_gt1[64 : 64 + 64 * 2])
            chunk_id = ds.get_chunk_id(1, 0)
            assert chunk_id == 3
            assert ds.from_chunk_id_to_doc_id(3) == 1
            assert ds.from_chunk_id_to_doc_id(4) == 1
            assert ds.from_chunk_id_to_doc_id(5) == 1
            assert ds.from_chunk_id_to_doc_id(6) == 1
            assert ds.from_chunk_id_to_doc_id(7) == 1
            assert ds.from_chunk_id_to_doc_id(8) == 1
            assert ds.from_chunk_id_to_doc_id(9) == 1
            with pytest.raises(ValueError):
                ds.from_chunk_id_to_doc_id(10)
            assert np.array_equal(ds.get_chunk(chunk_id), padded_gt2[0 : chunk_size * 2])
            assert np.array_equal(ds.get_chunk(chunk_id + 1), gt2[stride : stride + chunk_size * 2])
            assert np.array_equal(ds.get_chunk(chunk_id + 2), gt2[stride * 2 : stride * 2 + chunk_size * 2])
            assert np.array_equal(ds.get_chunk(chunk_id + 3), gt2[stride * 3 : stride * 3 + chunk_size * 2])
            assert ds.get_chunk_id(1, 64) == 5
            assert ds.get_chunk_id(1, 128) == 7
            assert ds.get_chunk_id(1, 192) == 9
            with pytest.raises(ValueError):
                ds.get_chunk_id(0, 256)
            chunk_id = ds.get_chunk_id(1, 64)
            assert np.array_equal(ds.get_chunk(chunk_id), padded_gt2[64:192])
            multi_chunks = ds.get_chunk(slice(0, ds.chunks))
            assert np.array_equal(multi_chunks[0], padded_gt1[0 : chunk_size * 2])
            assert np.array_equal(multi_chunks[1], padded_gt1[stride : stride + chunk_size * 2])
            assert np.array_equal(multi_chunks[2], padded_gt1[stride * 2 : stride * 2 + chunk_size * 2])
            assert np.array_equal(multi_chunks[3], padded_gt2[0 : chunk_size * 2])
            assert np.array_equal(multi_chunks[4], padded_gt2[stride : stride + chunk_size * 2])
            assert np.array_equal(multi_chunks[5], padded_gt2[stride * 2 : stride * 2 + chunk_size * 2])
            assert np.array_equal(multi_chunks[6], padded_gt2[stride * 3 : stride * 3 + chunk_size * 2])
            assert np.array_equal(multi_chunks[7], padded_gt2[stride * 4 : stride * 4 + chunk_size * 2])
            assert np.array_equal(multi_chunks[8], padded_gt2[stride * 5 : stride * 5 + chunk_size * 2])
            assert np.array_equal(multi_chunks[9], padded_gt2[stride * 6 : stride * 6 + chunk_size * 2])
        finally:
            os.remove(index_file)
            os.remove(bin_file)

    @pytest.mark.unit
    def test_create_retrieval_data_index(self):

        chunk_size = 64
        pad_id = 0
        sentence1 = torch.arange(0, 200, 2, dtype=torch.int64)
        padded_size = chunk_size - (len(sentence1) % chunk_size)
        gt1 = np.pad(sentence1, (0, padded_size), 'constant', constant_values=pad_id)
        padded_gt1 = np.pad(sentence1, (0, padded_size + chunk_size), 'constant', constant_values=pad_id)

        sentence2 = torch.arange(1, 500, 2, dtype=torch.int64)
        padded_size = chunk_size - (len(sentence2) % chunk_size)
        gt2 = np.pad(sentence2, (0, padded_size), 'constant', constant_values=pad_id)
        padded_gt2 = np.pad(sentence2, (0, padded_size + chunk_size), 'constant', constant_values=pad_id)

        data_file = '/tmp/test'
        index_file = data_file + '.idx'
        bin_file = data_file + '.bin'
        try:
            builder = MMapRetrievalIndexedDatasetBuilder(bin_file, chunk_size, pad_id, True)
            builder.add_item(sentence1)
            builder.add_item(sentence2)
            builder.finalize(index_file)
            # load the data
            ds = MMapRetrievalIndexedDataset(data_file)
            assert np.array_equal(ds.get(0), gt1)
            assert np.array_equal(ds.get(1), gt2)
            fetch1, fetch2 = ds[0:2]
            assert np.array_equal(fetch1, gt1)
            assert np.array_equal(fetch2, gt2)
            chunk_id = ds.get_chunk_id(0, 64)
            assert chunk_id == 1
            assert ds.from_chunk_id_to_doc_id(0) == 0
            assert ds.from_chunk_id_to_doc_id(1) == 0
            with pytest.raises(ValueError):
                ds.get_chunk_id(0, 128)
            assert np.array_equal(ds.get_chunk(chunk_id), padded_gt1[64 : 64 + 64 * 2])
            chunk_id = ds.get_chunk_id(1, 0)
            assert chunk_id == 2
            assert ds.from_chunk_id_to_doc_id(2) == 1
            assert ds.from_chunk_id_to_doc_id(3) == 1
            assert ds.from_chunk_id_to_doc_id(4) == 1
            assert ds.from_chunk_id_to_doc_id(5) == 1
            with pytest.raises(ValueError):
                ds.from_chunk_id_to_doc_id(6)
            assert np.array_equal(ds.get_chunk(chunk_id), padded_gt2[0:128])
            assert np.array_equal(ds.get_chunk(chunk_id + 1), padded_gt2[64:192])
            assert np.array_equal(ds.get_chunk(chunk_id + 2), padded_gt2[128:256])
            assert np.array_equal(ds.get_chunk(chunk_id + 3), padded_gt2[192:320])
            assert ds.get_chunk_id(1, 64) == 3
            assert ds.get_chunk_id(1, 128) == 4
            assert ds.get_chunk_id(1, 192) == 5
            with pytest.raises(ValueError):
                ds.get_chunk_id(0, 256)
            chunk_id = ds.get_chunk_id(1, 64)
            assert np.array_equal(ds.get_chunk(chunk_id), padded_gt2[64:192])
            multi_chunks = ds.get_chunk(slice(0, ds.chunks))
            assert np.array_equal(multi_chunks[0], padded_gt1[0:128])
            assert np.array_equal(multi_chunks[1], padded_gt1[64 : 64 + 128])
            assert np.array_equal(multi_chunks[2], padded_gt2[0:128])
            assert np.array_equal(multi_chunks[3], padded_gt2[64 : 64 + 128])
            assert np.array_equal(multi_chunks[4], padded_gt2[128 : 128 + 128])
            assert np.array_equal(multi_chunks[5], padded_gt2[192 : 192 + 128])
        finally:
            os.remove(index_file)
            os.remove(bin_file)

    @pytest.mark.unit
    def test_knn_index(self):
        data_file = '/tmp/test'
        index_file = data_file + '.idx'
        K = 8
        index_files = [f'{data_file}_{i}.idx' for i in range(3)]
        merged_file = '/tmp/merged.idx'
        try:
            with KNNIndex.writer(index_file, K) as w:
                map_np0 = np.random.randint(0, 100, (50, K))
                w.write(map_np0)
                map_np1 = np.random.randint(0, 100, (50, K))
                w.write(map_np1)
                map_np2 = np.random.randint(0, 100, (50, K))
                w.write(map_np2)
            f = KNNIndex(index_file)
            assert f.K == K
            assert f.len == map_np0.shape[0] + map_np1.shape[0] + map_np2.shape[0]
            assert np.array_equal(map_np0, f.knn_map[:50])
            assert np.array_equal(map_np1, f.knn_map[50:100])
            assert np.array_equal(map_np2, f.knn_map[100:])
            assert np.array_equal(f.get_KNN_chunk_ids(5), map_np0[5])
            assert f.chunk_start_id == 0
            assert f.chunk_end_id == f.len

            with KNNIndex.writer(index_file, K, 100) as w:
                map_np0 = np.random.randint(0, 100, (50, K))
                w.write(map_np0)
                map_np1 = np.random.randint(0, 100, (50, K))
                w.write(map_np1)
                map_np2 = np.random.randint(0, 100, (50, K))
                w.write(map_np2)
            f = KNNIndex(index_file)
            assert f.K == K
            assert f.len == map_np0.shape[0] + map_np1.shape[0] + map_np2.shape[0]
            assert np.array_equal(map_np0, f.knn_map[:50])
            assert np.array_equal(map_np1, f.knn_map[50:100])
            assert np.array_equal(map_np2, f.knn_map[100:])
            assert np.array_equal(f.get_KNN_chunk_ids(5 + 100), map_np0[5])
            assert f.chunk_start_id == 100
            assert f.chunk_end_id == f.len + 100

            # test multiple sharding indices
            inputs = []
            start = 0
            for i in range(3):
                with KNNIndex.writer(index_files[i], K, offset=start) as w:
                    map_np0 = np.random.randint(0, 100, (50, K))
                    inputs.append(map_np0)
                    w.write(map_np0)
                    map_np1 = np.random.randint(0, 100, (50, K))
                    inputs.append(map_np1)
                    w.write(map_np1)
                f = KNNIndex(index_files[i])
                start += f.len
            merge_knn_files(index_files, merged_file)
            f = KNNIndex(merged_file)
            input_array = np.vstack(inputs)
            assert f.len == 100 * 3
            for i in range(300):
                assert np.array_equal(f.get_KNN_chunk_ids(i), input_array[i])
            assert f.chunk_start_id == 0
            assert f.chunk_end_id == f.len
            assert f.K == K

        finally:
            os.remove(index_file)
            for i in range(3):
                os.remove(index_files[i])
            os.remove(merged_file)

    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_MEGATRON_CORE, reason="megatron-core is not installed")
    def test_retro_dataset(self):

        chunk_size = 64
        pad_id = 0
        sentence1 = torch.arange(0, 200, 2, dtype=torch.int64)
        sentence2 = torch.arange(1, 500, 2, dtype=torch.int64)
        sentence3 = torch.arange(0, 300, 2, dtype=torch.int64)
        sentence4 = torch.arange(1, 400, 2, dtype=torch.int64)

        # test the case that
        # training data and retrieval data are different

        data_file = '/tmp/test_data'
        data_index_file = data_file + '.idx'
        data_bin_file = data_file + '.bin'
        db_file = '/tmp/test_db_data'
        db_index_file = db_file + '.idx'
        db_bin_file = db_file + '.bin'
        K = 8
        map_index_file = '/tmp/test_map.idx'
        index_path = '/tmp'

        cfg = OmegaConf.create({'data': {"index_mapping_dir": index_path}})

        # dummy tokenizer
        class Tokenizer:
            eos_id = 1
            pad_id = 0

        tokenizer = Tokenizer()

        num_samples = 100
        seq_len = 192
        name = 'test'
        data_prefix = 'pref'
        seed = 1
        _filename = index_path + '/' + data_prefix
        _filename += '_{}_indexmap'.format(name)
        _filename += '_{}ns'.format(num_samples)
        _filename += '_{}sl'.format(seq_len)
        _filename += '_{}s'.format(seed)
        doc_idx_filename = _filename + '_doc_idx.npy'
        sample_idx_filename = _filename + '_sample_idx.npy'
        shuffle_idx_filename = _filename + '_shuffle_idx.npy'

        try:
            builder = MMapRetrievalIndexedDatasetBuilder(data_bin_file, chunk_size, pad_id, False)
            builder.add_item(sentence1)
            builder.add_item(sentence2)
            builder.finalize(data_index_file)

            builder = MMapRetrievalIndexedDatasetBuilder(db_bin_file, chunk_size, pad_id, True)
            builder.add_item(sentence3)
            builder.add_item(sentence4)
            builder.finalize(db_index_file)

            # load the data
            data_index = MMapRetrievalIndexedDataset(data_file)
            db_index = MMapRetrievalIndexedDataset(db_file)

            with KNNIndex.writer(map_index_file, K) as w:
                map_np = np.random.randint(-3, db_index.chunks, (data_index.chunks, K))
                w.write(map_np)
            map_index = KNNIndex(map_index_file)

            documents = np.arange(0, data_index.sizes.shape[0])
            d = RETRODataset(
                cfg,
                None,
                tokenizer,
                name,
                data_prefix,
                documents,
                data_index,
                num_samples,
                seq_len,
                seed,
                map_index,
                db_index,
            )
            for i in range(len(d)):
                record = d[i]
                assert record['tokens'].shape[0] == seq_len
                assert record['labels'].shape[0] == seq_len
                assert record['retrieved_ids'].shape[0] == seq_len // chunk_size
                assert record['retrieved_ids'].shape[1] == K
                assert record['retrieved_ids'].shape[2] == chunk_size * 2
                assert record['tokens_mask'].shape[0] == seq_len

        finally:
            os.remove(data_bin_file)
            os.remove(data_index_file)
            os.remove(db_bin_file)
            os.remove(db_index_file)
            os.remove(map_index_file)
            os.remove(doc_idx_filename)
            os.remove(sample_idx_filename)
            os.remove(shuffle_idx_filename)

        # test the case that
        # training data and retrieval data are the same

        try:

            builder = MMapRetrievalIndexedDatasetBuilder(db_bin_file, chunk_size, pad_id, True)
            builder.add_item(sentence1)
            builder.add_item(sentence2)
            builder.add_item(sentence3)
            builder.add_item(sentence4)
            builder.finalize(db_index_file)

            # load the data
            data_index = MMapRetrievalIndexedDataset(db_file)
            db_index = MMapRetrievalIndexedDataset(db_file)

            with KNNIndex.writer(map_index_file, K) as w:
                map_np = np.random.randint(-3, db_index.chunks, (data_index.chunks, K))
                w.write(map_np)
            map_index = KNNIndex(map_index_file)

            documents = np.arange(0, data_index.sizes.shape[0])
            d = RETRODataset(
                cfg,
                None,
                tokenizer,
                name,
                data_prefix,
                documents,
                data_index,
                num_samples,
                seq_len,
                seed,
                map_index,
                db_index,
            )
            for i in range(len(d)):
                record = d[i]
                assert record['tokens'].shape[0] == seq_len
                assert record['labels'].shape[0] == seq_len
                assert record['retrieved_ids'].shape[0] == seq_len // chunk_size
                assert record['retrieved_ids'].shape[1] == K
                assert record['retrieved_ids'].shape[2] == chunk_size * 2
                assert record['tokens_mask'].shape[0] == seq_len

        finally:
            os.remove(db_bin_file)
            os.remove(db_index_file)
            os.remove(map_index_file)
            os.remove(doc_idx_filename)
            os.remove(sample_idx_filename)
            os.remove(shuffle_idx_filename)

    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_MEGATRON_CORE, reason="megatron-core is not installed")
    def test_retro_dataset_stride32(self):
        chunk_size = 64
        pad_id = 0
        sentence1 = torch.arange(0, 200, 2, dtype=torch.int64)
        sentence2 = torch.arange(1, 500, 2, dtype=torch.int64)
        sentence3 = torch.arange(0, 300, 2, dtype=torch.int64)
        sentence4 = torch.arange(1, 400, 2, dtype=torch.int64)

        # test the case that
        # training data and retrieval data are different

        data_file = '/tmp/test_data'
        data_index_file = data_file + '.idx'
        data_bin_file = data_file + '.bin'
        db_file = '/tmp/test_db_data'
        db_index_file = db_file + '.idx'
        db_bin_file = db_file + '.bin'
        K = 8
        map_index_file = '/tmp/test_map.idx'
        index_path = '/tmp'

        cfg = OmegaConf.create({'data': {"index_mapping_dir": index_path}})

        # dummy tokenizer
        class Tokenizer:
            eos_id = 1
            pad_id = 0

        tokenizer = Tokenizer()

        num_samples = 100
        stride = 32
        seq_len = 192
        name = 'test'
        data_prefix = 'pref'
        seed = 1
        _filename = index_path + '/' + data_prefix
        _filename += '_{}_indexmap'.format(name)
        _filename += '_{}ns'.format(num_samples)
        _filename += '_{}sl'.format(seq_len)
        _filename += '_{}s'.format(seed)
        doc_idx_filename = _filename + '_doc_idx.npy'
        sample_idx_filename = _filename + '_sample_idx.npy'
        shuffle_idx_filename = _filename + '_shuffle_idx.npy'

        try:
            builder = MMapRetrievalIndexedDatasetBuilder(data_bin_file, chunk_size, pad_id, False, stride=32)
            builder.add_item(sentence1)
            builder.add_item(sentence2)
            builder.finalize(data_index_file)

            builder = MMapRetrievalIndexedDatasetBuilder(db_bin_file, chunk_size, pad_id, True, stride=32)
            builder.add_item(sentence3)
            builder.add_item(sentence4)
            builder.finalize(db_index_file)

            # load the data
            data_index = MMapRetrievalIndexedDataset(data_file)
            db_index = MMapRetrievalIndexedDataset(db_file)

            with KNNIndex.writer(map_index_file, K) as w:
                map_np = np.random.randint(-3, db_index.chunks, (data_index.chunks, K))
                w.write(map_np)
            map_index = KNNIndex(map_index_file)

            documents = np.arange(0, data_index.sizes.shape[0])
            d = RETRODataset(
                cfg,
                None,
                tokenizer,
                name,
                data_prefix,
                documents,
                data_index,
                num_samples,
                seq_len,
                seed,
                map_index,
                db_index,
            )
            for i in range(len(d)):
                record = d[i]
                assert record['tokens'].shape[0] == seq_len
                assert record['labels'].shape[0] == seq_len
                assert record['retrieved_ids'].shape[0] == seq_len // chunk_size
                assert record['retrieved_ids'].shape[1] == K
                assert record['retrieved_ids'].shape[2] == chunk_size * 2
                assert record['tokens_mask'].shape[0] == seq_len

        finally:
            os.remove(data_bin_file)
            os.remove(data_index_file)
            os.remove(db_bin_file)
            os.remove(db_index_file)
            os.remove(map_index_file)
            os.remove(doc_idx_filename)
            os.remove(sample_idx_filename)
            os.remove(shuffle_idx_filename)

        # test the case that
        # training data and retrieval data are the same

        try:

            builder = MMapRetrievalIndexedDatasetBuilder(db_bin_file, chunk_size, pad_id, True, stride=32)
            builder.add_item(sentence1)
            builder.add_item(sentence2)
            builder.add_item(sentence3)
            builder.add_item(sentence4)
            builder.finalize(db_index_file)

            # load the data
            data_index = MMapRetrievalIndexedDataset(db_file)
            db_index = MMapRetrievalIndexedDataset(db_file)

            with KNNIndex.writer(map_index_file, K) as w:
                map_np = np.random.randint(-3, db_index.chunks, (data_index.chunks, K))
                w.write(map_np)
            map_index = KNNIndex(map_index_file)

            documents = np.arange(0, data_index.sizes.shape[0])
            d = RETRODataset(
                cfg,
                None,
                tokenizer,
                name,
                data_prefix,
                documents,
                data_index,
                num_samples,
                seq_len,
                seed,
                map_index,
                db_index,
            )
            for i in range(len(d)):
                record = d[i]
                assert record['tokens'].shape[0] == seq_len
                assert record['labels'].shape[0] == seq_len
                assert record['retrieved_ids'].shape[0] == seq_len // chunk_size
                assert record['retrieved_ids'].shape[1] == K
                assert record['retrieved_ids'].shape[2] == chunk_size * 2
                assert record['tokens_mask'].shape[0] == seq_len

        finally:
            os.remove(db_bin_file)
            os.remove(db_index_file)
            os.remove(map_index_file)
            os.remove(doc_idx_filename)
            os.remove(sample_idx_filename)
            os.remove(shuffle_idx_filename)

    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_MEGATRON_CORE, reason="megatron-core is not installed")
    def test_dedup(self):
        total = 1000
        id_start = np.array([0, 100, 200, 300, 500, 900])
        beg = 30
        end = 210
        chunk_id_to_doc_id_map = np.zeros((end - beg, 2), dtype=np.int64)
        build_map(id_start, chunk_id_to_doc_id_map, total, beg, end)
        for i in range(30, 100):
            assert_array_equal(chunk_id_to_doc_id_map[i - beg], id_start[0:2])
        for i in range(100, 200):
            assert_array_equal(chunk_id_to_doc_id_map[i - beg], id_start[1:3])
        for i in range(200, 210):
            assert_array_equal(chunk_id_to_doc_id_map[i - beg], id_start[2:4])
        beg = 5
        end = 100
        chunk_id_to_doc_id_map = np.zeros((end - beg, 2), dtype=np.int64)
        build_map(id_start, chunk_id_to_doc_id_map, total, beg, end)
        for i in range(beg, end):
            assert_array_equal(chunk_id_to_doc_id_map[i - beg], id_start[0:2])
        beg = 100
        end = 200
        chunk_id_to_doc_id_map = np.zeros((end - beg, 2), dtype=np.int64)
        build_map(id_start, chunk_id_to_doc_id_map, total, beg, end)
        for i in range(beg, end):
            assert_array_equal(chunk_id_to_doc_id_map[i - beg], id_start[1:3])
        beg = 900
        end = 1000
        chunk_id_to_doc_id_map = np.zeros((end - beg, 2), dtype=np.int64)
        build_map(id_start, chunk_id_to_doc_id_map, total, beg, end)
        for i in range(beg, end):
            assert_array_equal(chunk_id_to_doc_id_map[i - beg], np.array([900, 1000]))
        beg = 150
        end = 250
        chunk_id_to_doc_id_map = np.zeros((end - beg, 2), dtype=np.int64)
        build_map(id_start, chunk_id_to_doc_id_map, total, beg, end)
        for i in range(beg, 200):
            assert_array_equal(chunk_id_to_doc_id_map[i - beg], id_start[1:3])
        for i in range(200, end):
            assert_array_equal(chunk_id_to_doc_id_map[i - beg], id_start[2:4])

        I = np.arange(1000)[None, :]
        tmp_neighbors = np.ones_like(I) * -1
        with pytest.raises(ValueError):
            dedup(chunk_id_to_doc_id_map, I, tmp_neighbors, 0, beg)

        I = np.arange(1000)[None, :]
        tmp_neighbors = np.ones_like(I) * -1
        with pytest.raises(ValueError):
            dedup(chunk_id_to_doc_id_map, I, tmp_neighbors, 250, beg)

        for i in range(beg, 200):
            I = np.arange(1000)[None, :]
            tmp_neighbors = np.ones_like(I) * -1
            dedup(chunk_id_to_doc_id_map, I, tmp_neighbors, i, beg)
            gt = np.array(list(range(100)) + list(range(200, 1000)) + ([-1] * 100))
            assert_array_equal(tmp_neighbors[0], gt)

        for i in range(200, 250):
            I = np.arange(1000)[None, :]
            tmp_neighbors = np.ones_like(I) * -1
            dedup(chunk_id_to_doc_id_map, I, tmp_neighbors, i, beg)
            gt = np.array(list(range(200)) + list(range(300, 1000)) + ([-1] * 100))
            assert_array_equal(tmp_neighbors[0], gt)

        I = np.arange(1000)[None, :]
        I = np.repeat(I, 70, axis=0)
        tmp_neighbors = np.ones_like(I) * -1
        dedup(chunk_id_to_doc_id_map, I, tmp_neighbors, 180, beg)
        gt0 = np.array(list(range(100)) + list(range(200, 1000)) + ([-1] * 100))
        gt1 = np.array(list(range(200)) + list(range(300, 1000)) + ([-1] * 100))
        for i in range(20):
            assert_array_equal(tmp_neighbors[i], gt0)
        for i in range(20, 70):
            assert_array_equal(tmp_neighbors[i], gt1)
