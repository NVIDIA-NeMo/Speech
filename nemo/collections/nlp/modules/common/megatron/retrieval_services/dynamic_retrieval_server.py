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

import base64
import logging
import pickle
import threading
import time
from collections import namedtuple
from typing import List

import faiss
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_restful import Api

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.modules.common.megatron.retrieval_services.static_retrieval_server import (
    FaissRetrievalResource,
)
from nemo.collections.nlp.modules.common.megatron.retrieval_services.util import lock, request_data

# define this type to mimic the indexed dataset
DType = namedtuple('DType', ['dtype'])


class ChunkStore:
    """
    ChunkStore maps chunk id to tokens. It is used as an in memory storage for dynamic retrieval DB.
    """

    def __init__(self, chunk_size, pad_id):
        self.store = {}
        self._count = 0
        self.no_retrieval = np.ones(2 * chunk_size, dtype=np.int64) * pad_id
        self.chunk_size = chunk_size
        self.store[-1] = self.no_retrieval
        field = DType(dtype=np.int64)
        self._index = field

    def add(self, chunk):
        self.store[self._count] = chunk
        self._count += 1

    def get_chunk(self, neighbor_id):
        return self.store[neighbor_id]

    def reset(self):
        self._count = 0
        self.store = {}
        self.store[-1] = self.no_retrieval


class DynamicRetrievalResource(FaissRetrievalResource):
    """
    Dynamic Faiss Retrieval Flask resource.
    The PUT method is to get KNN tokens, add new chunks, reset index.
    """

    def __init__(
        self,
        index,
        tokenizer,
        chunk_size,
        stride,
        store,
        ctx_bert_ip,
        ctx_bert_port,
        query_bert_ip,
        query_bert_port,
        output_filename,
    ):
        super().__init__(index, tokenizer, store, query_bert_ip, query_bert_port)
        self.chunk_size = chunk_size
        self.stride = stride
        self.pad_id = self.tokenizer.pad_id
        self.ctx_bert_ip = ctx_bert_ip
        self.ctx_bert_port = ctx_bert_port
        self.output_filename = output_filename

    def put(self):
        data = request.get_json()
        if 'neighbors' in data:
            sentences = data['sentences']
            # do knn query
            num_neighbors = data['neighbors']
            with lock:  # Need to get lock to keep multiple threads from hitting code
                neighbors = self.get_knn(sentences, num_neighbors)
            return jsonify(neighbors.tolist())
        elif 'reset' in data:
            with lock:  # Need to get lock to keep multiple threads from hitting code
                self.reset()
            return "success"
        elif 'index_name' in data:
            with lock:
                # serialize the index
                index = self.index
                if hasattr(faiss, 'index_gpu_to_cpu'):
                    index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(index, data['index_name'] + '_' + self.output_filename + '.index')
                # save the data
                with open(self.output_filename + '.pkl', 'bw') as f:
                    pickle.dump(self.ds, f)
                return "success"
        else:
            sentences = data['sentences']
            add_eos = data['add_eos']
            # update the index
            with lock:  # Need to get lock to keep multiple threads from hitting code
                self.add_docs_to_index(sentences, add_eos)
            return "success"

    def reset(self):
        self.index.reset()
        self.ds.reset()

    def add_docs_to_index(self, docs: List[str], add_eos: bool = True):
        """
        Add documents to the Faiss index
        Args:
            docs: List[str], list of documents that is going to be added to the index
            add_eos: bool, whether add the eos in the end
        """
        for doc in docs:
            token_ids = self.tokenizer.text_to_ids(doc)
            # append eos in the end
            if add_eos:
                token_ids.append(self.tokenizer.eos_id)
            np_array = np.array(token_ids, dtype=np.int32)
            padded_size = self.chunk_size - (len(np_array) % self.chunk_size)
            # for retrieval database, added one more chunk in the end as padding
            padded_size += self.chunk_size
            np_array = np.pad(np_array, (0, padded_size), 'constant', constant_values=self.pad_id)
            chunk_texts = []
            for i in range(0, len(np_array), self.stride):
                if i + 2 * self.chunk_size <= len(np_array):
                    chunk = np_array[i : i + 2 * self.chunk_size]
                    self.ds.add(chunk)
                    chunk_texts.append(self.tokenizer.ids_to_text(chunk))
            emb = request_data(chunk_texts, self.ctx_bert_ip, self.ctx_bert_port)
            emb_data = base64.b64decode(emb.encode())
            emb = pickle.loads(emb_data)
            self.index.add(emb)  # add vectors to the index


class DynamicRetrievalServer(object):
    """
    Flask Dynamic Retrieval server, which helps to build dynamic retrieval index.
    """

    def __init__(
        self,
        faiss_devices: str,
        tokenizer: TokenizerSpec,
        chunk_size: int = 64,
        stride: int = 32,
        faiss_index: str = None,
        store_file: str = None,
        ctx_bert_ip: str = None,
        ctx_bert_port: int = 0,
        query_bert_ip: str = None,
        query_bert_port: int = 0,
        output_filename: str = 'dynamic_db',
    ):
        self.app = Flask(__name__, static_url_path='')
        has_gpu = torch.cuda.is_available() and hasattr(faiss, "index_gpu_to_cpu")
        embedding_dim = request_data({}, ctx_bert_ip, ctx_bert_port)['dim']

        if faiss_index is not None:
            self.index = faiss.read_index(faiss_index)
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)  # build the index
        self.pad_id = tokenizer.pad_id
        self.chunk_size = chunk_size
        self.stride = stride
        if store_file is not None:
            with open(store_file, 'rb') as f:
                self.store = pickle.load(f)
        else:
            self.store = ChunkStore(chunk_size, self.pad_id)

        if faiss_devices is None or not torch.cuda.is_available():
            device_list = None
        else:
            device_list = ['cuda:' + str(device) for device in faiss_devices.split(',')]

        if has_gpu and device_list is not None:
            beg = time.time()
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co, ngpu=len(device_list))
            end = time.time()
            logging.info(f'convert Faiss db to GPU takes {end - beg} s')

        self.tokenizer = tokenizer

        api = Api(self.app)
        api.add_resource(
            DynamicRetrievalResource,
            '/knn',
            resource_class_args=[
                self.index,
                self.tokenizer,
                self.chunk_size,
                self.stride,
                self.store,
                ctx_bert_ip,
                ctx_bert_port,
                query_bert_ip,
                query_bert_port,
                output_filename,
            ],
        )

    def run(self, url, port=None):
        threading.Thread(target=lambda: self.app.run(host=url, threaded=True, port=port)).start()
