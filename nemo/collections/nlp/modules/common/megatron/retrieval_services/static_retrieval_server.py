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
from typing import List, Union

import faiss
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import MMapRetrievalIndexedDataset
from nemo.collections.nlp.modules.common.megatron.retrieval_services.util import lock, request_data


class FaissRetrievalResource(Resource):
    """
    Static Faiss Retrieval Flask resource.
    The PUT method is to get KNN tokens.
    """

    def __init__(
        self, index, tokenizer, ds, query_bert_ip, query_bert_port,
    ):
        # server
        self.index = index
        self.tokenizer = tokenizer
        self.ds = ds
        self.query_bert_ip = query_bert_ip
        self.query_bert_port = query_bert_port
        self.chunk_size = ds.chunk_size
        pad_id = self.tokenizer.pad_id
        self.no_retrieval = np.ones((1, 1, 2 * self.chunk_size), dtype=ds._index.dtype) * pad_id

    def put(self):
        data = request.get_json()
        sentences = data['sentences']
        num_neighbors = data['neighbors']
        with lock:  # Need to get lock to keep multiple threads from hitting code
            neighbors = self.get_knn(sentences, num_neighbors)
        return jsonify(neighbors.tolist())
        # check keys

    def get_knn(self, query: Union[List[str], str, torch.Tensor], neighbors: int):
        if neighbors == 0:
            # use padding
            return np.repeat(self.no_retrieval, len(query), 0).astype(np.int64)
        single_sentence = False
        if isinstance(query, str):
            single_sentence = True
            query = [query]
        elif isinstance(query, torch.Tensor):
            sentence_list = []
            for q in query:
                text = self.tokenizer.ids_to_text(q)
                sentence_list.append(text)
            query = sentence_list
        emb = request_data(query, self.query_bert_ip, self.query_bert_port)
        emb_data = base64.b64decode(emb.encode())
        emb = pickle.loads(emb_data)
        if self.index.ntotal == 0:
            # A workaround to fix searching an empty Faiss index
            knn = [[-1] * neighbors for i in range(len(emb))]
        else:
            _, knn = self.index.search(emb, neighbors)
        results = []
        for sentence_neighbors in knn:
            chunks = []
            for neighbor_chunk_id in sentence_neighbors:
                chunk_id = self.ds.get_chunk(neighbor_chunk_id)
                chunks.append(chunk_id)
            chunks = np.stack(chunks, axis=0).astype(np.int64)
            results.append(chunks)
        if single_sentence:
            # unpack the single sentence input
            return results[0]
        return np.stack(results, axis=0).astype(np.int64)


class RetrievalServer(object):
    """
    Flask Retrieval server, which helps to get the KNN tokens given the query chunk
    """

    def __init__(
        self,
        faiss_index: str,
        faiss_devices: str,
        nprobe: int,
        retrieval_index: str,
        tokenizer: TokenizerSpec,
        query_bert_ip: str,
        query_bert_port: int = None,
    ):
        self.app = Flask(__name__, static_url_path='')
        # server
        has_gpu = torch.cuda.is_available() and hasattr(faiss, "index_gpu_to_cpu")

        if faiss_devices is None or not torch.cuda.is_available():
            device_list = None
        else:
            device_list = ['cuda:' + str(device) for device in faiss_devices.split(',')]

        self.index = faiss.read_index(faiss_index)
        if has_gpu and device_list is not None:
            beg = time.time()
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co, ngpu=len(device_list))
            end = time.time()
            logging.info(f'convert Faiss db to GPU takes {end - beg} s')
        self.index.nprobe = nprobe
        self.tokenizer = tokenizer
        self.ds = MMapRetrievalIndexedDataset(retrieval_index)
        api = Api(self.app)
        api.add_resource(
            FaissRetrievalResource,
            '/knn',
            resource_class_args=[self.index, self.tokenizer, self.ds, query_bert_ip, query_bert_port],
        )

    def run(self, url, port=None):
        threading.Thread(target=lambda: self.app.run(host=url, threaded=True, port=port)).start()
