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

import abc
import base64
import json
import logging
import pickle
import threading
import time
from typing import List, Union

import faiss
import numpy as np
import requests
import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from sentence_transformers import SentenceTransformer

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import MMapRetrievalIndexedDataset

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

lock = threading.Lock()
headers = {"Content-Type": "application/json"}

PORT_NUM = 17179
PORT_NUM_DYN = 17180
PORT_NUM_BERT = 17181


def request_data(data, port=PORT_NUM):
    resp = requests.put('http://localhost:{}/knn'.format(port), data=json.dumps(data), headers=headers)
    return resp.json()


class RetrievalService:
    """
    Abstract class for Retrieval Service. 
    """

    @abc.abstractmethod
    def get_knn(self, query: Union[List[str], str, torch.Tensor], neighbors: int):
        pass

    @abc.abstractmethod
    def add_docs_to_index(self, docs: List[str], add_eos: bool = True):
        """
        Add documents to the Faiss index
        Args:
            docs: List[str], list of documents that is going to be added to the index
            add_eos: bool, whether add the eos in the end
        """
        raise NotImplementedError()


class ChunkStore:
    """
    ChunkStore maps chunk id to tokens. It is used as an in memory storage for dynamic retrieval DB.
    """

    def __init__(self, chunk_size, pad_id):
        self.store = {}
        self._count = 0
        self.no_retrieval = np.ones(2 * chunk_size, dtype=np.int64) * pad_id
        self.store[-1] = self.no_retrieval

    def add(self, chunk):
        self.store[self._count] = chunk
        self._count += 1

    def get_chunk(self, neighbor_id):
        return self.store[neighbor_id]

    def reset(self):
        self._count = 0
        self.store = {}
        self.store[-1] = self.no_retrieval


class SentenceBertResource(Resource):
    """
    SentenceBERT Flask resource.
    The PUT method is to get token/str embedding.
    """

    def __init__(
        self, bert_model, tokenizer, pool, sentence_bert_batch,
    ):
        # server
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.pool = pool
        self.sentence_bert_batch = sentence_bert_batch
        self.embedding_dim = self.bert_model.get_sentence_embedding_dimension()

    def put(self):
        data = request.get_json()
        if isinstance(data, dict):
            return jsonify({'dim': self.embedding_dim})
        sentences = data
        emb = self.get_emb(sentences)
        str_emb = base64.b64encode(pickle.dumps(emb))
        return str_emb.decode('ascii')

    def get_emb(self, query: Union[List[str], str, torch.Tensor]):
        if isinstance(query, str):
            query = [query]
        elif isinstance(query, torch.Tensor):
            sentence_list = []
            for q in query:
                text = self.tokenizer.ids_to_text(q)
                sentence_list.append(text)
            query = sentence_list
        emb = self.bert_model.encode_multi_process(
            sentences=query, pool=self.pool, batch_size=self.sentence_bert_batch
        )
        return emb


class SentenceBertServer(object):
    """
    Flask SentenceBERT server, which helps to calculate str/token embeddings
    """

    def __init__(
        self,
        devices: str,
        tokenizer: TokenizerSpec,
        sentence_bert: str = 'all-mpnet-base-v2',
        sentence_bert_batch: int = 4,
    ):
        self.app = Flask(__name__, static_url_path='')

        if devices is None or not torch.cuda.is_available():
            device_list = None
        else:
            device_list = ['cuda:' + str(device) for device in devices.split(',')]

        self.bert_model = SentenceTransformer(sentence_bert)
        self.tokenizer = tokenizer
        self.pool = self.bert_model.start_multi_process_pool(device_list)
        self.sentence_bert_batch = sentence_bert_batch
        api = Api(self.app)
        api.add_resource(
            SentenceBertResource,
            '/knn',
            resource_class_args=[self.bert_model, self.tokenizer, self.pool, self.sentence_bert_batch,],
        )

    def run(self, url, port=PORT_NUM_BERT):
        threading.Thread(target=lambda: self.app.run(host=url, threaded=True, port=port)).start()


def start_sentence_bert_server(
    devices: str, tokenizer: TokenizerSpec, sentence_bert: str = 'all-mpnet-base-v2', sentence_bert_batch: int = 4
):
    """
    Start the sentence bert server method.
    It only starts the server at rank 0 worker.
    Doesn't support multiple nodes yet.
    """

    if torch.distributed.get_rank() == 0:
        server = SentenceBertServer(devices, tokenizer, sentence_bert, sentence_bert_batch,)
        server.run("0.0.0.0")
    torch.distributed.barrier()


class FaissRetrievalResource(Resource):
    """
    Static Faiss Retrieval Flask resource.
    The PUT method is to get KNN tokens.
    """

    def __init__(
        self, index, tokenizer, ds,
    ):
        # server
        self.index = index
        self.tokenizer = tokenizer
        self.ds = ds

    def put(self):
        data = request.get_json()
        sentences = data['sentences']
        num_neighbors = data['neighbors']
        with lock:  # Need to get lock to keep multiple threads from hitting code
            neighbors = self.get_knn(sentences, num_neighbors)
        return jsonify(neighbors.tolist())
        # check keys

    def get_knn(self, query: Union[List[str], str, torch.Tensor], neighbors: int):
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
        emb = request_data(query, PORT_NUM_BERT)
        emb_data = base64.b64decode(emb.encode())
        emb = pickle.loads(emb_data)
        D, knn = self.index.search(emb, neighbors)
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
        self, faiss_index: str, faiss_devices: str, nprobe: int, retrieval_index: str, tokenizer: TokenizerSpec,
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
            logging.info('convert Faiss db to GPU takes', end - beg)
        self.index.nprobe = nprobe
        self.tokenizer = tokenizer
        self.ds = MMapRetrievalIndexedDataset(retrieval_index)
        api = Api(self.app)
        api.add_resource(
            FaissRetrievalResource, '/knn', resource_class_args=[self.index, self.tokenizer, self.ds,],
        )

    def run(self, url, port=PORT_NUM):
        threading.Thread(target=lambda: self.app.run(host=url, threaded=True, port=port)).start()


class DynamicRetrievalResource(FaissRetrievalResource):
    """
    Dynamic Faiss Retrieval Flask resource.
    The PUT method is to get KNN tokens, add new chunks, reset index.
    """

    def __init__(
        self, index, tokenizer, chunk_size, stride, store,
    ):
        self.index = index
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.stride = stride
        self.pad_id = self.tokenizer.pad_id
        self.ds = store

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
            emb = request_data(chunk_texts, PORT_NUM_BERT)
            emb_data = base64.b64decode(emb.encode())
            emb = pickle.loads(emb_data)
            self.index.add(emb)  # add vectors to the index


class DynamicRetrievalServer(object):
    """
    Flask Dynamic Retrieval server, which helps to build dynamic retrieval index.
    """

    def __init__(
        self, faiss_devices: str, tokenizer: TokenizerSpec, chunk_size: int = 64, stride: int = 32,
    ):
        self.app = Flask(__name__, static_url_path='')
        has_gpu = torch.cuda.is_available() and hasattr(faiss, "index_gpu_to_cpu")
        embedding_dim = request_data({}, PORT_NUM_BERT)['dim']
        self.index = faiss.IndexFlatL2(embedding_dim)  # build the index
        self.pad_id = tokenizer.pad_id
        self.chunk_size = chunk_size
        self.stride = stride
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
            logging.info('convert Faiss db to GPU takes', end - beg)

        self.tokenizer = tokenizer

        api = Api(self.app)
        api.add_resource(
            DynamicRetrievalResource,
            '/knn',
            resource_class_args=[self.index, self.tokenizer, self.chunk_size, self.stride, self.store,],
        )

    def run(self, url, port=PORT_NUM_DYN):
        threading.Thread(target=lambda: self.app.run(host=url, threaded=True, port=port)).start()


class FaissRetrievalService(RetrievalService):
    """
    Top level static retrieval service class.
    It starts the server at rank 0 worker, currently doesn't support multiple nodes yet.
    It implements the retrieval services interface, has a simple client to do KNN queries.
    """

    def __init__(
        self, faiss_index: str, faiss_devices: str, nprobe: int, retrieval_index: str, tokenizer: TokenizerSpec,
    ):
        self.updatable = False
        self.tokenizer = tokenizer
        ds = MMapRetrievalIndexedDataset(retrieval_index)
        self.chunk_size = ds.chunk_size
        pad_id = self.tokenizer.pad_id
        # batch, neighbors, 2*chunk_size
        self.no_retrieval = np.ones((1, 1, 2 * self.chunk_size), dtype=ds._index.dtype) * pad_id
        if torch.distributed.get_rank() == 0:
            server = RetrievalServer(faiss_index, faiss_devices, nprobe, retrieval_index, tokenizer)
            server.run("0.0.0.0")
        torch.distributed.barrier()

    def get_knn(self, query: Union[List[str], str, torch.Tensor], neighbors):
        if isinstance(query, torch.Tensor):
            sentence_list = []
            for q in query:
                text = self.tokenizer.ids_to_text(q)
                sentence_list.append(text)
            query = sentence_list
        if neighbors == 0:
            # use padding
            return np.repeat(self.no_retrieval, len(query), 0).astype(np.int64)
        data = {'sentences': query}
        data['neighbors'] = neighbors
        result = request_data(data, PORT_NUM)
        result = np.array(result)
        return result


class DynamicFaissRetrievalService(RetrievalService):
    """
    Top level dynamic retrieval service class.
    It starts the server at rank 0 worker, currently doesn't support multiple nodes yet.
    It implements the retrieval services interface, has a simple client to add, reset and query
    the dynamic retrieval index.
    """

    def __init__(
        self, faiss_devices: str, tokenizer: TokenizerSpec, chunk_size: int, stride: int,
    ):
        self.updatable = True
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        pad_id = self.tokenizer.pad_id
        # batch, neighbors, 2*chunk_size
        self.no_retrieval = np.ones((1, 1, 2 * self.chunk_size), dtype=np.int64) * pad_id
        if torch.distributed.get_rank() == 0:
            server = DynamicRetrievalServer(faiss_devices, tokenizer, chunk_size, stride)
            server.run("0.0.0.0")
        torch.distributed.barrier()

    def get_knn(self, query: Union[List[str], str, torch.Tensor], neighbors):
        if isinstance(query, torch.Tensor):
            sentence_list = []
            for q in query:
                text = self.tokenizer.ids_to_text(q)
                sentence_list.append(text)
            query = sentence_list
        if neighbors == 0:
            # use padding
            return np.repeat(self.no_retrieval, len(query), 0).astype(np.int64)
        data = {'sentences': query}
        data['neighbors'] = neighbors
        result = request_data(data, PORT_NUM_DYN)
        result = np.array(result)
        return result

    def add_docs_to_index(self, query: List[str], add_eos: bool = True):
        """
        Add documents to the Faiss index
        Args:
            docs: List[str], list of documents that is going to be added to the index
            add_eos: bool, whether add the eos in the end
        """
        if isinstance(query, torch.Tensor):
            sentence_list = []
            for q in query:
                text = self.tokenizer.ids_to_text(q)
                sentence_list.append(text)
            query = sentence_list
        data = {'sentences': query, 'add_eos': add_eos}
        return request_data(data, PORT_NUM_DYN)


class ComboRetrievalService(RetrievalService):
    """
    Top level retrieval service class.
    It combines other retrieval services as a combo retrieval service.
    It uses `weights` to determine the number of neighbors for each of the retrieval service members.
    """

    def __init__(self, retrieval_services, weights):
        self.retrieval_services = retrieval_services
        self.updatable = any([service.updatable for service in retrieval_services])
        weights = np.array(weights)
        # normalize the weights
        self.weights = weights / weights.sum()
        self.chunk_size = self.retrieval_services[0].chunk_size

    def update_weights(self, weights):
        weights = np.array(weights)
        # normalize the weights
        self.weights = weights / weights.sum()

    def get_knn(self, query: Union[List[str], str, torch.Tensor], neighbors):
        if neighbors == 0:
            return self.retrieval_services[0].get_knn(query, 0)
        total_neighbors = 0
        results = []
        for i, service in enumerate(self.retrieval_services):
            k = int(neighbors * self.weights[i])
            if i == len(self.retrieval_services) - 1:
                k = neighbors - total_neighbors
            total_neighbors += k
            if k == 0:
                # empty, skip it
                continue
            result = service.get_knn(query, k)
            results.append(result)
        return np.concatenate(results, axis=1)

    def add_docs_to_index(self, query: List[str], add_eos: bool = True):
        """
        Add documents to the Faiss index
        Args:
            docs: List[str], list of documents that is going to be added to the index
            add_eos: bool, whether add the eos in the end
        """
        output = 'success'
        if not self.updatable:
            return output
        for i, service in enumerate(self.retrieval_services):
            if service.updatable:
                service.add_docs_to_index(query, add_eos)
        return output
