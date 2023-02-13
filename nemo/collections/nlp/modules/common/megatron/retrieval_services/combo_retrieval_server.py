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

import pickle
import threading
from typing import List, Union

import faiss
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.modules.common.megatron.retrieval_services.retrieval_service import (
    DynamicFaissRetrievalService,
    FaissRetrievalService,
)
from nemo.collections.nlp.modules.common.megatron.retrieval_services.util import lock

weights = None


class ComboRetrievalResource(Resource):
    """
    Combo Faiss Retrieval Flask resource.
    The PUT method is to get KNN tokens, add new chunks, reset index.
    """

    def __init__(self, retrieval_services, weight_container):
        self.retrieval_services = retrieval_services
        self.updatable = any([service.updatable for service in retrieval_services])

        self.weight_container = weight_container
        weights = np.array(weight_container[0])
        # normalize the weights
        weights = weights / weights.sum()
        self.weight_container[0] = weights

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
        elif 'update_weight' in data:
            with lock:
                self.update_weights(data['update_weight'])
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
        else:
            sentences = data['sentences']
            add_eos = data['add_eos']
            # update the index
            with lock:  # Need to get lock to keep multiple threads from hitting code
                self.add_docs_to_index(sentences, add_eos)
            return "success"

    def reset(self):
        output = 'success'
        if not self.updatable:
            return 'no dynamic service, no action is performed'
        for i, service in enumerate(self.retrieval_services):
            if service.updatable:
                service.reset()
        return output

    def update_weights(self, weights):
        weights = np.array(weights)
        # normalize the weights
        weights = weights / weights.sum()
        self.weight_container[0] = weights

    def get_knn(self, query: Union[List[str], str, torch.Tensor], neighbors):
        weights = self.weight_container[0]
        if neighbors == 0:
            return self.retrieval_services[0].get_knn(query, 0)
        total_neighbors = 0
        results = []
        for i, service in enumerate(self.retrieval_services):
            k = int(neighbors * weights[i])
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
            if not self.updatable:
                return 'no dynamic service, no action is performed'
        for i, service in enumerate(self.retrieval_services):
            if service.updatable:
                service.add_docs_to_index(query, add_eos)
        return output

    def write_index(self, index_name: str):
        """
        write the dynamic index into a file
        Args:
            index_name: str, index name
        """
        output = 'success'
        if not self.updatable:
            if not self.updatable:
                return 'no dynamic service, no action is performed'
        for i, service in enumerate(self.retrieval_services):
            if service.updatable:
                service.write_index(index_name)
        return output


class ComboRetrievalServer(object):
    """
    Flask Combo Retrieval server, which helps to aggregate different retrieval services
    """

    def __init__(
        self, tokenizer: TokenizerSpec, services_cfg: list,
    ):
        self.app = Flask(__name__, static_url_path='')
        services = []
        weights = []
        for service_cfg in services_cfg:
            weights.append(service_cfg.weight)
            if service_cfg.type == 'FaissRetrievalService':
                service = FaissRetrievalService(
                    tokenizer=tokenizer, service_ip=service_cfg.service_ip, service_port=service_cfg.service_port
                )
            elif service_cfg.type == 'DynamicFaissRetrievalService':
                service = DynamicFaissRetrievalService(
                    tokenizer=tokenizer, service_ip=service_cfg.service_ip, service_port=service_cfg.service_port
                )
            else:
                raise ValueError(f'Unsupported retrieval service {service_cfg.type}')
            services.append(service)
        self.weight_container = [weights]
        self.tokenizer = tokenizer

        api = Api(self.app)
        api.add_resource(
            ComboRetrievalResource, '/knn', resource_class_args=[services, self.weight_container,],
        )

    def run(self, url, port=None):
        threading.Thread(target=lambda: self.app.run(host=url, threaded=True, port=port)).start()
