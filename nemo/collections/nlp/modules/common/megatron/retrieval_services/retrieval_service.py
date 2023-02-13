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

import abc
import logging
import threading
from typing import List, Union

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.modules.common.megatron.retrieval_services.util import request_data

log = logging.getLogger('retrieval')
log.setLevel(logging.ERROR)

lock = threading.Lock()

PORT_NUM = 17179
PORT_NUM_DYN = 17180


class RetrievalService:
    """
    Abstract class for Retrieval Service. 
    """

    @abc.abstractmethod
    def get_knn(self, query: Union[List[str], str, torch.Tensor], neighbors: int):
        """Get K-nearest neighbor chunks based on the input query

        Args:
            query (Union[List[str], str, torch.Tensor]): query str, list of str or token ids in torch.Tensor type
            neighbors (int): number of neighbors to query
        """
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


class FaissRetrievalService(RetrievalService):
    """
    Static retrieval service client class.
    It implements the retrieval services interface, has a simple client to do KNN queries.
    """

    def __init__(
        self, tokenizer: TokenizerSpec, service_ip: str = None, service_port: int = None, updatable: bool = False,
    ):
        self.updatable = updatable
        self.tokenizer = tokenizer
        self.service_ip = service_ip
        self.service_port = service_port

    def get_knn(self, query: Union[List[str], str, torch.Tensor], neighbors):
        """Get K-nearest neighbor chunks based on the input query

        Args:
            query (Union[List[str], str, torch.Tensor]): query str, list of str or token ids in torch.Tensor type
            neighbors (int): number of neighbors to query
        """

        if isinstance(query, torch.Tensor):
            sentence_list = []
            for q in query:
                text = self.tokenizer.ids_to_text(q)
                sentence_list.append(text)
            query = sentence_list
        data = {'sentences': query}
        data['neighbors'] = neighbors
        result = request_data(data, self.service_ip, self.service_port)
        result = np.array(result)
        return result


class DynamicFaissRetrievalService(FaissRetrievalService):
    """
    Dynamic retrieval service client class.
    It implements the retrieval services interface, has a simple client to add, reset and query
    the dynamic retrieval index.
    """

    def __init__(
        self, tokenizer: TokenizerSpec, service_ip: str = None, service_port: int = None,
    ):
        super().__init__(tokenizer=tokenizer, service_ip=service_ip, service_port=service_port, updatable=True)

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
        return request_data(data, self.service_ip, self.service_port)

    def write_index(self, index_name: str):
        """
        Write the dynamic index and document storage into file
        Args:
            index_name: str, the index name used for the file name
        """
        data = {'index_name': index_name}
        return request_data(data, self.service_ip, self.service_port)

    def reset(self):
        """
        Write the dynamic index and document storage into file
        Args:
            index_name: str, the index name used for the file name
        """
        data = {'reset': None}
        return request_data(data, self.service_ip, self.service_port)


class ComboRetrievalService(DynamicFaissRetrievalService):
    """
    Combo retrieval service client class.
    It implements the retrieval services interface, has a simple client to add, reset, query, update weights
    """

    def __init__(
        self, tokenizer: TokenizerSpec, service_ip: str = None, service_port: int = None,
    ):
        super().__init__(tokenizer=tokenizer, service_ip=service_ip, service_port=service_port)

    def update_weights(self, weights: List[float]):
        """ update the weights between the children services
        Args:
            weights (List[float]): weights for children services
        """
        data = {"update_weight": weights}
        return request_data(data, self.service_ip, self.service_port)
