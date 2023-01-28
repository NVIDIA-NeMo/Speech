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

import base64
import pickle
import threading
import time
from collections import OrderedDict
from typing import List, Union

import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from sentence_transformers import SentenceTransformer

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

PORT_NUM_START = 17190
# a global dict, map bert name to port number
BERT_MODEL_MAP = OrderedDict()


def get_available_port_num():
    output = PORT_NUM_START
    return output + len(BERT_MODEL_MAP)


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
        name: str,
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
        self.name = name
        self.tokenizer = tokenizer
        self.pool = self.bert_model.start_multi_process_pool(device_list)
        self.sentence_bert_batch = sentence_bert_batch
        api = Api(self.app)
        api.add_resource(
            SentenceBertResource,
            '/knn',
            resource_class_args=[self.bert_model, self.tokenizer, self.pool, self.sentence_bert_batch,],
        )

    def run(self, url):
        port = BERT_MODEL_MAP[self.name]
        threading.Thread(target=lambda: self.app.run(host=url, threaded=True, port=port)).start()


def start_sentence_bert_server(
    name: str,
    devices: str,
    tokenizer: TokenizerSpec,
    sentence_bert: str = 'all-mpnet-base-v2',
    sentence_bert_batch: int = 4,
):
    """
    Start the sentence bert server method.
    It only starts the server at rank 0 worker.
    Doesn't support multiple nodes yet.
    """
    # register the bert model port number
    port_num = get_available_port_num()
    BERT_MODEL_MAP[name] = port_num

    if torch.distributed.is_initialized():
        # doesn't handle multiple nodes yet.
        # need to set ip address properly for it to work in multiple nodes environment
        if torch.distributed.get_rank() == 0:
            server = SentenceBertServer(name, devices, tokenizer, sentence_bert, sentence_bert_batch,)
            server.run("0.0.0.0")
            # sleep to make sure the sentence bert server is full started.
            time.sleep(2)
        torch.distributed.barrier()
    else:
        server = SentenceBertServer(name, devices, tokenizer, sentence_bert, sentence_bert_batch,)
        server.run("0.0.0.0")
        # sleep to make sure the sentence bert server is full started.
        time.sleep(2)
