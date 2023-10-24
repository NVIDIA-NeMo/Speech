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
"""Utilities for generating text."""

import json
import threading

import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource

from nemo.collections.nlp.modules.common.retro_inference_strategies import (
    RetroModelTextGenerationStrategy,
    RetroQAModelTextGenerationStrategy,
)
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.utils import logging

GENERATE_NUM = 0
lock = threading.Lock()

API_ALLOWED_KEYS = set(
    [
        'all_probs',
        'sentences',
        "task_ids",
        "tokens_to_generate",
        "temperature",
        "add_BOS",
        "greedy",
        "top_k",
        "top_p",
        "neighbors",
        "repetition_penalty",
        "min_tokens_to_generate",
        "end_strings",
        "compute_logprob",
    ]
)


class MegatronGenerate(Resource):
    def __init__(self, model, inference_strategy=None):
        self.model = model
        self.inference_strategy = inference_strategy

    @staticmethod
    def send_do_generate():
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)

    def put(self):
        logging.info("request IP: " + str(request.remote_addr))
        logging.info(json.dumps(request.get_json()))
        # check keys
        for key in request.get_json().keys():
            if key not in API_ALLOWED_KEYS:
                logging.error(f"The request key {key} is not allowed")

        sentences = request.get_json()["sentences"]
        if isinstance(sentences, tuple):  # Input can be text or tensor
            if len(sentences[0]) != len(sentences[1]) or sentences[0] > 128:
                return "Maximum number of sentences is 128", 400
        elif len(sentences) > 128:
            return "Maximum number of sentences is 128", 400

        task_ids = None  # Used for ptuned/prompt tuned models only
        if "task_ids" in request.get_json():
            task_ids = request.get_json()["task_ids"]
            if not isinstance(sentences, tuple):
                return "Input at 'sentences' must by a tuple of two tensors like:\
                    (context_tokens_tensor, context_length_tensor) if task ids are given"
            if len(task_ids) != len(sentences[0]):
                return "Each sentence must have a corresponding task id for p-tuned/prompt-tuned models"

        tokens_to_generate = 64  # Choosing hopefully sane default.  Full sequence is slow
        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if tokens_to_generate < 1:
                return "tokens_to_generate must be an integer greater than 0"

        all_probs = False
        if "all_probs" in request.get_json():
            all_probs = request.get_json()["all_probs"]
            if not isinstance(all_probs, bool):
                return "all_probs must be a boolean value"

        temperature = 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (type(temperature) == int or type(temperature) == float):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"

        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"

        greedy = False
        if "greedy" in request.get_json():
            greedy = request.get_json()["greedy"]
            if not isinstance(greedy, bool):
                return "greedy must be a boolean value"

        top_k = 0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not (type(top_k) == int or type(top_k) == float):
                return "top_k must be a positive integer number"
            if not (0 <= top_k):
                return "top_k must be a positive integer number"

        top_p = 0.9
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not (type(top_p) == int or type(top_p) == float):
                return "top_p must be a positive number less than or equal to 1.0"
            if not (0.0 <= top_p <= 1.0):
                return "top_p must be a positive number less than or equal to 1.0"

        repetition_penalty = 1.2
        if "repetition_penalty" in request.get_json():
            repetition_penalty = request.get_json()["repetition_penalty"]
            if not (type(repetition_penalty) == int or type(repetition_penalty) == float):
                return "repetition_penalty must be a positive number no less than 1.0"
            if not (1.0 <= repetition_penalty):
                return "repetition_penalty must be a positive number no less than 1.0"

        end_strings = ['<|endoftext|>']
        if 'end_strings' in request.get_json():
            end_strings = request.get_json()['end_strings']
            if not isinstance(end_strings, list):
                return "expect end_strings to be a list of strings"
            if not all([isinstance(s, str) for s in end_strings]):
                return "expect end_strings to be a list of strings"

        min_tokens_to_generate = 0
        if "min_tokens_to_generate" in request.get_json():
            min_tokens_to_generate = request.get_json()["min_tokens_to_generate"]
            if not isinstance(min_tokens_to_generate, int):
                return "min_tokens_to_generate must be an integer no less than 0"
            if min_tokens_to_generate < 0:
                return "min_tokens_to_generate must be an integer no less than 0"

        neighbors = None
        if "neighbors" in request.get_json():
            neighbors = request.get_json()["neighbors"]
            if not isinstance(neighbors, int):
                return "num of neighbors must be an integer no less than 0"
            if neighbors < 0:
                return "num of neighbors must be an integer no less than 0"

        compute_logprob = False
        if "compute_logprob" in request.get_json():
            compute_logprob = request.get_json()["compute_logprob"]
            if not isinstance(compute_logprob, bool):
                return "compute_logprob must be a boolean value"

        with lock:  # Need to get lock to keep multiple threads from hitting code
            MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
            extra = {}
            if task_ids is not None:
                extra['task_ids'] = task_ids
            if self.inference_strategy is not None:
                extra['strategy'] = self.inference_strategy
                # RETRO specific arguments
                if isinstance(
                    self.inference_strategy, (RetroModelTextGenerationStrategy, RetroQAModelTextGenerationStrategy)
                ):
                    if neighbors is not None:
                        self.inference_strategy.update_neighbors(neighbors)

            output = generate(
                self.model,
                sentences,
                tokens_to_generate,
                all_probs,
                temperature,
                add_BOS,
                top_k,
                top_p,
                greedy,
                repetition_penalty,
                end_strings=end_strings,
                min_tokens_to_generate=min_tokens_to_generate,
                compute_logprob=compute_logprob,
                **extra,
            )
            for k in output:
                if isinstance(output[k], torch.Tensor):
                    output[k] = output[k].tolist()
        if not all_probs:
            del output['full_logprob']

        if self.inference_strategy is not None:
            if isinstance(
                self.inference_strategy, (RetroModelTextGenerationStrategy, RetroQAModelTextGenerationStrategy)
            ):
                retrieved_doc = self.inference_strategy.retrieved_text
                output['retrieved'] = retrieved_doc
        return jsonify(output)


class MegatronServer(object):
    def __init__(self, model, inference_strategy=None):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/generate', resource_class_args=[model, inference_strategy])

    def run(self, url, port=5000):
        self.app.run(url, threaded=True, port=port, debug=False)
