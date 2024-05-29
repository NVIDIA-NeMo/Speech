# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod

import numpy as np

from nemo.deploy.utils import str_list2numpy

use_pytriton = True
try:
    from pytriton.client import DecoupledModelClient, ModelClient
except Exception:
    use_pytriton = False


class NemoQueryLLMBase(ABC):
    def __init__(self, url, model_name):
        self.url = url
        self.model_name = model_name

    @abstractmethod
    def query_llm(
        self,
        prompts,
        stop_words_list=None,
        bad_words_list=None,
        no_repeat_ngram_size=None,
        max_output_token=512,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        random_seed=None,
        task_id=None,
        lora_uids=None,
        init_timeout=60.0,
    ):
        pass


class NemoQueryLLM(NemoQueryLLMBase):
    """
    Sends a query to Triton for LLM inference

    Example:
        from nemo.deploy import NemoQueryLLM

        nq = NemoQueryLLM(url="localhost", model_name="GPT-2B")

        prompts = ["hello, testing GPT inference", "another GPT inference test?"]
        output = nq.query_llm(
            prompts=prompts,
            max_output_len=100,
            top_k=1,
            top_p=0.0,
            temperature=0.0,
        )
        print("prompts: ", prompts)
    """

    def __init__(self, url, model_name):
        super().__init__(
            url=url,
            model_name=model_name,
        )

    def query_llm(
        self,
        prompts,
        stop_words_list=None,
        bad_words_list=None,
        no_repeat_ngram_size=None,
        max_output_token=512,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        random_seed=None,
        task_id=None,
        lora_uids=None,
        init_timeout=60.0,
    ):
        """
        Query the Triton server synchronously and return a list of responses.

        Args:
            prompts (List(str)): list of sentences.
            max_output_token (int): max generated tokens.
            top_k (int): limits us to a certain number (K) of the top tokens to consider.
            top_p (float): limits us to the top tokens within a certain probability mass (p).
            temperature (float): A parameter of the softmax function, which is the last layer in the network.
            random_seed (int): Seed to condition sampling.
            stop_words_list (List(str)): list of stop words.
            bad_words_list (List(str)): list of bad words.
            no_repeat_ngram_size (int): no repeat ngram size.
            task_id (str): downstream task id if virtual tokens are used.
            init_timeout (flat): timeout for the connection.
        """

        prompts = str_list2numpy(prompts)
        inputs = {"prompts": prompts}

        if max_output_token is not None:
            inputs["max_output_token"] = np.full(prompts.shape, max_output_token, dtype=np.int_)

        if top_k is not None:
            inputs["top_k"] = np.full(prompts.shape, top_k, dtype=np.int_)

        if top_p is not None:
            inputs["top_p"] = np.full(prompts.shape, top_p, dtype=np.single)

        if temperature is not None:
            inputs["temperature"] = np.full(prompts.shape, temperature, dtype=np.single)

        if random_seed is not None:
            inputs["random_seed"] = np.full(prompts.shape, random_seed, dtype=np.int_)

        if stop_words_list is not None:
            inputs["stop_words_list"] = str_list2numpy(stop_words_list)
        if bad_words_list is not None:
            inputs["bad_words_list"] = str_list2numpy(bad_words_list)

        if no_repeat_ngram_size is not None:
            inputs["no_repeat_ngram_size"] = np.full(prompts.shape, no_repeat_ngram_size, dtype=np.single)

        if task_id is not None:
            task_id = np.char.encode(task_id, "utf-8")
            inputs["task_id"] = np.full((prompts.shape[0], len([task_id])), task_id)

        if lora_uids is not None:
            lora_uids = np.char.encode(lora_uids, "utf-8")
            inputs["lora_uids"] = np.full((prompts.shape[0], len(lora_uids)), lora_uids)

        with ModelClient(self.url, self.model_name, init_timeout_s=init_timeout) as client:
            result_dict = client.infer_batch(**inputs)
            output_type = client.model_config.outputs[0].dtype

            if output_type == np.bytes_:
                sentences = np.char.decode(result_dict["outputs"].astype("bytes"), "utf-8")
                return sentences
            else:
                return result_dict["outputs"]

    def query_llm_streaming(
        self,
        prompts,
        stop_words_list=None,
        bad_words_list=None,
        no_repeat_ngram_size=None,
        max_output_token=512,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        random_seed=None,
        task_id=None,
        lora_uids=None,
        init_timeout=60.0,
    ):
        """
        Query the Triton server using streaming.

        Args:
            prompts (List(str)): list of sentences.
            max_output_token (int): max generated tokens.
            top_k (int): limits us to a certain number (K) of the top tokens to consider.
            top_p (float): limits us to the top tokens within a certain probability mass (p).
            temperature (float): A parameter of the softmax function, which is the last layer in the network.
            random_seed (int): Seed to condition sampling.
            stop_words_list (List(str)): list of stop words.
            bad_words_list (List(str)): list of bad words.
            no_repeat_ngram_size (int): no repeat ngram size.
            task_id (str): downstream task id if virtual tokens are used.
            init_timeout (flat): timeout for the connection.
        """

        prompts = str_list2numpy(prompts)
        inputs = {"prompts": prompts}

        if max_output_token is not None:
            inputs["max_output_token"] = np.full(prompts.shape, max_output_token, dtype=np.int_)

        if top_k is not None:
            inputs["top_k"] = np.full(prompts.shape, top_k, dtype=np.int_)

        if top_p is not None:
            inputs["top_p"] = np.full(prompts.shape, top_p, dtype=np.single)

        if temperature is not None:
            inputs["temperature"] = np.full(prompts.shape, temperature, dtype=np.single)

        if random_seed is not None:
            inputs["random_seed"] = np.full(prompts.shape, random_seed, dtype=np.int_)

        if stop_words_list is not None:
            stop_words_list = np.char.encode(stop_words_list, "utf-8")
            inputs["stop_words_list"] = np.full((prompts.shape[0], len(stop_words_list)), stop_words_list)

        if bad_words_list is not None:
            bad_words_list = np.char.encode(bad_words_list, "utf-8")
            inputs["bad_words_list"] = np.full((prompts.shape[0], len(bad_words_list)), bad_words_list)

        if no_repeat_ngram_size is not None:
            inputs["no_repeat_ngram_size"] = np.full(prompts.shape, no_repeat_ngram_size, dtype=np.single)

        if task_id is not None:
            task_id = np.char.encode(task_id, "utf-8")
            inputs["task_id"] = np.full((prompts.shape[0], len([task_id])), task_id)

        if lora_uids is not None:
            lora_uids = np.char.encode(lora_uids, "utf-8")
            inputs["lora_uids"] = np.full((prompts.shape[0], len(lora_uids)), lora_uids)

        with DecoupledModelClient(self.url, self.model_name, init_timeout_s=init_timeout) as client:
            for partial_result_dict in client.infer_batch(**inputs):
                output_type = client.model_config.outputs[0].dtype
                if output_type == np.bytes_:
                    sentences = np.char.decode(partial_result_dict["outputs"].astype("bytes"), "utf-8")
                    yield sentences
                else:
                    yield partial_result_dict["outputs"]
