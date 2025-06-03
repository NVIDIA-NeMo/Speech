# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
import time
from abc import ABC
from typing import List, Optional

import numpy as np

from nemo.deploy.utils import str_list2numpy

use_pytriton = True
try:
    from pytriton.client import DecoupledModelClient, ModelClient
except Exception:
    use_pytriton = False


class NemoQueryLLMBase(ABC):
    """
    Abstract base class for querying a Large Language Model (LLM).

    Args:
    url (str): The URL of the inference server.
    model_name (str): The name of the model to be queried.
    """

    def __init__(self, url, model_name):
        self.url = url
        self.model_name = model_name


class NemoQueryLLMPyTorch(NemoQueryLLMBase):
    """
    Sends a query to Triton for LLM inference

    Example:
        from nemo.deploy import NemoTritonQueryLLMPyTorch

        nq = NemoTritonQueryLLMPyTorch(url="localhost", model_name="GPT-2B")

        prompts = ["hello, testing GPT inference", "another GPT inference test?"]
        output = nq.query_llm(
            prompts=prompts,
            max_length=100,
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

    # these arguments are explicitly defined in order to make it clear to user what they can pass
    # names and optionality should exactly match the get_triton_input() results for MegatronGPTDeployable
    def query_llm(
        self,
        prompts: List[str],
        use_greedy: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        add_BOS: Optional[bool] = None,
        all_probs: Optional[bool] = None,
        compute_logprob: Optional[bool] = None,
        end_strings: Optional[List[str]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        apply_chat_template: bool = False,
        n_top_logprobs: Optional[int] = None,
        init_timeout: float = 60.0,
        echo: Optional[bool] = None,
    ):
        """
        Query the Triton server synchronously and return a list of responses.

        Args:
            prompts (List(str)): list of sentences.
            use_greedy (bool): use greedy sampling, effectively the same as top_k=1
            temperature (float): A parameter of the softmax function, which is the last layer in the network.
            top_k (int): limits us to a certain number (K) of the top tokens to consider.
            top_p (float): limits us to the top tokens within a certain probability mass (p).
            repetition_penalty (float): penalty applied to repeated sequences, 1.0 means no penalty.
            add_BOS (bool): whether or not to add a BOS (beginning of sentence) token.
            all_probs (bool): when using compute_logprob, returns probabilities for all tokens in vocabulary.
            compute_logprob (bool): get back probabilities of all tokens in the sequence.
            end_strings (List(str)): list of strings which will terminate generation when they appear in the output.
            min_length (int): min generated tokens.
            max_length (int): max generated tokens.
            apply_chat_template (bool): applies chat template if its a chat model. Default: False
            init_timeout (flat): timeout for the connection.
        """
        prompts = str_list2numpy(prompts)
        inputs = {
            "prompts": prompts,
        }
        if use_greedy is not None:
            inputs["use_greedy"] = np.full(prompts.shape, use_greedy, dtype=np.bool_)
        if temperature is not None:
            inputs["temperature"] = np.full(prompts.shape, temperature, dtype=np.single)
        if top_k is not None:
            inputs["top_k"] = np.full(prompts.shape, top_k, dtype=np.int_)
        if top_p is not None:
            inputs["top_p"] = np.full(prompts.shape, top_p, dtype=np.single)
        if repetition_penalty is not None:
            inputs["repetition_penalty"] = np.full(prompts.shape, repetition_penalty, dtype=np.single)
        if add_BOS is not None:
            inputs["add_BOS"] = np.full(prompts.shape, add_BOS, dtype=np.bool_)
        if all_probs is not None:
            inputs["all_probs"] = np.full(prompts.shape, all_probs, dtype=np.bool_)
        if compute_logprob is not None:
            inputs["compute_logprob"] = np.full(prompts.shape, compute_logprob, dtype=np.bool_)
        if end_strings is not None:
            inputs["end_strings"] = str_list2numpy(end_strings)
        if min_length is not None:
            inputs["min_length"] = np.full(prompts.shape, min_length, dtype=np.int_)
        if max_length is not None:
            inputs["max_length"] = np.full(prompts.shape, max_length, dtype=np.int_)
        if apply_chat_template is not None:
            inputs["apply_chat_template"] = np.full(prompts.shape, apply_chat_template, dtype=np.bool_)
        if n_top_logprobs is not None:
            inputs["n_top_logprobs"] = np.full(prompts.shape, n_top_logprobs, dtype=np.int_)
        if echo is not None:
            inputs["echo"] = np.full(prompts.shape, echo, dtype=np.bool_)

        with ModelClient(self.url, self.model_name, init_timeout_s=init_timeout, inference_timeout_s=600) as client:
            result_dict = client.infer_batch(**inputs)
            output_type = client.model_config.outputs[0].dtype

            log_probs_output = None
            if "log_probs" in result_dict.keys():
                log_probs_output = result_dict["log_probs"]

            top_log_probs_output = None
            if "top_logprobs" in result_dict.keys():
                top_log_probs_output = result_dict["top_logprobs"]

            if output_type == np.bytes_:
                if "sentences" in result_dict.keys():
                    output = result_dict["sentences"]
                else:
                    return "Unknown output keyword."

                sentences = np.char.decode(output.astype("bytes"), "utf-8")
                openai_response = {
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{"text": sentences}],
                }

                if log_probs_output is not None:
                    # logprobs are stored under choices in openai format.
                    openai_response["choices"][0]["logprobs"] = {}
                    openai_response["choices"][0]["logprobs"]["token_logprobs"] = log_probs_output
                    # TODO athitten: get top_n_logprobs from mcore once available
                    if top_log_probs_output is not None:
                        # we take 1st element because cast_output adds an extra dimension
                        n_log_probs_output = [json.loads(top_log_prob[0]) for top_log_prob in top_log_probs_output]
                        openai_response["choices"][0]["logprobs"]["top_logprobs"] = n_log_probs_output
                return openai_response
            else:
                return result_dict["sentences"]


class NemoQueryLLMHF(NemoQueryLLMBase):
    """
    Sends a query to Triton for LLM inference

    Example:
        from nemo.deploy import NemoQueryLLMHF

        nq = NemoQueryLLMHF(url="localhost", model_name="GPT-2B")

        prompts = ["hello, testing GPT inference", "another GPT inference test?"]
        output = nq.query_llm(
            prompts=prompts,
            max_length=100,
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

    # these arguments are explicitly defined in order to make it clear to user what they can pass
    # names and optionality should exactly match the get_triton_input() results for HuggingFaceLLMDeploy
    def query_llm(
        self,
        prompts: List[str],
        use_greedy: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        add_BOS: Optional[bool] = None,
        all_probs: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        end_strings: Optional[List[str]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        init_timeout: float = 60.0,
    ):
        """
        Query the Triton server synchronously and return a list of responses.

        Args:
            prompts (List[str]): list of sentences.
            use_greedy (Optional[bool]): use greedy sampling, effectively the same as top_k=1
            temperature (Optional[float]): A parameter of the softmax function, which is the last layer in the network.
            top_k (Optional[int]): limits us to a certain number (K) of the top tokens to consider.
            top_p (Optional[float]): limits us to the top tokens within a certain probability mass (p).
            repetition_penalty (Optional[float]): penalty applied to repeated sequences, 1.0 means no penalty.
            add_BOS (Optional[bool]): whether or not to add a BOS (beginning of sentence) token.
            all_probs (Optional[bool]): when using compute_logprob, returns probabilities for all tokens in vocabulary.
            output_logits (Optional[bool]): whether to return logits for each token
            output_scores (Optional[bool]): whether to return scores for each token
            end_strings (Optional[List[str]]): list of strs which will stop generation when they appear in the output.
            min_length (Optional[int]): min generated tokens.
            max_length (Optional[int]): max generated tokens.
            init_timeout (float): timeout for the connection.
        """
        prompts = str_list2numpy(prompts)
        inputs = {
            "prompts": prompts,
        }
        if use_greedy is not None:
            inputs["use_greedy"] = np.full(prompts.shape, use_greedy, dtype=np.bool_)
        if temperature is not None:
            inputs["temperature"] = np.full(prompts.shape, temperature, dtype=np.single)
        if top_k is not None:
            inputs["top_k"] = np.full(prompts.shape, top_k, dtype=np.int_)
        if top_p is not None:
            inputs["top_p"] = np.full(prompts.shape, top_p, dtype=np.single)
        if repetition_penalty is not None:
            inputs["repetition_penalty"] = np.full(prompts.shape, repetition_penalty, dtype=np.single)
        if add_BOS is not None:
            inputs["add_BOS"] = np.full(prompts.shape, add_BOS, dtype=np.bool_)
        if all_probs is not None:
            inputs["all_probs"] = np.full(prompts.shape, all_probs, dtype=np.bool_)
        if output_logits is not None:
            inputs["output_logits"] = np.full(prompts.shape, output_logits, dtype=np.bool_)
        if output_scores is not None:
            inputs["output_scores"] = np.full(prompts.shape, output_scores, dtype=np.bool_)
        if end_strings is not None:
            inputs["end_strings"] = str_list2numpy(end_strings)
        if min_length is not None:
            inputs["min_length"] = np.full(prompts.shape, min_length, dtype=np.int_)
        if max_length is not None:
            inputs["max_length"] = np.full(prompts.shape, max_length, dtype=np.int_)

        with ModelClient(self.url, self.model_name, init_timeout_s=init_timeout) as client:
            result_dict = client.infer_batch(**inputs)
            output_type = client.model_config.outputs[0].dtype

            if output_type == np.bytes_:
                if "sentences" in result_dict.keys():
                    output = result_dict["sentences"]
                else:
                    return "Unknown output keyword."

                sentences = np.char.decode(output.astype("bytes"), "utf-8")
                openai_response = {
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{"text": sentences}],
                }
                if output_logits and "logits" in result_dict:
                    openai_response["logits"] = result_dict["logits"]
                if output_scores and "scores" in result_dict:
                    openai_response["scores"] = result_dict["scores"]
                return openai_response
            else:
                return result_dict["sentences"]


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
        min_output_len=None,
        max_output_len=None,
        top_k=None,
        top_p=None,
        temperature=None,
        random_seed=None,
        task_id=None,
        lora_uids=None,
        use_greedy: bool = None,
        repetition_penalty: float = None,
        add_BOS: bool = None,
        all_probs: bool = None,
        compute_logprob: bool = None,
        end_strings=None,
        init_timeout=60.0,
        openai_format_response: bool = False,
        output_context_logits: bool = False,
        output_generation_logits: bool = False,
    ):
        """
        Query the Triton server synchronously and return a list of responses.

        Args:
            prompts (List(str)): list of sentences.
            max_output_len (int): max generated tokens.
            top_k (int): limits us to a certain number (K) of the top tokens to consider.
            top_p (float): limits us to the top tokens within a certain probability mass (p).
            temperature (float): A parameter of the softmax function, which is the last layer in the network.
            random_seed (int): Seed to condition sampling.
            stop_words_list (List(str)): list of stop words.
            bad_words_list (List(str)): list of bad words.
            no_repeat_ngram_size (int): no repeat ngram size.
            task_id (str): downstream task id if virtual tokens are used.
            init_timeout (flat): timeout for the connection.
            openai_format_response: return response similar to OpenAI API format
            output_generation_logits: return generation logits from model on PyTriton
        """

        prompts = str_list2numpy(prompts)
        inputs = {"prompts": prompts}

        if min_output_len is not None:
            inputs["min_output_len"] = np.full(prompts.shape, max_output_len, dtype=np.int_)

        if max_output_len is not None:
            inputs["max_output_len"] = np.full(prompts.shape, max_output_len, dtype=np.int_)

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

        if use_greedy is not None:
            inputs["use_greedy"] = np.full(prompts.shape, use_greedy, dtype=np.bool_)

        if repetition_penalty is not None:
            inputs["repetition_penalty"] = np.full(prompts.shape, repetition_penalty, dtype=np.single)

        if add_BOS is not None:
            inputs["add_BOS"] = np.full(prompts.shape, add_BOS, dtype=np.bool_)

        if all_probs is not None:
            inputs["all_probs"] = np.full(prompts.shape, all_probs, dtype=np.bool_)

        if compute_logprob is not None:
            inputs["compute_logprob"] = np.full(prompts.shape, compute_logprob, dtype=np.bool_)

        if end_strings is not None:
            inputs["end_strings"] = str_list2numpy(end_strings)

        if output_context_logits is not None:
            inputs["output_context_logits"] = np.full(prompts.shape, output_context_logits, dtype=np.bool_)

        if output_generation_logits is not None:
            inputs["output_generation_logits"] = np.full(prompts.shape, output_generation_logits, dtype=np.bool_)

        with ModelClient(self.url, self.model_name, init_timeout_s=init_timeout) as client:
            result_dict = client.infer_batch(**inputs)
            output_type = client.model_config.outputs[0].dtype

            if output_type == np.bytes_:
                if "outputs" in result_dict.keys():
                    output = result_dict["outputs"]
                elif "sentences" in result_dict.keys():
                    output = result_dict["sentences"]
                else:
                    return "Unknown output keyword."

                sentences = np.char.decode(output.astype("bytes"), "utf-8")
                if openai_format_response:
                    openai_response = {
                        "id": f"cmpl-{int(time.time())}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": self.model_name,
                        "choices": [{"text": sentences}],
                    }
                    if output_generation_logits:
                        openai_response["choices"][0]["generation_logits"] = result_dict["generation_logits"]
                    if output_context_logits:
                        openai_response["choices"][0]["context_logits"] = result_dict["context_logits"]
                    return openai_response
                else:
                    return sentences
            else:
                return result_dict["outputs"]

    def query_llm_streaming(
        self,
        prompts,
        stop_words_list=None,
        bad_words_list=None,
        no_repeat_ngram_size=None,
        max_output_len=512,
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
            max_output_len (int): max generated tokens.
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

        if max_output_len is not None:
            inputs["max_output_len"] = np.full(prompts.shape, max_output_len, dtype=np.int_)

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
