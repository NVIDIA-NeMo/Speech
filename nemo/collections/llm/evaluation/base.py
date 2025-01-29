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

import re
import numpy as np

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm import tqdm

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.deploy.nlp import NemoQueryLLM
from nemo.utils import logging


class NeMoFWLMEval(LM):
    """
    NeMoFWLMEval is a wrapper class subclassing lm_eval.api.model.LM class, that defines how lm_eval interfaces with
    NeMo model deployed on PyTriton server.
    Created based on: https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.4/docs/model_guide.md
    """

    def __init__(
        self, model_name, api_url, tokenizer, batch_size, max_tokens_to_generate, temperature, top_p, top_k, add_bos
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_tokens_to_generate = max_tokens_to_generate
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.add_bos = add_bos
        super().__init__()

    def _generate_tokens_logits(self, payload, single_prediction_token: bool = False, return_logits: bool = False):
        """
        A private method that sends post request to the model on PyTriton server and returns either generated text or
        logits.
        """
        nq = NemoQueryLLM(url=self.api_url, model_name=payload['model'])

        output_context_logits = False
        output_generation_logits = False
        if return_logits:  # in case of loglikelihood type tasks
            if single_prediction_token:
                # In case of single token prediction like mmlu return only the generation logits
                output_generation_logits = True
            else:
                # In case of multiple token prediction return the full context logits
                output_context_logits = True
        response = nq.query_llm(
            prompts=payload['prompt'] if isinstance(payload['prompt'], list) else [payload['prompt']],
            max_output_len=payload['max_tokens'],
            top_k=payload['top_k'],
            top_p=payload['top_p'],
            temperature=payload['temperature'],
            output_context_logits=output_context_logits,
            output_generation_logits=output_generation_logits,
            openai_format_response=True,
        )

        if return_logits:  # loglikelihood type tasks, return just logits and not text
            if output_context_logits:
                return response["choices"][0]["context_logits"]
            else:
                return response["choices"][0]["generation_logits"]
        else:  # generate_until type tasks, return just text and not logits
            return str(response["choices"][0]["text"])

    def tokenizer_type(self, tokenizer):
        """
        Returns the type of the tokenizer.
        """
        if isinstance(tokenizer, AutoTokenizer):
            return "AutoTokenizer"
        elif isinstance(tokenizer, SentencePieceTokenizer):
            return "SentencePieceTokenizer"
        else:
            raise ValueError(
                "Tokenizer type is not one of SentencePieceTokenizer or HF's AutoTokenizer. Please check "
                "how to handle special tokens for this tokenizer"
            )

    def loglikelihood(self, requests: list[Instance]):
        """
        Defines the loglikelihood request. Takes input requests of type list[Instance] where Instance is a dataclass
        defined in lm_eval.api.instance. Each Instance conists of the input prompt, output prompt, request type(here
        loglikelihood) and other relevant args like few shot samples.
        """
        special_tokens_kwargs = {}
        tokenizer_type = self.tokenizer_type(self.tokenizer)
        if tokenizer_type == "SentencePieceTokenizer":
            special_tokens_kwargs['add_bos'] = self.add_bos
        elif tokenizer_type == "AutoTokenizer":
            special_tokens_kwargs['add_special_tokens'] = self.add_bos

        single_prediction_token = False
        # Assuming evaluating on only one benchmark/task at a time, hence all instances in requests are of the same
        # task.
        mmlu_regex_pattern = r"^mmlu_"
        if re.match(mmlu_regex_pattern, requests[0].task_name):
            # in case of mmlu the output token is one of 'a','b','c','d'
            single_prediction_token = True

        # Hard code max_tokens_to_generate to 1 to always generate just 1 token in case of loglikelihood type tasks
        self.max_tokens_to_generate = 1

        results = []
        for i in tqdm(range(0, len(requests), self.batch_size)):
            # Group requests into batches
            batch = requests[i : i + self.batch_size]
            prompts = []
            continuations = []
            continuation_encs = []
            num_ctx_tokens_list = []
            num_cont_tokens_list = []
            # Prepare inputs for the batch
            for request in batch:
                # get the input prompt from the request
                context = request.arguments[0]
                # get the output prompt from the request
                continuation = request.arguments[1]
                # get encoded tokens of context
                context_enc = self.tokenizer.tokenizer.encode(context, **special_tokens_kwargs)
                # get encoded tokens of continuation
                continuation_enc = self.tokenizer.tokenizer.encode(continuation, **special_tokens_kwargs)
                # for SentencePeice consider the encoded tokens from the 2nd token since first encoded token is space.
                if self.tokenizer_type(self.tokenizer) == "SentencePieceTokenizer":
                    context_enc = context_enc[1:]
                    continuation_enc = continuation_enc[1:]
                num_ctx_tokens = len(context_enc)
                num_cont_tokens = len(continuation_enc)
                # Delete the last token from continuation before passing it to the ip prompt by replacing with empty
                # string
                prompt = context + continuation.replace(self.tokenizer.tokenizer.decode(continuation_enc[-1]), "")

                prompts.append(prompt)
                continuations.append(continuation)
                continuation_encs.append(continuation_enc)
                num_ctx_tokens_list.append(num_ctx_tokens)
                num_cont_tokens_list.append(num_cont_tokens)

            # Create a single payload for the entire batch
            payload = {
                "model": self.model_name,
                "prompt": prompts,
                "max_tokens": self.max_tokens_to_generate,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
            }

            # Query the model deployed on PyTriton server with the batched payload to get the logits
            logits_batch = self._generate_tokens_logits(payload, single_prediction_token, return_logits=True)

            # Process each result in the batch
            for j, logits in enumerate(logits_batch):
                continuation_enc = continuation_encs[j]
                num_ctx_tokens = num_ctx_tokens_list[j]
                num_cont_tokens = num_cont_tokens_list[j]

                # In case of multiple token prediction where full context logits are returned (tasks other than mmlu),
                # get only logits corresponding to the continuation tokens from context logits tensor. context_logits
                # contains logits for all tokens in the ip prompt along with the logit for the next token prediction
                # after the final token in the prompt. Shape of context_logits: [1, #tokens_in_prompt+1, vocab_size].
                if not single_prediction_token:
                    # Discard zero padding if any
                    logits = logits[:, np.any(logits != 0, axis=(0, 2)), :]
                    # Get only logits corresponding to cont tokens
                    logits = logits[:, -num_cont_tokens:, :]
                # Convert logits to torch tensor to easily get logprobs wo manual implementation of log_softmax
                logProbs = F.log_softmax(torch.tensor(logits), dim=-1)
                # Convert encoded continuation tokens to torch tensor
                cont_toks = torch.tensor(continuation_enc, dtype=torch.long).unsqueeze(0)
                # Get the greedy token from the logits (i.e token with the highest prob)
                greedy_tokens = logProbs.argmax(dim=-1)
                # Check if all greedy_tokens match the the actual continuation tokens
                is_greedy = (greedy_tokens == cont_toks).all()
                # Get the logits corresponding to the actual continuation tokens
                logProbs_actual = torch.gather(logProbs, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
                # result is tuple of logProb of generating the continuation token and is_greedy
                result = (float(logProbs_actual.sum()), bool(is_greedy))
                # Append the result of this input in the batch to results list
                results.append(result)

        return results

    def loglikelihood_rolling(self, requests: list[Instance]):
        """
        Defines the loglikelihood_rolling request type. Yet to be implemented.
        """
        pass

    def generate_until(self, inputs: list[Instance]):
        """
        Defines the generate_until request type. Takes input requests of type list[Instance] where Instance is a
        dataclass defined in lm_eval.api.instance. Each Instance conists of the input prompt, output prompt, request
        type(here loglikelihood) and other relevant args like few shot samples.
        """
        results = []
        for instance in tqdm(inputs):
            # Access the 'arguments' attribute of the Instance which contains the input prompt string
            prompt = instance.arguments[0]
            # Create payload to query the model deployed on PyTriton server
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": self.max_tokens_to_generate,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
            }
            # Get the text generated by the model
            generated_text = self._generate_tokens_logits(payload)

            results.append(generated_text)

        return results


def wait_for_server_ready(url, triton_http_port, model_name, max_retries=600, retry_interval=2):
    """
    Wait for PyTriton server and model to be ready.

    Args:
        url (str): The URL of the Triton server (e.g., "grpc://0.0.0.0:8001").
        triton_http_port (int): http port of the triton server.
        model_name (str): The name of the deployed model.
        max_retries (int): Maximum number of retries before giving up.
        retry_interval (int): Time in seconds to wait between retries.

    Returns:
        bool: True if both the server and model are ready within the retries, False otherwise.
    """

    import time

    import requests
    from pytriton.client import ModelClient
    from pytriton.client.exceptions import PyTritonClientModelUnavailableError, PyTritonClientTimeoutError

    # If gRPC URL, extract HTTP URL from gRPC URL for health checks
    if url.startswith("grpc://"):
        # Extract the gRPC port using regex
        pattern = r":(\d+)"  # Matches a colon followed by one or more digits
        match = re.search(pattern, url)
        grpc_port = match.group(1)
        # Replace 'grpc' with 'http' and replace the grpc_port with http port
        url = url.replace("grpc://", "http://").replace(f":{grpc_port}", f":{triton_http_port}")
    health_url = f"{url}/v2/health/ready"

    for _ in range(max_retries):
        logging.info("Checking server and model readiness...")

        try:
            # Check server readiness using HTTP health endpoint
            response = requests.get(health_url)
            if response.status_code != 200:
                logging.info(f"Server is not ready. HTTP status code: {response.status_code}")
                time.sleep(retry_interval)
                continue
            logging.info("Server is ready.")

            # Check model readiness using ModelClient
            with ModelClient(url, model_name=model_name, init_timeout_s=retry_interval) as client:
                logging.info(f"Model '{model_name}' is ready.")
                return True

        except PyTritonClientTimeoutError:
            logging.info(f"Timeout: Server or model '{model_name}' not ready yet.")
        except PyTritonClientModelUnavailableError:
            logging.info(f"Model '{model_name}' is unavailable on the server.")
        except requests.exceptions.RequestException:
            logging.info(f"Pytriton server not ready yet. Retrying in {retry_interval} seconds...")

        # Wait before retrying
        time.sleep(retry_interval)

    logging.error(f"Server or model '{model_name}' not ready after {max_retries} attempts.")
    return False
