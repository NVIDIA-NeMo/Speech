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

import sys
from typing import List, Tuple, Union

from torch import Tensor

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class LengthParam(TypedDict):
    max_length: int  # The maximum length of the sequence to be generated.
    min_length: int  # The minimum length of the sequence to be generated.


class SamplingParam(TypedDict):
    use_greedy: bool  # Whether or not to use sampling ; use greedy decoding otherwise
    temperature: float  # sampling temperature
    top_k: int  # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_p: float  # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    repetition_penalty: float  # The parameter for repetition penalty. 1.0 means no penalty.
    add_BOS: bool  # add the bos token at the begining of the prompt
    all_probs: bool  # whether return the log prob for all the tokens in vocab
    compute_logprob: bool  # a flag used to compute logprob of all the input text, a very special case of running inference, default False
    end_strings: List[str]  # generation will stop when one of these tokens is generated


class OutputType(TypedDict):
    sentences: List[str]  # output sentences
    tokens: List[List[str]]  # output sentences borken into tokens
    logprob: List[List[float]]  # log prob of generated tokens
    full_logprob: List[List[float]]  # log prob of all the tokens in the vocab
    token_ids: List[List[int]]  # output sentence token ids
    offsets: List[List[int]]  # list of tokens start positions in text


class TextGeneration:
    """
    Interface for all text generation models.
    """

    def generate(
        self,
        inputs: Union[List[str], Tuple[Tensor, Tensor], List[dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
    ) -> OutputType:
        """
        Public method to generate text.

        Args:
            inputs (Union[List[str], Tensor, List[dict]]):
                Can be one of the 3 types: 

                1. List of strings. Each element of the list provides input prompt. The model will apply tokenizer on it.
                    E.g [‘sentence’, ‘sentence2’ … ]

                2. Tuple of Pytorch Tensors (context_tokens, context_lengths). The `context_tokens` has shape (batch_size, seq_length),  it's the batched sequences of tokens used as a prompst for the generation or as model inputs to the encoder.
                    The generative model will skip the tokenization and padding step.  The `context_lengths` has shape (batch_size,), it indicates the length of the context tokens for each of the input sequences.
                    E.g. ( torch.tensor([[23,5234,23,35,…], [223,323,23,23232,232,...] …]), torch.tensor([20, 30, …]))

                3. List of python dict objects. Used for prompt/p-tuning inputs where a set of key-value pairs are converted into input token embeddings for the model.
                    E.g. [{"prompt-tag": "sentiment", "sentence": "this is a good movie"},
                    {"prompt-tag": "qa", "context": "some context text", "question": "a simple question"} ... ]
                    where 'prompt-tag' is used to identify the type of NLP task to solve.

            length_params (LengthParam):
                a dictionary type which controls the sampling length.

                * max_length: int, The maximum length of the sequence to be generated.
                * min_length: int,  The minimum length of the sequence to be generated.

                If None, max_length is set to 30, and min_length is set to None

            sampling_params (SamplingParam):
                a dictionary type which contains the parameters for text sampling. It has the following keys

                * use_greedy: bool,  Whether or not to use sampling ; use greedy decoding otherwise
                * top_k: int, The number of highest probability vocabulary tokens to keep for top-k-filtering.
                * top_p: float, If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
                * repetition_penalty: float, The parameter for repetition penalty. 1.0 means no penalty.
                * add_BOS: bool, Whether add the bos token at the begining of the prompt
                * all_probs: bool  # whether return the log prob for all the tokens in vocab
                * compute_logprob: bool  # a flag used to compute logprob of all the input text, a very special case of running inference, default False
                * end_strings: List[str]  # generation will stop when one of these tokens is generated

                Default None, If it is None, use_greedy will be "True".

        Returns:
            It generates the output in a dictionary type. It has the following keys,

            * sentences: List[str], output sentences
            * tokens: List[List[str]], output sentences borken into tokens
            * logprob: List[List[float]],  log prob of generated tokens
            * full_logprob: List[List[float]], log prob of all the tokens in the vocab
            * token_ids: List[List[int]], output sentence token ids
            * offsets: List[List[int]]  # list of tokens start positions in text
        """
        raise NotImplementedError("please implement this method")
