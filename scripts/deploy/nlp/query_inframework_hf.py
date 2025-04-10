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

import argparse
import sys

from nemo.deploy.nlp import NemoQueryLLMHF


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Query a HuggingFace model deployed on Triton Inference Server",
    )
    parser.add_argument(
        "-u",
        "--url",
        default="0.0.0.0",
        type=str,
        help="URL of the Triton Inference Server (e.g. localhost or IP address)",
    )
    parser.add_argument(
        "-mn", "--model_name", required=True, type=str, help="Name of the model as deployed on Triton server"
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("-p", "--prompt", required=False, type=str, help="Text prompt to send to the model")
    prompt_group.add_argument(
        "-pf", "--prompt_file", required=False, type=str, help="Path to file containing the prompt text"
    )
    parser.add_argument(
        "-mol", "--max_output_len", default=128, type=int, help="Maximum number of tokens to generate in the response"
    )
    parser.add_argument(
        "-tk", "--top_k", default=1, type=int, help="Number of highest probability tokens to consider for sampling"
    )
    parser.add_argument(
        "-tpp", "--top_p", default=0.0, type=float, help="Cumulative probability threshold for token sampling"
    )
    parser.add_argument(
        "-t",
        "--temperature",
        default=1.0,
        type=float,
        help="Temperature for controlling randomness in sampling (higher = more random)",
    )
    parser.add_argument(
        "-it",
        "--init_timeout",
        default=60.0,
        type=float,
        help="Timeout in seconds when initializing connection to Triton server",
    )
    parser.add_argument(
        "-ol", "--output_logits", default=False, action='store_true', help="Return raw logits from model output"
    )
    parser.add_argument(
        "-os",
        "--output_scores",
        default=False,
        action='store_true',
        help="Return token probability scores from model output",
    )

    args = parser.parse_args(argv)
    return args


def query_llm(
    url,
    model_name,
    prompts,
    max_output_len=128,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    output_logits=False,
    output_scores=False,
    init_timeout=60.0,
):
    """Query a HuggingFace language model deployed on Triton Inference Server.

    Args:
        url (str): URL of the Triton Inference Server (e.g. localhost or IP address)
        model_name (str): Name of the model as deployed on Triton server
        prompts (List[str]): List of text prompts to send to the model
        max_output_len (int, optional): Maximum number of tokens to generate in the response. Defaults to 128.
        top_k (int, optional): Number of highest probability tokens to consider for sampling. Defaults to 1.
        top_p (float, optional): Cumulative probability threshold for token sampling. Defaults to 0.0.
        temperature (float, optional): Temperature for controlling randomness in sampling (higher = more random). Defaults to 1.0.
        output_logits (bool, optional): Return raw logits from model output. Defaults to False.
        output_scores (bool, optional): Return token probability scores from model output. Defaults to False.
        init_timeout (float, optional): Timeout in seconds when initializing connection to Triton server. Defaults to 60.0.

    Returns:
        List[str]: Generated text responses for each input prompt
    """

    nemo_query = NemoQueryLLMHF(url, model_name)
    return nemo_query.query_llm(
        prompts=prompts,
        max_length=max_output_len,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        output_logits=output_logits,
        output_scores=output_scores,
        init_timeout=init_timeout,
    )


def query(argv):
    """Query a HuggingFace language model deployed on Triton Inference Server using command line arguments.

    This function parses command line arguments and sends queries to a deployed model. It supports
    reading prompts either directly from command line or from a file.

    Args:
        argv (List[str]): Command line arguments passed to the script, excluding the script name.
            Expected arguments include:
            - url: URL of Triton server
            - model_name: Name of deployed model
            - prompt: Text prompt or prompt_file: Path to file containing prompt
            - max_output_len: Maximum tokens to generate
            - top_k: Top-k sampling parameter
            - top_p: Top-p sampling parameter
            - temperature: Sampling temperature
            - output_logits: Whether to return logits
            - output_scores: Whether to return scores
            - init_timeout: Connection timeout

    Returns:
        List[str]: Generated text responses from the model
    """

    args = get_args(argv)

    if args.prompt_file is not None:
        with open(args.prompt_file, "r") as f:
            args.prompt = f.read()

    outputs = query_llm(
        url=args.url,
        model_name=args.model_name,
        prompts=[args.prompt],
        max_output_len=args.max_output_len,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        output_logits=args.output_logits,
        output_scores=args.output_scores,
        init_timeout=args.init_timeout,
    )
    print(outputs)


if __name__ == '__main__':
    query(sys.argv[1:])
