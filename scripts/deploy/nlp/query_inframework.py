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

from nemo.deploy.nlp.query_llm import NemoQueryLLMPyTorch


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Queries Triton server running an in-framework Nemo model",
    )
    parser.add_argument("-u", "--url", default="0.0.0.0", type=str, help="url for the triton server")
    parser.add_argument("-mn", "--model_name", required=True, type=str, help="Name of the triton model")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("-p", "--prompt", required=False, type=str, help="Prompt")
    prompt_group.add_argument("-pf", "--prompt_file", required=False, type=str, help="File to read the prompt from")
    parser.add_argument("-mol", "--max_output_len", default=128, type=int, help="Max output token length")
    parser.add_argument("-tk", "--top_k", default=1, type=int, help="top_k")
    parser.add_argument("-tpp", "--top_p", default=0.0, type=float, help="top_p")
    parser.add_argument("-t", "--temperature", default=1.0, type=float, help="temperature")
    parser.add_argument("-it", "--init_timeout", default=60.0, type=float, help="init timeout for the triton server")

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
    init_timeout=60.0,
):
    nemo_query = NemoQueryLLMPyTorch(url, model_name)
    return nemo_query.query_llm(
        prompts=prompts,
        max_length=max_output_len,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        init_timeout=init_timeout,
    )


def query(argv):
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
        init_timeout=args.init_timeout,
    )
    print(outputs["sentences"][0][0])


if __name__ == '__main__':
    query(sys.argv[1:])
