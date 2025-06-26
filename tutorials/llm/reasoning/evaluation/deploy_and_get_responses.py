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

import argparse
import json
import signal
import subprocess

import requests

from nemo.collections.llm import api
from nemo.collections.llm.evaluation.base import wait_for_fastapi_server
from nemo.utils import logging

logging.setLevel(logging.INFO)

deploy_process = None
base_url = None
chat_url = None
model_name = None


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model on benchmark dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['gpqa_main', 'mmlu', 'gpqa_diamond'],
        help='Dataset to evaluate on (gpqa, mmlu)',
    )
    parser.add_argument(
        '--output_prefix', type=str, default='evaluation_results', help='Prefix for the output file name'
    )
    parser.add_argument(
        '--max_tokens', type=int, default=2048, help='Maximum number of tokens to generate in the response'
    )
    return parser.parse_args()


def create_benchmark_prompt(question, choice1, choice2, choice3, choice4):
    """Create benchmark prompt in the specified format"""
    prompt = f"""Given the following question and four candidate answers (A, B, C and D), choose the best answer.
        Question: {question} A. {choice1} B. {choice2} C. {choice3} D. {choice4}
        For simple problems, directly provide the answer with minimal explanation. For complex problems, use step-by-step format. Always conclude with: The final answer is [the_answer_letter], where the [the_answer_letter] is one of A, B, C or D."""
    return prompt


def load_model(checkpoint_path):
    """Initialize and load the model for inference"""
    global deploy_process, base_url, chat_url, model_name

    SCRIPTS_PATH = "/opt/NeMo/scripts"
    WORKSPACE = "."

    deploy_script = f"{SCRIPTS_PATH}/deploy/nlp/deploy_in_fw_oai_server_eval.py"
    deploy_process = subprocess.Popen(
        ['python', deploy_script, '--nemo_checkpoint', checkpoint_path],
    )

    base_url = "http://0.0.0.0:8886"
    model_name = "triton_model"
    chat_url = f"{base_url}/v1/chat/completions/"

    wait_for_fastapi_server(base_url=base_url, max_retries=600, retry_interval=10)
    logging.info("Model loaded and server is ready for inference")


def get_response(prompt, max_tokens):
    chat_payload = {
        "messages": [{"role": "system", "content": "detailed thinking on"}, {"role": "user", "content": prompt}],
        "model": model_name,
        "max_tokens": max_tokens,
    }
    response = requests.post(chat_url, json=chat_payload)
    return response.content.decode()


def main():
    args = parse_args()

    # Determine dataset file and output file based on dataset selection
    dataset_files = {
        'gpqa_main': 'gpqa_dataset.jsonl',
        'mmlu': 'mmlu_dataset_test.jsonl',
        'gpqa_diamond': 'gpqa_diamond_dataset.jsonl',
    }

    dataset_file = dataset_files[args.dataset]
    output_file = f"{args.output_prefix}_{args.dataset}_evaluation.jsonl"

    try:
        with open(dataset_file, "r") as f:
            problems = [json.loads(line) for line in f]

        load_model(args.checkpoint_path)

        # Open output file once before the loop
        with open(output_file, "w") as f:
            for i, problem in enumerate(problems):
                print(f"\n{'='*70}")
                print(f"Problem {i+1}/{len(problems)}")

                prompt = create_benchmark_prompt(
                    problem['Question'],
                    problem['Choice 1'],
                    problem['Choice 2'],
                    problem['Choice 3'],
                    problem['Choice 4'],
                )

                response = get_response(prompt, args.max_tokens)

                # Create result entry
                result = {
                    "question": problem['Question'],
                    "choices": {
                        "A": problem['Choice 1'],
                        "B": problem['Choice 2'],
                        "C": problem['Choice 3'],
                        "D": problem['Choice 4'],
                    },
                    "expected_answer": problem['Answer'],
                    "model_response": response,
                }

                # Write to JSONL file
                f.write(json.dumps(result) + "\n")

            print(f"All results written to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Killing the server...")
        deploy_process.send_signal(signal.SIGINT)


if __name__ == "__main__":
    main()
