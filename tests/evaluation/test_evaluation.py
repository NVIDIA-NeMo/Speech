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
import signal
import subprocess

from nemo.collections.llm import evaluate
from nemo.collections.llm.evaluation.api import ApiEndpoint, ConfigParams, EvaluationConfig, EvaluationTarget
from nemo.collections.llm.evaluation.base import wait_for_fastapi_server
from nemo.utils import logging


def get_args():
    parser = argparse.ArgumentParser(
        description='Test evaluation with NVIDIA Evals Factory on nemo2 model deployed on PyTriton'
    )
    parser.add_argument('--nemo2_ckpt_path', type=str, help="NeMo 2.0 ckpt path")
    parser.add_argument('--tokenizer_path', type=str, default=None, help="Path to the tokenizer")
    parser.add_argument('--max_batch_size', type=int, help="Max BS for the model for deployment")
    parser.add_argument('--eval_type', type=str, help="Evaluation benchmark to run from NVIDIA Evals Factory")
    parser.add_argument('--limit', type=int, help="Limit evaluation to `limit` num of samples")
    parser.add_argument('--legacy_ckpt', action="store_true", help="Whether the nemo checkpoint is in legacy format")

    return parser.parse_args()


def run_deploy(args):
    return subprocess.Popen(
        [
            "python",
            "tests/evaluation/deploy_in_fw_script.py",
            "--nemo2_ckpt_path",
            args.nemo2_ckpt_path,
            "--max_batch_size",
            str(args.max_batch_size),
        ]
        + (["--legacy_ckpt"] if args.legacy_ckpt else []),
    )


if __name__ == '__main__':
    args = get_args()
    deploy_proc = run_deploy(args)

    # Evaluation code
    logging.info("Waiting for server readiness...")
    server_ready = wait_for_fastapi_server(base_url="http://0.0.0.0:8886", max_retries=120)
    if server_ready:
        logging.info("Starting evaluation...")
        api_endpoint = ApiEndpoint(url="http://0.0.0.0:8886/v1/completions/")
        eval_target = EvaluationTarget(api_endpoint=api_endpoint)
        # Run eval with just 1 sample from selected task
        eval_params = {
            "limit_samples": args.limit,
        }
        if args.tokenizer_path is not None:
            eval_params["extra"] = {
                "tokenizer_backend": "huggingface",
                "tokenizer": args.tokenizer_path,
            }
        eval_config = EvaluationConfig(type=args.eval_type, params=ConfigParams(**eval_params))
        evaluate(target_cfg=eval_target, eval_cfg=eval_config)
        logging.info("Evaluation completed.")
        deploy_proc.send_signal(signal.SIGINT)
    else:
        deploy_proc.send_signal(signal.SIGINT)
        raise RuntimeError("Server is not ready. Please look the deploy process log for the error")
