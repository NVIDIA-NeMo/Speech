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

import argparse
import subprocess

from nemo.collections.llm import evaluate
from nemo.collections.llm.evaluation.api import ApiEndpoint, ConfigParams, EvaluationConfig, EvaluationTarget
from nemo.collections.llm.evaluation.base import wait_for_server_ready
from nemo.utils import logging


def get_args():
    parser = argparse.ArgumentParser(
        description='Test evaluation with lm-eval-harness on nemo2 model deployed on PyTriton'
    )
    parser.add_argument('--nemo2_ckpt_path', type=str, help="NeMo 2.0 ckpt path")
    parser.add_argument('--max_batch_size', type=int, help="Max BS for the model for deployment")
    parser.add_argument(
        '--trtllm_dir',
        type=str,
        help="Folder for the trt-llm conversion, trt-llm engine gets saved \
                        in this specified dir",
    )
    parser.add_argument('--eval_type', type=str, help="Evaluation benchmark to run from lm-eval-harness")
    parser.add_argument('--limit', type=int, help="Limit evaluation to `limit` num of samples")

    return parser.parse_args()


def run_deploy(args):
    return subprocess.Popen(
        [
            "python",
            "tests/evaluation/deploy_script.py",
            "--nemo2_ckpt_path",
            args.nemo2_ckpt_path,
            "--max_batch_size",
            str(args.max_batch_size),
            "--trtllm_dir",
            args.trtllm_dir,
        ]
    )


if __name__ == '__main__':
    args = get_args()
    deploy_proc = run_deploy(args)

    # Evaluation code
    logging.info("Waiting for server readiness...")
    server_ready = wait_for_server_ready(max_retries=30)
    if server_ready:
        logging.info("Starting evaluation...")
        api_endpoint = ApiEndpoint(nemo_checkpoint_path=args.nemo2_ckpt_path)
        eval_target = EvaluationTarget(api_endpoint=api_endpoint)
        # Run eval with just 1 sample from arc_challenge
        eval_params = ConfigParams(limit_samples=args.limit)
        eval_config = EvaluationConfig(type=args.eval_type, params=eval_params)

        evaluate(target_cfg=eval_target, eval_cfg=eval_config)
        logging.info("Evaluation completed.")
    else:
        logging.error("Server is not ready.")
    # After evaluation, terminate deploy_proc
    deploy_proc.terminate()
    deploy_proc.wait()
