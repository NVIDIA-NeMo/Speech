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

from nemo.collections.llm import api
from nemo.utils import logging


def get_args():
    parser = argparse.ArgumentParser(
        description='Test evaluation with lm-eval-harness on nemo2 model deployed on PyTriton'
    )
    parser.add_argument('--nemo2_ckpt_path', type=str, help="NeMo 2.0 ckpt path")
    parser.add_argument('--max_batch_size', type=int, help="Max BS for the model")
    parser.add_argument('--legacy_ckpt', action="store_true", help="Whether the nemo checkpoint is in legacy format")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    try:
        api.deploy(
            nemo_checkpoint=args.nemo2_ckpt_path,
            max_batch_size=args.max_batch_size,
            fastapi_port=8886,
            legacy_ckpt=args.legacy_ckpt,
        )
    except Exception as e:
        logging.error(f"Deploy process encountered an error: {e}")
    logging.info("Deploy process terminated.")
