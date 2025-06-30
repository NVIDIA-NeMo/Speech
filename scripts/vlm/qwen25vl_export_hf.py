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
from pathlib import Path
from nemo.collections import llm


def main():
    parser = argparse.ArgumentParser(description="Export NeMo checkpoint to Hugging Face format.")
    parser.add_argument(
        "--path",
        type=str,
        default="/root/.cache/nemo/models/Qwen/Qwen2.5-VL-3B-Instruct/",
        help="Path to the NeMo checkpoint directory. (Default: /root/.cache/nemo/models/Qwen/Qwen2.5-VL-3B-Instruct/)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="path/to/converted/hf/ckpt",
        help="Path to save the converted Hugging Face checkpoint. (Default: path/to/converted/hf/ckpt)",
    )

    args = parser.parse_args()

    llm.export_ckpt(
        path=Path(args.path),
        target='hf',
        output_path=Path(args.output_path),
        overwrite=True,
    )


if __name__ == '__main__':
    main()
