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
"""Export Gemma3VL NeMo checkpoints to Hugging Face format."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from nemo.collections import llm


def main():
    parser = argparse.ArgumentParser(
        description=("Export NeMo vision language model checkpoint to Hugging Face format.")
    )
    parser.add_argument(
        "--nemo_ckpt_path",
        type=str,
        required=True,
        default=None,
        help="Path to the NeMo checkpoint directory.",
    )
    parser.add_argument(
        "--output_hf_path",
        type=str,
        required=True,
        default=None,
        help="Path to save the converted Hugging Face checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default=None,
        help="Name of the model on Hugging Face.",
    )

    args = parser.parse_args()

    llm.export_ckpt(
        path=Path(args.nemo_ckpt_path),
        target="hf",
        output_path=Path(args.output_hf_path),
        overwrite=True,
    )
    if args.model_name:
        # Copy necessary files if exist from HuggingFace for Gemma3VL model export.
        copy_file_list = [
            "preprocessor_config.json",
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ]
        for file_name in copy_file_list:
            try:
                downloaded_path = hf_hub_download(
                    repo_id=args.model_name,
                    filename=file_name,
                    local_dir=args.output_hf_path,
                )
                print(f"Downloaded {downloaded_path} during export gamma3vl models.")
            except Exception as e:
                print(f"Ignore {file_name} during export gamma3vl models.")


if __name__ == "__main__":
    main()
