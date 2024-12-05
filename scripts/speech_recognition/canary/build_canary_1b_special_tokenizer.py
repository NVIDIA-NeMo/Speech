#!/usr/bin/env python
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
from pathlib import Path

import click

from nemo.collections.common.tokenizers import CanaryTokenizer


@click.command()
@click.argument("output_dir", type=click.Path())
def main(output_dir: str) -> None:
    """
    Builds the special tokens tokenizer for NVIDIA Canary-1B model.
    It's intended to be used with CanaryTokenizer (a specialized AggregateTokenizer)
    under name ``spl_tokens``.
    """
    CanaryTokenizer.build_special_tokenizer(
        [
            "<|endoftext|>",
            "<|startoftranscript|>",
            "<|transcribe|>",
            "<|translate|>",
            "<|nopnc|>",
            "<|pnc|>",
            "<|nospeech|>",
        ]
        + [
            "<|en|>",
            "<|es|>",
            "<|de|>",
            "<|fr|>",
        ]
        + [f"<|spltoken{i}|>" for i in range(16)],
        model_dir=output_dir,
        force_rebuild=True,
    )


if __name__ == "__main__":
    main()
