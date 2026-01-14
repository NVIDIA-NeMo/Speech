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
"""Gemma3VL checkpoint import."""

import argparse

from nemo.collections import llm, vlm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

MODEL_DICT = {
    "gemma3_vl_4b_it": ("google/gemma-3-4b-it", llm.Gemma3Config4B),
    "gemma3_vl_27b_it": ("google/gemma-3-27b-it", llm.Gemma3Config27B),
}


def main(args: argparse.Namespace):
    hf_model_name, language_config_class = MODEL_DICT[args.model]
    tokenizer = AutoTokenizer(hf_model_name)
    language_transformer_config = language_config_class()

    # The default cross_entropy_fusion_impl is `te`, which will not calculate
    # loss properly with label < 0.
    language_transformer_config.cross_entropy_fusion_impl = "native"

    vision_transformer_config = vlm.Gemma3VLVisionConfig()
    vision_projection_config = vlm.Gemma3VLMultimodalProjectorConfig(
        input_size=vision_transformer_config.hidden_size,
        hidden_size=language_transformer_config.hidden_size,
    )
    gemma3vl_config = vlm.Gemma3VLConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        freeze_language_model=False,
        freeze_vision_model=True,
        freeze_vision_projection=True,
    )
    model = vlm.Gemma3VLModel(gemma3vl_config, tokenizer=tokenizer)

    llm.import_ckpt(model=model, source=f"hf://{hf_model_name}", overwrite=args.overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma3VL checkpoint import.")
    parser.add_argument("--model", type=str, required=False, default="gemma3_vl_4b_it")
    parser.add_argument("--overwrite", type=bool, required=False, default=False)
    parsed_args = parser.parse_args()
    main(parsed_args)
