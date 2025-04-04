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
from nemo.collections import vlm
from nemo.collections.llm import import_ckpt

HF_MODEL_ID_TO_NEMO_CLASS = {
    "llava-hf/llava-1.5-7b-hf": vlm.LlavaModel,
    "llava-hf/llava-1.5-13b-hf": vlm.LlavaModel,
    "meta-llama/Llama-3.2-11B-Vision": vlm.MLlamaModel,
    "meta-llama/Llama-3.2-90B-Vision": vlm.MLlamaModel,
    "meta-llama/Llama-3.2-11B-Vision-Instruct": vlm.MLlamaModel,
    "meta-llama/Llama-3.2-90B-Vision-Instruct": vlm.MLlamaModel,
    "OpenGVLab/InternViT-300M-448px-V2_5": vlm.InternViTModel,
    "google/siglip-base-patch16-224": vlm.SigLIPViTModel,
    "OpenGVLab/InternViT-6B-448px-V2_5": vlm.InternViTModel,
    "openai/clip-vit-large-patch14": vlm.CLIPViTModel,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Import NeMo checkpoint from Hugging Face format.")
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        required=True,
        help="Hugging Face model id or path to the Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the converted NeMo version Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "--nemo_class",
        type=str,
        default=None,
        help="If input is a local checkpoint path, specify the corresponding NeMo model class (e.g., 'vlm.LlavaModel').",
    )
    args = parser.parse_args()

    model_name_or_path = args.input_name_or_path
    local_path = Path(model_name_or_path)
    if local_path.exists():
        try:
            model_class = eval(args.nemo_class)
        except Exception as e:
            raise ValueError(f"Could not import the specified NeMo class '{args.nemo_class}': {e}")
    else:
        model_class = HF_MODEL_ID_TO_NEMO_CLASS[model_name_or_path]

    import_ckpt(model_class(), f"hf://{model_name_or_path}", output_path=args.output_path)
