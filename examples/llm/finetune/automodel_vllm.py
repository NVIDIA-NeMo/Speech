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

try:
    from nemo.export.vllm_hf_exporter import vLLMHFExporter
except ImportError:
    raise Exception("vLLM must be installed or activated in your environment:\n" "  source /opt/venv/bin/activate")

from nemo.deploy import DeployPyTriton
from nemo.deploy.nlp import NemoQueryLLM


def main():
    parser = argparse.ArgumentParser(
        description="Export a vLLM HF model, optionally apply LoRA, and optionally deploy to Triton."
    )

    # 1) Base model
    parser.add_argument(
        "--model",
        required=True,
        help="Local path or HuggingFace name of the base model",
    )

    # 2) Optional LoRA
    parser.add_argument(
        "--lora-model",
        help="Local path of a LoRA adapter to apply (optional)",
    )
    parser.add_argument(
        "--lora-name",
        default="lora_adapter",
        help="Logical name for the LoRA adapter (default: %(default)s)",
    )

    # 3) Optional Triton deploy
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy to Triton if set",
    )
    parser.add_argument(
        "--triton-model-name",
        help="Triton model name (required with --deploy)",
    )
    parser.add_argument(
        "--triton-model-version",
        type=int,
        default=1,
        help="Triton model version (default: %(default)s)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=64,
        help="Triton max batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8000,
        help="Triton HTTP port (default: %(default)s)",
    )
    parser.add_argument(
        "--address",
        default="0.0.0.0",
        help="Triton bind address (default: %(default)s)",
    )

    args = parser.parse_args()

    enable_lora = bool(args.lora_model)

    # Export base model (with LoRA enabled if requested)
    exporter = vLLMHFExporter()
    exporter.export(model=args.model, enable_lora=enable_lora)

    # Attach LoRA adapter if provided
    if enable_lora:
        exporter.add_lora_models(
            lora_model_name=args.lora_name,
            lora_model=args.lora_model,
        )

    # If not deploying, just do a local forward
    if not args.deploy:
        output = exporter.forward(
            input_texts=["How are you doing?"],
            lora_model_name=(args.lora_name if enable_lora else None),
        )
        print("Local forward output:", output)
        return

    # Validate Triton args
    if not args.triton_model_name:
        parser.error("--triton-model-name is required when --deploy is set")

    # Deploy to Triton
    server = DeployPyTriton(
        model=exporter,
        triton_model_name=args.triton_model_name,
        triton_model_version=args.triton_model_version,
        max_batch_size=args.max_batch_size,
        http_port=args.http_port,
        address=args.address,
    )

    try:
        server.deploy()
        server.run()

        # Query the deployed model
        client = NemoQueryLLM(
            url=f"localhost:{args.http_port}",
            model_name=args.triton_model_name,
        )
        resp = client.query_llm(
            prompts=["How are you doing?"],
            max_output_len=128,
            top_k=1,
            top_p=0.2,
            temperature=1.0,
        )
        print("Deployed Triton output:", resp)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
