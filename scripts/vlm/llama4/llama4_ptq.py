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

"""
Example of PTQ for Llama4:
  torchrun --nproc_per_node=8 \
    scripts/vlm/llama4/llama4_ptq.py \
    --calibration_tp 8 \
    --nemo_checkpoint "path/to/nemo_checkpoint" \
    --output_path "path/to/quantized_nemo_checkpoint" \
    --algorithm fp8 \
    --batch_size 1 \
    --export_format nemo \
    --legacy_ckpt \
"""

import argparse

import requests
import torch
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from PIL import Image
from transformers import AutoProcessor

from nemo.collections.llm.modelopt import ExportConfig, QuantizationConfig
from nemo.collections.llm.modelopt.quantization.quant_cfg_choices import get_quant_cfg_choices
from nemo.collections.vlm.api import ptq


def load_image(url):
    """Load image from URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error loading image from {url}: {e}")
        return None


# Define calibration dataset URLs
base_img_url = "http://images.cocodataset.org/val2017/"
images = [
    "000000039769.jpg",
    "000000002685.jpg",
    "000000004495.jpg",
    "000000005001.jpg",
    "000000003845.jpg",
    "000000011615.jpg",
    "000000010977.jpg",
    "000000010764.jpg",
    "000000010707.jpg",
    "000000010583.jpg",
    "000000010363.jpg",
    "000000010092.jpg",
    "000000009914.jpg",
    "000000009891.jpg",
    "000000009769.jpg",
    "000000009590.jpg",
    "000000009483.jpg",
    "000000009448.jpg",
    "000000009378.jpg",
    "000000008899.jpg",
]
quantization_images_url = [base_img_url + img_id for img_id in images]


def get_args():
    """Parses PTQ arguments."""
    QUANT_CFG_CHOICES_LIST = ["no_quant", *get_quant_cfg_choices()]
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="NeMo PTQ argument parser"
    )
    parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source NeMo 2.0 checkpoint")
    parser.add_argument("--decoder_type", type=str, help="Decoder type for Model-Optimizer")
    parser.add_argument("-ctp", "--calibration_tp", "--calib_tp", type=int, default=1)
    parser.add_argument("-cpp", "--calibration_pp", "--calib_pp", type=int, default=1)
    parser.add_argument(
        "--num_layers_in_first_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the first pipeline stage. If None, pipeline parallelism will default to evenly split layers.",
    )
    parser.add_argument(
        "--num_layers_in_last_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the last pipeline stage. If None, pipeline parallelism will default to evenly split layers.",
    )
    parser.add_argument(
        "-itp",
        "--inference_tp",
        "--tensor_parallelism_size",
        type=int,
        default=1,
        help="TRT-LLM engine TP size. (Only used when `--export_format` is 'trtllm')",
    )
    parser.add_argument(
        "-ipp",
        "--inference_pp",
        "--pipeline_parallelism_size",
        type=int,
        default=1,
        help="TRT-LLM engine PP size. (Only used when `--export_format` is 'trtllm')",
    )
    parser.add_argument("--devices", type=int, help="Number of GPUs to use per node")
    parser.add_argument("-nodes", "--num_nodes", type=int, help="Number of nodes used")
    parser.add_argument("-out", "--export_path", "--output_path", type=str, help="Path for the exported engine")
    parser.add_argument(
        "--export_format", default="trtllm", choices=["trtllm", "nemo", "hf"], help="Model format to export as"
    )
    parser.add_argument(
        "-algo",
        "--algorithm",
        type=str,
        default="fp8",
        choices=QUANT_CFG_CHOICES_LIST,
        help="Model-Optimizer quantization algorithm",
    )
    parser.add_argument(
        "-awq_bs", "--awq_block_size", type=int, default=128, help="Block size for AWQ quantization algorithms"
    )
    parser.add_argument("--sq_alpha", type=float, default=0.5, help="Smooth-Quant alpha parameter")
    parser.add_argument("--enable_kv_cache", help="Enables KV-cache quantization", action="store_true")
    parser.add_argument("--disable_kv_cache", dest="enable_kv_cache", action="store_false")
    parser.set_defaults(enable_kv_cache=None)
    parser.add_argument(
        "-dt", "--dtype", default="bf16", choices=["16", "bf16"], help="Default precision for non-quantized layers"
    )
    parser.add_argument("-bs", "--batch_size", default=64, type=int, help="Calibration batch size")
    parser.add_argument("-sl", "--seq_len", default=128, type=int, help="Length of the tokenized text")
    parser.add_argument(
        "-calib_size", "--calibration_dataset_size", default=512, type=int, help="Size of calibration dataset"
    )
    parser.add_argument(
        "-calib_ds",
        "--calibration_dataset",
        default="cnn_dailymail",
        type=str,
        help='Calibration dataset to be used. Should be "wikitext", "cnn_dailymail" or path to a local .json file',
    )
    parser.add_argument(
        "--generate_sample", help="Generate sample model output after performing PTQ", action="store_true"
    )
    parser.add_argument(
        "--trust_remote_code", help="Trust remote code when loading HuggingFace models", action="store_true"
    )
    parser.add_argument("--legacy_ckpt", help="Load ckpt saved with TE < 1.14", action="store_true")
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        help="Model HuggingFace ID to use.",
    )

    args = parser.parse_args()

    if args.export_path is None:
        if args.export_format == "trtllm":
            args.export_path = f"./qnemo_{args.algorithm}_tp{args.inference_tp}_pp{args.inference_pp}"
        else:
            args.export_path = f"./{args.export_format}_{args.algorithm}"

    if args.devices is None:
        args.devices = args.calibration_tp
    if args.num_nodes is None:
        args.num_nodes = args.calibration_pp

    return args


class SingleBatchIterator:
    def __init__(self, images, input_ids, position_ids):
        self.batch = dict(
            media=images,
            tokens=input_ids,
            position_ids=position_ids,
            attention_mask=None,
        )
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def llama4_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    batch = next(data_iterator)

    forward_args = {
        "images": batch["media"],
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def main():
    """Example NeMo 2.0 Post Training Quantization workflow"""
    args = get_args()

    def forward_loop(model):
        """Forward loop for quantization calibration."""
        # Initialize processor and tokenizer
        model_id = args.model_id
        processor = AutoProcessor.from_pretrained(model_id)

        for img_url in quantization_images_url:
            raw_image = load_image(img_url)
            if raw_image is None:
                continue
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful visual assistant."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": img_url},
                        {"type": "text", "text": "Can you describe this image?"},
                    ],
                },
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].cuda()
            images = inputs["pixel_values"].cuda()
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0)

            batch_iterator = SingleBatchIterator(images, input_ids, position_ids)
            fwd_bwd_function = get_forward_backward_func()
            with torch.no_grad():
                output = fwd_bwd_function(
                    forward_step_func=llama4_forward_step,
                    data_iterator=batch_iterator,
                    model=model,
                    num_microbatches=1,
                    forward_only=True,
                    seq_length=input_ids.size(1),
                    micro_batch_size=1,
                    collect_non_loss_data=True,
                )

    quantization_config = QuantizationConfig(
        algorithm=None if args.algorithm == "no_quant" else args.algorithm,
        awq_block_size=args.awq_block_size,
        sq_alpha=args.sq_alpha,
        enable_kv_cache=args.enable_kv_cache,
        calibration_dataset=args.calibration_dataset,
        calibration_dataset_size=args.calibration_dataset_size,
        calibration_batch_size=args.batch_size,
        calibration_seq_len=args.seq_len,
    )
    export_config = ExportConfig(
        export_format=args.export_format,
        path=args.export_path,
        decoder_type=args.decoder_type,
        inference_tp=args.inference_tp,
        inference_pp=args.inference_pp,
        dtype=args.dtype,
        generate_sample=args.generate_sample,
    )

    ptq(
        model_path=args.nemo_checkpoint,
        export_config=export_config,
        calibration_tp=args.calibration_tp,
        calibration_pp=args.calibration_pp,
        num_layers_in_first_pipeline_stage=args.num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=args.num_layers_in_last_pipeline_stage,
        devices=args.devices,
        num_nodes=args.num_nodes,
        quantization_config=quantization_config,
        legacy_ckpt=args.legacy_ckpt,
        trust_remote_code=args.trust_remote_code,
        forward_loop=forward_loop,
    )


if __name__ == "__main__":
    main()
