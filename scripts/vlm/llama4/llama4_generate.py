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
Example:
  python scripts/vlm/llama4/llama4_generate.py --local_model_path="path/to/converted_nemo_checkpoint"
"""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

import nemo.lightning as nl
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.vlm.inference.base import _setup_trainer_and_restore_model
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.utils.get_rank import get_last_rank


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


def main(args) -> None:
    # pylint: disable=C0115,C0116
    tp = args.tp
    pp = args.pp
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tp,
        expert_tensor_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        ckpt_include_optimizer=False,
        ckpt_load_strictness="log_all",
        pipeline_dtype=torch.bfloat16,
    )
    trainer = nl.Trainer(
        devices=min(tp * pp, 8),
        num_nodes=max(tp * pp // 8, 1),
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )

    # Decide whether to import or load the model based on the input arguments
    if args.load_from_hf:
        raise NotImplementedError("Please use the convert script to convert the HF checkpoint first.")
    else:
        path = Path(args.local_model_path)
        model: io.TrainerContext = io.load_context(path=ckpt_to_context_subdir(path), subpath="model")
        _setup_trainer_and_restore_model(path=path, trainer=trainer, model=model)

    model = model.module.cuda()
    model.eval()

    from transformers import AutoProcessor

    model_id = 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
    processor = AutoProcessor.from_pretrained(model_id)
    llama_tokenizer = AutoTokenizer(model_id)
    hf_tokenizer = llama_tokenizer.tokenizer

    url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    url2 = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
    )
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
                {"type": "image", "url": url1},
                {"type": "image", "url": url2},
                {"type": "text", "text": "Can you describe how these two images are similar, and how they differ?"},
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

    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    generated_ids = input_ids.clone()

    stop_tokens = [hf_tokenizer.vocab["<|eom|>"], hf_tokenizer.vocab["<|eot|>"]]
    # Greedy generation loop
    for step in range(50):
        with torch.no_grad():
            if torch.distributed.get_rank() == 0:
                print(step)
            fwd_bwd_function = get_forward_backward_func()

            iterator = SingleBatchIterator(images, input_ids, position_ids)

            output = fwd_bwd_function(
                forward_step_func=llama4_forward_step,
                data_iterator=iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                # All-gather operation
                dist.all_gather(gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group())
                # Concatenate along last dimension (dim=2)
                output = torch.cat(gathered_tensors, dim=2)

                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() in stop_tokens:
                break

    generated_texts = hf_tokenizer.decode(list(generated_ids[0]))
    if torch.distributed.get_rank() == 0:
        print("======== GENERATED TEXT OUTPUT ========")
        print(f"{generated_texts}")
        print("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama4 Multimodal Inference")
    parser.add_argument(
        "--load_from_hf",
        action="store_true",
        help="Flag to indicate whether to load the model from Hugging Face hub.",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default="/path/to/nemo_checkpoint",
        help="Local path to the model if not loading from Hugging Face.",
    )
    parser.add_argument('--tp', default=8)
    parser.add_argument('--pp', default=1)
    parser.add_argument('--num_experts', type=int, default=16)
    args = parser.parse_args()

    main(args)
