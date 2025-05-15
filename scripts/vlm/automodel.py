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
"""
Usage example:
    python scripts/vlm/gemma3_automodel.py --model llava-hf/llava-v1.6-mistral-7b-hf --data_path naver-clova-ix/cord-v2

Used Automodel for training and finetuning HF models. More details can be found at:
https://docs.nvidia.com/nemo-framework/user-guide/latest/automodel/index.html
"""


import fiddle as fdl
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger

from nemo import lightning as nl
from nemo.automodel.dist_utils import FirstRankPerNode
from nemo.collections import llm, vlm
from nemo.collections.vlm.hf.data.automodel_datasets import (
    mk_hf_vlm_dataset_cord_v2,
    mk_hf_vlm_dataset_fineweb_edu,
    mk_hf_vlm_dataset_rdr,
)


def make_strategy(strategy, model, devices, num_nodes, adapter_only=False):
    if strategy == 'auto':
        return pl.strategies.SingleDeviceStrategy(
            device='cuda:0',
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )
    elif strategy == 'ddp':
        return pl.strategies.DDPStrategy(
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
            find_unused_parameters=True,
        )
    elif strategy == 'fsdp2':
        return nl.FSDP2Strategy(
            data_parallel_size=devices * num_nodes,
            tensor_parallel_size=1,
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )
    else:
        raise NotImplementedError("Encountered unknown strategy")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='gemma3_automodel')
    parser.add_argument('--model', type=str, default="google/gemma-3-4b-it")
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'fsdp2'])
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--num-nodes', default=1, type=int)
    parser.add_argument('--mbs', default=1, type=int)
    parser.add_argument('--gbs', default=4, type=int)
    parser.add_argument(
        "--log_dir", type=str, required=False, default="/results", help="Directory for logging and checkpoints"
    )
    parser.add_argument('--accelerator', default='gpu', choices=['gpu'])
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--disable-ckpt', action='store_false')
    parser.add_argument('--use-4bit', help="Load model in 4bit", action="store_true")
    parser.add_argument(
        "--data_path",
        type=str,
        default="quintend/rdr-items",
        help="Path to the dataset. Can be a local path or a HF dataset name",
    )
    parser.add_argument("--peft", type=str, default="none", choices=["lora", "none"], help="Which peft to use")
    parser.add_argument("--freeze-vision-model", action="store_true", help="Freeze the vision model parameters")
    parser.add_argument("--freeze-language-model", action="store_true", help="Freeze the language model parameters")
    args = parser.parse_args()

    dataset_fn = None
    # Each new dataset might require a new data preperation function. Current we support quintend/rdr-items and
    # naver-clova-ix/cord-v2. Addtional datasets can be added here.
    if "rdr-items" in args.data_path:
        dataset_fn = mk_hf_vlm_dataset_rdr
    elif "cord-v2" in args.data_path:
        dataset_fn = mk_hf_vlm_dataset_cord_v2
    elif "fineweb-edu" in args.data_path:
        dataset_fn = mk_hf_vlm_dataset_fineweb_edu
    else:
        raise NotImplementedError

    processor = vlm.HFAutoModelForImageTextToText.configure_processor(args.model)

    with FirstRankPerNode():
        dataset = dataset_fn(args.data_path, processor, args.mbs, args.gbs)

    model_kwargs = {}

    # Gemma-3 recommends eager attention implementation
    if "google/gemma-3" in args.model:
        model_kwargs["attn_implementation"] = "eager"

    # Currently freezing language model is not supported as it gives error related to no grad
    # TODO: Fix this
    if args.freeze_language_model:
        raise ValueError("Freezing language model is not supported for current version of VLM automodel")

    model = vlm.HFAutoModelForImageTextToText(
        args.model,
        load_in_4bit=args.use_4bit,
        processor=processor,
        freeze_language_model=args.freeze_language_model,
        freeze_vision_model=args.freeze_vision_model,
        **model_kwargs,
    )

    peft = None
    if args.peft == 'lora':
        peft = llm.peft.LoRA(
            target_modules=['*_proj'],
            dim=16,
            lora_dtype=torch.bfloat16 if args.use_4bit else None,
        )
    nemo_logger = nl.NeMoLogger(
        log_dir=args.log_dir,
        name=args.name,
        wandb=WandbLogger(project=args.wandb_project, name=args.name) if args.wandb_project is not None else None,
    )

    llm.finetune(
        model=model,
        data=dataset,
        trainer=nl.Trainer(
            devices=args.devices,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
            strategy=make_strategy(args.strategy, model, args.devices, args.num_nodes, adapter_only=False),
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=max(1, args.gbs // args.mbs),
            gradient_clip_val=1,
            use_distributed_sampler=False,
            enable_checkpointing=args.disable_ckpt,
            precision='bf16-mixed',
            num_nodes=args.num_nodes,
        ),
        # llm.adam.pytorch_adam_with_flat_lr is returns a fiddle
        #  config, so we need to build the object from the config.
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=nemo_logger,
        peft=peft,
    )
