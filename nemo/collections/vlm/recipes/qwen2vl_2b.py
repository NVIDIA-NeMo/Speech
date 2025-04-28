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


from typing import Optional

import lightning.pytorch as pl
import nemo_run as run
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from transformers import Qwen2VLImageProcessor

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.recipes.log.default import tensorboard_logger
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.collections.vlm.qwen2vl.data.mock import Qwen2VLMockDataModule
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback

NAME = "qwen2vl_2b"

HF_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Qwen2VL 2B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Qwen2VL 2B model model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=qwen2vl_2b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(vlm.Qwen2VLModel, config=run.Config(vlm.Qwen2VLConfig2B))


@run.cli.factory(target=llm.finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'none',
) -> run.Partial:
    """
    Create a fine-tuning recipe for Qwen2VL 2B model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory qwen2vl_2b

        Python API usage:
            >>> recipe = finetune_recipe(name="qwen2vl_2b_finetune", num_nodes=1)
            >>> print(recipe)

    Note:
        This recipe uses the Mock dataset for fine-tuning.
    """

    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        encoder_pipeline_model_parallel_size=0,
        sequence_parallel=True,
        pipeline_dtype=torch.bfloat16,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        devices=num_gpus_per_node,
        limit_val_batches=10,
        log_every_n_steps=1,
        max_steps=10,
        num_nodes=num_nodes,
        plugins=bf16_mixed(),
        strategy=strategy,
        val_check_interval=1000,
        callbacks=[
            run.Config(TimingCallback),
            run.Config(MegatronCommOverlapCallback, tp_comm_overlap=True),
        ],
    )
    tokenizer = run.Config(AutoTokenizer, HF_MODEL_NAME)
    image_processor = run.Config(Qwen2VLImageProcessor)

    max_sequence_length = 4096

    language_transformer_config = run.Config(llm.Qwen2Config1P5B, seq_length=max_sequence_length)

    vision_transformer_config = run.Config(vlm.Qwen2VLVisionConfig)

    vision_projection_config = run.Config(
        vlm.MultimodalProjectorConfig,
        projector_type="mcore_mlp",
        input_size=vision_transformer_config.ffn_hidden_size,
        hidden_size=language_transformer_config.hidden_size,
        ffn_hidden_size=vision_transformer_config.ffn_hidden_size,
    )

    # Qwen2VL model configuration
    qwen2vl_config = run.Config(
        vlm.Qwen2VLConfig,
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        freeze_language_model=False,
        freeze_vision_model=True,
    )

    model = run.Config(vlm.Qwen2VLModel, qwen2vl_config, tokenizer=tokenizer)
    nemo_resume = run.Config(
        nl.AutoResume,
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
    )

    opt_config = run.Config(
        OptimizerConfig,
        optimizer='adam',
        lr=2.0e-06,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=True,
        bf16=True,
    )
    sched = run.Config(
        CosineAnnealingScheduler, max_steps=trainer.max_steps, warmup_steps=0, constant_steps=1000, min_lr=1.0e-07
    )
    opt = run.Config(MegatronOptimizerModule, opt_config, sched)

    recipe = run.Partial(
        llm.finetune,
        model=model,
        trainer=trainer,
        data=run.Config(
            Qwen2VLMockDataModule,
            seq_length=max_sequence_length,
            global_batch_size=128,
            micro_batch_size=2,
            tokenizer=tokenizer,
            image_processor=image_processor,
            num_workers=4,
        ),
        log=llm.default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=opt,
        resume=nemo_resume,
    )

    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.tensor_model_parallel_size = 1
        recipe.optim.config.lr = 2e-05
    elif peft_scheme.lower() == 'lora':
        recipe.peft = run.Config(
            vlm.LoRA,
            target_modules=[
                "linear_qkv",
                "linear_proj",
                "linear_fc1",
                "linear_fc2",
            ],
        )
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")

    return recipe


if __name__ == "__main__":
    env_vars = {
        "CUDA_VISIBLE_DEVICES": "0,1",
    }
    recipe = finetune_recipe(num_gpus_per_node=2)
    recipe.trainer.max_steps = 10
    recipe.trainer.val_check_interval = 10
    recipe.trainer.limit_val_batches = 0.0
    recipe.data.global_batch_size = 8
    recipe.trainer.strategy.tensor_model_parallel_size = 1

    executor = run.LocalExecutor(ntasks_per_node=2, launcher="torchrun", env_vars=env_vars)

    run.run(recipe, executor=executor, name="qwen2vl_2b_finetune")
