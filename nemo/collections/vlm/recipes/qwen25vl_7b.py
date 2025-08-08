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

from typing import Optional

import lightning.pytorch as pl
import nemo_run as run
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.recipes.log.default import tensorboard_logger
from nemo.collections.vlm.qwen2vl.model import Qwen25VLVisionConfig
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback

NAME = "qwen25vl_7b"

HF_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Qwen2.5VL 7B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Qwen2.5VL 7B model.

    Examples:
        CLI usage:
            $ nemo llm finetune model=qwen25vl_7b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """

    return run.Config(vlm.Qwen2VLModel, config=run.Config(vlm.Qwen2VLConfig))


@run.cli.factory(target=llm.finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'none',
) -> run.Partial:
    """
    Create a finetuning recipe for Qwen2.5VL 7B model.

    This function sets up a complete configuration for finetuning, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the finetuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.

    Returns:
        run.Partial: Partial configuration for finetuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory qwen25vl_7b

        Python API usage:
            >>> recipe = finetune_recipe(name="qwen25vl_7b_finetune", num_nodes=1)
            >>> print(recipe)

    Note:
        This recipe uses the Mock dataset for finetuning.
    """

    # Training strategy setup
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=True,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
        use_te_rng_tracker=True,
    )

    # tp_comm_overlap callback
    tp_comm_overlap_callback = run.Config(
        MegatronCommOverlapCallback, tp_comm_overlap=True, tp_comm_bootstrap_backend='nccl'
    )
    # Trainer setup
    trainer = run.Config(
        nl.Trainer,
        num_nodes=num_nodes,
        devices=num_gpus_per_node,
        max_steps=10,
        accelerator="gpu",
        strategy=strategy,
        plugins=run.Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
        callbacks=[run.Config(TimingCallback), tp_comm_overlap_callback],
        val_check_interval=10,
        limit_val_batches=0.0,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    max_sequence_length = 8192

    data = run.Config(
        vlm.Qwen2VLMockDataModule,
        seq_length=max_sequence_length,
        global_batch_size=64,
        micro_batch_size=1,
        tokenizer=run.Config(AutoTokenizer, HF_MODEL_NAME),
        image_processor=run.Config(Qwen2VLImageProcessor),
        num_workers=1,
    )

    # Submodules configurations
    language_transformer_config = run.Config(llm.Qwen25Config7B, seq_length=max_sequence_length)
    vision_transformer_config = run.Config(Qwen25VLVisionConfig)
    vision_projection_config = run.Config(
        vlm.MultimodalProjectorConfig,
        projector_type="mcore_mlp",
        input_size=vision_transformer_config.hidden_size * (vision_transformer_config.spatial_merge_size**2),
        hidden_size=language_transformer_config.hidden_size,
        ffn_hidden_size=vision_transformer_config.hidden_size * (vision_transformer_config.spatial_merge_size**2),
    )

    # Qwen25VL model configuration
    qwen25vl_config = run.Config(
        vlm.Qwen2VLConfig,
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        freeze_language_model=False,
        freeze_vision_model=True,
        enable_cuda_graph=True,
        use_te_rng_tracker=True,
        gradient_accumulation_fusion=True,
        cross_entropy_loss_fusion=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        masked_softmax_fusion=True,
        attention_softmax_in_fp32=True,
        apply_rope_fusion=True,
        tp_comm_overlap=True,
        tp_comm_overlap_rs_dgrad=True,
        overlap_p2p_comm=True,
    )
    model = run.Config(
        vlm.Qwen2VLModel,
        config=qwen25vl_config,
        model_version="qwen25-vl",
        tokenizer=run.Config(AutoTokenizer, HF_MODEL_NAME),
    )

    nemo_resume = run.Config(
        nl.AutoResume,
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
    )
    # Optimizer and scheduler setup
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
        CosineAnnealingScheduler,
        max_steps=10,
        warmup_steps=0,
        constant_steps=1000,
        min_lr=1.0e-07,
    )
    opt = run.Config(MegatronOptimizerModule, opt_config, sched)

    recipe = run.Partial(
        llm.finetune,
        model=model,
        trainer=trainer,
        data=data,
        log=llm.default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=opt,
        resume=nemo_resume,
    )
    return recipe


if __name__ == "__main__":
    env_vars = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    }
    recipe = finetune_recipe(num_gpus_per_node=8)

    executor = run.LocalExecutor(ntasks_per_node=8, launcher="torchrun", env_vars=env_vars)

    run.run(recipe, executor=executor, name="qwen2.5vl_7b_finetune")
