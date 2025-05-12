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
from typing import Callable, Optional

import lightning.pytorch as pl
import nemo_run as run

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
from nemo.collections.llm.gpt.model.deepseek import DeepSeekModel, DeepSeekV3Config
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.llm.recipes.deepseek import trainer
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.utils.exp_manager import TimingCallback

NAME = "deepseek_v3"


@run.cli.factory(name=NAME)
def model(use_mtp=False) -> run.Config[pl.LightningModule]:
    """
    Factory function to create a DeepSeek-V3 (671B) model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the DeepSeek V3 model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=deepseek_v3 ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    if use_mtp:
        conf = run.Config(DeepSeekV3Config, mtp_num_layers=1, mtp_loss_scaling_factor=0.1)
    else:
        conf = run.Config(DeepSeekV3Config)
    return run.Config(DeepSeekModel, config=conf)


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 128,
    num_gpus_per_node: int = 8,
    fn: Callable = pretrain,
    use_mtp: bool = True,
) -> run.Partial:
    """
    Create a pre-training recipe for DeepSeek-V3 (671B) model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        performance_mode (bool): If true, enables optimizations for maximum performance.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for pre-training.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory deepseek_v3
            $ nemo llm pretrain --factory "deepseek_v3(num_nodes=128, name='my_deepseek_v3')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="deepseek_v3_pretrain", num_nodes=128)
            >>> print(recipe)

    """
    recipe = run.Partial(
        fn,
        model=model(use_mtp),
        trainer=trainer(
            tensor_parallelism=1,
            pipeline_parallelism=16,
            expert_parallelism=64,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(MockDataModule, seq_length=4096, global_batch_size=4096, micro_batch_size=1),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )
    recipe.trainer.strategy.num_layers_in_first_pipeline_stage = 3
    recipe.trainer.strategy.num_layers_in_last_pipeline_stage = 2
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None
    recipe.trainer.strategy.expert_tensor_parallel_size = 1
    recipe.trainer.strategy.tensor_model_parallel_size = 2
    recipe.trainer.strategy.ddp.grad_reduce_in_fp32 = False

    recipe.log.ckpt.save_top_k = 2
    from datetime import timedelta

    recipe.log.ckpt.train_time_interval = run.Config(timedelta, minutes=60)

    # recompute
    recipe.model.config.recompute_granularity = "selective"
    recipe.model.config.recompute_modules = ["mla_up_proj", "layernorm"]

    # DeepEP
    recipe.model.config.moe_token_dispatcher_type = "flex"
    recipe.model.config.moe_enable_deepep = True
    recipe.model.config.moe_shared_expert_overlap = False

    garbage_collection_callback = run.Config(
        GarbageCollectionCallback,
        gc_interval_train=5,
        gc_interval_val=5,
    )
    comm_overlap_callback = run.Config(
        MegatronCommOverlapCallback,
        tp_comm_overlap=False,
    )

    recipe.trainer.callbacks.append(garbage_collection_callback)
    recipe.trainer.callbacks.append(comm_overlap_callback)

    return recipe


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    resume_path: str = "deepseek-ai/DeepSeek-V3-Base",
    name: str = "default",
    num_nodes: int = 5,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'lora',
    seq_length: Optional[int] = None,
    packed_sequence: Optional[bool] = None,
) -> run.Partial:
    """
    Create a fine-tuning recipe for DeepSeek-V3 (671B) model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        resume_path (str): Path to the NeMo checkpoint
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning.
            Allowed values: 'lora'/'dora'/'none'/None.
        seq_length (int): Maximum number of tokens per microbatch.
        packed_sequence (Optional[bool]): If true, fine-tuning sequences will be packed into batches up to the given
            maximum seq_length for better efficiency. By default, this value equals performance_mode.
    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory deepseek_v3
            $ nemo llm finetune --factory "deepseek_v3(num_nodes=5, name='my_deepseek_v3_finetune')"

        Python API usage:
            >>> recipe = finetune_recipe(name="deepseek_v3_finetune", num_nodes=6)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning. Be aware that fine-tuning the DeepSeek-V3 model
        requires substantial computational resources.
    """

    if seq_length is None:
        seq_length = 2048

    if num_nodes is None:
        if peft_scheme is None or peft_scheme.lower() == 'none':
            num_nodes = 64
        elif peft_scheme.lower() in ['lora', 'dora']:
            num_nodes = 5

    recipe = default_finetune_recipe(model(), resume_path, dir, name, num_nodes, num_gpus_per_node, packed_sequence)
    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.expert_model_parallel_size = 64
        recipe.trainer.strategy.tensor_model_parallel_size = 1
        recipe.trainer.strategy.pipeline_model_parallel_size = 8
        recipe.trainer.strategy.num_layers_in_first_pipeline_stage = 6
        recipe.trainer.strategy.num_layers_in_last_pipeline_stage = 7
        recipe.optim.config.lr = 5e-6
    elif peft_scheme.lower() in ['lora', 'dora']:
        recipe.peft = run.Config(PEFT_STR2CLS[peft_scheme.lower()])
        recipe.peft.target_modules = [
            'linear_q_down_proj',
            'linear_q_up_proj',
            'linear_kv_down_proj',
            'linear_kv_up_proj',
            'linear_proj',
        ]
        recipe.optim.config.use_distributed_optimizer = False
        recipe.model.config.cross_entropy_loss_fusion = False
        recipe.trainer.strategy.sequence_parallel = True
        recipe.trainer.strategy.tensor_model_parallel_size = 8
        recipe.trainer.strategy.expert_model_parallel_size = 1
        recipe.trainer.strategy.pipeline_model_parallel_size = 5
        recipe.trainer.strategy.num_layers_in_first_pipeline_stage = 13
        recipe.trainer.strategy.num_layers_in_last_pipeline_stage = 12
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")

    # Sequence length settings in the model and dataset must agree
    recipe.model.config.seq_length = seq_length
    recipe.data.seq_length = seq_length
    if packed_sequence:
        recipe.data.dataset_kwargs = {'pad_to_max_length': True}
        recipe.data.packed_sequence_specs = run.Config(PackedSequenceSpecs, packed_sequence_size=seq_length)

    return recipe
