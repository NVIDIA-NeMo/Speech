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
from lightning.pytorch.callbacks.callback import Callback

from nemo import lightning as nl
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.hf_auto_model_for_causal_lm import HFAutoModelForCausalLM
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import pytorch_adam_with_cosine_annealing
from nemo.utils.exp_manager import TimingCallback

NAME = "hf_auto_model_for_causal_lm"


@run.cli.factory(name=NAME)
def model(model_name, load_pretrained_weights) -> run.Config[pl.LightningModule]:
    """
    Factory function to create HFAutoModelForCausalLM model configurations.

    Args:
        model_name (str): Model id on HF.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the HFAutoModelForCausalLM.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory 'HFAutoModelForCausalLM(model_name="mistralai/Mistral-Nemo-Instruct-2407")'

        Python API usage:
            >>> model_config = model(model_name="mistralai/Mistral-Nemo-Instruct-2407")
            >>> print(model_config)
    """
    return run.Config(HFAutoModelForCausalLM, model_name=model_name, load_pretrained_weights=load_pretrained_weights)


def trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 2,
    sequence_parallelism: bool = False,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 100,
    callbacks: Optional[list[run.Config[Callback]]] = None,
    strategy: Optional[str] = 'ddp',
    gradient_clip_val: float = 1.0,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for HFAutoModelForCausalLM.

    This function sets up the distributed training strategy and other training parameters.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_type (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps.
        callbacks (Optional[list[run.Config[Callback]]]): List of callback configurations.
        strategy: Optional[str] = 'ddp': Parallelism strategy.
        gradient_clip_val: float = 1.0: gradient-clip value.
    Returns:
        run.Config[nl.Trainer]: Configuration for the NeMo Lightning Trainer.

    Examples:
        CLI usage:
            $ nemo llm pretrain trainer=HFAutoModelForCausalLM ...

        Python API usage:
            >>> trainer_config = trainer(num_nodes=2, num_gpus_per_node=8)
            >>> print(trainer_config)
    """
    strategy = str(strategy).lower()
    assert strategy in ['', 'ddp', 'fsdp'], strategy
    if strategy == 'fsdp':
        # See: https://github.com/Lightning-AI/pytorch-lightning/blob/8ad3e29816a63d8ce5c00ac104b14729a4176f4f/src/lightning/pytorch/plugins/precision/fsdp.py#L81
        gradient_clip_val = None

    trainer = run.Config(
        nl.Trainer,
        devices=num_gpus_per_node,
        max_steps=max_steps,
        accelerator='gpu',
        strategy=strategy,
        log_every_n_steps=1,
        limit_val_batches=0.0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=10,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        use_distributed_sampler=False,
    )

    return trainer


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    fn=pretrain,
    model_name: str = '',
) -> run.Partial:
    """
    Create a pre-training recipe for a HFAutoModelForCausalLM model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for pre-training.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory 'HFAutoModelForCausalLM(model_name="mistralai/Mistral-Nemo-Instruct-2407")'

        Python API usage:
            >>> recipe = pretrain_recipe(name="auto_pretrain", num_nodes=2, model_name="mistralai/Mistral-Nemo-Instruct-2407")
            >>> print(recipe)
    """
    return run.Partial(
        fn,
        model=model(model_name, load_pretrained_weights=False),
        trainer=trainer(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(MockDataModule, seq_length=4096, global_batch_size=512, micro_batch_size=1),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=pytorch_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'lora',
    model_name: str = '',
) -> run.Partial:
    """
    Create a fine-tuning recipe for a HFAutoModelForCausalLM model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning. Allowed values: 'lora', 'none'/None.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory hf_auto_model_for_causal_lm

        Python API usage:
            >>> recipe = finetune_recipe(name="llama3_8b_finetune", num_nodes=2)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning. For more information
        on fine-tuning LLMs with NeMo, see the fine-tuning guide in the
        `examples/llm/finetune/` directory.
    """
    recipe = run.Partial(
        finetune,
        model=model(model_name, load_pretrained_weights=True),
        trainer=trainer(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(MockDataModule, seq_length=4096, global_batch_size=512, micro_batch_size=1),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=pytorch_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )
    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.optim.config.lr = 5e-6
    elif peft_scheme.lower() == 'lora':
        recipe.peft = run.Config(LoRA, target_modules=['*_proj'])
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")
    return recipe
