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

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Union

import lightning.pytorch as pl
import nemo_run as run

from typing_extensions import Annotated

import nemo.lightning as nl
from nemo.lightning import (
    AutoResume,
    NeMoLogger,
    OptimizerModule,
    Trainer,
    configure_no_restart_validation_training_loop,
)
from nemo.lightning.pytorch.callbacks import PEFT, ModelTransform
from nemo.utils import logging

TokenizerType = Any


@run.cli.entrypoint(namespace="speechlm")
def train(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    tokenizer: Optional[TokenizerType] = None,
    model_transform: Optional[Union[PEFT, ModelTransform, Callable]] = None,
    # TODO: Fix export export: Optional[str] = None,
) -> Path:
    """
    Trains a model using the specified data and trainer, with optional tokenizer, source, and export.

    Args:
        model (pl.LightningModule): The model to be trained.
        data (pl.LightningDataModule): The data module containing training data.
        trainer (Trainer): The trainer instance configured with a MegatronStrategy.
        log (NeMoLogger): A nemologger instance.
        resume (Optional[Union[AutoResume, Resume]]): Resume training from a checkpoint.
        optim (Optional[OptimizerModule]): The optimizer module to be used. If not provided, the default optimizer
            from the model will be used.
        tokenizer (Optional[TokenizerType]): Tokenizer setting to be applied. Can be 'data' or 'model'
            or an instance of TokenizerSpec.
        export (Optional[str]): Filename to save the exported checkpoint after training.
        model_transform (Optional[Union[Callable[[nn.Module], nn.Module], PEFT]]): A model transform to be applied.

    Returns
    -------
        Path: The directory path where training artifacts are saved.

    Examples
    --------
        >>> from nemo.collections import llm
        >>> from nemo import lightning as nl
        >>> model = llm.MistralModel()
        >>> data = llm.SquadDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)
        >>> precision = nl.MegatronMixedPrecision(precision="bf16-mixed")
        >>> trainer = nl.Trainer(strategy=nl.MegatronStrategy(tensor_model_parallel_size=2), plugins=precision)
        >>> llm.train(model, data, trainer, tokenizer="data")
        PosixPath('/path/to/log_dir')
    """
    app_state = _setup(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer=tokenizer,
        model_transform=model_transform,
    )

    trainer.fit(model, data)

    return app_state.exp_dir


@run.cli.entrypoint(namespace="speechlm")
def pretrain(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
) -> Path:
    """
    Pretrains a model using the specified data and trainer, with optional logging, resuming, and optimization.

    This function is a wrapper around the `train` function, specifically configured for pretraining tasks.
    Note, by default it will use the tokenizer from the model.

    Args:
        model (pl.LightningModule): The model to be pretrained.
        data (pl.LightningDataModule): The data module containing pretraining data.
        trainer (Trainer): The trainer instance configured with a MegatronStrategy.
        log (NeMoLogger): A nemologger instance.
        resume (Optional[AutoResume]): Resume training from a checkpoint.
        optim (Optional[OptimizerModule]): The optimizer module to be used. If not provided, the default
            optimizer from the model will be used.

    Returns:
        Path: The directory path where pretraining artifacts are saved.

    Examples:
        >>> from nemo.collections import llm
        >>> from nemo import lightning as nl
        >>> model = llm.MistralModel()
        >>> data = llm.PretrainingDataModule(paths=[...], seq_length=4096, global_batch_size=16, micro_batch_size=2)
        >>> precision = nl.MegatronMixedPrecision(precision="bf16-mixed")
        >>> trainer = nl.Trainer(strategy=nl.MegatronStrategy(tensor_model_parallel_size=2), plugins=precision)
        >>> llm.pretrain(model, data, trainer)
        PosixPath('/path/to/log_dir')
    """
    _validate_config(model, data, trainer, log=log, resume=resume, optim=optim)
    return train(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer="data",
    )


@run.cli.entrypoint(namespace="speechlm")
def finetune(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    peft: Optional[Union[PEFT, ModelTransform, Callable]] = None,
) -> Path:
    """
    Finetunes a model using the specified data and trainer, with optional logging, resuming, and PEFT.

    Note, by default it will use the tokenizer from the model.

    Args:
        model (pl.LightningModule): The model to be finetuned.
        data (pl.LightningDataModule): The data module containing finetuning data.
        trainer (Trainer): The trainer instance configured with a MegatronStrategy.
        log (NeMoLogger): A nemologger instance.
        resume (Optional[AutoResume]): Resume training from a checkpoint.
        optim (Optional[OptimizerModule]): The optimizer module to be used. If not provided, the default
            optimizer from the model will be used.
        peft (Optional[PEFT]): A PEFT (Parameter-Efficient Fine-Tuning) configuration to be applied.

    Returns:
        Path: The directory path where finetuning artifacts are saved.

    Examples:
        >>> from nemo.collections import llm
        >>> from nemo import lightning as nl
        >>> model = llm.MistralModel()
        >>> data = llm.SquadDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)
        >>> precision = nl.MegatronMixedPrecision(precision="bf16-mixed")
        >>> trainer = nl.Trainer(strategy=nl.MegatronStrategy(tensor_model_parallel_size=2), plugins=precision)
        >>> llm.finetune(model, data, trainer, peft=llm.peft.LoRA()])
        PosixPath('/path/to/log_dir')
    """

    _validate_config(model, data, trainer, log=log, resume=resume, optim=optim, model_transform=peft)
    return train(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer="model",
        model_transform=peft,
    )


@run.cli.entrypoint(namespace="speechlm")
def validate(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    tokenizer: Optional[TokenizerType] = None,
    model_transform: Optional[Union[PEFT, ModelTransform, Callable]] = None,
) -> Path:
    """
    Validates a model using the specified data and trainer, with optional logging, resuming, and model transformations.

    Args:
        model (pl.LightningModule): The model to be validated.
        data (pl.LightningDataModule): The data module containing validation data.
        trainer (Trainer): The trainer instance configured with a MegatronStrategy.
        log (NeMoLogger): A nemologger instance.
        resume (Optional[AutoResume]): Resume from a checkpoint for validation.
        optim (Optional[OptimizerModule]): The optimizer module to be used. If not provided, the default optimizer
            from the model will be used.
        tokenizer (Optional[TokenizerType]): Tokenizer setting to be applied. Can be 'data' or 'model'
            or an instance of TokenizerSpec.
        model_transform (Optional[Union[Callable[[nn.Module], nn.Module], PEFT]]): A model transform to be applied.

    Returns:
        Path: The directory path where validation artifacts are saved.

    Examples:
        >>> from nemo.collections import llm
        >>> from nemo import lightning as nl
        >>> model = llm.MistralModel()
        >>> data = llm.SquadDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)
        >>> precision = nl.MegatronMixedPrecision(precision="bf16-mixed")
        >>> trainer = nl.Trainer(strategy=nl.MegatronStrategy(tensor_model_parallel_size=2), plugins=precision)
        >>> llm.validate(model, data, trainer, tokenizer="data")
        PosixPath('/path/to/log_dir')
    """
    app_state = _setup(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer=tokenizer,
        model_transform=model_transform,
    )

    trainer.validate(model, data)

    return app_state.exp_dir


def evaluate():
    """
    Evaluates NeMo SpeechLM model.
    """
    raise NotImplementedError("This function will be implemented later")


@run.cli.entrypoint(name="generate", namespace="speechlm")
def generate():
    """
    Generates text using a NeMo Speech model.
    """
    raise NotImplementedError("This function will be implemented later")


def _use_tokenizer(model: pl.LightningModule, data: pl.LightningDataModule, tokenizer: TokenizerType) -> None:
    if tokenizer == "data":
        _set_with_io(model, "tokenizer", data.tokenizer)
    elif tokenizer == "model":
        _set_with_io(data, "tokenizer", model.tokenizer)
    else:
        try:
            from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

            if isinstance(tokenizer, TokenizerSpec):
                _set_with_io(model, "tokenizer", tokenizer)
                _set_with_io(data, "tokenizer", tokenizer)
            else:
                raise ValueError(f"Expected TokenizerSpec or 'data' or 'model', got: {tokenizer}")
        except ImportError:
            raise ValueError("TokenizerSpec is not available")


def _setup(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Optional[NeMoLogger],
    resume: Optional[AutoResume],
    optim: Optional[OptimizerModule],
    tokenizer: Optional[TokenizerType],
    model_transform: Optional[Union[PEFT, ModelTransform, Callable]],
) -> Any:  # Return type is Any because app_state's type is not specified
    configure_no_restart_validation_training_loop(trainer)
    _log = log or NeMoLogger()
    if resume and isinstance(model_transform, PEFT) and _log.ckpt:
        logging.info("Disabling try_restore_best_ckpt restoration for adapters")
        _log.ckpt.try_restore_best_ckpt = False

    app_state = _log.setup(
        trainer,
        resume_if_exists=getattr(resume, "resume_if_exists", False),
        task_config=getattr(train, "__io__", None),
    )
    if resume is not None:
        resume.setup(trainer, model)

    if optim:
        optim.connect(model)
    if tokenizer:  # TODO: Improve this
        _use_tokenizer(model, data, tokenizer)

    if model_transform:
        _set_with_io(model, "model_transform", model_transform)

    # Add ModelTransform callback to Trainer if needed
    if getattr(model, "model_transform", None):
        if not any(isinstance(cb, ModelTransform) for cb in trainer.callbacks):
            if isinstance(model_transform, ModelTransform):
                trainer.callbacks.append(model_transform)
            else:
                trainer.callbacks.append(ModelTransform())

    return app_state


def _set_with_io(obj, attr, value):
    setattr(obj, attr, value)
    if hasattr(obj, "__io__") and hasattr(value, "__io__"):
        setattr(obj.__io__, attr, deepcopy(value.__io__))


def _validate_config(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Optional[NeMoLogger] = None,
    resume: Optional[AutoResume] = None,
    optim: Optional[OptimizerModule] = None,
    tokenizer: Optional[TokenizerType] = None,
    model_transform: Optional[Union[PEFT, ModelTransform, Callable]] = None,
) -> None:

    ## Model validation
    if hasattr(model, "config"):
        assert getattr(model.config, "seq_length", 1) > 0
        assert getattr(model.config, "max_position_embeddings", 1) > 0
        assert model.config.num_layers > 0
        assert model.config.hidden_size > 0
        assert model.config.num_attention_heads > 0
        assert model.config.ffn_hidden_size > 0

        if hasattr(model.config, "seq_length"):
            if getattr(model.config, "max_position_embeddings", None) is not None:
                assert model.config.seq_length <= model.config.max_position_embeddings
    else:
        assert not isinstance(trainer.strategy, nl.MegatronStrategy), "Expected model.config to exist"

    ## Data validation
    if hasattr(data, 'micro_batch_size'):
        assert data.micro_batch_size > 0
    if hasattr(data, 'global_batch_size'):
        assert data.global_batch_size > 0
    if hasattr(data, 'seq_length'):
        assert data.seq_length > 0

    if hasattr(data, 'micro_batch_size') and hasattr(data, 'global_batch_size'):
        assert (
            data.global_batch_size % data.micro_batch_size == 0
        ), "Global batch size must be divisible by micro batch size in data module."

    ## Trainer validation

    # MegatronStrategy validation
    if isinstance(trainer.strategy, nl.MegatronStrategy):
        # Basic validation
        assert trainer.strategy.tensor_model_parallel_size > 0
        assert trainer.strategy.pipeline_model_parallel_size > 0
        assert trainer.strategy.context_parallel_size > 0

        # DP validation
        assert (trainer.num_devices * trainer.num_nodes) % (
            trainer.strategy.tensor_model_parallel_size
            * trainer.strategy.pipeline_model_parallel_size
            * trainer.strategy.context_parallel_size
        ) == 0, "Number of GPUs must be divisible by the product of all parallelism sizes for data parallel."

        assert (
            data.global_batch_size
            % (
                data.micro_batch_size
                * (
                    (trainer.num_devices * trainer.num_nodes)
                    / (
                        trainer.strategy.tensor_model_parallel_size
                        * trainer.strategy.pipeline_model_parallel_size
                        * trainer.strategy.context_parallel_size
                    )
                )
            )
            == 0
        ), "Global batch size must be divisible by the product of micro batch size and data parallel size"

        # TP/SP validation
        if trainer.strategy.tensor_model_parallel_size == 1:
            if trainer.strategy.sequence_parallel == True:
                warnings.warn("Disabling sequence parallelism because tensor model parallelism is disabled")
                trainer.strategy.sequence_parallel = False

        # PP/VP validation
        if trainer.strategy.pipeline_model_parallel_size > 1:
            assert (
                trainer.strategy.pipeline_dtype is not None
            ), "pipeline_dtype must be set if pipeline model parallelism is enabled"
        else:
            if trainer.strategy.virtual_pipeline_model_parallel_size is not None:
                warnings.warn("Disabling virtual pipeline parallelism because pipeline model parallelism is disabled")
                trainer.strategy.virtual_pipeline_model_parallel_size = None
            if trainer.strategy.pipeline_dtype is not None:
                warnings.warn("Setting pipeline dtype to None because pipeline model parallelism is disabled")
                trainer.strategy.pipeline_dtype = None

        # CP validation
        if trainer.strategy.context_parallel_size > 1:
            if hasattr(model, "config"):
                if model.config.seq_length is not None:
                    assert (
                        model.config.seq_length % (trainer.strategy.context_parallel_size * 2) == 0
                    ), 'Sequence length must be divisible by 2 * context parallel size if context parallel is used.'

        # EP validation
        if trainer.strategy.expert_model_parallel_size > 1:
            if hasattr(model, "config"):
                assert (
                    model.config.num_moe_experts is not None
                ), "num_experts must be non None to use expert model parallelism"
                assert (
                    model.config.num_moe_experts % trainer.strategy.expert_model_parallel_size == 0
                ), "Number of experts should be a multiple of expert model parallel_size."
