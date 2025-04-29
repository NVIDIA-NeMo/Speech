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
import importlib
import json
import warnings
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import lightning.pytorch as pl
import nemo_run as run
import torch
from megatron.core import parallel_state
from rich.console import Console
from torch.distributed import all_gather_object
from typing_extensions import Annotated

import nemo.lightning as nl
from nemo.collections.llm import GPTModel, HFAutoModelForCausalLM
from nemo.collections.llm.evaluation.api import EvaluationConfig, EvaluationTarget, MisconfigurationError
from nemo.collections.llm.modelopt import (
    DistillationGPTModel,
    ExportConfig,
    PruningConfig,
    QuantizationConfig,
    Quantizer,
    prune_gpt_model,
    save_pruned_model,
    set_modelopt_spec_if_exists_in_ckpt,
    setup_trainer_and_restore_model_with_modelopt_spec,
)
from nemo.lightning import (
    AutoResume,
    NeMoLogger,
    OptimizerModule,
    Trainer,
    configure_no_restart_validation_training_loop,
    io,
)
from nemo.lightning.base import NEMO_MODELS_CACHE
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.pytorch.callbacks import PEFT, JitTransform, ModelTransform
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

if TYPE_CHECKING:
    from megatron.core.inference.common_inference_params import CommonInferenceParams
    from megatron.core.inference.inference_request import InferenceRequest


TokenizerType = Any
AnyPath = Union[Path, str]


@run.cli.entrypoint(namespace="llm")
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
    # [ModelOpt]: If modelopt_state exists, overwrite transformer_layer_spec to modelopt spec
    if resume:
        if resume.restore_config and resume.restore_config.path:
            set_modelopt_spec_if_exists_in_ckpt(model, resume.restore_config.path)
        elif resume.resume_from_path:
            set_modelopt_spec_if_exists_in_ckpt(model, resume.resume_from_path)

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


@run.cli.entrypoint(namespace="llm")
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


@run.cli.entrypoint(namespace="llm")
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


@run.cli.entrypoint(namespace="llm")
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


@run.cli.entrypoint(name="prune", namespace="llm")
def prune(
    nemo_checkpoint: str,
    save_path: str,
    pruning_config: PruningConfig,
    devices: int = 1,
    num_nodes: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    num_train_samples: int = 1024,
    data: pl.LightningDataModule | None = None,
    tokenizer_path: str | None = None,
    legacy_ckpt: bool = False,
) -> str:
    """
    Prunes a model using the specified data and trainer. Currently only supports GPT models.

    Args:
        nemo_checkpoint (str): The path to the NeMo checkpoint to be pruned.
        save_path (str): The path to save the pruned NeMo checkpoint.
        pruning_config (PruningConfig): The pruning configuration.
        devices (int): The number of devices to use for pruning.
        num_nodes (int): The number of nodes to use for pruning.
        tp_size (int): The tensor parallel size.
        pp_size (int): The pipeline parallel size.
        num_train_samples (int): Number of training samples for importance estimation using forward pass.
        data (pl.LightningDataModule): The data module for forward pass.
            Required if not dropping layers.
        tokenizer_path (str): Path to the tokenizer if not using model's tokenizer.
        legacy_ckpt (bool): If True, allow loading ckpt saved with older version of TE.
            Use for cases like missing state dict keys ending with `_extra_state`.

    Returns:
        str: The path to the pruned NeMo checkpoint.

    Examples:
        >>> from nemo.collections import llm
        >>> from nemo.collections.llm.modelopt.prune import PruningConfig
        >>> data = llm.PretrainingDataModule(
                paths=["1.0", "path/to/tokenized/data"],
                seq_length=256,
                global_batch_size=1,
                micro_batch_size=1,
            )
        >>> llm.prune(
                nemo_checkpoint="path/to/llama3.1-8b",
                save_path="path/to/pruned_llama_model",
                pruning_config=PruningConfig(target_ffn_hidden_size=9216, target_hidden_size=3072),
                data=data
            )
    """
    if data is not None:
        assert data.global_batch_size == data.micro_batch_size, "Global batch size must be equal to micro batch size"
        steps = num_train_samples // data.global_batch_size
    else:
        steps = num_train_samples

    model, trainer = setup_trainer_and_restore_model_with_modelopt_spec(
        model_path=nemo_checkpoint,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        devices=devices,
        num_nodes=num_nodes,
        inference_only=True,
        tokenizer_path=tokenizer_path,
        legacy_ckpt=legacy_ckpt,
        strategy_kwargs={"sequence_parallel": False, "replace_progress_bar": False},
        trainer_kwargs={"max_steps": steps, "limit_val_batches": steps, "val_check_interval": steps},
        model_config_overrides={"sequence_parallel": False},
    )
    prune_gpt_model(model, pruning_config, data, trainer)
    save_pruned_model(trainer, save_path)

    console = Console()
    console.print(f"[green]✓ Pruning succeded, pruned checkpoint saved to {save_path}[/green]")

    return save_path


@run.cli.entrypoint(name="distill", namespace="llm")
def distill(
    student_model_path: AnyPath,
    teacher_model_path: AnyPath,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    tokenizer: Optional[TokenizerType] = None,
    model_transform: Optional[Union[PEFT, ModelTransform, Callable]] = None,
) -> Path:
    """
    Distills a teacher model into a student model using special Knowledge-Distillation losses.

    Note that this requires an existing NeMo 2.0 checkpoint of the student model as well, as
    the model class is not known beforehand.
    This script currently supports instances of ``nemo.collections.llm.GPTModel`` for now.

    Args:
        student_model_path (Path): Path to student model NeMo checkpoint to be trained.
        teacher_model_path (Path): Path to teacher model NeMo checkpoint to distill from.
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
        >>> student = "/path/to/student/nemo/ckpt"  # <-- change me
        >>> teacher = "/path/to/teacher/nemo/ckpt"  # <-- change me
        >>> data = llm.SquadDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)
        >>> precision = nl.MegatronMixedPrecision(precision="bf16-mixed")
        >>> trainer = nl.Trainer(strategy=nl.MegatronStrategy(tensor_model_parallel_size=2), plugins=precision)
        >>> llm.distill(student, teacher, data, trainer, tokenizer="model")
        PosixPath('/path/to/log_dir')
    """
    _student_model = io.load_context(ckpt_to_context_subdir(student_model_path), subpath="model")
    _teacher_model = io.load_context(ckpt_to_context_subdir(teacher_model_path), subpath="model")
    assert isinstance(_student_model, GPTModel), "Only models based on `llm.GPTModel` are supported currently."
    assert isinstance(_teacher_model, GPTModel), "Only models based on `llm.GPTModel` are supported currently."

    if tokenizer is None:
        tokenizer = getattr(_student_model, "tokenizer", None) or getattr(_teacher_model, "tokenizer", None)
        assert tokenizer is not None, "Tokenizer neither provided nor found in models."

    model = DistillationGPTModel(
        _student_model.config,
        _teacher_model.config,
        teacher_ckpt_path=teacher_model_path,
    )
    model.__io__ = _student_model.__io__

    return train(
        model=model,
        data=data,
        optim=optim,
        tokenizer=tokenizer,
        trainer=trainer,
        log=log,
        resume=resume,
        model_transform=model_transform,
    )


@run.cli.entrypoint(name="ptq", namespace="llm")
def ptq(
    model_path: str,
    export_config: ExportConfig,
    calibration_tp: int = 1,
    calibration_pp: int = 1,
    num_layers_in_first_pipeline_stage: int | None = None,
    num_layers_in_last_pipeline_stage: int | None = None,
    devices: int | None = None,
    num_nodes: int | None = None,
    quantization_config: Annotated[Optional[QuantizationConfig], run.Config[QuantizationConfig]] = None,
    forward_loop: Callable | None = None,
    tokenizer_path: str | None = None,
    legacy_ckpt: bool = False,
    trust_remote_code: bool = False,
) -> Path:
    """
    Applies Post-Training Quantization (PTQ) for a model using the specified quantization and export configs. It runs
    calibration for a small dataset to collect scaling factors low-precision GEMMs used by desired quantization method.
    By default, this function produces TensorRT-LLM checkpoint ready for deployment using nemo.export and nemo.deploy
    modules or direcly using TensorRT-LLM library.

    The function can be used through the NeMo CLI in the following way:
    ```bash
    # Run calibration using tensor parallel set to 8 and export quantized checkpoint with tensor parallel equal 2
    nemo llm ptq run.executor=torchrun run.executor.ntasks_per_node=8 \
        model_path=/models/Llama-3-70B \
        export_config.path=/models/Llama-3-70B-FP8 \
        calibration_tp=8 \
        export_config.inference_tp=2

    # Choose different quantization method, for example, INT8 SmoothQuant
    nemo llm ptq run.executor=torchrun run.executor.ntasks_per_node=1 \
        model_path=/models/Llama-3-8B \
        export_config.path=/models/Llama-3-8B-INT8_SQ \
        quantization_config.algorithm=int8_sq

    # Export as NeMo checkpoint instead
    nemo llm ptq run.executor=torchrun \
        model_path=/models/Llama-3-8B \
        export_config.path=/models/Llama-3-8B-INT8_SQ \
        quantization_config.algorithm=int8_sq \
        export_config.export_format=nemo

    # Quantize HF AutoModel checkpoint.
    nemo llm ptq run.executor=torchrun run.executor.ntasks_per_node=1 \
        model_path=/models/Llama-3-70B-HF \
        export_config.path=/models/Llama-3-70B-HF-FP8 \
        export_config.export_format=hf
    ```

    Args:
        model_path (str): The path to model to be quantized.
        calibration_tp (int): Calibration tensor parallelism.
        calibration_pp (int): Calibration pipeline parallelism.
        num_layers_in_first_pipeline_stage (int): Number of layers in the first pipeline stage.
        num_layers_in_last_pipeline_stage (int): Number of layers in the last pipeline stage.
        export_config (ExportConfig): Export configuration for output checkpoint.
        devices (int): Number of devices to use for calibration. Default: calibration_tp.
        num_nodes (int): Number of nodes to use for calibration. Default: calibration_pp.
        quantization_config (QuantizationConfig): Configuration for quantization algorithm.
        forward_loop (Callable): Forward loop to use for calibration.
            If not provided, a forward loop will be created using the calibration dataset.
        tokenizer_path (str): Path to the tokenizer if not using model's tokenizer.
        legacy_ckpt (bool): If True, allow loading ckpt saved with older version of TE.
        trust_remote_code (bool): Trust remote code when loading HuggingFace models.

    Returns:
        Path: The path where the quantized checkpoint has been saved after calibration.
    """
    if not quantization_config:
        quantization_config = QuantizationConfig()
    if devices is None:
        devices = calibration_tp
    if num_nodes is None:
        num_nodes = calibration_pp

    quantizer = Quantizer(quantization_config, export_config)
    assert Path(model_path).exists(), f"Path {model_path} does not exist"
    is_automodel = (Path(model_path) / 'config.json').exists()

    trainer = None
    if is_automodel:
        assert export_config.export_format != "nemo", "Automodel PTQ does not support export format nemo"
        model = HFAutoModelForCausalLM(model_name=model_path, trust_remote_code=trust_remote_code, device_map="auto")
        model.configure_model()
    else:
        model, trainer = setup_trainer_and_restore_model_with_modelopt_spec(
            model_path=model_path,
            tensor_model_parallel_size=calibration_tp,
            pipeline_model_parallel_size=calibration_pp,
            num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
            devices=devices,
            num_nodes=num_nodes,
            inference_only=True,
            tokenizer_path=tokenizer_path,
            legacy_ckpt=legacy_ckpt,
            strategy_kwargs={"sequence_parallel": False, "lazy_init": True},
            trainer_kwargs={},
            model_config_overrides={"sequence_parallel": False},
        )

    model = quantizer.quantize(model, forward_loop)
    quantizer.export(model, model_path, trainer)

    if is_global_rank_zero():
        console = Console()
        console.print(f"[green]✓ PTQ succeded, quantized checkpoint exported to {export_config.path}[/green]")
    return export_config.path


@run.cli.entrypoint(namespace="llm")
def deploy(
    nemo_checkpoint: AnyPath = None,
    backend: str = "in-framework",
    model_type: str = "llama",
    triton_model_name: str = "triton_model",
    triton_model_version: Optional[int] = 1,
    triton_http_port: int = 8000,
    triton_grpc_port: int = 8001,
    triton_http_address: str = "0.0.0.0",
    triton_model_repository: AnyPath = None,
    start_fastapi_server: bool = True,
    fastapi_http_address: str = "0.0.0.0",
    fastapi_port: int = 8080,
    num_gpus: int = 1,
    num_nodes: int = 1,
    tensor_parallelism_size: int = 1,
    pipeline_parallelism_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    max_input_len: int = 4096,
    max_output_len: int = 256,
    max_batch_size: int = 8,
    output_context_logits: bool = True,
    output_generation_logits: bool = True,
):
    """
    Deploys nemo model on a PyTriton server either "in-framework" or by converting to trtllm depending on the backend.
    This deploy method is intended to be used for evaluation.

    Args:
        nemo_checkpoint (Path): Path for nemo checkpoint.
        backend (str): options: "in-framework" or "trtllm". Deploys nemo2 checkpoint directly on Pytriton server wo any
        conversion if "in-framework". If "trtllm", exports nemo2 model to trtllm and deploys on PyTriton.
        Default: "in-framework".
        model_type (str): Type of the model. Choices: gpt, llama, falcon, starcoder. Default: llama.
        triton_model_name (str): Name for the model that gets deployed on PyTriton. Please ensure that the same model
        name is passed to the evalute method for the model to be accessible while sending evalution requests.
        Default: 'triton_model'.
        triton_model_version (Optional[int]): Version for the triton model. Default: 1.
        triton_http_port (int): HTTP port for the PyTriton server. Default: 8000.
        triton_grpc_port (int): gRPC Port for the PyTriton server. Default: 8001.
        triton_http_address (str): HTTP address for the PyTriton server. Default:  "0.0.0.0".
        triton_model_repository (Path): Folder for the trt-llm conversion, trt-llm engine gets saved in this specified
        path. If None, saves it in /tmp dir. Default: None.
        start_fastapi_server (bool): Starts FastAPI server which acts as a proxy in between to expose the
        v1/completions and v1/chat/completions OpenAI (OAI) compatible endpoints as PyTriton does not expose a
        standard HTTP/REST API. Only supported for "in-framework" deployment and not with "trtllm" backend.
        Default: True.
        fastapi_http_address (str): HTTP address for FastAPI interface/server.  Default: "0.0.0.0". OAI endpoints via
        FastAPI interface are only supported for "in-framework" backend.
        fastapi_port (int): Port for FastAPI interface/server. Applicable only for "in-framework" backend.
        Default: 8080.
        num_gpus (int): Number of GPUs per node for export to trtllm and deploy. Default: 1.
        tensor_parallelism_size (int): Tensor parallelism size. Default: 1.
        pipeline_parallelism_size (int): Pipeline parallelism size. Default: 1.
        dtype (str): dtype of the TensorRT-LLM model. Default: "bfloat16".
        max_input_len (int): Max input length of the model. Default: 4096.
        max_output_len (int): Max output length of the model. Default: 256.
        max_batch_size (int): Max batch size of the model. Default: 8.
        openai_format_response (bool): Return the response from PyTriton server in OpenAI compatible format. Needs to
        be True while running evaluation. Default: True.
        output_context_logits (bool): If True builds trtllm engine with 'gather_context_logits=True'. Default: True.
        context_logits are used to compute the logProb of the output token in multi-token prediction benchmarks.
        Used only with "trtllm" backend.
        output_generation_logits (bool): If True builds trtllm engine with gather_generation_logits set to True.
        generation_logits are used to compute the logProb of the output token in case of single token prediction
        benchmarks (like MMLU, lambada). Default: True. Used only with "trtllm" backend.
    """
    import os
    import uvicorn

    from nemo.deploy import DeployPyTriton

    if backend == "in-framework":
        assert (
            start_fastapi_server is True
        ), 'in-framework deployment exposes OAI API endpoints v1/completions and \
        v1/chat/completions hence needs fastAPI interface to expose these endpoints to PyTriton. Please set \
        start_fastapi_server to True'
        if triton_http_port == fastapi_port:
            raise ValueError("FastAPI port and Triton server port cannot use the same port. Please change them")
        # Store triton ip, port relevant for FastAPI as env vars to be accessible by fastapi_interface_to_pytriton.py
        os.environ["TRITON_HTTP_ADDRESS"] = triton_http_address
        os.environ["TRITON_PORT"] = str(triton_http_port)

        try:
            from nemo.deploy.nlp.megatronllm_deployable import MegatronLLMDeployableNemo2
        except Exception as e:
            raise ValueError(
                f"Unable to import MegatronLLMDeployable, due to: {type(e).__name__}: {e} cannot run "
                f"evaluation with in-framework deployment"
            )

        triton_deployable = MegatronLLMDeployableNemo2(
            nemo_checkpoint_filepath=nemo_checkpoint,
            num_devices=num_gpus,
            num_nodes=num_nodes,
            tensor_model_parallel_size=tensor_parallelism_size,
            pipeline_model_parallel_size=pipeline_parallelism_size,
            context_parallel_size=context_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            inference_max_seq_length=max_input_len,
        )

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                try:
                    nm = DeployPyTriton(
                        model=triton_deployable,
                        triton_model_name=triton_model_name,
                        triton_model_version=triton_model_version,
                        max_batch_size=max_batch_size,
                        http_port=triton_http_port,
                        grpc_port=triton_grpc_port,
                        address=triton_http_address,
                    )

                    logging.info("Triton deploy function will be called.")
                    nm.deploy()
                    nm.run()
                except Exception as error:
                    logging.error("Error message has occurred during deploy function. Error message: " + str(error))
                    return

                try:
                    if start_fastapi_server:
                        try:
                            logging.info("REST service will be started.")
                            uvicorn.run(
                                'nemo.deploy.service.fastapi_interface_to_pytriton:app',
                                host=fastapi_http_address,
                                port=fastapi_port,
                                reload=True,
                            )
                        except Exception as error:
                            logging.error(
                                "Error message has occurred during REST service start. Error message: " + str(error)
                            )
                    logging.info("Model serving on Triton will be started.")
                    nm.serve()
                except Exception as error:
                    logging.error("Error message has occurred during deploy function. Error message: " + str(error))
                    return

                logging.info("Model serving will be stopped.")
                nm.stop()
            elif torch.distributed.get_rank() > 0:
                triton_deployable.generate_other_ranks()

    elif backend == "trtllm":
        from nemo.collections.llm.deploy.base import get_trtllm_deployable, unset_environment_variables

        unset_environment_variables()  ## Required for export to trtllm on clusters.
        triton_deployable = get_trtllm_deployable(
            nemo_checkpoint,
            model_type,
            triton_model_repository,
            num_gpus,
            tensor_parallelism_size,
            pipeline_parallelism_size,
            max_input_len,
            max_output_len,
            max_batch_size,
            dtype,
            output_context_logits,
            output_generation_logits,
        )
        try:
            nm = DeployPyTriton(
                model=triton_deployable,
                triton_model_name=triton_model_name,
                triton_model_version=triton_model_version,
                max_batch_size=max_batch_size,
                http_port=triton_http_port,
                grpc_port=triton_grpc_port,
                address=triton_http_address,
            )

            logging.info("Triton deploy function will be called.")
            nm.deploy()
            nm.run()
        except Exception as error:
            logging.error("Error message has occurred during deploy function. Error message: " + str(error))
            return

        try:
            logging.info("Model serving on Triton will be started.")
            nm.serve()
        except Exception as error:
            logging.error("Error message has occurred during deploy function. Error message: " + str(error))
            return

        logging.info("Model serving will be stopped.")
        nm.stop()


@run.cli.entrypoint(namespace="llm")
def evaluate(
    target_cfg: EvaluationTarget,
    eval_cfg: EvaluationConfig = EvaluationConfig(type="gsm8k"),
) -> dict:
    """
    Evaluates nemo model deployed on PyTriton server using nvidia-lm-eval

    Args:
        target_cfg (EvaluationTarget): target of the evaluation. Providing model_id and
            url in EvaluationTarget.api_endpoint is required to run evaluations.
        eval_cfg (EvaluationConfig): configuration for evaluations. Default type (task): gsm8k.
    """
    from nemo.collections.llm.evaluation.base import _legacy_evaluate, find_framework, wait_for_fastapi_server

    if target_cfg.api_endpoint.nemo_checkpoint_path is not None:
        _legacy_evaluate(target_cfg=target_cfg, eval_cfg=eval_cfg)
        return

    import yaml

    eval_type_components = eval_cfg.type.split(".")
    if len(eval_type_components) == 2:
        framework_name, task_name = eval_type_components
    elif len(eval_type_components) == 1:
        framework_name, task_name = None, eval_type_components[0]
    else:
        raise MisconfigurationError(
            "eval_type must follow 'framework_name.task_name'. No additional dots are allowed."
        )

    if framework_name is None:
        framework_module_name = find_framework(task_name)
    else:
        framework_module_name = f"core_evals.{framework_name}"
    try:
        evaluate = importlib.import_module(".evaluate", package=framework_module_name)
    except ImportError:
        raise ImportError(
            f"Please ensure that {framework_module_name} is installed in your env "
            f"as it is required to run {eval_cfg.type} evaluation"
        )

    base_url, _ = target_cfg.api_endpoint.url.split('/v1')
    server_ready = wait_for_fastapi_server(base_url=base_url, model_name=target_cfg.api_endpoint.model_id)
    if not server_ready:
        raise RuntimeError("Server not ready for evaluation")

    results = evaluate.evaluate_accuracy(
        target_cfg=target_cfg,
        eval_cfg=eval_cfg,
    )
    results_dict = results.model_dump()

    logging.info("========== RESULTS ==========")
    logging.info(yaml.dump(results_dict))

    return results_dict


@run.cli.entrypoint(name="import", namespace="llm")
def import_ckpt(
    model: pl.LightningModule,
    source: str,
    output_path: Optional[AnyPath] = None,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    """
    Imports a checkpoint into a model using the model's associated importer, typically for
    the purpose of fine-tuning a community model trained in an external framework, such as
    Hugging Face.

    This function can be used both programmatically and through the NeMo CLI:

    CLI Usage:
    ```bash
    # Import Llama 3 8B from HuggingFace (saves to $NEMO_MODELS_CACHE)
    nemo llm import llama3_8b source="hf://meta-llama/Llama-3.1-8B"

    # Import with custom output path
    nemo llm import llama3_8b source="hf://meta-llama/Llama-3.1-8B" output_path="/path/to/save"

    # Force overwrite existing checkpoint
    nemo llm import llama3_8b source="hf://meta-llama/Llama-3.1-8B" overwrite=true
    ```

    Python Usage:
    ```python
    model = Mistral7BModel()
    imported_path = import_ckpt(model, "hf://mistralai/Mistral-7B-v0.1")
    ```

    The importer component of the model reads the checkpoint data from the specified source
    and transforms it into the right format. This is particularly useful for adapting
    models that have been pre-trained in different environments or frameworks to be fine-tuned
    or further developed within the current system.

    For instance, using `import_ckpt(Mistral7BModel(), "hf")` initiates the import process
    by searching for a registered model importer tagged with "hf". In NeMo, `HFMistral7BImporter`
    is registered under this tag via:
    `@io.model_importer(Mistral7BModel, "hf", default_path="mistralai/Mistral-7B-v0.1")`.
    This links `Mistral7BModel` to `HFMistral7BImporter`, designed for HuggingFace checkpoints.

    Args:
        model (pl.LightningModule): The model into which the checkpoint will be imported.
            This model must implement the ConnectorMixin.
        source (str): The source from which the checkpoint will be imported. This can be
            a file path, URL, or any other string identifier that the model's importer
            can recognize.
        output_path (Optional[Path]): The path where the imported checkpoint will be stored.
            If not specified, the checkpoint will be saved to $NEMO_MODELS_CACHE
            (defaults to ~/.cache/nemo/models/ if the environment variable is not set).
        overwrite (bool): If set to True, existing files at the output path will be overwritten.
            This is useful for model updates where retaining old checkpoint files is not required.

    Returns:
        Path: The path where the checkpoint has been saved after import.

    Raises:
        ValueError: If the model does not implement ConnectorMixin, indicating a lack of
            necessary importer functionality.
        FileExistsError: If the output path is provided (that is, when not using models cache)
            and it exists and overwrite is not set to True.
    """
    if output_path:
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output path {output_path} exists. Use overwrite=True to force overwrite.")

    output = io.import_ckpt(model=model, source=source, output_path=output_path, overwrite=overwrite, **kwargs)

    console = Console()
    if output_path:
        console.print(f"[green]✓ Checkpoint imported to {output}[/green]")
    else:
        console.print(f"[green] $NEMO_MODELS_CACHE={NEMO_MODELS_CACHE} [/green]")

    # Display directory structure as a tree
    dir_tree = _build_directory_tree(output, root_name="Imported Checkpoint")
    console.print(dir_tree)

    return output


def load_connector_from_trainer_ckpt(path: AnyPath, target: str) -> io.ModelConnector:
    # pylint: disable=C0116
    if not isinstance(path, Path):
        path = Path(path)
    return io.load_context(path, subpath="model").exporter(target, path)


@run.cli.entrypoint(name="export", namespace="llm")
def export_ckpt(
    path: AnyPath,
    target: str,
    output_path: Optional[AnyPath] = None,
    overwrite: bool = False,
    load_connector: Callable[[Path, str], io.ModelConnector] = load_connector_from_trainer_ckpt,
    **kwargs,
) -> Path:
    """
    Exports a checkpoint from a model using the model's associated exporter, typically for
    the purpose of sharing a model that has been fine-tuned or customized within NeMo.

    This function can be used both programmatically and through the NeMo CLI:

    CLI Usage:
    ```bash
    # Export model to HuggingFace format (saves to {checkpoint_path}/hf/)
    nemo llm export /path/to/model.nemo target="hf"

    # Export with custom output path
    nemo llm export /path/to/model.nemo target="hf" output_path="/path/to/save"

    # Force overwrite existing export
    nemo llm export /path/to/model.nemo target="hf" overwrite=true
    ```

    Python Usage:
    ```python
    nemo_ckpt_path = Path("/path/to/model.nemo")
    export_path = export_ckpt(nemo_ckpt_path, "hf")
    ```

    The exporter component of the model reads the model's state from the specified path and
    exports it into the format specified by the 'target' identifier. This is particularly
    useful for adapting models that have been developed or fine-tuned within NeMo to be
    compatible with other environments or frameworks.

    Args:
        path (Path): The path to the model's checkpoint file from which data will be exported.
        target (str): The identifier for the exporter that defines the format of the export
            (e.g., "hf" for HuggingFace format).
        output_path (Optional[Path]): The path where the exported checkpoint will be saved.
            If not specified, defaults to {checkpoint_path}/{target}/.
        overwrite (bool): If set to True, existing files at the output path will be overwritten.
            This is useful for model updates where retaining old checkpoint files is not required.
        load_connector (Callable[[Path, str], ModelConnector]): A function to load the appropriate
            exporter based on the model and target format. Defaults to `load_connector_from_trainer_ckpt`.

    Returns:
        Path: The path where the checkpoint has been saved after export.

    Raises:
        ValueError: If the model does not implement ConnectorMixin, indicating a lack of
            necessary exporter functionality.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if output_path and not isinstance(output_path, Path):
        output_path = Path(output_path)

    output = io.export_ckpt(path, target, output_path, overwrite, load_connector, **kwargs)

    console = Console()
    console.print(f"[green]✓ Checkpoint exported to {output}[/green]")

    return output


@run.cli.entrypoint(name="generate", namespace="llm")
def generate(
    path: AnyPath,
    trainer: nl.Trainer,
    prompts: Optional[list[str]] = None,
    encoder_prompts: Optional[list[str]] = None,
    input_dataset: Optional[Union[pl.LightningDataModule, str]] = None,
    params_dtype: torch.dtype = torch.bfloat16,
    add_BOS: bool = False,
    max_batch_size: int = 4,
    random_seed: Optional[int] = None,
    inference_batch_times_seqlen_threshold: int = 1000,
    inference_params: Optional["CommonInferenceParams"] = None,
    text_only: bool = False,
    output_path: Optional[AnyPath] = None,
) -> list[Union["InferenceRequest", str]]:
    """
    Generates text using a NeMo LLM model.

    This function takes a checkpoint path and a list of prompts,
    and generates text based on the loaded model and parameters.
    It returns a list of generated text, either as a string or as an InferenceRequest object.

    Python Usage:
    ```python
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=2,
        num_nodes=1,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
    )
    prompts = [
        "Hello, how are you?",
        "How many r's are in the word 'strawberry'?",
        "Which number is bigger? 10.119 or 10.19?",
    ]

    if __name__ == "__main__":
        results = api.generate(
            path=os.path.join(os.environ["NEMO_HOME"], "models", "meta-llama/Meta-Llama-3-8B"),
            prompts=prompts,
            trainer=trainer,
            inference_params=CommonInferenceParams(temperature=0.1, top_k=10, num_tokens_to_generate=512),
            text_only=True,
        )
    ```

    Args:
        path (Union[Path, str]): The path to the model checkpoint.
        prompts (list[str]): The list of prompts to generate text for.
        trainer (nl.Trainer): The trainer object.
        encoder_prompts (Optional[list[str]], optional): The list of encoder prompts. Defaults to None.
        input_dataset (Optional[Union[pl.LightningDataModule, str]], optional): The input data module or jsonl file.
            Test set will be used for generation for data modules. Defaults to None.
        params_dtype (torch.dtype, optional): The data type of the model parameters. Defaults to torch.bfloat16.
        add_BOS (bool, optional): Whether to add the beginning of sequence token. Defaults to False.
        max_batch_size (int, optional): The maximum batch size. Defaults to 4.
        random_seed (Optional[int], optional): The random seed. Defaults to None.
        inference_batch_times_seqlen_threshold (int, optional): If batch-size times sequence-length is smaller than
            this threshold then we will not use pipelining, otherwise we will. Defaults to 1000.
        inference_params (Optional["CommonInferenceParams"], optional): The inference parameters defined in
            Mcore's CommonInferenceParams. Defaults to None.
        text_only (bool, optional): Whether to return only the generated text as a string. Defaults to False.
        output_path (Optional[Union[Path, str]], optional): The path to save the generated text or test dataset
            predictions. Defaults to None.

    Returns:
        list[Union["InferenceRequest", str]]: A list of generated text,
            either as a string or as an InferenceRequest object.
    """
    from nemo.collections.llm import inference

    if input_dataset is not None:
        input_path = input_dataset if isinstance(input_dataset, str) else input_dataset.test_path
        with open(input_path) as f:
            dataset = [json.loads(sample) for sample in f.readlines()]
            inputs = [sample["input"] for sample in dataset]
    elif prompts is not None:
        inputs = prompts
    else:
        raise ValueError("Either prompts or input_dataset must be provided.")

    inference_wrapped_model, mcore_tokenizer = inference.setup_model_and_tokenizer(
        path=path,
        trainer=trainer,
        params_dtype=params_dtype,
        inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
    )

    max_seq_length = inference_params.num_tokens_to_generate + max(len(mcore_tokenizer.tokenize(p)) for p in inputs)
    # set kv cache allocation to only num tokens in prompt + max tokens to generate
    inference_wrapped_model.inference_wrapper_config.inference_max_seq_length = max_seq_length
    inference_wrapped_model.inference_context.max_sequence_length = max_seq_length

    dp_size = trainer.strategy.distributed_sampler_kwargs['num_replicas']
    dp_rank = trainer.strategy.distributed_sampler_kwargs['rank']
    chunk_size = (len(inputs) + dp_size - 1) // dp_size
    start_idx = dp_rank * chunk_size
    end_idx = min(start_idx + chunk_size, len(inputs))
    inputs_on_this_dp_rank = inputs[start_idx:end_idx]

    results_on_this_dp_rank = inference.generate(
        model=inference_wrapped_model,
        tokenizer=mcore_tokenizer,
        prompts=inputs_on_this_dp_rank,
        encoder_prompts=encoder_prompts,
        add_BOS=add_BOS,
        max_batch_size=max_batch_size,
        random_seed=random_seed,
        inference_params=inference_params,
    )
    gathered_results = [None] * dp_size

    all_gather_object(
        gathered_results,
        [r.generated_text if text_only else r for r in results_on_this_dp_rank],
        group=parallel_state.get_data_parallel_group(),
    )
    gathered_results = [result for sublist in gathered_results for result in sublist]

    assert len(gathered_results) == len(inputs)

    if output_path is not None and is_global_rank_zero():
        with open(output_path, "w") as f:
            for sample, pred in zip(dataset if input_dataset else inputs, gathered_results):
                if type(sample) == dict:
                    sample["label"] = sample.pop("output", None)
                    sample["prediction"] = pred if text_only else pred.generated_text
                elif type(sample) == str:
                    sample = {"input": sample, "prediction": pred if text_only else pred.generated_text}
                f.write(json.dumps(sample) + "\n")
        logging.info(f"Predictions written to {output_path}")

    return gathered_results


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
    # Move jit callback at the end ensure it's applied on top of any model transformations (peft)
    jit_cb = None
    for i, cb in enumerate(trainer.callbacks):
        if isinstance(cb, JitTransform):
            assert jit_cb is None
            jit_cb = trainer.callbacks.pop(i)
    if jit_cb is not None:
        trainer.callbacks.append(jit_cb)
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

    # Model validation
    if hasattr(model, "config"):
        assert getattr(model.config, "seq_length", 1) > 0
        assert getattr(model.config, "max_position_embeddings", 1) > 0
        assert model.config.num_layers > 0
        assert model.config.hidden_size > 0
        assert model.config.num_attention_heads > 0
        assert model.config.ffn_hidden_size > 0
    else:
        assert not isinstance(trainer.strategy, nl.MegatronStrategy), "Expected model.config to exist"

    # Data validation
    assert data.micro_batch_size > 0
    if isinstance(trainer.strategy, nl.MegatronStrategy):
        assert data.global_batch_size > 0
        assert data.seq_length > 0

        assert (
            data.global_batch_size % data.micro_batch_size == 0
        ), "Global batch size must be divisible by micro batch size in data module."

    # Trainer validation

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


def _build_directory_tree(path, tree=None, root_name=None):
    """Build a Rich Tree representation of a directory structure."""
    from rich.tree import Tree

    path = Path(path)
    if tree is None:
        tree = Tree(f"[bold blue]{root_name or path.name}[/bold blue]")

    # Sort to have directories first, then files
    items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))

    for item in items:
        if item.is_dir():
            branch = tree.add(f"[bold cyan]{item.name}/[/bold cyan]")
            _build_directory_tree(item, branch)
        else:
            # Color differently based on file extension
            if item.suffix in ('.json', '.jsonl'):
                tree.add(f"[yellow]{item.name}[/yellow]")
            elif item.suffix in ('.pt', '.bin', '.ckpt', '.nemo'):
                tree.add(f"[magenta]{item.name}[/magenta]")
            elif item.suffix in ('.py', '.sh'):
                tree.add(f"[green]{item.name}[/green]")
            else:
                tree.add(f"[white]{item.name}[/white]")

    return tree
