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
import json
import os
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
from nemo.collections.llm.quantization import ExportConfig, QuantizationConfig
from nemo.lightning import (
    AutoResume,
    NeMoLogger,
    OptimizerModule,
    Trainer,
    configure_no_restart_validation_training_loop,
    io,
)
from nemo.lightning.base import NEMO_MODELS_CACHE
from nemo.lightning.pytorch.callbacks import PEFT, JitTransform, ModelTransform
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

if TYPE_CHECKING:
    from megatron.core.inference.common_inference_params import CommonInferenceParams
    from megatron.core.inference.inference_request import InferenceRequest


TokenizerType = Any


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


@run.cli.entrypoint(name="ptq", namespace="llm")
def ptq(
    nemo_checkpoint: str,
    export_config: ExportConfig,
    calibration_tp: int = 1,
    calibration_pp: int = 1,
    quantization_config: Annotated[Optional[QuantizationConfig], run.Config[QuantizationConfig]] = None,
) -> Path:
    """
    Applies Post-Training Quantization (PTQ) for a model using the specified quantization and export configs. It runs
    calibration for a small dataset to collect scaling factors low-precision GEMMs used by desired quantization method.
    This function produces TensorRT-LLM checkpoint ready for deployment using nemo.export and nemo.deploy modules
    or direcly using TensorRT-LLM library.
    The function can be used through the NeMo CLI in the following way:
    ```bash
    # Run calibration using tensor parallel set to 8 and export quantized checkpoint with tensor parallel equal 2
    nemo llm ptq nemo_checkpoint=/models/Llama-3-70B \
        export_config.path=/models/Llama-3-70B-FP8 \
        calibration_tp=8 \
        export_config.inference_tp=2
    # Choose different quantization method, for example, INT8 SmoothQuant
    nemo llm ptq nemo_checkpoint=/models/Llama-3-8B \
        export_config.path=/models/Llama-3-8B-INT8_SQ \
        quantization_config.algorithm=int8_sq
    ```
    Args:
        nemo_checkpoint (str): The path to model to be quantized.
        calibration_tp (int): Calibration tensor parallelism.
        calibration_pp (int): Calibration pipeline parallelism.
        quantization_config (QuantizationConfig): Configuration for quantization algorithm.
        export_config (ExportConfig): Export configuration for TensorRT-LLM checkpoint.
    Returns:
        Path: The path where the quantized checkpoint has been saved after calibration.
    """
    if not quantization_config:
        quantization_config = QuantizationConfig()

    if export_config.path is None:
        raise ValueError("The export_config.path needs to be specified, got None.")

    from nemo.collections.llm import quantization

    quantizer = quantization.Quantizer(quantization_config, export_config)

    model = quantization.load_with_modelopt_layer_spec(nemo_checkpoint, calibration_tp, calibration_pp)

    model = quantizer.quantize(model)

    quantizer.export(model, nemo_checkpoint)

    console = Console()
    console.print(f"[green]✓ PTQ succeded, quantized checkpoint exported to {export_config.path}[/green]")

    return export_config.path


@run.cli.entrypoint(namespace="llm")
def deploy(
    nemo_checkpoint: Path = None,
    model_type: str = "llama",
    triton_model_name: str = "triton_model",
    triton_model_version: Optional[int] = 1,
    triton_http_port: int = 8000,
    triton_grpc_port: int = 8001,
    triton_http_address: str = "0.0.0.0",
    triton_model_repository: Path = None,
    num_gpus: int = 1,
    tensor_parallelism_size: int = 1,
    pipeline_parallelism_size: int = 1,
    dtype: str = "bfloat16",
    max_input_len: int = 256,
    max_output_len: int = 256,
    max_batch_size: int = 8,
    output_generation_logits: bool = True,
):
    """
    Deploys nemo model on a PyTriton server by converting the nemo ckpt to trtllm.

    Args:
        nemo_checkpoint (Path): Path for nemo checkpoint.
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
        num_gpus (int): Number of GPUs for export to trtllm and deploy. Default: 1.
        tensor_parallelism_size (int): Tensor parallelism size. Default: 1.
        pipeline_parallelism_size (int): Pipeline parallelism size. Default: 1.
        dtype (str): dtype of the TensorRT-LLM model. Default: "bfloat16".
        max_input_len (int): Max input length of the model. Default: 256.
        max_output_len (int): Max output length of the model. Default: 256.
        max_batch_size (int): Max batch size of the model. Default: 8.
        Needs to be True to be able to run evaluation. Default: True.
        openai_format_response (bool): Return the response from PyTriton server in OpenAI compatible format. Needs to
        be True while running evaluation. Default: True.
        output_generation_logits (bool): If True builds trtllm engine with gather_generation_logits set to True.
        generation_logits are used to compute the logProb of the output token. Default: True.
    """
    from nemo.collections.llm.deploy.base import get_trtllm_deployable, unset_environment_variables
    from nemo.deploy import DeployPyTriton

    unset_environment_variables()

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


def evaluate(
    nemo_checkpoint_path: Path,
    url: str = "grpc://0.0.0.0:8001",
    triton_http_port: int = 8000,
    model_name: str = "triton_model",
    eval_task: str = "gsm8k",
    num_fewshot: Optional[int] = None,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    # inference params
    max_tokens_to_generate: Optional[int] = 256,
    temperature: Optional[float] = 0.000000001,
    top_p: Optional[float] = 0.0,
    top_k: Optional[int] = 1,
    add_bos: Optional[bool] = False,
):
    """
    Evaluates nemo model deployed on PyTriton server (via trtllm) using lm-evaluation-harness
    (https://github.com/EleutherAI/lm-evaluation-harness/tree/main).

    Args:
        nemo_checkpoint_path (Path): Path for nemo 2.0 checkpoint. This is used to get the tokenizer from the ckpt
        which is required to tokenize the evaluation input and output prompts.
        url (str): grpc service url that were used in the deploy method above in the format: grpc://{grpc_service_ip}:{grpc_port}.
        triton_http_port (int): HTTP port that was used for the PyTriton server in the deploy method. Default: 8000.
        Please pass the triton_http_port if using a custom port in the deploy method.
        model_name (str): Name of the model that is deployed on PyTriton server. It should be the same as
        triton_model_name passed to the deploy method above to be able to launch evaluation. Deafult: "triton_model".
        eval_task (str): task to be evaluated on. For ex: "gsm8k", "gsm8k_cot", "mmlu", "lambada". Default: "gsm8k".
        These are the tasks that are supported currently. Any other task of type generate_until or loglikelihood from
        lm-evaluation-harness can be run, but only the above mentioned ones are tested. Tasks of type
        loglikelihood_rolling are not supported yet.
        num_fewshot (int): number of examples in few-shot context. Default: None.
        limit (Union[int, float]): Limit the number of examples per task. If <1 (i.e float val between 0 and 1), limit
        is a percentage of the total number of examples. If int say x, then run evaluation only on x number of samples
        from the eval dataset. Default: None, which means eval is run the entire dataset.
        bootstrap_iters (int): Number of iterations for bootstrap statistics, used when calculating stderrs. Set to 0
        for no stderr calculations to be performed. Default: 100000.
        # inference params
        max_tokens_to_generate (int): max tokens to generate. Default: 256.
        temperature: Optional[float]: float value between 0 and 1. temp of 0 indicates greedy decoding, where the token
        with highest prob is chosen. Temperature can't be set to 0.0 currently, due to a bug with TRTLLM
        (# TODO to be investigated). Hence using a very samll value as the default. Default: 0.000000001.
        top_p: Optional[float]: float value between 0 and 1. limits to the top tokens within a certain probability.
        top_p=0 means the model will only consider the single most likely token for the next prediction. Default: 0.0.
        top_k: Optional[int]: limits to a certain number (K) of the top tokens to consider. top_k=1 means the model
        will only consider the single most likely token for the next prediction. Default: 1
        add_bos: Optional[bool]: whether a special token representing the beginning of a sequence should be added when
        encoding a string. Default: False since typically for CausalLM its set to False. If needed set add_bos to True.
    """
    try:
        # lm-evaluation-harness import
        from lm_eval import evaluator
    except ImportError:
        raise ImportError(
            "Please ensure that lm-evaluation-harness is installed in your env as it is required " "to run evaluations"
        )

    from nemo.collections.llm import evaluation

    # Get tokenizer from nemo ckpt. This works only with NeMo 2.0 ckpt.
    tokenizer = io.load_context(nemo_checkpoint_path + "/context", subpath="model.tokenizer")
    # Wait for server to be ready before starting evaluation
    evaluation.wait_for_server_ready(url=url, triton_http_port=triton_http_port, model_name=model_name)
    # Create an object of the NeMoFWLM which is passed as a model to evaluator.simple_evaluate
    model = evaluation.NeMoFWLMEval(
        model_name, url, tokenizer, max_tokens_to_generate, temperature, top_p, top_k, add_bos
    )
    results = evaluator.simple_evaluate(
        model=model,
        tasks=eval_task,
        limit=limit,
        num_fewshot=num_fewshot,
        bootstrap_iters=bootstrap_iters,
    )

    print("score", results["results"][eval_task])


@run.cli.entrypoint(name="import", namespace="llm")
def import_ckpt(
    model: pl.LightningModule,
    source: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
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
    """
    output = io.import_ckpt(model=model, source=source, output_path=output_path, overwrite=overwrite)

    console = Console()
    if output_path:
        console.print(f"[green]✓ Checkpoint imported to {output}[/green]")
    else:
        console.print(f"[green] $NEMO_MODELS_CACHE={NEMO_MODELS_CACHE} [/green]")
        console.print(f"[green]✓ Checkpoint imported to {output}[/green]")

    return output


def load_connector_from_trainer_ckpt(path: Path, target: str) -> io.ModelConnector:
    return io.load_context(path, subpath="model").exporter(target, path)


@run.cli.entrypoint(name="export", namespace="llm")
def export_ckpt(
    path: Path,
    target: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    load_connector: Callable[[Path, str], io.ModelConnector] = load_connector_from_trainer_ckpt,
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
    output = io.export_ckpt(path, target, output_path, overwrite, load_connector)

    console = Console()
    console.print(f"[green]✓ Checkpoint exported to {output}[/green]")

    return output


@run.cli.entrypoint(name="generate", namespace="llm")
def generate(
    path: Union[Path, str],
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
    output_path: Optional[Union[Path, str]] = None,
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
    assert data.micro_batch_size > 0
    assert data.global_batch_size > 0
    assert data.seq_length > 0

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
