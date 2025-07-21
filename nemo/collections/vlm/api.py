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

from pathlib import Path
from typing import Callable, Optional

import nemo_run as run
from rich.console import Console
from typing_extensions import Annotated

from nemo.collections.llm.modelopt import ExportConfig, QuantizationConfig, Quantizer
from nemo.collections.vlm import HFAutoModelForImageTextToText
from nemo.collections.vlm.modelopt import setup_trainer_and_restore_model_with_modelopt_spec
from nemo.utils.get_rank import is_global_rank_zero


@run.cli.entrypoint(name="ptq", namespace="vlm")
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
    legacy_ckpt: bool = False,
    trust_remote_code: bool = False,
) -> Path:
    """
    Applies Post-Training Quantization (PTQ) for a vision-language model using the
    specified quantization and export configs.
    It runs calibration for a small dataset to collect scaling factors low-precision
    GEMMs used by desired quantization method.
    By default, this function produces TensorRT-LLM checkpoint ready for deployment using nemo.export and nemo.deploy
    modules or directly using TensorRT-LLM library.

    Args:
        model_path (str): The path to model to be quantized.
        export_config (ExportConfig): Export configuration for output checkpoint.
        calibration_tp (int): Calibration tensor parallelism.
        calibration_pp (int): Calibration pipeline parallelism.
        num_layers_in_first_pipeline_stage (int): Number of layers in the first pipeline stage.
        num_layers_in_last_pipeline_stage (int): Number of layers in the last pipeline stage.
        devices (int): Number of devices to use for calibration. Default: calibration_tp.
        num_nodes (int): Number of nodes to use for calibration. Default: calibration_pp.
        quantization_config (QuantizationConfig): Configuration for quantization algorithm.
        forward_loop (Callable): Forward loop to use for calibration.
            If not provided, a forward loop will be created using the calibration dataset.
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
        model = HFAutoModelForImageTextToText(
            model_name=model_path, trust_remote_code=trust_remote_code, device_map="auto"
        )
        model.configure_model()
    else:
        assert export_config.export_format != "hf", "Automodel PTQ does not support export format hf"
        model, trainer = setup_trainer_and_restore_model_with_modelopt_spec(
            model_path=model_path,
            tensor_model_parallel_size=calibration_tp,
            pipeline_model_parallel_size=calibration_pp,
            num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
            devices=devices,
            num_nodes=num_nodes,
            inference_only=True,
            legacy_ckpt=legacy_ckpt,
            strategy_kwargs={"sequence_parallel": False, "lazy_init": True},
            trainer_kwargs={},
            model_config_overrides={"sequence_parallel": False},
        )

    model = quantizer.quantize(model, forward_loop)
    quantizer.export(model, model_path, trainer)

    if is_global_rank_zero():
        console = Console()
        console.print(f"[green]✓ PTQ succeeded, quantized checkpoint exported to {export_config.path}[/green]")
    return export_config.path
