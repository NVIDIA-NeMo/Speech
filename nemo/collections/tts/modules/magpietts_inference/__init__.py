# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
MagpieTTS inference and evaluation subpackage.

This package provides modular components for:
- Model loading and configuration (utils.py)
- Batch inference (inference.py)
- Audio quality evaluation (evaluation.py)
- Metrics visualization (visualization.py)

Example Usage:
    from examples.tts.magpietts import (
        InferenceConfig,
        MagpieInferenceRunner,
        load_magpie_model,
        ModelLoadConfig,
    )

    # Load model
    model_config = ModelLoadConfig(
        nemo_file="/path/to/model.nemo",
        codecmodel_path="/path/to/codec.nemo",
    )
    model, checkpoint_name = load_magpie_model(model_config)

    # Log architecture summary and retrieve MoE info + FLOPs metrics
    moe_info, flops_per_component = log_model_architecture_summary(model)

    # Create runner and run inference
    inference_config = InferenceConfig()
    runner = MagpieInferenceRunner(model, inference_config)
"""

from nemo.collections.tts.modules.magpietts_inference.evaluation import (
    DEFAULT_VIOLIN_METRICS,
    EvaluationConfig,
    compute_mean_with_confidence_interval,
    evaluate_generated_audio_dir,
)
from nemo.collections.tts.modules.magpietts_inference.inference import InferenceConfig, MagpieInferenceRunner
from nemo.collections.tts.modules.magpietts_inference.utils import (
    ModelLoadConfig,
    compute_ffn_flops_per_token,
    get_experiment_name_from_checkpoint_path,
    load_magpie_model,
    log_model_architecture_summary,
)
from nemo.collections.tts.modules.magpietts_inference.visualization import create_combined_box_plot, create_violin_plot

__all__ = [
    # Utils
    "ModelLoadConfig",
    "load_magpie_model",
    "compute_ffn_flops_per_token",
    "get_experiment_name_from_checkpoint_path",
    "log_model_architecture_summary",
    # Inference
    "InferenceConfig",
    "MagpieInferenceRunner",
    # Evaluation
    "EvaluationConfig",
    "evaluate_generated_audio_dir",
    "compute_mean_with_confidence_interval",
    "DEFAULT_VIOLIN_METRICS",
    # Visualization
    "create_violin_plot",
    "create_combined_box_plot",
]
