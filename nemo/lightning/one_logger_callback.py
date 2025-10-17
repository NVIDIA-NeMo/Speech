# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
OneLogger callback for NeMo training.

This module provides a callback that integrates OneLogger telemetry with NeMo training.
"""
import os
from typing import Any, Dict

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.training_telemetry.api.callbacks import on_app_start
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from nv_one_logger.training_telemetry.integration.pytorch_lightning import TimeEventCallback as OneLoggerPTLCallback

from nemo.lightning.base_callback import BaseCallback

# Export all symbols for testing and usage
__all__ = ['OneLoggerNeMoCallback']


def get_one_logger_init_config() -> Dict[str, Any]:
    """Generate minimal configuration for OneLogger initialization.

    This function provides the absolute minimal configuration needed for OneLogger initialization.
    It only includes the required fields and uses defaults for everything else to avoid
    dependencies on exp_manager during early import.

    Returns:
        Dictionary containing minimal initialization configuration
    """
    if "EXP_NAME" in os.environ:
        session_tag = os.environ.get("EXP_NAME")  # For NeMo v1
    else:
        session_tag = os.environ.get("SLURM_JOB_NAME", "nemo-run")

    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Minimal configuration - required fields only
    init_config = {
        # Required fields (from OneLoggerConfig) - no defaults
        "application_name": "nemo",
        "session_tag_or_fn": session_tag,
        # Important fields with defaults - provide if available from config
        "enable_for_current_rank": _should_enable_for_current_rank(),
        "world_size_or_fn": world_size,
    }

    return init_config


def _get_base_callback_config(
    trainer: Any,
    global_batch_size: int,
    seq_length: int,
) -> Dict[str, Any]:
    """Generate base configuration for OneLogger training telemetry.

    This function provides the common configuration needed for both NeMo v1 and v2.
    It extracts basic training information from trainer object and uses provided
    batch size and sequence length values.

    Args:
        trainer: PyTorch Lightning trainer instance
        global_batch_size: Global batch size (calculated by version-specific function)
        seq_length: Sequence length (calculated by version-specific function)

    Returns:
        Dictionary containing base training callback configuration
    """
    # Extract values from trainer
    # Get job name from multiple sources in order of reliability
    if "EXP_NAME" in os.environ:
        job_name = os.environ.get("EXP_NAME")  # For NeMo v1
    else:
        job_name = os.environ.get("SLURM_JOB_NAME", "nemo-run")

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    max_steps = getattr(trainer, 'max_steps', 1)
    log_every_n_steps = getattr(trainer, 'log_every_n_steps', 10)
    micro_batch_size = global_batch_size // world_size
    # Get PERF_VERSION_TAG from environment
    perf_version_tag = os.environ.get('PERF_VERSION_TAG', '0.0.0')

    # Calculate performance tag
    perf_tag = f"{job_name}_{perf_version_tag}_bf{global_batch_size}_se{seq_length}_ws{world_size}"

    # Calculate train samples target
    train_samples_target = max_steps * global_batch_size

    # Fallback values
    is_save_checkpoint_enabled = False
    is_validation_iterations_enabled = False
    save_checkpoint_strategy = "sync"

    checkpoint_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]
    is_save_checkpoint_enabled = len(checkpoint_callbacks) > 0

    val_check_interval = getattr(trainer, 'val_check_interval', -1)
    is_validation_iterations_enabled = val_check_interval > 0

    # Check for async_save in trainer strategy (handle both dict and object cases)
    if hasattr(trainer, 'strategy') and trainer.strategy is not None:
        if isinstance(trainer.strategy, dict):
            if trainer.strategy.get('async_save', False):
                save_checkpoint_strategy = "async"
        else:
            if hasattr(trainer.strategy, 'async_save') and trainer.strategy.async_save:
                save_checkpoint_strategy = "async"

    for callback in checkpoint_callbacks:
        if hasattr(callback, 'async_save') and callback.async_save:
            save_checkpoint_strategy = "async"
            break

    # Base training telemetry configuration
    base_config = {
        # Performance tag (REQUIRED in TrainingTelemetryConfig)
        "perf_tag_or_fn": perf_tag,
        # Batch information (REQUIRED in TrainingTelemetryConfig)
        "global_batch_size_or_fn": global_batch_size,
        "micro_batch_size_or_fn": micro_batch_size,
        "seq_length_or_fn": seq_length,
        # Training targets
        "train_iterations_target_or_fn": max_steps,
        "train_samples_target_or_fn": train_samples_target,
        # Logging frequency
        "log_every_n_train_iterations": log_every_n_steps,
        'is_validation_iterations_enabled_or_fn': is_validation_iterations_enabled,
        'is_save_checkpoint_enabled_or_fn': is_save_checkpoint_enabled,
        'save_checkpoint_strategy': save_checkpoint_strategy,
    }

    return base_config


def get_nemo_v1_callback_config(trainer: Any) -> Dict[str, Any]:
    """Generate NeMo v1 specific configuration for OneLogger training callback.

    This function provides NeMo v1 specific configuration by extracting values from
    the exp_manager_config object and trainer object.

    Args:
        trainer: PyTorch Lightning trainer instance

    Returns:
        Dictionary containing NeMo v1 training callback configuration
    """
    global_batch_size = 1  # Default fallback
    seq_length = 1  # Default fallback

    if (
        hasattr(trainer, 'lightning_module')
        and trainer.lightning_module is not None
        and hasattr(trainer.lightning_module, 'cfg')
    ):
        model_cfg = trainer.lightning_module.cfg
        if hasattr(model_cfg, 'train_ds'):
            train_ds = model_cfg.train_ds
            micro_batch_size = getattr(train_ds, 'batch_size', None)
            if micro_batch_size is not None:
                # Standard fixed-size batching
                global_batch_size = int(micro_batch_size) * int(os.environ.get('WORLD_SIZE', 1))
            else:
                # Try bucketing average first if available
                if hasattr(train_ds, 'bucket_batch_size'):
                    # For ASR with bucketing, use the average batch size
                    bucket_batch_sizes = train_ds.bucket_batch_size
                    # Handle both ListConfig and regular list types
                    if hasattr(bucket_batch_sizes, '__len__') and len(bucket_batch_sizes) > 0:
                        # Convert to list if it's a ListConfig, otherwise use as is
                        bucket_list = (
                            list(bucket_batch_sizes) if hasattr(bucket_batch_sizes, '__iter__') else bucket_batch_sizes
                        )
                        avg_batch_size = sum(bucket_list) / len(bucket_list)
                        global_batch_size = int(avg_batch_size) * int(os.environ.get('WORLD_SIZE', 1))
        if hasattr(model_cfg, 'encoder') and hasattr(model_cfg.encoder, 'd_model'):
            seq_length = model_cfg.encoder.d_model

    # Get base configuration with calculated values
    config = _get_base_callback_config(
        trainer=trainer,
        global_batch_size=global_batch_size,
        seq_length=seq_length,
    )

    return config


def get_nemo_v2_callback_config(
    trainer: Any,
    data: Any,
) -> Dict[str, Any]:
    """Generate NeMo v2 specific configuration for the OneLogger training callback.

    This function extracts the global batch size and sequence length from the provided NeMo v2 data module,
    and uses them to construct the configuration dictionary for the OneLogger training callback.

    Args:
        trainer: PyTorch Lightning trainer instance.
        data: NeMo v2 data module (required).

    Returns:
        Dictionary containing the NeMo v2 training callback configuration.
    """
    # NeMo v2: Extract batch size and sequence length from data module (most reliable source)
    global_batch_size = 1  # Default fallback
    seq_length = 1  # Default fallback

    if data is not None:
        seq_length = data.seq_length
        # Prefer explicit global_batch_size if provided by the data module
        if hasattr(data, 'global_batch_size') and getattr(data, 'global_batch_size') is not None:
            global_batch_size = int(getattr(data, 'global_batch_size'))
        else:
            # Fall back to micro_batch_size multiplied by WORLD_SIZE when global_batch_size is unavailable
            micro_batch_size = getattr(data, 'micro_batch_size', None)
            if micro_batch_size is not None:
                world_size = int(os.environ.get('WORLD_SIZE', 1))
                global_batch_size = int(micro_batch_size) * world_size

    # Get base configuration with calculated values
    config = _get_base_callback_config(
        trainer=trainer,
        global_batch_size=global_batch_size,
        seq_length=seq_length,
    )

    return config


def _should_enable_for_current_rank() -> bool:
    """Determine if OneLogger should be enabled for the current rank.

    Uses environment variables instead of torch.distributed to avoid circular imports.
    In distributed training, typically only rank 0 (or the last rank) should
    enable OneLogger to avoid duplicate telemetry data.

    Returns:
        True if OneLogger should be enabled for the current rank, False otherwise
    """
    rank = int(os.environ.get('RANK', -1))
    # Enable for rank 0 or the last rank (common pattern)
    return rank == 0


class OneLoggerNeMoCallback(OneLoggerPTLCallback, BaseCallback):
    """Adapter extending OneLogger's PTL callback with init + config update.

    __init__ configures the provider from meta info, then calls super().__init__.
    update_config computes TrainingTelemetryConfig and applies it.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, '_initialized', False):
            return
        init_config = get_one_logger_init_config()
        one_logger_config = OneLoggerConfig(**init_config)
        TrainingTelemetryProvider.instance().with_base_config(
            one_logger_config
        ).with_export_config().configure_provider()
        # Initialize underlying OneLogger PTL callback
        super().__init__(TrainingTelemetryProvider.instance(), call_on_app_start=False)
        # Explicitly signal application start after provider configuration
        on_app_start()

    def update_config(self, nemo_version: str, trainer: Trainer, **kwargs) -> None:
        # Avoid this function being called multiple times
        if TrainingTelemetryProvider.instance().config.telemetry_config is not None:
            return
        if nemo_version == 'v1':
            config = get_nemo_v1_callback_config(trainer=trainer)
        elif nemo_version == 'v2':
            # v2 expects data module in kwargs
            data = kwargs.get('data', None)
            config = get_nemo_v2_callback_config(trainer=trainer, data=data)
        else:
            config = get_nemo_v1_callback_config(trainer=trainer)
        training_telemetry_config = TrainingTelemetryConfig(**config)
        TrainingTelemetryProvider.instance().set_training_telemetry_config(training_telemetry_config)
