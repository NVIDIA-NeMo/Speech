# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import sys
from typing import Optional, Union

from lightning_fabric.utilities.exceptions import MisconfigurationException
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.common.metrics.perf_metrics import FLOPsMeasurementCallback
from nemo.collections.nlp.parts.nlp_overrides import (
    CustomProgressBar,
    FSDPMixedPrecisionPlugin,
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPDDPStrategyNotebook,
    NLPFSDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils import logging
from nemo.utils.callbacks.dist_ckpt_io import (
    AsyncFinalizableCheckpointIO,
    AsyncFinalizerCallback,
    DistributedCheckpointIO,
)


class MegatronTrainerBuilder:
    """
    Builder type to hide complex configuration of PTL Trainers for Megatron LLM models.
    Can be extended to change behavior for a specific model.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def _training_strategy(self) -> Union[NLPDDPStrategy, NLPFSDPStrategy]:
        """
        Returns a DDP or a FSDP strategy passed to Trainer.strategy.
        """
        # check interactive environment
        _IS_INTERACTIVE = hasattr(sys, "ps1") or bool(sys.flags.interactive)
        if _IS_INTERACTIVE and self.cfg.trainer.devices == 1:
            logging.info("Detected interactive environment, using NLPDDPStrategyNotebook")
            return NLPDDPStrategyNotebook(
                no_ddp_communication_hook=True,
                find_unused_parameters=False,
            )

        if self.cfg.model.get('fsdp', False):
            assert (
                not self.cfg.model.optim.get('name') == 'distributed_fused_adam'
                and not self.cfg.model.optim.get('name') == 'mcore_distributed_optim'
            ), 'Distributed optimizer cannot be used with FSDP.'
            sharded_checkpoint = self.cfg.model.get('fsdp_sharded_checkpoint', False)
            if self.cfg.model.get('tensor_model_parallel_size', 1) > 1:
                assert not sharded_checkpoint, 'FSDP sharded checkpoint is not supported when TP size > 1.'
            if self.cfg.model.get('megatron_amp_O2', False):
                logging.info('Torch FSDP is not compatible with O2 precision recipe. Setting O2 `False`.')
                self.cfg.model.megatron_amp_O2 = False
            return NLPFSDPStrategy(
                limit_all_gathers=self.cfg.model.get('fsdp_limit_all_gathers', True),
                sharding_strategy=self.cfg.model.get('fsdp_sharding_strategy', 'full'),
                cpu_offload=self.cfg.model.get('fsdp_cpu_offload', False),
                grad_reduce_dtype=self.cfg.model.get('fsdp_grad_reduce_dtype', 32),
                sharded_checkpoint=sharded_checkpoint,
                precision=self.cfg.trainer.precision,
                nccl_communicator_config_path=self.cfg.model.get('nccl_communicator_config_path', None),
                sharp=self.cfg.model.get('sharp', False),
                use_orig_params=self.cfg.model.get('fsdp_use_orig_params', False),
            )

        return NLPDDPStrategy(
            no_ddp_communication_hook=True,
            gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
            find_unused_parameters=False,
            nccl_communicator_config_path=self.cfg.model.get('nccl_communicator_config_path', None),
            sharp=self.cfg.model.get('sharp', False),
            dist_ckpt_parallel_save=self.cfg.model.get('dist_ckpt_parallel_dist_opt', True),
        )

    def _grad_scaler(self) -> GradScaler:
        """
        Returns a scaler for precision plugins.
        """
        return GradScaler(
            init_scale=self.cfg.model.get('native_amp_init_scale', 2**32),
            growth_interval=self.cfg.model.get('native_amp_growth_interval', 1000),
            hysteresis=self.cfg.model.get('hysteresis', 2),
        )

    def _plugins(self) -> list:
        """
        Returns:
            plugins: list of plugins passed to Trainer.plugins including precision plugins.
        """
        megatron_amp_O2 = self.cfg.model.get('megatron_amp_O2', False)
        with_distributed_adam = (
            (
                self.cfg.model.optim.get('name') == 'distributed_fused_adam'
                or self.cfg.model.optim.get('name') == 'mcore_distributed_optim'
            )
            if self.cfg.model.get('optim')
            else False
        )

        plugins = []
        if self.cfg.trainer.precision in [16, '16', 'bf16', '16-mixed', 'bf16-mixed']:
            scaler = None
            if self.cfg.trainer.precision in [16, '16', '16-mixed']:
                if not self.cfg.model.get('fsdp', False):
                    scaler = self._grad_scaler()
                plugin_precision = '16-mixed'
            else:
                plugin_precision = 'bf16-mixed'

            if megatron_amp_O2 and not with_distributed_adam:
                plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
            else:
                if self.cfg.model.get('fsdp', False):
                    plugins.append(FSDPMixedPrecisionPlugin(precision=plugin_precision, scaler=scaler))
                else:
                    plugins.append(
                        PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler)
                    )
            self.cfg.trainer.precision = None

        if self.cfg.get('cluster_type', None) == 'BCP':
            plugins.append(TorchElasticEnvironment())

        # Use dist-ckt for non-FSDP MCore models
        use_dist_ckpt = not self.cfg.model.get('fsdp', False) and (
            self.cfg.model.get('mcore_gpt', False) or self.cfg.model.get('mcore_bert', False)
        )
        async_save = self.cfg.get('exp_manager', {}).get('checkpoint_callback_params', {}).get('async_save', False)
        if use_dist_ckpt:
            checkpoint_io = DistributedCheckpointIO.from_config(self.cfg.model, async_save)
            if async_save:
                checkpoint_io = AsyncFinalizableCheckpointIO(checkpoint_io)
            plugins.append(checkpoint_io)
        elif async_save:
            raise MisconfigurationException(
                'exp_manager.checkpoint_callback_params.async_save=True without'
                'distributed checkpoints is currently not supported'
            )

        return plugins

    def _callbacks(self, callbacks: Optional[list]) -> list:
        """
        Returns:
            callbacks: list of callbacks passed to Trainer.callbacks.
        """
        if callbacks is None:
            callbacks = []
        # enable_progress_bar is True by default. If cfg.trainer.enable_progress_bar=False, CustomProgressBar is not appended to callbacks
        if 'enable_progress_bar' not in self.cfg.trainer or self.cfg.trainer.enable_progress_bar:
            callbacks.append(CustomProgressBar())

        if self.cfg.get('exp_manager', {}).get('checkpoint_callback_params', {}).get('async_save', False):
            callbacks.append(AsyncFinalizerCallback())

        if self.cfg.get('exp_manager', {}).get('log_tflops_per_sec_per_gpu', True):
            callbacks.append(FLOPsMeasurementCallback(self.cfg))

        return callbacks

    def create_trainer(self, callbacks=None) -> Trainer:
        # cfg.trainer.precision becomes None in Trainer if precision_plugins exist since both precision plugins and precision
        precision = self.cfg.trainer.precision
        strategy = self._training_strategy()
        plugins = self._plugins()
        callbacks = self._callbacks(callbacks)
        trainer = Trainer(plugins=plugins, strategy=strategy, **self.cfg.trainer, callbacks=callbacks)
        # Restore the precision value after Trainer is built.
        self.cfg.trainer.precision = precision
        return trainer


class MegatronBertTrainerBuilder(MegatronTrainerBuilder):
    """Builder for BERT model Trainer with overrides."""

    def _grad_scaler(self) -> GradScaler:
        return GradScaler(
            init_scale=self.cfg.model.get('native_amp_init_scale', 2**32),
            growth_interval=self.cfg.model.get('native_amp_growth_interval', 1000),
        )


class MegatronT5TrainerBuilder(MegatronTrainerBuilder):
    """Builder for T5 model Trainer with overrides."""

    def _callbacks(self, callbacks: Optional[list]) -> list:
        callbacks = super()._callbacks(callbacks)
        callbacks.append(ModelSummary(max_depth=3))
        return callbacks

    def create_trainer(self, callbacks=None) -> Trainer:
        strategy = self._training_strategy()
        plugins = self._plugins()
        callbacks = self._callbacks(callbacks)
        return Trainer(plugins=plugins, strategy=strategy, **self.cfg.trainer, callbacks=callbacks)


class MegatronStableDiffusionTrainerBuilder(MegatronTrainerBuilder):
    """Builder for SD model Trainer with overrides."""

    def _training_strategy(self) -> NLPDDPStrategy:
        """
        Returns a ddp strategy passed to Trainer.strategy.
        """
        ddp_overlap = self.cfg.model.get("ddp_overlap", True)
        if ddp_overlap:
            return NLPDDPStrategy(
                no_ddp_communication_hook=False,
                gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
                find_unused_parameters=True,
                bucket_cap_mb=256,
            )
        else:
            return NLPDDPStrategy(
                no_ddp_communication_hook=True,
                gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
                find_unused_parameters=False,
            )


class MegatronLMPPTrainerBuilder(MegatronTrainerBuilder):
    """Builder for scripts where grad scaler is turned off for pipeline parallel LM model. E.g. PEFT tuning scripts"""

    def _grad_scaler(self) -> GradScaler:
        return GradScaler(
            init_scale=self.cfg.model.get("native_amp_init_scale", 2**32),
            growth_interval=self.cfg.model.get("native_amp_growth_interval", 1000),
            hysteresis=self.cfg.model.get("hysteresis", 2),
            enabled=False if self.cfg.model.pipeline_model_parallel_size > 1 else True,
        )
