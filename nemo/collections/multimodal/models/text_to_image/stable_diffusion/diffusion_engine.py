# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractclassmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
import torch._dynamo
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from safetensors.torch import load_file as load_safetensors
from torch._dynamo import optimize
from torch.optim.lr_scheduler import LambdaLR

from nemo.collections.multimodal.data.stable_diffusion.stable_diffusion_dataset import (
    build_sdxl_precached_text_train_valid_datasets,
    build_sdxl_train_valid_datasets,
    build_train_valid_precached_datasets,
)
from nemo.collections.multimodal.modules.stable_diffusion.attention import LinearWrapper
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from nemo.collections.multimodal.parts.stable_diffusion.utils import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP, PEFTConfig
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes import ModelPT, Serialization
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin
from nemo.utils import logging, model_utils

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import get_num_microbatches

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

UNCONDITIONAL_CONFIG = {
    "target": "sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}


class DiffusionEngine(nn.Module, Serialization):
    def __init__(self, cfg, model_parallel_config):
        super().__init__()
        unet_config = cfg.unet_config
        denoiser_config = cfg.denoiser_config
        first_stage_config = cfg.first_stage_config
        conditioner_config = cfg.conditioner_config
        sampler_config = cfg.get('sampler_config', None)
        optimizer_config = cfg.get('optimizer_config', None)
        scheduler_config = cfg.get('scheduler_config', None)
        loss_fn_config = cfg.get('loss_fn_config', None)
        network_wrapper = cfg.get('network_wrapper', None)
        compile_model = cfg.get('compile_model', False)
        self.config = model_parallel_config

        self.channels_last = cfg.get('channels_last', False)
        self.log_keys = cfg.get('log_keys', None)
        self.input_key = cfg.get('input_key', 'images')
        # Precaching
        self.precache_mode = cfg.get('precache_mode')

        self.loss_fn = DiffusionEngine.from_config_dict(loss_fn_config) if loss_fn_config is not None else None

        model = DiffusionEngine.from_config_dict(unet_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(model, compile_model=compile_model)
        if cfg.get('inductor', False):
            torch._dynamo.config.cache_size_limit = 256
            torch._dynamo.config.dynamic_shapes = False
            torch._dynamo.config.automatic_dynamic_shapes = False
            torch._dynamo.config.suppress_errors = True
            self.model = torch.compile(self.model)

        self.denoiser = DiffusionEngine.from_config_dict(denoiser_config)
        self.sampler = instantiate_from_config(sampler_config) if sampler_config is not None else None

        self.conditioner = DiffusionEngine.from_config_dict(default(conditioner_config, UNCONDITIONAL_CONFIG))
        self.scheduler_config = scheduler_config
        # Precaching
        self.precache_mode = cfg.get('precache_mode')
        self._init_first_stage(first_stage_config)
        self.model_type = None

        self.rng = torch.Generator(
            device=torch.cuda.current_device(),
        )

        self.use_ema = False  # TODO use_ema need to switch to NeMo style
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = cfg.scale_factor
        self.disable_first_stage_autocast = cfg.disable_first_stage_autocast
        self.no_cond_log = cfg.get('no_cond_log', False)

        if self.channels_last:
            if self.first_stage_model:
                self.first_stage_model = self.first_stage_model.to(memory_format=torch.channels_last)
            self.model = self.model.to(memory_format=torch.channels_last)

    def _init_first_stage(self, config):
        if self.precache_mode == 'both':
            logging.info('Do not intialize VAE when caching image features.')
            self.first_stage_model = None
            return
        model = DiffusionEngine.from_config_dict(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            out = self.first_stage_model.decode(z)
        return out

    # same as above but differentiable
    def differentiable_decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch, rng=self.rng)
        loss_mean = loss.mean()
        log_prefix = 'train' if self.training else 'val'
        loss_dict = {f"{log_prefix}/loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(params, lr=lr, **cfg.get("params", dict()))

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = DiffusionEngine.from_config_dict(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape, generator=self.rng).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(self.model, input, sigma, c, **kwargs)
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if ((self.log_keys is None) or (embedder.input_key in self.log_keys)) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = ["x".join([str(xx) for xx in x[i].tolist()]) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        # only required for pipeline parallelism
        pass

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        z = self.encode_first_stage(x)
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        return log


class MegatronDiffusionEngine(NLPAdapterModelMixin, MegatronBaseModel):
    """Megatron DiffusionEngine Model."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None
        super().__init__(cfg, trainer=trainer)

        self._validate_trainer()

        # megatron_amp_O2 is not yet supported in diffusion models
        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)
        self.use_fsdp = cfg.get('fsdp', False)

        self.model = self.model_provider_func()

        self.conditioning_keys = []

        if self.trainer.precision in ['bf16', 'bf16-mixed']:
            self.autocast_dtype = torch.bfloat16
        elif self.trainer.precision in [32, '32', '32-true']:
            self.autocast_dtype = torch.float
        elif self.trainer.precision in [16, '16', '16-mixed']:
            self.autocast_dtype = torch.half
        else:
            raise ValueError('precision must be in ["32-true", "16-mixed", "bf16-mixed"]')

    def get_module_list(self):
        if isinstance(self.model, list):
            return [model.module if isinstance(model, Float16Module) else model for model in self.model]
        elif isinstance(self.model, Float16Module):
            return [self.model.module]
        else:
            return [self.model]

    def model_provider_func(self, pre_process=True, post_process=True):
        """Model depends on pipeline paralellism."""
        model = DiffusionEngine(cfg=self.cfg, model_parallel_config=self.model_parallel_config)
        return model

    # def forward(self, x, c, *args, **kwargs):
    #     output_tensor = self.model(x, c, *args, **kwargs)
    #     return output_tensor

    def forward(self, dataloader_iter, batch_idx):
        loss = self.training_step(dataloader_iter, batch_idx)
        return loss

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.cfg.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:
            assert self.cfg.scale_factor == 1.0, 'rather not use custom rescaling and std-rescaling simultaneously'
            batch[self.cfg.first_stage_key] = batch[self.cfg.first_stage_key].cuda(non_blocking=True)
            self.model.on_train_batch_start(batch, batch_idx)

    def fwd_bwd_step(self, dataloader_iter, forward_only):
        tensor_shape = None  # Placeholder

        # handle asynchronous grad reduction
        no_sync_func = None
        if not forward_only and self.with_distributed_adam:
            no_sync_func = partial(
                self._optimizer.no_sync,
                greedy_grad_copy=self.megatron_amp_O2,
            )

        # pipeline schedules will get these from self.model.config
        for module in self.get_module_list():
            module.config.no_sync_func = no_sync_func
        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=dataloader_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=None,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        loss_dict = {}
        if losses_reduced_per_micro_batch:
            if (not forward_only) or self.cfg.data.get('validation_drop_last', True):
                # average loss across micro batches
                for key in losses_reduced_per_micro_batch[0]:
                    loss_tensors_list = [loss_reduced[key] for loss_reduced in losses_reduced_per_micro_batch]
                    loss_tensor = torch.stack(loss_tensors_list)
                    loss_dict[key] = loss_tensor.mean()
                loss_mean = loss_dict["train/loss"]
            else:
                raise NotImplementedError("Losses of micro batches sizes must be uniform!")
        else:
            if forward_only:
                loss_mean = []
            else:
                loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())

        return loss_mean, loss_dict

    def training_step(self, dataloader_iter):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        Batch should be a list of microbatches and those microbatches should on CPU.
        Microbatches are then moved to GPU during the pipeline.
        The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        self._optimizer.zero_grad()

        loss_mean, loss_dict = self.fwd_bwd_step(dataloader_iter, False)

        torch.distributed.broadcast(loss_mean, get_last_rank())

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.use_fsdp:
            pass
        elif self.with_distributed_adam:
            # gradients are reduced internally in distributed optimizer
            pass
        elif self.megatron_amp_O2:
            # # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            # if self.cfg.get('pipeline_model_parallel_size', 1) > 1 or self.cfg.get('sequence_parallel', False):
            #     # main grads are stored in the MainParamsOptimizer wrapper
            #     self._optimizer.allreduce_main_grads()
            self._optimizer.allreduce_main_grads()
        elif not self.cfg.get('ddp_overlap', True):
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.cfg.precision in [16, '16', '16-mixed']:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, rank_zero_only=True, batch_size=1)
        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log('global_step', self.trainer.global_step + 1, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log(
            'consumed_samples',
            self.compute_consumed_samples(self.trainer.global_step + 1 - self.init_global_step),
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )
        return loss_mean

    def backward(self, *args, **kwargs):
        """LightningModule hook to do backward.
        We want this to do nothing since we run backward in the fwd/bwd functions from apex.
        No need to call it here.
        """
        pass

    def optimizer_zero_grad(self, *args, **kwargs):
        """LightningModule hook to zero grad.
        We want this to do nothing as we are zeroing grads during the training_step.
        """
        pass

    def _append_sequence_parallel_module_grads(self, module, grads):
        """Helper method for allreduce_sequence_parallel_gradients"""

        for param in module.parameters():
            sequence_parallel_param = getattr(param, 'sequence_parallel', False)
            if sequence_parallel_param and param.requires_grad:
                if self.megatron_amp_O2:
                    grad = param.main_grad
                else:
                    grad = param.grad
                grads.append(grad.data)

    def get_forward_output_and_loss_func(self):
        def process_batch(batch):
            """Prepares the global batch for apex fwd/bwd functions.
            Global batch is a list of micro batches.
            """
            # SD has more dedicated structure for encoding, so we enable autocasting here as well
            with torch.cuda.amp.autocast(
                self.autocast_dtype in (torch.half, torch.bfloat16),
                dtype=self.autocast_dtype,
            ):
                if self.model.precache_mode == 'both':
                    x = batch[self.model.input_key].to(torch.cuda.current_device())
                    if self.model.channels_last:
                        x = x.to(memory_format=torch.channels_last, non_blocking=True)
                    else:
                        x = x.to(memory_format=torch.contiguous_format, non_blocking=True)
                else:
                    x = batch[self.model.input_key].to(torch.cuda.current_device())
                    if self.model.channels_last:
                        x = x.permute(0, 3, 1, 2).to(memory_format=torch.channels_last, non_blocking=True)
                    else:
                        x = rearrange(x, "b h w c -> b c h w")
                        x = x.to(memory_format=torch.contiguous_format, non_blocking=True)
                    x = self.model.encode_first_stage(x)

                batch['global_step'] = self.trainer.global_step

            return x, batch

        def fwd_output_and_loss_func(dataloader_iter, model):
            batch, _, _ = next(dataloader_iter)
            x, batch = process_batch(batch)

            loss, loss_dict = model(x, batch)

            def dummy(output_tensor):
                return loss, loss_dict

            # output_tensor, and a function to convert output_tensor to loss + loss_dict
            return loss, dummy

        return fwd_output_and_loss_func

    def validation_step(self, dataloader_iter, batch_idx):
        loss, val_loss_dict = self.fwd_bwd_step(dataloader_iter, batch_idx, True)

        self.log_dict(val_loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)

        return loss

    def setup(self, stage=None):
        """PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """
        self.model.rng.manual_seed(self.cfg.seed + 100 * parallel_state.get_data_parallel_rank())

        # log number of parameters
        if isinstance(self.model, list):
            num_parameters_on_device = sum(
                [sum([p.nelement() for p in model_module.parameters()]) for model_module in self.model]
            )
        else:
            num_parameters_on_device = sum([p.nelement() for p in self.model.parameters()])

        # to be summed across data parallel group
        total_num_parameters = torch.tensor(num_parameters_on_device).cuda(non_blocking=True)

        torch.distributed.all_reduce(total_num_parameters, group=parallel_state.get_model_parallel_group())

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
            f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
            f'Total number of model parameters: {total_num_parameters:.2e}.'
        )

        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        # allowing restored models to optionally setup datasets
        self.build_train_valid_test_datasets()

        # Batch size need to be provided for webdatset
        self._num_micro_batches = get_num_microbatches()
        self._micro_batch_size = self.cfg.micro_batch_size

        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        self.setup_test_data(self.cfg.data)

    def build_train_valid_test_datasets(self):
        logging.info('Building datasets for Stable Diffusion...')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")

        if self.model.precache_mode == 'text':
            logging.info('Precahing text only.')
            build_dataset_cls = build_sdxl_precached_text_train_valid_datasets
        elif self.model.precache_mode == 'both':
            logging.info('Precaching text and image.')
            build_dataset_cls = build_train_valid_precached_datasets
        elif self.model.precache_mode is None:
            build_dataset_cls = build_sdxl_train_valid_datasets
        else:
            raise ValueError("unsupported precache mode provided. Check your config file.")
        self._train_ds, self._validation_ds = build_dataset_cls(
            model_cfg=self.cfg, consumed_samples=self.compute_consumed_samples(0)
        )
        self._test_ds = None

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building datasets for LatentDiffusion.')
        return self._train_ds, self._validation_ds, self._test_ds

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds') and self._train_ds is not None:
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = torch.utils.data.DataLoader(
                self._train_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds') and self._validation_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            self._validation_dl = torch.utils.data.DataLoader(
                self._validation_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
            )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds') and self._test_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = torch.utils.data.DataLoader(
                self._test_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
        When using pipeline parallelism, we need the global batch to remain on the CPU,
        since the memory overhead will be too high when using a large number of microbatches.
        Microbatches are transferred from CPU to GPU inside the pipeline.
        """
        return batch

    def _validate_trainer(self):
        """Certain trainer configurations can break training.
        Here we try to catch them and raise an error.
        """
        if self.trainer.accumulate_grad_batches > 1:
            raise ValueError(
                f'Gradient accumulation is done within training_step. trainer.accumulate_grad_batches must equal 1'
            )

    @classmethod
    def list_available_models(cls):
        return None

    def parameters(self):
        if isinstance(self.model, list):
            return itertools.chain.from_iterable(module.parameters() for module in self.model)
        else:
            return self.model.parameters()

    def _check_and_add_adapter(self, name, module, peft_name, peft_cfg, name_key_to_mcore_mixins=None):
        if isinstance(module, AdapterModuleMixin):
            if isinstance(module, LinearWrapper):
                peft_cfg.in_features, peft_cfg.out_features = module.in_features, module.out_features
            else:
                return
            if model_utils.import_class_by_path(peft_cfg._target_) in module.get_accepted_adapter_types():
                module.add_adapter(
                    name=peft_name,
                    cfg=peft_cfg,
                    base_model_cfg=self.cfg,
                    model_parallel_config=self.model_parallel_config,
                )
