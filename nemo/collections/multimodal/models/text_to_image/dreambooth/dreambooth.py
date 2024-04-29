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
from functools import partial
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch._inductor import config as inductor_config

from nemo.collections.multimodal.data.dreambooth.dreambooth_dataset import DreamBoothDataset
from nemo.collections.multimodal.modules.stable_diffusion.distributions.distributions import (
    DiagonalGaussianDistribution,
)
from nemo.collections.multimodal.modules.stable_diffusion.encoders.modules import LoraWrapper
from nemo.collections.multimodal.parts.utils import randn_like
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingRandomSampler
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.common import Serialization
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin
from nemo.utils import logging

try:
    from apex import amp
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def _collate_fn(examples, with_prior_preservation=False):
    if with_prior_preservation:
        prompts = [[example["instance_prompt"], example["reg_prompt"]] for example in examples]
        images = [example["instance_images"] for example in examples] + [example["reg_images"] for example in examples]
    else:
        prompts = [[example["instance_prompt"]] for example in examples]
        images = [example["instance_images"] for example in examples]

    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()

    return prompts, images


class DreamBooth(torch.nn.Module, Serialization):
    def __init__(self, cfg, model_parallel_config):
        super().__init__()
        self.cfg = cfg
        self.config = model_parallel_config
        self.with_prior_preservation = self.cfg.with_prior_preservation
        self.num_reg_images = self.cfg.data.num_reg_images
        self.prior_loss_weight = self.cfg.prior_loss_weight
        self.num_images_per_prompt = self.cfg.data.num_images_per_prompt

        self.train_text_encoder = self.cfg.train_text_encoder
        self.instantiate_text_encoder(self.cfg.cond_stage_config)

        self.inductor = self.cfg.inductor
        self.inductor_cudagraphs = self.cfg.inductor_cudagraphs

        self.instantiate_vae(self.cfg.first_stage_config)
        self.instantiate_unet(self.cfg.unet_config)

        self.scale_factor = self.cfg.scale_factor
        self.num_timesteps = self.cfg.noise_scheduler.timesteps
        self.parameterization = self.cfg.noise_scheduler.parameterization
        self.get_noise_scheduler(self.cfg.noise_scheduler)

        self.model_type = None
        self.rng = torch.Generator(device=torch.cuda.current_device(),)

        self.use_cached_latents = self.cfg.use_cached_latents

        if self.cfg.channels_last:
            self.unet = self.unet.to(memory_format=torch.channels_last)

    def instantiate_unet(self, cfg):
        self.unet = DreamBooth.from_config_dict(cfg)
        self.unet.train()
        if self.inductor:
            # TorchInductor with CUDA graph can lead to OOM
            inductor_config.triton.cudagraphs = self.inductor_cudagraphs
            torch._dynamo.config.dynamic_shapes = False
            torch._dynamo.config.automatic_dynamic_shapes = False
            self.unet = torch.compile(self.unet)

    def instantiate_vae(self, cfg):
        model = DreamBooth.from_config_dict(cfg)
        self.vae = model.eval()
        self.vae.train = disabled_train
        for param in self.vae.parameters():
            param.requires_grad = False

    def instantiate_text_encoder(self, cfg):
        model = DreamBooth.from_config_dict(cfg)
        if self.train_text_encoder:
            self.text_encoder = model.train()
            if (not hasattr(model, 'lora_layers')) or len(
                model.lora_layers
            ) == 0:  # if no lora, train all the parameters
                for param in self.text_encoder.parameters():
                    param.requires_grad = True
        else:
            self.text_encoder = model.eval()
            self.text_encoder.train = disabled_train
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def get_noise_scheduler(self, cfg):
        model = DreamBooth.from_config_dict(cfg)
        self.noise_scheduler = model.eval()

    def forward(self, batch):

        x, cond = batch
        if self.use_cached_latents:
            x = DiagonalGaussianDistribution(x)
            latents = x.sample().detach() * self.scale_factor
        else:
            latents = self.vae.encode(x).sample().detach()
            latents = latents * self.scale_factor

        noise = randn_like(latents, generator=self.rng)
        t = torch.randint(0, self.num_timesteps, (latents.shape[0],), generator=self.rng, device=latents.device).long()
        x_noisy = self.noise_scheduler(x_start=latents, t=t, noise=noise)

        # cond = self.text_encoder([t[0] for t in batch["prompts"]])
        # if self.with_prior_preservation:
        #     cond_prior = self.text_encoder([t[1] for t in batch["prompts"]])
        #     cond = torch.cat([cond, cond_prior], dim=0)

        model_output = self.unet(x_noisy, t, cond)

        if self.parameterization == "x0":
            target = latents
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        if self.with_prior_preservation:
            model_pred, model_pred_prior = torch.chunk(model_output, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
            prior_loss = torch.nn.functional.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
            loss = loss + prior_loss * self.prior_loss_weight

        else:
            loss = torch.nn.functional.mse_loss(target.float(), model_output.float(), reduction="mean")
        return loss

    def parameters(self):
        params = list(self.unet.parameters())
        if self.train_text_encoder:
            # print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.text_encoder.parameters())
        return params

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        pass


class MegatronDreamBooth(NLPAdapterModelMixin, MegatronBaseModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
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
        self.model = self.model_provider_func()

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
        model = DreamBooth(cfg=self.cfg, model_parallel_config=self.model_parallel_config)
        return model

    def forward(self, batch):
        output_tensor = self.model(batch)
        return output_tensor

    def fwd_bwd_step(self, dataloader_iter, forward_only):
        tensor_shape = None  # Placeholder

        # handle asynchronous grad reduction
        no_sync_func = None
        if not forward_only and self.with_distributed_adam:
            no_sync_func = partial(self._optimizer.no_sync, greedy_grad_copy=self.megatron_amp_O2,)

        # pipeline schedules will get these from self.model.config
        for module in self.get_module_list():
            module.config.no_sync_func = no_sync_func

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
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

        # losses_reduced_per_micro_batch is a list of dictionaries
        # [{"loss": 0.1}, {"loss": 0.2}, ...] which are from gradient accumulation steps
        # only the last stages of the pipeline return losses
        loss_dict = {}
        if losses_reduced_per_micro_batch:
            if (not forward_only) or self.cfg.data.get('validation_drop_last', True):
                # average loss across micro batches
                prefix = 'train'
                for key in losses_reduced_per_micro_batch[0]:
                    loss_tensors_list = [loss_reduced[key] for loss_reduced in losses_reduced_per_micro_batch]
                    loss_tensor = torch.stack(loss_tensors_list)
                    loss_dict[f'{prefix}/{key}'] = loss_tensor.mean()
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

        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()

        loss_mean, loss_dict = self.fwd_bwd_step(dataloader_iter, False)

        torch.distributed.broadcast(loss_mean, get_last_rank())

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.with_distributed_adam:
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
                self.log('loss_scale', loss_scale, prog_bar=True, batch_size=1)

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

    def validation_step(self, dataloader_iter):
        loss, val_loss_dict = self.fwd_bwd_step(dataloader_iter, True)

        self.log_dict(val_loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)

        return loss

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from apex.
            No need to call it here.
        """
        pass

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        pass

    def _append_sequence_parallel_module_grads(self, module, grads):
        """ Helper method for allreduce_sequence_parallel_gradients"""

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
            # noise_map, condition
            prompts, images = batch
            # DB has more dedicated structure for encoding, so we enable autocasting here as well
            with torch.cuda.amp.autocast(
                self.autocast_dtype in (torch.half, torch.bfloat16), dtype=self.autocast_dtype,
            ):
                images = images.cuda(non_blocking=True)

                cond = self.model.text_encoder([t[0] for t in prompts])
                if self.cfg.with_prior_preservation:
                    cond_prior = self.model.text_encoder([t[1] for t in prompts])
                    cond = torch.cat([cond, cond_prior], dim=0)

            return images, cond

        def fwd_output_and_loss_func(dataloader_iter, model):
            batch, _, _ = next(dataloader_iter)
            batch = process_batch(batch)
            batch = [x.cuda(non_blocking=True) for x in batch]
            loss = model(batch)

            def dummy(output_tensor):
                return loss, {'loss': loss}

            return loss, dummy

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(batch, model):
            raise NotImplementedError

        return fwd_output_only_func

    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
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

        # Batch size need to be provided for webdatset
        self._num_micro_batches = get_num_microbatches()
        self._micro_batch_size = self.cfg.micro_batch_size

        self.setup_training_data(self.cfg.data)
        self.setup_complete = True

    def setup_training_data(self, cfg):
        if self.cfg.with_prior_preservation:
            if cfg.regularization_dir is None:
                raise ValueError("Regularization images must be provided to train with prior preservation loss")
            if cfg.regularization_prompt is None:
                raise ValueError("Regularization prompts must be provided to train with prior preservation loss")

        self.train_dataset = DreamBoothDataset(
            instance_data_root=cfg.instance_dir,
            instance_prompt=cfg.instance_prompt,
            with_prior_preservation=self.cfg.with_prior_preservation,
            reg_data_root=cfg.regularization_dir if self.cfg.with_prior_preservation else None,
            reg_prompt=cfg.regularization_prompt if self.cfg.with_prior_preservation else None,
            size=cfg.resolution,
            center_crop=cfg.center_crop,
            load_cache_latents=self.model.use_cached_latents,
            cached_instance_data_root=self.cfg.data.get("cached_instance_dir", None),
            cached_reg_data_root=self.cfg.data.get("cached_reg_dir", None)
            if self.cfg.with_prior_preservation
            else None,
            vae=self.model.vae,
            text_encoder=self.model.text_encoder,
        )

        batch_sampler = MegatronPretrainingRandomSampler(
            total_samples=len(self.train_dataset),
            consumed_samples=self.compute_consumed_samples(0),
            micro_batch_size=self.cfg.micro_batch_size,
            global_batch_size=self.cfg.global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=True,
        )

        self._train_dl = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=partial(_collate_fn, with_prior_preservation=self.cfg.with_prior_preservation),
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def setup_validation_data(self, cfg):
        pass

    def setup_test_data(self, cfg):
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """ PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
            When using pipeline parallelism, we need the global batch to remain on the CPU,
            since the memory overhead will be too high when using a large number of microbatches.
            Microbatches are transferred from CPU to GPU inside the pipeline.
        """
        return batch

    def _validate_trainer(self):
        """ Certain trainer configurations can break training.
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

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: Any = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        """
        Loads ModelPT from checkpoint, with some maintenance of restoration.
        For documentation, please refer to LightningModule.load_from_checkpoin() documentation.
        """
        checkpoint = None
        try:
            cls._set_model_restore_state(is_being_restored=True)
            # TODO: replace with proper PTL API
            with pl_legacy_patch():
                if map_location is not None:
                    checkpoint = pl_load(checkpoint_path, map_location=map_location)
                else:
                    checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

            if hparams_file is not None:
                extension = hparams_file.split(".")[-1]
                if extension.lower() == "csv":
                    hparams = load_hparams_from_tags_csv(hparams_file)
                elif extension.lower() in ("yml", "yaml"):
                    hparams = load_hparams_from_yaml(hparams_file)
                else:
                    raise ValueError(".csv, .yml or .yaml is required for `hparams_file`")

                hparams["on_gpu"] = False

                # overwrite hparams by the given file
                checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

            # for past checkpoint need to add the new key
            if cls.CHECKPOINT_HYPER_PARAMS_KEY not in checkpoint:
                checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = {}
            # override the hparams with values that were passed in
            cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].get('cfg', checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY])
            # TODO: can we do this without overriding?
            config_kwargs = kwargs.copy()
            if 'trainer' in config_kwargs:
                config_kwargs.pop('trainer')
            cfg.update(config_kwargs)

            # Disable individual unet/vae weights loading otherwise the model will look for these partial ckpts and raise error
            if cfg:
                if cfg.get('unet_config') and cfg.get('unet_config').get('from_pretrained'):
                    cfg.unet_config.from_pretrained = None
                if cfg.get('first_stage_config') and cfg.get('first_stage_config').get('from_pretrained'):
                    cfg.first_stage_config.from_pretrained = None
                ## Now when we covert ckpt to nemo, let's always get rid of those _orig_mod
                if cfg.get('inductor'):
                    cfg.inductor = False
                ## Append some dummy configs that DB didn't support
                if not cfg.get('channels_last'):
                    cfg.channels_last = True
                if not cfg.get('capture_cudagraph_iters'):
                    cfg.capture_cudagraph_iters = -1

            # compatibility for stable diffusion old checkpoint tweaks
            first_key = list(checkpoint['state_dict'].keys())[0]
            if first_key == "betas":
                # insert "model." into for megatron wrapper
                new_state_dict = {}
                for key in checkpoint['state_dict'].keys():
                    new_key = "model." + key
                    new_state_dict[new_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'] = new_state_dict
            elif (
                first_key == 'model.text_encoder.transformer.text_model.embeddings.position_ids'
                or first_key == 'model.text_encoder.model.language_model.embedding.position_embeddings'
            ):
                # remap state keys from dreambooth when using HF clip
                new_state_dict = {}
                for key in checkpoint['state_dict'].keys():
                    new_key = key.replace('._orig_mod', "")
                    new_key = new_key.replace('unet', 'model.diffusion_model')
                    new_key = new_key.replace('vae', 'first_stage_model')
                    new_key = new_key.replace('text_encoder', 'cond_stage_model')
                    new_key = new_key.replace('.noise_scheduler', '')
                    new_state_dict[new_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'] = new_state_dict

            # compatibility for inductor in inference
            if not cfg.get('inductor', False):
                new_state_dict = {}
                for key in checkpoint['state_dict'].keys():
                    new_key = key.replace('._orig_mod', '', 1)
                    new_state_dict[new_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'] = new_state_dict

            if cfg.get('megatron_amp_O2', False):
                new_state_dict = {}
                for key in checkpoint['state_dict'].keys():
                    new_key = key.replace('model.', 'model.module.', 1)
                    new_state_dict[new_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'] = new_state_dict

            if 'cfg' in kwargs:
                model = ptl_load_state(cls, checkpoint, strict=strict, **kwargs)
            else:
                model = ptl_load_state(cls, checkpoint, strict=strict, cfg=cfg, **kwargs)
                # cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].cfg

            checkpoint = model

        finally:
            cls._set_model_restore_state(is_being_restored=False)
        return checkpoint

    def _check_and_add_adapter(self, name, module, peft_name, peft_cfg, name_key_to_mcore_mixins=None):
        from nemo.collections.multimodal.modules.stable_diffusion.attention import LinearWrapper

        if isinstance(module, AdapterModuleMixin):
            if isinstance(module, LinearWrapper):
                peft_cfg.in_features, peft_cfg.out_features = module.in_features, module.out_features
            elif isinstance(module, LoraWrapper):
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
