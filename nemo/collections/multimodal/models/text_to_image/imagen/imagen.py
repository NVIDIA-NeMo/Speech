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
import itertools
from datetime import datetime
from functools import partial
from typing import Any

import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer

from nemo.collections.multimodal.data.imagen.imagen_dataset import build_train_valid_datasets
from nemo.collections.multimodal.models.text_to_image.imagen.precond import ContinousDDPMPrecond, EDMPrecond
from nemo.collections.multimodal.modules.imagen.diffusionmodules.nets import EfficientUNetModel, UNetModel
from nemo.collections.multimodal.modules.imagen.encoder.t5encoder import T5Encoder
from nemo.collections.multimodal.modules.imagen.sampler.sampler import DDPMSampler, EDMSampler
from nemo.collections.multimodal.parts.imagen.utils import random_dropout
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.common import Serialization
from nemo.utils import logging

try:
    from apex import amp

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

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

try:
    from apex.contrib.group_norm import GroupNorm

    OPT_GROUP_NORM = True
except Exception:
    print('Fused optimized group norm has not been installed.')
    OPT_GROUP_NORM = False

DUMMY_TENSOR = torch.tensor([1.0])


class Imagen(torch.nn.Module, Serialization):
    def __init__(self, cfg, model_parallel_config):
        super().__init__()
        self.cfg = cfg
        self.config = model_parallel_config
        # Make sure the initialization on different GPUs are the same
        self.unet_type = cfg.get('unet_type', 'base')
        self.noise_cond_aug = cfg.get('noise_cond_aug', False)
        if self.unet_type == 'base':
            logging.info('Initializing UNet.')
            unet = UNetModel(**cfg.unet, text_embed_dim=cfg.conditioning.embed_dim)
        elif self.unet_type == 'sr':
            logging.info('Initializing Efficient-UNet.')
            unet = EfficientUNetModel(
                **cfg.unet, text_embed_dim=cfg.conditioning.embed_dim, noise_cond_aug=self.noise_cond_aug
            )
        elif self.unet_type == 'sr-unet':
            logging.info('Initializing UNet for SR model.')
            unet = UNetModel(**cfg.unet, text_embed_dim=cfg.conditioning.embed_dim, noise_cond_aug=self.noise_cond_aug)
        else:
            raise NotImplemented(f'{self.unet_type} UNet is not implemented.')

        self.channels_last = cfg.get('channels_last', False)
        if self.channels_last:
            assert OPT_GROUP_NORM, 'Training in channels last format requires optmized group norm implementation.'
            logging.info('Training in torch channels last format.')
            unet = unet.to(memory_format=torch.channels_last)

        # Preconditioning
        self.preconditioning_type = cfg.get('preconditioning_type', 'DDPM')
        if self.preconditioning_type == 'DDPM':
            logging.info('Preconditioned with Continous DDPM')
            self.model = ContinousDDPMPrecond(unet=unet, **cfg.preconditioning, noise_cond_aug=self.noise_cond_aug)
            self.sampler = DDPMSampler(unet_type=self.unet_type, denoiser=self.model.scheduler)
        elif self.preconditioning_type == 'EDM':
            logging.info('Preconditioned with EDM')
            self.model = EDMPrecond(unet=unet, **cfg.preconditioning, noise_cond_aug=self.noise_cond_aug)
            self.sampler = EDMSampler(unet_type=self.unet_type)
        else:
            raise NotImplemented(f'{self.preconditioning_type} preconditioning is not implemented.')

        self.rng = None
        self.conditioning = cfg.conditioning
        self.text_drop_rate = cfg.conditioning.drop_rate
        self.model_type = None
        self.image_size = cfg.unet.image_size

    def setup_rng(self):
        # We need to set different rng seed for different GPUs/ different runs;
        # otherwise, the noise map and time will be exactly the same.
        self.rng = torch.Generator(device=torch.cuda.current_device())
        self.rng_seed = int(datetime.now().timestamp()) + self.cfg.seed + parallel_state.get_data_parallel_rank()
        logging.info(f'RNG seed set as {self.rng_seed} for rank {parallel_state.get_data_parallel_rank()}')
        self.rng.manual_seed(self.rng_seed)
        self.model.set_rng(self.rng)

    @property
    def unet(self):
        return self.model.unet

    def get_text_encoder(self, encoder_path=None):
        # TODO Assume using T5 for all
        return T5Encoder(max_seq_len=self.conditioning.token_length, encoder_path=encoder_path)

    def forward(self, x_start, text_embed, text_mask, x_lowres=None):
        if self.unet_type == 'base':
            assert x_lowres[0].item() == DUMMY_TENSOR.item(), 'Base model should have no low-resolution conditioning'
            x_lowres = None
        else:
            assert x_lowres[0].dim() not in [0, 1], 'SR model should have low-resolution conditioning'

        if self.channels_last:
            x_start = x_start.to(memory_format=torch.channels_last)
            if x_lowres is not None:
                x_lowres = x_lowres.to(memory_format=torch.channels_last)

        # Apply random dropout to text embedding
        text_embed = random_dropout(text_embed, drop_rate=self.text_drop_rate)
        # UNet Forward Pass
        low_res_cond = {'x_low_res': x_lowres} if x_lowres is not None else {}
        # UNet Forward Pass and compute loss
        loss = self.model.compute_loss(
            x0=x_start,
            text_embed=text_embed,
            text_mask=text_mask,
            time=None,  # Randomly Sample
            noise=None,  # Randomly Sample
            **low_res_cond,
        )
        return loss, {'train/loss': loss}

    @torch.no_grad()
    def sample_image(
        self,
        noise_map,
        text_encoding,
        text_mask,
        x_low_res=None,
        cond_scale=1.0,
        sampling_steps=None,
        thresholding_method='dynamic',
    ):
        return self.sampler(
            self.model, noise_map, text_encoding, text_mask, x_low_res, cond_scale, sampling_steps, thresholding_method
        )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        # only required for pipeline parallelism
        pass


class MegatronImagen(MegatronBaseModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        with open_dict(cfg):
            cfg.hidden_size = cfg.unet.embed_dim
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

        self.online_encoding = cfg.conditioning.get("online_encoding", False)
        self.text_encoder_path = cfg.conditioning.get("encoder_path", None)

    def get_module_list(self):
        if isinstance(self.model, list):
            return [model.module if isinstance(model, Float16Module) else model for model in self.model]
        elif isinstance(self.model, Float16Module):
            return [self.model.module]
        else:
            return [self.model]

    def model_provider_func(self, pre_process=True, post_process=True):
        """Model depends on pipeline paralellism."""
        model = Imagen(cfg=self.cfg, model_parallel_config=self.model_parallel_config)
        return model

    def get_forward_output_and_loss_func(self):
        def process_batch(batch):
            """Prepares the batch for megatron fwd/bwd functions.
            Global batch is a list of micro batches.
            """
            # Base model and SR models have slightly different batch input:
            # Base model would only require images (64x64),
            # while SR models (both SR256 and SR1024) require low-res image (64x64) and
            # actual (cropped) image (256x256)
            if self.cfg.unet_type == 'base':
                x_start = batch['images']
                # Pass in DUMMY_TENSOR because megatron requires each input to be
                # tensor (not None) with same batch size (first dim)
                x_lowres = DUMMY_TENSOR.repeat(x_start.shape[0])
            elif self.cfg.unet_type == 'sr' or self.cfg.unet_type == 'sr-unet':
                x_start = batch['images_256']
                x_lowres = batch['images_64']
            else:
                raise NotImplemented(f'Unknown UNet type: {self.cfg.unet_type}')

            if self.cfg.conditioning.get("online_encoding", False):
                input_text = batch["raw_text"]
                # Encode the text embeddings using text encoder.
                with torch.no_grad():
                    text_embed, text_mask = self.text_encoder.encode(input_text)
            else:
                text_conditioning_key = self.cfg.conditioning.out_key
                text_embed = batch[f'{text_conditioning_key}_embeddings']
                text_mask = batch[f'{text_conditioning_key}_mask']
            return [x_start, text_embed, text_mask, x_lowres]

        def fwd_output_and_loss_func(dataloader_iter, model):
            batch, _, _ = next(dataloader_iter)
            batch = process_batch(batch)
            batch = [x.cuda(non_blocking=True) for x in batch]
            loss, loss_dict = model(*batch)

            def dummy(output_tensor):
                return loss, loss_dict

            # output_tensor, and a function to convert output_tensor to loss + loss_dict
            return loss, dummy

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(batch, model):
            raise NotImplementedError

        return fwd_output_only_func

    def build_train_valid_test_datasets(self):
        logging.info('Building datasets for Imagen...')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        self._train_ds, self._validation_ds = build_train_valid_datasets(
            model_cfg=self.cfg, consumed_samples=self.compute_consumed_samples(0)
        )
        # We do not have test dataset
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

    def fwd_bwd_step(self, dataloader_iter, forward_only):
        tensor_shape = None

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

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = get_forward_backward_func()

        # TODO @akhattar: add num_micro_batches_with_partial_activation_checkpoints when ready
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
                for key in losses_reduced_per_micro_batch[0]:
                    loss_tensors_list = [loss_reduced[key] for loss_reduced in losses_reduced_per_micro_batch]
                    loss_tensor = torch.stack(loss_tensors_list)
                    loss_dict[key] = loss_tensor.mean()
                    loss_mean = loss_dict["train/loss"]
            else:
                # Get the total loss since micro batches sizes are not uniform
                raise NotImplementedError("Losses of micro batches sizes must be uniform!")
        else:
            # we're not on the last pipeline stage so no losses
            if forward_only:
                loss_mean = []
            else:
                loss_mean = torch.tensor(0.0).cuda()

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

        # we zero grads here because we also call backward in the megatron-core fwd/bwd functions
        self._optimizer.zero_grad()

        loss_mean, loss_dict = self.fwd_bwd_step(dataloader_iter, False)

        torch.distributed.broadcast(loss_mean, get_last_rank())

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.with_distributed_adam:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            self._optimizer._finish_bucket_grad_sync()
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

    def validation_step(self, dataloader_iter):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions."""

        loss, val_loss_dict = self.fwd_bwd_step(dataloader_iter, True)

        self.log_dict(val_loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)
        return loss

    def setup(self, stage=None):
        """PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """

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
        # Setup RNG seed in model
        self.model.setup_rng()

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

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.online_encoding:
            # Removing the weights relating to Text encoder when saving the checkpoints
            frozen_weights_keys = [k for k in checkpoint['state_dict'].keys() if k.startswith("text_encoder")]
            for k in frozen_weights_keys:
                del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint) -> None:
        # make sure inductor naming is consistent with checkpoint's
        inductor_enabled = self.cfg.get('inductor', False)
        state_dict = checkpoint['state_dict']
        inductor_checkpoint = False
        for (
            k,
            v,
        ) in state_dict.items():
            if '_orig_mod' in k:
                inductor_checkpoint = True
                break

        if inductor_enabled and not inductor_checkpoint:
            # ckpt needs to be converted to inductor-format weights (add .orig_mod)
            logging.info('Add .orig_mod to all weight keys.')
            new_state_dict = {}
            for k, v in state_dict.items():
                idx = k.find('._orig_mod')
                new_key = k[:idx] + k[idx + len('._orig_mod') :]
                new_state_dict[new_key] = v
            checkpoint['state_dict'] = new_state_dict
        elif not inductor_enabled and inductor_checkpoint:
            # ckpt needs to be converted to non-inductor-format weights (remove .orig_mod)
            logging.info('Remove .orig_mod to all weight keys.')
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("._orig_mod", "")
                new_state_dict[new_key] = v
            checkpoint['state_dict'] = new_state_dict
        super().on_load_checkpoint(checkpoint)

    def on_fit_start(self) -> None:
        if self.online_encoding:
            # if encoding text online, set up text_encoder here (after loading checkpoints) instead of in __init__.
            # This is because text encoder weights are not saved, so the encoder must be loaded after other weights
            # are loaded.
            logging.info(
                f'Setting up pretrained text encoder: {self.text_encoder_path or "download or use cached t5-11b"}'
            )
            self.text_encoder = self.model.get_text_encoder(encoder_path=self.text_encoder_path).to(
                torch.cuda.current_device()
            )
            self.text_encoder.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad = False
