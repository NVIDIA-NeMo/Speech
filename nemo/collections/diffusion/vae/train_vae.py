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


from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import nemo_run as run
import torch
import torch.distributed
import torch.utils.checkpoint
import torchvision
import wandb
from autovae import VAEGenerator
from contperceptual_loss import LPIPSWithDiscriminator
from diffusers import AutoencoderKL
from megatron.core import parallel_state
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.energon import DefaultTaskEncoder, ImageSample
from torch import Tensor, nn

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.data.diffusion_energon_datamodule import DiffusionDataModule
from nemo.collections.diffusion.train import pretrain
from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.lightning.io.mixin import IOMixin
from nemo.lightning.megatron_parallel import DataT, MegatronLossReduction, ReductionT
from nemo.lightning.pytorch.optim import OptimizerModule


class AvgLossReduction(MegatronLossReduction):
    """Performs average loss reduction across micro-batches."""

    def forward(self, batch: DataT, forward_out: Tensor) -> Tuple[Tensor, ReductionT]:
        """
        Forward pass for loss reduction.

        Args:
            batch: The batch of data.
            forward_out: The output tensor from forward computation.

        Returns:
            A tuple of (loss, reduction dictionary).
        """
        loss = forward_out.mean()
        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[ReductionT]) -> Tensor:
        """
        Reduce losses across multiple micro-batches by averaging them.

        Args:
            losses_reduced_per_micro_batch: A sequence of loss dictionaries.

        Returns:
            The averaged loss tensor.
        """
        losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return losses.mean()


class VAE(MegatronModule):
    """Variational Autoencoder (VAE) module."""

    def __init__(self, config, pretrained_model_name_or_path, search_vae=False):
        """
        Initialize the VAE model.

        Args:
            config: Transformer configuration.
            pretrained_model_name_or_path: Path or name of the pretrained model.
            search_vae: Flag to indicate whether to search for a target VAE using AutoVAE.
        """
        super().__init__(config)
        if search_vae:
            # Get VAE automatically from AutoVAE
            self.vae = VAEGenerator(input_resolution=1024, compression_ratio=16)
            # Below line is commented out due to an undefined 'generator' variable in original code snippet.
            # self.vae = generator.search_for_target_vae(parameters_budget=895.178707, cuda_max_mem=0)
        else:
            self.vae = AutoencoderKL.from_config(pretrained_model_name_or_path, weight_dtype=torch.bfloat16)

        sdxl_vae = AutoencoderKL.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0', subfolder="vae", weight_dtype=torch.bfloat16
        )
        sd_dict = sdxl_vae.state_dict()
        vae_dict = self.vae.state_dict()
        pre_dict = {k: v for k, v in sd_dict.items() if (k in vae_dict) and (vae_dict[k].numel() == v.numel())}
        self.vae.load_state_dict(pre_dict, strict=False)
        del sdxl_vae

        self.vae_loss = LPIPSWithDiscriminator(
            disc_start=50001,
            logvar_init=0.0,
            kl_weight=0.000001,
            pixelloss_weight=1.0,
            disc_num_layers=3,
            disc_in_channels=3,
            disc_factor=1.0,
            disc_weight=0.5,
            perceptual_weight=1.0,
            use_actnorm=False,
            disc_conditional=False,
            disc_loss="hinge",
        )

    def forward(self, target, global_step):
        """
        Forward pass through the VAE.

        Args:
            target: Target images.
            global_step: Current global step.

        Returns:
            A tuple (aeloss, log_dict_ae, pred) containing the loss, log dictionary, and predictions.
        """
        posterior = self.vae.encode(target).latent_dist
        z = posterior.sample()
        pred = self.vae.decode(z).sample
        aeloss, log_dict_ae = self.vae_loss(
            inputs=target,
            reconstructions=pred,
            posteriors=posterior,
            optimizer_idx=0,
            global_step=global_step,
            last_layer=self.vae.decoder.conv_out.weight,
        )
        return aeloss, log_dict_ae, pred

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """
        Set input tensor.

        Args:
            input_tensor: The input tensor to the model.
        """
        pass


class VAEModel(GPTModel):
    """A GPTModel wrapper for the VAE."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        optim: Optional[OptimizerModule] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        """
        Initialize the VAEModel.

        Args:
            pretrained_model_name_or_path: Path or name of the pretrained model.
            optim: Optional optimizer module.
            model_transform: Optional function to transform the model.
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        config = TransformerConfig(num_layers=1, hidden_size=1, num_attention_heads=1)
        self.model_type = ModelType.encoder_or_decoder
        super().__init__(config, optim=optim, model_transform=model_transform)

    def configure_model(self) -> None:
        """Configure the model by initializing the module."""
        if not hasattr(self, "module"):
            self.module = VAE(self.config, self.pretrained_model_name_or_path)

    def data_step(self, dataloader_iter) -> Dict[str, Any]:
        """
        Perform a single data step to fetch a batch from the iterator.

        Args:
            dataloader_iter: The dataloader iterator.

        Returns:
            A dictionary with 'pixel_values' ready for the model.
        """
        batch = next(dataloader_iter)[0]
        return {'pixel_values': batch.image.to(device='cuda', dtype=torch.bfloat16, non_blocking=True)}

    def forward(self, *args, **kwargs):
        """
        Forward pass through the underlying module.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of forward pass of self.module.
        """
        return self.module(*args, **kwargs)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch: The input batch.
            batch_idx: Batch index.

        Returns:
            The loss tensor.
        """
        loss, log_dict_ae, pred = self(batch["pixel_values"], self.global_step)

        if torch.distributed.get_rank() == 0:
            self.log_dict(log_dict_ae)

        return loss

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        """
        Perform a single validation step.

        Args:
            batch: The input batch.
            batch_idx: Batch index.

        Returns:
            The loss tensor.
        """
        loss, log_dict_ae, pred = self(batch["pixel_values"], self.global_step)

        image = torch.cat([batch["pixel_values"].cpu(), pred.cpu()], axis=0)
        image = (image + 0.5).clamp(0, 1)

        # wandb is on the last rank for megatron, first rank for nemo
        wandb_rank = 0

        if parallel_state.get_data_parallel_src_rank() == wandb_rank:
            if torch.distributed.get_rank() == wandb_rank:
                gather_list = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            else:
                gather_list = None
            torch.distributed.gather_object(
                image, gather_list, wandb_rank, group=parallel_state.get_data_parallel_group()
            )
            if gather_list is not None:
                self.log_dict(log_dict_ae)
                wandb.log(
                    {
                        "Original (left), Reconstruction (right)": [
                            wandb.Image(torchvision.utils.make_grid(image)) for _, image in enumerate(gather_list)
                        ]
                    },
                )

        return loss

    @property
    def training_loss_reduction(self) -> AvgLossReduction:
        """Returns the loss reduction method for training."""
        if not self._training_loss_reduction:
            self._training_loss_reduction = AvgLossReduction()
        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> AvgLossReduction:
        """Returns the loss reduction method for validation."""
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = AvgLossReduction()
        return self._validation_loss_reduction

    def on_validation_model_zero_grad(self) -> None:
        """
        Hook to handle zero grad on validation model step.
        Used here to skip first validation on resume.
        """
        super().on_validation_model_zero_grad()
        if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
            self.trainer.sanity_checking = True
            self._restarting_skip_val_flag = False


def crop_image(img, divisor=16):
    """
    Crop the image so that both dimensions are divisible by the given divisor.

    Args:
        img: Image tensor.
        divisor: The divisor to use for cropping.

    Returns:
        The cropped image tensor.
    """
    h, w = img.shape[-2], img.shape[-1]

    delta_h = h % divisor
    delta_w = w % divisor

    delta_h_top = delta_h // 2
    delta_h_bottom = delta_h - delta_h_top

    delta_w_left = delta_w // 2
    delta_w_right = delta_w - delta_w_left

    img_cropped = img[..., delta_h_top : h - delta_h_bottom, delta_w_left : w - delta_w_right]

    return img_cropped


class ImageTaskEncoder(DefaultTaskEncoder, IOMixin):
    """Image task encoder that crops and normalizes the image."""

    def encode_sample(self, sample: ImageSample) -> ImageSample:
        """
        Encode a single image sample by cropping and shifting its values.

        Args:
            sample: An image sample.

        Returns:
            The transformed image sample.
        """
        sample = super().encode_sample(sample)
        sample.image = crop_image(sample.image, 16)
        sample.image -= 0.5
        return sample


@run.cli.factory(target=llm.train)
def train_vae() -> run.Partial:
    """
    Training factory function for VAE.

    Returns:
        A run.Partial recipe for training.
    """
    recipe = pretrain()
    recipe.model = run.Config(
        VAEModel,
        pretrained_model_name_or_path='nemo/collections/diffusion/vae/vae16x/config.json',
    )
    recipe.data = run.Config(
        DiffusionDataModule,
        task_encoder=run.Config(ImageTaskEncoder),
        global_batch_size=24,
        num_workers=10,
    )
    recipe.optim.lr_scheduler = run.Config(nl.lr_scheduler.WarmupHoldPolicyScheduler, warmup_steps=100, hold_steps=1e9)
    recipe.optim.config.lr = 5e-6
    recipe.optim.config.weight_decay = 1e-2
    recipe.log.log_dir = 'nemo_experiments/train_vae'
    recipe.trainer.val_check_interval = 1000
    recipe.trainer.callbacks[0].every_n_train_steps = 1000

    return recipe


if __name__ == "__main__":
    run.cli.main(llm.train, default_factory=train_vae)
