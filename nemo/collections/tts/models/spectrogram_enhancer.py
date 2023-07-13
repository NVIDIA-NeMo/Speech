# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The following is largely based on code from https://github.com/lucidrains/stylegan2-pytorch

from random import random, randrange
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.tensorboard.writer import SummaryWriter

from nemo.collections.tts.losses.spectrogram_enhancer_losses import (
    ConsistencyLoss,
    GeneratorLoss,
    GradientPenaltyLoss,
    HingeLoss,
)
from nemo.collections.tts.parts.utils.helpers import mask_sequence_tensor, to_device_recursive
from nemo.core import Exportable, ModelPT, PretrainedModelInfo, typecheck
from nemo.core.neural_types import LengthsType, MelSpectrogramType, NeuralType
from nemo.core.neural_types.elements import BoolType
from nemo.utils import logging


class SpectrogramEnhancerModel(ModelPT, Exportable):
    """
    GAN-based model to add details to blurry spectrograms from TTS models like Tacotron or FastPitch. Based on StyleGAN 2 [1]
    [1] Karras et. al. - Analyzing and Improving the Image Quality of StyleGAN (https://arxiv.org/abs/1912.04958)
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        self.spectrogram_model = None
        super().__init__(cfg=cfg, trainer=trainer)

        self.generator = instantiate(cfg.generator)
        self.discriminator = instantiate(cfg.discriminator)

        self.generator_loss = GeneratorLoss()
        self.discriminator_loss = HingeLoss()
        self.consistency_loss = ConsistencyLoss(cfg.consistency_loss_weight)
        self.gradient_penalty_loss = GradientPenaltyLoss(cfg.gradient_penalty_loss_weight)

    def move_to_correct_device(self, e):
        return to_device_recursive(e, next(iter(self.generator.parameters())).device)

    def normalize_spectrograms(self, spectrogram: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        spectrogram = spectrogram - self._cfg.spectrogram_min_value
        spectrogram = spectrogram / (self._cfg.spectrogram_max_value - self._cfg.spectrogram_min_value)
        return mask_sequence_tensor(spectrogram, lengths)

    def unnormalize_spectrograms(self, spectrogram: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        spectrogram = spectrogram * (self._cfg.spectrogram_max_value - self._cfg.spectrogram_min_value)
        spectrogram = spectrogram + self._cfg.spectrogram_min_value
        return mask_sequence_tensor(spectrogram, lengths)

    def generate_zs(self, batch_size: int = 1, mixing: bool = False):
        if mixing and self._cfg.mixed_prob < random():
            mixing_point = randrange(1, self.generator.num_layers)
            first_part = [torch.randn(batch_size, self._cfg.latent_dim)] * mixing_point
            second_part = [torch.randn(batch_size, self._cfg.latent_dim)] * (self.generator.num_layers - mixing_point)
            zs = [*first_part, *second_part]
        else:
            zs = [torch.randn(batch_size, self._cfg.latent_dim)] * self.generator.num_layers

        return self.move_to_correct_device(zs)

    def generate_noise(self, batch_size: int = 1) -> torch.Tensor:
        noise = torch.rand(batch_size, self._cfg.n_bands, 4096, 1)
        return self.move_to_correct_device(noise)

    def pad_spectrograms(self, spectrograms):
        multiplier = self.generator.upsample_factor
        *_, max_length = spectrograms.shape
        return F.pad(spectrograms, (0, multiplier - max_length % multiplier))

    @typecheck(
        input_types={
            "input_spectrograms": NeuralType(("B", "D", "T_spec"), MelSpectrogramType()),
            "lengths": NeuralType(("B",), LengthsType()),
            "mixing": NeuralType(None, BoolType(), optional=True),
            "normalize": NeuralType(None, BoolType(), optional=True),
        }
    )
    def forward(
        self, *, input_spectrograms: torch.Tensor, lengths: torch.Tensor, mixing: bool = False, normalize: bool = True,
    ):
        """
        Generator forward pass. Noise inputs will be generated.

        input_spectrograms: batch of spectrograms, typically synthetic
        lengths: length for every spectrogam in the batch
        mixing: style mixing, usually True during training
        normalize: normalize spectrogram range to ~[0, 1], True for normal use

        returns: batch of enhanced spectrograms

        For explanation of style mixing refer to [1]
        [1] Karras et. al. - A Style-Based Generator Architecture for Generative Adversarial Networks, 2018 (https://arxiv.org/abs/1812.04948)
        """

        return self.forward_with_custom_noise(
            input_spectrograms=input_spectrograms,
            lengths=lengths,
            mixing=mixing,
            normalize=normalize,
            zs=None,
            ws=None,
            noise=None,
        )

    def forward_with_custom_noise(
        self,
        input_spectrograms: torch.Tensor,
        lengths: torch.Tensor,
        zs: Optional[List[torch.Tensor]] = None,
        ws: Optional[List[torch.Tensor]] = None,
        noise: Optional[torch.Tensor] = None,
        mixing: bool = False,
        normalize: bool = True,
    ):
        """
        Generator forward pass. Noise inputs will be generated if None.

        input_spectrograms: batch of spectrograms, typically synthetic
        lenghts: length for every spectrogam in the batch
        zs: latent noise inputs on the unit sphere (either this or ws or neither)
        ws: latent noise inputs in the style space (either this or zs or neither)
        noise: per-pixel indepentent gaussian noise
        mixing: style mixing, usually True during training
        normalize: normalize spectrogram range to ~[0, 1], True for normal use

        returns: batch of enhanced spectrograms

        For explanation of style mixing refer to [1]
        For definititions of z, w [2]
        [1] Karras et. al. - A Style-Based Generator Architecture for Generative Adversarial Networks, 2018 (https://arxiv.org/abs/1812.04948)
        [2] Karras et. al. - Analyzing and Improving the Image Quality of StyleGAN, 2019 (https://arxiv.org/abs/1912.04958)
        """
        batch_size, *_, max_length = input_spectrograms.shape

        # generate noise
        if zs is not None and ws is not None:
            raise ValueError(
                "Please specify either zs or ws or neither, but not both. It is not clear which one to use."
            )

        if zs is None:
            zs = self.generate_zs(batch_size, mixing)
        if ws is None:
            ws = [self.generator.style_mapping(z) for z in zs]
        if noise is None:
            noise = self.generate_noise(batch_size)

        input_spectrograms = rearrange(input_spectrograms, "b c l -> b 1 c l")
        # normalize if needed, mask and pad appropriately
        if normalize:
            input_spectrograms = self.normalize_spectrograms(input_spectrograms, lengths)
        input_spectrograms = self.pad_spectrograms(input_spectrograms)

        # the main call
        enhanced_spectrograms = self.generator(input_spectrograms, lengths, ws, noise)

        # denormalize if needed, mask and remove padding
        if normalize:
            enhanced_spectrograms = self.unnormalize_spectrograms(enhanced_spectrograms, lengths)
        enhanced_spectrograms = enhanced_spectrograms[:, :, :, :max_length]
        enhanced_spectrograms = rearrange(enhanced_spectrograms, "b 1 c l -> b c l")

        return enhanced_spectrograms

    def training_step(self, batch, batch_idx, optimizer_idx):
        input_spectrograms, target_spectrograms, lengths = batch

        with torch.no_grad():
            input_spectrograms = self.normalize_spectrograms(input_spectrograms, lengths)
            target_spectrograms = self.normalize_spectrograms(target_spectrograms, lengths)

        # train discriminator
        if optimizer_idx == 0:
            enhanced_spectrograms = self.forward(
                input_spectrograms=input_spectrograms, lengths=lengths, mixing=True, normalize=False
            )
            enhanced_spectrograms = rearrange(enhanced_spectrograms, "b c l -> b 1 c l")
            fake_logits = self.discriminator(enhanced_spectrograms, input_spectrograms, lengths)

            target_spectrograms_ = rearrange(target_spectrograms, "b c l -> b 1 c l").requires_grad_()
            real_logits = self.discriminator(target_spectrograms_, input_spectrograms, lengths)
            d_loss = self.discriminator_loss(real_logits, fake_logits)
            self.log("d_loss", d_loss, prog_bar=True)

            if batch_idx % self._cfg.gradient_penalty_loss_every_n_steps == 0:
                gp_loss = self.gradient_penalty_loss(target_spectrograms_, real_logits)
                self.log("d_loss_gp", gp_loss, prog_bar=True)
                return d_loss + gp_loss

            return d_loss

        # train generator
        if optimizer_idx == 1:
            enhanced_spectrograms = self.forward(
                input_spectrograms=input_spectrograms, lengths=lengths, mixing=True, normalize=False
            )

            input_spectrograms = rearrange(input_spectrograms, "b c l -> b 1 c l")
            enhanced_spectrograms = rearrange(enhanced_spectrograms, "b c l -> b 1 c l")

            fake_logits = self.discriminator(enhanced_spectrograms, input_spectrograms, lengths)
            g_loss = self.generator_loss(fake_logits)
            c_loss = self.consistency_loss(input_spectrograms, enhanced_spectrograms, lengths)

            self.log("g_loss", g_loss, prog_bar=True)
            self.log("c_loss", c_loss, prog_bar=True)

            with torch.no_grad():
                target_spectrograms = rearrange(target_spectrograms, "b c l -> b 1 c l")
                self.log_illustration(target_spectrograms, input_spectrograms, enhanced_spectrograms, lengths)
            return g_loss + c_loss

    def configure_optimizers(self):
        generator_opt = instantiate(self._cfg.generator_opt, params=self.generator.parameters(),)
        discriminator_opt = instantiate(self._cfg.discriminator_opt, params=self.discriminator.parameters())
        return [discriminator_opt, generator_opt], []

    def setup_training_data(self, train_data_config):
        dataset = instantiate(train_data_config.dataset)
        self._train_dl = torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, **train_data_config.dataloader_params
        )

    def setup_validation_data(self, val_data_config):
        """
        There is no validation step for this model.
        It is not clear whether any of used losses is a sensible metric for choosing between two models.
        This might change in the future.
        """
        pass

    @classmethod
    def list_available_models(cls):
        list_of_models = []

        # en, multi speaker, LibriTTS, 16000 Hz
        # stft 25ms 10ms matching ASR params
        # for use during Enhlish ASR training/adaptation
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_spectrogram_enhancer_for_asr_finetuning",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_fastpitch_spectrogram_enhancer_for_asr_finetuning/versions/1.20.0/files/tts_en_spectrogram_enhancer_for_asr_finetuning.nemo",
            description="This model is trained to add details to synthetic spectrograms."
            " It was trained on pairs of real-synthesized spectrograms generated by FastPitch."
            " STFT parameters follow ASR with 25 ms window and 10 ms hop."
            " It is supposed to be used in conjunction with that model for ASR training/adaptation.",
            class_=cls,
        )
        list_of_models.append(model)

        return list_of_models

    def log_illustration(self, target_spectrograms, input_spectrograms, enhanced_spectrograms, lengths):
        if self.global_rank != 0:
            return

        if not self.loggers:
            return

        step = self.trainer.global_step // 2  # because of G/D training
        if step % self.trainer.log_every_n_steps != 0:
            return

        idx = 0
        length = int(lengths.flatten()[idx].item())
        tensor = torch.stack(
            [
                enhanced_spectrograms - input_spectrograms,
                input_spectrograms,
                enhanced_spectrograms,
                target_spectrograms,
            ],
            dim=0,
        ).cpu()[:, idx, :, :, :length]

        grid = torchvision.utils.make_grid(tensor, nrow=1).clamp(0.0, 1.0)

        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                writer: SummaryWriter = logger.experiment
                writer.add_image("spectrograms", grid, global_step=step)
                writer.flush()
            elif isinstance(logger, WandbLogger):
                logger.log_image("spectrograms", [grid], caption=["residual, input, output, ground truth"], step=step)
            else:
                logging.warning("Unsupported logger type: %s", str(type(logger)))
