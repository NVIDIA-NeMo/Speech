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

import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


class LPIPSWithDiscriminator(nn.Module):
    """
    A perceptual loss module that combines LPIPS with an adversarial discriminator
    for improved reconstruction quality in variational autoencoders. This class
    calculates a combination of pixel-level, perceptual (LPIPS), KL, and adversarial
    losses for training a VAE model with a discriminator.
    """

    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
    ):
        """
        Initializes the LPIPSWithDiscriminator module.

        Args:
            disc_start (int): Iteration at which to start discriminator updates.
            logvar_init (float): Initial value for the log variance parameter.
            kl_weight (float): Weight for the KL divergence term.
            pixelloss_weight (float): Weight for the pixel-level reconstruction loss.
            disc_num_layers (int): Number of layers in the discriminator.
            disc_in_channels (int): Number of input channels for the discriminator.
            disc_factor (float): Scaling factor for the discriminator loss.
            disc_weight (float): Weight applied to the discriminator gradient balancing.
            perceptual_weight (float): Weight for the LPIPS perceptual loss.
            use_actnorm (bool): Whether to use actnorm in the discriminator.
            disc_conditional (bool): Whether the discriminator is conditional on an additional input.
            disc_loss (str): Type of GAN loss to use ("hinge" or "vanilla").
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(1) * logvar_init)

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Computes an adaptive weight that balances the reconstruction (NLL) and the
        adversarial (GAN) losses. This ensures stable training by adjusting the
        impact of the discriminator’s gradient on the generator.

        Args:
            nll_loss (torch.Tensor): The negative log-likelihood loss.
            g_loss (torch.Tensor): The generator (adversarial) loss.
            last_layer (torch.nn.Parameter, optional): Last layer parameters of the model
                for gradient-based calculations. If None, uses self.last_layer[0].

        Returns:
            torch.Tensor: The computed adaptive weight for balancing the discriminator.
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self, inputs, reconstructions, posteriors, optimizer_idx, global_step, last_layer=None, cond=None, weights=None
    ):
        """
        Forward pass for computing the combined loss. Depending on the optimizer index,
        this either computes the generator loss (including pixel, perceptual, KL, and
        adversarial terms) or the discriminator loss.

        Args:
            inputs (torch.Tensor): Original inputs to reconstruct.
            reconstructions (torch.Tensor): Reconstructed outputs from the model.
            posteriors (object): Posteriors from the VAE model for KL computation.
            optimizer_idx (int): Indicates which optimizer is being updated
                (0 for generator, 1 for discriminator).
            global_step (int): Current training iteration step.
            last_layer (torch.nn.Parameter, optional): The last layer's parameters for
                adaptive weight calculation.
            cond (torch.Tensor, optional): Conditional input for the discriminator.
            weights (torch.Tensor, optional): Sample-wise weighting for the losses.

        Returns:
            (torch.Tensor, dict): A tuple of (loss, log_dict) where loss is the computed loss
            for the current optimizer and log_dict is a dictionary of intermediate values
            for logging and debugging.
        """
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {
                "total_loss": loss.clone().detach().mean(),
                "logvar": self.logvar.detach().item(),
                "kl_loss": kl_loss.detach().mean(),
                "nll_loss": nll_loss.detach().mean(),
                "rec_loss": rec_loss.detach().mean(),
                "d_weight": d_weight.detach(),
                "disc_factor": torch.tensor(disc_factor),
                "g_loss": g_loss.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1:
            # discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "disc_loss": d_loss.clone().detach().mean(),
                "logits_real": logits_real.detach().mean(),
                "logits_fake": logits_fake.detach().mean(),
            }
            return d_loss, log
