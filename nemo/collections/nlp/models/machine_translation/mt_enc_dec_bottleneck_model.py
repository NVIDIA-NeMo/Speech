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

from typing import Dict, Optional

import numpy as np
import torch
from lightning.pytorch import Trainer

from nemo.collections.common.losses import NLLLoss
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTBottleneckModelConfig
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.core.classes.common import typecheck
from nemo.utils import timers

__all__ = ['MTBottleneckModel']


def build_linear_or_identity(input_dim, output_dim):
    """
    Auxiliary method to return FC layer when input_dim != output_dim
    else return identity
    """
    if input_dim != output_dim:
        model = torch.nn.Linear(input_dim, output_dim)
    else:
        model = torch.nn.Identity()

    return model


class MTBottleneckModel(MTEncDecModel):
    """
    Machine translation model which supports bottleneck architecture,
    NLL, VAE, and MIM loss.

    Supported losses:
      1) nll - Conditional cross entropy (the usual NMT loss)
      2) mim - MIM learning framework. A latent variable model with good
                      reconstruction and compressed latent representation.
                      https://arxiv.org/pdf/2003.02645.pdf
      3) vae - VAE learning framework. A latent variable model which learns
                      good probability estimation over observations and
                      a regularized latent representation.
                      https://arxiv.org/pdf/1312.6114.pdf
    """

    def __init__(self, cfg: MTBottleneckModelConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        self.model_type: str = cfg.get("model_type", "nll")
        self.min_logv: float = cfg.get("min_logv", -6)
        self.latent_size: int = cfg.get("latent_size", -1)
        self.non_recon_warmup_batches: int = cfg.get("non_recon_warmup_batches", 200000)
        self.recon_per_token: bool = cfg.get("recon_per_token", True)
        self.log_timing: bool = cfg.get("log_timing", True)

        # if True, translation uses the mean of latent for VAE and MIM
        self.deterministic_translate = True

        # latent_size -1 will take value of encoder.hidden_size
        if self.latent_size < 0:
            self.latent_size = self.encoder.hidden_size

        if not self.recon_per_token:
            # disable reduction for train and eval loss
            self.eval_loss_fn = NLLLoss(ignore_index=self.decoder_tokenizer.pad_id, reduction='none')
            self.loss_fn._per_token_reduction = False

        if self.model_type not in ["nll", "mim", "vae"]:
            raise ValueError(f"Unknown model_type = {self.model_type}")

        # project bridge dimension back to decoder hidden dimensions
        self.latent2hidden = build_linear_or_identity(self.latent_size, self.decoder.hidden_size)

        if self.model_type == "nll":
            # project dimension of encoder hidden to latent dimension
            self.hidden2latent_mean = build_linear_or_identity(self.encoder.hidden_size, self.latent_size)
        else:
            # MIM or VAE requires two independent projections for mean/variance
            # project dimension of encoder hidden to latent dimension
            self.hidden2latent_mean = torch.nn.Linear(self.encoder.hidden_size, self.latent_size)
            # for probabilistic latent variable models we also need variance
            self.hidden2latent_logv = torch.nn.Linear(self.encoder.hidden_size, self.latent_size)

    def _validate_encoder_decoder_hidden_size(self):
        """
        Validate encoder and decoder hidden sizes, and enforce same size.
        We support here encoder/decoder with different hidden_size, so do nothing.
        """
        pass

    def eval_epoch_end(self, outputs, mode, global_rank):
        # call parent for logging
        super().eval_epoch_end(outputs, mode, global_rank)

        # if user specifies one validation dataloader, then PTL reverts to giving a list of dictionary instead of a list of list of dictionary
        if isinstance(outputs[0], dict):
            outputs = [outputs]

        for dataloader_idx, output in enumerate(outputs):
            # add logs if available in outputs
            log_dict = {}
            for x in output:
                if "log" in x:
                    for k, v in x["log"].items():
                        log_dict[k] = log_dict.get(k, []) + [v]

            for k, v in log_dict.items():
                if dataloader_idx == 0:
                    self.log(f"{mode}_{k}", np.mean(v), sync_dist=True)
                else:
                    self.log(f"{mode}_{k}_dl_index_{dataloader_idx}", np.mean(v), sync_dist=True)

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []

        return result

    def encode_latent(self, hidden):
        """
        Sample latent code z with reparameterization from bridge for
        probabilistic latent variable models (e.g., mim, vae),
        or return value for non-probabilistic models (nll)
        """
        # all models have mean
        z_mean = self.hidden2latent_mean(hidden)

        if self.model_type == "nll":
            # reconstruction only
            z = z_mean
            z_logv = torch.zeros_like(z)
        else:
            # mim or vae

            # sample posterior q(z|x) for MIM and VAE
            z_logv = self.hidden2latent_logv(hidden)
            # avoid numerical instability for MIM
            z_logv = z_logv.clamp_min(self.min_logv)
            # sample z with reparameterization
            e = torch.randn_like(z_mean)
            z = e * torch.exp(0.5 * z_logv) + z_mean

        return z, z_mean, z_logv

    def loss(
        self, z, z_mean, z_logv, z_mask, tgt_log_probs, tgt, tgt_mask, tgt_labels, train=False, return_info=False
    ):
        """
        Compute the loss from latent (z) and target (x).

        train - If True enables loss annealing, and label smoothing
        """

        recon_loss_fn = self.loss_fn if train else self.eval_loss_fn

        info_dict = {}

        if self.recon_per_token:
            log_p_x_given_z_per_token = -recon_loss_fn(log_probs=tgt_log_probs, labels=tgt_labels)

            log_p_x_given_z = log_p_x_given_z_per_token
            log_p_x_given_z_per_token = log_p_x_given_z_per_token.detach()
        else:
            # averaging of log_p_x_given_z per sample
            output_mask = (tgt_labels != self.decoder_tokenizer.pad_id).type_as(tgt_log_probs)

            log_p_x_given_z_per_token = (
                -recon_loss_fn(
                    log_probs=tgt_log_probs,
                    labels=tgt_labels,
                ).view(tgt_log_probs.shape[:2])
                * output_mask
            )

            # probability per sample
            log_p_x_given_z = log_p_x_given_z_per_token.sum(-1).mean()

            tokens = output_mask.sum()
            log_p_x_given_z_per_token = log_p_x_given_z_per_token.sum().detach() / tokens

            info_dict["log_p_x_given_z"] = log_p_x_given_z.detach().cpu()

        info_dict["log_p_x_given_z_per_token"] = log_p_x_given_z_per_token.detach().cpu()

        # loss warmup during training only
        if train:
            trainer = self.trainer
            # if we do not have a trainer ignore annealing
            if trainer is None:
                # ignore warmup and auxiliary loss
                warmup_coef = 1.0
            else:
                global_step = self.trainer.global_step

                warmup_coef = min(global_step / self.non_recon_warmup_batches, 1)
        else:
            # ignore warmup and auxiliary loss
            warmup_coef = 1.0

        info_dict["warmup_coef_recon"] = warmup_coef

        if self.model_type in ["mim", "vae"]:
            # tokens = tgt_mask.sum()
            q_z_given_x = torch.distributions.Normal(
                loc=z_mean,
                scale=torch.exp(0.5 * z_logv),
            )
            # average latent distribution to match averaging of observations
            if self.recon_per_token:
                # average latent per dimension - to heuristically match per-token reconstruction
                log_q_z_given_x = q_z_given_x.log_prob(z).mean(-1).mean(-1).mean()
            else:
                log_q_z_given_x = q_z_given_x.log_prob(z).sum(-1).sum(-1).mean()

            # build prior distribution
            p_z = torch.distributions.Normal(
                loc=torch.zeros_like(z),
                scale=torch.ones_like(z),
            )
            if self.recon_per_token:
                # average latent distribution similar to averaging of observations
                log_p_z = p_z.log_prob(z).mean(-1).mean(-1).mean()
            else:
                log_p_z = p_z.log_prob(z).sum(-1).sum(-1).mean()

            if self.model_type == "mim":
                loss_terms = 0.5 * (log_q_z_given_x + log_p_z)
            elif self.model_type == "vae":
                # KL divergence -Dkl( q(z|x) || p(z) )
                loss_terms = log_p_z - log_q_z_given_x

            # show loss value for reconstruction but train with MIM/VAE loss
            loss = -(log_p_x_given_z + warmup_coef * loss_terms)

            info_dict["log_q_z_given_x"] = log_q_z_given_x.detach().cpu()
            info_dict["log_var_q_z_given_x"] = z_logv.detach().mean().cpu()
            info_dict["log_p_z"] = log_p_z.detach().cpu()
            info_dict["kl_div_q_p"] = (log_q_z_given_x - log_p_z).detach().cpu()

        elif self.model_type == "nll":
            loss = -log_p_x_given_z

        if return_info:
            return loss, info_dict
        else:
            return loss

    @typecheck()
    def forward(self, src, src_mask, tgt, tgt_mask, timer=None):
        """
        return_info - if True, returns loss, info_dict with additional information
                      regarding the loss that can be logged
        """
        if self.validate_input_ids:
            # test src/tgt for id range (i.e., hellp in catching wrong tokenizer)
            self.test_encoder_ids(src, raise_error=True)
            self.test_decoder_ids(tgt, raise_error=True)

        if timer is not None:
            timer.start("encoder")
        enc_hiddens, enc_mask = self.encoder(
            input_ids=src,
            encoder_mask=src_mask,
            return_mask=True,
        )

        # build posterior distribution q(x|z)
        z, z_mean, z_logv = self.encode_latent(hidden=enc_hiddens)
        z_mask = enc_mask

        if timer is not None:
            timer.stop("encoder")

        if timer is not None:
            timer.start("decoder")

        # decoding cross attention context
        context_hiddens = self.latent2hidden(z)

        tgt_hiddens = self.decoder(
            input_ids=tgt,
            decoder_mask=tgt_mask,
            encoder_embeddings=context_hiddens,
            encoder_mask=enc_mask,
        )

        # build decoding distribution
        tgt_log_probs = self.log_softmax(hidden_states=tgt_hiddens)

        if timer is not None:
            timer.stop("decoder")

        return z, z_mean, z_logv, z_mask, tgt_log_probs

    @torch.no_grad()
    def batch_translate(
        self, src: torch.LongTensor, src_mask: torch.LongTensor, return_beam_scores: bool = False, cache={}
    ):
        """
        Translates a minibatch of inputs from source language to target language.
        Args:
            src: minibatch of inputs in the src language (batch x seq_len)
            src_mask: mask tensor indicating elements to be ignored (batch x seq_len)
        Returns:
            translations: a list strings containing detokenized translations
            inputs: a list of string containing detokenized inputs
        """
        mode = self.training
        timer = cache.get("timer", None)
        try:
            self.eval()

            # build posterior distribution q(x|z)
            if ("z" not in cache) or ("z_mean" not in cache) or ("z_mask" not in cache):
                if timer is not None:
                    timer.start("encoder")
                enc_hiddens, enc_mask = self.encoder(input_ids=src, encoder_mask=src_mask, return_mask=True)
                z, z_mean, _ = self.encode_latent(hidden=enc_hiddens)
                if timer is not None:
                    timer.stop("encoder")
            else:
                enc_mask = cache["z_mask"]
                z = cache["z"]
                z_mean = cache["z_mean"]

            if getattr(self, "deterministic_translate", True):
                z = z_mean

            if timer is not None:
                timer.start("sampler")
            # decoding cross attention context
            context_hiddens = self.latent2hidden(z)

            best_translations = self.beam_search(
                encoder_hidden_states=context_hiddens,
                encoder_input_mask=enc_mask,
                return_beam_scores=return_beam_scores,
            )
            if timer is not None:
                timer.stop("sampler")

            if return_beam_scores:
                all_translations, scores, best_translations = best_translations
                scores = scores.view(-1)
                all_translations = self.ids_to_postprocessed_text(
                    all_translations, self.decoder_tokenizer, self.target_processor, filter_beam_ids=True
                )

            best_translations = self.ids_to_postprocessed_text(
                best_translations, self.decoder_tokenizer, self.target_processor, filter_beam_ids=True
            )
            inputs = self.ids_to_postprocessed_text(
                src, self.encoder_tokenizer, self.source_processor, filter_beam_ids=False
            )

        finally:
            self.train(mode=mode)
        if return_beam_scores:
            return inputs, all_translations, scores.data.cpu().numpy().tolist(), best_translations

        return inputs, best_translations

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        z, z_mean, z_logv, z_mask, tgt_log_probs = self(src_ids, src_mask, tgt_ids, tgt_mask)
        train_loss, info_dict = self.loss(
            z=z,
            z_mean=z_mean,
            z_logv=z_logv,
            z_mask=z_mask,
            tgt_log_probs=tgt_log_probs,
            tgt=tgt_ids,
            tgt_mask=tgt_mask,
            tgt_labels=labels,
            train=True,
            return_info=True,
        )
        tensorboard_logs = {
            'train_loss': train_loss,
            'lr': self._optimizer.param_groups[0]['lr'],
        }
        tensorboard_logs.update(info_dict)

        return {'loss': train_loss, 'log': tensorboard_logs}

    def eval_step(self, batch, batch_idx, mode, dataloader_idx=0):
        if self.log_timing:
            timer = timers.NamedTimer()
        else:
            timer = None

        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)

        if self.multilingual:
            self.source_processor = self.source_processor_list[dataloader_idx]
            self.target_processor = self.target_processor_list[dataloader_idx]

        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        z, z_mean, z_logv, z_mask, tgt_log_probs = self(src_ids, src_mask, tgt_ids, tgt_mask, timer=timer)
        eval_loss, info_dict = self.loss(
            z=z,
            z_mean=z_mean,
            z_logv=z_logv,
            z_mask=z_mask,
            tgt_log_probs=tgt_log_probs,
            tgt=tgt_ids,
            tgt_mask=tgt_mask,
            tgt_labels=labels,
            train=False,
            return_info=True,
        )
        # pass cache to sampler in order to reuse encoder's output
        cache = dict(
            z=z,
            z_mean=z_mean,
            z_mask=z_mask,
            timer=timer,
        )

        inputs, translations = self.batch_translate(src=src_ids, src_mask=src_mask, cache=cache)

        num_measurements = labels.shape[0] * labels.shape[1]
        if dataloader_idx == 0:
            getattr(self, f'{mode}_loss')(
                loss=eval_loss,
                num_measurements=num_measurements,
            )
        else:
            getattr(self, f'{mode}_loss_{dataloader_idx}')(
                loss=eval_loss,
                num_measurements=num_measurements,
            )
        np_tgt = tgt_ids.detach().cpu().numpy()
        ground_truths = [self.decoder_tokenizer.ids_to_text(tgt) for tgt in np_tgt]
        ground_truths = [self.target_processor.detokenize(tgt.split(' ')) for tgt in ground_truths]
        num_non_pad_tokens = np.not_equal(np_tgt, self.decoder_tokenizer.pad_id).sum().item()

        # collect logs
        log_dict = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in info_dict.items()}
        # add timing if required
        if timer is not None:
            for k, v in timer.export().items():
                log_dict[f"{k}_timing"] = v

        return {
            'inputs': inputs,
            'translations': translations,
            'ground_truths': ground_truths,
            'num_non_pad_tokens': num_non_pad_tokens,
            'log': log_dict,
        }
