# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import List

import numpy as np
import omegaconf
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch import nn

from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.models.base import NeedsNormalizer
from nemo.collections.tts.parts.utils.helpers import (
    binarize_attention,
    g2p_backward_compatible_support,
    get_mask_from_lengths,
    plot_alignment_to_numpy,
)
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


class AlignerModel(NeedsNormalizer, ModelPT):
    """Speech-to-text alignment model (https://arxiv.org/pdf/2108.10447.pdf) that is used to learn alignments between mel spectrogram and text."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Setup normalizer
        self.normalizer = None
        self.text_normalizer_call = None
        self.text_normalizer_call_kwargs = {}
        self._setup_normalizer(cfg)

        # Setup tokenizer
        self.tokenizer = None
        self._setup_tokenizer(cfg)
        assert self.tokenizer is not None

        num_tokens = len(self.tokenizer.tokens)
        self.tokenizer_pad = self.tokenizer.pad
        self.tokenizer_unk = self.tokenizer.oov

        super().__init__(cfg=cfg, trainer=trainer)

        self.embed = nn.Embedding(num_tokens, cfg.symbols_embedding_dim)
        self.preprocessor = instantiate(cfg.preprocessor)
        self.alignment_encoder = instantiate(cfg.alignment_encoder)

        self.forward_sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.add_bin_loss = False
        self.bin_loss_scale = 0.0
        self.bin_loss_start_ratio = cfg.bin_loss_start_ratio
        self.bin_loss_warmup_epochs = cfg.bin_loss_warmup_epochs

    def _setup_tokenizer(self, cfg):
        text_tokenizer_kwargs = {}
        if "g2p" in cfg.text_tokenizer:
            # for backward compatibility
            if (
                self._is_model_being_restored()
                and (cfg.text_tokenizer.g2p.get('_target_', None) is not None)
                and cfg.text_tokenizer.g2p["_target_"].startswith("nemo_text_processing.g2p")
            ):
                cfg.text_tokenizer.g2p["_target_"] = g2p_backward_compatible_support(
                    cfg.text_tokenizer.g2p["_target_"]
                )

            g2p_kwargs = {}

            if "phoneme_dict" in cfg.text_tokenizer.g2p:
                g2p_kwargs["phoneme_dict"] = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict',
                    cfg.text_tokenizer.g2p.phoneme_dict,
                )

            if "heteronyms" in cfg.text_tokenizer.g2p:
                g2p_kwargs["heteronyms"] = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms',
                    cfg.text_tokenizer.g2p.heteronyms,
                )

            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p, **g2p_kwargs)

        self.tokenizer = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)

    def forward(self, *, spec, spec_len, text, text_len, attn_prior=None):
        with torch.amp.autocast(self.device.type, enabled=False):
            attn_soft, attn_logprob = self.alignment_encoder(
                queries=spec,
                keys=self.embed(text).transpose(1, 2),
                mask=get_mask_from_lengths(text_len).unsqueeze(-1) == 0,
                attn_prior=attn_prior,
            )

        return attn_soft, attn_logprob

    def _metrics(self, attn_soft, attn_logprob, spec_len, text_len):
        loss, bin_loss, attn_hard = 0.0, None, None

        forward_sum_loss = self.forward_sum_loss(attn_logprob=attn_logprob, in_lens=text_len, out_lens=spec_len)
        loss += forward_sum_loss

        if self.add_bin_loss:
            attn_hard = binarize_attention(attn_soft, text_len, spec_len)
            bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft)
            loss += bin_loss

        return loss, forward_sum_loss, bin_loss, attn_hard

    def on_train_epoch_start(self):
        bin_loss_start_epoch = np.ceil(self.bin_loss_start_ratio * self._trainer.max_epochs)

        # Add bin loss when current_epoch >= bin_start_epoch
        if not self.add_bin_loss and self.current_epoch >= bin_loss_start_epoch:
            logging.info(f"Using hard attentions after epoch: {self.current_epoch}")
            self.add_bin_loss = True

        if self.add_bin_loss:
            self.bin_loss_scale = min((self.current_epoch - bin_loss_start_epoch) / self.bin_loss_warmup_epochs, 1.0)

    def training_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, attn_prior = batch
        spec, spec_len = self.preprocessor(input_signal=audio, length=audio_len)
        attn_soft, attn_logprob = self(
            spec=spec, spec_len=spec_len, text=text, text_len=text_len, attn_prior=attn_prior
        )

        loss, forward_sum_loss, bin_loss, _ = self._metrics(attn_soft, attn_logprob, spec_len, text_len)

        train_log = {
            'train_forward_sum_loss': forward_sum_loss,
            'train_bin_loss': torch.tensor(1.0).to(forward_sum_loss.device) if bin_loss is None else bin_loss,
        }
        return {'loss': loss, 'progress_bar': {k: v.detach() for k, v in train_log.items()}, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, attn_prior = batch
        spec, spec_len = self.preprocessor(input_signal=audio, length=audio_len)
        attn_soft, attn_logprob = self(
            spec=spec, spec_len=spec_len, text=text, text_len=text_len, attn_prior=attn_prior
        )

        loss, forward_sum_loss, bin_loss, attn_hard = self._metrics(attn_soft, attn_logprob, spec_len, text_len)

        # plot once per epoch
        if batch_idx == 0 and isinstance(self.logger, WandbLogger) and HAVE_WANDB:
            if attn_hard is None:
                attn_hard = binarize_attention(attn_soft, text_len, spec_len)

            attn_matrices = []
            for i in range(min(5, audio.shape[0])):
                attn_matrices.append(
                    wandb.Image(
                        plot_alignment_to_numpy(
                            np.fliplr(np.rot90(attn_soft[i, 0, : spec_len[i], : text_len[i]].data.cpu().numpy()))
                        ),
                        caption=f"attn soft",
                    ),
                )

                attn_matrices.append(
                    wandb.Image(
                        plot_alignment_to_numpy(
                            np.fliplr(np.rot90(attn_hard[i, 0, : spec_len[i], : text_len[i]].data.cpu().numpy()))
                        ),
                        caption=f"attn hard",
                    )
                )

            self.logger.experiment.log({"attn_matrices": attn_matrices})

        val_log = {
            'val_loss': loss,
            'val_forward_sum_loss': forward_sum_loss,
            'val_bin_loss': torch.tensor(1.0).to(forward_sum_loss.device) if bin_loss is None else bin_loss,
        }
        self.log_dict(val_log, prog_bar=False, on_epoch=True, logger=True, sync_dist=True)

    def _loader(self, cfg):
        try:
            _ = cfg.dataset.manifest_filepath
        except omegaconf.errors.MissingMandatoryValue:
            logging.warning("manifest_filepath was skipped. No dataset for this model.")
            return None

        dataset = instantiate(
            cfg.dataset,
            text_normalizer=self.normalizer,
            text_normalizer_call_kwargs=self.text_normalizer_call_kwargs,
            text_tokenizer=self.tokenizer,
        )
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        return list_of_models
