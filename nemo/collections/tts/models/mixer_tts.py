# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import contextlib
from typing import List, Optional

import numpy as np
import omegaconf
import torch
import transformers
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F
from transformers import AlbertTokenizer

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import (
    EnglishCharsTokenizer,
    EnglishPhonemesTokenizer,
)
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.fastpitch import average_features, regulate_len
from nemo.collections.tts.parts.utils.helpers import (
    binarize_attention_parallel,
    get_mask_from_lengths,
    plot_pitch_to_numpy,
    plot_spectrogram_to_numpy,
)
from nemo.core import Exportable
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    LengthsType,
    LogprobsType,
    MelSpectrogramType,
    ProbsType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging, model_utils


class MixerTTSModel(SpectrogramGenerator, Exportable):
    """Mixer-TTS and Mixer-TTS-X models (https://arxiv.org/abs/2110.03584) that is used to generate mel spectrogram from text."""

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

        self.pitch_loss_scale = cfg.pitch_loss_scale
        self.durs_loss_scale = cfg.durs_loss_scale
        self.mel_loss_scale = cfg.mel_loss_scale

        self.aligner = instantiate(cfg.alignment_module)
        self.forward_sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.add_bin_loss = False
        self.bin_loss_scale = 0.0
        self.bin_loss_start_ratio = cfg.bin_loss_start_ratio
        self.bin_loss_warmup_epochs = cfg.bin_loss_warmup_epochs

        self.cond_on_lm_embeddings = cfg.get("cond_on_lm_embeddings", False)

        if self.cond_on_lm_embeddings:
            self.lm_padding_value = (
                self._train_dl.dataset.lm_padding_value
                if self._train_dl is not None
                else self._get_lm_padding_value(cfg.lm_model)
            )
            self.lm_embeddings = self._get_lm_embeddings(cfg.lm_model)
            self.lm_embeddings.weight.requires_grad = False

            self.self_attention_module = instantiate(
                cfg.self_attention_module, n_lm_tokens_channels=self.lm_embeddings.weight.shape[1]
            )

        self.encoder = instantiate(cfg.encoder, num_tokens=num_tokens, padding_idx=self.tokenizer_pad)
        self.symbol_emb = self.encoder.to_embed

        self.duration_predictor = instantiate(cfg.duration_predictor)

        self.pitch_mean, self.pitch_std = float(cfg.pitch_mean), float(cfg.pitch_std)
        self.pitch_predictor = instantiate(cfg.pitch_predictor)
        self.pitch_emb = instantiate(cfg.pitch_emb)

        self.preprocessor = instantiate(cfg.preprocessor)

        self.decoder = instantiate(cfg.decoder)
        self.proj = nn.Linear(self.decoder.d_model, cfg.n_mel_channels)

    def _setup_normalizer(self, cfg):
        if "text_normalizer" in cfg:
            normalizer_kwargs = {}

            if "whitelist" in cfg.text_normalizer:
                normalizer_kwargs["whitelist"] = self.register_artifact(
                    'text_normalizer.whitelist', cfg.text_normalizer.whitelist
                )

            try:
                import nemo_text_processing

                self.normalizer = instantiate(cfg.text_normalizer, **normalizer_kwargs)
            except Exception as e:
                logging.error(e)
                raise ImportError(
                    "`nemo_text_processing` not installed, see https://github.com/NVIDIA/NeMo-text-processing for more details"
                )

            self.text_normalizer_call = self.normalizer.normalize
            if "text_normalizer_call_kwargs" in cfg:
                self.text_normalizer_call_kwargs = cfg.text_normalizer_call_kwargs

    def _setup_tokenizer(self, cfg):
        text_tokenizer_kwargs = {}
        if "g2p" in cfg.text_tokenizer:
            # for backward compatibility
            if self._is_model_being_restored() and cfg.text_tokenizer.g2p.get('_target_', None):
                cfg.text_tokenizer.g2p['_target_'] = cfg.text_tokenizer.g2p['_target_'].replace(
                    "nemo_text_processing.g2p", "nemo.collections.tts.g2p"
                )
                logging.warning("This checkpoint support will be dropped after r1.18.0.")

            g2p_kwargs = {}

            if "phoneme_dict" in cfg.text_tokenizer.g2p:
                g2p_kwargs["phoneme_dict"] = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict', cfg.text_tokenizer.g2p.phoneme_dict,
                )

            if "heteronyms" in cfg.text_tokenizer.g2p:
                g2p_kwargs["heteronyms"] = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms', cfg.text_tokenizer.g2p.heteronyms,
                )

            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p, **g2p_kwargs)

        self.tokenizer = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)

    def _get_lm_model_tokenizer(self, lm_model="albert"):
        if getattr(self, "_lm_model_tokenizer", None) is not None:
            return self._lm_model_tokenizer

        if self._train_dl is not None and self._train_dl.dataset is not None:
            self._lm_model_tokenizer = self._train_dl.dataset.lm_model_tokenizer

        if lm_model == "albert":
            self._lm_model_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        else:
            raise NotImplementedError(
                f"{lm_model} lm model is not supported. Only albert is supported at this moment."
            )

        return self._lm_model_tokenizer

    def _get_lm_embeddings(self, lm_model="albert"):
        if lm_model == "albert":
            return transformers.AlbertModel.from_pretrained('albert-base-v2').embeddings.word_embeddings
        else:
            raise NotImplementedError(
                f"{lm_model} lm model is not supported. Only albert is supported at this moment."
            )

    def _get_lm_padding_value(self, lm_model="albert"):
        if lm_model == "albert":
            return transformers.AlbertTokenizer.from_pretrained('albert-base-v2')._convert_token_to_id('<pad>')
        else:
            raise NotImplementedError(
                f"{lm_model} lm model is not supported. Only albert is supported at this moment."
            )

    def _metrics(
        self,
        true_durs,
        true_text_len,
        pred_durs,
        true_pitch,
        pred_pitch,
        true_spect=None,
        pred_spect=None,
        true_spect_len=None,
        attn_logprob=None,
        attn_soft=None,
        attn_hard=None,
        attn_hard_dur=None,
    ):
        text_mask = get_mask_from_lengths(true_text_len)
        mel_mask = get_mask_from_lengths(true_spect_len)
        loss = 0.0

        # Dur loss and metrics
        durs_loss = F.mse_loss(pred_durs, (true_durs + 1).float().log(), reduction='none')
        durs_loss = durs_loss * text_mask.float()
        durs_loss = durs_loss.sum() / text_mask.sum()

        durs_pred = pred_durs.exp() - 1
        durs_pred = torch.clamp_min(durs_pred, min=0)
        durs_pred = durs_pred.round().long()

        acc = ((true_durs == durs_pred) * text_mask).sum().float() / text_mask.sum() * 100
        acc_dist_1 = (((true_durs - durs_pred).abs() <= 1) * text_mask).sum().float() / text_mask.sum() * 100
        acc_dist_3 = (((true_durs - durs_pred).abs() <= 3) * text_mask).sum().float() / text_mask.sum() * 100

        pred_spect = pred_spect.transpose(1, 2)

        # Mel loss
        mel_loss = F.mse_loss(pred_spect, true_spect, reduction='none').mean(dim=-2)
        mel_loss = mel_loss * mel_mask.float()
        mel_loss = mel_loss.sum() / mel_mask.sum()

        loss = loss + self.durs_loss_scale * durs_loss + self.mel_loss_scale * mel_loss

        # Aligner loss
        bin_loss, ctc_loss = None, None
        ctc_loss = self.forward_sum_loss(attn_logprob=attn_logprob, in_lens=true_text_len, out_lens=true_spect_len)
        loss = loss + ctc_loss
        if self.add_bin_loss:
            bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft)
            loss = loss + self.bin_loss_scale * bin_loss
        true_avg_pitch = average_features(true_pitch.unsqueeze(1), attn_hard_dur).squeeze(1)

        # Pitch loss
        pitch_loss = F.mse_loss(pred_pitch, true_avg_pitch, reduction='none')  # noqa
        pitch_loss = (pitch_loss * text_mask).sum() / text_mask.sum()

        loss = loss + self.pitch_loss_scale * pitch_loss

        return loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss, mel_loss, ctc_loss, bin_loss

    @torch.jit.unused
    def run_aligner(self, text, text_len, text_mask, spect, spect_len, attn_prior):
        text_emb = self.symbol_emb(text)
        attn_soft, attn_logprob = self.aligner(
            spect, text_emb.permute(0, 2, 1), mask=text_mask == 0, attn_prior=attn_prior,
        )
        attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        assert torch.all(torch.eq(attn_hard_dur.sum(dim=1), spect_len))
        return attn_soft, attn_logprob, attn_hard, attn_hard_dur

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_len": NeuralType(('B',), LengthsType()),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType(), optional=True),
            "spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType(), optional=True),
            "spect_len": NeuralType(('B',), LengthsType(), optional=True),
            "attn_prior": NeuralType(('B', 'T_spec', 'T_text'), ProbsType(), optional=True),
            "lm_tokens": NeuralType(('B', 'T_lm_tokens'), TokenIndex(), optional=True),
        },
        output_types={
            "pred_spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "durs_predicted": NeuralType(('B', 'T_text'), TokenDurationType()),
            "log_durs_predicted": NeuralType(('B', 'T_text'), TokenLogDurationType()),
            "pitch_predicted": NeuralType(('B', 'T_text'), RegressionValuesType()),
            "attn_soft": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
            "attn_logprob": NeuralType(('B', 'S', 'T_spec', 'T_text'), LogprobsType()),
            "attn_hard": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
            "attn_hard_dur": NeuralType(('B', 'T_text'), TokenDurationType()),
        },
    )
    def forward(self, text, text_len, pitch=None, spect=None, spect_len=None, attn_prior=None, lm_tokens=None):
        if self.training:
            assert pitch is not None

        text_mask = get_mask_from_lengths(text_len).unsqueeze(2)

        enc_out, enc_mask = self.encoder(text, text_mask)

        # Aligner
        attn_soft, attn_logprob, attn_hard, attn_hard_dur = None, None, None, None
        if spect is not None:
            attn_soft, attn_logprob, attn_hard, attn_hard_dur = self.run_aligner(
                text, text_len, text_mask, spect, spect_len, attn_prior
            )

        if self.cond_on_lm_embeddings:
            lm_emb = self.lm_embeddings(lm_tokens)
            lm_features = self.self_attention_module(
                enc_out, lm_emb, lm_emb, q_mask=enc_mask.squeeze(2), kv_mask=lm_tokens != self.lm_padding_value
            )

        # Duration predictor
        log_durs_predicted = self.duration_predictor(enc_out, enc_mask)
        durs_predicted = torch.clamp(log_durs_predicted.exp() - 1, 0)

        # Pitch predictor
        pitch_predicted = self.pitch_predictor(enc_out, enc_mask)

        # Avg pitch, add pitch_emb
        if not self.training:
            if pitch is not None:
                pitch = average_features(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
                pitch_emb = self.pitch_emb(pitch.unsqueeze(1))
            else:
                pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))
        else:
            pitch = average_features(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
            pitch_emb = self.pitch_emb(pitch.unsqueeze(1))

        enc_out = enc_out + pitch_emb.transpose(1, 2)

        if self.cond_on_lm_embeddings:
            enc_out = enc_out + lm_features

        # Regulate length
        len_regulated_enc_out, dec_lens = regulate_len(attn_hard_dur, enc_out)

        dec_out, dec_lens = self.decoder(len_regulated_enc_out, get_mask_from_lengths(dec_lens).unsqueeze(2))
        pred_spect = self.proj(dec_out)

        return (
            pred_spect,
            durs_predicted,
            log_durs_predicted,
            pitch_predicted,
            attn_soft,
            attn_logprob,
            attn_hard,
            attn_hard_dur,
        )

    def infer(
        self,
        text,
        text_len=None,
        text_mask=None,
        spect=None,
        spect_len=None,
        attn_prior=None,
        use_gt_durs=False,
        lm_tokens=None,
        pitch=None,
    ):
        if text_mask is None:
            text_mask = get_mask_from_lengths(text_len).unsqueeze(2)

        enc_out, enc_mask = self.encoder(text, text_mask)

        # Aligner
        attn_hard_dur = None
        if use_gt_durs:
            attn_soft, attn_logprob, attn_hard, attn_hard_dur = self.run_aligner(
                text, text_len, text_mask, spect, spect_len, attn_prior
            )

        if self.cond_on_lm_embeddings:
            lm_emb = self.lm_embeddings(lm_tokens)
            lm_features = self.self_attention_module(
                enc_out, lm_emb, lm_emb, q_mask=enc_mask.squeeze(2), kv_mask=lm_tokens != self.lm_padding_value
            )

        # Duration predictor
        log_durs_predicted = self.duration_predictor(enc_out, enc_mask)
        durs_predicted = torch.clamp(log_durs_predicted.exp() - 1, 0)

        # Avg pitch, pitch predictor
        if use_gt_durs and pitch is not None:
            pitch = average_features(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
            pitch_emb = self.pitch_emb(pitch.unsqueeze(1))
        else:
            pitch_predicted = self.pitch_predictor(enc_out, enc_mask)
            pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))

        # Add pitch emb
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        if self.cond_on_lm_embeddings:
            enc_out = enc_out + lm_features

        if use_gt_durs:
            if attn_hard_dur is not None:
                len_regulated_enc_out, dec_lens = regulate_len(attn_hard_dur, enc_out)
            else:
                raise NotImplementedError
        else:
            len_regulated_enc_out, dec_lens = regulate_len(durs_predicted, enc_out)

        dec_out, _ = self.decoder(len_regulated_enc_out, get_mask_from_lengths(dec_lens).unsqueeze(2))
        pred_spect = self.proj(dec_out)

        return pred_spect

    def on_train_epoch_start(self):
        bin_loss_start_epoch = np.ceil(self.bin_loss_start_ratio * self._trainer.max_epochs)

        # Add bin loss when current_epoch >= bin_start_epoch
        if not self.add_bin_loss and self.current_epoch >= bin_loss_start_epoch:
            logging.info(f"Using hard attentions after epoch: {self.current_epoch}")
            self.add_bin_loss = True

        if self.add_bin_loss:
            self.bin_loss_scale = min((self.current_epoch - bin_loss_start_epoch) / self.bin_loss_warmup_epochs, 1.0)

    def training_step(self, batch, batch_idx):
        attn_prior, lm_tokens = None, None
        if self.cond_on_lm_embeddings:
            audio, audio_len, text, text_len, attn_prior, pitch, _, lm_tokens = batch
        else:
            audio, audio_len, text, text_len, attn_prior, pitch, _ = batch

        spect, spect_len = self.preprocessor(input_signal=audio, length=audio_len)

        # pitch normalization
        zero_pitch_idx = pitch == 0
        pitch = (pitch - self.pitch_mean) / self.pitch_std
        pitch[zero_pitch_idx] = 0.0

        (pred_spect, _, pred_log_durs, pred_pitch, attn_soft, attn_logprob, attn_hard, attn_hard_dur,) = self(
            text=text,
            text_len=text_len,
            pitch=pitch,
            spect=spect,
            spect_len=spect_len,
            attn_prior=attn_prior,
            lm_tokens=lm_tokens,
        )

        (loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss, mel_loss, ctc_loss, bin_loss,) = self._metrics(
            pred_durs=pred_log_durs,
            pred_pitch=pred_pitch,
            true_durs=attn_hard_dur,
            true_text_len=text_len,
            true_pitch=pitch,
            true_spect=spect,
            pred_spect=pred_spect,
            true_spect_len=spect_len,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
        )

        train_log = {
            'train_loss': loss,
            'train_durs_loss': durs_loss,
            'train_pitch_loss': torch.tensor(1.0).to(durs_loss.device) if pitch_loss is None else pitch_loss,
            'train_mel_loss': mel_loss,
            'train_durs_acc': acc,
            'train_durs_acc_dist_3': acc_dist_3,
            'train_ctc_loss': torch.tensor(1.0).to(durs_loss.device) if ctc_loss is None else ctc_loss,
            'train_bin_loss': torch.tensor(1.0).to(durs_loss.device) if bin_loss is None else bin_loss,
        }

        return {'loss': loss, 'progress_bar': {k: v.detach() for k, v in train_log.items()}, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        attn_prior, lm_tokens = None, None
        if self.cond_on_lm_embeddings:
            audio, audio_len, text, text_len, attn_prior, pitch, _, lm_tokens = batch
        else:
            audio, audio_len, text, text_len, attn_prior, pitch, _ = batch

        spect, spect_len = self.preprocessor(input_signal=audio, length=audio_len)

        # pitch normalization
        zero_pitch_idx = pitch == 0
        pitch = (pitch - self.pitch_mean) / self.pitch_std
        pitch[zero_pitch_idx] = 0.0

        (pred_spect, _, pred_log_durs, pred_pitch, attn_soft, attn_logprob, attn_hard, attn_hard_dur,) = self(
            text=text,
            text_len=text_len,
            pitch=pitch,
            spect=spect,
            spect_len=spect_len,
            attn_prior=attn_prior,
            lm_tokens=lm_tokens,
        )

        (loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss, mel_loss, ctc_loss, bin_loss,) = self._metrics(
            pred_durs=pred_log_durs,
            pred_pitch=pred_pitch,
            true_durs=attn_hard_dur,
            true_text_len=text_len,
            true_pitch=pitch,
            true_spect=spect,
            pred_spect=pred_spect,
            true_spect_len=spect_len,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
        )

        # without ground truth internal features except for durations
        pred_spect, _, pred_log_durs, pred_pitch, attn_soft, attn_logprob, attn_hard, attn_hard_dur = self(
            text=text,
            text_len=text_len,
            pitch=None,
            spect=spect,
            spect_len=spect_len,
            attn_prior=attn_prior,
            lm_tokens=lm_tokens,
        )

        *_, with_pred_features_mel_loss, _, _ = self._metrics(
            pred_durs=pred_log_durs,
            pred_pitch=pred_pitch,
            true_durs=attn_hard_dur,
            true_text_len=text_len,
            true_pitch=pitch,
            true_spect=spect,
            pred_spect=pred_spect,
            true_spect_len=spect_len,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
        )

        val_log = {
            'val_loss': loss,
            'val_durs_loss': durs_loss,
            'val_pitch_loss': torch.tensor(1.0).to(durs_loss.device) if pitch_loss is None else pitch_loss,
            'val_mel_loss': mel_loss,
            'val_with_pred_features_mel_loss': with_pred_features_mel_loss,
            'val_durs_acc': acc,
            'val_durs_acc_dist_3': acc_dist_3,
            'val_ctc_loss': torch.tensor(1.0).to(durs_loss.device) if ctc_loss is None else ctc_loss,
            'val_bin_loss': torch.tensor(1.0).to(durs_loss.device) if bin_loss is None else bin_loss,
        }
        self.log_dict(val_log, prog_bar=False, on_epoch=True, logger=True, sync_dist=True)

        if batch_idx == 0 and self.current_epoch % 5 == 0 and isinstance(self.logger, WandbLogger):
            specs = []
            pitches = []
            for i in range(min(3, spect.shape[0])):
                specs += [
                    wandb.Image(
                        plot_spectrogram_to_numpy(spect[i, :, : spect_len[i]].data.cpu().numpy()),
                        caption=f"gt mel {i}",
                    ),
                    wandb.Image(
                        plot_spectrogram_to_numpy(pred_spect.transpose(1, 2)[i, :, : spect_len[i]].data.cpu().numpy()),
                        caption=f"pred mel {i}",
                    ),
                ]

                pitches += [
                    wandb.Image(
                        plot_pitch_to_numpy(
                            average_features(pitch.unsqueeze(1), attn_hard_dur)
                            .squeeze(1)[i, : text_len[i]]
                            .data.cpu()
                            .numpy(),
                            ylim_range=[-2.5, 2.5],
                        ),
                        caption=f"gt pitch {i}",
                    ),
                ]

                pitches += [
                    wandb.Image(
                        plot_pitch_to_numpy(pred_pitch[i, : text_len[i]].data.cpu().numpy(), ylim_range=[-2.5, 2.5]),
                        caption=f"pred pitch {i}",
                    ),
                ]

            self.logger.experiment.log({"specs": specs, "pitches": pitches})

    @typecheck(
        input_types={
            "tokens": NeuralType(('B', 'T_text'), TokenIndex(), optional=True),
            "tokens_len": NeuralType(('B'), LengthsType(), optional=True),
            "lm_tokens": NeuralType(('B', 'T_lm_tokens'), TokenIndex(), optional=True),
            "raw_texts": [NeuralType(optional=True)],
            "lm_model": NeuralType(optional=True),
        },
        output_types={"spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),},
    )
    def generate_spectrogram(
        self,
        tokens: Optional[torch.Tensor] = None,
        tokens_len: Optional[torch.Tensor] = None,
        lm_tokens: Optional[torch.Tensor] = None,
        raw_texts: Optional[List[str]] = None,
        norm_text_for_lm_model: bool = True,
        lm_model: str = "albert",
    ):
        if tokens is not None:
            if tokens_len is None:
                # It is assumed that padding is consecutive and only at the end
                tokens_len = (tokens != self.tokenizer.pad).sum(dim=-1)
        else:
            if raw_texts is None:
                raise ValueError("raw_texts must be specified if tokens is None")

            t_seqs = [self.tokenizer(t) for t in raw_texts]
            tokens = torch.nn.utils.rnn.pad_sequence(
                sequences=[torch.tensor(t, dtype=torch.long, device=self.device) for t in t_seqs],
                batch_first=True,
                padding_value=self.tokenizer.pad,
            )
            tokens_len = torch.tensor([len(t) for t in t_seqs], dtype=torch.long, device=tokens.device)

        if self.cond_on_lm_embeddings and lm_tokens is None:
            if raw_texts is None:
                raise ValueError("raw_texts must be specified if lm_tokens is None")

            lm_model_tokenizer = self._get_lm_model_tokenizer(lm_model)
            lm_padding_value = lm_model_tokenizer._convert_token_to_id('<pad>')
            lm_space_value = lm_model_tokenizer._convert_token_to_id('▁')

            assert isinstance(self.tokenizer, EnglishCharsTokenizer) or isinstance(
                self.tokenizer, EnglishPhonemesTokenizer
            )

            if norm_text_for_lm_model and self.text_normalizer_call is not None:
                raw_texts = [self.text_normalizer_call(t, **self.text_normalizer_call_kwargs) for t in raw_texts]

            preprocess_texts_as_tts_input = [self.tokenizer.text_preprocessing_func(t) for t in raw_texts]
            lm_tokens_as_ids_list = [
                lm_model_tokenizer.encode(t, add_special_tokens=False) for t in preprocess_texts_as_tts_input
            ]

            if self.tokenizer.pad_with_space:
                lm_tokens_as_ids_list = [[lm_space_value] + t + [lm_space_value] for t in lm_tokens_as_ids_list]

            lm_tokens = torch.full(
                (len(lm_tokens_as_ids_list), max([len(t) for t in lm_tokens_as_ids_list])),
                fill_value=lm_padding_value,
                device=tokens.device,
            )
            for i, lm_tokens_i in enumerate(lm_tokens_as_ids_list):
                lm_tokens[i, : len(lm_tokens_i)] = torch.tensor(lm_tokens_i, device=tokens.device)

        pred_spect = self.infer(tokens, tokens_len, lm_tokens=lm_tokens).transpose(1, 2)
        return pred_spect

    def parse(self, text: str, normalize=True) -> torch.Tensor:
        if self.training:
            logging.warning("parse() is meant to be called in eval mode.")
        if normalize and self.text_normalizer_call is not None:
            text = self.text_normalizer_call(text, **self.text_normalizer_call_kwargs)

        eval_phon_mode = contextlib.nullcontext()
        if hasattr(self.tokenizer, "set_phone_prob"):
            eval_phon_mode = self.tokenizer.set_phone_prob(prob=1.0)

        with eval_phon_mode:
            tokens = self.tokenizer.encode(text)
        return torch.tensor(tokens).long().unsqueeze(0).to(self.device)

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
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_lj_mixertts",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_lj_mixertts/versions/1.6.0/files/tts_en_lj_mixertts.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz with and can be used to generate female English voices with an American accent.",
            class_=cls,  # noqa
        )
        list_of_models.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_lj_mixerttsx",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_lj_mixerttsx/versions/1.6.0/files/tts_en_lj_mixerttsx.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz with and can be used to generate female English voices with an American accent.",
            class_=cls,  # noqa
        )
        list_of_models.append(model)

        return list_of_models

    # Methods for model exportability
    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "lm_tokens": NeuralType(('B', 'T_lm_tokens'), TokenIndex(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
        }

    def input_example(self, max_text_len=10, max_lm_tokens_len=10):
        text = torch.randint(
            low=0, high=len(self.tokenizer.tokens), size=(1, max_text_len), device=self.device, dtype=torch.long,
        )

        inputs = {'text': text}

        if self.cond_on_lm_embeddings:
            inputs['lm_tokens'] = torch.randint(
                low=0,
                high=self.lm_embeddings.weight.shape[0],
                size=(1, max_lm_tokens_len),
                device=self.device,
                dtype=torch.long,
            )

        return (inputs,)

    def forward_for_export(self, text, lm_tokens=None):
        text_mask = (text != self.tokenizer_pad).unsqueeze(2)
        spect = self.infer(text=text, text_mask=text_mask, lm_tokens=lm_tokens).transpose(1, 2)
        return spect.to(torch.float)
