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


import contextlib

import omegaconf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.cuda.amp import autocast
from torch.nn import functional as F

from nemo.collections.tts.data.tts_dataset import DistributedBucketSampler
from nemo.collections.tts.losses.vits_losses import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, KlLoss
from nemo.collections.tts.models.base import TextToWaveform
from nemo.collections.tts.modules.vits_modules import MultiPeriodDiscriminator
from nemo.collections.tts.parts.utils.helpers import clip_grad_value_, plot_spectrogram_to_numpy, slice_segments
from nemo.collections.tts.torch.tts_data_types import SpeakerID
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import AudioSignal, FloatType, Index, IntType, TokenIndex
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.optim.lr_scheduler import CosineAnnealing
from nemo.utils import logging, model_utils
from nemo.utils.decorators.experimental import experimental

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


@experimental
class VitsModel(TextToWaveform):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        # Convert to Hydra 1.0 compatible DictConfig

        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # setup normalizer
        self.normalizer = None
        self.text_normalizer_call = None
        self.text_normalizer_call_kwargs = {}
        self._setup_normalizer(cfg)

        # setup tokenizer
        self.tokenizer = None
        self._setup_tokenizer(cfg)
        assert self.tokenizer is not None

        num_tokens = len(self.tokenizer.tokens)
        self.tokenizer_pad = self.tokenizer.pad

        super().__init__(cfg=cfg, trainer=trainer)

        self.audio_to_melspec_processor = instantiate(cfg.preprocessor, highfreq=cfg.train_ds.dataset.highfreq)

        self.feat_matching_loss = FeatureMatchingLoss()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.kl_loss = KlLoss()

        self.net_g = instantiate(
            cfg.synthesizer,
            n_vocab=num_tokens,
            spec_channels=cfg.n_fft // 2 + 1,
            segment_size=cfg.segment_size // cfg.n_window_stride,
            padding_idx=self.tokenizer_pad,
        )

        self.net_d = MultiPeriodDiscriminator(cfg.use_spectral_norm)

        self.automatic_optimization = False

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
                self.text_normalizer_call = self.normalizer.normalize
            except Exception as e:
                logging.error(e)
                raise ImportError(
                    "`nemo_text_processing` not installed, see https://github.com/NVIDIA/NeMo-text-processing for more details"
                )
            if "text_normalizer_call_kwargs" in cfg:
                self.text_normalizer_call_kwargs = cfg.text_normalizer_call_kwargs

    def _setup_tokenizer(self, cfg):
        text_tokenizer_kwargs = {}
        if "g2p" in cfg.text_tokenizer and cfg.text_tokenizer.g2p is not None:
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

    def parse(self, text: str, normalize=True) -> torch.tensor:
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

    def configure_optimizers(self):
        optim_config = self._cfg.optim.copy()
        OmegaConf.set_struct(optim_config, False)
        sched_config = optim_config.pop("sched", None)
        OmegaConf.set_struct(optim_config, True)

        optim_g = instantiate(optim_config, params=self.net_g.parameters(),)
        optim_d = instantiate(optim_config, params=self.net_d.parameters(),)

        if sched_config is not None:
            if sched_config.name == 'ExponentialLR':
                scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=sched_config.lr_decay)
                scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=sched_config.lr_decay)
            elif sched_config.name == 'CosineAnnealing':
                scheduler_g = CosineAnnealing(
                    optimizer=optim_g, max_steps=sched_config.max_steps, min_lr=sched_config.min_lr,
                )
                scheduler_d = CosineAnnealing(
                    optimizer=optim_d, max_steps=sched_config.max_steps, min_lr=sched_config.min_lr,
                )
            else:
                raise ValueError("Unknown optimizer.")

            scheduler_g_dict = {'scheduler': scheduler_g, 'interval': 'step'}
            scheduler_d_dict = {'scheduler': scheduler_d, 'interval': 'step'}
            return [optim_g, optim_d], [scheduler_g_dict, scheduler_d_dict]
        else:
            return [optim_g, optim_d]

    # for inference
    @typecheck(
        input_types={
            "tokens": NeuralType(('B', 'T_text'), TokenIndex()),
            "speakers": NeuralType(('B',), Index(), optional=True),
            "noise_scale": NeuralType(('B',), FloatType(), optional=True),
            "length_scale": NeuralType(('B',), FloatType(), optional=True),
            "noise_scale_w": NeuralType(('B',), FloatType(), optional=True),
            "max_len": NeuralType(('B',), IntType(), optional=True),
        }
    )
    def forward(self, tokens, speakers=None, noise_scale=1, length_scale=1, noise_scale_w=1.0, max_len=1000):
        text_len = torch.tensor([tokens.size(-1)]).to(int).to(tokens.device)
        audio_pred, attn, y_mask, (z, z_p, m_p, logs_p) = self.net_g.infer(
            tokens,
            text_len,
            speakers=speakers,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            max_len=max_len,
        )
        return audio_pred, attn, y_mask, (z, z_p, m_p, logs_p)

    def training_step(self, batch, batch_idx):
        speakers = None
        if SpeakerID in self._train_dl.dataset.sup_data_types_set:
            (audio, audio_len, text, text_len, speakers) = batch
        else:
            (audio, audio_len, text, text_len) = batch

        spec, spec_lengths = self.audio_to_melspec_processor(audio, audio_len, linear_spec=True)

        with autocast(enabled=True):
            audio_pred, l_length, attn, ids_slice, text_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(
                text, text_len, spec, spec_lengths, speakers
            )

        audio_pred = audio_pred.float()

        audio_pred_mel, _ = self.audio_to_melspec_processor(audio_pred.squeeze(1), audio_len, linear_spec=False)

        audio = slice_segments(audio.unsqueeze(1), ids_slice * self.cfg.n_window_stride, self._cfg.segment_size)
        audio_mel, _ = self.audio_to_melspec_processor(audio.squeeze(1), audio_len, linear_spec=False)

        with autocast(enabled=True):
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(audio, audio_pred.detach())

        with autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = self.disc_loss(
                disc_real_outputs=y_d_hat_r, disc_generated_outputs=y_d_hat_g
            )
            loss_disc_all = loss_disc

        # get optimizers
        optim_g, optim_d = self.optimizers()

        # train discriminator
        optim_d.zero_grad()
        self.manual_backward(loss_disc_all)
        norm_d = clip_grad_value_(self.net_d.parameters(), None)
        optim_d.step()

        with autocast(enabled=True):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(audio, audio_pred)
        # Generator
        with autocast(enabled=False):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(audio_mel, audio_pred_mel) * self._cfg.c_mel
            loss_kl = self.kl_loss(z_p=z_p, logs_q=logs_q, m_p=m_p, logs_p=logs_p, z_mask=z_mask) * self._cfg.c_kl
            loss_fm = self.feat_matching_loss(fmap_r=fmap_r, fmap_g=fmap_g)
            loss_gen, losses_gen = self.gen_loss(disc_outputs=y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        # train generator
        optim_g.zero_grad()
        self.manual_backward(loss_gen_all)
        norm_g = clip_grad_value_(self.net_g.parameters(), None)
        optim_g.step()

        schedulers = self.lr_schedulers()
        if schedulers is not None:
            sch1, sch2 = schedulers
            if (
                self.trainer.is_last_batch
                and isinstance(sch1, torch.optim.lr_scheduler.ExponentialLR)
                or isinstance(sch1, CosineAnnealing)
            ):
                sch1.step()
                sch2.step()

        metrics = {
            "loss_gen": loss_gen,
            "loss_fm": loss_fm,
            "loss_mel": loss_mel,
            "loss_dur": loss_dur,
            "loss_kl": loss_kl,
            "loss_gen_all": loss_gen_all,
            "loss_disc_all": loss_disc_all,
            "grad_gen": norm_g,
            "grad_disc": norm_d,
        }

        for i, v in enumerate(losses_gen):
            metrics[f"loss_gen_i_{i}"] = v

        for i, v in enumerate(losses_disc_r):
            metrics[f"loss_disc_r_{i}"] = v

        for i, v in enumerate(losses_disc_g):
            metrics[f"loss_disc_g_{i}"] = v

        self.log_dict(metrics, on_step=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        speakers = None
        if self.cfg.n_speakers > 1:
            (audio, audio_len, text, text_len, speakers) = batch
        else:
            (audio, audio_len, text, text_len) = batch

        audio_pred, _, mask, *_ = self.net_g.infer(text, text_len, speakers, max_len=1000)

        audio_pred = audio_pred.squeeze()
        audio_pred_len = mask.sum([1, 2]).long() * self._cfg.validation_ds.dataset.hop_length

        mel, mel_lengths = self.audio_to_melspec_processor(audio, audio_len)
        audio_pred_mel, audio_pred_mel_len = self.audio_to_melspec_processor(audio_pred, audio_pred_len)

        # plot audio once per epoch
        if batch_idx == 0 and isinstance(self.logger, WandbLogger) and HAVE_WANDB:
            logger = self.logger.experiment

            specs = []
            audios = []
            specs += [
                wandb.Image(
                    plot_spectrogram_to_numpy(mel[0, :, : mel_lengths[0]].data.cpu().numpy()),
                    caption=f"val_mel_target",
                ),
                wandb.Image(
                    plot_spectrogram_to_numpy(audio_pred_mel[0, :, : audio_pred_mel_len[0]].data.cpu().numpy()),
                    caption=f"val_mel_predicted",
                ),
            ]

            audios += [
                wandb.Audio(
                    audio[0, : audio_len[0]].data.cpu().to(torch.float).numpy(),
                    caption=f"val_wav_target",
                    sample_rate=self._cfg.sample_rate,
                ),
                wandb.Audio(
                    audio_pred[0, : audio_pred_len[0]].data.cpu().to(torch.float).numpy(),
                    caption=f"val_wav_predicted",
                    sample_rate=self._cfg.sample_rate,
                ),
            ]

            logger.log({"specs": specs, "audios": audios})

    def _loader(self, cfg):
        try:
            _ = cfg['dataset']['manifest_filepath']
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

    def train_dataloader(self):
        # default used by the Trainer
        dataset = instantiate(
            self.cfg.train_ds.dataset,
            text_normalizer=self.normalizer,
            text_normalizer_call_kwargs=self.text_normalizer_call_kwargs,
            text_tokenizer=self.tokenizer,
        )

        train_sampler = DistributedBucketSampler(dataset, **self.cfg.train_ds.batch_sampler)

        dataloader = torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, batch_sampler=train_sampler, **self.cfg.train_ds.dataloader_params,
        )
        return dataloader

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_lj_vits",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_lj_vits/versions/1.13.0/files/vits_ljspeech_fp16_full.nemo",
            description="This model is trained on LJSpeech audio sampled at 22050Hz. This model has been tested on generating female English "
            "voices with an American accent.",
            class_=cls,
        )
        list_of_models.append(model)
        return list_of_models

    @typecheck(
        input_types={"tokens": NeuralType(('B', 'T_text'), TokenIndex(), optional=True),},
        output_types={"audio": NeuralType(('B', 'T_audio'), AudioSignal())},
    )
    def convert_text_to_waveform(self, *, tokens, speakers=None):
        audio = self(tokens=tokens, speakers=speakers)[0].squeeze(1)
        return audio
