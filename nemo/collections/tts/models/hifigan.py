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

import itertools

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.loggers.wandb import WandbLogger

from nemo.collections.tts.losses.hifigan_losses import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss
from nemo.collections.tts.models.base import Vocoder
from nemo.collections.tts.modules.hifigan_modules import MultiPeriodDiscriminator, MultiScaleDiscriminator
from nemo.collections.tts.parts.utils.helpers import get_batch_size, get_num_workers, plot_spectrogram_to_numpy
from nemo.core.classes import Exportable
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import AudioSignal, MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.optim.lr_scheduler import CosineAnnealing, compute_max_steps
from nemo.utils import logging, model_utils

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


class HifiGanModel(Vocoder, Exportable):
    """
    HiFi-GAN model (https://arxiv.org/abs/2010.05646) that is used to generate audio from mel spectrogram.
    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        super().__init__(cfg=cfg, trainer=trainer)

        self.audio_to_melspec_precessor = instantiate(cfg.preprocessor)
        # We use separate preprocessor for training, because we need to pass grads and remove pitch fmax limitation
        self.trg_melspec_fn = instantiate(cfg.preprocessor, highfreq=None, use_grads=True)
        self.generator = instantiate(cfg.generator)
        self.mpd = MultiPeriodDiscriminator(debug=cfg.debug if "debug" in cfg else False)
        self.msd = MultiScaleDiscriminator(debug=cfg.debug if "debug" in cfg else False)
        self.feature_loss = FeatureMatchingLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()

        self.l1_factor = cfg.get("l1_loss_factor", 45)

        self.sample_rate = self._cfg.preprocessor.sample_rate
        self.stft_bias = None

        self.input_as_mel = False
        if self._train_dl:
            self.input_as_mel = self._train_dl.dataset.load_precomputed_mel

        self.automatic_optimization = False

    def _get_max_steps(self):
        return compute_max_steps(
            max_epochs=self._cfg.max_epochs,
            accumulate_grad_batches=self.trainer.accumulate_grad_batches,
            limit_train_batches=self.trainer.limit_train_batches,
            num_workers=get_num_workers(self.trainer),
            num_samples=len(self._train_dl.dataset),
            batch_size=get_batch_size(self._train_dl),
            drop_last=self._train_dl.drop_last,
        )

    @staticmethod
    def get_warmup_steps(max_steps, warmup_steps, warmup_ratio):
        if warmup_steps is not None and warmup_ratio is not None:
            raise ValueError(f'Either use warmup_steps or warmup_ratio for scheduler')

        if warmup_steps is not None:
            return warmup_steps

        if warmup_ratio is not None:
            return warmup_ratio * max_steps

        raise ValueError(f'Specify warmup_steps or warmup_ratio for scheduler')

    def configure_optimizers(self):
        optim_config = self._cfg.optim.copy()

        OmegaConf.set_struct(optim_config, False)
        sched_config = optim_config.pop("sched", None)
        OmegaConf.set_struct(optim_config, True)

        optim_g = instantiate(optim_config, params=self.generator.parameters(),)
        optim_d = instantiate(optim_config, params=itertools.chain(self.msd.parameters(), self.mpd.parameters()),)

        # Backward compatibility
        if sched_config is None and 'sched' in self._cfg:
            sched_config = self._cfg.sched

        if sched_config is not None:
            max_steps = self._cfg.get("max_steps", None)
            if max_steps is None or max_steps < 0:
                max_steps = self._get_max_steps()

            warmup_steps = HifiGanModel.get_warmup_steps(
                max_steps=max_steps,
                warmup_steps=sched_config.get("warmup_steps", None),
                warmup_ratio=sched_config.get("warmup_ratio", None),
            )

            scheduler_g = CosineAnnealing(
                optimizer=optim_g, max_steps=max_steps, min_lr=sched_config.min_lr, warmup_steps=warmup_steps,
            )  # Use warmup to delay start
            sch1_dict = {
                'scheduler': scheduler_g,
                'interval': 'step',
            }

            scheduler_d = CosineAnnealing(optimizer=optim_d, max_steps=max_steps, min_lr=sched_config.min_lr,)
            sch2_dict = {
                'scheduler': scheduler_d,
                'interval': 'step',
            }

            return [optim_g, optim_d], [sch1_dict, sch2_dict]
        else:
            return [optim_g, optim_d]

    @typecheck()
    def forward(self, *, spec):
        """
        Runs the generator, for inputs and outputs see input_types, and output_types
        """
        return self.generator(x=spec)

    @typecheck(
        input_types={"spec": NeuralType(('B', 'C', 'T'), MelSpectrogramType())},
        output_types={"audio": NeuralType(('B', 'T'), AudioSignal())},
    )
    def convert_spectrogram_to_audio(self, spec: 'torch.tensor') -> 'torch.tensor':
        return self(spec=spec).squeeze(1)

    def training_step(self, batch, batch_idx):
        if self.input_as_mel:
            # Pre-computed spectrograms will be used as input
            audio, audio_len, audio_mel = batch
        else:
            audio, audio_len = batch
            audio_mel, _ = self.audio_to_melspec_precessor(audio, audio_len)

        # Mel as input for L1 mel loss
        audio_trg_mel, _ = self.trg_melspec_fn(audio, audio_len)
        audio = audio.unsqueeze(1)

        audio_pred = self.generator(x=audio_mel)
        audio_pred_mel, _ = self.trg_melspec_fn(audio_pred.squeeze(1), audio_len)

        optim_g, optim_d = self.optimizers()

        # Train discriminator
        optim_d.zero_grad()
        mpd_score_real, mpd_score_gen, _, _ = self.mpd(y=audio, y_hat=audio_pred.detach())
        loss_disc_mpd, _, _ = self.discriminator_loss(
            disc_real_outputs=mpd_score_real, disc_generated_outputs=mpd_score_gen
        )
        msd_score_real, msd_score_gen, _, _ = self.msd(y=audio, y_hat=audio_pred.detach())
        loss_disc_msd, _, _ = self.discriminator_loss(
            disc_real_outputs=msd_score_real, disc_generated_outputs=msd_score_gen
        )
        loss_d = loss_disc_msd + loss_disc_mpd
        self.manual_backward(loss_d)
        optim_d.step()

        # Train generator
        optim_g.zero_grad()
        loss_mel = F.l1_loss(audio_pred_mel, audio_trg_mel)
        _, mpd_score_gen, fmap_mpd_real, fmap_mpd_gen = self.mpd(y=audio, y_hat=audio_pred)
        _, msd_score_gen, fmap_msd_real, fmap_msd_gen = self.msd(y=audio, y_hat=audio_pred)
        loss_fm_mpd = self.feature_loss(fmap_r=fmap_mpd_real, fmap_g=fmap_mpd_gen)
        loss_fm_msd = self.feature_loss(fmap_r=fmap_msd_real, fmap_g=fmap_msd_gen)
        loss_gen_mpd, _ = self.generator_loss(disc_outputs=mpd_score_gen)
        loss_gen_msd, _ = self.generator_loss(disc_outputs=msd_score_gen)
        loss_g = loss_gen_msd + loss_gen_mpd + loss_fm_msd + loss_fm_mpd + loss_mel * self.l1_factor
        self.manual_backward(loss_g)
        optim_g.step()

        # Run schedulers
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            sch1, sch2 = schedulers
            sch1.step()
            sch2.step()

        metrics = {
            "g_loss_fm_mpd": loss_fm_mpd,
            "g_loss_fm_msd": loss_fm_msd,
            "g_loss_gen_mpd": loss_gen_mpd,
            "g_loss_gen_msd": loss_gen_msd,
            "g_loss": loss_g,
            "d_loss_mpd": loss_disc_mpd,
            "d_loss_msd": loss_disc_msd,
            "d_loss": loss_d,
            "global_step": self.global_step,
            "lr": optim_g.param_groups[0]['lr'],
        }
        self.log_dict(metrics, on_step=True, sync_dist=True)
        self.log("g_l1_loss", loss_mel, prog_bar=True, logger=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        if self.input_as_mel:
            audio, audio_len, audio_mel = batch
            audio_mel_len = [audio_mel.shape[1]] * audio_mel.shape[0]
        else:
            audio, audio_len = batch
            audio_mel, audio_mel_len = self.audio_to_melspec_precessor(audio, audio_len)
        audio_pred = self(spec=audio_mel)

        # Perform bias denoising
        pred_denoised = self._bias_denoise(audio_pred, audio_mel).squeeze(1)
        pred_denoised_mel, _ = self.audio_to_melspec_precessor(pred_denoised, audio_len)

        if self.input_as_mel:
            gt_mel, gt_mel_len = self.audio_to_melspec_precessor(audio, audio_len)
        audio_pred_mel, _ = self.audio_to_melspec_precessor(audio_pred.squeeze(1), audio_len)
        loss_mel = F.l1_loss(audio_mel, audio_pred_mel)

        self.log_dict({"val_loss": loss_mel}, on_epoch=True, sync_dist=True)

        # Plot audio once per epoch
        if batch_idx == 0 and isinstance(self.logger, WandbLogger) and HAVE_WANDB:
            clips = []
            specs = []
            for i in range(min(5, audio.shape[0])):
                clips += [
                    wandb.Audio(
                        audio[i, : audio_len[i]].data.cpu().numpy(),
                        caption=f"real audio {i}",
                        sample_rate=self.sample_rate,
                    ),
                    wandb.Audio(
                        audio_pred[i, 0, : audio_len[i]].data.cpu().numpy().astype('float32'),
                        caption=f"generated audio {i}",
                        sample_rate=self.sample_rate,
                    ),
                    wandb.Audio(
                        pred_denoised[i, : audio_len[i]].data.cpu().numpy(),
                        caption=f"denoised audio {i}",
                        sample_rate=self.sample_rate,
                    ),
                ]
                specs += [
                    wandb.Image(
                        plot_spectrogram_to_numpy(audio_mel[i, :, : audio_mel_len[i]].data.cpu().numpy()),
                        caption=f"input mel {i}",
                    ),
                    wandb.Image(
                        plot_spectrogram_to_numpy(audio_pred_mel[i, :, : audio_mel_len[i]].data.cpu().numpy()),
                        caption=f"output mel {i}",
                    ),
                    wandb.Image(
                        plot_spectrogram_to_numpy(pred_denoised_mel[i, :, : audio_mel_len[i]].data.cpu().numpy()),
                        caption=f"denoised mel {i}",
                    ),
                ]
                if self.input_as_mel:
                    specs += [
                        wandb.Image(
                            plot_spectrogram_to_numpy(gt_mel[i, :, : audio_mel_len[i]].data.cpu().numpy()),
                            caption=f"gt mel {i}",
                        ),
                    ]

            self.logger.experiment.log({"audio": clips, "specs": specs})

    def _bias_denoise(self, audio, mel):
        def stft(x):
            comp = torch.stft(x.squeeze(1), n_fft=1024, hop_length=256, win_length=1024, return_complex=True)
            comp = torch.view_as_real(comp)
            real, imag = comp[..., 0], comp[..., 1]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase

        def istft(mags, phase):
            comp = torch.stack([mags * torch.cos(phase), mags * torch.sin(phase)], dim=-1)
            x = torch.istft(torch.view_as_complex(comp), n_fft=1024, hop_length=256, win_length=1024)
            return x

        # Create bias tensor
        if self.stft_bias is None or self.stft_bias.shape[0] != audio.shape[0]:
            audio_bias = self(spec=torch.zeros_like(mel, device=mel.device))
            self.stft_bias, _ = stft(audio_bias)
            self.stft_bias = self.stft_bias[:, :, 0][:, :, None]

        audio_mags, audio_phase = stft(audio)
        audio_mags = audio_mags - self.cfg.get("denoise_strength", 0.0025) * self.stft_bias
        audio_mags = torch.clamp(audio_mags, 0.0)
        audio_denoised = istft(audio_mags, audio_phase).unsqueeze(1)

        return audio_denoised

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg["dataloader_params"]):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

    def setup_test_data(self, cfg):
        pass

    @classmethod
    def list_available_models(cls) -> 'Optional[Dict[str, str]]':
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_hifigan",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_hifigan/versions/1.0.0rc1/files/tts_hifigan.nemo",
            description="This model is trained on LJSpeech audio sampled at 22050Hz and mel spectrograms generated from"
            " Tacotron2, TalkNet, and FastPitch. This model has been tested on generating female English "
            "voices with an American accent.",
            class_=cls,
        )
        list_of_models.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_lj_hifigan_ft_mixertts",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_lj_hifigan/versions/1.6.0/files/tts_en_lj_hifigan_ft_mixertts.nemo",
            description="This model is trained on LJSpeech audio sampled at 22050Hz and mel spectrograms generated from"
            " Mixer-TTS. This model has been tested on generating female English voices with an American accent.",
            class_=cls,
        )
        list_of_models.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_lj_hifigan_ft_mixerttsx",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_lj_hifigan/versions/1.6.0/files/tts_en_lj_hifigan_ft_mixerttsx.nemo",
            description="This model is trained on LJSpeech audio sampled at 22050Hz and mel spectrograms generated from"
            " Mixer-TTS-X. This model has been tested on generating female English voices with an American accent.",
            class_=cls,
        )
        list_of_models.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_hifitts_hifigan_ft_fastpitch",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_multispeaker_fastpitchhifigan/versions/1.10.0/files/tts_en_hifitts_hifigan_ft_fastpitch.nemo",
            description="This model is trained on HiFiTTS audio sampled at 44100Hz and mel spectrograms generated from"
            " FastPitch. This model has been tested on generating male and female English voices with an American accent.",
            class_=cls,
        )
        list_of_models.append(model)

        # de-DE, single male speaker, 22050 Hz, Thorsten Müller’s German Neutral-TTS Dataset, 21.02
        model = PretrainedModelInfo(
            pretrained_model_name="tts_de_hifigan_singleSpeaker_thorstenNeutral_2102",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_de_fastpitchhifigan/versions/1.15.0/files/tts_de_hifigan_thorstens2102.nemo",
            description="This model is finetuned from the HiFiGAN pretrained checkpoint `tts_en_lj_hifigan_ft_mixerttsx`"
            " by the mel-spectrograms generated from the FastPitch checkpoint `tts_de_fastpitch_singleSpeaker_thorstenNeutral_2102`."
            " This model has been tested on generating male German neutral voices.",
            class_=cls,
        )
        list_of_models.append(model)

        # de-DE, single male speaker, 22050 Hz, Thorsten Müller’s German Neutral-TTS Dataset, 22.10
        model = PretrainedModelInfo(
            pretrained_model_name="tts_de_hifigan_singleSpeaker_thorstenNeutral_2210",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_de_fastpitchhifigan/versions/1.15.0/files/tts_de_hifigan_thorstens2210.nemo",
            description="This model is finetuned from the HiFiGAN pretrained checkpoint `tts_en_lj_hifigan_ft_mixerttsx`"
            " by the mel-spectrograms generated from the FastPitch checkpoint `tts_de_fastpitch_singleSpeaker_thorstenNeutral_2210`."
            " This model has been tested on generating male German neutral voices.",
            class_=cls,
        )
        list_of_models.append(model)

        # de-DE, multi-speaker, 5 speakers, 44100 Hz, HUI-Audio-Corpus-German Clean.
        model = PretrainedModelInfo(
            pretrained_model_name="tts_de_hui_hifigan_ft_fastpitch_multispeaker_5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_de_fastpitch_multispeaker_5/versions/1.11.0/files/tts_de_hui_hifigan_ft_fastpitch_multispeaker_5.nemo",
            description="This model is finetuned from the HiFiGAN pretrained checkpoint `tts_en_hifitts_hifigan_ft_fastpitch` "
            "by the mel-spectrograms generated from the FastPitch checkpoint `tts_de_fastpitch_multispeaker_5`. This model "
            "has been tested on generating male and female German voices.",
            class_=cls,
        )
        list_of_models.append(model)

        # Spanish, multi-speaker, 44100 Hz, Latin American Spanish OpenSLR
        model = PretrainedModelInfo(
            pretrained_model_name="tts_es_hifigan_ft_fastpitch_multispeaker",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_es_multispeaker_fastpitchhifigan/versions/1.15.0/files/tts_es_hifigan_ft_fastpitch_multispeaker.nemo",
            description="This model is trained on the audio from 6 crowdsourced Latin American Spanish OpenSLR "
            "datasets and finetuned on the mel-spectrograms generated from the FastPitch checkpoint "
            "`tts_es_fastpitch_multispeaker`. This model has been tested on generating male and female "
            "Spanish voices with Latin American accents.",
            class_=cls,
        )
        list_of_models.append(model)

        # zh, single female speaker, 22050Hz, SFSpeech Bilingual Chinese/English dataset, improved model using richer
        # dict and jieba word segmenter for polyphone disambiguation.
        model = PretrainedModelInfo(
            pretrained_model_name="tts_zh_hifigan_sfspeech",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_zh_fastpitch_hifigan_sfspeech/versions/1.15.0/files/tts_zh_hifigan_sfspeech.nemo",
            description="This model is finetuned from the HiFiGAN pretrained checkpoint `tts_en_lj_hifigan_ft_mixerttsx`"
            " by the mel-spectrograms generated from the FastPitch checkpoint `tts_zh_fastpitch_sfspeech`."
            " This model has been tested on generating female Mandarin Chinese voices.",
            class_=cls,
        )
        list_of_models.append(model)

        return list_of_models

    def load_state_dict(self, state_dict, strict=True):
        # Override load_state_dict to give us some flexibility to be backward-compatible with old checkpoints
        new_state_dict = {}
        num_resblocks = len(self.cfg['generator']['resblock_kernel_sizes'])
        for k, v in state_dict.items():
            new_k = k
            if 'resblocks' in k:
                parts = k.split(".")
                # only do this is the checkpoint type is older
                if len(parts) == 6:
                    layer = int(parts[2])
                    new_layer = f"{layer // num_resblocks}.{layer % num_resblocks}"
                    new_k = f"generator.resblocks.{new_layer}.{'.'.join(parts[3:])}"
            new_state_dict[new_k] = v
        super().load_state_dict(new_state_dict, strict=strict)

    # Methods for model exportability
    def _prepare_for_export(self, **kwargs):
        if self.generator is not None:
            try:
                self.generator.remove_weight_norm()
            except ValueError:
                return

    @property
    def input_types(self):
        return {
            "spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'S', 'T'), AudioSignal(self.sample_rate)),
        }

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        par = next(self.parameters())
        mel = torch.randn((max_batch, self.cfg['preprocessor']['nfilt'], max_dim), device=self.device, dtype=par.dtype)
        return ({'spec': mel},)

    def forward_for_export(self, spec):
        """
        Runs the generator, for inputs and outputs see input_types, and output_types
        """
        return self.generator(x=spec)
