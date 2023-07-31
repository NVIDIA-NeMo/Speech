# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.tts.losses.audio_codec_loss import (
    MultiResolutionMelLoss,
    RelativeFeatureMatchingLoss,
    TimeDomainLoss,
)
from nemo.collections.tts.modules.common import GaussianDropout
from nemo.collections.tts.parts.utils.callbacks import LoggingCallback
from nemo.collections.tts.parts.utils.helpers import get_batch_size, get_num_workers
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import AudioSignal, EncodedRepresentation, Index, LengthsType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.optim.lr_scheduler import compute_max_steps, prepare_lr_scheduler
from nemo.utils import model_utils
from nemo.utils.decorators import experimental


@experimental
class AudioCodecModel(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        super().__init__(cfg=cfg, trainer=trainer)

        self.sample_rate = cfg.sample_rate
        self.samples_per_frame = cfg.samples_per_frame

        self.disc_update_prob = cfg.get("disc_update_prob", 1.0)
        self.audio_encoder = instantiate(cfg.audio_encoder)

        # Optionally, add gaussian noise to encoder output as an information bottleneck
        encoder_noise_stdev = cfg.get("encoder_noise_stdev", 0.0)
        if encoder_noise_stdev:
            self.encoder_noise = GaussianDropout(stdev=encoder_noise_stdev)
        else:
            self.encoder_noise = None

        if "vector_quantizer" in cfg:
            self.vector_quantizer = instantiate(cfg.vector_quantizer)
        else:
            self.vector_quantizer = None

        self.audio_decoder = instantiate(cfg.audio_decoder)
        self.discriminator = instantiate(cfg.discriminator)

        mel_loss_dim = cfg.get("mel_loss_dim", 64)
        mel_loss_resolutions = cfg.mel_loss_resolutions
        self.time_domain_loss_scale = cfg.get("time_domain_loss_scale", 1.0)
        self.mel_loss_scale = cfg.get("mel_loss_scale", 1.0)
        mel_loss_l1_scale = cfg.get("mel_loss_l1_scale", 1.0)
        self.gen_loss_scale = cfg.get("gen_loss_scale", 1.0)
        self.feature_loss_scale = cfg.get("feature_loss_scale", 1.0)

        self.time_domain_loss_fn = TimeDomainLoss()
        self.mel_loss_fn = MultiResolutionMelLoss(
            sample_rate=self.sample_rate,
            mel_dim=mel_loss_dim,
            resolutions=mel_loss_resolutions,
            l1_scale=mel_loss_l1_scale,
        )
        self.gen_loss_fn = instantiate(cfg.generator_loss)
        self.disc_loss_fn = instantiate(cfg.discriminator_loss)
        self.feature_loss_fn = RelativeFeatureMatchingLoss()

        self.log_config = cfg.get("log_config", None)
        self.lr_schedule_interval = None
        self.automatic_optimization = False

    @typecheck(
        input_types={
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "encoded": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        },
    )
    def encode_audio(self, audio: torch.Tensor, audio_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        audio, audio_len = self.pad_audio(audio, audio_len)
        encoded, encoded_len = self.audio_encoder(audio=audio, audio_len=audio_len)
        return encoded, encoded_len

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        },
    )
    def decode_audio(self, inputs: torch.Tensor, input_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        audio, audio_len = self.audio_decoder(inputs=inputs, input_len=input_len)
        return audio, audio_len

    @typecheck(
        input_types={
            "encoded": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('N', 'B', 'T_encoded'), Index())},
    )
    def quantize_encode(self, encoded: torch.Tensor, encoded_len: torch.Tensor) -> torch.Tensor:
        if not self.vector_quantizer:
            raise ValueError("Cannot quantize without quantizer")

        indices = self.vector_quantizer.encode(inputs=encoded, input_len=encoded_len)
        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(('N', 'B', 'T_encoded'), Index()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"quantized": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),},
    )
    def quantize_decode(self, indices: torch.Tensor, encoded_len: torch.Tensor) -> torch.Tensor:
        if not self.vector_quantizer:
            raise ValueError("Cannot dequantize without quantizer")

        quantized = self.vector_quantizer.decode(indices=indices, input_len=encoded_len)
        return quantized

    @typecheck(
        input_types={
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "output_audio": NeuralType(('B', 'T_audio'), EncodedRepresentation()),
            "output_audio_len": NeuralType(tuple('B'), LengthsType()),
        },
    )
    def forward(self, audio: torch.Tensor, audio_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        audio, audio_len = self.pad_audio(audio, audio_len)
        encoded, encoded_len = self.encode_audio(audio=audio, audio_len=audio_len)

        if self.vector_quantizer:
            indices = self.quantize_encode(encoded=encoded, encoded_len=encoded_len)
            quantized = self.quantize_decode(indices=indices, encoded_len=encoded_len)
            output_audio, output_audio_len = self.decode_audio(inputs=quantized, input_len=encoded_len)
        else:
            output_audio, output_audio_len = self.decode_audio(inputs=encoded, input_len=encoded_len)

        return output_audio, output_audio_len

    # Zero pad the end of the audio so that we do not have a partial end frame.
    def pad_audio(self, audio, audio_len):
        padded_len = self.samples_per_frame * torch.ceil(audio_len / self.samples_per_frame).int()
        max_len = padded_len.max().item()
        num_padding = max_len - audio.shape[1]
        padded_audio = F.pad(audio, (0, num_padding))
        return padded_audio, padded_len

    def _process_batch(self, batch):
        # [B, T_audio]
        audio = batch.get("audio")
        # [B]
        audio_len = batch.get("audio_lens")
        audio, audio_len = self.pad_audio(audio, audio_len)

        # [B, D, T_encoded]
        encoded, encoded_len = self.audio_encoder(audio=audio, audio_len=audio_len)

        if self.encoder_noise is not None:
            encoded = self.encoder_noise(encoded)

        if self.vector_quantizer:
            encoded, _, commit_loss = self.vector_quantizer(inputs=encoded, input_len=encoded_len)
        else:
            commit_loss = None

        # [B, T]
        audio_gen, audio_gen_len = self.audio_decoder(inputs=encoded, input_len=encoded_len)

        return audio, audio_len, audio_gen, commit_loss

    def training_step(self, batch, batch_idx):
        optim_gen, optim_disc = self.optimizers()
        optim_gen.zero_grad()

        audio, audio_len, audio_gen, commit_loss = self._process_batch(batch)

        if self.disc_update_prob < random.random():
            loss_disc = None
        else:
            # Train discriminator
            optim_disc.zero_grad()

            disc_scores_real, disc_scores_gen, _, _ = self.discriminator(
                audio_real=audio, audio_gen=audio_gen.detach()
            )
            loss_disc = self.disc_loss_fn(disc_scores_real=disc_scores_real, disc_scores_gen=disc_scores_gen)
            train_disc_loss = loss_disc

            self.manual_backward(train_disc_loss)
            optim_disc.step()

        loss_time_domain = self.time_domain_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
        train_loss_time_domain = self.time_domain_loss_scale * loss_time_domain

        loss_mel = self.mel_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
        train_loss_mel = self.mel_loss_scale * loss_mel

        _, disc_scores_gen, fmaps_real, fmaps_gen = self.discriminator(audio_real=audio, audio_gen=audio_gen)

        loss_gen = self.gen_loss_fn(disc_scores_gen=disc_scores_gen)
        train_loss_gen = self.gen_loss_scale * loss_gen

        loss_feature = self.feature_loss_fn(fmaps_real=fmaps_real, fmaps_gen=fmaps_gen)
        train_loss_feature = self.feature_loss_scale * loss_feature

        loss_gen_all = train_loss_time_domain + train_loss_mel + train_loss_gen + train_loss_feature
        if commit_loss is not None:
            loss_gen_all += commit_loss

        self.manual_backward(loss_gen_all)
        optim_gen.step()

        self.update_lr()

        metrics = {
            "g_loss_time_domain": loss_time_domain,
            "g_loss_mel": loss_mel,
            "g_loss_gen": loss_gen,
            "g_loss_feature": loss_feature,
            "g_loss": loss_gen_all,
            "global_step": self.global_step,
            "lr": optim_gen.param_groups[0]['lr'],
        }

        if loss_disc is not None:
            metrics["d_loss"] = loss_disc

        if commit_loss is not None:
            metrics["g_loss_commit"] = commit_loss

        self.log_dict(metrics, on_step=True, sync_dist=True)
        self.log("t_loss", train_loss_mel, prog_bar=True, logger=False, sync_dist=True)

    def training_epoch_end(self, outputs):
        self.update_lr("epoch")

    def validation_step(self, batch, batch_idx):
        audio, audio_len, audio_gen, _ = self._process_batch(batch)
        loss_audio = self.time_domain_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
        loss_mel = self.mel_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
        metrics = {"val_loss": loss_audio + loss_mel, "val_loss_audio": loss_audio, "val_loss_mel": loss_mel}
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

    @staticmethod
    def _setup_train_dataloader(cfg):
        dataset = instantiate(cfg.dataset)
        sampler = dataset.get_sampler(cfg.dataloader_params.batch_size)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, sampler=sampler, **cfg.dataloader_params
        )
        return data_loader

    @staticmethod
    def _setup_test_dataloader(cfg):
        dataset = instantiate(cfg.dataset)
        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)
        return data_loader

    def setup_training_data(self, cfg):
        self._train_dl = self._setup_train_dataloader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_test_dataloader(cfg)

    def setup_test_data(self, cfg):
        pass

    @property
    def max_steps(self):
        if "max_steps" in self._cfg:
            return self._cfg.get("max_steps")

        if "max_epochs" not in self._cfg:
            raise ValueError("Must specify 'max_steps' or 'max_epochs'.")

        if "steps_per_epoch" in self._cfg:
            return self._cfg.max_epochs * self._cfg.steps_per_epoch

        return compute_max_steps(
            max_epochs=self._cfg.max_epochs,
            accumulate_grad_batches=self.trainer.accumulate_grad_batches,
            limit_train_batches=self.trainer.limit_train_batches,
            num_workers=get_num_workers(self.trainer),
            num_samples=len(self._train_dl.dataset),
            batch_size=get_batch_size(self._train_dl),
            drop_last=self._train_dl.drop_last,
        )

    def configure_optimizers(self):
        optim_config = self._cfg.optim.copy()

        OmegaConf.set_struct(optim_config, False)
        sched_config = optim_config.pop("sched", None)
        OmegaConf.set_struct(optim_config, True)

        gen_params = itertools.chain(self.audio_encoder.parameters(), self.audio_decoder.parameters())
        disc_params = self.discriminator.parameters()
        optim_g = instantiate(optim_config, params=gen_params)
        optim_d = instantiate(optim_config, params=disc_params)

        if sched_config is None:
            return [optim_g, optim_d]

        OmegaConf.set_struct(sched_config, False)
        sched_config["max_steps"] = self.max_steps
        OmegaConf.set_struct(sched_config, True)

        scheduler_g = prepare_lr_scheduler(
            optimizer=optim_g, scheduler_config=sched_config, train_dataloader=self._train_dl
        )

        scheduler_d = prepare_lr_scheduler(
            optimizer=optim_d, scheduler_config=sched_config, train_dataloader=self._train_dl
        )

        self.lr_schedule_interval = scheduler_g["interval"]

        return [optim_g, optim_d], [scheduler_g, scheduler_d]

    def update_lr(self, interval="step"):
        schedulers = self.lr_schedulers()
        if schedulers is not None and self.lr_schedule_interval == interval:
            sch1, sch2 = schedulers
            sch1.step()
            sch2.step()

    def configure_callbacks(self):
        if not self.log_config:
            return []

        data_loader = self._setup_test_dataloader(self.log_config)
        generators = instantiate(self.log_config.generators)
        log_dir = Path(self.log_config.log_dir) if self.log_config.log_dir else None
        log_callback = LoggingCallback(
            generators=generators,
            data_loader=data_loader,
            log_epochs=self.log_config.log_epochs,
            epoch_frequency=self.log_config.epoch_frequency,
            output_dir=log_dir,
            loggers=self.trainer.loggers,
            log_tensorboard=self.log_config.log_tensorboard,
            log_wandb=self.log_config.log_wandb,
        )

        return [log_callback]

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []
