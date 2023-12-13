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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.losses.fastpitchloss import DurationLoss, EnergyLoss, MelLoss, PitchLoss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.fastpitch import FastPitchModule
from nemo.collections.tts.parts.mixins import FastPitchAdapterModelMixin
from nemo.collections.tts.parts.utils.callbacks import LoggingCallback
from nemo.collections.tts.parts.utils.helpers import (
    batch_from_ragged,
    g2p_backward_compatible_support,
    plot_alignment_to_numpy,
    plot_spectrogram_to_numpy,
    process_batch,
    sample_tts_input,
)
from nemo.core.classes import Exportable
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    Index,
    LengthsType,
    MelSpectrogramType,
    ProbsType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging, model_utils


@dataclass
class G2PConfig:
    _target_: str = "nemo.collections.tts.g2p.models.en_us_arpabet.EnglishG2p"
    phoneme_dict: str = "scripts/tts_dataset_files/cmudict-0.7b_nv22.10"
    heteronyms: str = "scripts/tts_dataset_files/heteronyms-052722"
    phoneme_probability: float = 0.5


@dataclass
class TextTokenizer:
    _target_: str = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.EnglishPhonemesTokenizer"
    punct: bool = True
    stresses: bool = True
    chars: bool = True
    apostrophe: bool = True
    pad_with_space: bool = True
    add_blank_at: bool = True
    g2p: G2PConfig = field(default_factory=lambda: G2PConfig())


@dataclass
class TextTokenizerConfig:
    text_tokenizer: TextTokenizer = field(default_factory=lambda: TextTokenizer())


class FastPitchModel(SpectrogramGenerator, Exportable, FastPitchAdapterModelMixin):
    """FastPitch model (https://arxiv.org/abs/2006.06873) that is used to generate mel spectrogram from text."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Setup normalizer
        self.normalizer = None
        self.text_normalizer_call = None
        self.text_normalizer_call_kwargs = {}
        self._setup_normalizer(cfg)

        self.learn_alignment = cfg.get("learn_alignment", False)

        # Setup vocabulary (=tokenizer) and input_fft_kwargs (supported only with self.learn_alignment=True)
        input_fft_kwargs = {}
        if self.learn_alignment:
            self.vocab = None

            self.ds_class = cfg.train_ds.dataset._target_
            self.ds_class_name = self.ds_class.split(".")[-1]
            if not self.ds_class in [
                "nemo.collections.tts.data.dataset.TTSDataset",
                "nemo.collections.tts.data.text_to_speech_dataset.TextToSpeechDataset",
                "nemo.collections.tts.torch.data.TTSDataset",
            ]:
                raise ValueError(f"Unknown dataset class: {self.ds_class}.")

            self._setup_tokenizer(cfg)
            assert self.vocab is not None
            input_fft_kwargs["n_embed"] = len(self.vocab.tokens)
            input_fft_kwargs["padding_idx"] = self.vocab.pad

        self._parser = None
        self._tb_logger = None
        super().__init__(cfg=cfg, trainer=trainer)

        self.bin_loss_warmup_epochs = cfg.get("bin_loss_warmup_epochs", 100)
        self.log_images = cfg.get("log_images", False)
        self.log_train_images = False

        default_prosody_loss_scale = 0.1 if self.learn_alignment else 1.0
        dur_loss_scale = cfg.get("dur_loss_scale", default_prosody_loss_scale)
        pitch_loss_scale = cfg.get("pitch_loss_scale", default_prosody_loss_scale)
        energy_loss_scale = cfg.get("energy_loss_scale", default_prosody_loss_scale)

        self.mel_loss_fn = MelLoss()
        self.pitch_loss_fn = PitchLoss(loss_scale=pitch_loss_scale)
        self.duration_loss_fn = DurationLoss(loss_scale=dur_loss_scale)
        self.energy_loss_fn = EnergyLoss(loss_scale=energy_loss_scale)

        self.aligner = None
        if self.learn_alignment:
            aligner_loss_scale = cfg.get("aligner_loss_scale", 1.0)
            self.aligner = instantiate(self._cfg.alignment_module)
            self.forward_sum_loss_fn = ForwardSumLoss(loss_scale=aligner_loss_scale)
            self.bin_loss_fn = BinLoss(loss_scale=aligner_loss_scale)

        self.preprocessor = instantiate(self._cfg.preprocessor)
        input_fft = instantiate(self._cfg.input_fft, **input_fft_kwargs)
        output_fft = instantiate(self._cfg.output_fft)
        duration_predictor = instantiate(self._cfg.duration_predictor)
        pitch_predictor = instantiate(self._cfg.pitch_predictor)
        speaker_encoder = instantiate(self._cfg.get("speaker_encoder", None))
        energy_embedding_kernel_size = cfg.get("energy_embedding_kernel_size", 0)
        energy_predictor = instantiate(self._cfg.get("energy_predictor", None))

        # [TODO] may remove if we change the pre-trained config
        # cfg: condition_types = [ "add" ]
        n_speakers = cfg.get("n_speakers", 0)
        speaker_emb_condition_prosody = cfg.get("speaker_emb_condition_prosody", False)
        speaker_emb_condition_decoder = cfg.get("speaker_emb_condition_decoder", False)
        speaker_emb_condition_aligner = cfg.get("speaker_emb_condition_aligner", False)
        min_token_duration = cfg.get("min_token_duration", 0)
        use_log_energy = cfg.get("use_log_energy", True)
        if n_speakers > 1 and "add" not in input_fft.cond_input.condition_types:
            input_fft.cond_input.condition_types.append("add")
        if speaker_emb_condition_prosody:
            duration_predictor.cond_input.condition_types.append("add")
            pitch_predictor.cond_input.condition_types.append("add")
        if speaker_emb_condition_decoder:
            output_fft.cond_input.condition_types.append("add")
        if speaker_emb_condition_aligner and self.aligner is not None:
            self.aligner.cond_input.condition_types.append("add")

        self.fastpitch = FastPitchModule(
            input_fft,
            output_fft,
            duration_predictor,
            pitch_predictor,
            energy_predictor,
            self.aligner,
            speaker_encoder,
            n_speakers,
            cfg.symbols_embedding_dim,
            cfg.pitch_embedding_kernel_size,
            energy_embedding_kernel_size,
            cfg.n_mel_channels,
            min_token_duration,
            cfg.max_token_duration,
            use_log_energy,
        )
        self._input_types = self._output_types = None
        self.export_config = {
            "emb_range": (0, self.fastpitch.encoder.word_emb.num_embeddings),
            "enable_volume": False,
            "enable_ragged_batches": False,
        }
        if self.fastpitch.speaker_emb is not None:
            self.export_config["num_speakers"] = cfg.n_speakers

        self.log_config = cfg.get("log_config", None)

        # Adapter modules setup (from FastPitchAdapterModelMixin)
        self.setup_adapters()

    def _get_default_text_tokenizer_conf(self):
        text_tokenizer: TextTokenizerConfig = TextTokenizerConfig()
        return OmegaConf.create(OmegaConf.to_yaml(text_tokenizer))

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
                    'text_tokenizer.g2p.phoneme_dict', cfg.text_tokenizer.g2p.phoneme_dict,
                )

            if "heteronyms" in cfg.text_tokenizer.g2p:
                g2p_kwargs["heteronyms"] = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms', cfg.text_tokenizer.g2p.heteronyms,
                )

            # for backward compatability
            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p, **g2p_kwargs)

        # TODO @xueyang: rename the instance of tokenizer because vocab is misleading.
        self.vocab = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            tb_logger = self.logger.experiment
            for logger in self.trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    tb_logger = logger.experiment
                    break
            self._tb_logger = tb_logger
        return self._tb_logger

    @property
    def parser(self):
        if self._parser is not None:
            return self._parser

        if self.learn_alignment:
            self._parser = self.vocab.encode
        else:
            self._parser = parsers.make_parser(
                labels=self._cfg.labels,
                name='en',
                unk_id=-1,
                blank_id=-1,
                do_normalize=True,
                abbreviation_version="fastpitch",
                make_table=False,
            )
        return self._parser

    def parse(self, str_input: str, normalize=True) -> torch.tensor:
        if self.training:
            logging.warning("parse() is meant to be called in eval mode.")

        if normalize and self.text_normalizer_call is not None:
            str_input = self.text_normalizer_call(str_input, **self.text_normalizer_call_kwargs)

        if self.learn_alignment:
            eval_phon_mode = contextlib.nullcontext()
            if hasattr(self.vocab, "set_phone_prob"):
                eval_phon_mode = self.vocab.set_phone_prob(prob=1.0)

            # Disable mixed g2p representation if necessary
            with eval_phon_mode:
                tokens = self.parser(str_input)
        else:
            tokens = self.parser(str_input)

        x = torch.tensor(tokens).unsqueeze_(0).long().to(self.device)
        return x

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
            "energy": NeuralType(('B', 'T_audio'), RegressionValuesType(), optional=True),
            "speaker": NeuralType(('B'), Index(), optional=True),
            "pace": NeuralType(optional=True),
            "spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType(), optional=True),
            "attn_prior": NeuralType(('B', 'T_spec', 'T_text'), ProbsType(), optional=True),
            "mel_lens": NeuralType(('B'), LengthsType(), optional=True),
            "input_lens": NeuralType(('B'), LengthsType(), optional=True),
            # reference_* data is used for multi-speaker FastPitch training
            "reference_spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType(), optional=True),
            "reference_spec_lens": NeuralType(('B'), LengthsType(), optional=True),
        }
    )
    def forward(
        self,
        *,
        text,
        durs=None,
        pitch=None,
        energy=None,
        speaker=None,
        pace=1.0,
        spec=None,
        attn_prior=None,
        mel_lens=None,
        input_lens=None,
        reference_spec=None,
        reference_spec_lens=None,
    ):
        return self.fastpitch(
            text=text,
            durs=durs,
            pitch=pitch,
            energy=energy,
            speaker=speaker,
            pace=pace,
            spec=spec,
            attn_prior=attn_prior,
            mel_lens=mel_lens,
            input_lens=input_lens,
            reference_spec=reference_spec,
            reference_spec_lens=reference_spec_lens,
        )

    @typecheck(output_types={"spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType())})
    def generate_spectrogram(
        self,
        tokens: 'torch.tensor',
        speaker: Optional[int] = None,
        pace: float = 1.0,
        reference_spec: Optional['torch.tensor'] = None,
        reference_spec_lens: Optional['torch.tensor'] = None,
    ) -> torch.tensor:
        if self.training:
            logging.warning("generate_spectrogram() is meant to be called in eval mode.")
        if isinstance(speaker, int):
            speaker = torch.tensor([speaker]).to(self.device)
        spect, *_ = self(
            text=tokens,
            durs=None,
            pitch=None,
            speaker=speaker,
            pace=pace,
            reference_spec=reference_spec,
            reference_spec_lens=reference_spec_lens,
        )
        return spect

    def training_step(self, batch, batch_idx):
        attn_prior, durs, speaker, energy, reference_audio, reference_audio_len = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if self.learn_alignment:
            if self.ds_class == "nemo.collections.tts.data.text_to_speech_dataset.TextToSpeechDataset":
                batch_dict = batch
            else:
                batch_dict = process_batch(batch, self._train_dl.dataset.sup_data_types_set)
            audio = batch_dict.get("audio")
            audio_lens = batch_dict.get("audio_lens")
            text = batch_dict.get("text")
            text_lens = batch_dict.get("text_lens")
            attn_prior = batch_dict.get("align_prior_matrix", None)
            pitch = batch_dict.get("pitch", None)
            energy = batch_dict.get("energy", None)
            speaker = batch_dict.get("speaker_id", None)
            reference_audio = batch_dict.get("reference_audio", None)
            reference_audio_len = batch_dict.get("reference_audio_lens", None)
        else:
            audio, audio_lens, text, text_lens, durs, pitch, speaker = batch

        mels, spec_len = self.preprocessor(input_signal=audio, length=audio_lens)
        reference_spec, reference_spec_len = None, None
        if reference_audio is not None:
            reference_spec, reference_spec_len = self.preprocessor(
                input_signal=reference_audio, length=reference_audio_len
            )

        (
            mels_pred,
            _,
            _,
            log_durs_pred,
            pitch_pred,
            attn_soft,
            attn_logprob,
            attn_hard,
            attn_hard_dur,
            pitch,
            energy_pred,
            energy_tgt,
        ) = self(
            text=text,
            durs=durs,
            pitch=pitch,
            energy=energy,
            speaker=speaker,
            pace=1.0,
            spec=mels if self.learn_alignment else None,
            reference_spec=reference_spec,
            reference_spec_lens=reference_spec_len,
            attn_prior=attn_prior,
            mel_lens=spec_len,
            input_lens=text_lens,
        )
        if durs is None:
            durs = attn_hard_dur

        mel_loss = self.mel_loss_fn(spect_predicted=mels_pred, spect_tgt=mels)
        dur_loss = self.duration_loss_fn(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens)
        loss = mel_loss + dur_loss
        if self.learn_alignment:
            ctc_loss = self.forward_sum_loss_fn(attn_logprob=attn_logprob, in_lens=text_lens, out_lens=spec_len)
            bin_loss_weight = min(self.current_epoch / self.bin_loss_warmup_epochs, 1.0) * 1.0
            bin_loss = self.bin_loss_fn(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight
            loss += ctc_loss + bin_loss

        pitch_loss = self.pitch_loss_fn(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
        energy_loss = self.energy_loss_fn(energy_predicted=energy_pred, energy_tgt=energy_tgt, length=text_lens)
        loss += pitch_loss + energy_loss

        self.log("t_loss", loss)
        self.log("t_mel_loss", mel_loss)
        self.log("t_dur_loss", dur_loss)
        self.log("t_pitch_loss", pitch_loss)
        if energy_tgt is not None:
            self.log("t_energy_loss", energy_loss)
        if self.learn_alignment:
            self.log("t_ctc_loss", ctc_loss)
            self.log("t_bin_loss", bin_loss)

        # Log images to tensorboard
        if self.log_images and self.log_train_images and isinstance(self.logger, TensorBoardLogger):
            self.log_train_images = False

            self.tb_logger.add_image(
                "train_mel_target",
                plot_spectrogram_to_numpy(mels[0].data.cpu().float().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = mels_pred[0].data.cpu().float().numpy()
            self.tb_logger.add_image(
                "train_mel_predicted", plot_spectrogram_to_numpy(spec_predict), self.global_step, dataformats="HWC",
            )
            if self.learn_alignment:
                attn = attn_hard[0].data.cpu().float().numpy().squeeze()
                self.tb_logger.add_image(
                    "train_attn", plot_alignment_to_numpy(attn.T), self.global_step, dataformats="HWC",
                )
                soft_attn = attn_soft[0].data.cpu().float().numpy().squeeze()
                self.tb_logger.add_image(
                    "train_soft_attn", plot_alignment_to_numpy(soft_attn.T), self.global_step, dataformats="HWC",
                )

        return loss

    def validation_step(self, batch, batch_idx):
        attn_prior, durs, speaker, energy, reference_audio, reference_audio_len = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if self.learn_alignment:
            if self.ds_class == "nemo.collections.tts.data.text_to_speech_dataset.TextToSpeechDataset":
                batch_dict = batch
            else:
                batch_dict = process_batch(batch, self._train_dl.dataset.sup_data_types_set)
            audio = batch_dict.get("audio")
            audio_lens = batch_dict.get("audio_lens")
            text = batch_dict.get("text")
            text_lens = batch_dict.get("text_lens")
            attn_prior = batch_dict.get("align_prior_matrix", None)
            pitch = batch_dict.get("pitch", None)
            energy = batch_dict.get("energy", None)
            speaker = batch_dict.get("speaker_id", None)
            reference_audio = batch_dict.get("reference_audio", None)
            reference_audio_len = batch_dict.get("reference_audio_lens", None)
        else:
            audio, audio_lens, text, text_lens, durs, pitch, speaker = batch

        mels, mel_lens = self.preprocessor(input_signal=audio, length=audio_lens)
        reference_spec, reference_spec_len = None, None
        if reference_audio is not None:
            reference_spec, reference_spec_len = self.preprocessor(
                input_signal=reference_audio, length=reference_audio_len
            )

        # Calculate val loss on ground truth durations to better align L2 loss in time
        (mels_pred, _, _, log_durs_pred, pitch_pred, _, _, _, attn_hard_dur, pitch, energy_pred, energy_tgt,) = self(
            text=text,
            durs=durs,
            pitch=pitch,
            energy=energy,
            speaker=speaker,
            pace=1.0,
            spec=mels if self.learn_alignment else None,
            reference_spec=reference_spec,
            reference_spec_lens=reference_spec_len,
            attn_prior=attn_prior,
            mel_lens=mel_lens,
            input_lens=text_lens,
        )
        if durs is None:
            durs = attn_hard_dur

        mel_loss = self.mel_loss_fn(spect_predicted=mels_pred, spect_tgt=mels)
        dur_loss = self.duration_loss_fn(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens)
        pitch_loss = self.pitch_loss_fn(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
        energy_loss = self.energy_loss_fn(energy_predicted=energy_pred, energy_tgt=energy_tgt, length=text_lens)
        loss = mel_loss + dur_loss + pitch_loss + energy_loss

        val_outputs = {
            "val_loss": loss,
            "mel_loss": mel_loss,
            "dur_loss": dur_loss,
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss if energy_tgt is not None else None,
            "mel_target": mels if batch_idx == 0 else None,
            "mel_pred": mels_pred if batch_idx == 0 else None,
        }
        self.validation_step_outputs.append(val_outputs)
        return val_outputs

    def on_validation_epoch_end(self):
        collect = lambda key: torch.stack([x[key] for x in self.validation_step_outputs]).mean()
        val_loss = collect("val_loss")
        mel_loss = collect("mel_loss")
        dur_loss = collect("dur_loss")
        pitch_loss = collect("pitch_loss")
        self.log("val_loss", val_loss, sync_dist=True)
        self.log("val_mel_loss", mel_loss, sync_dist=True)
        self.log("val_dur_loss", dur_loss, sync_dist=True)
        self.log("val_pitch_loss", pitch_loss, sync_dist=True)
        if self.validation_step_outputs[0]["energy_loss"] is not None:
            energy_loss = collect("energy_loss")
            self.log("val_energy_loss", energy_loss, sync_dist=True)

        _, _, _, _, _, spec_target, spec_predict = self.validation_step_outputs[0].values()

        if self.log_images and isinstance(self.logger, TensorBoardLogger):
            self.tb_logger.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(spec_target[0].data.cpu().float().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = spec_predict[0].data.cpu().float().numpy()
            self.tb_logger.add_image(
                "val_mel_predicted", plot_spectrogram_to_numpy(spec_predict), self.global_step, dataformats="HWC",
            )
            self.log_train_images = True
        self.validation_step_outputs.clear()  # free memory)

    def _setup_train_dataloader(self, cfg):
        phon_mode = contextlib.nullcontext()
        if hasattr(self.vocab, "set_phone_prob"):
            phon_mode = self.vocab.set_phone_prob(self.vocab.phoneme_probability)

        with phon_mode:
            dataset = instantiate(cfg.dataset, text_tokenizer=self.vocab,)

        sampler = dataset.get_sampler(cfg.dataloader_params.batch_size, world_size=self.trainer.world_size)
        return torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, sampler=sampler, **cfg.dataloader_params
        )

    def _setup_test_dataloader(self, cfg):
        phon_mode = contextlib.nullcontext()
        if hasattr(self.vocab, "set_phone_prob"):
            phon_mode = self.vocab.set_phone_prob(0.0)

        with phon_mode:
            dataset = instantiate(cfg.dataset, text_tokenizer=self.vocab,)

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloader_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg.dataloader_params):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        if self.ds_class == "nemo.collections.tts.data.dataset.TTSDataset":
            phon_mode = contextlib.nullcontext()
            if hasattr(self.vocab, "set_phone_prob"):
                phon_mode = self.vocab.set_phone_prob(prob=None if name == "val" else self.vocab.phoneme_probability)

            with phon_mode:
                dataset = instantiate(
                    cfg.dataset,
                    text_normalizer=self.normalizer,
                    text_normalizer_call_kwargs=self.text_normalizer_call_kwargs,
                    text_tokenizer=self.vocab,
                )
        else:
            dataset = instantiate(cfg.dataset)

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        if self.ds_class == "nemo.collections.tts.data.text_to_speech_dataset.TextToSpeechDataset":
            self._train_dl = self._setup_train_dataloader(cfg)
        else:
            self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        if self.ds_class == "nemo.collections.tts.data.text_to_speech_dataset.TextToSpeechDataset":
            self._validation_dl = self._setup_test_dataloader(cfg)
        else:
            self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="val")

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    def configure_callbacks(self):
        if not self.log_config:
            return []

        sample_ds_class = self.log_config.dataset._target_
        if sample_ds_class != "nemo.collections.tts.data.text_to_speech_dataset.TextToSpeechDataset":
            raise ValueError(f"Logging callback only supported for TextToSpeechDataset, got {sample_ds_class}")

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
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []

        # en-US, single speaker, 22050Hz, LJSpeech (ARPABET).
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_fastpitch",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_fastpitch/versions/1.8.1/files/tts_en_fastpitch_align.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz with and can be used to generate female English voices with an American accent. It is ARPABET-based.",
            class_=cls,
        )
        list_of_models.append(model)

        # en-US, single speaker, 22050Hz, LJSpeech (IPA).
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_fastpitch_ipa",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_fastpitch/versions/IPA_1.13.0/files/tts_en_fastpitch_align_ipa.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz with and can be used to generate female English voices with an American accent. It is IPA-based.",
            class_=cls,
        )
        list_of_models.append(model)

        # en-US, multi-speaker, 44100Hz, HiFiTTS.
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_fastpitch_multispeaker",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_multispeaker_fastpitchhifigan/versions/1.10.0/files/tts_en_fastpitch_multispeaker.nemo",
            description="This model is trained on HiFITTS sampled at 44100Hz with and can be used to generate male and female English voices with an American accent.",
            class_=cls,
        )
        list_of_models.append(model)

        # de-DE, single male speaker, grapheme-based tokenizer, 22050 Hz, Thorsten Müller’s German Neutral-TTS Dataset, 21.02
        model = PretrainedModelInfo(
            pretrained_model_name="tts_de_fastpitch_singleSpeaker_thorstenNeutral_2102",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_de_fastpitchhifigan/versions/1.15.0/files/tts_de_fastpitch_thorstens2102.nemo",
            description="This model is trained on a single male speaker data in Thorsten Müller’s German Neutral 21.02 Dataset sampled at 22050Hz and can be used to generate male German voices.",
            class_=cls,
        )
        list_of_models.append(model)

        # de-DE, single male speaker, grapheme-based tokenizer, 22050 Hz, Thorsten Müller’s German Neutral-TTS Dataset, 22.10
        model = PretrainedModelInfo(
            pretrained_model_name="tts_de_fastpitch_singleSpeaker_thorstenNeutral_2210",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_de_fastpitchhifigan/versions/1.15.0/files/tts_de_fastpitch_thorstens2210.nemo",
            description="This model is trained on a single male speaker data in Thorsten Müller’s German Neutral 22.10 Dataset sampled at 22050Hz and can be used to generate male German voices.",
            class_=cls,
        )
        list_of_models.append(model)

        # de-DE, multi-speaker, 5 speakers, 44100 Hz, HUI-Audio-Corpus-German Clean.
        model = PretrainedModelInfo(
            pretrained_model_name="tts_de_fastpitch_multispeaker_5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_de_fastpitch_multispeaker_5/versions/1.11.0/files/tts_de_fastpitch_multispeaker_5.nemo",
            description="This model is trained on 5 speakers in HUI-Audio-Corpus-German clean subset sampled at 44100Hz with and can be used to generate male and female German voices.",
            class_=cls,
        )
        list_of_models.append(model)

        # es, 174 speakers, 44100Hz, OpenSLR (IPA)
        model = PretrainedModelInfo(
            pretrained_model_name="tts_es_fastpitch_multispeaker",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_es_multispeaker_fastpitchhifigan/versions/1.15.0/files/tts_es_fastpitch_multispeaker.nemo",
            description="This model is trained on 174 speakers in 6 crowdsourced Latin American Spanish OpenSLR datasets sampled at 44100Hz and can be used to generate male and female Spanish voices with Latin American accents.",
            class_=cls,
        )
        list_of_models.append(model)

        # zh, single female speaker, 22050Hz, SFSpeech Bilingual Chinese/English dataset, improved model using richer
        # dict and jieba word segmenter for polyphone disambiguation.
        model = PretrainedModelInfo(
            pretrained_model_name="tts_zh_fastpitch_sfspeech",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_zh_fastpitch_hifigan_sfspeech/versions/1.15.0/files/tts_zh_fastpitch_sfspeech.nemo",
            description="This model is trained on a single female speaker in SFSpeech Bilingual Chinese/English dataset"
            " sampled at 22050Hz and can be used to generate female Mandarin Chinese voices. It is improved"
            " using richer dict and jieba word segmenter for polyphone disambiguation.",
            class_=cls,
        )
        list_of_models.append(model)

        # en, multi speaker, LibriTTS, 16000 Hz
        # stft 25ms 10ms matching ASR params
        # for use during Enhlish ASR training/adaptation
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_fastpitch_for_asr_finetuning",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_fastpitch_spectrogram_enhancer_for_asr_finetuning/versions/1.20.0/files/tts_en_fastpitch_for_asr_finetuning.nemo",
            description="This model is trained on LibriSpeech, train-960 subset."
            " STFT parameters follow those commonly used in ASR: 25 ms window, 10 ms hop."
            " This model is supposed to be used with its companion SpetrogramEnhancer for "
            " ASR fine-tuning. Usage for regular TTS tasks is not advised.",
            class_=cls,
        )
        list_of_models.append(model)

        return list_of_models

    # Methods for model exportability
    def _prepare_for_export(self, **kwargs):
        super()._prepare_for_export(**kwargs)

        tensor_shape = ('T') if self.export_config["enable_ragged_batches"] else ('B', 'T')

        # Define input_types and output_types as required by export()
        self._input_types = {
            "text": NeuralType(tensor_shape, TokenIndex()),
            "pitch": NeuralType(tensor_shape, RegressionValuesType()),
            "pace": NeuralType(tensor_shape),
            "volume": NeuralType(tensor_shape, optional=True),
            "batch_lengths": NeuralType(('B'), optional=True),
            "speaker": NeuralType(('B'), Index(), optional=True),
        }
        self._output_types = {
            "spect": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "num_frames": NeuralType(('B'), TokenDurationType()),
            "durs_predicted": NeuralType(('B', 'T'), TokenDurationType()),
            "log_durs_predicted": NeuralType(('B', 'T'), TokenLogDurationType()),
            "pitch_predicted": NeuralType(('B', 'T'), RegressionValuesType()),
        }
        if self.export_config["enable_volume"]:
            self._output_types["volume_aligned"] = NeuralType(('B', 'T'), RegressionValuesType())

    def _export_teardown(self):
        self._input_types = self._output_types = None

    @property
    def disabled_deployment_input_names(self):
        """Implement this method to return a set of input names disabled for export"""
        disabled_inputs = set()
        if self.fastpitch.speaker_emb is None:
            disabled_inputs.add("speaker")
        if not self.export_config["enable_ragged_batches"]:
            disabled_inputs.add("batch_lengths")
        if not self.export_config["enable_volume"]:
            disabled_inputs.add("volume")
        return disabled_inputs

    @property
    def input_types(self):
        return self._input_types

    @property
    def output_types(self):
        return self._output_types

    def input_example(self, max_batch=1, max_dim=44):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        par = next(self.fastpitch.parameters())
        inputs = sample_tts_input(self.export_config, par.device, max_batch=max_batch, max_dim=max_dim)
        if 'enable_ragged_batches' not in self.export_config:
            inputs.pop('batch_lengths', None)
        return (inputs,)

    def forward_for_export(self, text, pitch, pace, volume=None, batch_lengths=None, speaker=None):
        if self.export_config["enable_ragged_batches"]:
            text, pitch, pace, volume_tensor, lens = batch_from_ragged(
                text, pitch, pace, batch_lengths, padding_idx=self.fastpitch.encoder.padding_idx, volume=volume
            )
            if volume is not None:
                volume = volume_tensor
        return self.fastpitch.infer(text=text, pitch=pitch, pace=pace, volume=volume, speaker=speaker)

    def interpolate_speaker(
        self, original_speaker_1, original_speaker_2, weight_speaker_1, weight_speaker_2, new_speaker_id
    ):
        """
        This method performs speaker interpolation between two original speakers the model is trained on.

        Inputs:
            original_speaker_1: Integer speaker ID of first existing speaker in the model
            original_speaker_2: Integer speaker ID of second existing speaker in the model
            weight_speaker_1: Floating point weight associated in to first speaker during weight combination
            weight_speaker_2: Floating point weight associated in to second speaker during weight combination
            new_speaker_id: Integer speaker ID of new interpolated speaker in the model
        """
        if self.fastpitch.speaker_emb is None:
            raise Exception(
                "Current FastPitch model is not a multi-speaker FastPitch model. Speaker interpolation can only \
                be performed with a multi-speaker model"
            )
        n_speakers = self.fastpitch.speaker_emb.weight.data.size()[0]
        if original_speaker_1 >= n_speakers or original_speaker_2 >= n_speakers or new_speaker_id >= n_speakers:
            raise Exception(
                f"Parameters original_speaker_1, original_speaker_2, new_speaker_id should be less than the total \
                total number of speakers FastPitch was trained on (n_speakers = {n_speakers})."
            )
        speaker_emb_1 = (
            self.fastpitch.speaker_emb(torch.tensor(original_speaker_1, dtype=torch.int32).cuda()).clone().detach()
        )
        speaker_emb_2 = (
            self.fastpitch.speaker_emb(torch.tensor(original_speaker_2, dtype=torch.int32).cuda()).clone().detach()
        )
        new_speaker_emb = weight_speaker_1 * speaker_emb_1 + weight_speaker_2 * speaker_emb_2
        self.fastpitch.speaker_emb.weight.data[new_speaker_id] = new_speaker_emb
