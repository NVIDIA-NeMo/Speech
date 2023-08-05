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

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer
from nemo.collections.tts.losses.radttsloss import AttentionBinarizationLoss, RADTTSLoss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.parts.utils.helpers import (
    batch_from_ragged,
    g2p_backward_compatible_support,
    plot_alignment_to_numpy,
    regulate_len,
    sample_tts_input,
)
from nemo.core.classes import Exportable
from nemo.core.classes.common import typecheck
from nemo.core.neural_types.elements import (
    Index,
    MelSpectrogramType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.optim.radam import RAdam
from nemo.utils import logging
from nemo.utils.decorators import experimental


@experimental
class RadTTSModel(SpectrogramGenerator, Exportable):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        self.normalizer = None
        self.text_normalizer_call = None
        self.text_normalizer_call_kwargs = {}
        self._setup_normalizer(cfg)

        self.tokenizer = None
        self._setup_tokenizer(cfg)

        assert self.tokenizer is not None

        self.tokenizer_pad = self.tokenizer.pad
        self.tokenizer_unk = self.tokenizer.oov

        self.text_tokenizer_pad_id = None
        self.tokens = None

        super().__init__(cfg=cfg, trainer=trainer)
        self.feat_loss_weight = 1.0
        self.model_config = cfg.modelConfig
        self.train_config = cfg.trainerConfig
        self.optim = cfg.optim
        self.criterion = RADTTSLoss(
            self.train_config.sigma,
            self.model_config.n_group_size,
            self.model_config.dur_model_config,
            self.model_config.f0_model_config,
            self.model_config.energy_model_config,
            vpred_model_config=self.model_config.v_model_config,
            loss_weights=self.train_config.loss_weights,
        )

        self.attention_kl_loss = AttentionBinarizationLoss()
        self.model = instantiate(cfg.modelConfig)
        self._parser = None
        self._tb_logger = None
        self.cfg = cfg
        self.log_train_images = False
        self.export_config = {
            "emb_range": (0, self.model.embedding.num_embeddings),
            "enable_volume": True,
            "enable_ragged_batches": False,
            "num_speakers": self.model_config.n_speakers,
        }
        # print("intial self normalizer", self.normalizer)

    def batch_dict(self, batch_data):
        if len(batch_data) < 14:
            spk_id = torch.tensor([0] * (batch_data[3]).size(0)).cuda().to(self.device)
            v_m = batch_data[9]
            p_v = batch_data[10]
        else:
            spk_id = batch_data[13]
            v_m = batch_data[9]
            p_v = batch_data[10]
        batch_data_dict = {
            "audio": batch_data[0],
            "audio_lens": batch_data[1],
            "text": batch_data[2],
            "text_lens": batch_data[3],
            "log_mel": batch_data[4],
            "log_mel_lens": batch_data[5],
            "align_prior_matrix": batch_data[6],
            "pitch": batch_data[7],
            "pitch_lens": batch_data[8],
            "voiced_mask": v_m,
            "p_voiced": p_v,
            "energy": batch_data[11],
            "energy_lens": batch_data[12],
            "speaker_id": spk_id,
        }
        return batch_data_dict

    def training_step(self, batch, batch_idx):
        batch = self.batch_dict(batch)
        mel = batch['log_mel']
        speaker_ids = batch['speaker_id']
        text = batch['text']
        in_lens = batch['text_lens']
        out_lens = batch['log_mel_lens']
        attn_prior = batch['align_prior_matrix']
        f0 = batch['pitch']
        voiced_mask = batch['voiced_mask']
        energy_avg = batch['energy']

        if (
            self.train_config.binarization_start_iter >= 0
            and self.global_step >= self.train_config.binarization_start_iter
        ):
            # binarization training phase
            binarize = True
        else:
            # no binarization, soft-only
            binarize = False

        outputs = self.model(
            mel,
            speaker_ids,
            text,
            in_lens,
            out_lens,
            binarize_attention=binarize,
            attn_prior=attn_prior,
            f0=f0,
            energy_avg=energy_avg,
            voiced_mask=voiced_mask,
        )
        loss_outputs = self.criterion(outputs, in_lens, out_lens)

        loss = None
        for k, (v, w) in loss_outputs.items():
            if w > 0:
                loss = v * w if loss is None else loss + v * w

        if binarize and self.global_step >= self.train_config.kl_loss_start_iter:
            binarization_loss = self.attention_kl_loss(outputs['attn'], outputs['attn_soft'])
            loss += binarization_loss
        else:
            binarization_loss = torch.zeros_like(loss)
        loss_outputs['binarization_loss'] = (binarization_loss, 1.0)

        for k, (v, w) in loss_outputs.items():
            self.log("train/" + k, loss_outputs[k][0], on_step=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        batch = self.batch_dict(batch)
        speaker_ids = batch['speaker_id']
        text = batch['text']
        in_lens = batch['text_lens']
        out_lens = batch['log_mel_lens']
        attn_prior = batch['align_prior_matrix']
        f0 = batch['pitch']
        voiced_mask = batch['voiced_mask']
        energy_avg = batch['energy']
        mel = batch['log_mel']
        if (
            self.train_config.binarization_start_iter >= 0
            and self.global_step >= self.train_config.binarization_start_iter
        ):
            # binarization training phase
            binarize = True
        else:
            # no binarization, soft-only
            binarize = False
        outputs = self.model(
            mel,
            speaker_ids,
            text,
            in_lens,
            out_lens,
            binarize_attention=True,
            attn_prior=attn_prior,
            f0=f0,
            energy_avg=energy_avg,
            voiced_mask=voiced_mask,
        )
        loss_outputs = self.criterion(outputs, in_lens, out_lens)

        loss = None
        for k, (v, w) in loss_outputs.items():
            if w > 0:
                loss = v * w if loss is None else loss + v * w

        if (
            binarize
            and self.train_config.kl_loss_start_iter >= 0
            and self.global_step >= self.train_config.kl_loss_start_iter
        ):
            binarization_loss = self.attention_kl_loss(outputs['attn'], outputs['attn_soft'])
            loss += binarization_loss
        else:
            binarization_loss = torch.zeros_like(loss)
        loss_outputs['binarization_loss'] = binarization_loss

        val_outputs = {
            "loss_outputs": loss_outputs,
            "attn": outputs["attn"] if batch_idx == 0 else None,
            "attn_soft": outputs["attn_soft"] if batch_idx == 0 else None,
            "audiopaths": "audio_1" if batch_idx == 0 else None,
        }
        self.validation_step_outputs.append(val_outputs)
        return val_outputs

    def on_validation_epoch_end(self):

        loss_outputs = self.validation_step_outputs[0]["loss_outputs"]

        for k, v in loss_outputs.items():
            if k != "binarization_loss":
                self.log("val/" + k, loss_outputs[k][0], sync_dist=True, on_epoch=True)

        attn = self.validation_step_outputs[0]["attn"]
        attn_soft = self.validation_step_outputs[0]["attn_soft"]

        self.tb_logger.add_image(
            'attention_weights_mas',
            plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy().T, title="audio"),
            self.global_step,
            dataformats='HWC',
        )

        self.tb_logger.add_image(
            'attention_weights',
            plot_alignment_to_numpy(attn_soft[0, 0].data.cpu().numpy().T, title="audio"),
            self.global_step,
            dataformats='HWC',
        )
        self.log_train_images = True
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        logging.info("Initializing %s optimizer" % (self.optim.name))
        if len(self.train_config.finetune_layers):
            for name, param in model.named_parameters():
                if any([l in name for l in self.train_config.finetune_layers]):  # short list hack
                    logging.info("Fine-tuning parameter", name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if self.optim.name == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.optim.lr, weight_decay=self.optim.weight_decay
            )
        elif self.optim.name == 'RAdam':  # False for inference riva
            optimizer = RAdam(self.model.parameters(), lr=self.optim.lr, weight_decay=self.optim.weight_decay)
        else:
            logging.info("Unrecognized optimizer %s! Please choose the right optimizer" % (self.optim.name))
            exit(1)

        return optimizer

    def _loader(self, cfg):
        try:
            _ = cfg.dataset.manifest_filepath
        except omegaconf.errors.MissingMandatoryValue:
            logging.warning("manifest_filepath was skipped. No dataset for this model.")
            return None
        # print("inside loader self normalizer", self.normalizer)
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

    @typecheck(
        input_types={
            "tokens": NeuralType(('B', 'T_text'), TokenIndex(), optional=True),
            "speaker": NeuralType(('B'), Index(), optional=True),
            "sigma": NeuralType(optional=True),
        },
        output_types={"spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),},
    )
    def generate_spectrogram(self, tokens: 'torch.tensor', speaker: int = 0, sigma: float = 1.0) -> torch.tensor:
        self.eval()
        if self.training:
            logging.warning("generate_spectrogram() is meant to be called in eval mode.")
        speaker = torch.tensor([speaker]).long().cuda().to(self.device)
        outputs = self.model.infer(speaker, tokens, sigma=sigma)

        spect = outputs['mel']
        return spect

    @property
    def parser(self):
        if self._parser is not None:
            return self._parser
        return self._parser

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

            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p, **g2p_kwargs)

        self.tokenizer = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)
        if isinstance(self.tokenizer, BaseTokenizer):
            self.text_tokenizer_pad_id = self.tokenizer.pad
            self.tokens = self.tokenizer.tokens
        else:
            if text_tokenizer_pad_id is None:
                raise ValueError(f"text_tokenizer_pad_id must be specified if text_tokenizer is not BaseTokenizer")

            if tokens is None:
                raise ValueError(f"tokens must be specified if text_tokenizer is not BaseTokenizer")

            self.text_tokenizer_pad_id = text_tokenizer_pad_id
            self.tokens = tokens

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
            self.text_normalizer_call = self.normalizer.normalize
            if "text_normalizer_call_kwargs" in cfg:
                self.text_normalizer_call_kwargs = cfg.text_normalizer_call_kwargs

    def parse(self, text: str, normalize=False) -> torch.Tensor:
        if self.training:
            logging.warning("parse() is meant to be called in eval mode.")
        if normalize and self.text_normalizer_call is not None:
            text = self.text_normalizer_call(text, **self.text_normalizer_call_kwargs)

        eval_phon_mode = contextlib.nullcontext()
        if hasattr(self.tokenizer, "set_phone_prob"):
            eval_phon_mode = self.tokenizer.set_phone_prob(prob=1)
            print("changed to one")

        with eval_phon_mode:
            tokens = self.tokenizer.encode(text)
        print("text to token phone_prob")

        return torch.tensor(tokens).long().unsqueeze(0).cuda().to(self.device)

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

    def load_state_dict(self, state_dict, strict=True):
        # Override load_state_dict to be backward-compatible with old checkpoints
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("projection_fn.weight", "projection_fn.conv.weight")
            k = k.replace("projection_fn.bias", "projection_fn.conv.bias")
            new_state_dict[k] = v
        super().load_state_dict(new_state_dict, strict=strict)

    # Methods for model exportability
    @property
    def input_types(self):
        return self._input_types

    @property
    def output_types(self):
        return self._output_types

    def _prepare_for_export(self, **kwargs):
        self.model.remove_norms()
        super()._prepare_for_export(**kwargs)

        tensor_shape = ('T') if self.export_config["enable_ragged_batches"] else ('B', 'T')

        # Define input_types and output_types as required by export()
        self._input_types = {
            "text": NeuralType(tensor_shape, TokenIndex()),
            "batch_lengths": NeuralType(('B')),
            "speaker_id": NeuralType(('B'), Index()),
            "speaker_id_text": NeuralType(('B'), Index()),
            "speaker_id_attributes": NeuralType(('B'), Index()),
            "pitch": NeuralType(tensor_shape, RegressionValuesType()),
            "pace": NeuralType(tensor_shape),
        }
        self._output_types = {
            "spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "num_frames": NeuralType(('B'), TokenDurationType()),
            "durs_predicted": NeuralType(('B', 'T_text'), TokenDurationType()),
        }
        if self.export_config["enable_volume"]:
            self._input_types["volume"] = NeuralType(tensor_shape, optional=True)
            self._output_types["volume_aligned"] = NeuralType(('B', 'T_spec'), RegressionValuesType())

    def input_example(self, max_batch=1, max_dim=400):
        par = next(self.model.parameters())
        inputs = sample_tts_input(self.export_config, par.device, max_batch=max_batch, max_dim=max_dim)
        speaker = inputs.pop("speaker")
        inp = inputs['text']
        pad_id = self.tokenizer.pad
        inp[inp == pad_id] = pad_id - 1 if pad_id > 0 else pad_id + 1

        inputs.update(
            {'speaker_id': speaker, 'speaker_id_text': speaker, 'speaker_id_attributes': speaker,}
        )
        new_inputs = {
            'text': inp,
            'batch_lengths': inputs['batch_lengths'],
            'speaker_id': speaker,
            'speaker_id_text': speaker,
            'speaker_id_attributes': speaker,
            'pitch': inputs['pitch'],
            'pace': inputs['pace'],
            'volume': inputs['volume'],
        }

        return (new_inputs,)

    def forward_for_export(
        self, text, batch_lengths, speaker_id, speaker_id_text, speaker_id_attributes, pitch, pace, volume,
    ):
        if self.export_config["enable_ragged_batches"]:
            text, pitch, pace, volume_tensor, lens = batch_from_ragged(
                text, pitch, pace, batch_lengths=batch_lengths, padding_idx=self.tokenizer_pad, volume=volume,
            )
            if volume is not None:
                volume = volume_tensor
        else:
            lens = batch_lengths.to(dtype=torch.int64)

        (mel, n_frames, dur, _, _) = self.model.infer(
            speaker_id,
            text,
            speaker_id_text=speaker_id_text,
            speaker_id_attributes=speaker_id_attributes,
            sigma=0.7,
            f0_mean=0.0,
            f0_std=0.0,
            in_lens=lens,
            pitch_shift=pitch,
            pace=pace,
        ).values()
        ret_values = (mel.float(), n_frames, dur.float())

        if volume is not None:
            # Need to reshape as in infer patch
            durs_predicted = dur.float()
            truncated_length = torch.max(lens)
            volume_extended, _ = regulate_len(
                durs_predicted,
                volume[:, :truncated_length].unsqueeze(-1),
                pace[:, :truncated_length],
                group_size=self.model.n_group_size,
                dur_lens=lens,
            )
            volume_extended = volume_extended.squeeze(2).float()
            ret_values = ret_values + (volume_extended,)
        return ret_values
