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
import itertools
from typing import Iterable, Optional

import editdistance
import librosa
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.combined_loader import CombinedLoader

from nemo.collections.asr.losses.angularloss import AngularSoftmaxLoss
from nemo.collections.tts.data.dataset import TTSDataset
from nemo.collections.tts.modules.ssl_tts import GreedyCTCDecoder
from nemo.collections.tts.torch.tts_tokenizers import BaseTokenizer, EnglishCharsTokenizer
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.optim.lr_scheduler import WarmupPolicy
from nemo.utils import logging
from nemo.utils.decorators import experimental


@experimental
class SSLDisentangler(ModelPT):
    """
    SSLDisentangler is a Conformer based model for extracting disentangled content and speaker embeddings
    from an audio waveform. This model uses a pre-trained Conformer SSL model. To extract the linguistic content 
    and speaker representations using a pre-trained Conformer, two randomly initialized downstream 
    heads are added and the entire setup is finetuned in multi-task manner for speech recognition and speaker verification. 
    These representations can be used by FastPitchModel_SSL for voice conversion by swapping the speaker embedding 
    of a given source utterance, with the speaker embedding of a target speaker.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor_disentangler = SSLDisentangler.from_config_dict(self._cfg.preprocessor)
        self.encoder = SSLDisentangler.from_config_dict(self._cfg.encoder)
        self._text_tokenizer = EnglishCharsTokenizer(add_blank_at="last")
        self._tb_logger = None

        self.downstream_nets = torch.nn.ModuleDict()
        for task in self._cfg.downstream_heads.task_names:

            if task == 'speaker_verification':
                # setting up downstream heads and loss functions for speaker verification task
                in_dim = self._cfg.encoder.d_model
                out_dim = self._cfg.downstream_heads.speaker_embed_size
                num_speakers = self._cfg.downstream_heads.num_speakers
                self.downstream_nets[task] = torch.nn.Linear(in_dim, out_dim)
                self.sv_linear = torch.nn.Linear(out_dim, num_speakers)
                self.sv_loss = AngularSoftmaxLoss(scale=30, margin=0.4)

            elif task == 'content':
                # setting up downstream heads and loss functions for text/content recognition task
                in_dim = self._cfg.encoder.d_model
                out_dim = self._cfg.downstream_heads.content_embed_size
                num_chars = len(self._text_tokenizer.tokens)  # list of english tokens
                self.downstream_nets[task] = torch.nn.Linear(in_dim, out_dim)
                self.content_linear = torch.nn.Linear(out_dim, num_chars)
                self.ctc_loss = torch.nn.CTCLoss(blank=self._text_tokenizer.blank, zero_infinity=True)
                self.pitch_augment = self._cfg.get('pitch_augment', False)
                self.augment_ctc = self._cfg.get('augment_ctc', False)
                self.aug_loss_type = self._cfg.get('aug_loss_type', 'mse')
                self.stop_gradient = self._cfg.get('stop_gradient', False)
                assert (
                    self.stop_gradient and self.augment_ctc
                ) == False, "stop_gradient and augment_ctc cannot be true at the same time"
                self.mse_loss = torch.nn.MSELoss()

                self.ctc_decoder = GreedyCTCDecoder(self._text_tokenizer.tokens, self._text_tokenizer.blank)

            else:
                raise ValueError(f"{task} is not a valid task. Task must be speaker_verification or content.")

        self.automatic_optimization = False

        stft_cfg = self._cfg.preprocessor
        librosa_mel_filter = librosa.filters.mel(
            sr=stft_cfg.sample_rate, n_fft=stft_cfg.n_fft, n_mels=stft_cfg.features, fmin=0, fmax=8000
        )
        fb = torch.tensor(librosa_mel_filter, dtype=torch.float,).unsqueeze(0)

        self.register_buffer("fb", fb)

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="ssl_en_conformer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:ssl_en_conformer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/ssl_en_conformer_large/versions/1.10.1/files/ssl_en_conformer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="ssl_en_conformer_xlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:ssl_en_conformer_xlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/ssl_en_conformer_xlarge/versions/1.10.0/files/ssl_en_conformer_xlarge.nemo",
        )
        results.append(model)

        return results

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            tb_logger = self.logger.experiment
            if isinstance(self.logger, Iterable):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            self._tb_logger = tb_logger
        return self._tb_logger

    def __setup_dataloader_from_config(self, data_config):

        if hasattr(self, '_text_tokenizer') and isinstance(self._text_tokenizer, BaseTokenizer):
            _text_tokenizer = self._text_tokenizer

        else:
            if hasattr(self, '_text_tokenizer') and not isinstance(self._text_tokenizer, BaseTokenizer):
                logging.warning(f"test_tokenizer is set but not a BaseTokenizer. Will be set to EnglishCharsTokenizer")

            _text_tokenizer = self._text_tokenizer = EnglishCharsTokenizer(add_blank_at="last")

        for task in self._cfg.downstream_heads.task_names:
            if task == 'speaker_verification':
                sv_dataset = TTSDataset(
                    manifest_filepath=data_config['manifest_speaker_verification_fp'],
                    sample_rate=self._cfg.sample_rate,
                    text_tokenizer=_text_tokenizer,
                    segment_max_duration=data_config['segment_max_duration'],
                    sup_data_types=['speaker_id'],
                    sup_data_path=data_config['sup_data_path'],
                    pad_multiple=data_config.get('pad_multiple', 1),
                )
                sv_loader = torch.utils.data.DataLoader(
                    sv_dataset,
                    batch_size=data_config['batch_size_sv'],
                    collate_fn=sv_dataset.general_collate_fn,
                    shuffle=data_config['shuffle'],
                    num_workers=data_config.get('num_workers_sv', 0),
                    pin_memory=data_config.get('pin_memory', False),
                )

            elif task == 'content':
                content_dataset = TTSDataset(
                    manifest_filepath=data_config['manifest_content_fp'],
                    sample_rate=self._cfg.sample_rate,
                    text_tokenizer=_text_tokenizer,
                    min_duration=data_config['min_duration_content'],
                    max_duration=data_config['max_duration_content'],
                    pitch_augment=data_config.get('pitch_augment', False),
                    cache_pitch_augment=data_config.get('cache_pitch_augment', True),
                    sup_data_path=data_config['sup_data_path'],
                    pad_multiple=data_config.get('pad_multiple', 1),
                )
                content_loader = torch.utils.data.DataLoader(
                    content_dataset,
                    batch_size=data_config['batch_size_content'],
                    collate_fn=content_dataset.general_collate_fn,
                    shuffle=data_config['shuffle'],
                    num_workers=data_config.get('num_workers_content', 0),
                    pin_memory=data_config.get('pin_memory', False),
                )

            else:
                raise ValueError(f"{task} is not a valid task. Task must be speaker_verification or content.")

        loaders = {"sv": sv_loader, "content": content_loader}
        return loaders

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(self._cfg.train_ds)

    def setup_validation_data(self, cfg):
        self._validation_dl = CombinedLoader(self.__setup_dataloader_from_config(self._cfg.validation_ds))

    def configure_optimizers(self):
        optim_backbone_config = self._cfg.optim_backbone.copy()
        optim_downstream_config = self._cfg.optim_downstream.copy()

        OmegaConf.set_struct(optim_backbone_config, False)
        sched_backbone_config = optim_backbone_config.pop("sched", None)
        OmegaConf.set_struct(optim_backbone_config, True)

        OmegaConf.set_struct(optim_downstream_config, False)
        sched_downstream_config = optim_downstream_config.pop("sched", None)
        OmegaConf.set_struct(optim_downstream_config, True)

        optim_backbone = instantiate(optim_backbone_config, params=self.encoder.parameters(),)
        optim_downstream = instantiate(
            optim_downstream_config,
            params=itertools.chain(
                self.downstream_nets.parameters(),
                self.sv_linear.parameters(),
                self.content_linear.parameters(),
                self.sv_loss.parameters(),
            ),
        )

        if sched_backbone_config is not None and sched_downstream_config is not None:

            scheduler_backbone = WarmupPolicy(
                optimizer=optim_backbone,
                max_steps=None,
                min_lr=sched_backbone_config.min_lr,
                warmup_steps=sched_backbone_config.warmup_steps,
            )  # Use warmup to delay start
            sch1_dict = {
                'scheduler': scheduler_backbone,
                'interval': 'step',
            }

            scheduler_downstream = WarmupPolicy(
                optimizer=optim_downstream,
                max_steps=None,
                min_lr=sched_downstream_config.min_lr,
                warmup_steps=sched_downstream_config.warmup_steps,
            )
            sch2_dict = {
                'scheduler': scheduler_downstream,
                'interval': 'step',
            }

            return [optim_backbone, optim_downstream], [sch1_dict, sch2_dict]
        else:
            return [optim_backbone, optim_downstream]

    def forward(self, input_signal=None, input_signal_length=None, normalize_content=True):

        processed_signal, processed_signal_length = self.preprocessor_disentangler(
            input_signal=input_signal, length=input_signal_length,
        )

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)  # b,c,t

        for task in self._cfg.downstream_heads.task_names:
            if task == "speaker_verification":
                speaker_embedding = self.downstream_nets['speaker_verification'](encoded[:, :, 0])
                l2_norm = torch.norm(speaker_embedding, p=2, dim=-1, keepdim=True)
                speaker_embedding_normalized = speaker_embedding / l2_norm
                speaker_logits = self.sv_linear(speaker_embedding_normalized)

            elif task == "content":
                encoded_btc = encoded.permute(0, 2, 1)
                content_embedding = self.downstream_nets['content'](encoded_btc)
                if normalize_content:
                    l2_norm_content = torch.norm(content_embedding, p=2, dim=-1, keepdim=True)
                    content_embedding = content_embedding / l2_norm_content

                content_logits = self.content_linear(content_embedding)
                content_log_probs = content_logits.log_softmax(dim=2)
                content_log_probs = content_log_probs.permute(1, 0, 2)  # t,b,c for ctc

            else:
                raise ValueError(f"{task} is not a valid task. Task must be speaker_verification or content.")

        return (
            speaker_logits,
            speaker_embedding_normalized,
            content_embedding,
            content_log_probs,
            encoded_len,
        )

    def forward_for_export(self, input_signal=None, input_signal_length=None, normalize_content=True):
        # Same as forward right now. Earlier version of encoder had a different forward for export.
        # This function is still kept for compatibility with older evaluation/inference scripts.
        return self.forward(
            input_signal=input_signal, input_signal_length=input_signal_length, normalize_content=normalize_content,
        )

    def training_step(self, batch, batch_idx):
        loss = 0.0
        optim_backbone, optim_downstream = self.optimizers()
        schedulers = self.lr_schedulers()

        for key in batch.keys():
            if key == 'sv':
                signal = batch[key]['audio']
                signal_len = batch[key]['audio_lens']
                speaker_id = batch[key]['speaker_id']

                sv_logits, sv_emb, _, _, _ = self.forward(input_signal=signal, input_signal_length=signal_len)
                pred_speaker = torch.argmax(sv_logits, dim=1)

                sv_loss = self.sv_loss(logits=sv_logits, labels=speaker_id)
                loss += sv_loss
                if not self._cfg.combined_loss:
                    optim_backbone.zero_grad()
                    optim_downstream.zero_grad()
                    self.manual_backward(sv_loss)
                    optim_backbone.step()
                    optim_downstream.step()

                correct = pred_speaker.eq(speaker_id.data.view_as(pred_speaker)).sum().item()
                acc = (correct / len(speaker_id)) * 100

                self.log("t_sv_loss", sv_loss.item())
                self.log("t_sv_accuracy", acc)

            elif key == "content":
                content_loss = 0
                signal = batch[key]['audio']
                signal_len = batch[key]['audio_lens']
                target = batch[key]['text']  # (B, T)
                target_len = batch[key]['text_lens']

                _, _, content_embedding, content_log_probs, encoded_len = self.forward(
                    input_signal=signal, input_signal_length=signal_len
                )

                ctc_loss = self.ctc_loss(content_log_probs, target, encoded_len, target_len)
                # check if ctc loss is nan
                if torch.isfinite(ctc_loss):
                    self.log("t_ctc_loss", ctc_loss.item())
                    content_loss += ctc_loss
                else:
                    logging.warning(f"ctc_loss is not finite")

                if self.pitch_augment:
                    augmented_signal = batch[key]['audio_shifted']
                    if self.stop_gradient:
                        with torch.no_grad():
                            _, _, content_embedding_aug, content_log_probs_aug, _ = self.forward(
                                input_signal=augmented_signal, input_signal_length=signal_len
                            )
                    else:
                        _, _, content_embedding_aug, content_log_probs_aug, _ = self.forward(
                            input_signal=augmented_signal, input_signal_length=signal_len
                        )
                    if self.aug_loss_type == "mse":
                        sim_loss = self.mse_loss(content_embedding, content_embedding_aug)
                    elif self.aug_loss_type == "cosine":

                        cosine_similarity = torch.nn.functional.cosine_similarity(
                            content_embedding, content_embedding_aug, dim=-1
                        ).mean()

                        sim_loss = 1.0 - cosine_similarity

                    content_loss += self._cfg.augment_sim_alpha * sim_loss
                    self.log("t_sim_loss", sim_loss.item())

                    if self.augment_ctc:
                        ctc_loss_aug = self.ctc_loss(content_log_probs_aug, target, encoded_len, target_len)
                        if torch.isfinite(ctc_loss_aug):
                            content_loss += ctc_loss_aug
                            self.log("t_ctc_loss_aug", ctc_loss_aug.item())
                        else:
                            logging.warning(f"ctc_loss_aug is not finite. Add min duration to avoid getting here.")

                loss += content_loss

                if not self._cfg.combined_loss:
                    optim_backbone.zero_grad()
                    optim_downstream.zero_grad()
                    self.manual_backward(content_loss)
                    optim_backbone.step()
                    optim_downstream.step()

                if isinstance(content_loss, torch.Tensor):
                    self.log("t_content_loss", content_loss.item())

        if self._cfg.combined_loss:
            optim_backbone.zero_grad()
            optim_downstream.zero_grad()
            self.manual_backward(loss)
            optim_backbone.step()
            optim_downstream.step()

        if schedulers is not None:
            sch1, sch2 = schedulers
            sch1.step()
            sch2.step()

        if self.trainer.global_step % 10 == 0:
            self.log("lr_backbone", optim_backbone.param_groups[0]['lr'])
            self.log("lr_downstream", optim_downstream.param_groups[0]['lr'])
            self.log("t_loss", loss)

    def validation_step(self, batch, batch_idx):

        loss_total = 0
        for key in batch.keys():
            if key == 'sv':
                signal = batch[key]['audio']
                signal_len = batch[key]['audio_lens']
                speaker_id = batch[key]['speaker_id']
                sv_logits, sv_emb, _, _, _ = self.forward(input_signal=signal, input_signal_length=signal_len)

                pred_speaker = torch.argmax(sv_logits, dim=1)
                sv_loss = self.sv_loss(logits=sv_logits, labels=speaker_id)
                loss_total += sv_loss

                correct = pred_speaker.eq(speaker_id.data.view_as(pred_speaker)).sum().item()
                acc = (correct / len(speaker_id)) * 100
                acc_val = torch.as_tensor(acc)

            if key == 'content':
                content_loss = 0
                signal = batch[key]['audio']
                signal_len = batch[key]['audio_lens']
                target = batch[key]['text']  # (B, T)
                target_len = batch[key]['text_lens']

                _, _, content_embedding, content_log_probs, encoded_len = self.forward(
                    input_signal=signal, input_signal_length=signal_len
                )

                ctc_loss = self.ctc_loss(content_log_probs, target, encoded_len, target_len)

                # check if ctc loss is nan
                if torch.isfinite(ctc_loss):
                    content_loss += ctc_loss
                else:
                    logging.warning(f"ctc_loss is not finite. Add min duration to avoid getting here.")

                if self.pitch_augment:
                    augmented_signal = batch[key]['audio_shifted']
                    _, _, content_embedding_aug, content_log_probs_aug, _ = self.forward(
                        input_signal=augmented_signal, input_signal_length=signal_len
                    )
                    if self.aug_loss_type == "mse":
                        sim_loss = self.mse_loss(content_embedding, content_embedding_aug)
                    elif self.aug_loss_type == "cosine":
                        cosine_similarity = torch.nn.functional.cosine_similarity(
                            content_embedding, content_embedding_aug, dim=-1
                        ).mean()
                        sim_loss = 1.0 - cosine_similarity

                    content_loss += self._cfg.augment_sim_alpha * sim_loss

                loss_total += content_loss
                cers = []
                for _idx in range(target.shape[0]):
                    item_log_prob = content_log_probs[:, _idx, :][: encoded_len[_idx]].cpu()
                    item_target = target[_idx][: target_len[_idx]].cpu()
                    _, predicted_str = self.ctc_decoder(item_log_prob)
                    tokenizer = self._text_tokenizer
                    target_str = tokenizer.sep.join(tokenizer._id2token[t] for t in item_target.tolist())
                    ed = editdistance.eval(predicted_str, target_str)
                    if max(len(predicted_str), len(target_str)) > 0:
                        normalized_ed = (1.0 * ed) / max(len(predicted_str), len(target_str))
                    else:
                        normalized_ed = 1.0
                    cers.append(normalized_ed)

        return {
            'val_loss': loss_total.cpu(),
            'sv_loss': sv_loss.cpu(),
            'ctc_loss': ctc_loss.cpu(),
            'content_loss': content_loss.cpu(),
            'accuracy_sv': acc_val.cpu(),
            'cer': torch.tensor(cers).mean().cpu(),
        }

    def on_validation_epoch_end(self, outputs):
        collect = lambda key: torch.stack([x[key] for x in outputs if torch.isfinite(x[key])]).mean()
        val_loss = collect("val_loss")
        val_sv_loss = collect("sv_loss")
        val_ctc_loss = collect("ctc_loss")
        val_content_loss = collect("content_loss")
        accuracy_sv = collect("accuracy_sv")
        cer = collect("cer")
        self.log("val_loss", val_loss)
        self.log("sv_loss", val_sv_loss)
        self.log("val_ctc_loss", val_ctc_loss)
        self.log("val_content_loss", val_content_loss)
        self.log("accuracy_sv", accuracy_sv)
        self.log("cer", cer)
