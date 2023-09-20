# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import json
import os
import tempfile
from typing import List, Optional

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER, CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.parts.mixins import ASRBPEMixin, InterCTCMixin
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.multimodal.speech_cv.models.visual_rnnt_models import VisualEncDecRNNTModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging, model_utils


class VisualEncDecHybridRNNTCTCModel(VisualEncDecRNNTModel, ASRBPEMixin, InterCTCMixin):
    """Base class for hybrid RNNT/CTC models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        if 'aux_ctc' not in self.cfg:
            raise ValueError(
                "The config need to have a section for the CTC decoder named as aux_ctc for Hybrid models."
            )
        with open_dict(self.cfg.aux_ctc):
            if "feat_in" not in self.cfg.aux_ctc.decoder or (
                not self.cfg.aux_ctc.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self.cfg.aux_ctc.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self.cfg.aux_ctc.decoder or not self.cfg.aux_ctc.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

            if self.cfg.aux_ctc.decoder.num_classes < 1 and self.cfg.aux_ctc.decoder.vocabulary is not None:
                logging.info(
                    "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                        self.cfg.aux_ctc.decoder.num_classes, len(self.cfg.aux_ctc.decoder.vocabulary)
                    )
                )
                self.cfg.aux_ctc.decoder["num_classes"] = len(self.cfg.aux_ctc.decoder.vocabulary)

        self.ctc_decoder = VisualEncDecHybridRNNTCTCModel.from_config_dict(self.cfg.aux_ctc.decoder)
        self.ctc_loss_weight = self.cfg.aux_ctc.get("ctc_loss_weight", 0.5)

        self.ctc_loss = CTCLoss(
            num_classes=self.ctc_decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
        )

        ctc_decoding_cfg = self.cfg.aux_ctc.get('decoding', None)
        if ctc_decoding_cfg is None:
            ctc_decoding_cfg = OmegaConf.structured(CTCDecodingConfig)
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg

        self.ctc_decoding = CTCDecoding(self.cfg.aux_ctc.decoding, vocabulary=self.ctc_decoder.vocabulary)
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self.cfg.get("log_prediction", False),
        )

        # setting the RNNT decoder as the default one
        self.use_rnnt_decoder = True

        # setting up interCTC loss (from InterCTCMixin)
        self.setup_interctc(decoder_name='decoder', loss_name='loss', wer_name='_wer')

    @torch.no_grad()
    def transcribe(
        self,
        paths2video_files: List[str],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
    ) -> (List[str], Optional[List['Hypothesis']]):
        """
        Uses greedy decoding to transcribe video files. Use this method for debugging and prototyping.

        Args:

        paths2video_files: (a list) of paths to video files.
        batch_size: (int) batch size to use during inference. \
        Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
        With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.

        Returns:
            Returns a tuple of 2 items -
            * A list of greedy transcript texts / Hypothesis
            * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
        """
        if self.use_rnnt_decoder:
            return super().transcribe(
                paths2video_files=paths2video_files,
                batch_size=batch_size,
                return_hypotheses=return_hypotheses,
                partial_hypothesis=partial_hypothesis,
                num_workers=num_workers,
                channel_selector=channel_selector,
            )

        if paths2video_files is None or len(paths2video_files) == 0:
            return {}
        # We will store transcriptions here
        hypotheses = []
        all_hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        try:

            # Switch model to evaluation mode
            self.eval()
            # Freeze the visual front-end, encoder and decoder modules
            self.video_front_end.freeze()
            self.encoder.freeze()
            self.decoder.freeze()
            self.joint.freeze()
            if hasattr(self, 'ctc_decoder'):
                self.ctc_decoder.freeze()

            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                    for video_file in paths2video_files:
                        entry = {'video_filepath': video_file, 'duration': 100000, 'text': ''}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'paths2video_files': paths2video_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                    'channel_selector': channel_selector,
                }

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
                    encoded, encoded_len = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )

                    logits = self.ctc_decoder(encoder_output=encoded)
                    best_hyp, all_hyp = self.ctc_decoding.ctc_decoder_predictions_tensor(
                        logits, encoded_len, return_hypotheses=return_hypotheses,
                    )
                    if return_hypotheses:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            best_hyp[idx].y_sequence = logits[idx][: encoded_len[idx]]
                            if best_hyp[idx].alignments is None:
                                best_hyp[idx].alignments = best_hyp[idx].y_sequence
                    del logits

                    hypotheses += best_hyp
                    if all_hyp is not None:
                        all_hypotheses += all_hyp
                    else:
                        all_hypotheses += best_hyp

                    del encoded
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)

            logging.set_verbosity(logging_level)
            if mode is True:
                self.video_front_end.unfreeze()
                self.encoder.unfreeze()
                self.decoder.unfreeze()
                self.joint.unfreeze()
                if hasattr(self, 'ctc_decoder'):
                    self.ctc_decoder.unfreeze()
        return hypotheses, all_hypotheses

    def change_vocabulary(
        self,
        new_vocabulary: List[str],
        decoding_cfg: Optional[DictConfig] = None,
        ctc_decoding_cfg: Optional[DictConfig] = None,
    ):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning a pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
                this is target alphabet.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            ctc_decoding_cfg: A config for CTC decoding, which is optional and can be used to change decoding type.

        Returns: None

        """
        super().change_vocabulary(new_vocabulary=new_vocabulary, decoding_cfg=decoding_cfg)

        # set up the new tokenizer for the CTC decoder
        if hasattr(self, 'ctc_decoder'):
            if self.ctc_decoder.vocabulary == new_vocabulary:
                logging.warning(
                    f"Old {self.ctc_decoder.vocabulary} and new {new_vocabulary} match. Not changing anything."
                )
            else:
                if new_vocabulary is None or len(new_vocabulary) == 0:
                    raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
                decoder_config = self.ctc_decoder.to_config_dict()
                new_decoder_config = copy.deepcopy(decoder_config)
                new_decoder_config['vocabulary'] = new_vocabulary
                new_decoder_config['num_classes'] = len(new_vocabulary)

                del self.ctc_decoder
                self.ctc_decoder = VisualEncDecHybridRNNTCTCModel.from_config_dict(new_decoder_config)
                del self.ctc_loss
                self.ctc_loss = CTCLoss(
                    num_classes=self.ctc_decoder.num_classes_with_blank - 1,
                    zero_infinity=True,
                    reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
                )

                if ctc_decoding_cfg is None:
                    # Assume same decoding config as before
                    logging.info("No `ctc_decoding_cfg` passed when changing decoding strategy, using internal config")
                    ctc_decoding_cfg = self.cfg.aux_ctc.decoding

                # Assert the decoding config with all hyper parameters
                ctc_decoding_cls = OmegaConf.structured(CTCDecodingConfig)
                ctc_decoding_cls = OmegaConf.create(OmegaConf.to_container(ctc_decoding_cls))
                ctc_decoding_cfg = OmegaConf.merge(ctc_decoding_cls, ctc_decoding_cfg)

                self.ctc_decoding = CTCDecoding(decoding_cfg=ctc_decoding_cfg, vocabulary=self.ctc_decoder.vocabulary)

                self.ctc_wer = WER(
                    decoding=self.ctc_decoding,
                    use_cer=self.ctc_wer.use_cer,
                    log_prediction=self.ctc_wer.log_prediction,
                    dist_sync_on_step=True,
                )

                # Update config
                with open_dict(self.cfg.aux_ctc):
                    self.cfg.aux_ctc.decoding = ctc_decoding_cfg

                with open_dict(self.cfg.aux_ctc):
                    self.cfg.aux_ctc.decoder = new_decoder_config

                ds_keys = ['train_ds', 'validation_ds', 'test_ds']
                for key in ds_keys:
                    if key in self.cfg:
                        with open_dict(self.cfg[key]):
                            self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

                logging.info(f"Changed the tokenizer of the CTC decoder to {self.ctc_decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig, decoder_type: str = None):
        """
        Changes decoding strategy used during RNNT decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            decoder_type: (str) Can be set to 'rnnt' or 'ctc' to switch between appropriate decoder in a
                model having RNN-T and CTC decoders. Defaults to None, in which case RNN-T decoder is
                used. If set to 'ctc', it raises error if 'ctc_decoder' is not an attribute of the model.
        """
        if decoder_type is None or decoder_type == 'rnnt':
            self.use_rnnt_decoder = True
            return super().change_decoding_strategy(decoding_cfg=decoding_cfg)

        assert decoder_type == 'ctc' and hasattr(self, 'ctc_decoder')
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.aux_ctc.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.ctc_decoding = CTCDecoding(decoding_cfg=decoding_cfg, vocabulary=self.ctc_decoder.vocabulary)

        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.ctc_wer.use_cer,
            log_prediction=self.ctc_wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Update config
        with open_dict(self.cfg.aux_ctc):
            self.cfg.aux_ctc.decoding = decoding_cfg

        self.use_rnnt_decoder = False
        logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.aux_ctc.decoding)}")

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # If fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            # if AccessMixin.is_access_enabled():
            #    AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(encoded, encoded_len, transcript, transcript_len)
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            # if AccessMixin.is_access_enabled():
            #    AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )

            # Add Interctc Losses
            ctc_loss, interctc_tensorboard_logs = self.add_interctc_losses(
                ctc_loss, transcript, transcript_len, compute_wer=((batch_nb + 1) % log_every_n_steps == 0)
            )
            tensorboard_logs.update(interctc_tensorboard_logs)

            tensorboard_logs['train_rnnt_loss'] = loss_value
            tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs['train_loss'] = loss_value
            if (sample_id + 1) % log_every_n_steps == 0:
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    target_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})

        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO: add support for CTC decoding
        signal, signal_len, transcript, transcript_len, sample_id = batch

        # forward() only performs encoder forward
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        best_hyp_text, all_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, best_hyp_text))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )

                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(encoded, encoded_len, transcript, transcript_len)
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        log_probs = self.ctc_decoder(encoder_output=encoded)
        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )

            # Add interCTC losses
            ctc_loss, interctc_tensorboard_logs = self.add_interctc_losses(
                ctc_loss, transcript, transcript_len, compute_wer=True, log_wer_num_denom=True, log_prefix="val_",
            )
            tensorboard_logs.update(interctc_tensorboard_logs)

            tensorboard_logs['val_ctc_loss'] = ctc_loss
            tensorboard_logs['val_rnnt_loss'] = loss_value
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs['val_loss'] = loss_value
        self.ctc_wer.update(
            predictions=log_probs, targets=transcript, target_lengths=transcript_len, predictions_lengths=encoded_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        return tensorboard_logs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    """
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {
            'test_wer_num': logs['val_wer_num'],
            'test_wer_denom': logs['val_wer_denom'],
            # 'test_wer': logs['val_wer'],
        }
        if 'val_loss' in logs:
            test_logs['test_loss'] = logs['val_loss']

        if self.ctc_loss_weight > 0:
            test_logs['test_wer_num_ctc'] = logs['val_wer_num_ctc']
            test_logs['test_wer_denom_ctc'] = logs['val_wer_denom_ctc']
            if 'val_ctc_loss' in logs:
                test_logs['test_ctc_loss'] = logs['val_ctc_loss']
            if 'val_rnnt_loss' in logs:
                test_logs['test_rnnt_loss'] = logs['val_rnnt_loss']

        return test_logs
    """

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**val_loss_log, 'val_wer': wer_num.float() / wer_denom}
        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['val_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['val_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['val_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom

        metrics = {**val_loss_log, 'log': tensorboard_logs}
        self.finalize_interctc_metrics(metrics, outputs, prefix="val_")
        return metrics

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            test_loss_log = {'test_loss': test_loss_mean}
        else:
            test_loss_log = {}
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**test_loss_log, 'test_wer': wer_num.float() / wer_denom}

        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['test_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['test_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['test_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom

        metrics = {**test_loss_log, 'log': tensorboard_logs}
        self.finalize_interctc_metrics(metrics, outputs, prefix="test_")
        return metrics

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        return results
