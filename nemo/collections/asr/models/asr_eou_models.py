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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torchmetrics import Accuracy

from nemo.collections.asr.data.audio_to_eou_label_lhotse import (
    EOB_LABEL,
    EOB_STRING,
    EOU_LABEL,
    EOU_STRING,
    AudioToTextEOUBatch,
    LhotseSpeechToTextBpeEOUDataset,
)
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel, EncDecRNNTBPEModel
from nemo.collections.asr.modules.conformer_encoder import ConformerMultiLayerFeatureExtractor
from nemo.collections.asr.parts.mixins import TranscribeConfig
from nemo.collections.asr.parts.utils.eou_utils import (
    EOUResult,
    cal_eou_metrics_from_frame_labels,
    flatten_nested_list,
)
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.common.losses import CrossEntropyLoss
from nemo.core.classes.common import Serialization
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging

__all__ = ['EncDecRNNTBPEEOUModel', 'EncDecHybridRNNTCTCBPEEOUModel']


@dataclass
class EOUPrediction:
    eou_probs: Optional[List[float]] = None
    eob_probs: Optional[List[float]] = None
    eou_preds: Optional[List[bool]] = None
    eob_preds: Optional[List[bool]] = None


class ASREOUModelMixin:

    def _patch_decoding_cfg(self, cfg: DictConfig):
        """
        Patch the decoding config as needed for EOU computation
        """
        with open_dict(cfg):
            cfg.decoding.preserve_alignments = True
            cfg.decoding.compute_timestamps = True

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """
        PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
        """
        batch = move_data_to_device(batch, device)
        return batch

    def _get_text_from_tokens(self, tokens: torch.Tensor, tokens_len: Optional[torch.Tensor] = None) -> List[str]:
        """
        Convert tokens to text.
        Args:
            tokens: tensor of tokens
        Returns:
            text: list of text
        """
        text_list = []
        for i in range(len(tokens)):
            tokens_i = tokens[i]
            if tokens_len is not None:
                tokens_i = tokens[i][: tokens_len[i]]
            tokens_i = [int(x) for x in tokens_i if x < self.tokenizer.vocab_size]
            text = self.tokenizer.ids_to_text(tokens_i)
            text_list.append(text)
        return text_list

    def _get_eou_predictions_from_hypotheses(
        self, hypotheses: List[Hypothesis], batch: AudioToTextEOUBatch
    ) -> List[EOUPrediction]:
        """
        Get EOU predictions from the hypotheses.
        Args:
            hypotheses: batch of hypotheses
        Returns:
            eou_predictions: list of EOU predictions
        """
        eou_predictions = []

        for hyp in hypotheses:
            # Process one hypothesis at a time
            eou_probs = []
            eob_probs = []
            eou_preds = []
            eob_preds = []
            if isinstance(hyp.alignments, tuple):
                # CTC
                probs = torch.softmax(hyp.alignments[0], dim=-1)  # [time, num_classes]
                tokens = hyp.alignments[1]
                eou_probs = probs[:, self.eou_token].tolist()
                eob_probs = probs[:, self.eob_token].tolist()
                eou_preds = [int(x) == self.eou_token for x in tokens]
                eob_preds = [int(x) == self.eob_token for x in tokens]
            else:
                # RNNT, each timestamp has a list of (prob, token) tuples
                for alignment in hyp.alignments:
                    # Process for each timestamp
                    probs = torch.softmax(torch.stack([a[0] for a in alignment], dim=0), dim=-1)  # unfold RNNT preds
                    tokens = torch.stack([a[1] for a in alignment], dim=0)  # unfold RNNT preds

                    # Get the max prob for eou and eob
                    # and check if eou and eob are predicted
                    max_eou_prob = probs[:, self.eou_token].max().item()
                    max_eob_prob = probs[:, self.eob_token].max().item()
                    eou_pred = torch.any(tokens == self.eou_token).item()
                    eob_pred = torch.any(tokens == self.eob_token).item()

                    eou_probs.append(max_eou_prob)
                    eob_probs.append(max_eob_prob)
                    eou_preds.append(eou_pred)
                    eob_preds.append(eob_pred)

            eou_predictions.append(
                EOUPrediction(
                    eou_probs=eou_probs,
                    eob_probs=eob_probs,
                    eou_preds=eou_preds,
                    eob_preds=eob_preds,
                )
            )

        return eou_predictions

    def _pad_to_same_length(self, eou_labels: List[float], eou_preds: List[float]) -> Tuple[List[float], List[float]]:
        """
        Pad the EOU labels and predictions to the same length.
        Args:
            eou_labels: list of EOU labels
            eou_preds: list of EOU predictions
        Returns:
            eou_labels: list of EOU labels, padded to the same length
            eou_preds: list of EOU predictions, padded to the same length
        """
        if len(eou_labels) < len(eou_preds):
            eou_labels = eou_labels + [0] * (len(eou_preds) - len(eou_labels))
        elif len(eou_labels) > len(eou_preds):
            eou_preds = eou_preds + [0] * (len(eou_labels) - len(eou_preds))
        return eou_labels, eou_preds

    def _calculate_eou_metrics(
        self, eou_predictions: List[EOUPrediction], batch: AudioToTextEOUBatch
    ) -> Tuple[List, List]:
        """
        Calculate EOU metrics.
        Args:
            eou_predictions: list of EOU predictions
            batch: batch of data
        Returns:
            eou_metrics_list: list of EOU metrics, each is of type EOUResult
            eob_metrics_list: list of EOB metrics, each is of type EOUResult
        """
        # Get the ground truth EOU labels
        eou_labels = batch.eou_targets
        eou_labels_len = batch.eou_target_lengths

        # Calculate EOU metrics
        eou_metrics_list = []
        eob_metrics_list = []
        for i, eou_prediction in enumerate(eou_predictions):
            eou_preds_i = [float(x) for x in eou_prediction.eou_preds]
            eob_preds_i = [float(x) for x in eou_prediction.eob_preds]

            eou_labels_i = (eou_labels[i][: eou_labels_len[i]] == EOU_LABEL).float().tolist()
            eob_labels_i = (eou_labels[i][: eou_labels_len[i]] == EOB_LABEL).float().tolist()

            # Pad the EOU labels and predictions to the same length with zeros
            eou_labels_i, eou_preds_i = self._pad_to_same_length(eou_labels_i, eou_preds_i)
            eob_labels_i, eob_preds_i = self._pad_to_same_length(eob_labels_i, eob_preds_i)

            # Calculate EOU metrics
            eou_metrics = cal_eou_metrics_from_frame_labels(
                prediction=eou_preds_i,
                reference=eou_labels_i,
                threshold=0.0,
                collar=0.0,
                frame_len_in_secs=self.frame_len_in_secs,
            )  # type: EOUResult

            eob_metrics = cal_eou_metrics_from_frame_labels(
                prediction=eob_preds_i,
                reference=eob_labels_i,
                threshold=0.0,
                collar=0.0,
                frame_len_in_secs=self.frame_len_in_secs,
            )

            eou_metrics_list.append(eou_metrics)
            eob_metrics_list.append(eob_metrics)

        return eou_metrics_list, eob_metrics_list

    def _get_percentiles(self, values: List[float], percentiles: List[float], tag: str = "") -> Dict[str, float]:
        """
        Get the percentiles of a list of values.
        Args:
            values: list of values
            percentiles: list of percentiles
        Returns:
            metrics: Dict of percentiles
        """
        if len(values) == 0:
            return [0.0] * len(percentiles)
        results = np.percentile(values, percentiles).tolist()
        metrics = {}
        if tag:
            tag += "_"
        for i, p in enumerate(percentiles):
            metrics[f'{tag}p{int(p)}'] = float(results[i])
        return metrics

    def _aggregate_eou_metrics(self, outputs: List[dict], mode: str, is_ctc: bool = False):
        if f'{mode}_eou_metrics' not in outputs[0] and not is_ctc:
            return {}
        if f'{mode}_eou_metrics_ctc' not in outputs[0] and is_ctc:
            return {}

        # Aggregate EOU/EOB metrics
        eou_metrics = []  # type: List[EOUResult]
        eob_metrics = []  # type: List[EOUResult]
        for x in outputs:
            if is_ctc:
                eou_metrics.extend(x[f'{mode}_eou_metrics_ctc'])
                eob_metrics.extend(x[f'{mode}_eob_metrics_ctc'])
            else:
                eou_metrics.extend(x[f'{mode}_eou_metrics'])
                eob_metrics.extend(x[f'{mode}_eob_metrics'])
        num_eou_utterances = sum([x.num_utterances for x in eou_metrics])
        eou_latency = flatten_nested_list([x.latency for x in eou_metrics])
        eou_early_cutoff = flatten_nested_list([x.early_cutoff for x in eou_metrics])

        num_eob_utterances = sum([x.num_utterances for x in eob_metrics])
        eob_latency = flatten_nested_list([x.latency for x in eob_metrics])
        eob_early_cutoff = flatten_nested_list([x.early_cutoff for x in eob_metrics])

        eou_avg_num_early_cutoff = len(eou_early_cutoff) / num_eou_utterances if num_eou_utterances > 0 else 0.0
        eob_avg_num_early_cutoff = len(eob_early_cutoff) / num_eob_utterances if num_eob_utterances > 0 else 0.0
        if len(eou_latency) == 0:
            eou_latency = [0.0]
        if len(eou_early_cutoff) == 0:
            eou_early_cutoff = [0.0]
        if len(eob_latency) == 0:
            eob_latency = [0.0]
        if len(eob_early_cutoff) == 0:
            eob_early_cutoff = [0.0]

        eou_missing = [x.missing for x in eou_metrics]
        eob_missing = [x.missing for x in eob_metrics]

        tensorboard_logs = {}
        target_percentiles = [50, 90, 95]
        eou_latency_metrics = self._get_percentiles(eou_latency, target_percentiles, tag=f'{mode}_eou_latency')
        eou_early_cutoff_metrics = self._get_percentiles(
            eou_early_cutoff, target_percentiles, tag=f'{mode}_eou_early_cutoff'
        )
        eob_latency_metrics = self._get_percentiles(eob_latency, target_percentiles, tag=f'{mode}_eob_latency')
        eob_early_cutoff_metrics = self._get_percentiles(
            eob_early_cutoff, target_percentiles, tag=f'{mode}_eob_early_cutoff'
        )

        tensorboard_logs.update(eou_latency_metrics)
        tensorboard_logs.update(eou_early_cutoff_metrics)
        tensorboard_logs.update(eob_latency_metrics)
        tensorboard_logs.update(eob_early_cutoff_metrics)

        tensorboard_logs[f'{mode}_eou_early_cutoff_avg_num'] = eou_avg_num_early_cutoff
        tensorboard_logs[f'{mode}_eob_early_cutoff_avg_num'] = eob_avg_num_early_cutoff

        tensorboard_logs[f'{mode}_eou_missing'] = (
            sum(eou_missing) / num_eou_utterances if num_eou_utterances > 0 else 0.0
        )
        tensorboard_logs[f'{mode}_eob_missing'] = (
            sum(eob_missing) / num_eob_utterances if num_eob_utterances > 0 else 0.0
        )

        return tensorboard_logs

    @rank_zero_only
    def _maybe_save_predictions(self, outputs: List[Dict], mode: str = "val", dataloader_idx: int = 0):
        """
        Save predictions to disk.
        Args:
            outputs: list of outputs
            mode: mode of the model, either 'val' or 'test'
        """

        if not self.cfg.get('save_pred_to_file', None):
            return

        output_file = Path(self.cfg.save_pred_to_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if getattr(self, '_validation_names', None):
            output_file = output_file.with_name(f"{self._validation_names[dataloader_idx]}_{output_file.name}")
        else:
            output_file = output_file.with_suffix(f'.{dataloader_idx}.json')

        manifest = []
        for output in outputs:
            for i in range(len(output[f'{mode}_sample_id'])):
                item = {
                    "sample_id": output[f'{mode}_sample_id'][i],
                    "audio_filepath": output[f'{mode}_audio_filepath'][i],
                    "eou_text": output[f'{mode}_text_gt'][i],
                    "eou_pred_text": output[f'{mode}_text_pred'][i],
                    "is_backchannel": bool(str(output[f'{mode}_text_gt'][i]).endswith(EOB_STRING)),
                }
                if f"{mode}_text_pred_ctc" in output:
                    item["eou_pred_text_ctc"] = output[f"{mode}_text_pred_ctc"][i]

                eou_metrics = {f"eou_{k}": v for k, v in output[f"{mode}_eou_metrics"][i].to_dict().items()}
                eob_metrics = {f"eob_{k}": v for k, v in output[f"{mode}_eob_metrics"][i].to_dict().items()}
                item.update(eou_metrics)
                item.update(eob_metrics)
                manifest.append(item)
        write_manifest(output_file, manifest)
        logging.info(f"Predictions saved to {output_file}")
        return output_file


class EncDecRNNTBPEEOUModel(EncDecRNNTBPEModel, ASREOUModelMixin):
    def __init__(self, cfg: DictConfig, trainer):

        self._patch_decoding_cfg(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        self.eou_token = self.tokenizer.token_to_id(EOU_STRING)
        self.eob_token = self.tokenizer.token_to_id(EOB_STRING)
        self.frame_len_in_secs = self.cfg.preprocessor.window_stride * self.cfg.encoder.subsampling_factor

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
            return_hypotheses=True,
        )

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer:
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        cfg = OmegaConf.create(config) if not isinstance(config, DictConfig) else config
        dataset = LhotseSpeechToTextBpeEOUDataset(
            cfg=cfg, tokenizer=self.tokenizer, return_cuts=config.get("do_transcribe", False)
        )
        return get_lhotse_dataloader_from_config(
            config,
            # During transcription, the model is initially loaded on the CPU.
            # To ensure the correct global_rank and world_size are set,
            # these values must be passed from the configuration.
            global_rank=self.global_rank if not config.get("do_transcribe", False) else config.get("global_rank"),
            world_size=self.world_size if not config.get("do_transcribe", False) else config.get("world_size"),
            dataset=dataset,
            tokenizer=self.tokenizer,
        )

    def _transcribe_forward(self, batch: AudioToTextEOUBatch, trcfg: TranscribeConfig):
        encoded, encoded_len = self.forward(input_signal=batch.audio_signal, input_signal_length=batch.audio_lengths)
        output = dict(encoded=encoded, encoded_len=encoded_len)
        return output

    def training_step(self, batch: AudioToTextEOUBatch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal = batch.audio_signal
        signal_len = batch.audio_lengths
        transcript = batch.text_tokens
        transcript_len = batch.text_token_lengths

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

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If experimental fused Joint-Loss-WER is used
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
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch: AudioToTextEOUBatch, batch_idx, dataloader_idx=0):
        signal = batch.audio_signal
        signal_len = batch.audio_lengths

        # forward() only performs encoder forward
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        best_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        return list(best_hyp_text)

    def validation_pass(self, batch: AudioToTextEOUBatch, batch_idx: int, dataloader_idx: int = 0):
        signal = batch.audio_signal
        signal_len = batch.audio_lengths
        transcript = batch.text_tokens
        transcript_len = batch.text_token_lengths

        # forward() only performs encoder forward
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}

        if self.cfg.get('save_pred_to_file', None):
            text_gt = self._get_text_from_tokens(transcript, transcript_len)
            tensorboard_logs['val_sample_id'] = batch.sample_ids
            tensorboard_logs['val_audio_filepath'] = batch.audio_filepaths
            tensorboard_logs['val_text_gt'] = text_gt
        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )

                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            hypotheses = self.wer.get_hypotheses()

            if self.cfg.get('save_pred_to_file', None):
                text_pred = self._get_text_from_tokens([x.y_sequence for x in hypotheses])
                tensorboard_logs['val_text_pred'] = text_pred

            if self.cfg.get('calculate_eou_metrics', True):
                eou_predictions = self._get_eou_predictions_from_hypotheses(hypotheses, batch)
                eou_metrics_list, eob_metrics_list = self._calculate_eou_metrics(eou_predictions, batch)
            else:
                eou_metrics_list = []
                eob_metrics_list = []

            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer
            tensorboard_logs['val_eou_metrics'] = eou_metrics_list
            tensorboard_logs['val_eob_metrics'] = eob_metrics_list

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
                keep_hypotheses=True,
            )

            hypotheses = self.joint.get_hypotheses()

            if self.cfg.get('save_pred_to_file', None):
                text_pred = self._get_text_from_tokens([x.y_sequence for x in hypotheses])
                tensorboard_logs['val_text_pred'] = text_pred

            if self.cfg.get('calculate_eou_metrics', True):
                eou_predictions = self._get_eou_predictions_from_hypotheses(hypotheses, batch)
                eou_metrics_list, eob_metrics_list = self._calculate_eou_metrics(eou_predictions, batch)
            else:
                eou_metrics_list = []
                eob_metrics_list = []

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer
            tensorboard_logs['val_eou_metrics'] = eou_metrics_list
            tensorboard_logs['val_eob_metrics'] = eob_metrics_list

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def multi_inference_epoch_end(self, outputs, dataloader_idx: int = 0, mode: str = "val"):
        assert mode in ['val', 'test'], f"Invalid mode: {mode}. Must be 'val' or 'test'."

        if not outputs:
            logging.warning(
                f"No outputs received for {mode} dataloader {dataloader_idx}. Skipping epoch end processing."
            )
            return {}

        self._maybe_save_predictions(outputs, mode=mode, dataloader_idx=dataloader_idx)

        # Aggregate WER metrics
        if self.compute_eval_loss:
            loss_mean = torch.stack([x[f'{mode}_loss'] for x in outputs]).mean()
            loss_log = {f'{mode}_loss': loss_mean}
        else:
            loss_log = {}
        wer_num = torch.stack([x[f'{mode}_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x[f'{mode}_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**loss_log, f'{mode}_wer': wer_num.float() / wer_denom}

        if self.cfg.get('calculate_eou_metrics', True):
            eou_metrics = self._aggregate_eou_metrics(outputs, mode=mode)
        tensorboard_logs.update(eou_metrics)

        return {**loss_log, 'log': tensorboard_logs}

    # def test_step(self, batch: AudioToTextEOUBatch, batch_idx, dataloader_idx=0):
    #     # logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
    #     # test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}

    #     signal = batch.audio_signal
    #     signal_len = batch.audio_lengths
    #     transcript = batch.text_tokens
    #     transcript_len = batch.text_token_lengths

    #     # forward() only performs encoder forward
    #     encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
    #     del signal

    #     tensorboard_logs = {}
    #     hypotheses = self.decoding.rnnt_decoder_predictions_tensor(
    #         encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=True
    #     )
    #     eou_predictions = self._get_eou_predictions_from_hypotheses(hypotheses, batch)
    #     eou_metrics_list, eob_metrics_list = self._calculate_eou_metrics(eou_predictions, batch)
    #     tensorboard_logs['test_eou_metrics'] = eou_metrics_list
    #     tensorboard_logs['test_eob_metrics'] = eob_metrics_list

    #     test_logs = tensorboard_logs
    #     if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
    #         self.test_step_outputs[dataloader_idx].append(test_logs)
    #     else:
    #         self.test_step_outputs.append(test_logs)
    #     return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_inference_epoch_end(outputs, dataloader_idx, mode='val')

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_inference_epoch_end(outputs, dataloader_idx, mode='test')

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": AudioToTextEOUBatch,
            "inputs": [
                {"type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input", "name": "audio_signal"},
                {"type": NeuralType(("B",), LengthsType()), "seq_length": "input", "name": "audio_lengths"},
                {
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "name": "text_tokens",
                    "vocab_size": self.tokenizer.vocab_size,
                },
                {"type": NeuralType(("B",), LengthsType()), "seq_length": "output", "name": "text_token_lengths"},
                {
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "name": "eou_targets",
                    "vocab_size": 4,
                },
                {"type": NeuralType(("B",), LengthsType()), "seq_length": "output", "name": "eou_target_lengths"},
            ],
        }


class EncDecHybridRNNTCTCBPEEOUModel(EncDecHybridRNNTCTCBPEModel, ASREOUModelMixin):
    def __init__(self, cfg: DictConfig, trainer):
        self._patch_decoding_cfg(cfg)
        if cfg.aux_ctc.get('decoding', None) is not None:
            with open_dict(cfg):
                cfg.aux_ctc.decoding.preserve_alignments = True
                cfg.aux_ctc.decoding.compute_timestamps = True

        super().__init__(cfg=cfg, trainer=trainer)

        self.eou_token = self.tokenizer.token_to_id(EOU_STRING)
        self.eob_token = self.tokenizer.token_to_id(EOB_STRING)
        self.frame_len_in_secs = self.cfg.preprocessor.window_stride * self.cfg.encoder.subsampling_factor

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
            return_hypotheses=True,
        )

        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self.cfg.get("log_prediction", False),
            return_hypotheses=True,
        )

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer:
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        cfg = OmegaConf.create(config) if not isinstance(config, DictConfig) else config
        dataset = LhotseSpeechToTextBpeEOUDataset(
            cfg=cfg, tokenizer=self.tokenizer, return_cuts=config.get("do_transcribe", False)
        )
        return get_lhotse_dataloader_from_config(
            config,
            # During transcription, the model is initially loaded on the CPU.
            # To ensure the correct global_rank and world_size are set,
            # these values must be passed from the configuration.
            global_rank=self.global_rank if not config.get("do_transcribe", False) else config.get("global_rank"),
            world_size=self.world_size if not config.get("do_transcribe", False) else config.get("world_size"),
            dataset=dataset,
            tokenizer=self.tokenizer,
        )

    def training_step(self, batch: AudioToTextEOUBatch, batch_nb):
        signal = batch.audio_signal
        signal_len = batch.audio_lengths
        transcript = batch.text_tokens
        transcript_len = batch.text_token_lengths

        new_batch = (signal, signal_len, transcript, transcript_len)
        return super().training_step(new_batch, batch_nb)

    def predict_step(self, batch: AudioToTextEOUBatch, batch_idx, dataloader_idx=0):
        signal = batch.audio_signal
        signal_len = batch.audio_lengths
        transcript = batch.text_tokens
        transcript_len = batch.text_token_lengths
        sample_ids = batch.sample_ids
        new_batch = (signal, signal_len, transcript, transcript_len, sample_ids)
        return super().predict_step(new_batch, batch_idx, dataloader_idx)

    def validation_pass(self, batch: AudioToTextEOUBatch, batch_idx: int, dataloader_idx: int = 0):
        signal = batch.audio_signal
        signal_len = batch.audio_lengths
        transcript = batch.text_tokens
        transcript_len = batch.text_token_lengths

        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}

        if self.cfg.get('save_pred_to_file', None):
            text_gt = self._get_text_from_tokens(transcript, transcript_len)
            tensorboard_logs['val_sample_id'] = batch.sample_ids
            tensorboard_logs['val_audio_filepath'] = batch.audio_filepaths
            tensorboard_logs['val_text_gt'] = text_gt

        loss_value = None

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )
                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )

            hypotheses = self.wer.get_hypotheses()

            if self.cfg.get('save_pred_to_file', None):
                text_pred = self._get_text_from_tokens([x.y_sequence for x in hypotheses])
                tensorboard_logs['val_text_pred'] = text_pred

            eou_predictions = self._get_eou_predictions_from_hypotheses(hypotheses, batch)
            eou_metrics_list, eob_metrics_list = self._calculate_eou_metrics(eou_predictions, batch)

            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer
            tensorboard_logs['val_eou_metrics'] = eou_metrics_list
            tensorboard_logs['val_eob_metrics'] = eob_metrics_list
            tensorboard_logs['val_text_pred'] = text_pred

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
                keep_hypotheses=True,
            )
            hypotheses = self.joint.get_hypotheses()

            if self.cfg.get('save_pred_to_file', None):
                text_pred = self._get_text_from_tokens([x.y_sequence for x in hypotheses])
                tensorboard_logs['val_text_pred'] = text_pred

            eou_predictions = self._get_eou_predictions_from_hypotheses(hypotheses, batch)

            eou_metrics_list, eob_metrics_list = self._calculate_eou_metrics(eou_predictions, batch)

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer
            tensorboard_logs['val_eou_metrics'] = eou_metrics_list
            tensorboard_logs['val_eob_metrics'] = eob_metrics_list

        log_probs = self.ctc_decoder(encoder_output=encoded)
        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss
            tensorboard_logs['val_rnnt_loss'] = loss_value
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs['val_loss'] = loss_value

        self.ctc_wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        hypotheses_ctc = self.ctc_wer.get_hypotheses()

        if self.cfg.get('save_pred_to_file', None):
            text_pred_ctc = self._get_text_from_tokens([x.y_sequence for x in hypotheses_ctc])
            tensorboard_logs['val_text_pred_ctc'] = text_pred_ctc

        eou_predictions_ctc = self._get_eou_predictions_from_hypotheses(hypotheses_ctc, batch)
        eou_metrics_list_ctc, eob_metrics_list_ctc = self._calculate_eou_metrics(eou_predictions_ctc, batch)

        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()

        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer
        tensorboard_logs['val_eou_metrics_ctc'] = eou_metrics_list_ctc
        tensorboard_logs['val_eob_metrics_ctc'] = eob_metrics_list_ctc

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            compute_loss=self.compute_eval_loss,
            log_wer_num_denom=True,
            log_prefix="val_",
        )
        if self.compute_eval_loss:
            # overriding total loss value. Note that the previous
            # rnnt + ctc loss is available in metrics as "val_final_loss" now
            tensorboard_logs['val_loss'] = loss_value
        tensorboard_logs.update(additional_logs)
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        return tensorboard_logs

    def multi_inference_epoch_end(self, outputs, dataloader_idx: int = 0, mode: str = "val"):
        assert mode in ['val', 'test'], f"Invalid mode: {mode}. Must be 'val' or 'test'."
        self._maybe_save_predictions(outputs, mode=mode, dataloader_idx=dataloader_idx)

        # Aggregate WER metrics
        if self.compute_eval_loss:
            loss_mean = torch.stack([x[f'{mode}_loss'] for x in outputs]).mean()
            loss_log = {f'{mode}_loss': loss_mean}
        else:
            loss_log = {}
        wer_num = torch.stack([x[f'{mode}_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x[f'{mode}_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**loss_log, f'{mode}_wer': wer_num.float() / wer_denom}

        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['val_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['val_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['val_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom

        eou_metrics = self._aggregate_eou_metrics(outputs, mode)
        tensorboard_logs.update(eou_metrics)

        eou_metrics_ctc = self._aggregate_eou_metrics(outputs, mode, is_ctc=True)
        for key, value in eou_metrics_ctc.items():
            tensorboard_logs[f'{key}_ctc'] = value

        return {**loss_log, 'log': tensorboard_logs}

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_inference_epoch_end(outputs, dataloader_idx, mode='val')

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_inference_epoch_end(outputs, dataloader_idx, mode='test')


class EncDecHybridASRFrameEOUModel(EncDecHybridRNNTCTCBPEModel, ASREOUModelMixin):
    def __init__(self, cfg: DictConfig, trainer):
        super().__init__(cfg=cfg, trainer=trainer)
        self.frame_len_in_secs = self.cfg.preprocessor.window_stride * self.cfg.encoder.subsampling_factor
        self.layer_idx_list = self.cfg.get('layer_idx_list', [])
        assert isinstance(self.layer_idx_list, (list, ListConfig)), "cfg.layer_idx_list must be a list"
        num_encoder_layers = len(self.encoder.layers)
        if -1 not in self.layer_idx_list and num_encoder_layers - 1 not in self.layer_idx_list:
            self.layer_idx_list.append(num_encoder_layers - 1)
        self.encoder = ConformerMultiLayerFeatureExtractor(self.encoder, self.layer_idx_list)
        self.aggregator = Serialization.from_config_dict(cfg.aggregator)
        self.eou_encoder = Serialization.from_config_dict(cfg.eou_encoder) if cfg.eou_encoder is not None else None
        self.eou_decoder = Serialization.from_config_dict(cfg.eou_decoder)
        self.num_eou_classes = cfg.num_eou_classes
        self.rnnt_loss_weight = cfg.rnnt_loss_weight
        self.ctc_loss_weight = cfg.ctc_loss_weight
        self.eou_loss_weight = cfg.eou_loss_weight
        self.use_ctc_pred = cfg.get('use_ctc_pred', False)
        self.eou_loss = self._setup_eou_loss()

        if cfg.freeze_encoder:
            self.encoder.freeze()
        if cfg.freeze_rnnt:
            self.decoder.freeze()
            self.joint.freeze()
        if cfg.freeze_ctc:
            self.ctc_decoder.freeze()

        self.macro_accuracy = Accuracy(num_classes=self.num_eou_classes, average='macro', task="multiclass")

    def _setup_eou_loss(self):
        if "eou_loss" in self.cfg:
            weight = self.cfg.eou_loss.get("weight", None)
            if weight in [None, "none", "None"]:
                weight = [1.0] * self.num_eou_classes
            elif len(weight) != self.num_eou_classes:
                raise ValueError(
                    f"Length of weight must match the number of classes {self.num_eou_classes}, but got {weight}"
                )
            logging.info(f"Using cross-entropy with weights: {weight}")
        else:
            weight = [1.0] * self.num_eou_classes
        return CrossEntropyLoss(logits_ndim=3, weight=weight)

    def get_label_masks(self, labels: torch.Tensor, labels_len: torch.Tensor) -> torch.Tensor:
        mask = torch.arange(labels.size(1))[None, :].to(labels.device) < labels_len[:, None]
        return mask.to(labels.device, dtype=bool)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        cfg = OmegaConf.create(config) if not isinstance(config, DictConfig) else config
        dataset = LhotseSpeechToTextBpeEOUDataset(
            cfg=cfg, tokenizer=self.tokenizer, return_cuts=config.get("do_transcribe", False)
        )
        return get_lhotse_dataloader_from_config(
            config,
            # During transcription, the model is initially loaded on the CPU.
            # To ensure the correct global_rank and world_size are set,
            # these values must be passed from the configuration.
            global_rank=self.global_rank if not config.get("do_transcribe", False) else config.get("global_rank"),
            world_size=self.world_size if not config.get("do_transcribe", False) else config.get("world_size"),
            dataset=dataset,
            tokenizer=self.tokenizer,
        )

    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len

    def get_eou_prediction(
        self,
        encoded_all: List[torch.Tensor],
        encoded_len_all: List[torch.Tensor],
        ctc_pred: Optional[torch.Tensor] = None,
    ):
        if ctc_pred is not None and self.use_ctc_pred:
            encoded_all[-1] = ctc_pred
        eou_encoded, eou_encoded_len = self.aggregator(encoded_all, encoded_len_all)
        if self.eou_encoder is not None:
            eou_encoded, eou_encoded_len = self.eou_encoder(eou_encoded, eou_encoded_len)
        eou_pred = self.eou_decoder(eou_encoded)
        return eou_pred, eou_encoded_len

    def trim_eou_preds_labels(
        self,
        eou_pred: torch.Tensor,
        eou_pred_len: torch.Tensor,
        eou_labels: torch.Tensor,
        eou_labels_len: torch.Tensor,
    ):
        seq_len = eou_pred.size(1)
        if eou_labels.size(1) > seq_len:
            eou_labels = eou_labels[:, :seq_len]
            eou_labels_len = eou_labels_len.clamp(max=seq_len)
        elif eou_labels.size(1) < seq_len:
            seq_len = eou_labels.size(1)
            eou_pred = eou_pred[:, :seq_len]
            eou_pred_len = eou_pred_len.clamp(max=seq_len)

        # get the min between the eou_encoded_len and eou_labels_len
        eou_valid_len = torch.min(eou_pred_len, eou_labels_len)

        return eou_pred, eou_labels, eou_valid_len

    def get_eou_loss(
        self,
        eou_pred: torch.Tensor,
        eou_pred_len: torch.Tensor,
        eou_labels: torch.Tensor,
        eou_labels_len: torch.Tensor,
    ):
        eou_pred, eou_labels, eou_valid_len = self.trim_eou_preds_labels(
            eou_pred, eou_pred_len, eou_labels, eou_labels_len
        )
        eou_loss = self.eou_loss(
            logits=eou_pred,
            labels=eou_labels,
            loss_mask=self.get_label_masks(eou_labels, eou_valid_len),
        )
        return eou_loss

    def training_step(self, batch: AudioToTextEOUBatch, batch_nb):
        signal = batch.audio_signal
        signal_len = batch.audio_lengths
        transcript = batch.text_tokens
        transcript_len = batch.text_token_lengths
        eou_labels = batch.eou_targets
        eou_labels_len = batch.eou_target_lengths

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        encoded_all, encoded_len_all = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        encoded = encoded_all[-1]
        encoded_len = encoded_len_all[-1]

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        if (sample_id + 1) % log_every_n_steps == 0:
            compute_wer = True
        else:
            compute_wer = False

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        loss_value = None
        if self.rnnt_loss_weight > 0:
            # If fused Joint-Loss-WER is not used
            if not self.joint.fuse_loss_wer:
                # Compute full joint and loss
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )

                # Add auxiliary losses, if registered
                loss_value = self.add_auxiliary_losses(loss_value)

                if compute_wer:
                    self.wer.update(
                        predictions=encoded,
                        predictions_lengths=encoded_len,
                        targets=transcript,
                        targets_lengths=transcript_len,
                    )
                    _, scores, words = self.wer.compute()
                    self.wer.reset()
                    tensorboard_logs.update({'training_batch_wer': scores.float() / words})

            else:  # If fused Joint-Loss-WER is used
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

                if compute_wer:
                    tensorboard_logs.update({'training_batch_wer': wer})

        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded)
            ctc_log_probs = log_probs
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['train_rnnt_loss'] = loss_value
            tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})
        elif self.use_ctc_pred:
            ctc_log_probs = self.ctc_decoder(encoder_output=encoded)
        else:
            ctc_log_probs = None

        eou_pred, eou_pred_len = self.get_eou_prediction(encoded_all, encoded_len_all, ctc_log_probs)
        eou_loss = self.get_eou_loss(eou_pred, eou_pred_len, eou_labels, eou_labels_len)
        loss_value = loss_value + self.eou_loss_weight * eou_loss if loss_value is not None else eou_loss
        tensorboard_logs['train_eou_loss'] = eou_loss

        # note that we want to apply interctc independent of whether main ctc
        # loss is used or not (to allow rnnt + interctc training).
        # assuming ``ctc_loss_weight=0.3`` and interctc is applied to a single
        # layer with weight of ``0.1``, the total loss will be
        # ``loss = 0.9 * (0.3 * ctc_loss + 0.7 * rnnt_loss) + 0.1 * interctc_loss``
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=compute_wer
        )
        tensorboard_logs.update(additional_logs)
        tensorboard_logs['train_loss'] = loss_value
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch: AudioToTextEOUBatch, batch_idx, dataloader_idx=0):
        signal = batch.audio_signal
        signal_len = batch.audio_lengths
        sample_ids = batch.sample_ids

        encoded_all, encoded_len_all = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        best_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded_all[-1], encoded_lengths=encoded_len_all[-1], return_hypotheses=False
        )
        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.cpu().detach().numpy()

        eou_pred, eou_pred_len = self.get_eou_prediction(encoded_all, encoded_len_all)
        eou_predictions = [eou_pred[i][: eou_pred_len[i]] for i in range(len(eou_pred))]
        return zip(sample_ids, best_hyp_text, eou_predictions)

    def validation_pass(self, batch: AudioToTextEOUBatch, batch_idx: int, dataloader_idx: int = 0):
        signal = batch.audio_signal
        signal_len = batch.audio_lengths
        transcript = batch.text_tokens
        transcript_len = batch.text_token_lengths
        eou_labels = batch.eou_targets
        eou_labels_len = batch.eou_target_lengths

        # forward() only performs encoder forward
        encoded_all, encoded_len_all = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}

        if self.cfg.get('save_pred_to_file', None):
            text_gt = self._get_text_from_tokens(transcript, transcript_len)
            tensorboard_logs['val_sample_id'] = batch.sample_ids
            tensorboard_logs['val_audio_filepath'] = batch.audio_filepaths
            tensorboard_logs['val_text_gt'] = text_gt

        loss_value = None
        encoded = encoded_all[-1]
        encoded_len = encoded_len_all[-1]
        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )
                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )

            if self.cfg.get('save_pred_to_file', None):
                hypotheses = self.wer.get_hypotheses()
                text_pred = self._get_text_from_tokens([x.y_sequence for x in hypotheses])
                tensorboard_logs['val_text_pred'] = text_pred

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
                keep_hypotheses=True,
            )
            if self.cfg.get('save_pred_to_file', None):
                hypotheses = self.joint.get_hypotheses()
                text_pred = self._get_text_from_tokens([x.y_sequence for x in hypotheses])
                tensorboard_logs['val_text_pred'] = text_pred

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
            tensorboard_logs['val_ctc_loss'] = ctc_loss
            tensorboard_logs['val_rnnt_loss'] = loss_value
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs['val_loss'] = loss_value
        self.ctc_wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )

        if self.cfg.get('save_pred_to_file', None):
            hypotheses_ctc = self.ctc_wer.get_hypotheses()
            text_pred_ctc = self._get_text_from_tokens([x.y_sequence for x in hypotheses_ctc])
            tensorboard_logs['val_text_pred_ctc'] = text_pred_ctc

        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            compute_loss=self.compute_eval_loss,
            log_wer_num_denom=True,
            log_prefix="val_",
        )
        if self.compute_eval_loss:
            # overriding total loss value. Note that the previous
            # rnnt + ctc loss is available in metrics as "val_final_loss" now
            tensorboard_logs['val_loss'] = loss_value
        tensorboard_logs.update(additional_logs)
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # Calculate EOU metrics
        eou_pred, eou_pred_len = self.get_eou_prediction(encoded_all, encoded_len_all, log_probs)
        eou_loss = self.get_eou_loss(eou_pred, eou_pred_len, eou_labels, eou_labels_len)
        tensorboard_logs['val_eou_loss'] = eou_loss

        eou_pred, eou_labels, eou_valid_len = self.trim_eou_preds_labels(
            eou_pred, eou_pred_len, eou_labels, eou_labels_len
        )

        for i in range(eou_pred.size(0)):
            self.macro_accuracy.update(preds=eou_pred[i][: eou_valid_len[i]], target=eou_labels[i][: eou_valid_len[i]])
        stats = self.macro_accuracy._final_state()
        tensorboard_logs['val_eou_acc_stats'] = stats
        self.macro_accuracy.reset()

        eou_predictions = self._get_eou_predictions_from_frames(eou_pred, eou_valid_len)
        eou_metrics_list, eob_metrics_list = self._calculate_eou_metrics(eou_predictions, batch)

        tensorboard_logs['val_eou_metrics'] = eou_metrics_list
        tensorboard_logs['val_eob_metrics'] = eob_metrics_list

        return tensorboard_logs

    def _get_eou_predictions_from_frames(
        self, eou_pred: torch.Tensor, eou_pred_len: torch.Tensor
    ) -> List[EOUPrediction]:
        eou_predictions = []
        for i in range(eou_pred.size(0)):
            eou_logits_i = eou_pred[i][: eou_pred_len[i]]  # [time, num_classes]
            eou_probs = eou_logits_i[:, EOU_LABEL].detach().cpu().numpy().tolist()
            eob_probs = eou_logits_i[:, EOB_LABEL].detach().cpu().numpy().tolist()
            eou_frame_prediction = eou_logits_i.argmax(dim=-1).cpu().numpy().tolist()
            eou_preds = [int(x == EOU_LABEL) for x in eou_frame_prediction]
            eob_preds = [int(x == EOB_LABEL) for x in eou_frame_prediction]
            eou_predictions.append(
                EOUPrediction(
                    eou_probs=eou_probs,
                    eob_probs=eob_probs,
                    eou_preds=eou_preds,
                    eob_preds=eob_preds,
                )
            )
        return eou_predictions

    def multi_inference_epoch_end(self, outputs, dataloader_idx: int = 0, mode: str = "val"):
        assert mode in ['val', 'test'], f"Invalid mode: {mode}. Must be 'val' or 'test'."
        self._maybe_save_predictions(outputs, mode=mode, dataloader_idx=dataloader_idx)

        # Aggregate WER metrics
        if self.compute_eval_loss:
            loss_mean = torch.stack([x[f'{mode}_loss'] for x in outputs]).mean()
            loss_log = {f'{mode}_loss': loss_mean}
        else:
            loss_log = {}

        eou_loss_mean = torch.stack([x[f'{mode}_eou_loss'] for x in outputs]).mean()
        loss_log[f'{mode}_eou_loss'] = eou_loss_mean

        wer_num = torch.stack([x[f'{mode}_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x[f'{mode}_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**loss_log, f'{mode}_wer': wer_num.float() / wer_denom}

        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x[f'{mode}_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x[f'{mode}_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['val_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom

        eou_metrics = self._aggregate_eou_metrics(outputs, mode)
        tensorboard_logs.update(eou_metrics)

        self.macro_accuracy.reset()
        self.macro_accuracy.tp = torch.stack([x[f'{mode}_eou_acc_stats'][0] for x in outputs]).sum(axis=0)
        self.macro_accuracy.fp = torch.stack([x[f'{mode}_eou_acc_stats'][1] for x in outputs]).sum(axis=0)
        self.macro_accuracy.tn = torch.stack([x[f'{mode}_eou_acc_stats'][2] for x in outputs]).sum(axis=0)
        self.macro_accuracy.fn = torch.stack([x[f'{mode}_eou_acc_stats'][3] for x in outputs]).sum(axis=0)
        macro_accuracy_score = self.macro_accuracy.compute()
        self.macro_accuracy.reset()
        tensorboard_logs[f'{mode}_eou_macro_acc'] = macro_accuracy_score

        return {**loss_log, 'log': tensorboard_logs}

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_inference_epoch_end(outputs, dataloader_idx, mode='val')

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_inference_epoch_end(outputs, dataloader_idx, mode='test')
