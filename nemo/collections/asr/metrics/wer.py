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

from typing import List, Optional, Tuple, Union

import editdistance
import jiwer
import torch
from torchmetrics import Metric

from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.submodules.multitask_decoding import AbstractMultiTaskDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding
from nemo.utils import logging

__all__ = ['word_error_rate', 'word_error_rate_detail', 'WER']


def move_dimension_to_the_front(tensor, dim_index):
    all_dims = list(range(tensor.ndim))
    return tensor.permute(*([dim_index] + all_dims[:dim_index] + all_dims[dim_index + 1 :]))


def word_error_rate(hypotheses: List[str], references: List[str], use_cer=False) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string.

    Hypotheses and references must have same length.

    Args:
        hypotheses (list): list of hypotheses
        references(list) : list of references
        use_cer (bool): set True to enable cer

    Returns:
        wer (float): average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        # May deprecate using editdistance in future release for here and rest of codebase
        # once we confirm jiwer is reliable.
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer


def word_error_rate_detail(
    hypotheses: List[str], references: List[str], use_cer=False
) -> Tuple[float, int, float, float, float]:
    """
    Computes Average Word Error Rate with details (insertion rate, deletion rate, substitution rate)
    between two texts represented as corresponding lists of string.

    Hypotheses and references must have same length.

    Args:
        hypotheses (list): list of hypotheses
        references(list) : list of references
        use_cer (bool): set True to enable cer

    Returns:
        wer (float): average word error rate
        words (int):  Total number of words/charactors of given reference texts
        ins_rate (float): average insertion error rate
        del_rate (float): average deletion error rate
        sub_rate (float): average substitution error rate
    """
    scores = 0
    words = 0
    ops_count = {'substitutions': 0, 'insertions': 0, 'deletions': 0}

    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )

    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()

        # To get rid of the issue that jiwer does not allow empty string
        if len(r_list) == 0:
            if len(h_list) != 0:
                errors = len(h_list)
                ops_count['insertions'] += errors
            else:
                errors = 0
        else:
            if use_cer:
                measures = jiwer.cer(r, h, return_dict=True)
            else:
                measures = jiwer.compute_measures(r, h)

            errors = measures['insertions'] + measures['deletions'] + measures['substitutions']
            ops_count['insertions'] += measures['insertions']
            ops_count['deletions'] += measures['deletions']
            ops_count['substitutions'] += measures['substitutions']

        scores += errors
        words += len(r_list)

    if words != 0:
        wer = 1.0 * scores / words
        ins_rate = 1.0 * ops_count['insertions'] / words
        del_rate = 1.0 * ops_count['deletions'] / words
        sub_rate = 1.0 * ops_count['substitutions'] / words
    else:
        wer, ins_rate, del_rate, sub_rate = float('inf'), float('inf'), float('inf'), float('inf')

    return wer, words, ins_rate, del_rate, sub_rate


def word_error_rate_per_utt(hypotheses: List[str], references: List[str], use_cer=False) -> Tuple[List[float], float]:
    """
    Computes Word Error Rate per utterance and the average WER
    between two texts represented as corresponding lists of string.

    Hypotheses and references must have same length.

    Args:
        hypotheses (list): list of hypotheses
        references(list) : list of references
        use_cer (bool): set True to enable cer

    Returns:
        wer_per_utt (List[float]): word error rate per utterance
        avg_wer (float): average word error rate
    """
    scores = 0
    words = 0
    wer_per_utt = []

    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )

    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()

        # To get rid of the issue that jiwer does not allow empty string
        if len(r_list) == 0:
            if len(h_list) != 0:
                errors = len(h_list)
                wer_per_utt.append(float('inf'))
        else:
            if use_cer:
                measures = jiwer.cer(r, h, return_dict=True)
                er = measures['cer']
            else:
                measures = jiwer.compute_measures(r, h)
                er = measures['wer']

            errors = measures['insertions'] + measures['deletions'] + measures['substitutions']
            wer_per_utt.append(er)

        scores += errors
        words += len(r_list)

    if words != 0:
        avg_wer = 1.0 * scores / words
    else:
        avg_wer = float('inf')

    return wer_per_utt, avg_wer


class WER(Metric):
    """
    This metric computes numerator and denominator for Overall Word Error Rate (WER) between prediction and reference
    texts. When doing distributed training/evaluation the result of ``res=WER(predictions, predictions_lengths, targets, target_lengths)``
    calls will be all-reduced between all workers using SUM operations. Here ``res`` contains three numbers
    ``res=[wer, total_levenstein_distance, total_number_of_words]``.

    If used with PytorchLightning LightningModule, include wer_numerator and wer_denominators inside validation_step
    results. Then aggregate (sum) then at the end of validation epoch to correctly compute validation WER.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            wer_num, wer_denom = self.__wer(predictions, predictions_len, transcript, transcript_len)
            self.val_outputs = {'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denom': wer_denom}
            return self.val_outputs

        def on_validation_epoch_end(self):
            ...
            wer_num = torch.stack([x['val_wer_num'] for x in self.val_outputs]).sum()
            wer_denom = torch.stack([x['val_wer_denom'] for x in self.val_outputs]).sum()
            tensorboard_logs = {'validation_loss': val_loss_mean, 'validation_avg_wer': wer_num / wer_denom}
            self.val_outputs.clear()  # free memory
            return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    Args:
        decoding: An instance of CTCDecoding or RNNTDecoding.
        use_cer: Whether to use Character Error Rate instead of Word Error Rate.
        log_prediction: Whether to log a single decoded sample per call.
        batch_dim_index: Index corresponding to batch dimension. (For RNNT.)
        dist_dync_on_step: Whether to perform reduction on forward pass of metric.

    Returns:
        res: a tuple of 3 zero dimensional float32 ``torch.Tensor` objects: a WER score, a sum of Levenstein's
            distances for all prediction - reference pairs, total number of words in all references.
    """

    full_state_update: bool = True

    def __init__(
        self,
        decoding: Union[AbstractCTCDecoding, AbstractRNNTDecoding, AbstractMultiTaskDecoding],
        use_cer=False,
        log_prediction=True,
        fold_consecutive=True,
        batch_dim_index=0,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.decoding = decoding
        self.use_cer = use_cer
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive
        self.batch_dim_index = batch_dim_index

        self.decode = None
        if isinstance(self.decoding, AbstractRNNTDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids, targets: self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=predictions, encoded_lengths=predictions_lengths
            )
        elif isinstance(self.decoding, AbstractCTCDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids, targets: self.decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=predictions,
                decoder_lengths=predictions_lengths,
                fold_consecutive=self.fold_consecutive,
            )
        elif isinstance(self.decoding, AbstractMultiTaskDecoding):
            self.decode = lambda predictions, prediction_lengths, predictions_mask, input_ids, targets: self.decoding.decode_predictions_tensor(
                encoder_hidden_states=predictions,
                encoder_input_mask=predictions_mask,
                decoder_input_ids=input_ids,
                return_hypotheses=False,
            )
        else:
            raise TypeError(f"WER metric does not support decoding of type {type(self.decoding)}")

        self.add_state("scores", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("words", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
        predictions_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        """
        Updates metric state.
        Args:
            predictions: an integer torch.Tensor of shape ``[Batch, Time, {Vocabulary}]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            prediction_lengths: an integer torch.Tensor of shape ``[Batch]``
            targets: an integer torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            target_lengths: an integer torch.Tensor of shape ``[Batch]``
            predictions_lengths: an integer torch.Tensor of shape ``[Batch]``
        """
        words = 0
        scores = 0
        references = []
        with torch.no_grad():
            tgt_lenths_cpu_tensor = targets_lengths.long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            # check batch_dim_index is first dim
            if self.batch_dim_index != 0:
                targets_cpu_tensor = move_dimension_to_the_front(targets_cpu_tensor, self.batch_dim_index)
            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[0]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = self.decoding.decode_tokens_to_str(target)
                references.append(reference)
            hypotheses, _ = self.decode(predictions, predictions_lengths, predictions_mask, input_ids, targets)

        if self.log_prediction:
            logging.info(f"\n")
            logging.info(f"reference:{references[0]}")
            logging.info(f"predicted:{hypotheses[0]}")

        for h, r in zip(hypotheses, references):
            if self.use_cer:
                h_list = list(h)
                r_list = list(r)
            else:
                h_list = h.split()
                r_list = r.split()
            words += len(r_list)
            # Compute Levenstein's distance
            scores += editdistance.eval(h_list, r_list)

        self.scores = torch.tensor(scores, device=self.scores.device, dtype=self.scores.dtype)
        self.words = torch.tensor(words, device=self.words.device, dtype=self.words.dtype)

    def compute(self):
        scores = self.scores.detach().float()
        words = self.words.detach().float()
        return scores / words, scores, words
