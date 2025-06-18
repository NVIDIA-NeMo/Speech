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

from typing import Literal, Optional, Sequence, TypeAlias, Union

import torch
from lhotse import CutSet
from lhotse.cut import MixedCut
from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update
from torchmetrics.text import SacreBLEUScore

from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.submodules.multitask_decoding import AbstractMultiTaskDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding
from nemo.utils import logging

__all__ = ['BLEU']

# Keyword to avoid mispelling issues.
BLEU_TOKENIZER = "bleu_tokenizer"


def _get_bleu_tokenizers_from_cuts(cuts):
    """
    Helper function for multi tokenizer BLEU evaluation.
    Looks for `bleu_tokenizer` property to pass to BLEU metric.
    """

    def _get_lang(c):
        return c.custom.get(BLEU_TOKENIZER)

    # Dataloader passes multiple types of cuts. Need to diambiguate to access custom.
    # TODO: resolve in lhotse backend.
    return [_get_lang(c.first_non_padding_cut) if isinstance(c, MixedCut) else _get_lang(c) for c in cuts]


def _move_dimension_to_the_front(tensor, dim_index):
    all_dims = list(range(tensor.ndim))
    return tensor.permute(*([dim_index] + all_dims[:dim_index] + all_dims[dim_index + 1 :]))


class BLEU(SacreBLEUScore):
    """
    This metric computes numerator, denominator, hypotheses lengths, and target lengths for Overall Bilingual Evaluation Understudy (BLEU)
    between prediction and reference texts. When doing distributed training/evaluation the result of
    ``res=BLEU.(predictions, predictions_lengths, targets, target_lengths)``
    calls will be all-reduced between all workers using SUM operations.

    If used with PytorchLightning LightningModule, include bleu_num bleur_den, bleu_pred_len, and bleu_target_len values inside
    validation_step results. Then aggregate (sum) then at the end of validation epoch to correctly compute validation BLEUR.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            bleu_values = self.bleu(predictions, predictions_len, transcript, transcript_len)
            self.val_outputs = {'val_loss': loss_value, **bleu_values}
            return self.val_outputs

        def on_validation_epoch_end(self):
            ...
            bleu_num = torch.stack([x['val_bleu_num'] for x in self.val_outputs]).sum()
            bleu_denom = torch.stack([x['val_bleu_denom'] for x in self.val_outputs]).sum()
            bleu_pred_len = torch.stack([x['val_bleu_pred_len'] for x in self.val_outputs]).sum()
            bleu_target_len = torch.stack([x['val_bleu_target_len'] for x in self.val_outputs]).sum()
            val_bleu = {"val_bleu": self.bleu._compute_bleu(bleu_pred_len, bleu_target_len, bleu_num, bleu_denom)}
            tensorboard_logs.update(val_bleu)
            self.val_outputs.clear()  # free memory
            return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    Args:
        decoding: Decoder instance (CTCDecoding, RNNTDecoding, or MultiTaskDecoding) for converting model outputs to text.
        tokenize: Tokenizer name for BLEU evaluation (affects BLEU score based on language/tokenization).
        n_gram: Maximum n-gram order for BLEU calculation (default: 4).
        lowercase: If True, lowercases all input texts before evaluation.
        weights: Optional sequence of float weights for each n-gram order.
        smooth: If True, applies smoothing to BLEU calculation.
        check_cuts_for_tokenizers: If True, will inspect cuts for a `BLEU_TOKENIZERS` attribute for 'on the fly' changes to tokenizer (see `cuts` argument in `update`).
        log_prediction: If True, logs the first reference and prediction in each batch.
        batch_dim_index: Index of the batch dimension in input tensors.
        dist_sync_on_step: If True, synchronizes metric state across distributed workers on each step.

    Returns:
        Dictionary containing BLEU score and component statistics (numerator, denominator, prediction_lengths, target_lengths).
    """

    full_state_update: bool = True
    SacreBLEUToken: TypeAlias = Literal[
        "none", "13a", "zh", "intl", "char", "ja-mecab", "ko-mecab", "flores101", "flores200"
    ]

    def __init__(
        self,
        decoding: Union[AbstractCTCDecoding, AbstractRNNTDecoding, AbstractMultiTaskDecoding],
        bleu_tokenizer: SacreBLEUToken = "13a",
        n_gram: int = 4,
        lowercase: bool = False,
        weights: Optional[Sequence[float]] = None,
        smooth: bool = False,
        check_cuts_for_bleu_tokenizers: bool = False,
        log_prediction=False,
        fold_consecutive=True,
        batch_dim_index=0,
        dist_sync_on_step=False,
        sync_on_compute=True,
        **kwargs,
    ):
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive
        self.batch_dim_index = batch_dim_index

        self.decoding = decoding
        self._init_decode()

        self.check_cuts = check_cuts_for_bleu_tokenizers
        super().__init__(
            tokenize=bleu_tokenizer,
            n_gram=n_gram,
            lowercase=lowercase,
            weights=weights,
            smooth=smooth,
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=sync_on_compute,
        )

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
        predictions_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        cuts: Optional[CutSet] = None,
        **kwargs,  # To allow easy swapping of metrics without worrying about var alignment.
    ):
        """
        Updates metric state.
        Args:
            predictions: an integer torch.Tensor of shape ``[Batch, Time, {Vocabulary}]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            predictions_lengths: an integer torch.Tensor of shape ``[Batch]``
            targets: an integer torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            target_lengths: an integer torch.Tensor of shape ``[Batch]``
            predictions_mask: a bool torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``). Required for MultiTaskDecoding.
            input_ids: an int torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``). Required for MultiTaskDecoding.
            cuts: a CutSet of ``length == batch size``. If `self.check_cuts`, inspects each element
                for SacreBLEU tokenizer type for corresponding element in batch. If a sequence element is ``None``,
                the initial tokenizer type from ``BLEU.__init__`` is used. If ``cuts == None`` then all elements
                in batch are tokenized with initial tokenizer type.
        """
        tokenizers = None
        if self.check_cuts:
            assert (
                len(cuts) == targets_lengths.shape[0]
            ), f"BLEU metrics configured for multiple tokenizers, but got only '{len(cuts)}' samples for '{targets_lengths.shape[0]}' predictions."
            tokenizers = _get_bleu_tokenizers_from_cuts(cuts)

        with torch.no_grad():
            # get predictions
            hypotheses = (
                self.decode(predictions, predictions_lengths, predictions_mask, input_ids)
                if predictions.numel() > 0
                else []
            )

            # Get references
            if self.batch_dim_index != 0:
                targets = _move_dimension_to_the_front(targets, self.batch_dim_index)
            targets_cpu_tensor = targets.long().cpu()
            tgt_lenths_cpu_tensor = targets_lengths.long().cpu()
            for idx, tgt_len in enumerate(tgt_lenths_cpu_tensor):
                target = targets_cpu_tensor[idx][:tgt_len].tolist()
                reference = self.decoding.decode_tokens_to_str(target)
                tok = tokenizers[idx] if tokenizers else None  # `None` arg uses default tokenizer

                # TODO: the backend implementation of this has a lot of cpu to gpu operations. Should reimplement
                # for speedup.
                self.preds_len, self.target_len = _bleu_score_update(
                    [hypotheses[idx].text],
                    [[reference]],  # Nested list as BLEU permits multiple references per prediction.
                    self.numerator,
                    self.denominator,
                    self.preds_len,
                    self.target_len,
                    self.n_gram,
                    self._get_tokenizer(tok),
                )
                if hypotheses and self.log_prediction and idx == 0:
                    logging.info("\n")
                    logging.info(f"BLEU reference:{reference}")
                    logging.info(f"BLEU predicted:{hypotheses[idx].text}")

    def compute(self, return_all_metrics=True, prefix=""):
        """
        Returns BLEU values and component metrics.

        Args:
            return_all_metrics: bool flag. On True, BLEU and composite metrics returned. If False, returns
                only BLEU. Default: True.
            prefix: str to prepend to metric value keys.

        Returns:
            Dict: key-value pairs of BLEU metrics and values. Keys are prepended with prefix flag.
        """
        bleu = super().compute()
        if return_all_metrics:
            return {
                f"{prefix}bleu": bleu,
                f"{prefix}bleu_pred_len": self.preds_len.detach().float(),
                f"{prefix}bleu_target_len": self.target_len.detach().float(),
                f"{prefix}bleu_num": self.numerator.detach().float(),
                f"{prefix}bleu_denom": self.denominator.detach().float(),
            }
        return {
            f"{prefix}bleu": bleu,
        }

    # Adding wrapper to avoid imports and extra variables over the namespace
    def _compute_bleu(
        self,
        predictions_lengths,
        targets_lengths,
        numerator,
        denominator,
    ):
        return _bleu_score_compute(
            predictions_lengths, targets_lengths, numerator, denominator, self.n_gram, self.weights, self.smooth
        )

    # Wrapper for tokenizer access. Uses default if None.
    def _get_tokenizer(self, tokenize=None):
        if not self.check_cuts or tokenize is None:
            return self.tokenizer
        elif tokenize not in self.tokenizer._TOKENIZE_FN:
            raise KeyError(
                f"Sample passed BLEU tokenizer key '{tokenize}' but BLEU config only support '{self.tokenizer._TOKENIZE_FN.keys()}'"
            )
        # Lower level function of torchmetric SacreBLEU call. See:
        # https://github.com/Lightning-AI/torchmetrics/blob/5b8b757c71d1b0f0f056c0df63e3fd772974e8b0/src/torchmetrics/functional/text/sacre_bleu.py#L166-L168
        tokenizer_fn = getattr(self.tokenizer, self.tokenizer._TOKENIZE_FN[tokenize])
        return lambda line: self.tokenizer._lower(tokenizer_fn(line), self.tokenizer.lowercase).split()

    def _init_decode(self):
        self.decode = None
        if isinstance(self.decoding, AbstractRNNTDecoding):
            # Just preload all potential SacreBLEU tokenizers.
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids: self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=predictions, encoded_lengths=predictions_lengths
            )
        elif isinstance(self.decoding, AbstractCTCDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids: self.decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=predictions,
                decoder_lengths=predictions_lengths,
                fold_consecutive=self.fold_consecutive,
            )
        elif isinstance(self.decoding, AbstractMultiTaskDecoding):
            self.decode = lambda predictions, prediction_lengths, predictions_mask, input_ids: self.decoding.decode_predictions_tensor(
                encoder_hidden_states=predictions,
                encoder_input_mask=predictions_mask,
                decoder_input_ids=input_ids,
                return_hypotheses=False,
            )
        else:
            raise TypeError(f"BLEU metric does not support decoding of type {type(self.decoding)}")
