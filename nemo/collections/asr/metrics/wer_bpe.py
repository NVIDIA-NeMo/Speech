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
from typing import List

import editdistance
import torch
from torchmetrics import Metric

from nemo.collections.asr.metrics.wer import AbstractCTCDecoding, CTCDecodingConfig
from nemo.collections.asr.parts.submodules import ctc_beam_decoding
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.tokenizers.aggregate_tokenizer import DummyTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging


class CTCBPEDecoding(AbstractCTCDecoding):
    """
    Used for performing CTC auto-regressive / non-auto-regressive decoding of the logprobs for subword based
    models.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.
            strategy: str value which represents the type of decoding that can occur.
                Possible values are :
                -   greedy (for greedy decoding).
                -   beam (for DeepSpeed KenLM based decoding).

            compute_timestamps: A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrite intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.

            ctc_timestamp_type: A str value, which represents the types of timestamps that should be calculated.
                Can take the following values - "char" for character/subword time stamps, "word" for word level
                time stamps and "all" (default), for both character level and word level time stamps.

            word_seperator: Str token representing the seperator between words.

            preserve_alignments: Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `logprobs` in it. Here, `logprobs` is a torch.Tensors.

            confidence_cfg: A dict-like object which contains the following key-value pairs related to confidence
                scores. In order to obtain hypotheses with confidence scores, please utilize
                `ctc_decoder_predictions_tensor` function with the `preserve_frame_confidence` flag set to True.

                preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
                    generated during decoding. When set to true, the Hypothesis will contain
                    the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of floats.
                preserve_token_confidence: Bool flag which preserves the history of per-token confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `token_confidence` in it. Here, `token_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized tokens.
                preserve_word_confidence: Bool flag which preserves the history of per-word confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `word_confidence` in it. Here, `word_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized words.
                exclude_blank: Bool flag indicating that blank token confidence scores are to be excluded
                    from the `token_confidence`.
                aggregation: Which aggregation type to use for collapsing per-token confidence into per-word confidence.
                    Valid options are `mean`, `min`, `max`, `prod`.
                method_cfg: A dict-like object which contains the method name and settings to compute per-frame
                    confidence scores.

                    name: The method name (str).
                        Supported values:
                            - 'max_prob' for using the maximum token probability as a confidence.
                            - 'entropy' for using a normalized entropy of a log-likelihood vector.

                    entropy_type: Which type of entropy to use (str).
                        Used if confidence_method_cfg.name is set to `entropy`.
                        Supported values:
                            - 'gibbs' for the (standard) Gibbs entropy. If the alpha (α) is provided,
                                the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                                Note that for this entropy, the alpha should comply the following inequality:
                                (log(V)+2-sqrt(log^2(V)+4))/(2*log(V)) <= α <= (1+log(V-1))/log(V-1)
                                where V is the model vocabulary size.
                            - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                                Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/Tsallis_entropy
                            - 'renyi' for the Rényi entropy.
                                Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

                    alpha: Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                        When the alpha equals one, scaling is not applied to 'max_prob',
                        and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

                    entropy_norm: A mapping of the entropy value to the interval [0,1].
                        Supported values:
                            - 'lin' for using the linear mapping.
                            - 'exp' for using exponential mapping with linear shift.

            batch_dim_index: Index of the batch dimension of ``targets`` and ``predictions`` parameters of
                ``ctc_decoder_predictions_tensor`` methods. Can be either 0 or 1.

            The config may further contain the following sub-dictionaries:
            "greedy":
                preserve_alignments: Same as above, overrides above value.
                compute_timestamps: Same as above, overrides above value.
                preserve_frame_confidence: Same as above, overrides above value.
                confidence_method_cfg: Same as above, overrides confidence_cfg.method_cfg.

            "beam":
                beam_size: int, defining the beam size for beam search. Must be >= 1.
                    If beam_size == 1, will perform cached greedy search. This might be slightly different
                    results compared to the greedy search above.

                return_best_hypothesis: optional bool, whether to return just the best hypothesis or all of the
                    hypotheses after beam search has concluded. This flag is set by default.

                beam_alpha: float, the strength of the Language model on the final score of a token.
                    final_score = acoustic_score + beam_alpha * lm_score + beam_beta * seq_length.

                beam_beta: float, the strength of the sequence length penalty on the final score of a token.
                    final_score = acoustic_score + beam_alpha * lm_score + beam_beta * seq_length.

                kenlm_path: str, path to a KenLM ARPA or .binary file (depending on the strategy chosen).
                    If the path is invalid (file is not found at path), will raise a deferred error at the moment
                    of calculation of beam search, so that users may update / change the decoding strategy
                    to point to the correct file.

        tokenizer: NeMo tokenizer object, which inherits from TokenizerSpec.
    """

    def __init__(self, decoding_cfg, tokenizer: TokenizerSpec):
        blank_id = tokenizer.tokenizer.vocab_size
        self.tokenizer = tokenizer

        super().__init__(decoding_cfg=decoding_cfg, blank_id=blank_id)

        # Finalize Beam Search Decoding framework
        if isinstance(self.decoding, ctc_beam_decoding.AbstractBeamCTCInfer):
            if hasattr(self.tokenizer.tokenizer, 'get_vocab'):
                vocab_dict = self.tokenizer.tokenizer.get_vocab()
                if isinstance(self.tokenizer.tokenizer, DummyTokenizer):  # AggregateTokenizer.DummyTokenizer
                    vocab = vocab_dict
                else:
                    vocab = list(vocab_dict.keys())
                self.decoding.set_vocabulary(vocab)
                self.decoding.set_tokenizer(tokenizer)
            else:
                logging.warning("Could not resolve the vocabulary of the tokenizer !")

            self.decoding.set_decoding_type('subword')

    def _aggregate_token_confidence(self, hypothesis: Hypothesis) -> List[float]:
        """
        Implemented by subclass in order to aggregate token confidence to a word-level confidence.

        **Note**: Only supports Sentencepiece based tokenizers!

        Args:
            hypothesis: Hypothesis

        Returns:
            A list of word-level confidence scores.
        """
        return self._aggregate_token_confidence_subwords_sentencepiece(
            self.decode_tokens_to_str(hypothesis.text[0]).split(), hypothesis.token_confidence, hypothesis.text[0]
        )

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to decoder a token list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        hypothesis = self.tokenizer.ids_to_text(tokens)
        return hypothesis

    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Implemented by subclass in order to decode a token id list into a token list.
        A token list is the string representation of each token id.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded tokens.
        """
        token_list = self.tokenizer.ids_to_tokens(tokens)
        return token_list


class WERBPE(Metric):
    """
    This metric computes numerator and denominator for Overall Word Error Rate for BPE tokens (WER-BPE) between
    prediction and reference texts. When doing distributed training/evaluation the result of
    ``res=WERBPE(predictions, targets, target_lengths)`` calls will be all-reduced between all workers using SUM
    operations. Here ``res`` contains three numbers  ``res=[wer, total_levenstein_distance, total_number_of_words]``.

    If used with PytorchLightning LightningModule, include wer_numerator and wer_denominators inside validation_step
    results. Then aggregate (sum) then at the end of validation epoch to correctly compute validation WER.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            wer_num, wer_denom = self.__wer(predictions, transcript, transcript_len)
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
        decoding: An instance of CTCBPEDecoding.
        use_cer: Whether to compute word-error-rate or character-error-rate.
        log_prediction: Whether to log a single decoded sample per call.
        fold_consecutive: Whether repeated consecutive tokens should be folded into one when decoding.

    Returns:
        res: a tuple of 3 zero dimensional float32 ``torch.Tensor` objects: a WER score, a sum of Levenstein's
            distances for all prediction - reference pairs, total number of words in all references.
    """

    full_state_update: bool = True

    def __init__(
        self,
        decoding: CTCBPEDecoding,
        use_cer=False,
        log_prediction=True,
        fold_consecutive=True,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.decoding = decoding
        self.tokenizer = self.decoding.tokenizer
        self.blank_id = self.decoding.tokenizer.tokenizer.vocab_size
        self.use_cer = use_cer
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive

        self.add_state("scores", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("words", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictions_lengths: torch.Tensor = None,
    ):
        """
        Updates metric state.
        Args:
            predictions: an integer torch.Tensor of shape ``[Batch, Time, {Vocabulary}]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            targets: an integer torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            target_lengths: an integer torch.Tensor of shape ``[Batch]``
            predictions_lengths: an integer torch.Tensor of shape ``[Batch]``
        """
        words = 0
        scores = 0
        references = []
        with torch.no_grad():
            targets_cpu_tensor = targets.long().cpu()
            tgt_lenths_cpu_tensor = target_lengths.long().cpu()

            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[0]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = self.decoding.decode_tokens_to_str(target)
                references.append(reference)

            hypotheses, _ = self.decoding.ctc_decoder_predictions_tensor(
                predictions, predictions_lengths, fold_consecutive=self.fold_consecutive
            )

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
        # return torch.tensor([scores, words]).to(predictions.device)

    def compute(self):
        scores = self.scores.detach().float()
        words = self.words.detach().float()
        return scores / words, scores, words


@dataclass
class CTCBPEDecodingConfig(CTCDecodingConfig):
    pass
