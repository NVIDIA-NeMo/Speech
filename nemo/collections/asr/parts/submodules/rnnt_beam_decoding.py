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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from nemo.collections.asr.modules import rnnt_abstract
from nemo.collections.asr.parts.utils.rnnt_utils import (
    HATJointOutput,
    Hypothesis,
    NBestHypotheses,
    is_prefix,
    select_k_expansions,
)
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, HypothesisType, LengthsType, NeuralType
from nemo.utils import logging

try:
    import kenlm

    KENLM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KENLM_AVAILABLE = False


def pack_hypotheses(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
    for idx, hyp in enumerate(hypotheses):  # type: rnnt_utils.Hypothesis
        hyp.y_sequence = torch.tensor(hyp.y_sequence, dtype=torch.long)

        if hyp.dec_state is not None:
            hyp.dec_state = _states_to_device(hyp.dec_state)

        # Remove -1 from timestep
        if hyp.timestep is not None and len(hyp.timestep) > 0 and hyp.timestep[0] == -1:
            hyp.timestep = hyp.timestep[1:]

    return hypotheses


def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state


class BeamRNNTInfer(Typing):
    """
    Beam Search implementation ported from ESPNet implementation -
    https://github.com/espnet/espnet/blob/master/espnet/nets/beam_search_transducer.py

    Sequence level beam decoding or batched-beam decoding, performed auto-repressively
    depending on the search type chosen.

    Args:
        decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.

        beam_size: number of beams for beam search. Must be a positive integer >= 1.
            If beam size is 1, defaults to stateful greedy search.
            This greedy search might result in slightly different results than
            the greedy results obtained by GreedyRNNTInfer due to implementation differences.

            For accurate greedy results, please use GreedyRNNTInfer or GreedyBatchedRNNTInfer.

        search_type: str representing the type of beam search to perform.
            Must be one of ['beam', 'tsd', 'alsd']. 'nsc' is currently not supported.

            Algoritm used:

                `beam` - basic beam search strategy. Larger beams generally result in better decoding,
                    however the time required for the search also grows steadily.

                `tsd` - time synchronous decoding. Please refer to the paper:
                    [Alignment-Length Synchronous Decoding for RNN Transducer](https://ieeexplore.ieee.org/document/9053040)
                    for details on the algorithm implemented.

                    Time synchronous decoding (TSD) execution time grows by the factor T * max_symmetric_expansions.
                    For longer sequences, T is greater, and can therefore take a long time for beams to obtain
                    good results. This also requires greater memory to execute.

                `alsd` - alignment-length synchronous decoding. Please refer to the paper:
                    [Alignment-Length Synchronous Decoding for RNN Transducer](https://ieeexplore.ieee.org/document/9053040)
                    for details on the algorithm implemented.

                    Alignment-length synchronous decoding (ALSD) execution time is faster than TSD, with growth
                    factor of T + U_max, where U_max is the maximum target length expected during execution.

                    Generally, T + U_max < T * max_symmetric_expansions. However, ALSD beams are non-unique,
                    therefore it is required to use larger beam sizes to achieve the same (or close to the same)
                    decoding accuracy as TSD.

                    For a given decoding accuracy, it is possible to attain faster decoding via ALSD than TSD.

                `maes` = modified adaptive expansion searcn. Please refer to the paper:
                    [Accelerating RNN Transducer Inference via Adaptive Expansion Search](https://ieeexplore.ieee.org/document/9250505)

                    Modified Adaptive Synchronous Decoding (mAES) execution time is adaptive w.r.t the
                    number of expansions (for tokens) required per timestep. The number of expansions can usually
                    be constrained to 1 or 2, and in most cases 2 is sufficient.

                    This beam search technique can possibly obtain superior WER while sacrificing some evaluation time.

        score_norm: bool, whether to normalize the scores of the log probabilities.

        return_best_hypothesis: bool, decides whether to return a single hypothesis (the best out of N),
            or return all N hypothesis (sorted with best score first). The container class changes based
            this flag -
            When set to True (default), returns a single Hypothesis.
            When set to False, returns a NBestHypotheses container, which contains a list of Hypothesis.

        # The following arguments are specific to the chosen `search_type`

        tsd_max_sym_exp_per_step: Used for `search_type=tsd`. The maximum symmetric expansions allowed
            per timestep during beam search. Larger values should be used to attempt decoding of longer
            sequences, but this in turn increases execution time and memory usage.

        alsd_max_target_len: Used for `search_type=alsd`. The maximum expected target sequence length
            during beam search. Larger values allow decoding of longer sequences at the expense of
            execution time and memory.

        # The following two flags are placeholders and unused until `nsc` implementation is stabilized.
        nsc_max_timesteps_expansion: Unused int.

        nsc_prefix_alpha: Unused int.

        # mAES flags
        maes_num_steps: Number of adaptive steps to take. From the paper, 2 steps is generally sufficient. int > 1.

        maes_prefix_alpha: Maximum prefix length in prefix search. Must be an integer, and is advised to keep this as 1
            in order to reduce expensive beam search cost later. int >= 0.

        maes_expansion_beta: Maximum number of prefix expansions allowed, in addition to the beam size.
            Effectively, the number of hypothesis = beam_size + maes_expansion_beta. Must be an int >= 0,
            and affects the speed of inference since large values will perform large beam search in the next step.

        maes_expansion_gamma: Float pruning threshold used in the prune-by-value step when computing the expansions.
            The default (2.3) is selected from the paper. It performs a comparison (max_log_prob - gamma <= log_prob[v])
            where v is all vocabulary indices in the Vocab set and max_log_prob is the "most" likely token to be
            predicted. Gamma therefore provides a margin of additional tokens which can be potential candidates for
            expansion apart from the "most likely" candidate.
            Lower values will reduce the number of expansions (by increasing pruning-by-value, thereby improving speed
            but hurting accuracy). Higher values will increase the number of expansions (by reducing pruning-by-value,
            thereby reducing speed but potentially improving accuracy). This is a hyper parameter to be experimentally
            tuned on a validation set.

        softmax_temperature: Scales the logits of the joint prior to computing log_softmax.

        preserve_alignments: Bool flag which preserves the history of alignments generated during
            beam decoding (sample). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of Tensor (of length V + 1).

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.

            NOTE: `preserve_alignments` is an invalid argument for any `search_type`
            other than basic beam search.

        ngram_lm_model: str
            The path to the N-gram LM
        ngram_lm_alpha: float
            Alpha weight of N-gram LM
        tokens_type: str
            Tokenization type ['subword', 'char']
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "partial_hypotheses": [NeuralType(elements_type=HypothesisType(), optional=True)],  # must always be last
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        beam_size: int,
        search_type: str = 'default',
        score_norm: bool = True,
        return_best_hypothesis: bool = True,
        tsd_max_sym_exp_per_step: Optional[int] = 50,
        alsd_max_target_len: Union[int, float] = 1.0,
        nsc_max_timesteps_expansion: int = 1,
        nsc_prefix_alpha: int = 1,
        maes_num_steps: int = 2,
        maes_prefix_alpha: int = 1,
        maes_expansion_gamma: float = 2.3,
        maes_expansion_beta: int = 2,
        language_model: Optional[Dict[str, Any]] = None,
        softmax_temperature: float = 1.0,
        preserve_alignments: bool = False,
        ngram_lm_model: Optional[str] = None,
        ngram_lm_alpha: float = 0.0,
        hat_subtract_ilm: bool = False,
        hat_ilm_weight: float = 0.0,
    ):
        self.decoder = decoder_model
        self.joint = joint_model

        self.blank = decoder_model.blank_idx
        self.vocab_size = decoder_model.vocab_size
        self.search_type = search_type
        self.return_best_hypothesis = return_best_hypothesis

        if beam_size < 1:
            raise ValueError("Beam search size cannot be less than 1!")

        self.beam_size = beam_size
        self.score_norm = score_norm
        self.max_candidates = beam_size

        if self.beam_size == 1:
            logging.info("Beam size of 1 was used, switching to sample level `greedy_search`")
            self.search_algorithm = self.greedy_search
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "nsc":
            raise NotImplementedError("`nsc` (Constrained Beam Search) has not been implemented.")
            # self.search_algorithm = self.nsc_beam_search
        elif search_type == "maes":
            self.search_algorithm = self.modified_adaptive_expansion_search
        else:
            raise NotImplementedError(
                f"The search type ({search_type}) supplied is not supported!\n"
                f"Please use one of : (default, tsd, alsd, nsc)"
            )

        if tsd_max_sym_exp_per_step is None:
            tsd_max_sym_exp_per_step = -1

        if search_type in ['tsd', 'alsd', 'nsc'] and not self.decoder.blank_as_pad:
            raise ValueError(
                f"Search type was chosen as '{search_type}', however the decoder module provided "
                f"does not support the `blank` token as a pad value. {search_type} requires "
                f"the blank token as pad value support in order to perform batched beam search."
                f"Please chose one of the other beam search methods, or re-train your model "
                f"with this support."
            )

        self.tsd_max_symmetric_expansion_per_step = tsd_max_sym_exp_per_step
        self.alsd_max_target_length = alsd_max_target_len
        self.nsc_max_timesteps_expansion = nsc_max_timesteps_expansion
        self.nsc_prefix_alpha = int(nsc_prefix_alpha)
        self.maes_prefix_alpha = int(maes_prefix_alpha)
        self.maes_num_steps = int(maes_num_steps)
        self.maes_expansion_gamma = float(maes_expansion_gamma)
        self.maes_expansion_beta = int(maes_expansion_beta)

        if self.search_type == 'maes' and self.maes_prefix_alpha < 0:
            raise ValueError("`maes_prefix_alpha` must be a positive integer.")

        if self.search_type == 'maes' and self.vocab_size < beam_size + maes_expansion_beta:
            raise ValueError(
                f"beam_size ({beam_size}) + expansion_beta ({maes_expansion_beta}) "
                f"should be smaller or equal to vocabulary size ({self.vocab_size})."
            )

        if search_type == 'maes':
            self.max_candidates += maes_expansion_beta

        if self.search_type == 'maes' and self.maes_num_steps < 2:
            raise ValueError("`maes_num_steps` must be greater than 1.")

        if softmax_temperature != 1.0 and language_model is not None:
            logging.warning(
                "Softmax temperature is not supported with LM decoding." "Setting softmax-temperature value to 1.0."
            )

            self.softmax_temperature = 1.0
        else:
            self.softmax_temperature = softmax_temperature
        self.language_model = language_model
        self.preserve_alignments = preserve_alignments

        self.token_offset = 0

        if ngram_lm_model:
            if KENLM_AVAILABLE:
                self.ngram_lm = kenlm.Model(ngram_lm_model)
                self.ngram_lm_alpha = ngram_lm_alpha
            else:
                raise ImportError(
                    "KenLM package (https://github.com/kpu/kenlm) is not installed. " "Use ngram_lm_model=None."
                )
        else:
            self.ngram_lm = None

        if hat_subtract_ilm:
            assert hasattr(self.joint, "return_hat_ilm")
            assert search_type == "maes"
        self.hat_subtract_ilm = hat_subtract_ilm
        self.hat_ilm_weight = hat_ilm_weight

    @typecheck()
    def __call__(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
    ) -> Union[Hypothesis, NBestHypotheses]:
        """Perform general beam search.

        Args:
            encoder_output: Encoded speech features (B, D_enc, T_max)
            encoded_lengths: Lengths of the encoder outputs

        Returns:
            Either a list containing a single Hypothesis (when `return_best_hypothesis=True`,
            otherwise a list containing a single NBestHypotheses, which itself contains a list of
            Hypothesis. This list is sorted such that the best hypothesis is the first element.
        """
        # Preserve decoder and joint training state
        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training

        # setup hat outputs mode
        return_hat_ilm_default = False
        if self.hat_subtract_ilm:
            assert hasattr(self.joint, "return_hat_ilm")
            return_hat_ilm_default = self.joint.return_hat_ilm
            self.joint.return_hat_ilm = self.hat_subtract_ilm

        with torch.inference_mode():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)

            self.decoder.eval()
            self.joint.eval()

            hypotheses = []
            with tqdm(
                range(encoder_output.size(0)),
                desc='Beam search progress:',
                total=encoder_output.size(0),
                unit='sample',
            ) as idx_gen:

                _p = next(self.joint.parameters())
                dtype = _p.dtype

                # Decode every sample in the batch independently.
                for batch_idx in idx_gen:
                    inseq = encoder_output[batch_idx : batch_idx + 1, : encoded_lengths[batch_idx], :]  # [1, T, D]
                    logitlen = encoded_lengths[batch_idx]

                    if inseq.dtype != dtype:
                        inseq = inseq.to(dtype=dtype)

                    # Extract partial hypothesis if exists
                    partial_hypothesis = partial_hypotheses[batch_idx] if partial_hypotheses is not None else None

                    # Execute the specific search strategy
                    nbest_hyps = self.search_algorithm(
                        inseq, logitlen, partial_hypotheses=partial_hypothesis
                    )  # sorted list of hypothesis

                    # Prepare the list of hypotheses
                    nbest_hyps = pack_hypotheses(nbest_hyps)

                    # Pack the result
                    if self.return_best_hypothesis:
                        best_hypothesis = nbest_hyps[0]  # type: Hypothesis
                    else:
                        best_hypothesis = NBestHypotheses(nbest_hyps)  # type: NBestHypotheses
                    hypotheses.append(best_hypothesis)

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)
        if self.hat_subtract_ilm:
            self.joint.return_hat_ilm = return_hat_ilm_default

        return (hypotheses,)

    def sort_nbest(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Sort hypotheses by score or score given sequence length.

        Args:
            hyps: list of hypotheses

        Return:
            hyps: sorted list of hypotheses
        """
        if self.score_norm:
            return sorted(hyps, key=lambda x: x.score / len(x.y_sequence), reverse=True)
        else:
            return sorted(hyps, key=lambda x: x.score, reverse=True)

    def greedy_search(
        self, h: torch.Tensor, encoded_lengths: torch.Tensor, partial_hypotheses: Optional[Hypothesis] = None
    ) -> List[Hypothesis]:
        """Greedy search implementation for transducer.
        Generic case when beam size = 1. Results might differ slightly due to implementation details
        as compared to `GreedyRNNTInfer` and `GreedyBatchRNNTInfer`.

        Args:
            h: Encoded speech features (1, T_max, D_enc)

        Returns:
            hyp: 1-best decoding results
        """
        if self.preserve_alignments:
            # Alignments is a 2-dimensional dangling list representing T x U
            alignments = [[]]
        else:
            alignments = None

        # Initialize zero state vectors
        dec_state = self.decoder.initialize_state(h)

        # Construct initial hypothesis
        hyp = Hypothesis(
            score=0.0, y_sequence=[self.blank], dec_state=dec_state, timestep=[-1], length=encoded_lengths
        )

        if partial_hypotheses is not None:
            if len(partial_hypotheses.y_sequence) > 0:
                hyp.y_sequence = [int(partial_hypotheses.y_sequence[-1].cpu().numpy())]
                hyp.dec_state = partial_hypotheses.dec_state
                hyp.dec_state = _states_to_device(hyp.dec_state, h.device)

        cache = {}

        # Initialize state and first token
        y, state, _ = self.decoder.score_hypothesis(hyp, cache)

        for i in range(int(encoded_lengths)):
            hi = h[:, i : i + 1, :]  # [1, 1, D]

            not_blank = True
            symbols_added = 0

            # TODO: Figure out how to remove this hard coding afterwords
            while not_blank and (symbols_added < 5):
                ytu = torch.log_softmax(self.joint.joint(hi, y) / self.softmax_temperature, dim=-1)  # [1, 1, 1, V + 1]
                ytu = ytu[0, 0, 0, :]  # [V + 1]

                # max() requires float
                if ytu.dtype != torch.float32:
                    ytu = ytu.float()

                logp, pred = torch.max(ytu, dim=-1)  # [1, 1]
                pred = pred.item()

                if self.preserve_alignments:
                    # insert logprobs into last timestep
                    alignments[-1].append((ytu.to('cpu'), torch.tensor(pred, dtype=torch.int32)))

                if pred == self.blank:
                    not_blank = False

                    if self.preserve_alignments:
                        # convert Ti-th logits into a torch array
                        alignments.append([])  # blank buffer for next timestep
                else:
                    # Update state and current sequence
                    hyp.y_sequence.append(int(pred))
                    hyp.score += float(logp)
                    hyp.dec_state = state
                    hyp.timestep.append(i)

                    # Compute next state and token
                    y, state, _ = self.decoder.score_hypothesis(hyp, cache)
                symbols_added += 1

        # Remove trailing empty list of alignments
        if self.preserve_alignments:
            if len(alignments[-1]) == 0:
                del alignments[-1]

        # attach alignments to hypothesis
        hyp.alignments = alignments

        # Remove the original input label if partial hypothesis was provided
        if partial_hypotheses is not None:
            hyp.y_sequence = hyp.y_sequence[1:]

        return [hyp]

    def default_beam_search(
        self, h: torch.Tensor, encoded_lengths: torch.Tensor, partial_hypotheses: Optional[Hypothesis] = None
    ) -> List[Hypothesis]:
        """Beam search implementation.

        Args:
            x: Encoded speech features (1, T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results
        """
        # Initialize states
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))
        blank_tensor = torch.tensor([self.blank], device=h.device, dtype=torch.long)

        # Precompute some constants for blank position
        ids = list(range(self.vocab_size + 1))
        ids.remove(self.blank)

        # Used when blank token is first vs last token
        if self.blank == 0:
            index_incr = 1
        else:
            index_incr = 0

        # Initialize zero vector states
        dec_state = self.decoder.initialize_state(h)

        # Initialize first hypothesis for the beam (blank)
        kept_hyps = [Hypothesis(score=0.0, y_sequence=[self.blank], dec_state=dec_state, timestep=[-1], length=0)]
        cache = {}

        if partial_hypotheses is not None:
            if len(partial_hypotheses.y_sequence) > 0:
                kept_hyps[0].y_sequence = [int(partial_hypotheses.y_sequence[-1].cpu().numpy())]
                kept_hyps[0].dec_state = partial_hypotheses.dec_state
                kept_hyps[0].dec_state = _states_to_device(kept_hyps[0].dec_state, h.device)

        if self.preserve_alignments:
            kept_hyps[0].alignments = [[]]

        for i in range(int(encoded_lengths)):
            hi = h[:, i : i + 1, :]  # [1, 1, D]
            hyps = kept_hyps
            kept_hyps = []

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                # update decoder state and get next score
                y, state, lm_tokens = self.decoder.score_hypothesis(max_hyp, cache)  # [1, 1, D]

                # get next token
                ytu = torch.log_softmax(self.joint.joint(hi, y) / self.softmax_temperature, dim=-1)  # [1, 1, 1, V + 1]
                ytu = ytu[0, 0, 0, :]  # [V + 1]

                # preserve alignments
                if self.preserve_alignments:
                    logprobs = ytu.cpu().clone()

                # remove blank token before top k
                top_k = ytu[ids].topk(beam_k, dim=-1)

                # Two possible steps - blank token or non-blank token predicted
                ytu = (
                    torch.cat((top_k[0], ytu[self.blank].unsqueeze(0))),
                    torch.cat((top_k[1] + index_incr, blank_tensor)),
                )

                # for each possible step
                for logp, k in zip(*ytu):
                    # construct hypothesis for step
                    new_hyp = Hypothesis(
                        score=(max_hyp.score + float(logp)),
                        y_sequence=max_hyp.y_sequence[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                        timestep=max_hyp.timestep[:],
                        length=encoded_lengths,
                    )

                    if self.preserve_alignments:
                        new_hyp.alignments = copy.deepcopy(max_hyp.alignments)

                    # if current token is blank, dont update sequence, just store the current hypothesis
                    if k == self.blank:
                        kept_hyps.append(new_hyp)
                    else:
                        # if non-blank token was predicted, update state and sequence and then search more hypothesis
                        new_hyp.dec_state = state
                        new_hyp.y_sequence.append(int(k))
                        new_hyp.timestep.append(i)

                        hyps.append(new_hyp)

                    # Determine whether the alignment should be blank or token
                    if self.preserve_alignments:
                        if k == self.blank:
                            new_hyp.alignments[-1].append(
                                (logprobs.clone(), torch.tensor(self.blank, dtype=torch.int32))
                            )
                        else:
                            new_hyp.alignments[-1].append(
                                (logprobs.clone(), torch.tensor(new_hyp.y_sequence[-1], dtype=torch.int32))
                            )

                # keep those hypothesis that have scores greater than next search generation
                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )

                # If enough hypothesis have scores greater than next search generation,
                # stop beam search.
                if len(kept_most_prob) >= beam:
                    if self.preserve_alignments:
                        # convert Ti-th logits into a torch array
                        for kept_h in kept_most_prob:
                            kept_h.alignments.append([])  # blank buffer for next timestep

                    kept_hyps = kept_most_prob
                    break

        # Remove trailing empty list of alignments
        if self.preserve_alignments:
            for h in kept_hyps:
                if len(h.alignments[-1]) == 0:
                    del h.alignments[-1]

        # Remove the original input label if partial hypothesis was provided
        if partial_hypotheses is not None:
            for hyp in kept_hyps:
                if hyp.y_sequence[0] == partial_hypotheses.y_sequence[-1] and len(hyp.y_sequence) > 1:
                    hyp.y_sequence = hyp.y_sequence[1:]

        return self.sort_nbest(kept_hyps)

    def time_sync_decoding(
        self, h: torch.Tensor, encoded_lengths: torch.Tensor, partial_hypotheses: Optional[Hypothesis] = None
    ) -> List[Hypothesis]:
        """Time synchronous beam search implementation.
        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoded speech features (1, T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results
        """
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not supported")

        # Precompute some constants for blank position
        ids = list(range(self.vocab_size + 1))
        ids.remove(self.blank)

        # Used when blank token is first vs last token
        if self.blank == 0:
            index_incr = 1
        else:
            index_incr = 0

        # prepare the batched beam states
        beam = min(self.beam_size, self.vocab_size)
        beam_state = self.decoder.initialize_state(
            torch.zeros(beam, device=h.device, dtype=h.dtype)
        )  # [L, B, H], [L, B, H] (for LSTMs)

        # Initialize first hypothesis for the beam (blank)
        B = [
            Hypothesis(
                y_sequence=[self.blank],
                score=0.0,
                dec_state=self.decoder.batch_select_state(beam_state, 0),
                timestep=[-1],
                length=0,
            )
        ]
        cache = {}

        # Initialize alignments
        if self.preserve_alignments:
            for hyp in B:
                hyp.alignments = [[]]

        for i in range(int(encoded_lengths)):
            hi = h[:, i : i + 1, :]

            # Update caches
            A = []
            C = B

            h_enc = hi

            # For a limited number of symmetric expansions per timestep "i"
            for v in range(self.tsd_max_symmetric_expansion_per_step):
                D = []

                # Decode a batch of beam states and scores
                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score_hypothesis(C, cache, beam_state)

                # Extract the log probabilities and the predicted tokens
                beam_logp = torch.log_softmax(
                    self.joint.joint(h_enc, beam_y) / self.softmax_temperature, dim=-1
                )  # [B, 1, 1, V + 1]
                beam_logp = beam_logp[:, 0, 0, :]  # [B, V + 1]
                beam_topk = beam_logp[:, ids].topk(beam, dim=-1)

                seq_A = [h.y_sequence for h in A]

                for j, hyp in enumerate(C):
                    # create a new hypothesis in A
                    if hyp.y_sequence not in seq_A:
                        # If the sequence is not in seq_A, add it as the blank token
                        # In this step, we dont add a token but simply update score
                        _temp_hyp = Hypothesis(
                            score=(hyp.score + float(beam_logp[j, self.blank])),
                            y_sequence=hyp.y_sequence[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            timestep=hyp.timestep[:],
                            length=encoded_lengths,
                        )

                        # Preserve the blank token alignment
                        if self.preserve_alignments:
                            _temp_hyp.alignments = copy.deepcopy(hyp.alignments)
                            _temp_hyp.alignments[-1].append(
                                (beam_logp[j].clone(), torch.tensor(self.blank, dtype=torch.int32)),
                            )

                        A.append(_temp_hyp)
                    else:
                        # merge the existing blank hypothesis score with current score.
                        dict_pos = seq_A.index(hyp.y_sequence)

                        A[dict_pos].score = np.logaddexp(
                            A[dict_pos].score, (hyp.score + float(beam_logp[j, self.blank]))
                        )

                if v < self.tsd_max_symmetric_expansion_per_step:
                    for j, hyp in enumerate(C):
                        # for each current hypothesis j
                        # extract the top token score and top token id for the jth hypothesis
                        for logp, k in zip(beam_topk[0][j], beam_topk[1][j] + index_incr):
                            # create new hypothesis and store in D
                            # Note: This loop does *not* include the blank token!
                            new_hyp = Hypothesis(
                                score=(hyp.score + float(logp)),
                                y_sequence=(hyp.y_sequence + [int(k)]),
                                dec_state=self.decoder.batch_select_state(beam_state, j),
                                lm_state=hyp.lm_state,
                                timestep=hyp.timestep[:] + [i],
                                length=encoded_lengths,
                            )

                            # Preserve token alignment
                            if self.preserve_alignments:
                                new_hyp.alignments = copy.deepcopy(hyp.alignments)
                                new_hyp.alignments[-1].append(
                                    (beam_topk[0].clone().cpu(), torch.tensor(k, dtype=torch.int32)),
                                )

                            D.append(new_hyp)

                # Prune beam
                C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]

                if self.preserve_alignments:
                    # convert Ti-th logits into a torch array
                    for C_i in C:
                        # Check if the last token emitted at last timestep was a blank
                        # If so, move to next timestep
                        logp, label = C_i.alignments[-1][-1]  # The last alignment of this step
                        if int(label) == self.blank:
                            C_i.alignments.append([])  # blank buffer for next timestep

            # Prune beam
            B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]

            if self.preserve_alignments:
                # convert Ti-th logits into a torch array
                for B_i in B:
                    # Check if the last token emitted at last timestep was a blank
                    # If so, move to next timestep
                    logp, label = B_i.alignments[-1][-1]  # The last alignment of this step
                    if int(label) == self.blank:
                        B_i.alignments.append([])  # blank buffer for next timestep

        # Remove trailing empty list of alignments
        if self.preserve_alignments:
            for h in B:
                if len(h.alignments[-1]) == 0:
                    del h.alignments[-1]

        return self.sort_nbest(B)

    def align_length_sync_decoding(
        self, h: torch.Tensor, encoded_lengths: torch.Tensor, partial_hypotheses: Optional[Hypothesis] = None
    ) -> List[Hypothesis]:
        """Alignment-length synchronous beam search implementation.
        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoded speech features (1, T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results
        """
        # delay this import here instead of at the beginning to avoid circular imports.
        from nemo.collections.asr.modules.rnnt import RNNTDecoder, StatelessTransducerDecoder

        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not supported")

        # Precompute some constants for blank position
        ids = list(range(self.vocab_size + 1))
        ids.remove(self.blank)

        # Used when blank token is first vs last token
        if self.blank == 0:
            index_incr = 1
        else:
            index_incr = 0

        # prepare the batched beam states
        beam = min(self.beam_size, self.vocab_size)

        h = h[0]  # [T, D]
        h_length = int(encoded_lengths)
        beam_state = self.decoder.initialize_state(
            torch.zeros(beam, device=h.device, dtype=h.dtype)
        )  # [L, B, H], [L, B, H] for LSTMS

        # compute u_max as either a specific static limit,
        # or a multiple of current `h_length` dynamically.
        if type(self.alsd_max_target_length) == float:
            u_max = int(self.alsd_max_target_length * h_length)
        else:
            u_max = int(self.alsd_max_target_length)

        # Initialize first hypothesis for the beam (blank)
        B = [
            Hypothesis(
                y_sequence=[self.blank],
                score=0.0,
                dec_state=self.decoder.batch_select_state(beam_state, 0),
                timestep=[-1],
                length=0,
            )
        ]

        # Initialize alignments
        if self.preserve_alignments:
            B[0].alignments = [[]]

        final = []
        cache = {}

        # ALSD runs for T + U_max steps
        for i in range(h_length + u_max):
            # Update caches
            A = []
            B_ = []
            h_states = []

            # preserve the list of batch indices which are added into the list
            # and those which are removed from the list
            # This is necessary to perform state updates in the correct batch indices later
            batch_ids = list(range(len(B)))  # initialize as a list of all batch ids
            batch_removal_ids = []  # update with sample ids which are removed

            for bid, hyp in enumerate(B):
                u = len(hyp.y_sequence) - 1
                t = i - u

                if t > (h_length - 1):
                    batch_removal_ids.append(bid)
                    continue

                B_.append(hyp)
                h_states.append((t, h[t]))

            if B_:
                # Compute the subset of batch ids which were *not* removed from the list above
                sub_batch_ids = None
                if len(B_) != beam:
                    sub_batch_ids = batch_ids
                    for id in batch_removal_ids:
                        # sub_batch_ids contains list of ids *that were not removed*
                        sub_batch_ids.remove(id)

                    # extract the states of the sub batch only.
                    if isinstance(self.decoder, RNNTDecoder):
                        # LSTM decoder, state is [layer x batch x hidden]
                        beam_state_ = [
                            beam_state[state_id][:, sub_batch_ids, :] for state_id in range(len(beam_state))
                        ]
                    elif isinstance(self.decoder, StatelessTransducerDecoder):
                        # stateless decoder, state is [batch x hidden]
                        beam_state_ = [beam_state[state_id][sub_batch_ids, :] for state_id in range(len(beam_state))]
                    else:
                        raise NotImplementedError("Unknown decoder type.")

                else:
                    # If entire batch was used (none were removed), simply take all the states
                    beam_state_ = beam_state

                # Decode a batch/sub-batch of beam states and scores
                beam_y, beam_state_, beam_lm_tokens = self.decoder.batch_score_hypothesis(B_, cache, beam_state_)

                # If only a subset of batch ids were updated (some were removed)
                if sub_batch_ids is not None:
                    # For each state in the RNN (2 for LSTM)
                    for state_id in range(len(beam_state)):
                        # Update the current batch states with the sub-batch states (in the correct indices)
                        # These indices are specified by sub_batch_ids, the ids of samples which were updated.
                        if isinstance(self.decoder, RNNTDecoder):
                            # LSTM decoder, state is [layer x batch x hidden]
                            beam_state[state_id][:, sub_batch_ids, :] = beam_state_[state_id][...]
                        elif isinstance(self.decoder, StatelessTransducerDecoder):
                            # stateless decoder, state is [batch x hidden]
                            beam_state[state_id][sub_batch_ids, :] = beam_state_[state_id][...]
                        else:
                            raise NotImplementedError("Unknown decoder type.")
                else:
                    # If entire batch was updated, simply update all the states
                    beam_state = beam_state_

                # h_states = list of [t, h[t]]
                # so h[1] here is a h[t] of shape [D]
                # Simply stack all of the h[t] within the sub_batch/batch (T <= beam)
                h_enc = torch.stack([h[1] for h in h_states])  # [T=beam, D]
                h_enc = h_enc.unsqueeze(1)  # [B=beam, T=1, D]; batch over the beams

                # Extract the log probabilities and the predicted tokens
                beam_logp = torch.log_softmax(
                    self.joint.joint(h_enc, beam_y) / self.softmax_temperature, dim=-1
                )  # [B=beam, 1, 1, V + 1]
                beam_logp = beam_logp[:, 0, 0, :]  # [B=beam, V + 1]
                beam_topk = beam_logp[:, ids].topk(beam, dim=-1)

                for j, hyp in enumerate(B_):
                    # For all updated samples in the batch, add it as the blank token
                    # In this step, we dont add a token but simply update score
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(beam_logp[j, self.blank])),
                        y_sequence=hyp.y_sequence[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                        timestep=hyp.timestep[:],
                        length=i,
                    )

                    if self.preserve_alignments:
                        new_hyp.alignments = copy.deepcopy(hyp.alignments)

                        # Add the alignment of blank at this step
                        new_hyp.alignments[-1].append(
                            (beam_logp[j].clone().cpu(), torch.tensor(self.blank, dtype=torch.int32))
                        )

                    # Add blank prediction to A
                    A.append(new_hyp)

                    # If the prediction "timestep" t has reached the length of the input sequence
                    # we can add it to the "finished" hypothesis list.
                    if h_states[j][0] == (h_length - 1):
                        final.append(new_hyp)

                    # Here, we carefully select the indices of the states that we want to preserve
                    # for the next token (non-blank) update.
                    if sub_batch_ids is not None:
                        h_states_idx = sub_batch_ids[j]
                    else:
                        h_states_idx = j

                    # for each current hypothesis j
                    # extract the top token score and top token id for the jth hypothesis
                    for logp, k in zip(beam_topk[0][j], beam_topk[1][j] + index_incr):
                        # create new hypothesis and store in A
                        # Note: This loop does *not* include the blank token!
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            y_sequence=(hyp.y_sequence[:] + [int(k)]),
                            dec_state=self.decoder.batch_select_state(beam_state, h_states_idx),
                            lm_state=hyp.lm_state,
                            timestep=hyp.timestep[:] + [i],
                            length=i,
                        )

                        if self.preserve_alignments:
                            new_hyp.alignments = copy.deepcopy(hyp.alignments)

                            # Add the alignment of Uj for this beam candidate at this step
                            new_hyp.alignments[-1].append(
                                (beam_logp[j].clone().cpu(), torch.tensor(new_hyp.y_sequence[-1], dtype=torch.int32))
                            )

                        A.append(new_hyp)

                # Prune and recombine same hypothesis
                # This may cause next beam to be smaller than max beam size
                # Therefore larger beam sizes may be required for better decoding.
                B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
                B = self.recombine_hypotheses(B)

                if self.preserve_alignments:
                    # convert Ti-th logits into a torch array
                    for B_i in B:
                        # Check if the last token emitted at last timestep was a blank
                        # If so, move to next timestep
                        logp, label = B_i.alignments[-1][-1]  # The last alignment of this step
                        if int(label) == self.blank:
                            B_i.alignments.append([])  # blank buffer for next timestep

            # If B_ is empty list, then we may be able to early exit
            elif len(batch_ids) == len(batch_removal_ids):
                # break early
                break

        if final:
            # Remove trailing empty list of alignments
            if self.preserve_alignments:
                for h in final:
                    if len(h.alignments[-1]) == 0:
                        del h.alignments[-1]

            return self.sort_nbest(final)
        else:
            # Remove trailing empty list of alignments
            if self.preserve_alignments:
                for h in B:
                    if len(h.alignments[-1]) == 0:
                        del h.alignments[-1]

            return B

    def modified_adaptive_expansion_search(
        self, h: torch.Tensor, encoded_lengths: torch.Tensor, partial_hypotheses: Optional[Hypothesis] = None
    ) -> List[Hypothesis]:
        """
        Based on/modified from https://ieeexplore.ieee.org/document/9250505

        Args:
            h: Encoded speech features (1, T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results
        """
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not supported")

        h = h[0]  # [T, D]

        # prepare the batched beam states
        beam = min(self.beam_size, self.vocab_size)
        beam_state = self.decoder.initialize_state(
            torch.zeros(beam, device=h.device, dtype=h.dtype)
        )  # [L, B, H], [L, B, H] for LSTMS

        # Initialize first hypothesis for the beam (blank)
        init_tokens = [
            Hypothesis(
                y_sequence=[self.blank],
                score=0.0,
                dec_state=self.decoder.batch_select_state(beam_state, 0),
                timestep=[-1],
                length=0,
            )
        ]

        cache = {}

        # Initialize alignment buffer
        if self.preserve_alignments:
            for hyp in init_tokens:
                hyp.alignments = [[]]

        # Decode a batch of beam states and scores
        beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score_hypothesis(init_tokens, cache, beam_state)
        state = self.decoder.batch_select_state(beam_state, 0)

        # Setup ngram LM:
        if self.ngram_lm:
            init_lm_state = kenlm.State()
            self.ngram_lm.BeginSentenceWrite(init_lm_state)

        # TODO: Setup LM
        if self.language_model is not None:
            # beam_lm_states, beam_lm_scores = self.lm.buff_predict(
            #     None, beam_lm_tokens, 1
            # )
            # lm_state = select_lm_state(
            #     beam_lm_states, 0, self.lm_layers, self.is_wordlm
            # )
            # lm_scores = beam_lm_scores[0]
            raise NotImplementedError()
        else:
            lm_state = None
            lm_scores = None

        # Initialize first hypothesis for the beam (blank) for kept hypotheses
        kept_hyps = [
            Hypothesis(
                y_sequence=[self.blank],
                score=0.0,
                dec_state=state,
                dec_out=[beam_dec_out[0]],
                lm_state=lm_state,
                lm_scores=lm_scores,
                timestep=[-1],
                length=0,
            )
        ]
        if self.ngram_lm:
            kept_hyps[0].ngram_lm_state = init_lm_state

        # Initialize alignment buffer
        if self.preserve_alignments:
            for hyp in kept_hyps:
                hyp.alignments = [[]]

        for t in range(encoded_lengths):
            enc_out_t = h[t : t + 1].unsqueeze(0)  # [1, 1, D]

            # Perform prefix search to obtain hypothesis
            hyps = self.prefix_search(
                sorted(kept_hyps, key=lambda x: len(x.y_sequence), reverse=True),
                enc_out_t,
                prefix_alpha=self.maes_prefix_alpha,
            )  # type: List[Hypothesis]
            kept_hyps = []

            # Prepare output tensor
            beam_enc_out = enc_out_t

            # List that contains the blank token emisions
            list_b = []
            duplication_check = [hyp.y_sequence for hyp in hyps]

            # Repeat for number of mAES steps
            for n in range(self.maes_num_steps):
                # Pack the decoder logits for all current hypothesis
                beam_dec_out = torch.stack([h.dec_out[-1] for h in hyps])  # [H, 1, D]

                # Extract the log probabilities
                ytm, ilm_ytm = self.resolve_joint_output(beam_enc_out, beam_dec_out)
                beam_logp, beam_idx = ytm.topk(self.max_candidates, dim=-1)

                beam_logp = beam_logp[:, 0, 0, :]  # [B, V + 1]
                beam_idx = beam_idx[:, 0, 0, :]  # [B, max_candidates]

                # Compute k expansions for all the current hypotheses
                k_expansions = select_k_expansions(
                    hyps, beam_idx, beam_logp, self.maes_expansion_gamma, self.maes_expansion_beta
                )

                # List that contains the hypothesis after prefix expansion
                list_exp = []
                for i, hyp in enumerate(hyps):  # For all hypothesis
                    for k, new_score in k_expansions[i]:  # for all expansion within these hypothesis
                        new_hyp = Hypothesis(
                            y_sequence=hyp.y_sequence[:],
                            score=new_score,
                            dec_out=hyp.dec_out[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_scores=hyp.lm_scores,
                            timestep=hyp.timestep[:],
                            length=t,
                        )
                        if self.ngram_lm:
                            new_hyp.ngram_lm_state = hyp.ngram_lm_state

                        # If the expansion was for blank
                        if k == self.blank:
                            list_b.append(new_hyp)
                        else:
                            # If the expansion was a token
                            # new_hyp.y_sequence.append(int(k))
                            if (new_hyp.y_sequence + [int(k)]) not in duplication_check:
                                new_hyp.y_sequence.append(int(k))
                                new_hyp.timestep.append(t)

                                # Setup ngram LM:
                                if self.ngram_lm:
                                    lm_score, new_hyp.ngram_lm_state = self.compute_ngram_score(
                                        hyp.ngram_lm_state, int(k)
                                    )
                                    if self.hat_subtract_ilm:
                                        new_hyp.score += self.ngram_lm_alpha * lm_score - float(
                                            self.hat_ilm_weight * ilm_ytm[i, 0, 0, k]
                                        )
                                    else:
                                        new_hyp.score += self.ngram_lm_alpha * lm_score

                                # TODO: Setup LM
                                if self.language_model is not None:
                                    # new_hyp.score += self.lm_weight * float(
                                    #     hyp.lm_scores[k]
                                    # )
                                    pass

                                list_exp.append(new_hyp)

                        # Preserve alignments
                        if self.preserve_alignments:
                            new_hyp.alignments = copy.deepcopy(hyp.alignments)

                            if k == self.blank:
                                new_hyp.alignments[-1].append(
                                    (beam_logp[i].clone().cpu(), torch.tensor(self.blank, dtype=torch.int32)),
                                )
                            else:
                                new_hyp.alignments[-1].append(
                                    (
                                        beam_logp[i].clone().cpu(),
                                        torch.tensor(new_hyp.y_sequence[-1], dtype=torch.int32),
                                    ),
                                )

                # If there were no token expansions in any of the hypotheses,
                # Early exit
                if not list_exp:
                    kept_hyps = sorted(list_b, key=lambda x: x.score, reverse=True)[:beam]

                    # Update aligments with next step
                    if self.preserve_alignments:
                        # convert Ti-th logits into a torch array
                        for h_i in kept_hyps:
                            # Check if the last token emitted at last timestep was a blank
                            # If so, move to next timestep
                            logp, label = h_i.alignments[-1][-1]  # The last alignment of this step
                            if int(label) == self.blank:
                                h_i.alignments.append([])  # blank buffer for next timestep

                    # Early exit
                    break

                else:
                    # Initialize the beam states for the hypotheses in the expannsion list
                    beam_state = self.decoder.batch_initialize_states(
                        beam_state,
                        [hyp.dec_state for hyp in list_exp],
                        # [hyp.y_sequence for hyp in list_exp],  # <look into when this is necessary>
                    )

                    # Decode a batch of beam states and scores
                    beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score_hypothesis(
                        list_exp,
                        cache,
                        beam_state,
                        # self.language_model is not None,
                    )

                    # TODO: Setup LM
                    if self.language_model is not None:
                        # beam_lm_states = create_lm_batch_states(
                        #     [hyp.lm_state for hyp in list_exp],
                        #     self.lm_layers,
                        #     self.is_wordlm,
                        # )
                        # beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                        #     beam_lm_states, beam_lm_tokens, len(list_exp)
                        # )
                        pass

                    # If this isnt the last mAES step
                    if n < (self.maes_num_steps - 1):
                        # For all expanded hypothesis
                        for i, hyp in enumerate(list_exp):
                            # Preserve the decoder logits for the current beam
                            hyp.dec_out.append(beam_dec_out[i])
                            hyp.dec_state = self.decoder.batch_select_state(beam_state, i)

                            # TODO: Setup LM
                            if self.language_model is not None:
                                # hyp.lm_state = select_lm_state(
                                #     beam_lm_states, i, self.lm_layers, self.is_wordlm
                                # )
                                # hyp.lm_scores = beam_lm_scores[i]
                                pass

                        # Copy the expanded hypothesis
                        hyps = list_exp[:]

                        # Update aligments with next step
                        if self.preserve_alignments:
                            # convert Ti-th logits into a torch array
                            for h_i in hyps:
                                # Check if the last token emitted at last timestep was a blank
                                # If so, move to next timestep
                                logp, label = h_i.alignments[-1][-1]  # The last alignment of this step
                                if int(label) == self.blank:
                                    h_i.alignments.append([])  # blank buffer for next timestep

                    else:
                        # Extract the log probabilities
                        beam_logp, _ = self.resolve_joint_output(beam_enc_out, beam_dec_out)
                        beam_logp = beam_logp[:, 0, 0, :]

                        # For all expansions, add the score for the blank label
                        for i, hyp in enumerate(list_exp):
                            hyp.score += float(beam_logp[i, self.blank])

                            # Preserve the decoder's output and state
                            hyp.dec_out.append(beam_dec_out[i])
                            hyp.dec_state = self.decoder.batch_select_state(beam_state, i)

                            # TODO: Setup LM
                            if self.language_model is not None:
                                # hyp.lm_state = select_lm_state(
                                #     beam_lm_states, i, self.lm_layers, self.is_wordlm
                                # )
                                # hyp.lm_scores = beam_lm_scores[i]
                                pass

                        # Finally, update the kept hypothesis of sorted top Beam candidates
                        kept_hyps = sorted(list_b + list_exp, key=lambda x: x.score, reverse=True)[:beam]

                        # Update aligments with next step
                        if self.preserve_alignments:
                            # convert Ti-th logits into a torch array
                            for h_i in kept_hyps:
                                # Check if the last token emitted at last timestep was a blank
                                # If so, move to next timestep
                                logp, label = h_i.alignments[-1][-1]  # The last alignment of this step
                                if int(label) == self.blank:
                                    h_i.alignments.append([])  # blank buffer for next timestep

        # Remove trailing empty list of alignments
        if self.preserve_alignments:
            for h in kept_hyps:
                if len(h.alignments[-1]) == 0:
                    del h.alignments[-1]

        # Sort the hypothesis with best scores
        return self.sort_nbest(kept_hyps)

    def recombine_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Recombine hypotheses with equivalent output sequence.

        Args:
            hypotheses (list): list of hypotheses

        Returns:
           final (list): list of recombined hypotheses
        """
        final = []

        for hyp in hypotheses:
            seq_final = [f.y_sequence for f in final if f.y_sequence]

            if hyp.y_sequence in seq_final:
                seq_pos = seq_final.index(hyp.y_sequence)

                final[seq_pos].score = np.logaddexp(final[seq_pos].score, hyp.score)
            else:
                final.append(hyp)

        return hypotheses

    def resolve_joint_output(self, enc_out: torch.Tensor, dec_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resolve output types for RNNT and HAT joint models
        """

        joint_output = self.joint.joint(enc_out, dec_out)
        if torch.is_tensor(joint_output):
            ytm = torch.log_softmax(joint_output / self.softmax_temperature, dim=-1)
            ilm_ytm = None
        elif self.hat_subtract_ilm and isinstance(joint_output, HATJointOutput):
            ytm, ilm_ytm = joint_output.hat_logprobs, joint_output.ilm_logprobs
        else:
            raise TypeError(
                f"Joint output ({type(joint_output)}) must be torch.Tensor or HATJointOutput in case of HAT joint"
            )

        return ytm, ilm_ytm

    def prefix_search(
        self, hypotheses: List[Hypothesis], enc_out: torch.Tensor, prefix_alpha: int
    ) -> List[Hypothesis]:
        """
        Prefix search for NSC and mAES strategies.
        Based on https://arxiv.org/pdf/1211.3711.pdf
        """

        for j, hyp_j in enumerate(hypotheses[:-1]):
            for hyp_i in hypotheses[(j + 1) :]:
                curr_id = len(hyp_j.y_sequence)
                pref_id = len(hyp_i.y_sequence)

                if is_prefix(hyp_j.y_sequence, hyp_i.y_sequence) and (curr_id - pref_id) <= prefix_alpha:
                    logp, ilm_logp = self.resolve_joint_output(enc_out, hyp_i.dec_out[-1])
                    logp = logp[0, 0, 0, :]
                    curr_score = hyp_i.score + float(logp[hyp_j.y_sequence[pref_id]])
                    # Setup ngram LM:
                    if self.ngram_lm:
                        lm_score, next_state = self.compute_ngram_score(
                            hyp_i.ngram_lm_state, int(hyp_j.y_sequence[pref_id])
                        )
                        if self.hat_subtract_ilm:
                            curr_score += self.ngram_lm_alpha * lm_score - self.hat_ilm_weight * float(
                                ilm_logp[0, 0, hyp_j.y_sequence[pref_id]]
                            )
                        else:
                            curr_score += self.ngram_lm_alpha * lm_score

                    for k in range(pref_id, (curr_id - 1)):
                        logp, ilm_logp = self.resolve_joint_output(enc_out, hyp_j.dec_out[k])
                        logp = logp[0, 0, 0, :]
                        curr_score += float(logp[hyp_j.y_sequence[k + 1]])
                        # Setup ngram LM:
                        if self.ngram_lm:
                            lm_score, next_state = self.compute_ngram_score(next_state, int(hyp_j.y_sequence[k + 1]))
                            if self.hat_subtract_ilm:
                                curr_score += self.ngram_lm_alpha * lm_score - self.hat_ilm_weight * float(
                                    ilm_logp[0, 0, hyp_j.y_sequence[k + 1]]
                                )
                            else:
                                curr_score += self.ngram_lm_alpha * lm_score

                    hyp_j.score = np.logaddexp(hyp_j.score, curr_score)

        return hypotheses

    def compute_ngram_score(self, current_lm_state: "kenlm.State", label: int) -> Tuple[float, "kenlm.State"]:
        """
        Score computation for kenlm ngram language model.
        """

        if self.token_offset:
            label = chr(label + self.token_offset)
        else:
            label = str(label)
        next_state = kenlm.State()
        lm_score = self.ngram_lm.BaseScore(current_lm_state, label, next_state)
        lm_score *= 1.0 / np.log10(np.e)

        return lm_score, next_state

    def set_decoding_type(self, decoding_type: str):

        # Please check train_kenlm.py in scripts/asr_language_modeling/ to find out why we need
        # TOKEN_OFFSET for BPE-based models
        if decoding_type == 'subword':
            from nemo.collections.asr.parts.submodules.ctc_beam_decoding import DEFAULT_TOKEN_OFFSET

            self.token_offset = DEFAULT_TOKEN_OFFSET


@dataclass
class BeamRNNTInferConfig:
    beam_size: int
    search_type: str = 'default'
    score_norm: bool = True
    return_best_hypothesis: bool = True
    tsd_max_sym_exp_per_step: Optional[int] = 50
    alsd_max_target_len: float = 1.0
    nsc_max_timesteps_expansion: int = 1
    nsc_prefix_alpha: int = 1
    maes_num_steps: int = 2
    maes_prefix_alpha: int = 1
    maes_expansion_gamma: float = 2.3
    maes_expansion_beta: int = 2
    language_model: Optional[Dict[str, Any]] = None
    softmax_temperature: float = 1.0
    preserve_alignments: bool = False
    ngram_lm_model: Optional[str] = None
    ngram_lm_alpha: Optional[float] = 0.0
    hat_subtract_ilm: bool = False
    hat_ilm_weight: float = 0.0
