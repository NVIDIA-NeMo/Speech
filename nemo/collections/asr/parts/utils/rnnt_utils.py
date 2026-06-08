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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from nemo.collections.asr.parts.context_biasing.biasing_multi_model import BiasingRequestItemConfig
from nemo.collections.asr.parts.submodules.transducer_decoding import BatchedHyps


@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms.

    score: A float score obtained from an AbstractRNNTDecoder module's score_hypothesis method.

    y_sequence: Either a sequence of integer ids pointing to some vocabulary, or a packed torch.Tensor
        behaving in the same manner. dtype must be torch.Long in the latter case.

    dec_state: A list (or list of list) of LSTM-RNN decoder states. Can be None.

    text: (Optional) A decoded string after processing via CTC / RNN-T decoding (removing the CTC/RNNT
        `blank` tokens, and optionally merging word-pieces). Should be used as decoded string for
        Word Error Rate calculation.

    timestamp: (Optional) A list of integer indices representing at which index in the decoding
        process did the token appear. Should be of same length as the number of non-blank tokens.

    alignments: (Optional) Represents the CTC / RNNT token alignments as integer tokens along an axis of
        time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of integer indices.
        For RNNT, represented as a dangling list of list of integer indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).
        The set of valid indices **includes** the CTC / RNNT blank token in order to represent alignments.

    frame_confidence: (Optional) Represents the CTC / RNNT per-frame confidence scores as token probabilities
        along an axis of time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of float indices.
        For RNNT, represented as a dangling list of list of float indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).

    token_confidence: (Optional) Represents the CTC / RNNT per-token confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    word_confidence: (Optional) Represents the CTC / RNNT per-word confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    length: Represents the length of the sequence (the original length without padding), otherwise
        defaults to 0.

    y: (Unused) A list of torch.Tensors representing the list of hypotheses.

    lm_state: (Unused) A dictionary state cache used by an external Language Model.

    lm_scores: (Unused) Score of the external Language Model.

    ngram_lm_state: (Optional) State of the external n-gram Language Model.

    tokens: (Optional) A list of decoded tokens (can be characters or word-pieces.

    last_token (Optional): A token or batch of tokens which was predicted in the last step.

    last_frame (Optional): Index of the last decoding step hypothesis was updated including blank token prediction.

    xatt_scores (Optional): List of cross-attention scores for each decoder layer. Each element of the list is a
        Tensor of shape num heads x decoder input len x encoder output len (HxUxT). This is useful only for AED models.
    """

    score: float
    y_sequence: Union[List[int], torch.Tensor]
    text: Optional[str] = None
    dec_out: Optional[List[torch.Tensor]] = None
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
    timestamp: Union[List[int], torch.Tensor] = field(default_factory=list)
    alignments: Optional[Union[List[int], List[List[int]]]] = None
    frame_confidence: Optional[Union[List[float], List[List[float]]]] = None
    token_confidence: Optional[List[float]] = None
    word_confidence: Optional[List[float]] = None
    length: Union[int, torch.Tensor] = 0
    y: List[torch.tensor] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    lm_scores: Optional[torch.Tensor] = None
    ngram_lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    tokens: Optional[Union[List[int], torch.Tensor]] = None
    last_token: Optional[torch.Tensor] = None
    token_duration: Optional[torch.Tensor] = None
    last_frame: Optional[int] = None
    biasing_cfg: BiasingRequestItemConfig | None = None
    non_blank_step_confidence_precomputed: list[float] | None = None
    xatt_scores: Optional[List[torch.Tensor]] = None

    @property
    def non_blank_frame_confidence(self) -> List[float]:
        """Get per-frame confidence for non-blank tokens according to self.timestamp

        Returns:
            List with confidence scores. The length of the list is the same as `timestamp`.
        """
        if self.non_blank_step_confidence_precomputed is not None:
            return self.non_blank_step_confidence_precomputed

        non_blank_frame_confidence = []
        # self.timestamp can be a dict for RNNT
        timestamp = self.timestamp['timestep'] if isinstance(self.timestamp, dict) else self.timestamp
        if len(timestamp) != 0 and self.frame_confidence is not None:
            if any(isinstance(i, list) for i in self.frame_confidence):  # rnnt
                t_prev = -1
                offset = 0
                for t in timestamp:
                    if t != t_prev:
                        t_prev = t
                        offset = 0
                    else:
                        offset += 1
                    non_blank_frame_confidence.append(self.frame_confidence[t][offset])
            else:  # ctc
                non_blank_frame_confidence = [self.frame_confidence[t] for t in timestamp]
        return non_blank_frame_confidence

    @property
    def words(self) -> List[str]:
        """Get words from self.text

        Returns:
            List with words (str).
        """
        return [] if self.text is None else self.text.split()

    def merge_(self, other: "Hypothesis") -> "Hypothesis":
        """Merge (inplace) current hypothesis with another one."""
        self.score += other.score
        if self.y_sequence is None:
            self.y_sequence = other.y_sequence
        elif isinstance(self.y_sequence, torch.Tensor):
            self.y_sequence = torch.cat((self.y_sequence, other.y_sequence), dim=0)
        else:
            self.y_sequence.extend(other.y_sequence)
        self.dec_state = other.dec_state
        if self.timestamp is None:
            self.timestamp = other.timestamp
        elif isinstance(self.timestamp, torch.Tensor):
            self.timestamp = torch.cat((self.timestamp, other.timestamp), dim=0)
        else:
            self.timestamp.extend(other.timestamp)
        self.length += other.length
        self.last_token = other.last_token
        if self.alignments is None:
            self.alignments = other.alignments
        else:
            self.alignments.extend(other.alignments)
        if self.frame_confidence is None:
            self.frame_confidence = other.frame_confidence
        else:
            self.frame_confidence.extend(other.frame_confidence)
        # Invalidated. Need to rerun decode_hypothesis here.
        self.text = None
        self.biasing_cfg = other.biasing_cfg or self.biasing_cfg
        if self.non_blank_step_confidence_precomputed is None:
            self.non_blank_step_confidence_precomputed = other.non_blank_step_confidence_precomputed
        else:
            # non_blank_step_confidence_precomputed should be filled in by the decoding algorithm
            # should be consistent between hyps
            assert other.non_blank_step_confidence_precomputed is not None
            self.non_blank_step_confidence_precomputed.extend(other.non_blank_step_confidence_precomputed)
        return self

    def clean_decoding_state_(self):
        """Clean the decoding state to save memory."""
        self.dec_state = None

    def has_biasing_request(self) -> bool:
        """Return True if contains non-empty biasing request"""
        return self.biasing_cfg is not None and (not self.biasing_cfg.is_empty())

    @classmethod
    def empty_with_biasing_cfg(cls, biasing_cfg: BiasingRequestItemConfig):
        """Constructor of empty hypothesis with biasing request"""
        return cls(y_sequence=[], score=0.0, biasing_cfg=biasing_cfg)


@dataclass
class NBestHypotheses:
    """List of N best hypotheses"""

    n_best_hypotheses: Optional[List[Hypothesis]]


@dataclass
class HATJointOutput:
    """HATJoint outputs for beam search decoding

    hat_logprobs: standard HATJoint outputs as for RNNTJoint

    ilm_logprobs: internal language model probabilities (for ILM subtraction)
    """

    hat_logprobs: Optional[torch.Tensor] = None
    ilm_logprobs: Optional[torch.Tensor] = None


def is_prefix(x: List[int], pref: List[int]) -> bool:
    """
    Obtained from https://github.com/espnet/espnet.

    Check if pref is a prefix of x.

    Args:
        x: Label ID sequence.
        pref: Prefix label ID sequence.

    Returns:
        : Whether pref is a prefix of x.
    """
    if len(pref) >= len(x):
        return False

    for i in range(len(pref)):
        if pref[i] != x[i]:
            return False

    return True


def select_k_expansions(
    hyps: List[Hypothesis],
    topk_idxs: torch.Tensor,
    topk_logps: torch.Tensor,
    gamma: float,
    beta: int,
) -> List[Tuple[int, Hypothesis]]:
    """
    Obtained from https://github.com/espnet/espnet

    Return K hypotheses candidates for expansion from a list of hypothesis.
    K candidates are selected according to the extended hypotheses probabilities
    and a prune-by-value method. Where K is equal to beam_size + beta.

    Args:
        hyps: Hypotheses.
        topk_idxs: Indices of candidates hypothesis. Shape = [B, num_candidates]
        topk_logps: Log-probabilities for hypotheses expansions. Shape = [B, V + 1]
        gamma: Allowed logp difference for prune-by-value method.
        beta: Number of additional candidates to store.

    Return:
        k_expansions: Best K expansion hypotheses candidates.
    """
    k_expansions = []

    for i, hyp in enumerate(hyps):
        hyp_i = [(int(k), hyp.score + float(v)) for k, v in zip(topk_idxs[i], topk_logps[i])]
        k_best_exp_val = max(hyp_i, key=lambda x: x[1])

        k_best_exp_idx = k_best_exp_val[0]
        k_best_exp = k_best_exp_val[1]

        expansions = sorted(
            filter(lambda x: (k_best_exp - gamma) <= x[1], hyp_i),
            key=lambda x: x[1],
        )

        if len(expansions) > 0:
            k_expansions.append(expansions)
        else:
            k_expansions.append([(k_best_exp_idx, k_best_exp)])

    return k_expansions


def batched_hyps_to_hypotheses(batched_hyps: BatchedHyps, batch_size=None) -> list[Hypothesis]:
    """
    Convert batched hypotheses to a list of Hypothesis objects.
    Keep this function separate to allow for jit compilation for BatchedHyps class (see tests)

    Args:
        batched_hyps: BatchedHyps object
        batch_size: Batch Size to retrieve hypotheses. When working with CUDA graphs the batch size for all tensors
            is constant, thus we need here the real batch size to return only necessary hypotheses

    Returns:
        list of Hypothesis objects
    """
    assert batch_size is None or batch_size <= batched_hyps.scores.shape[0]
    num_hyps = batched_hyps.scores.shape[0] if batch_size is None else batch_size
    # NB: clone is not necessary anymore, since CUDA graph decoder always returns an independent copy
    scores = batched_hyps.scores.cpu()

    lengths_nb, transcript_nb, timestamps_nb, durations_nb, step_confidence_nb = batched_hyps.get_data_without_blank()
    lengths_nb, transcript_nb, timestamps_nb = lengths_nb.cpu(), transcript_nb.cpu(), timestamps_nb.cpu()
    if durations_nb is not None:
        durations_nb = durations_nb.cpu()
    if step_confidence_nb is not None:
        step_confidence_nb = step_confidence_nb.cpu()
    lengths_nb = lengths_nb.tolist()
    hypotheses = [
        Hypothesis(
            score=scores[i].item(),
            y_sequence=transcript_nb[i, : lengths_nb[i]],
            timestamp=timestamps_nb[i, : lengths_nb[i]],
            token_duration=(durations_nb[i, : lengths_nb[i]] if durations_nb is not None else torch.empty(0)),
            alignments=None,
            dec_state=None,
            non_blank_step_confidence_precomputed=(
                step_confidence_nb[i, : lengths_nb[i]].tolist() if step_confidence_nb is not None else None
            ),
        )
        for i in range(num_hyps)
    ]

    if batched_hyps.with_blank_steps:
        if batched_hyps.with_logits:
            logits = batched_hyps.logits.cpu()
        if batched_hyps.with_step_confidence:
            step_confidence = batched_hyps.step_confidence.cpu()
        labels = batched_hyps.transcript.cpu()

        # for each hypothesis - aggregate alignment using unique_consecutive for time indices (~itertools.groupby)
        for i in range(len(hypotheses)):
            if batched_hyps.with_logits:
                hypotheses[i].alignments = []
            if batched_hyps.with_step_confidence:
                hypotheses[i].frame_confidence = []
            _, grouped_counts = torch.unique_consecutive(
                batched_hyps.timestamps[i, : batched_hyps.current_lengths[i]], return_counts=True
            )
            start = 0
            for timestamp_cnt in grouped_counts.tolist():
                if batched_hyps.with_logits:
                    hypotheses[i].alignments.append(
                        [(logits[i, start + j], labels[i, start + j]) for j in range(timestamp_cnt)]
                    )
                if batched_hyps.with_step_confidence:
                    hypotheses[i].frame_confidence.append(
                        [step_confidence[i, start + j] for j in range(timestamp_cnt)]
                    )
                start += timestamp_cnt
    return hypotheses
