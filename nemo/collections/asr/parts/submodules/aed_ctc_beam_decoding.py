# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Time-synchronous beam search over CTC frames with the attention decoder's label scores fused in.

The search walks the CTC output frame by frame.
Every hypothesis is an alignment path.
Its score is the sum of the chosen frame labels' CTC log-probabilities (`ctc_weight`)
and, whenever the path starts a new label (a non-blank frame label after a blank or after a different label),
the attention decoder's log-probability of that label given the labels so far (`aed_weight`).
Paths with the same label sequence are recombined
(`recombination`: `max` keeps the best path's score, `sum` the log-sum of the paths),
and the decoder's end-of-sequence score closes every path after the last frame.
Runs of frames that share a confident argmax label can be merged beforehand (`ctc_soft_collapse_threshold`),
which shortens the loop and lets a path change its label at every remaining frame.

The decoder runs over fixed-capacity key/value caches with one position per emitted label,
so hypotheses with different numbers of labels share one batched step,
and the cross-attention projections are computed once per sequence.
The step reuses the decoder's own modules
(embedding, norms, projections, feed-forward, rotary tables and the output layer)
and only re-expresses the attention over the caches,
so it applies to every `TransformerDecoderNM` with pre-LayerNorm blocks and absolute or rotary positions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from nemo.collections.asr.parts.submodules.multitask_beam_decoding import AEDBeamInfer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core import Typing, typecheck
from nemo.core.neural_types import ChannelType, HypothesisType, LabelsType, LogprobsType, MaskType, NeuralType

__all__ = [
    "AEDCTCBeamInferConfig",
    "AEDCTCBeamSearchResult",
    "AEDCTCTimeSyncBeamSearch",
    "TransformerAEDCTCBeamInfer",
    "soft_collapse_repeated",
]

NEG_INF = float("-inf")


class TransformerAEDCTCBeamInfer(AEDBeamInfer, Typing):
    """The joint CTC and attention-decoder beam search as the decoding engine of an AED model with a CTC head.

    `forward` needs the CTC log-probabilities of the batch next to the encoder states.
    The hypotheses hold the emitted labels without the prompt
    and without the tokenizer's boundary and padding ids, which the CTC head can produce as frame labels;
    `partial_hypotheses` are not supported.

    Args:
        transformer_decoder: a `TransformerDecoderNM` with pre-LayerNorm blocks
        log_softmax_module: the output layer, a `TokenClassifier` returning log-probabilities
        tokenizer: provides `bos`, `eos` and `pad`
        search_type: unused, kept for the common engine interface
        beam_size: hypotheses kept per sequence after every frame
        aed_weight: scale of the decoder's label log-probabilities
        ctc_weight: scale of the CTC frame log-probabilities
        recombination: `max`, `sum` or None, see `AEDCTCTimeSyncBeamSearch`
        ctc_soft_collapse_threshold: merge runs of confident repeated CTC frames before the search
            (`soft_collapse_repeated`); None searches over all frames
        length_normalization: exponent of the hypothesis length dividing the final scores
        return_best_hypothesis: return the best hypothesis per sequence instead of the whole beam
        preserve_alignments: keep the frame labels of every hypothesis in `Hypothesis.alignments`
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {
            "encoder_hidden_states": NeuralType(tuple(('B', 'T', 'D')), ChannelType()),
            "encoder_input_mask": NeuralType(tuple(('B', 'T')), MaskType()),
            "decoder_input_ids": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "partial_hypotheses": NeuralType(optional=True),
            "ctc_log_probs": NeuralType(('B', 'T', 'C'), LogprobsType(), optional=True),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(
        self,
        transformer_decoder: torch.nn.Module,
        log_softmax_module: torch.nn.Module,
        tokenizer: TokenizerSpec,
        search_type: str = 'default',
        beam_size: int = 4,
        aed_weight: float = 1.0,
        ctc_weight: float = 1.0,
        recombination: Optional[str] = "max",
        ctc_soft_collapse_threshold: Optional[float] = None,
        length_normalization: float = 0.0,
        return_best_hypothesis: bool = True,
        preserve_alignments: bool = False,
    ):
        super().__init__(
            transformer_decoder=transformer_decoder,
            log_softmax_module=log_softmax_module,
            tokenizer=tokenizer,
            search_type=search_type,
            return_best_hypothesis=return_best_hypothesis,
            preserve_alignments=preserve_alignments,
        )
        self.beam_size = beam_size
        self.ctc_soft_collapse_threshold = ctc_soft_collapse_threshold
        self.special_ids = torch.tensor(sorted({tokenizer.bos, tokenizer.eos, tokenizer.pad}), dtype=torch.long)
        self.beam_search = AEDCTCTimeSyncBeamSearch(
            transformer_decoder,
            log_softmax_module,
            beam_size=beam_size,
            bos=tokenizer.bos,
            eos=tokenizer.eos,
            pad=tokenizer.pad,
            aed_weight=aed_weight,
            ctc_weight=ctc_weight,
            recombination=recombination,
            length_normalization=length_normalization,
        )

    @typecheck()
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_input_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
        ctc_log_probs: Optional[torch.Tensor] = None,
    ):
        """Returns a list of hypotheses given an input batch of encoder states and CTC log-probabilities.

        Args:
            encoder_hidden_states: encoder output of shape (B, T, D)
            encoder_input_mask: nonzero for valid frames, shape (B, T)
            decoder_input_ids: decoder prompt of shape (B, P)
            partial_hypotheses: not supported
            ctc_log_probs: log-probabilities of the CTC head over the same frames, shape (B, T, C)

        Returns:
            A tuple with the list of hypotheses of the batch,
            each a `Hypothesis` or, when the whole beam is returned, an `NBestHypotheses`.
        """
        if ctc_log_probs is None:
            raise ValueError("The joint AED and CTC beam search needs the CTC log-probabilities of the batch")
        if ctc_log_probs.shape[1] != encoder_hidden_states.shape[1]:
            raise ValueError("The CTC log-probabilities must cover the encoder frames")
        with torch.inference_mode():
            self.transformer_decoder.eval()
            self.log_softmax_module.eval()
            lengths = encoder_input_mask.to(torch.bool).sum(dim=1)
            if self.ctc_soft_collapse_threshold is not None:
                ctc_log_probs, lengths = soft_collapse_repeated(
                    ctc_log_probs, lengths, self.ctc_soft_collapse_threshold
                )
            result = self.beam_search(
                encoder_hidden_states,
                encoder_input_mask,
                ctc_log_probs,
                lengths,
                decoder_input_ids,
                return_alignments=self.preserve_alignments,
            )
            labels, lengths, scores = result.labels.cpu(), result.lengths.cpu(), result.scores.cpu()
            alignments = result.alignments.cpu() if result.alignments is not None else None
            packed_result = []
            for b in range(labels.shape[0]):
                hypotheses = [
                    self._hypothesis(
                        labels[b, k, : lengths[b, k]],
                        float(scores[b, k]),
                        alignments[b, k].tolist() if alignments is not None else None,
                    )
                    for k in range(self.beam_size)
                    if k == 0 or torch.isfinite(scores[b, k])  # recombined-away paths carry no hypothesis
                ]
                packed_result.append(hypotheses[0] if self.return_best_hypothesis else NBestHypotheses(hypotheses))
        self.transformer_decoder.train()
        self.log_softmax_module.train()
        return (packed_result,)

    def _hypothesis(self, labels: torch.Tensor, score: float, alignment: Optional[List[int]]) -> Hypothesis:
        labels = labels[~torch.isin(labels, self.special_ids)]
        return Hypothesis(score=score, y_sequence=labels.long(), timestamp=[], alignments=alignment)


@dataclass
class AEDCTCBeamInferConfig:
    """Options of the joint AED and CTC beam search, see `AEDCTCTimeSyncBeamSearch`.

    `ctc_soft_collapse_threshold` merges runs of CTC frames with a confident shared argmax label
    before the search (`soft_collapse_repeated`); None searches over all frames.
    """

    beam_size: int = 4
    search_type: str = 'default'
    aed_weight: float = 1.0
    ctc_weight: float = 1.0
    recombination: Optional[str] = "max"
    ctc_soft_collapse_threshold: Optional[float] = None
    length_normalization: float = 0.0
    return_best_hypothesis: bool = True
    preserve_alignments: bool = False


@dataclass
class AEDCTCBeamSearchResult:
    """The beam of every sequence after the last frame, sorted by score.

    Attributes:
        labels: emitted label ids without the prompt, shape (B, beam, L), `pad` beyond each hypothesis
        lengths: number of labels of every hypothesis, shape (B, beam)
        scores: ranking scores, shape (B, beam); `-inf` marks a hypothesis that was recombined into another one
        alignments: frame labels including blank, shape (B, beam, T), or None when not requested
    """

    labels: torch.Tensor
    lengths: torch.Tensor
    scores: torch.Tensor
    alignments: Optional[torch.Tensor]


class AEDCTCTimeSyncBeamSearch:
    """Time-synchronous joint CTC and attention-decoder beam search, see the module docstring.

    The CTC blank is the class after the decoder vocabulary, where NeMo's CTC decoders place it.

    Args:
        transformer_decoder: a `TransformerDecoderNM`
            (or a module with the same `embedding`, `decoder` and `vocab_size`) with pre-LayerNorm blocks
        log_softmax_module: the output layer, a `TokenClassifier` returning log-probabilities
        beam_size: hypotheses kept per sequence after every frame
        bos: starts the decoder when no prompt is given
        eos: the end-of-sequence label whose decoder score closes every hypothesis after the last frame
        pad: fills the returned label tensors beyond each hypothesis
        aed_weight: scale of the decoder's label log-probabilities, including the end-of-sequence score
        ctc_weight: scale of the CTC frame log-probabilities
        recombination: `max` keeps the best of the paths with the same label sequence,
            `sum` their log-sum, None keeps all paths apart
        length_normalization: exponent of the hypothesis length (labels plus the end token)
            that divides the final scores before ranking the beam; 0 ranks by the joint score
    """

    def __init__(
        self,
        transformer_decoder: torch.nn.Module,
        log_softmax_module: torch.nn.Module,
        *,
        beam_size: int,
        bos: int,
        eos: int,
        pad: int,
        aed_weight: float = 1.0,
        ctc_weight: float = 1.0,
        recombination: Optional[str] = "max",
        length_normalization: float = 0.0,
    ):
        if beam_size < 1:
            raise ValueError(f"beam_size must be positive, got {beam_size}")
        if recombination not in (None, "max", "sum"):
            raise ValueError(f"recombination must be 'max', 'sum' or None, got {recombination!r}")
        stack = transformer_decoder.decoder
        if not all(layer.pre_ln for layer in stack.layers):
            raise ValueError("The joint AED and CTC beam search needs pre-LayerNorm decoder blocks")
        if not log_softmax_module.log_softmax:
            raise ValueError("The output layer must return log-probabilities")
        self.embedding = transformer_decoder.embedding
        self.layers = stack.layers
        self.final_norm = stack.final_layer_norm
        self.classifier = log_softmax_module
        self.vocab_size = transformer_decoder.vocab_size
        self.blank = self.vocab_size
        attention = self.layers[0].first_sub_layer
        self.heads = attention.num_attention_heads
        self.head_dim = attention.attn_head_size
        self.beam_size = beam_size
        self.bos, self.eos, self.pad = bos, eos, pad
        self.aed_weight = aed_weight
        self.ctc_weight = ctc_weight
        self.recombination = recombination
        self.length_normalization = length_normalization

    @torch.no_grad()
    def __call__(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_input_mask: torch.Tensor,
        ctc_log_probs: torch.Tensor,
        ctc_lengths: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        return_alignments: bool = False,
    ) -> AEDCTCBeamSearchResult:
        """Run the search over a batch.

        Args:
            encoder_hidden_states: encoder output of shape (B, T, D)
            encoder_input_mask: nonzero for valid encoder frames, shape (B, T)
            ctc_log_probs: CTC log-probabilities with the blank class last, shape (B, T', C),
                for example merged by `soft_collapse_repeated`
            ctc_lengths: valid CTC frames per sequence, shape (B,)
            decoder_input_ids: decoder prompt of shape (B, P), shared by the hypotheses of a sequence;
                one `bos` per sequence when None
            return_alignments: also return the frame labels of every hypothesis

        Returns:
            The sorted beam of every sequence.
        """
        batch = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device
        dtype = encoder_hidden_states.dtype
        beam = self.beam_size
        rows = batch * beam
        classes = ctc_log_probs.shape[-1]
        if classes != self.blank + 1:
            raise ValueError(
                f"Expected {self.blank + 1} CTC classes (the decoder vocabulary plus blank), got {classes}"
            )
        if decoder_input_ids is None:
            decoder_input_ids = torch.full((batch, 1), self.bos, dtype=torch.long, device=device)
        prompt_length = decoder_input_ids.shape[1]
        if prompt_length < 1:
            raise ValueError("The decoder prompt needs at least one token")
        frames = int(ctc_lengths.max()) if batch else 0
        capacity = prompt_length + frames  # at most one label per frame
        if capacity > self.embedding.max_sequence_length:
            raise ValueError(
                f"The prompt and {frames} CTC frames need {capacity} decoder positions, "
                f"but the embedding has {self.embedding.max_sequence_length}"
            )
        encoder_mask = encoder_input_mask.to(torch.bool)
        encoder = self._encoder_projections(encoder_hidden_states)
        for layer in self.layers:
            rotary = getattr(layer.first_sub_layer, "rotary_pos_enc", None)
            if rotary is not None:
                rotary.extend_pe(capacity, device, dtype)
        # Two cache sets: every frame reorders one into the other, see _reorder.
        caches, spare = (
            [
                (
                    torch.zeros(rows, capacity, self.heads, self.head_dim, device=device, dtype=dtype),
                    torch.zeros(rows, capacity, self.heads, self.head_dim, device=device, dtype=dtype),
                )
                for _ in self.layers
            ]
            for _ in range(2)
        )
        row_index = torch.arange(rows, device=device)
        batch_offset = (torch.arange(batch, device=device) * beam)[:, None]
        class_index = torch.arange(classes, device=device)

        # The prompt fills the caches; its last token's log-probabilities score the first label.
        positions = torch.zeros(rows, dtype=torch.long, device=device)
        prompt = decoder_input_ids.repeat_interleave(beam, dim=0)
        label_log_probs = None
        for step in range(prompt_length):
            label_log_probs = self._decoder_step(prompt[:, step], positions, caches, encoder, encoder_mask)
            positions += 1
        label_log_probs = label_log_probs * self.aed_weight

        score = torch.full((batch, beam), NEG_INF, device=device, dtype=dtype)
        score[:, 0] = 0.0  # one live path per sequence at the start
        score = score.view(rows)
        last_frame_label = torch.full((rows,), self.blank, dtype=torch.long, device=device)
        last_label = prompt[:, -1].clone()
        history = torch.zeros(rows, max(frames, 1), dtype=torch.long, device=device)
        history_length = torch.zeros(rows, dtype=torch.long, device=device)
        blank_only = torch.full((classes,), NEG_INF, device=device, dtype=dtype)
        blank_only[self.blank] = 0.0
        frame_labels, backrefs = [], []

        for t in range(frames):
            # Beyond a sequence's frames only blank continues a path, at no cost.
            ctc_t = torch.where((t < ctc_lengths)[:, None], ctc_log_probs[:, t], blank_only)
            # A frame label starts a new label after blank or when it differs from the previous frame label;
            # the decoder scores it then, blank and repeated labels add nothing.
            starts_label = (last_frame_label == self.blank)[:, None] | (
                class_index[None, :] != last_frame_label[:, None]
            )
            label_scores = torch.cat([label_log_probs, label_log_probs.new_zeros(rows, 1)], dim=1)
            ext = score[:, None] + self.ctc_weight * ctc_t.repeat_interleave(beam, dim=0)
            ext = ext + torch.where(starts_label, label_scores, 0.0)
            score, flat = torch.topk(ext.view(batch, beam * classes), beam, dim=1)
            score = score.view(rows)
            source = flat // classes  # (B, beam) -> previous beam index
            frame_label = (flat % classes).view(rows)
            source_rows = (source + batch_offset).view(rows)

            last_frame_label = last_frame_label.index_select(0, source_rows)
            last_label = last_label.index_select(0, source_rows)
            history = history.index_select(0, source_rows)
            history_length = history_length.index_select(0, source_rows)
            positions = positions.index_select(0, source_rows)
            label_log_probs = label_log_probs.index_select(0, source_rows)
            used = min(capacity, prompt_length + t + 1)  # slots written so far, plus the one this frame may write
            for (keys, values), (spare_keys, spare_values) in zip(caches, spare):
                self._reorder(keys, spare_keys, source_rows, used)
                self._reorder(values, spare_values, source_rows, used)
            caches, spare = spare, caches

            new_label = (frame_label != self.blank) & (frame_label != last_frame_label)
            label = torch.where(new_label, frame_label, last_label)
            history[row_index, history_length] = torch.where(new_label, label, history[row_index, history_length])
            history_length = history_length + new_label.long()
            if self.recombination:
                score, new_label = self._recombine(score, new_label, history, history_length, batch)

            # The decoder runs for every path and the result is kept where the path got a new label;
            # the other paths' slot at their position is written again by their next label before it is read.
            new_log_probs = self._decoder_step(label, positions, caches, encoder, encoder_mask)
            label_log_probs = torch.where(new_label[:, None], new_log_probs * self.aed_weight, label_log_probs)
            positions = positions + new_label.long()
            last_frame_label = frame_label
            last_label = label
            if return_alignments:
                frame_labels.append(frame_label.view(batch, beam))
                backrefs.append(source)

        score = score + label_log_probs[:, self.eos]  # every path ends after the last frame
        ranking = score
        if self.length_normalization:
            ranking = score / (history_length + 1).to(score.dtype) ** self.length_normalization
        ranking = ranking.view(batch, beam)
        order = ranking.argsort(dim=1, descending=True)
        order_rows = (order + batch_offset).view(rows)
        lengths = history_length.index_select(0, order_rows)
        max_length = int(lengths.max()) if rows else 0
        labels = history.index_select(0, order_rows)[:, :max_length]
        labels = labels.masked_fill(torch.arange(max_length, device=device)[None, :] >= lengths[:, None], self.pad)
        alignments = None
        if return_alignments:
            alignments = torch.empty(batch, beam, frames, dtype=torch.long, device=device)
            index = order
            for t in reversed(range(frames)):
                alignments[:, :, t] = frame_labels[t].gather(1, index)
                index = backrefs[t].gather(1, index)
        return AEDCTCBeamSearchResult(
            labels=labels.view(batch, beam, max_length),
            lengths=lengths.view(batch, beam),
            scores=ranking.gather(1, order),
            alignments=alignments,
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.heads, self.head_dim)

    @staticmethod
    def _rotate(rotary, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """NeMo's rotary embedding of `x` (rows, heads, head_dim) at one position per row."""
        rotated_dim = rotary.d_k_rot
        cos = rotary.cos.index_select(0, positions).to(x.dtype)[:, None, :]
        sin = rotary.sin.index_select(0, positions).to(x.dtype)[:, None, :]
        head = x[..., :rotated_dim]
        half = rotated_dim // 2
        rotated = head * cos + torch.cat((-head[..., half:], head[..., :half]), dim=-1) * sin
        if rotated_dim == x.shape[-1]:
            return rotated
        return torch.cat((rotated, x[..., rotated_dim:]), dim=-1)

    def _encoder_projections(self, encoder_hidden_states: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Cross-attention keys and values (B, T, heads, head_dim) of every layer, keys pre-scaled."""
        projections = []
        for layer in self.layers:
            cross = layer.second_sub_layer
            keys = self._split_heads(cross.key_net(encoder_hidden_states)) / cross.attn_scale
            values = self._split_heads(cross.value_net(encoder_hidden_states))
            projections.append((keys, values))
        return projections

    def _decoder_step(
        self,
        labels: torch.Tensor,
        positions: torch.Tensor,
        caches: List[Tuple[torch.Tensor, torch.Tensor]],
        encoder: List[Tuple[torch.Tensor, torch.Tensor]],
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Log-probabilities (rows, vocabulary) of the next label after feeding `labels` at `positions`.

        The new keys and values of every layer are written into `caches` at the rows' positions.
        A row attends to the encoder projections of the sequence `row // beam_size`.
        """
        rows = labels.shape[0]
        batch = encoder_mask.shape[0]
        row_index = torch.arange(rows, device=labels.device)
        x = self.embedding(labels[:, None], start_pos=positions)[:, 0]
        capacity = caches[0][0].shape[1]
        attend = torch.arange(capacity, device=labels.device)[None, :] <= positions[:, None]
        for layer, (keys, values), (encoder_keys, encoder_values) in zip(self.layers, caches, encoder):
            attention = layer.first_sub_layer
            normed = layer.layer_norm_1(x)
            # NeMo scales queries and keys by d_k**-1/4 each; the keys are cached scaled.
            query = self._split_heads(attention.query_net(normed)) / attention.attn_scale
            key = self._split_heads(attention.key_net(normed)) / attention.attn_scale
            value = self._split_heads(attention.value_net(normed))
            # Only attention blocks with rotary positions carry the table.
            rotary = getattr(attention, "rotary_pos_enc", None)
            if rotary is not None:
                query, key = self._rotate(rotary, query, positions), self._rotate(rotary, key, positions)
            keys[row_index, positions] = key
            values[row_index, positions] = value
            scores = torch.einsum("rhd,rjhd->rhj", query, keys).masked_fill(~attend[:, None, :], NEG_INF)
            context = torch.einsum("rhj,rjhd->rhd", torch.softmax(scores, dim=-1), values).reshape(rows, -1)
            x = attention.out_projection(context) + x

            cross = layer.second_sub_layer
            query = self._split_heads(cross.query_net(layer.layer_norm_2(x))) / cross.attn_scale
            query = query.view(batch, rows // batch, self.heads, self.head_dim)
            scores = torch.einsum("bkhd,bjhd->bkhj", query, encoder_keys)
            scores = scores.masked_fill(~encoder_mask[:, None, None, :], NEG_INF)
            context = torch.einsum("bkhj,bjhd->bkhd", torch.softmax(scores, dim=-1), encoder_values)
            x = cross.out_projection(context.reshape(rows, -1)) + x

            x = layer.third_sub_layer(layer.layer_norm_3(x)) + x
        if self.final_norm is not None:
            x = self.final_norm(x)
        return self.classifier(hidden_states=x[:, None])[:, 0]

    @staticmethod
    def _reorder(source: torch.Tensor, destination: torch.Tensor, source_rows: torch.Tensor, used: int) -> None:
        """Copy the rows `source_rows` of the `used` leading cache slots into the other cache set.

        Slots at or beyond `used` were never written in either set, so they stay zero;
        the masked attention multiplies them by zero weights,
        which must not meet stale or uninitialized values.
        """
        destination[:, :used].copy_(source[:, :used].index_select(0, source_rows))

    def _recombine(
        self,
        score: torch.Tensor,
        new_label: torch.Tensor,
        history: torch.Tensor,
        history_length: torch.Tensor,
        batch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keep one path per label sequence; the decoder is not re-run for the paths that were merged away."""
        beam = self.beam_size
        labels = history.view(batch, beam, -1)
        lengths = history_length.view(batch, beam)
        same = (labels[:, :, None, :] == labels[:, None, :, :]).all(dim=-1) & (
            lengths[:, :, None] == lengths[:, None, :]
        )
        scores = score.view(batch, beam)
        others = torch.where(same, scores[:, None, :], NEG_INF)  # (B, path, other path)
        recombined = torch.logsumexp(others, dim=-1) if self.recombination == "sum" else scores
        keep = others.argmax(dim=-1) == torch.arange(beam, device=score.device)[None, :]
        recombined = torch.where(keep, recombined, NEG_INF)
        # Label sequences change only with a new label,
        # so a sequence's paths are recombined in the frames where one of them received a label.
        new_label = new_label.view(batch, beam)
        gate = new_label.any(dim=1, keepdim=True)
        scores = torch.where(gate, recombined, scores)
        new_label = torch.where(gate, new_label & keep, new_label)
        return scores.view(-1), new_label.view(-1)


def soft_collapse_repeated(
    log_probs: torch.Tensor, lengths: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge runs of frames that share a confident argmax label into one frame.

    A frame is merged into its predecessor
    when both have the same argmax label and both argmax probabilities are at least `threshold`;
    the first frame of a run is kept.
    The merged frame takes the per-class maximum over its frames and is renormalized with log-softmax,
    so it stays a distribution.

    Args:
        log_probs: log-probabilities of shape (B, T, C)
        lengths: valid frames per sequence, shape (B,)
        threshold: probability threshold of the argmax labels

    Returns:
        The merged log-probabilities of shape (B, T', C), where T' is the longest merged sequence,
        zero beyond each sequence's merged length, and the merged lengths of shape (B,).
    """
    frames = log_probs.shape[1]
    valid = torch.arange(frames, device=log_probs.device)[None, :] < lengths[:, None]
    argmax = log_probs.argmax(dim=-1)
    confident = log_probs.gather(-1, argmax[..., None])[..., 0].exp() >= threshold
    merged_into_previous = torch.zeros_like(confident)
    merged_into_previous[:, 1:] = (argmax[:, 1:] == argmax[:, :-1]) & confident[:, 1:] & confident[:, :-1]
    keep = ~merged_into_previous & valid
    merged_lengths = keep.sum(dim=1)
    # Every frame maps to the merged frame of its run; padded frames map to the last one with a -inf source.
    index = (keep.long().cumsum(dim=1) - 1).clamp(min=0)
    source = log_probs.masked_fill(~valid[..., None], NEG_INF)
    merged = torch.full_like(log_probs, NEG_INF)
    merged.scatter_reduce_(1, index[..., None].expand_as(log_probs), source, reduce="amax", include_self=False)
    merged_frames = int(merged_lengths.max()) if merged_lengths.numel() else 0
    merged = merged[:, :merged_frames]
    merged_valid = torch.arange(merged_frames, device=log_probs.device)[None, :] < merged_lengths[:, None]
    merged = torch.where(merged_valid[..., None], torch.log_softmax(merged, dim=-1), torch.zeros_like(merged))
    return merged, merged_lengths
