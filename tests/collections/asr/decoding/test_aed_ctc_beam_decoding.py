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

"""The joint AED and CTC time-synchronous beam search against a plain per-hypothesis reference."""

import math
from unittest.mock import Mock

import pytest
import torch

from nemo.collections.asr.modules.transformer.transformer import TransformerDecoderNM
from nemo.collections.asr.parts.submodules.aed_ctc_beam_decoding import (
    AEDCTCTimeSyncBeamSearch,
    TransformerAEDCTCBeamInfer,
    soft_collapse_repeated,
)
from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier
from nemo.collections.asr.parts.utils.rnnt_utils import NBestHypotheses

HIDDEN = 16


def _modules(vocab: int, seed: int = 0):
    """A small pre-LayerNorm decoder with its log-softmax output layer and random weights."""
    torch.manual_seed(seed)
    decoder = TransformerDecoderNM(
        vocab_size=vocab,
        hidden_size=HIDDEN,
        num_layers=2,
        inner_size=24,
        num_attention_heads=2,
        max_sequence_length=64,
        num_token_types=0,
        pre_ln=True,
    )
    classifier = TokenClassifier(hidden_size=HIDDEN, num_classes=vocab, log_softmax=True, use_transformer_init=False)
    for parameter in list(decoder.parameters()) + list(classifier.parameters()):
        parameter.data.normal_(0, 0.4)
    return decoder.eval(), classifier.eval()


def _tokenizer(vocab: int):
    """BOS and EOS are the last two ids, pad aliases EOS."""
    tokenizer = Mock()
    tokenizer.bos, tokenizer.eos, tokenizer.pad, tokenizer.vocab_size = vocab - 2, vocab - 1, vocab - 1, vocab
    tokenizer.ids_to_text = lambda ids: " ".join(str(i) for i in ids)
    return tokenizer


def _full_pass_log_probs(decoder, classifier, ids, encoder_states, encoder_mask):
    """The decoder's log-probabilities of the next label after the prefix `ids`, from a full pass."""
    ids = torch.tensor([ids], dtype=torch.long)
    hidden, _ = decoder.decoder(
        decoder.embedding(ids), torch.ones_like(ids).float(), encoder_states[None], encoder_mask[None].float()
    )
    return classifier(hidden_states=hidden)[0, -1]


def _reference_search(
    decoder,
    classifier,
    encoder_states,
    encoder_mask,
    ctc_log_probs,
    length,
    prompt,
    *,
    vocab,
    beam,
    aed_weight,
    ctc_weight,
    recombination,
):
    """The search of one sequence over explicit hypotheses, recomputing the decoder on the full prefix."""
    blank, eos = vocab, vocab - 1
    classes = vocab + 1

    def label_log_probs(labels):
        return _full_pass_log_probs(decoder, classifier, prompt + labels, encoder_states, encoder_mask) * aed_weight

    hyps = [dict(score=0.0, labels=[], last=blank, log_probs=label_log_probs([]), new=False)]
    for t in range(length):
        scores = []
        for hyp in hyps:
            for c in range(classes):
                score = hyp["score"] + ctc_weight * float(ctc_log_probs[t, c])
                if c != blank and (hyp["last"] == blank or c != hyp["last"]):
                    score += float(hyp["log_probs"][c])
                scores.append(score)
        top, index = torch.topk(torch.tensor(scores, dtype=torch.float32), beam)
        new_hyps = []
        for score, flat in zip(top.tolist(), index.tolist()):
            hyp, c = hyps[flat // classes], flat % classes
            new = c != blank and c != hyp["last"]
            labels = hyp["labels"] + [c] if new else hyp["labels"]
            new_hyps.append(dict(score=score, labels=labels, last=c, log_probs=hyp["log_probs"], new=new))
        if recombination and any(hyp["new"] for hyp in new_hyps):
            original = [hyp["score"] for hyp in new_hyps]
            for i, hyp in enumerate(new_hyps):
                same = [k for k, other in enumerate(new_hyps) if other["labels"] == hyp["labels"]]
                best = max(same, key=lambda k: (original[k], -k))  # the first of equal scores, as argmax
                if best != i:
                    hyp["score"], hyp["new"] = -math.inf, False
                elif recombination == "sum":
                    hyp["score"] = float(torch.logsumexp(torch.tensor([original[k] for k in same]), dim=0))
        for hyp in new_hyps:
            if hyp["new"]:
                hyp["log_probs"] = label_log_probs(hyp["labels"])
        hyps = new_hyps
    for hyp in hyps:
        hyp["score"] += float(hyp["log_probs"][eos])
    hyps.sort(key=lambda hyp: -hyp["score"])
    return [(hyp["labels"], hyp["score"]) for hyp in hyps]


def _naive_soft_collapse(log_probs, length, threshold):
    """Soft collapse of one sequence in plain Python."""
    probs = log_probs.exp()
    frames = []
    for t in range(length):
        peak = int(probs[t].argmax())
        if (
            frames
            and int(probs[t - 1].argmax()) == peak
            and float(probs[t, peak]) >= threshold
            and float(probs[t - 1, peak]) >= threshold
        ):
            frames[-1] = torch.maximum(frames[-1], log_probs[t])
        else:
            frames.append(log_probs[t].clone())
    return torch.stack([torch.log_softmax(frame, dim=-1) for frame in frames])


def _check_against_reference(decoder, classifier, vocab, *, batch, beam, prompt_length, recombination):
    """Random encoder states and peaky CTC frames of a batch with different lengths against the reference."""
    bos = vocab - 2
    frames = 7
    encoder_states = torch.randn(batch, 6, HIDDEN)
    encoder_lengths = torch.tensor([6, 5, 3][:batch])
    encoder_mask = torch.arange(6)[None, :] < encoder_lengths[:, None]
    # peaky CTC frames with repeated labels, so that paths merge and repeat
    logits = torch.randn(batch, frames, vocab + 1) * 2
    logits[:, :, vocab] += 1.5
    logits[0, 1:3, 4] += 4
    if batch > 1:
        logits[1, 2:5, 2] += 4
    ctc_log_probs = torch.log_softmax(logits, dim=-1)
    ctc_lengths = torch.tensor([7, 5, 2][:batch])
    prompt = torch.tensor([[bos, 5][:prompt_length]] * batch)
    search = AEDCTCTimeSyncBeamSearch(
        decoder,
        classifier,
        beam_size=beam,
        bos=bos,
        eos=vocab - 1,
        pad=vocab - 1,
        aed_weight=0.7,
        ctc_weight=1.0,
        recombination=recombination,
    )
    result = search(encoder_states, encoder_mask, ctc_log_probs, ctc_lengths, prompt, return_alignments=True)
    assert result.labels.shape[:2] == (batch, beam) and result.alignments.shape == (batch, beam, frames)
    with torch.no_grad():
        for b in range(batch):
            expected = _reference_search(
                decoder,
                classifier,
                encoder_states[b],
                encoder_mask[b],
                ctc_log_probs[b],
                int(ctc_lengths[b]),
                prompt[b].tolist(),
                vocab=vocab,
                beam=beam,
                aed_weight=0.7,
                ctc_weight=1.0,
                recombination=recombination,
            )
            for k, (labels, score) in enumerate(expected):
                if math.isinf(score):
                    # A recombined-away path keeps its score -inf; its labels are arbitrary.
                    assert math.isinf(float(result.scores[b, k]))
                    continue
                assert result.labels[b, k, : result.lengths[b, k]].tolist() == labels
                assert torch.all(result.labels[b, k, result.lengths[b, k] :] == vocab - 1)
                assert abs(float(result.scores[b, k]) - score) < 1e-3
                # the alignment collapses to the label sequence
                collapsed, previous = [], vocab
                for frame_label in result.alignments[b, k].tolist()[: int(ctc_lengths[b])]:
                    if frame_label != vocab and frame_label != previous:
                        collapsed.append(frame_label)
                    previous = frame_label
                assert collapsed == labels
                assert all(
                    frame_label == vocab for frame_label in result.alignments[b, k].tolist()[int(ctc_lengths[b]) :]
                )


class TestSoftCollapseRepeated:
    @pytest.mark.unit
    def test_matches_naive_collapse(self):
        torch.manual_seed(1)
        logits = torch.randn(3, 9, 5) * 3
        logits[0, 2:6, 1] += 20  # a confident run of label 1 over four frames
        logits[1, 0:2, 3] += 20  # a confident run starting at the first frame
        log_probs = torch.log_softmax(logits, dim=-1)
        lengths = torch.tensor([9, 7, 4])
        merged, merged_lengths = soft_collapse_repeated(log_probs, lengths, threshold=0.8)
        for b in range(3):
            expected = _naive_soft_collapse(log_probs[b], int(lengths[b]), 0.8)
            assert int(merged_lengths[b]) == expected.shape[0]
            torch.testing.assert_close(merged[b, : expected.shape[0]], expected)
            assert torch.all(merged[b, expected.shape[0] :] == 0)
        # the forced run of four frames became one frame; the output covers the longest merged sequence
        assert int(merged_lengths[0]) <= 6 and merged.shape[1] == int(merged_lengths.max())


class TestTimeSyncSearch:
    @pytest.mark.unit
    def test_cached_step_matches_full_pass(self):
        vocab = 11
        decoder, classifier = _modules(vocab)
        search = AEDCTCTimeSyncBeamSearch(decoder, classifier, beam_size=2, bos=9, eos=10, pad=10)
        torch.manual_seed(2)
        encoder_states = torch.randn(2, 6, HIDDEN)
        encoder_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]).bool()
        prefixes = [[9, 3, 4, 1], [9, 7], [9, 2, 2], [9]]  # rows: sequence 0 beams 0 and 1, sequence 1
        rows = len(prefixes)
        caches = [
            (torch.zeros(rows, 5, 2, HIDDEN // 2), torch.zeros(rows, 5, 2, HIDDEN // 2))
            for _ in decoder.decoder.layers
        ]
        encoder = search._encoder_projections(encoder_states)
        positions = torch.zeros(rows, dtype=torch.long)
        latest = [None] * rows
        with torch.no_grad():
            for step in range(max(len(prefix) for prefix in prefixes)):
                active = torch.tensor([step < len(prefix) for prefix in prefixes])
                labels = torch.tensor([prefix[min(step, len(prefix) - 1)] for prefix in prefixes])
                log_probs = search._decoder_step(labels, positions, caches, encoder, encoder_mask)
                for row in range(rows):
                    if active[row]:
                        latest[row] = log_probs[row]
                positions = positions + active.long()
            for row, prefix in enumerate(prefixes):
                expected = _full_pass_log_probs(
                    decoder, classifier, prefix, encoder_states[row // 2], encoder_mask[row // 2]
                )
                torch.testing.assert_close(latest[row], expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("recombination", ["max", "sum", None])
    @pytest.mark.parametrize("prompt_length", [1, 2])
    def test_matches_reference_search(self, recombination, prompt_length):
        vocab = 11
        decoder, classifier = _modules(vocab)
        torch.manual_seed(3)
        _check_against_reference(
            decoder, classifier, vocab, batch=3, beam=4, prompt_length=prompt_length, recombination=recombination
        )

    @pytest.mark.unit
    def test_matches_reference_search_at_production_beam(self):
        # The production configuration: beam 64 over a vocabulary with the two boundary labels and the blank,
        # recombination by maximum, a one-token prompt.
        vocab = 70
        decoder, classifier = _modules(vocab, seed=5)
        torch.manual_seed(5)
        _check_against_reference(decoder, classifier, vocab, batch=2, beam=64, prompt_length=1, recombination="max")

    @pytest.mark.unit
    def test_length_normalization_ranks_by_normalized_score(self):
        vocab = 11
        decoder, classifier = _modules(vocab)
        torch.manual_seed(4)
        encoder_states = torch.randn(1, 5, HIDDEN)
        encoder_mask = torch.ones(1, 5, dtype=torch.bool)
        ctc_log_probs = torch.log_softmax(torch.randn(1, 6, vocab + 1), dim=-1)
        lengths = torch.tensor([6])
        plain = AEDCTCTimeSyncBeamSearch(decoder, classifier, beam_size=4, bos=9, eos=10, pad=10)
        normalized = AEDCTCTimeSyncBeamSearch(
            decoder, classifier, beam_size=4, bos=9, eos=10, pad=10, length_normalization=1.0
        )
        plain_result = plain(encoder_states, encoder_mask, ctc_log_probs, lengths)
        normalized_result = normalized(encoder_states, encoder_mask, ctc_log_probs, lengths)
        plain_scores = {
            tuple(plain_result.labels[0, k, : plain_result.lengths[0, k]].tolist()): float(plain_result.scores[0, k])
            for k in range(4)
            if torch.isfinite(plain_result.scores[0, k])
        }
        for k in range(4):
            labels = tuple(normalized_result.labels[0, k, : normalized_result.lengths[0, k]].tolist())
            if labels in plain_scores:
                assert abs(float(normalized_result.scores[0, k]) - plain_scores[labels] / (len(labels) + 1)) < 1e-4
        assert torch.all(normalized_result.scores[0, :-1] >= normalized_result.scores[0, 1:])

    @pytest.mark.unit
    def test_rejects_wrong_blank_position(self):
        vocab = 11
        decoder, classifier = _modules(vocab)
        search = AEDCTCTimeSyncBeamSearch(decoder, classifier, beam_size=2, bos=9, eos=10, pad=10)
        with pytest.raises(ValueError, match="CTC classes"):
            search(torch.randn(1, 3, HIDDEN), torch.ones(1, 3), torch.zeros(1, 3, vocab), torch.tensor([3]))


class TestTransformerAEDCTCBeamInfer:
    @pytest.mark.unit
    def test_packs_hypotheses_without_special_ids(self):
        vocab = 11
        decoder, classifier = _modules(vocab)
        torch.manual_seed(5)
        encoder_states = torch.randn(2, 5, HIDDEN)
        encoder_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0]])
        logits = torch.randn(2, 5, vocab + 1)
        logits[0, 1, vocab - 1] += 8  # the CTC head emits the end label as a frame label
        ctc_log_probs = torch.log_softmax(logits, dim=-1)
        prompt = torch.full((2, 1), vocab - 2)
        best = TransformerAEDCTCBeamInfer(
            decoder, classifier, _tokenizer(vocab), beam_size=3, preserve_alignments=True
        )
        (hypotheses,) = best(
            encoder_hidden_states=encoder_states,
            encoder_input_mask=encoder_mask,
            decoder_input_ids=prompt,
            ctc_log_probs=ctc_log_probs,
        )
        assert len(hypotheses) == 2
        for hypothesis in hypotheses:
            assert torch.is_tensor(hypothesis.y_sequence) and hypothesis.y_sequence.dtype == torch.long
            assert not any(int(i) in (vocab - 2, vocab - 1) for i in hypothesis.y_sequence)
            assert len(hypothesis.alignments) == 5 and math.isfinite(hypothesis.score)
        n_best = TransformerAEDCTCBeamInfer(
            decoder, classifier, _tokenizer(vocab), beam_size=3, return_best_hypothesis=False
        )
        (n_best_hypotheses,) = n_best(
            encoder_hidden_states=encoder_states,
            encoder_input_mask=encoder_mask,
            decoder_input_ids=prompt,
            ctc_log_probs=ctc_log_probs,
        )
        for packed, hypothesis in zip(n_best_hypotheses, hypotheses):
            assert isinstance(packed, NBestHypotheses) and 1 <= len(packed.n_best_hypotheses) <= 3
            scores = [h.score for h in packed.n_best_hypotheses]
            assert scores == sorted(scores, reverse=True) and all(math.isfinite(s) for s in scores)
            assert torch.equal(packed.n_best_hypotheses[0].y_sequence, hypothesis.y_sequence)

    @pytest.mark.unit
    def test_needs_ctc_log_probs(self):
        vocab = 11
        decoder, classifier = _modules(vocab)
        infer = TransformerAEDCTCBeamInfer(decoder, classifier, _tokenizer(vocab), beam_size=2)
        with pytest.raises(ValueError, match="CTC log-probabilities"):
            infer(
                encoder_hidden_states=torch.randn(1, 4, HIDDEN),
                encoder_input_mask=torch.ones(1, 4),
                decoder_input_ids=torch.full((1, 1), vocab - 2),
            )

    @pytest.mark.unit
    def test_soft_collapse_shortens_the_search(self):
        vocab = 11
        decoder, classifier = _modules(vocab)
        torch.manual_seed(6)
        encoder_states = torch.randn(1, 8, HIDDEN)
        encoder_mask = torch.ones(1, 8)
        logits = torch.randn(1, 8, vocab + 1)
        logits[0, 2:6, 3] += 10  # a confident run of label 3
        ctc_log_probs = torch.log_softmax(logits, dim=-1)
        infer = TransformerAEDCTCBeamInfer(
            decoder,
            classifier,
            _tokenizer(vocab),
            beam_size=2,
            ctc_soft_collapse_threshold=0.8,
            preserve_alignments=True,
        )
        (hypotheses,) = infer(
            encoder_hidden_states=encoder_states,
            encoder_input_mask=encoder_mask,
            decoder_input_ids=torch.full((1, 1), vocab - 2),
            ctc_log_probs=ctc_log_probs,
        )
        assert len(hypotheses[0].alignments) == 5  # eight frames, the run of four merged into one
