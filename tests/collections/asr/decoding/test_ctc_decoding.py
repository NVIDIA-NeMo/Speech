# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
from functools import cached_property, lru_cache

import pytest
import torch
from omegaconf import DictConfig, OmegaConf


from nemo.collections.asr.parts.mixins import mixins
from nemo.collections.asr.parts.submodules.ctc_decoding import (
    CTCBPEDecoding,
    CTCBPEDecodingConfig,
    CTCDecoding,
    CTCDecodingConfig,
)
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from tests.collections.asr.decoding.test_timestamps import BaseTimestampsTest


def char_vocabulary():
    return [' ', 'a', 'b', 'c', 'd', 'e', 'f', '.']


@pytest.fixture()
@lru_cache(maxsize=8)
def tmp_tokenizer(test_data_dir):
    cfg = DictConfig({'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'})

    class _TmpASRBPE(mixins.ASRBPEMixin):
        def register_artifact(self, _, vocab_path):
            return vocab_path

    asrbpe = _TmpASRBPE()
    asrbpe._setup_tokenizer(cfg)
    return asrbpe.tokenizer


class TestCTCDecoding:
    @pytest.mark.unit
    def test_constructor(self):
        cfg = CTCDecodingConfig()
        vocab = char_vocabulary()
        decoding = CTCDecoding(decoding_cfg=cfg, vocabulary=vocab)
        assert decoding is not None

    @pytest.mark.unit
    def test_constructor_subword(self, tmp_tokenizer):
        cfg = CTCBPEDecodingConfig()
        decoding = CTCBPEDecoding(decoding_cfg=cfg, tokenizer=tmp_tokenizer)
        assert decoding is not None

    @pytest.mark.unit
    def test_char_decoding_greedy_forward(
        self,
    ):
        cfg = CTCDecodingConfig(strategy='greedy')
        vocab = char_vocabulary()
        decoding = CTCDecoding(decoding_cfg=cfg, vocabulary=vocab)

        B, T = 4, 20
        V = len(char_vocabulary()) + 1
        input_signal = torch.randn(size=(B, T, V))
        length = torch.randint(low=1, high=T, size=[B])

        with torch.no_grad():
            hypotheses = decoding.ctc_decoder_predictions_tensor(
                input_signal, length, fold_consecutive=True, return_hypotheses=False
            )
            texts = [hyp.text for hyp in hypotheses]

            for text in texts:
                assert isinstance(text, str)

    @pytest.mark.unit
    @pytest.mark.parametrize('alignments', [False, True])
    @pytest.mark.parametrize('timestamps', [False, True])
    def test_char_decoding_greedy_forward_hypotheses(self, alignments, timestamps):
        cfg = CTCDecodingConfig(strategy='greedy', preserve_alignments=alignments, compute_timestamps=timestamps)
        vocab = char_vocabulary()
        decoding = CTCDecoding(decoding_cfg=cfg, vocabulary=vocab)

        B, T = 4, 20
        V = len(char_vocabulary()) + 1
        input_signal = torch.randn(size=(B, T, V))
        length = torch.randint(low=1, high=T, size=[B])

        with torch.no_grad():
            hyps = decoding.ctc_decoder_predictions_tensor(
                input_signal, length, fold_consecutive=True, return_hypotheses=True
            )

            for idx, hyp in enumerate(hyps):
                assert isinstance(hyp, Hypothesis)
                assert torch.is_tensor(hyp.y_sequence)
                assert isinstance(hyp.text, str)

                # alignments check
                if alignments:
                    assert hyp.alignments is not None
                    assert isinstance(hyp.alignments, tuple)
                    assert len(hyp.alignments[0]) == length[idx]
                    assert len(hyp.alignments[1]) == length[idx]

                # timestamps check
                if timestamps:
                    BaseTimestampsTest.check_char_timestamps(hyp, decoding)

    @pytest.mark.unit
    def test_subword_decoding_greedy_forward(self, tmp_tokenizer):
        cfg = CTCBPEDecodingConfig(strategy='greedy')
        decoding = CTCBPEDecoding(decoding_cfg=cfg, tokenizer=tmp_tokenizer)

        B, T = 4, 20
        V = decoding.tokenizer.tokenizer.vocab_size + 1
        input_signal = torch.randn(size=(B, T, V))
        length = torch.randint(low=1, high=T, size=[B])

        with torch.no_grad():
            hypotheses = decoding.ctc_decoder_predictions_tensor(
                input_signal, length, fold_consecutive=True, return_hypotheses=False
            )
            texts = [hyp.text for hyp in hypotheses]

            for text in texts:
                assert isinstance(text, str)

    @pytest.mark.unit
    @pytest.mark.parametrize('alignments', [False, True])
    @pytest.mark.parametrize('timestamps', [False, True])
    @pytest.mark.pleasefixme
    def test_subword_decoding_greedy_forward_hypotheses(self, tmp_tokenizer, alignments, timestamps):
        cfg = CTCBPEDecodingConfig(strategy='greedy', preserve_alignments=alignments, compute_timestamps=timestamps)
        decoding = CTCBPEDecoding(decoding_cfg=cfg, tokenizer=tmp_tokenizer)

        B, T = 4, 20
        V = decoding.tokenizer.tokenizer.vocab_size + 1
        input_signal = torch.randn(size=(B, T, V))
        length = torch.randint(low=1, high=T, size=[B])

        with torch.no_grad():
            hyps = decoding.ctc_decoder_predictions_tensor(
                input_signal, length, fold_consecutive=True, return_hypotheses=True
            )

            for idx, hyp in enumerate(hyps):
                assert isinstance(hyp, Hypothesis)
                assert torch.is_tensor(hyp.y_sequence)
                assert isinstance(hyp.text, str)

                # alignments check
                if alignments:
                    assert hyp.alignments is not None
                    assert isinstance(hyp.alignments, tuple)
                    assert len(hyp.alignments[0]) == length[idx]
                    assert len(hyp.alignments[1]) == length[idx]

                # timestamps check
                if timestamps:
                    BaseTimestampsTest.check_subword_timestamps(hyp, decoding)

    @pytest.mark.unit
    @pytest.mark.parametrize('alignments', [False, True])
    @pytest.mark.parametrize('timestamps', [False, True])
    @pytest.mark.parametrize('preserve_frame_confidence', [False, True])
    @pytest.mark.parametrize('length_is_none', [False, True])
    @pytest.mark.parametrize(
        "logprobs_device",
        [
            torch.device("cpu"),
            pytest.param(
                torch.device("cuda"),
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason='CUDA required for test.',
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "length_device",
        [
            torch.device("cpu"),
            pytest.param(
                torch.device("cuda"),
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason='CUDA required for test.',
                ),
            ),
        ],
    )
    def test_batched_decoding_logprobs(
        self,
        tmp_tokenizer,
        alignments,
        timestamps,
        preserve_frame_confidence,
        length_is_none,
        logprobs_device,
        length_device,
    ):
        cfg = CTCBPEDecodingConfig(
            strategy='greedy',
            preserve_alignments=alignments,
            compute_timestamps=timestamps,
            confidence_cfg=ConfidenceConfig(preserve_frame_confidence=preserve_frame_confidence),
        )
        unbatched_decoding = CTCBPEDecoding(decoding_cfg=cfg, tokenizer=tmp_tokenizer)

        cfg.strategy = 'greedy_batch'
        batched_decoding = CTCBPEDecoding(decoding_cfg=cfg, tokenizer=tmp_tokenizer)

        torch.manual_seed(1)
        B, T = 4, 20
        V = unbatched_decoding.tokenizer.tokenizer.vocab_size + 1
        input_signal = torch.randn(size=(B, T, V), device=logprobs_device)
        # Set the blank index to a very high probability to make sure
        # that we always handle at least a few blanks.
        input_signal[:, 0, unbatched_decoding.tokenizer.tokenizer.vocab_size] = 1000
        input_signal[:, 1, unbatched_decoding.tokenizer.tokenizer.vocab_size] = 1000
        if length_is_none:
            length = None
        else:
            length = torch.randint(low=1, high=T, size=[B], device=length_device)

        with torch.inference_mode():
            hyps = unbatched_decoding.ctc_decoder_predictions_tensor(
                input_signal, length, fold_consecutive=True, return_hypotheses=True
            )

            batched_hyps = batched_decoding.ctc_decoder_predictions_tensor(
                input_signal, length, fold_consecutive=True, return_hypotheses=True
            )

            assert len(hyps) == len(batched_hyps) == B
            for hyp, batched_hyp in zip(hyps, batched_hyps):
                assert torch.abs(hyp.score - batched_hyp.score) <= 1e-5
                assert torch.all(hyp.y_sequence == batched_hyp.y_sequence)
                if timestamps:
                    assert hyp.timestamp == batched_hyp.timestamp
                if alignments:
                    assert torch.all(hyp.alignments[0] == batched_hyp.alignments[0])
                    assert torch.all(hyp.alignments[1] == batched_hyp.alignments[1])

    @pytest.mark.unit
    @pytest.mark.parametrize('timestamps', [False, True])
    @pytest.mark.parametrize('length_is_none', [False, True])
    @pytest.mark.parametrize(
        "labels_device",
        [
            torch.device("cpu"),
            pytest.param(
                torch.device("cuda"),
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason='CUDA required for test.',
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "length_device",
        [
            torch.device("cpu"),
            pytest.param(
                torch.device("cuda"),
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason='CUDA required for test.',
                ),
            ),
        ],
    )
    def test_batched_decoding_labels(self, tmp_tokenizer, timestamps, length_is_none, labels_device, length_device):
        cfg = CTCBPEDecodingConfig(strategy='greedy', compute_timestamps=timestamps)
        unbatched_decoding = CTCBPEDecoding(decoding_cfg=cfg, tokenizer=tmp_tokenizer)
        cfg.strategy = 'greedy_batch'
        batched_decoding = CTCBPEDecoding(decoding_cfg=cfg, tokenizer=tmp_tokenizer)

        torch.manual_seed(1)
        B, T = 4, 20
        V = unbatched_decoding.tokenizer.tokenizer.vocab_size + 1
        input_labels = torch.randint(V, size=(B, T), device=labels_device)
        # Set some indices to blank to make sure that we always handle
        # at least a few blanks.
        input_labels[:, 0] = unbatched_decoding.tokenizer.tokenizer.vocab_size
        input_labels[:, 1] = unbatched_decoding.tokenizer.tokenizer.vocab_size
        if length_is_none:
            length = None
        else:
            length = torch.randint(low=1, high=T, size=[B], device=length_device)

        with torch.inference_mode():
            hyps = unbatched_decoding.ctc_decoder_predictions_tensor(
                input_labels, length, fold_consecutive=True, return_hypotheses=True
            )

            batched_hyps = batched_decoding.ctc_decoder_predictions_tensor(
                input_labels, length, fold_consecutive=True, return_hypotheses=True
            )

            assert len(hyps) == len(batched_hyps) == B
            for hyp, batched_hyp in zip(hyps, batched_hyps):
                assert abs(hyp.score - batched_hyp.score) <= 1e-5
                assert torch.all(hyp.y_sequence == batched_hyp.y_sequence)
                if timestamps:
                    assert hyp.timestamp == batched_hyp.timestamp


class TestCTCTimestamps(BaseTimestampsTest):
    """CTC-specific timestamp tests that inherit from BaseTimestampsTest"""

    @cached_property
    def decoding_char(self):
        cfg = CTCDecodingConfig()
        vocab = char_vocabulary()
        decoding = CTCDecoding(decoding_cfg=cfg, vocabulary=vocab)
        return decoding

    @cached_property
    def decoding_subword_wpe(self):
        cfg = CTCBPEDecodingConfig(compute_timestamps=True)
        decoding = CTCBPEDecoding(decoding_cfg=cfg, tokenizer=self.tmp_tokenizer)
        return decoding

    @cached_property
    def decoding_subword_bpe(self):
        cfg = CTCBPEDecodingConfig(compute_timestamps=True)
        decoding = CTCBPEDecoding(decoding_cfg=cfg, tokenizer=self.bpe_tokenizer)
        return decoding

    @pytest.mark.unit
    def test_word_offsets_subword_wpe(self, tmp_tokenizer):
        self.tmp_tokenizer = tmp_tokenizer
        super().test_word_offsets_subword_wpe()

    @pytest.mark.unit
    def test_word_offsets_subword_wpe_other_delimiter(self, tmp_tokenizer):
        self.tmp_tokenizer = tmp_tokenizer
        super().test_word_offsets_subword_wpe_other_delimiter()
