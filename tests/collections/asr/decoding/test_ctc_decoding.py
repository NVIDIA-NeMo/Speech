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

import copy
import os
from functools import cached_property, lru_cache
from pathlib import Path

import jiwer
import pytest
import torch
from omegaconf import DictConfig, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.mixins import mixins
from nemo.collections.asr.parts.submodules.ctc_decoding import (
    CTCBPEDecoding,
    CTCBPEDecodingConfig,
    CTCDecoding,
    CTCDecodingConfig,
)
from nemo.collections.asr.parts.submodules.ngram_lm.ngram_lm_batched import NGramGPULanguageModel
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.core.utils.cuda_python_utils import skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported
from tests.collections.asr.decoding.test_timestamps import BaseTimestampsTest


@pytest.fixture(scope="module")
def audio_file(test_data_dir):
    return os.path.join(test_data_dir, "asr/test/an4/wav/cen3-mjwl-b.wav")


CTC_MODEL = "nvidia/stt_en_conformer_ctc_small"


@pytest.fixture(scope="module")
def kenlm_model_path(tmp_path_factory, test_data_dir):
    lm_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
    assert os.path.exists(lm_path), f"LM file not found: {lm_path}"
    lm_nemo_path = tmp_path_factory.mktemp("lm") / f"{lm_path.name}.nemo"
    NGramGPULanguageModel.from_file(lm_path, vocab_size=1024).save_to(f"{lm_nemo_path}")
    return f"{lm_nemo_path}"


@pytest.fixture(scope="module")
def ctc_model():
    model = ASRModel.from_pretrained(model_name=CTC_MODEL, map_location="cpu")
    model.eval()
    return model


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


class TestCTCGreedyDecodingWithNGPU_LM:
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    def test_ctc_decoding_gpulm(
        self,
        audio_file,
        kenlm_model_path,
        ctc_model,
    ):
        device = torch.device("cuda")
        model = ctc_model.to(device)

        gt_hyp = model.transcribe([audio_file], num_workers=None)

        decoding_config = copy.deepcopy(model.cfg.decoding)
        with open_dict(model.decoding.cfg) as cfg:
            cfg.greedy["ngram_lm_model"] = kenlm_model_path
            cfg.greedy["ngram_lm_alpha"] = 0.0
            model.change_decoding_strategy(cfg)
        lm_hyp = model.transcribe([audio_file], num_workers=None)

        assert gt_hyp[0].text == lm_hyp[0].text
        assert abs(gt_hyp[0].score - lm_hyp[0].score) <= 1e-3

        with open_dict(model.decoding.cfg) as cfg:
            cfg.greedy["ngram_lm_model"] = kenlm_model_path
            cfg.greedy["ngram_lm_alpha"] = 10.0
        model.change_decoding_strategy(cfg)
        lm_hyp = model.transcribe([audio_file], num_workers=None)
        assert gt_hyp[0].text != lm_hyp[0].text
        assert abs(gt_hyp[0].score - lm_hyp[0].score) > 1e-3

        model.change_decoding_strategy(decoding_config)


class TestCTCGreedyDecodingCudaGrpahs:
    """
    Tests CudaGraphs implementations from CTC models greedy decoding
    """

    @pytest.mark.with_downloads
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder can run only on CUDA")
    @pytest.mark.parametrize("force_mode", ["no_graphs", "no_while_loops", "full_graph"])
    def test_stated_stateless(self, audio_file, kenlm_model_path, ctc_model, force_mode: str):
        """
        Compares pure Pytorch and with three modes of statefull implementations for double floating point precision.
            1. Pure pytorch, but statefull implementation: no_graphs
            2. With CudaGrpahs: no_while_loops and full_graph.
        """
        if force_mode == "full_graph":
            skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

        device = torch.device("cuda")
        model = ctc_model.to(device)
        decoding_config = copy.deepcopy(model.cfg.decoding)

        with open_dict(model.decoding.cfg) as cfg:
            cfg.greedy["ngram_lm_model"] = kenlm_model_path
            cfg.greedy["ngram_lm_alpha"] = 0.2
            cfg.greedy["allow_cuda_graphs"] = False

            model.change_decoding_strategy(cfg)

        actual_hypotheses = model.transcribe([audio_file], num_workers=None)
        actual_transcripts = [hyp.text for hyp in actual_hypotheses]
        actual_scores = [hyp.score for hyp in actual_hypotheses]
        actual_timestamps = [hyp.timestamp for hyp in actual_hypotheses]

        # transcribe with use implementation with cuda graphs
        model.decoding.cfg["greedy"]["allow_cuda_graphs"] = True
        model.change_decoding_strategy(model.decoding.cfg)
        model.decoding.decoding.force_cuda_graphs_mode(mode=force_mode)

        cudagraph_hypotheses = model.transcribe([audio_file], num_workers=None)
        cudagraph_transcripts = [hyp.text for hyp in cudagraph_hypotheses]
        cudagraph_scores = [hyp.score for hyp in cudagraph_hypotheses]
        cudagraph_timestamps = [hyp.timestamp for hyp in cudagraph_hypotheses]

        for batch_idx in range(len(actual_transcripts)):
            assert len(actual_transcripts[batch_idx]) == len(cudagraph_transcripts[batch_idx])
            assert cudagraph_scores[batch_idx] == pytest.approx(
                actual_scores[batch_idx], abs=1e-2
            ), f"Scores mismatch for batch_idx {batch_idx}"
            assert (
                cudagraph_timestamps[batch_idx] == actual_timestamps[batch_idx]
            ), f"Timestamps mismatch for batch_idx {batch_idx}"

            wer = jiwer.wer(actual_transcripts[batch_idx], cudagraph_transcripts[batch_idx])

            assert wer <= 1e-3, "Cuda graph greedy decoder should match original decoder implementation."

            for actual, fast in zip(actual_transcripts[batch_idx], cudagraph_transcripts[batch_idx]):
                if actual != fast:
                    print("Erroneous samples in batch:", batch_idx)
                    print("Original transcript:", actual)
                    print("New transcript:", fast)

        model.change_decoding_strategy(decoding_config)
