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
import io
import math
import operator
import random
import string
from copy import deepcopy
from typing import List
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.metrics.bleu import (
    BLEU,
    BLEU_TOKENIZER,
    _get_bleu_tokenizers_from_cuts,
    _move_dimension_to_the_front,
)
from nemo.collections.asr.metrics.multitask import ConstraintParser, MultiTaskMetric
from nemo.collections.asr.metrics.wer import WER, word_error_rate, word_error_rate_detail, word_error_rate_per_utt
from nemo.collections.asr.parts.submodules.ctc_decoding import (
    AbstractCTCDecoding,
    CTCBPEDecoding,
    CTCBPEDecodingConfig,
    CTCDecoding,
    CTCDecodingConfig,
)
from nemo.collections.asr.parts.submodules.multitask_decoding import AbstractMultiTaskDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding, RNNTBPEDecoding, RNNTDecoding
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.tokenizers import CharTokenizer
from nemo.utils.config_utils import assert_dataclass_signature_match


def build_char_tokenizer_with_vocabulary(vocabulary: List[str]) -> CharTokenizer:
    with patch('pathlib.Path.open', Mock(return_value=io.StringIO('\n'.join([repr(char) for char in vocabulary])))):
        char_tokenizer = CharTokenizer('a_path_which_will_not_be_used')
    # For some reason `WERBPE` takes vocabulary size of inner tokenizer. Mock inner tokenizer.
    setattr(char_tokenizer, "tokenizer", Mock(vocab_size=char_tokenizer.vocab_size))
    return char_tokenizer


class TestWordErrorRate:

    vocabulary = [' '] + list(string.ascii_lowercase) + ["'"]
    char_tokenizer = build_char_tokenizer_with_vocabulary(vocabulary)

    def __string_to_ctc_tensor(self, txt: str, use_tokenizer: bool, as_logprobs: bool = False) -> torch.Tensor:
        # This function emulates how CTC output could like for txt
        if use_tokenizer:
            blank_id = self.char_tokenizer.vocab_size
            string_in_id_form = self.char_tokenizer.text_to_ids(txt)
        else:
            blank_id = len(self.vocabulary)
            char_to_ind = dict([(self.vocabulary[i], i) for i in range(len(self.vocabulary))])
            string_in_id_form = [char_to_ind[c] for c in txt]
        ctc_list = []
        prev_id = -1
        for c in string_in_id_form:
            # when character is repeated we need to insert CTC blank symbol
            if c != prev_id:
                ctc_list.append(c)
            else:
                ctc_list.append(blank_id)
                ctc_list.append(c)
            prev_id = c
        tensor = torch.Tensor(ctc_list).unsqueeze(0)

        if not as_logprobs:
            return tensor
        else:
            tensor = tensor.to(torch.int64)
            new_tensor = torch.nn.functional.one_hot(tensor[0], num_classes=blank_id)
            new_tensor = new_tensor.unsqueeze(0)  # [1, V, T]
            return new_tensor

    def __reference_string_to_tensor(self, txt: str, use_tokenizer: bool) -> torch.Tensor:
        # Reference tensors aren't produced by CTC logic
        if use_tokenizer:
            string_in_id_form = self.char_tokenizer.text_to_ids(txt)
        else:
            char_to_ind = dict([(self.vocabulary[i], i) for i in range(len(self.vocabulary))])
            string_in_id_form = [char_to_ind[c] for c in txt]
        return torch.Tensor(string_in_id_form).unsqueeze(0)

    def get_wer(self, wer, prediction: str, reference: str, use_tokenizer: bool):
        predictions_tensor = self.__string_to_ctc_tensor(prediction, use_tokenizer)
        targets_tensor = self.__reference_string_to_tensor(reference, use_tokenizer)
        if wer.batch_dim_index > 0:
            targets_tensor.transpose_(0, 1)
            predictions_tensor.transpose_(0, 1)
        wer(
            predictions=predictions_tensor,
            predictions_lengths=None,
            targets=targets_tensor,
            targets_lengths=torch.tensor([len(reference)]),
        )
        res, _, _ = wer.compute()
        res = res.detach().cpu()
        # return res[0] / res[1]
        return res.item()

    @pytest.mark.unit
    def test_wer_function(self):
        assert word_error_rate(hypotheses=['cat'], references=['cot']) == 1.0
        assert word_error_rate(hypotheses=['GPU'], references=['G P U']) == 1.0
        assert word_error_rate(hypotheses=['G P U'], references=['GPU']) == 3.0
        assert word_error_rate(hypotheses=['ducati motorcycle'], references=['motorcycle']) == 1.0
        assert word_error_rate(hypotheses=['ducati motorcycle'], references=['ducuti motorcycle']) == 0.5
        assert word_error_rate(hypotheses=['a B c'], references=['a b c']) == 1.0 / 3.0

        assert word_error_rate_detail(hypotheses=['cat'], references=['cot'])[0] == 1.0
        assert word_error_rate_detail(hypotheses=['GPU'], references=['G P U'])[0] == 1.0
        assert word_error_rate_detail(hypotheses=['G P U'], references=['GPU'])[0] == 3.0
        assert word_error_rate_detail(hypotheses=['ducati motorcycle'], references=['motorcycle'])[0] == 1.0
        assert word_error_rate_detail(hypotheses=['ducati motorcycle'], references=['ducuti motorcycle'])[0] == 0.5
        assert word_error_rate_detail(hypotheses=['a B c'], references=['a b c'])[0] == 1.0 / 3.0

        assert word_error_rate_detail(hypotheses=['cat'], references=['']) == (
            float("inf"),
            0,
            float("inf"),
            float("inf"),
            float("inf"),
        )
        assert word_error_rate_detail(hypotheses=['cat', ''], references=['', 'gpu']) == (
            2.0,
            1,
            1.0,
            1.0,
            0.0,
        )
        assert word_error_rate_detail(hypotheses=['cat'], references=['cot']) == (1.0, 1, 0.0, 0.0, 1.0)
        assert word_error_rate_detail(hypotheses=['G P U'], references=['GPU']) == (3.0, 1, 2.0, 0.0, 1.0)
        assert word_error_rate_detail(hypotheses=[''], references=['ducuti motorcycle'], use_cer=True) == (
            1.0,
            17,
            0.0,
            1.0,
            0.0,
        )

        assert word_error_rate_per_utt(hypotheses=['kat'], references=['cat']) == ([1.0], 1.0)
        assert word_error_rate_per_utt(hypotheses=['cat', ''], references=['', 'gpu']) == ([float("inf"), 1.0], 2.0)
        assert word_error_rate_per_utt(
            hypotheses=['ducuti motorcycle', 'G P U'], references=['ducati motorcycle', 'GPU']
        ) == ([0.5, 3.0], 4 / 3)
        assert word_error_rate_per_utt(
            hypotheses=['ducuti motorcycle', 'G P U'], references=['ducati motorcycle', 'GPU'], use_cer=True
        ) == ([1 / 17, 2 / 3], 0.15)

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_dim_index", [0, 1])
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_wer_metric_simple(self, batch_dim_index, test_wer_bpe):
        assert self.get_wer_ctc('cat', 'cot', test_wer_bpe) == 1.0
        assert self.get_wer_ctc('gpu', 'g p u', test_wer_bpe) == 1.0
        assert self.get_wer_ctc('g p u', 'gpu', test_wer_bpe) == 3.0
        assert self.get_wer_ctc('ducati motorcycle', 'motorcycle', test_wer_bpe) == 1.0
        assert self.get_wer_ctc('ducati motorcycle', 'ducuti motorcycle', test_wer_bpe) == 0.5
        assert abs(self.get_wer_ctc('a f c', 'a b c', test_wer_bpe) - 1.0 / 3.0) < 1e-6

    @pytest.mark.unit
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_wer_metric_randomized(self, test_wer_bpe):
        """This test relies on correctness of word_error_rate function."""

        def __random_string(length):
            return ''.join(random.choice(''.join(self.vocabulary)) for _ in range(length))

        for test_id in range(256):
            n1 = random.randint(1, 512)
            n2 = random.randint(1, 512)
            s1 = __random_string(n1)
            s2 = __random_string(n2)
            # skip empty strings as reference
            if s2.strip():
                assert (
                    abs(
                        self.get_wer_ctc(prediction=s1, reference=s2, test_wer_bpe=test_wer_bpe)
                        - word_error_rate(hypotheses=[s1], references=[s2])
                    )
                    < 1e-6
                )

    @pytest.mark.unit
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_wer_metric_decode(self, test_wer_bpe):
        decoding_config = {'strategy': 'greedy'}
        if test_wer_bpe:
            decoding = CTCBPEDecoding(decoding_config, self.char_tokenizer)
            wer = WER(decoding, use_cer=False)
        else:
            decoding = CTCDecoding(decoding_config, self.vocabulary.copy())
            wer = WER(decoding, use_cer=False)

        tokens = self.__string_to_ctc_tensor('cat', use_tokenizer=test_wer_bpe)[0].int().numpy().tolist()
        assert tokens == [3, 1, 20]

        tokens_decoded = wer.decoding.decode_ids_to_tokens(tokens)
        assert tokens_decoded == ['c', 'a', 't']

        str_decoded = wer.decoding.decode_ids_to_str(tokens)
        assert str_decoded == 'cat'

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_dim_index", [0, 1])
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_wer_metric_return_hypothesis(self, batch_dim_index, test_wer_bpe):
        decoding_config = {'strategy': 'greedy', 'batch_dim_index': batch_dim_index}
        wer = WER(CTCDecoding(decoding_config, self.vocabulary), use_cer=False)

        tensor = self.__string_to_ctc_tensor('cat', test_wer_bpe, as_logprobs=True).int()
        if batch_dim_index > 0:
            tensor.transpose_(0, 1)

        # pass batchsize 1 tensor, get back list of length 1 Hypothesis
        wer.decoding.preserve_alignments = True
        hyp = wer.decoding.ctc_decoder_predictions_tensor(tensor, return_hypotheses=True)
        hyp = hyp[0]
        assert isinstance(hyp, Hypothesis)

        sample = tensor[0] if batch_dim_index == 0 else tensor[:, 0, :]
        assert (hyp.y_sequence - torch.tensor([3, 1, 20])).sum() == 0
        assert hyp.score == 3  # sum of number of tokens in one hot representation
        assert hyp.text == 'cat'
        assert (hyp.alignments[0] == sample).all()
        assert hyp.length == 0

        length = torch.tensor([tensor.shape[1 - batch_dim_index]], dtype=torch.long)

        # pass batchsize 1 tensor, get back list of length 1 Hypothesis [add length info]
        hyp = wer.decoding.ctc_decoder_predictions_tensor(tensor, decoder_lengths=length, return_hypotheses=True)
        hyp = hyp[0]
        assert isinstance(hyp, Hypothesis)
        assert hyp.length == 3

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_dim_index", [0, 1])
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_wer_metric_subword_return_hypothesis(self, batch_dim_index, test_wer_bpe):
        decoding_config = {'strategy': 'greedy', 'batch_dim_index': batch_dim_index}
        wer = WER(CTCBPEDecoding(decoding_config, self.char_tokenizer), use_cer=False)

        tensor = self.__string_to_ctc_tensor('cat', test_wer_bpe, as_logprobs=True).int()
        if batch_dim_index > 0:
            tensor.transpose_(0, 1)

        # pass batchsize 1 tensor, get back list of length 1 Hypothesis
        wer.decoding.preserve_alignments = True
        hyp = wer.decoding.ctc_decoder_predictions_tensor(tensor, return_hypotheses=True)
        hyp = hyp[0]
        assert isinstance(hyp, Hypothesis)

        sample = tensor[0] if batch_dim_index == 0 else tensor[:, 0, :]
        assert (hyp.y_sequence - torch.tensor([3, 1, 20])).sum() == 0
        assert hyp.score == 3  # sum of number of tokens in one hot representation
        assert hyp.text == 'cat'
        assert (hyp.alignments[0] == sample).all()
        assert hyp.length == 0

        length = torch.tensor([tensor.shape[1 - batch_dim_index]], dtype=torch.long)

        # pass batchsize 1 tensor, get back list of length 1 Hypothesis [add length info]
        hyp = wer.decoding.ctc_decoder_predictions_tensor(tensor, decoder_lengths=length, return_hypotheses=True)
        hyp = hyp[0]
        assert isinstance(hyp, Hypothesis)
        assert hyp.length == 3

    def get_wer_ctc(self, prediction: str, reference: str, test_wer_bpe: bool):
        ctc_decoder_predictions_tensor_mock = Mock(
            return_value=[Hypothesis(score=1.0, y_sequence=[], text=prediction)]
        )
        if test_wer_bpe:
            decoding = Mock(
                blank_id=self.char_tokenizer.tokenizer.vocab_size,
                tokenizer=deepcopy(self.char_tokenizer),
                ctc_decoder_predictions_tensor=ctc_decoder_predictions_tensor_mock,
                decode_ids_to_str=self.char_tokenizer.ids_to_text,
                spec=CTCBPEDecoding,
            )
            wer = WER(decoding, use_cer=False)
        else:
            decoding = Mock(
                blank_id=len(self.vocabulary),
                labels_map=self.vocabulary.copy(),
                ctc_decoder_predictions_tensor=ctc_decoder_predictions_tensor_mock,
                decode_ids_to_str=self.decode_token_to_str_with_vocabulary_mock,
                spec=CTCDecoding,
            )
            wer = WER(decoding, use_cer=False)

        predictions_tensor = self.__string_to_ctc_tensor(prediction, test_wer_bpe)
        targets_tensor = self.__reference_string_to_tensor(reference, test_wer_bpe)

        wer(
            predictions=predictions_tensor,
            predictions_lengths=None,
            targets=targets_tensor,
            targets_lengths=torch.tensor([len(reference)]),
        )
        res, _, _ = wer.compute()
        res = res.detach().cpu()
        # return res[0] / res[1]
        return res.item()

    def decode_token_to_str_with_vocabulary_mock(self, ids):
        return ''.join([self.vocabulary[id_] for id_ in ids])

    def get_wer_rnnt(self, prediction: str, reference: str, batch_dim_index: int, test_wer_bpe: bool):
        rnnt_decoder_predictions_tensor_mock = Mock(
            return_value=[Hypothesis(score=1.0, y_sequence=[], text=prediction)]
        )
        if test_wer_bpe:
            decoding = Mock(
                blank_id=self.char_tokenizer.tokenizer.vocab_size,
                tokenizer=deepcopy(self.char_tokenizer),
                rnnt_decoder_predictions_tensor=rnnt_decoder_predictions_tensor_mock,
                decode_ids_to_str=self.char_tokenizer.ids_to_text,
                spec=RNNTBPEDecoding,
            )
            wer = WER(decoding, batch_dim_index=batch_dim_index, use_cer=False)
        else:
            decoding = Mock(
                blank_id=len(self.vocabulary),
                labels_map=self.vocabulary.copy(),
                rnnt_decoder_predictions_tensor=rnnt_decoder_predictions_tensor_mock,
                decode_ids_to_str=self.decode_token_to_str_with_vocabulary_mock,
                spec=RNNTDecoding,
            )
            wer = WER(decoding, batch_dim_index=batch_dim_index, use_cer=False)
        targets_tensor = self.__reference_string_to_tensor(reference, test_wer_bpe)
        if wer.batch_dim_index > 0:
            targets_tensor.transpose_(0, 1)

        # Create proper predictions tensor instead of passing None
        predictions_tensor = self.__string_to_ctc_tensor(prediction, test_wer_bpe)
        if wer.batch_dim_index > 0:
            predictions_tensor.transpose_(0, 1)

        wer(
            predictions=predictions_tensor,
            predictions_lengths=None,
            targets=targets_tensor,
            targets_lengths=torch.tensor([len(reference)]),
        )
        res, _, _ = wer.compute()
        res = res.detach().cpu()
        # return res[0] / res[1]
        return res.item()

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_dim_index", [0, 1])
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_rnnt_wer_metric_simple(self, batch_dim_index, test_wer_bpe):
        assert self.get_wer_rnnt('cat', 'cot', batch_dim_index, test_wer_bpe) == 1.0
        assert self.get_wer_rnnt('gpu', 'g p u', batch_dim_index, test_wer_bpe) == 1.0
        assert self.get_wer_rnnt('g p u', 'gpu', batch_dim_index, test_wer_bpe) == 3.0
        assert self.get_wer_rnnt('ducati motorcycle', 'motorcycle', batch_dim_index, test_wer_bpe) == 1.0
        assert self.get_wer_rnnt('ducati motorcycle', 'ducuti motorcycle', batch_dim_index, test_wer_bpe) == 0.5
        assert abs(self.get_wer_rnnt('a f c', 'a b c', batch_dim_index, test_wer_bpe) - 1.0 / 3.0) < 1e-6

    @pytest.mark.unit
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_rnnt_wer_metric_randomized(self, test_wer_bpe):
        """This test relies on correctness of word_error_rate function."""

        def __random_string(length):
            return ''.join(random.choice(''.join(self.vocabulary)) for _ in range(length))

        for test_id in range(256):
            n1 = random.randint(1, 512)
            n2 = random.randint(1, 512)
            s1 = __random_string(n1)
            s2 = __random_string(n2)
            # skip empty strings as reference
            if s2.strip():
                assert (
                    abs(
                        self.get_wer_rnnt(prediction=s1, reference=s2, batch_dim_index=0, test_wer_bpe=test_wer_bpe)
                        - word_error_rate(hypotheses=[s1], references=[s2])
                    )
                    < 1e-6
                )

    @pytest.mark.unit
    def test_char_decoding_logprobs(self):
        B, T, V = 1, 8, len(self.vocabulary)
        torch.manual_seed(0)
        decoder_outputs = torch.randn(B, T, V, dtype=torch.float32)
        decoder_lens = torch.randint(0, T, size=[B], dtype=torch.int32)
        decoder_lens[torch.randint(0, B, [1])[0]] = T

        decoding_cfg = CTCDecodingConfig()
        decoding = CTCDecoding(decoding_cfg, vocabulary=self.vocabulary)

        hyp = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestamp) == 0
        assert hyp.alignments is None

        # Preserve timestamps and alignments
        decoding_cfg = CTCDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        decoding = CTCDecoding(decoding_cfg, vocabulary=self.vocabulary)

        hyp = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestamp) == 4
        assert hyp.alignments is not None

    @pytest.mark.unit
    def test_subword_decoding_logprobs(self):
        B, T, V = 1, 8, self.char_tokenizer.vocab_size
        torch.manual_seed(0)
        decoder_outputs = torch.randn(B, T, V, dtype=torch.float32)
        decoder_lens = torch.randint(0, T, size=[B], dtype=torch.int32)
        decoder_lens[torch.randint(0, B, [1])[0]] = T

        decoding_cfg = CTCBPEDecodingConfig()
        decoding = CTCBPEDecoding(decoding_cfg, tokenizer=self.char_tokenizer)

        hyp = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestamp) == 0
        assert hyp.alignments is None

        # Preserve timestamps and alignments
        decoding_cfg = CTCBPEDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        decoding = CTCBPEDecoding(decoding_cfg, tokenizer=self.char_tokenizer)

        hyp = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestamp) == 4
        assert hyp.alignments is not None

    @pytest.mark.unit
    def test_char_decoding_labels(self):
        B, T, V = 1, 8, len(self.vocabulary)
        torch.manual_seed(0)
        decoder_outputs = torch.randint(0, V + 1, size=[B, T], dtype=torch.float32)
        decoder_lens = torch.randint(0, T, size=[B], dtype=torch.int32)
        decoder_lens[torch.randint(0, B, [1])[0]] = T

        decoding_cfg = CTCDecodingConfig()
        decoding = CTCDecoding(decoding_cfg, vocabulary=self.vocabulary)

        hyp = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestamp) == 0
        assert hyp.alignments is None

        # Preserve timestamps and alignments
        decoding_cfg = CTCDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        decoding = CTCDecoding(decoding_cfg, vocabulary=self.vocabulary)

        # Cannot compute alignments from labels
        with pytest.raises(ValueError):
            _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)

        # Preserve timestamps
        decoding_cfg = CTCDecodingConfig(preserve_alignments=False, compute_timestamps=True)
        decoding = CTCDecoding(decoding_cfg, vocabulary=self.vocabulary)

        hyp = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestamp) == 4
        assert hyp.alignments is None

    @pytest.mark.unit
    def test_subword_decoding_logprobs(self):
        B, T, V = 1, 8, self.char_tokenizer.vocab_size
        torch.manual_seed(0)
        decoder_outputs = torch.randn(B, T, V, dtype=torch.float32)
        decoder_lens = torch.randint(0, T, size=[B], dtype=torch.int32)
        decoder_lens[torch.randint(0, B, [1])[0]] = T

        decoding_cfg = CTCBPEDecodingConfig()
        decoding = CTCBPEDecoding(decoding_cfg, tokenizer=self.char_tokenizer)

        hyp = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestamp) == 0
        assert hyp.alignments is None

        # Preserve timestamps and alignments
        decoding_cfg = CTCBPEDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        decoding = CTCBPEDecoding(decoding_cfg, tokenizer=self.char_tokenizer)

        hyp = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestamp) == 4
        assert hyp.alignments is not None

    @pytest.mark.unit
    def test_subword_decoding_labels(self):
        B, T, V = 1, 8, self.char_tokenizer.vocab_size
        torch.manual_seed(0)
        decoder_outputs = torch.randint(0, V + 1, size=[B, T], dtype=torch.float32)
        decoder_lens = torch.randint(0, T, size=[B], dtype=torch.int32)
        decoder_lens[torch.randint(0, B, [1])[0]] = T

        decoding_cfg = CTCBPEDecodingConfig()
        decoding = CTCBPEDecoding(decoding_cfg, tokenizer=self.char_tokenizer)

        hyp = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestamp) == 0
        assert hyp.alignments is None

        # Preserve timestamps and alignments
        decoding_cfg = CTCBPEDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        decoding = CTCBPEDecoding(decoding_cfg, tokenizer=self.char_tokenizer)

        # Cannot compute alignments from labels
        with pytest.raises(ValueError):
            _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)

        # Preserve timestamps
        decoding_cfg = CTCBPEDecodingConfig(preserve_alignments=False, compute_timestamps=True)
        decoding = CTCBPEDecoding(decoding_cfg, tokenizer=self.char_tokenizer)

        hyp = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestamp) == 4
        assert hyp.alignments is None


class TestBLEUHelperFunctions:
    """Test BLEU helper functions"""

    @pytest.mark.unit
    def test_get_bleu_tokenizers_from_cuts_missing_tokenizer(self):
        """Test handling cuts without BLEU tokenizer"""
        cut = Mock()
        cut.custom = {}  # Empty dict, no BLEU_TOKENIZER key
        cut.first_non_padding_cut = cut

        cuts = [cut]
        tokenizers = _get_bleu_tokenizers_from_cuts(cuts)

        assert tokenizers == [None]

    @pytest.mark.unit
    def test_move_dimension_to_the_front(self):
        """Test moving tensor dimensions"""
        # Create test tensor [batch, time, features]
        tensor = torch.randn(2, 10, 128)

        # Move time dimension to front
        moved = _move_dimension_to_the_front(tensor, 1)
        assert moved.shape == (10, 2, 128)

        # Move features dimension to front
        moved = _move_dimension_to_the_front(tensor, 2)
        assert moved.shape == (128, 2, 10)

        # Move batch dimension to front (should be unchanged)
        moved = _move_dimension_to_the_front(tensor, 0)
        assert moved.shape == (2, 10, 128)


class TestBLEUMetric:
    """Test BLEU metric functionality"""

    vocabulary = [' '] + list(string.ascii_lowercase) + ["'"] + ['你', '好', '世', '界', '朋', '友']
    char_tokenizer = build_char_tokenizer_with_vocabulary(vocabulary)

    def create_mock_decoding(self, decode_type="ctc"):
        """Create mock decoding instance"""
        decoding = None
        if decode_type == "ctc":
            decoding = Mock(spec=AbstractCTCDecoding)
            decoding.decode_ids_to_str = lambda tokens: ''.join([self.vocabulary[id_] for id_ in tokens])
            decoding.ctc_decoder_predictions_tensor = Mock(
                return_value=[Hypothesis(score=1.0, y_sequence=[], text="hello world")]
            )
        elif decode_type == "rnnt":
            decoding = Mock(spec=AbstractRNNTDecoding)
            decoding.decode_ids_to_str = lambda tokens: ''.join([self.vocabulary[id_] for id_ in tokens])
            decoding.rnnt_decoder_predictions_tensor = Mock(
                return_value=[Hypothesis(score=1.0, y_sequence=[], text="hello world")]
            )
        elif decode_type == "multitask":
            decoding = Mock(spec=AbstractMultiTaskDecoding)
            decoding.decode_ids_to_str = lambda tokens: ''.join([self.vocabulary[id_] for id_ in tokens])
            decoding.decode_predictions_tensor = Mock(
                return_value=[Hypothesis(score=1.0, y_sequence=[], text="hello world")]
            )
        else:
            raise TypeError(f"`decode_type:` {decode_type} is invalid type for `create_mock_decoding'")
        return decoding

    def __reference_string_to_tensor(self, txt: str) -> torch.Tensor:
        """Convert reference string to tensor"""
        char_to_ind = dict([(self.vocabulary[i], i) for i in range(len(self.vocabulary))])
        string_in_id_form = [char_to_ind[c] for c in txt]
        return torch.tensor(string_in_id_form).unsqueeze(0)

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_dim_index", [0, 1])
    @pytest.mark.parametrize("decode_type", ["ctc", "rnnt", "multitask"])
    def test_bleu_initialization(self, batch_dim_index, decode_type):
        """Test BLEU metric initialization with different decoders"""
        decoding = self.create_mock_decoding(decode_type)

        bleu = BLEU(
            decoding=decoding, batch_dim_index=batch_dim_index, bleu_tokenizer="13a", n_gram=4, lowercase=False
        )

        assert bleu.batch_dim_index == batch_dim_index
        assert bleu.decoding == decoding
        assert bleu.n_gram == 4
        assert bleu.decode is not None

    @pytest.mark.unit
    def test_bleu_update_basic(self):
        """Test basic BLEU update functionality"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a")

        # Create test tensors
        batch_size = 2
        predictions = torch.randn(batch_size, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80, 90])
        targets = self.__reference_string_to_tensor("hello world")
        targets = targets.repeat(batch_size, 1)
        targets_lengths = torch.tensor([11, 11])  # Length of "hello world"

        # Mock decode function to return expected predictions
        decoding.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text="hello world"),
            Hypothesis(score=1.0, y_sequence=[], text="hello earth"),
        ]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        # Verify decode was called
        decoding.ctc_decoder_predictions_tensor.assert_called_once()

    @pytest.mark.unit
    def test_bleu_update_empty_predictions(self):
        """Test BLEU update with empty predictions"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a")

        # Create empty predictions tensor
        predictions = torch.empty(0, 100, len(self.vocabulary))
        predictions_lengths = torch.empty(0, dtype=torch.long)
        targets = torch.empty(0, 50, dtype=torch.long)
        targets_lengths = torch.empty(0, dtype=torch.long)

        # Should not raise an error
        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        # Decode should not be called for empty predictions
        decoding.ctc_decoder_predictions_tensor.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_dim_index", [0, 1])
    def test_bleu_update_different_batch_dims(self, batch_dim_index):
        """Test BLEU update with different batch dimension indices"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, batch_dim_index=batch_dim_index, bleu_tokenizer="13a")

        batch_size = 2
        time_steps = 50
        vocab_size = len(self.vocabulary)

        if batch_dim_index == 0:
            predictions = torch.randn(batch_size, time_steps, vocab_size)
            targets = torch.randint(0, vocab_size, (batch_size, time_steps))
        else:
            predictions = torch.randn(time_steps, batch_size, vocab_size)
            targets = torch.randint(0, vocab_size, (time_steps, batch_size))

        predictions_lengths = torch.tensor([40, 45])
        targets_lengths = torch.tensor([35, 40])

        decoding.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text="test"),
            Hypothesis(score=1.0, y_sequence=[], text="sample"),
        ]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        decoding.ctc_decoder_predictions_tensor.assert_called_once()

    @pytest.mark.unit
    def test_bleu_with_cuts_tokenizers(self):
        """Test BLEU with different tokenizers from cuts"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, check_cuts_for_tokenizers=True, bleu_tokenizer="13a")

        # Create cuts with different tokenizers
        cut1 = Mock()
        cut1.custom = {BLEU_TOKENIZER: "13a"}  # Use dict instead of Mock
        cut1.first_non_padding_cut = cut1

        cut2 = Mock()
        cut2.custom = {BLEU_TOKENIZER: "zh"}  # Use dict instead of Mock
        cut2.first_non_padding_cut = cut2

        cuts = [cut1, cut2]

        batch_size = 2
        predictions = torch.randn(batch_size, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80, 90])
        targets = self.__reference_string_to_tensor("test")
        targets = targets.repeat(batch_size, 1)
        targets_lengths = torch.tensor([4, 4])

        decoding.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text="test"),
            Hypothesis(score=1.0, y_sequence=[], text="测试"),
        ]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
            cuts=cuts,
        )

        decoding.ctc_decoder_predictions_tensor.assert_called_once()

    @pytest.mark.unit
    def test_bleu_cuts_length_mismatch(self):
        """Test BLEU with mismatched cuts and batch size"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, check_cuts_for_bleu_tokenizers=True)

        # Create cuts with wrong length
        cuts = [Mock()]  # Only 1 cut

        batch_size = 2  # But 2 samples
        predictions = torch.randn(batch_size, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80, 90])
        targets = torch.randint(0, len(self.vocabulary), (batch_size, 50))
        targets_lengths = torch.tensor([40, 45])

        with pytest.raises(AssertionError, match="BLEU metrics configured for multiple tokenizers"):
            bleu.update(
                predictions=predictions,
                predictions_lengths=predictions_lengths,
                targets=targets,
                targets_lengths=targets_lengths,
                cuts=cuts,
            )

    @pytest.mark.unit
    def test_bleu_perfect_match(self):
        """Test BLEU calculation with perfect prediction match"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a", n_gram=1)

        # Use very simple text to debug
        perfect_text = "hello"
        predictions = torch.randn(1, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80])
        targets = self.__reference_string_to_tensor(perfect_text)
        targets_lengths = torch.tensor([targets.shape[1]]).unsqueeze(0)

        # Mock decode to return exact match
        decoding.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text=perfect_text)
        ]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        result = bleu.compute(return_all_metrics=False)

        # Perfect match should give BLEU score close to 1.0 (allow small tolerance)
        assert result["bleu"].item() == 1.0

    @pytest.mark.unit
    def test_bleu_no_match(self):
        """Test BLEU calculation with no matching words"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a")

        predictions = torch.randn(1, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80])
        targets = self.__reference_string_to_tensor("hello world")
        targets_lengths = torch.tensor([11])  # "hello world"

        # Mock decode to return completely different text
        decoding.ctc_decoder_predictions_tensor.return_value = [Hypothesis(score=1.0, y_sequence=[], text="cat dog")]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        result = bleu.compute(return_all_metrics=False)
        # No matching words should give BLEU score of 0.0
        assert result["bleu"].item() == 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize("n_gram", [1, 2, 3, 4])
    def test_bleu_partial_match(self, n_gram):
        """Test BLEU calculation with partial word matches."""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a", n_gram=n_gram)

        predictions = torch.randn(1, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80])
        targets = self.__reference_string_to_tensor("the quick brown fox jumps")
        targets_lengths = torch.tensor([targets.shape[1]])  # Use actual tensor length

        # Mock decode to return partial match with overlapping n-grams
        decoding.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text="the quick brown fox runs")
        ]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        result = bleu.compute(return_all_metrics=False)["bleu"]

        # Manual BLEU calculation for different n-gram levels:
        # Reference: "the quick brown fox jumps" vs Prediction: "the quick brown fox runs"
        # 1-grams: 4/5 matches (the, quick, brown, fox match; jumps≠runs)
        # 2-grams: 3/4 matches (the quick, quick brown, brown fox match; fox jumps≠fox runs)
        # 3-grams: 2/3 matches (the quick brown, quick brown fox match; brown fox jumps≠brown fox runs)
        # 4-grams: 1/2 matches (the quick brown fox matches; quick brown fox jumps≠quick brown fox runs)

        p1, p2, p3, p4 = 4 / 5, 3 / 4, 2 / 3, 1 / 2

        # Calculate expected BLEU for each n-gram level
        expected_bleu = -1
        if n_gram == 1:
            expected_bleu = p1
        elif n_gram == 2:
            expected_bleu = math.sqrt(p1 * p2)
        elif n_gram == 3:
            expected_bleu = (p1 * p2 * p3) ** (1 / 3)
        elif n_gram == 4:
            expected_bleu = (p1 * p2 * p3 * p4) ** (1 / 4)
        else:
            raise ValueError(f"`n_gram` value of {n_gram} is not supported by `test_bleu_partial_match")

        # BP = 1 (same length: 5 words each)
        assert (
            abs(result.item() - expected_bleu) < 0.1
        ), f"Expected BLEU ≈ {expected_bleu:.3f}, got {result.item():.3f}"

    @pytest.mark.unit
    def test_bleu_empty_prediction(self):
        """Test BLEU calculation with empty prediction"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a")

        predictions = torch.randn(1, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80])
        targets = self.__reference_string_to_tensor("hello world")
        targets_lengths = torch.tensor([11])

        # Mock decode to return empty text
        decoding.ctc_decoder_predictions_tensor.return_value = [Hypothesis(score=1.0, y_sequence=[], text="")]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        result = bleu.compute()
        # Empty prediction should give BLEU score of 0.0
        assert result["bleu"].item() == 0.0

    @pytest.mark.unit
    def test_bleu_empty_reference(self):
        """Test BLEU calculation with empty reference"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a")

        predictions = torch.randn(1, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80])
        targets = self.__reference_string_to_tensor("")
        targets_lengths = torch.tensor([0])

        # Mock decode to return some text
        decoding.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text="hello world")
        ]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        result = bleu.compute()
        # Empty reference should give BLEU score of 0.0
        assert result["bleu"].item() == 0.0

    @pytest.mark.unit
    def test_bleu_multiple_samples(self):
        """Test BLEU calculation with multiple samples in batch"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a", n_gram=2)  # Use 2-gram for exact calculation

        # Test with 3 samples: perfect match, partial match, no match
        batch_size = 3
        predictions = torch.randn(batch_size, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80, 85, 90])

        # Create targets for each sample
        target1 = self.__reference_string_to_tensor("hello world")
        target2 = self.__reference_string_to_tensor("test case")
        target3 = self.__reference_string_to_tensor("example text")

        # Pad targets to same length
        max_len = max(target1.shape[1], target2.shape[1], target3.shape[1])
        targets = torch.zeros(batch_size, max_len, dtype=torch.long)
        targets[0, : target1.shape[1]] = target1[0]
        targets[1, : target2.shape[1]] = target2[0]
        targets[2, : target3.shape[1]] = target3[0]

        targets_lengths = torch.tensor([target1.shape[1], target2.shape[1], target3.shape[1]])

        # Mock decode with different quality predictions
        decoding.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text="hello world"),  # Perfect match
            Hypothesis(score=1.0, y_sequence=[], text="test different"),  # Partial match (1/2 words)
            Hypothesis(score=1.0, y_sequence=[], text="completely wrong"),  # No match (0/2 words)
        ]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        result = bleu.compute(return_all_metrics=False)["bleu"]

        # Corpus-level BLEU calculation:
        # Sample 1: "hello world" vs "hello world"
        #   1-grams: 2/2 matches, 2-grams: 1/1 matches
        # Sample 2: "test different" vs "test case"
        #   1-grams: 1/2 matches (only "test"), 2-grams: 0/1 matches
        # Sample 3: "completely wrong" vs "example text"
        #   1-grams: 0/2 matches, 2-grams: 0/1 matches
        #
        # Corpus totals:
        # p1 = (2 + 1 + 0) / (2 + 2 + 2) = 3/6 = 0.5
        # p2 = (1 + 0 + 0) / (1 + 1 + 1) = 1/3 ≈ 0.3333
        # BP = 1.0 (prediction length = reference length = 6 words total)
        # BLEU = sqrt(p1 * p2) = sqrt(0.5 * 0.3333) = sqrt(0.1667) ≈ 0.408

        p1 = 3 / 6  # 0.5
        p2 = 1 / 3  # 0.3333
        expected_bleu = math.sqrt(p1 * p2)  # ≈ 0.408

        # Allow reasonable tolerance for BLEU implementation differences
        assert (
            abs(result.item() - expected_bleu) < 0.1
        ), f"Expected BLEU ≈ {expected_bleu:.2f}, got {result.item():.2f}"

    @pytest.mark.unit
    @pytest.mark.parametrize("n_gram", [1, 2, 3, 4])
    def test_bleu_different_ngram_calculations(self, n_gram):
        """Test that different n-gram settings produce different BLEU scores"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a", n_gram=n_gram)

        predictions = torch.randn(1, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80])
        targets = self.__reference_string_to_tensor("the quick brown fox jumps")
        targets_lengths = torch.tensor([targets.shape[1]])  # Use actual tensor length

        # Mock decode with slightly different word order
        decoding.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text="quick brown fox the jumps")
        ]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        result = bleu.compute(return_all_metrics=False)["bleu"]
        # Different n-gram values should produce valid BLEU scores
        assert 0.0 <= result.item() <= 1.0

        # For this specific case:
        # - 1-gram should have high score (all words match)
        # - Higher n-grams should have lower scores due to word order
        if n_gram == 1:
            assert result.item() > 0.5  # Should be high since all words match
        else:
            # Higher n-grams might be lower due to order differences
            assert result.item() >= 0.0

    @pytest.mark.unit
    def test_bleu_multi_tokenization(self):
        """Test BLEU calculation with multiple tokenizers for different languages"""
        # Test without multi-tokenization (single tokenizer)
        decoding_single = self.create_mock_decoding("ctc")
        bleu_single = BLEU(
            decoding=decoding_single, bleu_tokenizer="13a", check_cuts_for_bleu_tokenizers=False, n_gram=2
        )

        # Test with multi-tokenization (tokenizers from cuts)
        decoding_multi = self.create_mock_decoding("ctc")
        bleu_multi = BLEU(decoding=decoding_multi, bleu_tokenizer="13a", check_cuts_for_bleu_tokenizers=True, n_gram=2)

        batch_size = 2
        predictions = torch.randn(batch_size, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80, 85])

        # Create English and Chinese targets
        english_target = self.__reference_string_to_tensor("hello world")
        chinese_target = self.__reference_string_to_tensor("你好世界")  # "hello world" in Chinese

        # Pad targets to same length
        max_len = max(english_target.shape[1], chinese_target.shape[1])
        targets = torch.zeros(batch_size, max_len, dtype=torch.long)
        targets[0, : english_target.shape[1]] = english_target[0]
        targets[1, : chinese_target.shape[1]] = chinese_target[0]

        targets_lengths = torch.tensor([english_target.shape[1], chinese_target.shape[1]])

        # Create cuts specifying different tokenizers
        cut1 = Mock()
        cut1.custom = {BLEU_TOKENIZER: "13a"}  # Use dict instead of Mock
        cut1.first_non_padding_cut = cut1

        cut2 = Mock()
        cut2.custom = {BLEU_TOKENIZER: "zh"}  # Use dict instead of Mock
        cut2.first_non_padding_cut = cut2

        cuts = [cut1, cut2]

        # Mock predictions - partial matches for both languages
        english_prediction = "hello earth"  # Partial match with English reference
        chinese_prediction = "你好朋友"  # Partial match with Chinese reference

        decoding_single.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text=english_prediction),
            Hypothesis(score=1.0, y_sequence=[], text=chinese_prediction),
        ]

        decoding_multi.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text=english_prediction),
            Hypothesis(score=1.0, y_sequence=[], text=chinese_prediction),
        ]

        # Test single tokenizer (13a for both samples)
        bleu_single.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )
        result_single = bleu_single.compute(return_all_metrics=False)["bleu"]

        # Test multi tokenizer (13a for English, zh for Chinese)
        bleu_multi.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
            cuts=cuts,
        )
        result_multi = bleu_multi.compute(return_all_metrics=False)["bleu"]

        # Calculate expected BLEU for multi-tokenization case:
        # English (13a tokenizer): "hello earth" vs "hello world"
        #   - 1-grams: "hello" matches, "earth" ≠ "world" → 1/2 = 0.5
        #   - 2-grams: "hello earth" ≠ "hello world" → 0/1 = 0.0
        # Chinese (zh tokenizer): "你好朋友" vs "你好世界"
        #   - If character-level: 1-grams: "你","好" match, "朋","友" ≠ "世","界" → 2/4 = 0.5
        #   - If character-level: 2-grams: "你好" matches, others don't → 1/3 ≈ 0.33
        #
        # Corpus-level calculation (character-level assumption):
        # p1 = (1 + 2) / (2 + 4) = 3/6 = 0.5
        # p2 = (0 + 1) / (1 + 3) = 1/4 = 0.25
        # BLEU = sqrt(p1 * p2) = sqrt(0.5 * 0.25) = sqrt(0.125) ≈ 0.354
        expected_multi_bleu = math.sqrt(0.5 * 0.25)  # ≈ 0.354

        # Assert the two methods produce different results
        assert (
            abs(result_single.item() - result_multi.item()) > 0.01
        ), "Multi-tokenization should produce different BLEU scores"

        # Assert multi-tokenization result is within expected range
        assert (
            abs(result_multi.item() - expected_multi_bleu) < 0.1
        ), f"Multi-tokenization BLEU should be near {expected_multi_bleu:.4f}, got {result_multi.item():.4f}"

    @pytest.mark.unit
    def test_bleu_empty_prediction(self):
        """Test BLEU calculation with empty prediction"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a")

        predictions = torch.randn(1, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80])
        targets = self.__reference_string_to_tensor("hello world")
        targets_lengths = torch.tensor([11])

        # Mock decode to return empty text
        decoding.ctc_decoder_predictions_tensor.return_value = [Hypothesis(score=1.0, y_sequence=[], text="")]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        result = bleu.compute()
        # Empty prediction should give BLEU score of 0.0
        assert result["bleu"].item() == 0.0

    @pytest.mark.unit
    def test_bleu_empty_reference(self):
        """Test BLEU calculation with empty reference"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a")

        predictions = torch.randn(1, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80])
        targets = self.__reference_string_to_tensor("")
        targets_lengths = torch.tensor([0])

        # Mock decode to return some text
        decoding.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text="hello world")
        ]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        result = bleu.compute()
        # Empty reference should give BLEU score of 0.0
        assert result["bleu"].item() == 0.0

    @pytest.mark.unit
    def test_bleu_multiple_samples(self):
        """Test BLEU calculation with multiple samples in batch"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a", n_gram=2)  # Use 2-gram for exact calculation

        # Test with 3 samples: perfect match, partial match, no match
        batch_size = 3
        predictions = torch.randn(batch_size, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80, 85, 90])

        # Create targets for each sample
        target1 = self.__reference_string_to_tensor("hello world")
        target2 = self.__reference_string_to_tensor("test case")
        target3 = self.__reference_string_to_tensor("example text")

        # Pad targets to same length
        max_len = max(target1.shape[1], target2.shape[1], target3.shape[1])
        targets = torch.zeros(batch_size, max_len, dtype=torch.long)
        targets[0, : target1.shape[1]] = target1[0]
        targets[1, : target2.shape[1]] = target2[0]
        targets[2, : target3.shape[1]] = target3[0]

        targets_lengths = torch.tensor([target1.shape[1], target2.shape[1], target3.shape[1]])

        # Mock decode with different quality predictions
        decoding.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text="hello world"),  # Perfect match
            Hypothesis(score=1.0, y_sequence=[], text="test different"),  # Partial match (1/2 words)
            Hypothesis(score=1.0, y_sequence=[], text="completely wrong"),  # No match (0/2 words)
        ]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        result = bleu.compute(return_all_metrics=False)["bleu"]

        # Corpus-level BLEU calculation:
        # Sample 1: "hello world" vs "hello world"
        #   1-grams: 2/2 matches, 2-grams: 1/1 matches
        # Sample 2: "test different" vs "test case"
        #   1-grams: 1/2 matches (only "test"), 2-grams: 0/1 matches
        # Sample 3: "completely wrong" vs "example text"
        #   1-grams: 0/2 matches, 2-grams: 0/1 matches
        #
        # Corpus totals:
        # p1 = (2 + 1 + 0) / (2 + 2 + 2) = 3/6 = 0.5
        # p2 = (1 + 0 + 0) / (1 + 1 + 1) = 1/3 ≈ 0.3333
        # BP = 1.0 (prediction length = reference length = 6 words total)
        # BLEU = sqrt(p1 * p2) = sqrt(0.5 * 0.3333) = sqrt(0.1667) ≈ 0.408

        p1 = 3 / 6  # 0.5
        p2 = 1 / 3  # 0.3333
        expected_bleu = math.sqrt(p1 * p2)  # ≈ 0.408

        # Allow reasonable tolerance for BLEU implementation differences
        assert (
            abs(result.item() - expected_bleu) < 0.1
        ), f"Expected BLEU ≈ {expected_bleu:.2f}, got {result.item():.2f}"

    @pytest.mark.unit
    @pytest.mark.parametrize("n_gram", [1, 2, 3, 4])
    def test_bleu_different_ngram_calculations(self, n_gram):
        """Test that different n-gram settings produce different BLEU scores"""
        decoding = self.create_mock_decoding("ctc")
        bleu = BLEU(decoding=decoding, bleu_tokenizer="13a", n_gram=n_gram)

        predictions = torch.randn(1, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80])
        targets = self.__reference_string_to_tensor("the quick brown fox jumps")
        targets_lengths = torch.tensor([targets.shape[1]])  # Use actual tensor length

        # Mock decode with slightly different word order
        decoding.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text="quick brown fox the jumps")
        ]

        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        result = bleu.compute(return_all_metrics=False)["bleu"]
        # Different n-gram values should produce valid BLEU scores
        assert 0.0 <= result.item() <= 1.0

        # For this specific case:
        # - 1-gram should have high score (all words match)
        # - Higher n-grams should have lower scores due to word order
        if n_gram == 1:
            assert result.item() > 0.5  # Should be high since all words match
        else:
            # Higher n-grams might be lower due to order differences
            assert result.item() >= 0.0


class TestBLEUEdgeCases:
    """Test BLEU edge cases and error conditions"""

    vocabulary = [' '] + list(string.ascii_lowercase) + ["'"] + ['你', '好', '世', '界', '朋', '友']

    def create_mock_decoding(self):
        """Create minimal mock decoding"""
        decoding = Mock(spec=AbstractCTCDecoding)
        decoding.decode_ids_to_str = lambda tokens: ''.join([self.vocabulary[id_] for id_ in tokens])
        decoding.ctc_decoder_predictions_tensor = Mock(return_value=[])
        return decoding

    @pytest.mark.unit
    def test_bleu_empty_hypotheses(self):
        """Test BLEU with empty hypotheses"""
        decoding = self.create_mock_decoding()
        # Return empty text hypothesis to match the number of targets
        decoding.ctc_decoder_predictions_tensor.return_value = [
            Hypothesis(score=1.0, y_sequence=[], text="")  # Empty text instead of empty list
        ]

        bleu = BLEU(decoding=decoding)

        predictions = torch.randn(1, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80])
        targets = torch.randint(0, len(self.vocabulary), (1, 50))
        targets_lengths = torch.tensor([40])

        # Should not raise an error even with empty hypotheses
        # The update method should handle empty hypotheses gracefully
        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )

        # Compute should also handle empty state
        result = bleu.compute(return_all_metrics=False)
        assert result["bleu"].item() == 0.0  # Empty hypotheses should give BLEU score of 0.0

    @pytest.mark.unit
    def test_bleu_zero_length_targets(self):
        """Test BLEU with zero-length targets"""
        decoding = self.create_mock_decoding()
        bleu = BLEU(decoding=decoding)

        predictions = torch.randn(1, 100, len(self.vocabulary))
        predictions_lengths = torch.tensor([80])
        targets = torch.randint(0, len(self.vocabulary), (1, 50))
        targets_lengths = torch.tensor([0])  # Zero length

        decoding.ctc_decoder_predictions_tensor.return_value = [Hypothesis(score=1.0, y_sequence=[], text="test")]

        # Should handle zero-length targets gracefully
        bleu.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=targets,
            targets_lengths=targets_lengths,
        )


class TestMultiTaskMetricConstraintFunctions:
    """Test the constraint parsing and evaluation functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.parser = ConstraintParser()

    @pytest.mark.unit
    def test_static_constraint_equality(self):
        """Test static constraint with equality operator"""
        parser = ConstraintParser()
        properties = {"task": "transcribe", "lang": "en"}

        # Test successful match
        result = parser._static_constraint(operator.eq, "task", "transcribe", properties)
        assert result is True

        # Test failed match
        result = parser._static_constraint(operator.eq, "task", "translate", properties)
        assert result is False

        # Test missing key
        result = parser._static_constraint(operator.eq, "missing", "value", properties)
        assert result is False

    @pytest.mark.unit
    def test_static_constraint_inequality(self):
        """Test static constraint with inequality operator"""
        parser = ConstraintParser()
        properties = {"task": "transcribe", "lang": "en"}

        result = parser._static_constraint(operator.ne, "task", "translate", properties)
        assert result is True

        result = parser._static_constraint(operator.ne, "task", "transcribe", properties)
        assert result is False

    @pytest.mark.unit
    def test_compare_constraint(self):
        """Test comparing two properties"""
        parser = ConstraintParser()
        properties = {"source_lang": "en", "target_lang": "en", "other": "different"}

        # Test equal properties
        result = parser._compare_constraint(operator.eq, "source_lang", "target_lang", properties)
        assert result is True

        # Test unequal properties
        result = parser._compare_constraint(operator.eq, "source_lang", "other", properties)
        assert result is False

        # Test missing property
        result = parser._compare_constraint(operator.eq, "source_lang", "missing", properties)
        assert result is False

    @pytest.mark.unit
    def test_logical_operations(self):
        """Test logical AND, OR, NOT operations"""
        parser = ConstraintParser()
        properties = {"task": "transcribe", "lang": "en"}

        # Create simple constraint functions
        true_constraint = lambda p: p.get("task") == "transcribe"
        false_constraint = lambda p: p.get("task") == "translate"

        # Test AND
        result = parser._logical_and(true_constraint, false_constraint, properties)
        assert result is False

        # Test OR
        result = parser._logical_or(true_constraint, false_constraint, properties)
        assert result is True

        result = parser._logical_or(false_constraint, false_constraint, properties)
        assert result is False

        # Test NOT
        result = parser._logical_not(true_constraint, properties)
        assert result is False

        result = parser._logical_not(false_constraint, properties)
        assert result is True


class TestMultiTaskMetricConstraintParsing:
    """Test the constraint string parsing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.parser = ConstraintParser()

    @pytest.mark.unit
    def test_simple_equality_constraint(self):
        """Test parsing simple equality constraints"""
        parser = ConstraintParser()
        constraint_fn = parser.parse_constraint(".task==transcribe")

        properties = {"task": "transcribe"}
        assert constraint_fn(properties) is True

        properties = {"task": "translate"}
        assert constraint_fn(properties) is False

    @pytest.mark.unit
    def test_simple_inequality_constraint(self):
        """Test parsing simple inequality constraints"""
        parser = ConstraintParser()
        constraint_fn = parser.parse_constraint(".task!=translate")

        properties = {"task": "transcribe"}
        assert constraint_fn(properties) is True

        properties = {"task": "translate"}
        assert constraint_fn(properties) is False

    @pytest.mark.unit
    def test_property_comparison_constraint(self):
        """Test parsing property-to-property comparisons"""
        parser = ConstraintParser()
        constraint_fn = parser.parse_constraint(".source_lang==.target_lang")

        properties = {"source_lang": "en", "target_lang": "en"}
        assert constraint_fn(properties) is True

        properties = {"source_lang": "en", "target_lang": "de"}
        assert constraint_fn(properties) is False

    @pytest.mark.unit
    def test_and_constraint(self):
        """Test parsing AND constraints"""
        parser = ConstraintParser()
        constraint_fn = parser.parse_constraint(".task==transcribe and .lang==en")

        properties = {"task": "transcribe", "lang": "en"}
        assert constraint_fn(properties) is True

        properties = {"task": "transcribe", "lang": "de"}
        assert constraint_fn(properties) is False

        properties = {"task": "translate", "lang": "en"}
        assert constraint_fn(properties) is False

    @pytest.mark.unit
    def test_or_constraint(self):
        """Test parsing OR constraints"""
        parser = ConstraintParser()
        constraint_fn = parser.parse_constraint(".task==transcribe or .task==translate")

        properties = {"task": "transcribe"}
        assert constraint_fn(properties) is True

        properties = {"task": "translate"}
        assert constraint_fn(properties) is True

        properties = {"task": "other"}
        assert constraint_fn(properties) is False

    @pytest.mark.unit
    def test_not_constraint(self):
        """Test parsing NOT constraints"""
        parser = ConstraintParser()
        constraint_fn = parser.parse_constraint("not .task==translate")

        properties = {"task": "transcribe"}
        assert constraint_fn(properties) is True

        properties = {"task": "translate"}
        assert constraint_fn(properties) is False

    @pytest.mark.unit
    def test_complex_constraint(self):
        """Test parsing complex nested constraints"""
        parser = ConstraintParser()
        constraint_fn = parser.parse_constraint(".task==transcribe and .source_lang==.target_lang")

        properties = {"task": "transcribe", "source_lang": "en", "target_lang": "en"}
        assert constraint_fn(properties) is True

        properties = {"task": "transcribe", "source_lang": "en", "target_lang": "de"}
        assert constraint_fn(properties) is False

        properties = {"task": "translate", "source_lang": "en", "target_lang": "en"}
        assert constraint_fn(properties) is False

    @pytest.mark.unit
    def test_parentheses_constraint(self):
        """Test parsing constraints with parentheses"""
        parser = ConstraintParser()
        # Basic parentheses
        constraint_fn = parser.parse_constraint("(.task==transcribe or .task==translate) and .lang==en")

        properties = {"task": "transcribe", "lang": "en"}
        assert constraint_fn(properties) is True

        properties = {"task": "translate", "lang": "en"}
        assert constraint_fn(properties) is True

        properties = {"task": "transcribe", "lang": "de"}
        assert constraint_fn(properties) is False

        properties = {"task": "other", "lang": "en"}
        assert constraint_fn(properties) is False

    @pytest.mark.unit
    def test_nested_parentheses_constraint(self):
        """Test parsing constraints with nested parentheses"""
        parser = ConstraintParser()
        constraint_fn = parser.parse_constraint("(.task==transcribe and (.lang==en or .lang==de)) or .task==translate")

        properties = {"task": "transcribe", "lang": "en"}
        assert constraint_fn(properties) is True

        properties = {"task": "transcribe", "lang": "de"}
        assert constraint_fn(properties) is True

        properties = {"task": "transcribe", "lang": "fr"}
        assert constraint_fn(properties) is False

        properties = {"task": "translate", "lang": "fr"}
        assert constraint_fn(properties) is True

    @pytest.mark.unit
    def test_parentheses_with_not_constraint(self):
        """Test parsing constraints with parentheses and NOT operator"""
        parser = ConstraintParser()
        constraint_fn = parser.parse_constraint("not (.task==transcribe and .lang==en)")

        properties = {"task": "transcribe", "lang": "en"}
        assert constraint_fn(properties) is False

        properties = {"task": "transcribe", "lang": "de"}
        assert constraint_fn(properties) is True

        properties = {"task": "translate", "lang": "en"}
        assert constraint_fn(properties) is True

    @pytest.mark.unit
    def test_complex_parentheses_constraint(self):
        """Test parsing complex constraints with multiple parentheses"""
        parser = ConstraintParser()
        constraint_fn = parser.parse_constraint(
            "(.task==transcribe or .task==translate) and (.source_lang!=.target_lang or .domain==special)"
        )

        properties = {"task": "transcribe", "source_lang": "en", "target_lang": "de", "domain": "general"}
        assert constraint_fn(properties) is True

        properties = {"task": "translate", "source_lang": "en", "target_lang": "en", "domain": "special"}
        assert constraint_fn(properties) is True

        properties = {"task": "transcribe", "source_lang": "en", "target_lang": "en", "domain": "general"}
        assert constraint_fn(properties) is False

        properties = {"task": "other", "source_lang": "en", "target_lang": "de", "domain": "general"}
        assert constraint_fn(properties) is False

    @pytest.mark.unit
    def test_invalid_constraint(self):
        """Test that invalid constraints raise errors"""
        parser = ConstraintParser()
        with pytest.raises(SyntaxError):
            parser.parse_constraint("invalid constraint format")


class TestMultiTaskMetricCutSplitting:
    """Test the cut splitting functionality"""

    def create_mock_cut(self, custom_data):
        """Helper to create mock cuts with custom data"""
        cut = Mock()
        cut.custom = custom_data
        # Handle MixedCut case
        cut.first_non_padding_cut = cut
        return cut

    @pytest.mark.unit
    def test_split_cuts_simple(self):
        """Test basic cut splitting"""
        # Create mock cuts
        cuts = [
            self.create_mock_cut({"task": "transcribe", "lang": "en"}),
            self.create_mock_cut({"task": "translate", "lang": "en"}),
            self.create_mock_cut({"task": "transcribe", "lang": "de"}),
        ]

        # Create MultiTaskMetric instance with mock metrics
        mock_model = Mock()
        mock_model.decoding = Mock()

        cfg = DictConfig(
            {
                "metrics": {
                    "wer": {
                        "constraint": ".task==transcribe",
                        "_target_": "nemo.collections.asr.metrics.WER",
                    },
                    "bleu": {
                        "constraint": ".task==translate",
                        "_target_": "nemo.collections.asr.metrics.BLEU",
                    },
                }
            }
        )

        with patch('nemo.collections.asr.metrics.multitask.MultiTaskMetric.from_config_dict') as mock_from_config:
            mock_wer = Mock()
            mock_bleu = Mock()
            mock_from_config.side_effect = [mock_wer, mock_bleu]

            multitask_metric = MultiTaskMetric(mock_model, cfg)

            cuts_split, idx_split = multitask_metric._split_cuts(cuts)

            # Check WER metric gets transcribe cuts (indices 0, 2)
            assert idx_split["wer"] == [0, 2]

            # Check BLEU metric gets translate cuts (index 1)
            assert idx_split["bleu"] == [1]

    @pytest.mark.unit
    def test_split_cuts_no_matches(self):
        """Test cut splitting with no matching cuts"""
        cuts = [
            self.create_mock_cut({"task": "other", "lang": "en"}),
        ]

        mock_model = Mock()
        mock_model.decoding = Mock()

        cfg = DictConfig(
            {
                "metrics": {
                    "wer": {
                        "constraint": ".task==transcribe",
                        "_target_": "nemo.collections.asr.metrics.WER",
                    }
                }
            }
        )

        with patch('nemo.collections.asr.metrics.multitask.MultiTaskMetric.from_config_dict') as mock_from_config:
            mock_wer = Mock()
            mock_from_config.return_value = mock_wer

            multitask_metric = MultiTaskMetric(mock_model, cfg)

            cuts_split, idx_split = multitask_metric._split_cuts(cuts)

            # No cuts should match
            assert idx_split["wer"] == []

    @pytest.mark.unit
    def test_split_cuts_empty_input(self):
        """Test cut splitting with empty input"""
        cuts = []

        mock_model = Mock()
        mock_model.decoding = Mock()

        cfg = DictConfig(
            {
                "metrics": {
                    "wer": {
                        "constraint": ".task==transcribe",
                        "_target_": "nemo.collections.asr.metrics.WER",
                    }
                }
            }
        )

        with patch('nemo.collections.asr.metrics.multitask.MultiTaskMetric.from_config_dict') as mock_from_config:
            mock_wer = Mock()
            mock_from_config.return_value = mock_wer

            multitask_metric = MultiTaskMetric(mock_model, cfg)

            cuts_split, idx_split = multitask_metric._split_cuts(cuts)

            assert idx_split["wer"] == []


class TestMultiTaskMetricUpdate:
    """Test the metric update functionality"""

    @pytest.fixture
    def mock_multitask_metric(self):
        """Create a MultiTaskMetric with mocked dependencies"""
        mock_model = Mock()
        mock_model.decoding = Mock()

        cfg = DictConfig(
            {
                "metrics": {
                    "wer": {
                        "constraint": ".task==transcribe",
                        "_target_": "nemo.collections.asr.metrics.WER",
                    },
                    "bleu": {
                        "constraint": ".task==translate",
                        "_target_": "nemo.collections.asr.metrics.BLEU",
                    },
                }
            }
        )

        with patch('nemo.collections.asr.metrics.multitask.MultiTaskMetric.from_config_dict') as mock_from_config:
            mock_wer = Mock()
            mock_bleu = Mock()
            mock_from_config.side_effect = [mock_wer, mock_bleu]

            multitask_metric = MultiTaskMetric(mock_model, cfg)

            # Store references for testing
            multitask_metric._mock_wer = mock_wer
            multitask_metric._mock_bleu = mock_bleu

            return multitask_metric

    @pytest.mark.unit
    def test_update_with_matching_cuts(self, mock_multitask_metric):
        """Test update with cuts that match constraints"""
        # Create mock cuts
        cuts = [Mock(), Mock()]
        cuts[0].custom = {"task": "transcribe"}
        cuts[0].first_non_padding_cut = cuts[0]
        cuts[1].custom = {"task": "translate"}
        cuts[1].first_non_padding_cut = cuts[1]

        # Create mock tensors
        batch_size = 2
        predictions = torch.randn(batch_size, 100, 1024)
        predictions_lengths = torch.tensor([80, 90])
        predictions_mask = torch.ones(batch_size, 100, dtype=torch.bool)
        targets = torch.randint(0, 1000, (batch_size, 50))
        targets_lengths = torch.tensor([40, 45])
        input_ids = torch.randint(0, 1000, (batch_size, 60))

        mock_multitask_metric.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            predictions_mask=predictions_mask,
            targets=targets,
            targets_lengths=targets_lengths,
            input_ids=input_ids,
            cuts=cuts,
        )

        # Verify WER metric was called with transcribe cut (index 0)
        mock_multitask_metric._mock_wer.update.assert_called_once()
        wer_call_args = mock_multitask_metric._mock_wer.update.call_args
        assert wer_call_args.kwargs['predictions'].shape[0] == 1  # One transcribe cut

        # Verify BLEU metric was called with translate cut (index 1)
        mock_multitask_metric._mock_bleu.update.assert_called_once()
        bleu_call_args = mock_multitask_metric._mock_bleu.update.call_args
        assert bleu_call_args.kwargs['predictions'].shape[0] == 1  # One translate cut

    @pytest.mark.unit
    def test_update_with_empty_indices(self, mock_multitask_metric):
        """Test update when no cuts match a metric's constraints"""
        # Create cuts that don't match WER constraint
        cuts = [Mock()]
        cuts[0].custom = {"task": "translate"}  # Only matches BLEU
        cuts[0].first_non_padding_cut = cuts[0]

        batch_size = 1
        predictions = torch.randn(batch_size, 100, 1024)
        predictions_lengths = torch.tensor([80])
        predictions_mask = torch.ones(batch_size, 100, dtype=torch.bool)
        targets = torch.randint(0, 1000, (batch_size, 50))
        targets_lengths = torch.tensor([40])
        input_ids = torch.randint(0, 1000, (batch_size, 60))

        mock_multitask_metric.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            predictions_mask=predictions_mask,
            targets=targets,
            targets_lengths=targets_lengths,
            input_ids=input_ids,
            cuts=cuts,
        )

        # WER should be called with empty tensors
        mock_multitask_metric._mock_wer.update.assert_called_once()
        wer_call_args = mock_multitask_metric._mock_wer.update.call_args
        assert wer_call_args.kwargs['predictions'].shape[0] == 0  # Empty

        # BLEU should be called with the one cut
        mock_multitask_metric._mock_bleu.update.assert_called_once()
        bleu_call_args = mock_multitask_metric._mock_bleu.update.call_args
        assert bleu_call_args.kwargs['predictions'].shape[0] == 1


class TestMultiTaskMetricCompute:
    """Test the metric compute functionality"""

    @pytest.mark.unit
    def test_compute_wer_metric(self):
        """Test compute for WER metric"""
        mock_model = Mock()
        mock_model.decoding = Mock()

        cfg = DictConfig(
            {
                "metrics": {
                    "wer": {
                        "constraint": ".task==transcribe",
                        "_target_": "nemo.collections.asr.metrics.WER",
                    }
                }
            }
        )

        with patch('nemo.collections.asr.metrics.multitask.MultiTaskMetric.from_config_dict') as mock_from_config:
            mock_wer = Mock()
            mock_wer.compute.return_value = {
                "val_wer": 0.1,
                "val_wer_num": 10.0,
                "val_wer_denom": 100.0,
            }  # wer, scores, words
            mock_from_config.return_value = mock_wer

            multitask_metric = MultiTaskMetric(mock_model, cfg)

            result = multitask_metric.compute(return_all_metrics=True, prefix="val_")

            expected = {"val_wer": 0.1, "val_wer_num": 10.0, "val_wer_denom": 100.0}
            assert result == expected

    @pytest.mark.unit
    def test_compute_bleu_metric(self):
        """Test compute for BLEU metric"""
        mock_model = Mock()
        mock_model.decoding = Mock()

        cfg = DictConfig(
            {
                "metrics": {
                    "bleu": {
                        "constraint": ".task==translate",
                        "_target_": "nemo.collections.asr.metrics.BLEU",
                    }
                }
            }
        )

        with patch('nemo.collections.asr.metrics.multitask.MultiTaskMetric.from_config_dict') as mock_from_config:
            mock_bleu = Mock()
            mock_bleu.compute.return_value = {"bleu": 25.5, "bleu_num": 1000.0}
            mock_from_config.return_value = mock_bleu

            multitask_metric = MultiTaskMetric(mock_model, cfg)

            result = multitask_metric.compute(prefix="test_")

            assert result == {"bleu": 25.5, "bleu_num": 1000.0}

    @pytest.mark.unit
    def test_reset_metrics(self):
        """Test that reset is called on all metrics"""
        mock_model = Mock()
        mock_model.decoding = Mock()

        cfg = DictConfig(
            {
                "metrics": {
                    "wer": {
                        "constraint": ".task==transcribe",
                        "_target_": "nemo.collections.asr.metrics.WER",
                    },
                    "bleu": {
                        "constraint": ".task==translate",
                        "_target_": "nemo.collections.asr.metrics.BLEU",
                    },
                }
            }
        )

        with patch('nemo.collections.asr.metrics.multitask.MultiTaskMetric.from_config_dict') as mock_from_config:
            mock_wer = Mock()
            mock_bleu = Mock()
            mock_from_config.side_effect = [mock_wer, mock_bleu]

            multitask_metric = MultiTaskMetric(mock_model, cfg)
            multitask_metric.reset()

            mock_wer.reset.assert_called_once()
            mock_bleu.reset.assert_called_once()


class TestMultiTaskMetricEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.unit
    def test_missing_custom_attribute(self):
        """Test handling of cuts without custom attribute"""
        mock_model = Mock()
        mock_model.decoding = Mock()

        cfg = DictConfig(
            {
                "metrics": {
                    "wer": {
                        "constraint": ".task==transcribe",
                        "_target_": "nemo.collections.asr.metrics.WER",
                    }
                }
            }
        )

        with patch('nemo.collections.asr.metrics.multitask.MultiTaskMetric.from_config_dict') as mock_from_config:
            mock_wer = Mock()
            mock_from_config.return_value = mock_wer

            multitask_metric = MultiTaskMetric(mock_model, cfg)

            # Create cut without custom attribute
            cut = Mock()
            cut.custom = {}  # Empty custom dict
            cut.first_non_padding_cut = cut

            cuts_split, idx_split = multitask_metric._split_cuts([cut])

            # Should not match any constraints
            assert idx_split["wer"] == []

    @pytest.mark.unit
    def test_complex_constraint_edge_cases(self):
        """Test complex constraints with edge cases"""
        parser = ConstraintParser()

        # Test constraint with missing properties
        constraint_fn = parser.parse_constraint(".missing_prop==value")
        result = constraint_fn({})
        assert result is False

        # Test constraint with None values
        constraint_fn = parser.parse_constraint(".prop==value")
        result = constraint_fn({"prop": None})
        assert result is False

    @pytest.mark.unit
    def test_operators_coverage(self):
        """Test that all operators are properly defined"""
        parser = ConstraintParser()
        assert "==" in parser.primitives
        assert "!=" in parser.primitives
        assert parser.primitives["=="] == operator.eq
        assert parser.primitives["!="] == operator.ne
