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
import dataclasses
import io
import random
import string
from copy import deepcopy
from typing import List
from unittest.mock import Mock, patch

import pytest
import torch
from torchmetrics.audio.snr import SignalNoiseRatio

from nemo.collections.asr.metrics.audio import AudioMetricWrapper
from nemo.collections.asr.metrics.rnnt_wer import RNNTWER
from nemo.collections.asr.metrics.rnnt_wer_bpe import RNNTBPEWER
from nemo.collections.asr.metrics.wer import (
    WER,
    CTCDecoding,
    CTCDecodingConfig,
    word_error_rate,
    word_error_rate_detail,
    word_error_rate_per_utt,
)
from nemo.collections.asr.metrics.wer_bpe import WERBPE, CTCBPEDecoding, CTCBPEDecodingConfig
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
        wer(predictions=predictions_tensor, targets=targets_tensor, target_lengths=torch.tensor([len(reference)]))
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
        assert word_error_rate_detail(hypotheses=['cat', ''], references=['', 'gpu']) == (2.0, 1, 1.0, 1.0, 0.0,)
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
            wer = WERBPE(decoding, use_cer=False)
        else:
            decoding = CTCDecoding(decoding_config, self.vocabulary.copy())
            wer = WER(decoding, use_cer=False)

        tokens = self.__string_to_ctc_tensor('cat', use_tokenizer=test_wer_bpe)[0].int().numpy().tolist()
        assert tokens == [3, 1, 20]

        tokens_decoded = wer.decoding.decode_ids_to_tokens(tokens)
        assert tokens_decoded == ['c', 'a', 't']

        str_decoded = wer.decoding.decode_tokens_to_str(tokens)
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
        hyp, _ = wer.decoding.ctc_decoder_predictions_tensor(tensor, return_hypotheses=True)
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
        hyp, _ = wer.decoding.ctc_decoder_predictions_tensor(tensor, decoder_lengths=length, return_hypotheses=True)
        hyp = hyp[0]
        assert isinstance(hyp, Hypothesis)
        assert hyp.length == 3

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_dim_index", [0, 1])
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_wer_metric_subword_return_hypothesis(self, batch_dim_index, test_wer_bpe):
        decoding_config = {'strategy': 'greedy', 'batch_dim_index': batch_dim_index}
        wer = WERBPE(CTCBPEDecoding(decoding_config, self.char_tokenizer), use_cer=False)

        tensor = self.__string_to_ctc_tensor('cat', test_wer_bpe, as_logprobs=True).int()
        if batch_dim_index > 0:
            tensor.transpose_(0, 1)

        # pass batchsize 1 tensor, get back list of length 1 Hypothesis
        wer.decoding.preserve_alignments = True
        hyp, _ = wer.decoding.ctc_decoder_predictions_tensor(tensor, return_hypotheses=True)
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
        hyp, _ = wer.decoding.ctc_decoder_predictions_tensor(tensor, decoder_lengths=length, return_hypotheses=True)
        hyp = hyp[0]
        assert isinstance(hyp, Hypothesis)
        assert hyp.length == 3

    def get_wer_ctc(self, prediction: str, reference: str, test_wer_bpe: bool):
        ctc_decoder_predictions_tensor_mock = Mock(return_value=([prediction], None))
        if test_wer_bpe:
            decoding = Mock(
                blank_id=self.char_tokenizer.tokenizer.vocab_size,
                tokenizer=deepcopy(self.char_tokenizer),
                ctc_decoder_predictions_tensor=ctc_decoder_predictions_tensor_mock,
                decode_tokens_to_str=self.char_tokenizer.ids_to_text,
            )
            wer = WERBPE(decoding, use_cer=False)
        else:
            decoding = Mock(
                blank_id=len(self.vocabulary),
                labels_map=self.vocabulary.copy(),
                ctc_decoder_predictions_tensor=ctc_decoder_predictions_tensor_mock,
                decode_tokens_to_str=self.decode_token_to_str_with_vocabulary_mock,
            )
            wer = WER(decoding, use_cer=False)
        targets_tensor = self.__reference_string_to_tensor(reference, test_wer_bpe)

        wer(
            predictions=None,
            predictions_lengths=None,
            targets=targets_tensor,
            target_lengths=torch.tensor([len(reference)]),
        )
        res, _, _ = wer.compute()
        res = res.detach().cpu()
        # return res[0] / res[1]
        return res.item()

    def decode_token_to_str_with_vocabulary_mock(self, ids):
        return ''.join([self.vocabulary[id_] for id_ in ids])

    def get_wer_rnnt(self, prediction: str, reference: str, batch_dim_index: int, test_wer_bpe: bool):
        rnnt_decoder_predictions_tensor_mock = Mock(return_value=([prediction], None))
        if test_wer_bpe:
            decoding = Mock(
                blank_id=self.char_tokenizer.tokenizer.vocab_size,
                tokenizer=deepcopy(self.char_tokenizer),
                rnnt_decoder_predictions_tensor=rnnt_decoder_predictions_tensor_mock,
                decode_tokens_to_str=self.char_tokenizer.ids_to_text,
            )
            wer = RNNTBPEWER(decoding, batch_dim_index=batch_dim_index, use_cer=False)
        else:
            decoding = Mock(
                blank_id=len(self.vocabulary),
                labels_map=self.vocabulary.copy(),
                rnnt_decoder_predictions_tensor=rnnt_decoder_predictions_tensor_mock,
                decode_tokens_to_str=self.decode_token_to_str_with_vocabulary_mock,
            )
            wer = RNNTWER(decoding, batch_dim_index=batch_dim_index, use_cer=False)
        targets_tensor = self.__reference_string_to_tensor(reference, test_wer_bpe)
        if wer.batch_dim_index > 0:
            targets_tensor.transpose_(0, 1)
        wer(
            encoder_output=None,
            encoded_lengths=None,
            targets=targets_tensor,
            target_lengths=torch.tensor([len(reference)]),
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

        hyp, _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestep) == 0
        assert hyp.alignments is None

        # Preserve timestamps and alignments
        decoding_cfg = CTCDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        decoding = CTCDecoding(decoding_cfg, vocabulary=self.vocabulary)

        hyp, _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestep) == 3
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

        hyp, _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestep) == 0
        assert hyp.alignments is None

        # Preserve timestamps and alignments
        decoding_cfg = CTCBPEDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        decoding = CTCBPEDecoding(decoding_cfg, tokenizer=self.char_tokenizer)

        hyp, _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestep) == 3
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

        hyp, _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestep) == 0
        assert hyp.alignments is None

        # Preserve timestamps and alignments
        decoding_cfg = CTCDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        decoding = CTCDecoding(decoding_cfg, vocabulary=self.vocabulary)

        # Cannot compute alignments from labels
        with pytest.raises(ValueError):
            hyp, _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)

        # Preserve timestamps
        decoding_cfg = CTCDecodingConfig(preserve_alignments=False, compute_timestamps=True)
        decoding = CTCDecoding(decoding_cfg, vocabulary=self.vocabulary)

        hyp, _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestep) == 3
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

        hyp, _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestep) == 0
        assert hyp.alignments is None

        # Preserve timestamps and alignments
        decoding_cfg = CTCBPEDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        decoding = CTCBPEDecoding(decoding_cfg, tokenizer=self.char_tokenizer)

        hyp, _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestep) == 3
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

        hyp, _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestep) == 0
        assert hyp.alignments is None

        # Preserve timestamps and alignments
        decoding_cfg = CTCBPEDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        decoding = CTCBPEDecoding(decoding_cfg, tokenizer=self.char_tokenizer)

        # Cannot compute alignments from labels
        with pytest.raises(ValueError):
            hyp, _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)

        # Preserve timestamps
        decoding_cfg = CTCBPEDecodingConfig(preserve_alignments=False, compute_timestamps=True)
        decoding = CTCBPEDecoding(decoding_cfg, tokenizer=self.char_tokenizer)

        hyp, _ = decoding.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lens, return_hypotheses=True)
        hyp = hyp[0]  # type: Hypothesis
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.length == torch.tensor(T, dtype=torch.int32)
        assert hyp.text != ''
        assert len(hyp.timestep) == 3
        assert hyp.alignments is None


class TestAudioMetricWrapper:
    def test_metric_full_batch(self):
        """Test metric on batches where all examples have equal length.
        """
        ref_metric = SignalNoiseRatio()
        wrapped_metric = AudioMetricWrapper(metric=SignalNoiseRatio())

        num_resets = 5
        num_batches = 10
        batch_size = 8
        num_channels = 2
        num_samples = 200

        batch_shape = (batch_size, num_channels, num_samples)

        for nr in range(num_resets):
            for nb in range(num_batches):
                target = torch.rand(*batch_shape)
                preds = target + torch.rand(1) * torch.rand(*batch_shape)

                # test forward for a single batch
                batch_value_wrapped = wrapped_metric(preds=preds, target=target)
                batch_value_ref = ref_metric(preds=preds, target=target)

                assert torch.allclose(
                    batch_value_wrapped, batch_value_ref
                ), f'Metric forward not matching for batch {nb}, reset {nr}'

            # test compute (over num_batches)
            assert torch.allclose(
                wrapped_metric.compute(), ref_metric.compute()
            ), f'Metric compute not matching for batch {nb}, reset {nr}'

            ref_metric.reset()
            wrapped_metric.reset()

    def test_input_length(self):
        """Test metric on batches where examples have different length.
        """
        ref_metric = SignalNoiseRatio()
        wrapped_metric = AudioMetricWrapper(metric=SignalNoiseRatio())

        num_resets = 5
        num_batches = 10
        batch_size = 8
        num_channels = 2
        num_samples = 200

        batch_shape = (batch_size, num_channels, num_samples)

        for nr in range(num_resets):
            for nb in range(num_batches):
                target = torch.rand(*batch_shape)
                preds = target + torch.rand(1) * torch.rand(*batch_shape)

                input_length = torch.randint(low=num_samples // 2, high=num_samples, size=(batch_size,))

                # test forward for a single batch
                batch_value_wrapped = wrapped_metric(preds=preds, target=target, input_length=input_length)

                # compute reference value, assuming batch reduction using averaging
                batch_value_ref = 0
                for b_idx, b_len in enumerate(input_length):
                    batch_value_ref += ref_metric(preds=preds[b_idx, ..., :b_len], target=target[b_idx, ..., :b_len])
                batch_value_ref /= batch_size  # average

                assert torch.allclose(
                    batch_value_wrapped, batch_value_ref
                ), f'Metric forward not matching for batch {nb}, reset {nr}'

            # test compute (over num_batches)
            assert torch.allclose(
                wrapped_metric.compute(), ref_metric.compute()
            ), f'Metric compute not matching for batch {nb}, reset {nr}'

            ref_metric.reset()
            wrapped_metric.reset()

    @pytest.mark.unit
    @pytest.mark.parametrize('channel', [0, 1])
    def test_channel(self, channel):
        """Test metric on a single channel from a batch.
        """
        ref_metric = SignalNoiseRatio()
        # select only a single channel
        wrapped_metric = AudioMetricWrapper(metric=SignalNoiseRatio(), channel=channel)

        num_resets = 5
        num_batches = 10
        batch_size = 8
        num_channels = 2
        num_samples = 200

        batch_shape = (batch_size, num_channels, num_samples)

        for nr in range(num_resets):
            for nb in range(num_batches):
                target = torch.rand(*batch_shape)
                preds = target + torch.rand(1) * torch.rand(*batch_shape)

                # varying length
                input_length = torch.randint(low=num_samples // 2, high=num_samples, size=(batch_size,))

                # test forward for a single batch
                batch_value_wrapped = wrapped_metric(preds=preds, target=target, input_length=input_length)

                # compute reference value, assuming batch reduction using averaging
                batch_value_ref = 0
                for b_idx, b_len in enumerate(input_length):
                    batch_value_ref += ref_metric(
                        preds=preds[b_idx, channel, :b_len], target=target[b_idx, channel, :b_len]
                    )
                batch_value_ref /= batch_size  # average

                assert torch.allclose(
                    batch_value_wrapped, batch_value_ref
                ), f'Metric forward not matching for batch {nb}, reset {nr}'

            # test compute (over num_batches)
            assert torch.allclose(
                wrapped_metric.compute(), ref_metric.compute()
            ), f'Metric compute not matching for batch {nb}, reset {nr}'

            ref_metric.reset()
            wrapped_metric.reset()
