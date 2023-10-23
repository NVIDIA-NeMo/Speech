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

import pytest
import torch
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.metrics.wer import CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.modules import RNNTDecoder, RNNTJoint, SampledRNNTJoint, StatelessTransducerDecoder
from nemo.collections.asr.parts.submodules import rnnt_beam_decoding as beam_decode
from nemo.collections.asr.parts.submodules import rnnt_greedy_decoding as greedy_decode
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__
from nemo.utils.config_utils import assert_dataclass_signature_match

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cpu_is_supported(
    __NUMBA_MINIMUM_VERSION__
) or numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


@pytest.fixture()
def hybrid_asr_model():
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}

    # fmt: off
    labels = [' ', 'a', 'b', 'c', 'd', 'e', 'f',
              'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
              'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
              'x', 'y', 'z', "'",
              ]
    # fmt: on

    model_defaults = {'enc_hidden': 1024, 'pred_hidden': 64}

    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                }
            ],
        },
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
        'prednet': {'pred_hidden': model_defaults['pred_hidden'], 'pred_rnn_layers': 1},
    }

    joint = {
        '_target_': 'nemo.collections.asr.modules.RNNTJoint',
        'jointnet': {'joint_hidden': 32, 'activation': 'relu'},
    }

    decoding = {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 30}}

    loss = {'loss_name': 'default', 'warprnnt_numba_kwargs': {'fastemit_lambda': 0.001}}

    aux_ctc = {
        'ctc_loss_weight': 0.3,
        'use_cer': False,
        'ctc_reduction': 'mean_batch',
        'decoder': {
            '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
            'feat_in': 1024,
            'num_classes': len(labels),
            'vocabulary': labels,
        },
        'decoding': DictConfig(CTCDecodingConfig),
    }

    modelConfig = DictConfig(
        {
            'labels': ListConfig(labels),
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'joint': DictConfig(joint),
            'decoding': DictConfig(decoding),
            'loss': DictConfig(loss),
            'aux_ctc': DictConfig(aux_ctc),
        }
    )

    model_instance = EncDecHybridRNNTCTCModel(cfg=modelConfig)
    return model_instance


class TestEncDecHybridRNNTCTCModel:
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_constructor(self, hybrid_asr_model):
        hybrid_asr_model.train()
        # TODO: make proper config and assert correct number of weights
        # Check to/from config_dict:
        confdict = hybrid_asr_model.to_config_dict()
        instance2 = EncDecHybridRNNTCTCModel.from_config_dict(confdict)
        assert isinstance(instance2, EncDecHybridRNNTCTCModel)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_forward(self, hybrid_asr_model):
        hybrid_asr_model = hybrid_asr_model.eval()

        hybrid_asr_model.preprocessor.featurizer.dither = 0.0
        hybrid_asr_model.preprocessor.featurizer.pad_to = 0

        hybrid_asr_model.compute_eval_loss = False

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                logprobs_ins, _ = hybrid_asr_model.forward(
                    input_signal=input_signal[i : i + 1], input_signal_length=length[i : i + 1]
                )
                logprobs_instance.append(logprobs_ins)
            logprobs_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            logprobs_batch, _ = hybrid_asr_model.forward(input_signal=input_signal, input_signal_length=length)

        assert logprobs_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_vocab_change(self, hybrid_asr_model):
        old_vocab = copy.deepcopy(hybrid_asr_model.joint.vocabulary)
        nw1 = hybrid_asr_model.num_weights
        hybrid_asr_model.change_vocabulary(new_vocabulary=old_vocab)
        # No change
        assert nw1 == hybrid_asr_model.num_weights
        new_vocab = copy.deepcopy(old_vocab)
        new_vocab.append('!')
        new_vocab.append('$')
        new_vocab.append('@')
        hybrid_asr_model.change_vocabulary(new_vocabulary=new_vocab)
        # fully connected + bias
        # rnn embedding + joint + bias
        pred_embedding = 3 * (hybrid_asr_model.decoder.pred_hidden)
        joint_joint = 3 * (hybrid_asr_model.joint.joint_hidden + 1)
        ctc_decoder = 3 * (hybrid_asr_model.ctc_decoder._feat_in + 1)
        assert hybrid_asr_model.num_weights == (nw1 + (pred_embedding + joint_joint) + ctc_decoder)
        assert hybrid_asr_model.ctc_decoder.vocabulary == hybrid_asr_model.joint.vocabulary

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_decoding_change(self, hybrid_asr_model):
        assert isinstance(hybrid_asr_model.decoding.decoding, greedy_decode.GreedyBatchedRNNTInfer)

        new_strategy = DictConfig({})
        new_strategy.strategy = 'greedy'
        new_strategy.greedy = DictConfig({'max_symbols': 10})
        hybrid_asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(hybrid_asr_model.decoding.decoding, greedy_decode.GreedyRNNTInfer)

        new_strategy = DictConfig({})
        new_strategy.strategy = 'beam'
        new_strategy.beam = DictConfig({'beam_size': 1})
        hybrid_asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(hybrid_asr_model.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert hybrid_asr_model.decoding.decoding.search_type == "default"

        new_strategy = DictConfig({})
        new_strategy.strategy = 'beam'
        new_strategy.beam = DictConfig({'beam_size': 2})
        hybrid_asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(hybrid_asr_model.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert hybrid_asr_model.decoding.decoding.search_type == "default"

        new_strategy = DictConfig({})
        new_strategy.strategy = 'tsd'
        new_strategy.beam = DictConfig({'beam_size': 2})
        hybrid_asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(hybrid_asr_model.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert hybrid_asr_model.decoding.decoding.search_type == "tsd"

        new_strategy = DictConfig({})
        new_strategy.strategy = 'alsd'
        new_strategy.beam = DictConfig({'beam_size': 2})
        hybrid_asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(hybrid_asr_model.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert hybrid_asr_model.decoding.decoding.search_type == "alsd"

        assert hybrid_asr_model.ctc_decoding is not None
        assert isinstance(hybrid_asr_model.ctc_decoding, CTCDecoding)
        assert hybrid_asr_model.ctc_decoding.cfg.strategy == "greedy"
        assert hybrid_asr_model.ctc_decoding.preserve_alignments is False
        assert hybrid_asr_model.ctc_decoding.compute_timestamps is False

        cfg = CTCDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        hybrid_asr_model.change_decoding_strategy(cfg, decoder_type="ctc")

        assert hybrid_asr_model.ctc_decoding.preserve_alignments is True
        assert hybrid_asr_model.ctc_decoding.compute_timestamps is True

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_decoding_type_change(self, hybrid_asr_model):
        assert isinstance(hybrid_asr_model.decoding.decoding, greedy_decode.GreedyBatchedRNNTInfer)

        new_strategy = DictConfig({})
        new_strategy.strategy = 'greedy'
        new_strategy.greedy = DictConfig({'max_symbols': 10})
        hybrid_asr_model.change_decoding_strategy(decoding_cfg=new_strategy, decoder_type='rnnt')
        assert isinstance(hybrid_asr_model.decoding.decoding, greedy_decode.GreedyRNNTInfer)
        assert hybrid_asr_model.cur_decoder == 'rnnt'

        hybrid_asr_model.change_decoding_strategy(decoding_cfg=new_strategy, decoder_type='ctc')
        assert isinstance(hybrid_asr_model.ctc_decoding, CTCDecoding)
        assert hybrid_asr_model.cur_decoder == 'ctc'

        hybrid_asr_model.change_decoding_strategy(decoding_cfg=new_strategy, decoder_type='rnnt')
        assert isinstance(hybrid_asr_model.decoding.decoding, greedy_decode.GreedyRNNTInfer)
        assert hybrid_asr_model.cur_decoder == 'rnnt'

    @pytest.mark.unit
    def test_GreedyRNNTInferConfig(self):
        IGNORE_ARGS = ['decoder_model', 'joint_model', 'blank_index']

        result = assert_dataclass_signature_match(
            greedy_decode.GreedyRNNTInfer, greedy_decode.GreedyRNNTInferConfig, ignore_args=IGNORE_ARGS
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_GreedyBatchedRNNTInferConfig(self):
        IGNORE_ARGS = ['decoder_model', 'joint_model', 'blank_index']

        result = assert_dataclass_signature_match(
            greedy_decode.GreedyBatchedRNNTInfer, greedy_decode.GreedyBatchedRNNTInferConfig, ignore_args=IGNORE_ARGS
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_BeamRNNTInferConfig(self):
        IGNORE_ARGS = ['decoder_model', 'joint_model', 'blank_index']

        result = assert_dataclass_signature_match(
            beam_decode.BeamRNNTInfer, beam_decode.BeamRNNTInferConfig, ignore_args=IGNORE_ARGS
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "greedy_class", [greedy_decode.GreedyRNNTInfer, greedy_decode.GreedyBatchedRNNTInfer],
    )
    def test_greedy_decoding(self, greedy_class):
        token_list = [" ", "a", "b", "c"]
        vocab_size = len(token_list)

        encoder_output_size = 4
        decoder_output_size = 4
        joint_output_shape = 4

        prednet_cfg = {'pred_hidden': decoder_output_size, 'pred_rnn_layers': 1}
        jointnet_cfg = {
            'encoder_hidden': encoder_output_size,
            'pred_hidden': decoder_output_size,
            'joint_hidden': joint_output_shape,
            'activation': 'relu',
        }

        decoder = RNNTDecoder(prednet_cfg, vocab_size)
        joint_net = RNNTJoint(jointnet_cfg, vocab_size, vocabulary=token_list)

        greedy = greedy_class(decoder, joint_net, blank_index=len(token_list) - 1, max_symbols_per_step=5)

        # (B, D, T)
        enc_out = torch.randn(1, encoder_output_size, 30)
        enc_len = torch.tensor([30], dtype=torch.int32)

        with torch.no_grad():
            _ = greedy(encoder_output=enc_out, encoded_lengths=enc_len)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "greedy_class", [greedy_decode.GreedyRNNTInfer],
    )
    def test_greedy_multi_decoding(self, greedy_class):
        token_list = [" ", "a", "b", "c"]
        vocab_size = len(token_list)

        encoder_output_size = 4
        decoder_output_size = 4
        joint_output_shape = 4

        prednet_cfg = {'pred_hidden': decoder_output_size, 'pred_rnn_layers': 1}
        jointnet_cfg = {
            'encoder_hidden': encoder_output_size,
            'pred_hidden': decoder_output_size,
            'joint_hidden': joint_output_shape,
            'activation': 'relu',
        }

        decoder = RNNTDecoder(prednet_cfg, vocab_size)
        joint_net = RNNTJoint(jointnet_cfg, vocab_size, vocabulary=token_list)

        greedy = greedy_class(decoder, joint_net, blank_index=len(token_list) - 1, max_symbols_per_step=5)

        # (B, D, T)
        enc_out = torch.randn(1, encoder_output_size, 30)
        enc_len = torch.tensor([30], dtype=torch.int32)

        with torch.no_grad():
            (partial_hyp) = greedy(encoder_output=enc_out, encoded_lengths=enc_len)
            partial_hyp = partial_hyp[0]
            _ = greedy(encoder_output=enc_out, encoded_lengths=enc_len, partial_hypotheses=partial_hyp)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "greedy_class", [greedy_decode.GreedyRNNTInfer, greedy_decode.GreedyBatchedRNNTInfer],
    )
    def test_greedy_decoding_stateless_decoder(self, greedy_class):
        token_list = [" ", "a", "b", "c"]
        vocab_size = len(token_list)

        encoder_output_size = 4
        decoder_output_size = 4
        joint_output_shape = 4

        prednet_cfg = {'pred_hidden': decoder_output_size, 'pred_rnn_layers': 1}
        jointnet_cfg = {
            'encoder_hidden': encoder_output_size,
            'pred_hidden': decoder_output_size,
            'joint_hidden': joint_output_shape,
            'activation': 'relu',
        }

        decoder = StatelessTransducerDecoder(prednet_cfg, vocab_size)
        joint_net = RNNTJoint(jointnet_cfg, vocab_size, vocabulary=token_list)

        greedy = greedy_class(decoder, joint_net, blank_index=len(token_list) - 1, max_symbols_per_step=5)

        # (B, D, T)
        enc_out = torch.randn(1, encoder_output_size, 30)
        enc_len = torch.tensor([30], dtype=torch.int32)

        with torch.no_grad():
            _ = greedy(encoder_output=enc_out, encoded_lengths=enc_len)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "greedy_class", [greedy_decode.GreedyRNNTInfer],
    )
    def test_greedy_multi_decoding_stateless_decoder(self, greedy_class):
        token_list = [" ", "a", "b", "c"]
        vocab_size = len(token_list)

        encoder_output_size = 4
        decoder_output_size = 4
        joint_output_shape = 4

        prednet_cfg = {'pred_hidden': decoder_output_size, 'pred_rnn_layers': 1}
        jointnet_cfg = {
            'encoder_hidden': encoder_output_size,
            'pred_hidden': decoder_output_size,
            'joint_hidden': joint_output_shape,
            'activation': 'relu',
        }

        decoder = StatelessTransducerDecoder(prednet_cfg, vocab_size)
        joint_net = RNNTJoint(jointnet_cfg, vocab_size, vocabulary=token_list)

        greedy = greedy_class(decoder, joint_net, blank_index=len(token_list) - 1, max_symbols_per_step=5)

        # (B, D, T)
        enc_out = torch.randn(1, encoder_output_size, 30)
        enc_len = torch.tensor([30], dtype=torch.int32)

        with torch.no_grad():
            (partial_hyp) = greedy(encoder_output=enc_out, encoded_lengths=enc_len)
            partial_hyp = partial_hyp[0]
            _ = greedy(encoder_output=enc_out, encoded_lengths=enc_len, partial_hypotheses=partial_hyp)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "greedy_class", [greedy_decode.GreedyRNNTInfer, greedy_decode.GreedyBatchedRNNTInfer],
    )
    def test_greedy_decoding_preserve_alignment(self, greedy_class):
        token_list = [" ", "a", "b", "c"]
        vocab_size = len(token_list)

        encoder_output_size = 4
        decoder_output_size = 4
        joint_output_shape = 4

        prednet_cfg = {'pred_hidden': decoder_output_size, 'pred_rnn_layers': 1}
        jointnet_cfg = {
            'encoder_hidden': encoder_output_size,
            'pred_hidden': decoder_output_size,
            'joint_hidden': joint_output_shape,
            'activation': 'relu',
        }

        decoder = RNNTDecoder(prednet_cfg, vocab_size)
        joint_net = RNNTJoint(jointnet_cfg, vocab_size, vocabulary=token_list)

        greedy = greedy_class(
            decoder, joint_net, blank_index=len(token_list) - 1, preserve_alignments=True, max_symbols_per_step=5
        )

        # (B, D, T)
        enc_out = torch.randn(1, encoder_output_size, 30)
        enc_len = torch.tensor([30], dtype=torch.int32)

        with torch.no_grad():
            hyp = greedy(encoder_output=enc_out, encoded_lengths=enc_len)[0][0]  # type: rnnt_utils.Hypothesis
            assert hyp.alignments is not None

            for t in range(len(hyp.alignments)):
                for u in range(len(hyp.alignments[t])):
                    logp, label = hyp.alignments[t][u]
                    assert torch.is_tensor(logp)
                    assert torch.is_tensor(label)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "greedy"},
            {"search_type": "default", "score_norm": False, "return_best_hypothesis": False},
            {"search_type": "alsd", "alsd_max_target_len": 20, "return_best_hypothesis": False},
            {"search_type": "tsd", "tsd_max_sym_exp_per_step": 3, "return_best_hypothesis": False},
            {"search_type": "maes", "maes_num_steps": 2, "maes_expansion_beta": 2, "return_best_hypothesis": False},
            {"search_type": "maes", "maes_num_steps": 3, "maes_expansion_beta": 1, "return_best_hypothesis": False},
        ],
    )
    def test_beam_decoding(self, beam_config):
        token_list = [" ", "a", "b", "c"]
        vocab_size = len(token_list)
        beam_size = 1 if beam_config["search_type"] == "greedy" else 2

        encoder_output_size = 4
        decoder_output_size = 4
        joint_output_shape = 4

        prednet_cfg = {'pred_hidden': decoder_output_size, 'pred_rnn_layers': 1}
        jointnet_cfg = {
            'encoder_hidden': encoder_output_size,
            'pred_hidden': decoder_output_size,
            'joint_hidden': joint_output_shape,
            'activation': 'relu',
        }

        decoder = RNNTDecoder(prednet_cfg, vocab_size)
        joint_net = RNNTJoint(jointnet_cfg, vocab_size, vocabulary=token_list)

        beam = beam_decode.BeamRNNTInfer(decoder, joint_net, beam_size=beam_size, **beam_config,)

        # (B, D, T)
        enc_out = torch.randn(1, encoder_output_size, 30)
        enc_len = torch.tensor([30], dtype=torch.int32)

        with torch.no_grad():
            _ = beam(encoder_output=enc_out, encoded_lengths=enc_len)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [{"search_type": "greedy"}, {"search_type": "default", "score_norm": False, "return_best_hypothesis": False},],
    )
    def test_beam_decoding_preserve_alignments(self, beam_config):
        token_list = [" ", "a", "b", "c"]
        vocab_size = len(token_list)
        beam_size = 1 if beam_config["search_type"] == "greedy" else 2

        encoder_output_size = 4
        decoder_output_size = 4
        joint_output_shape = 4

        prednet_cfg = {'pred_hidden': decoder_output_size, 'pred_rnn_layers': 1}
        jointnet_cfg = {
            'encoder_hidden': encoder_output_size,
            'pred_hidden': decoder_output_size,
            'joint_hidden': joint_output_shape,
            'activation': 'relu',
        }

        decoder = RNNTDecoder(prednet_cfg, vocab_size)
        joint_net = RNNTJoint(jointnet_cfg, vocab_size, vocabulary=token_list)

        beam = beam_decode.BeamRNNTInfer(
            decoder, joint_net, beam_size=beam_size, **beam_config, preserve_alignments=True
        )

        # (B, D, T)
        enc_out = torch.randn(1, encoder_output_size, 30)
        enc_len = torch.tensor([30], dtype=torch.int32)

        with torch.no_grad():
            hyp = beam(encoder_output=enc_out, encoded_lengths=enc_len)[0][0]  # type: rnnt_utils.Hypothesis

            if isinstance(hyp, rnnt_utils.NBestHypotheses):
                hyp = hyp.n_best_hypotheses[0]  # select top hypothesis only

            assert hyp.alignments is not None

            for t in range(len(hyp.alignments)):
                for u in range(len(hyp.alignments[t])):
                    logp, label = hyp.alignments[t][u]
                    assert torch.is_tensor(logp)
                    assert torch.is_tensor(label)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "greedy_class", [greedy_decode.GreedyRNNTInfer, greedy_decode.GreedyBatchedRNNTInfer],
    )
    def test_greedy_decoding_SampledRNNTJoint(self, greedy_class):
        token_list = [" ", "a", "b", "c"]
        vocab_size = len(token_list)

        encoder_output_size = 4
        decoder_output_size = 4
        joint_output_shape = 4

        prednet_cfg = {'pred_hidden': decoder_output_size, 'pred_rnn_layers': 1}
        jointnet_cfg = {
            'encoder_hidden': encoder_output_size,
            'pred_hidden': decoder_output_size,
            'joint_hidden': joint_output_shape,
            'activation': 'relu',
        }

        decoder = RNNTDecoder(prednet_cfg, vocab_size)
        joint_net = SampledRNNTJoint(jointnet_cfg, vocab_size, n_samples=2, vocabulary=token_list)

        greedy = greedy_class(decoder, joint_net, blank_index=len(token_list) - 1, max_symbols_per_step=5)

        # (B, D, T)
        enc_out = torch.randn(1, encoder_output_size, 30)
        enc_len = torch.tensor([30], dtype=torch.int32)

        with torch.no_grad():
            _ = greedy(encoder_output=enc_out, encoded_lengths=enc_len)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "greedy"},
            {"search_type": "default", "score_norm": False, "return_best_hypothesis": False},
            {"search_type": "alsd", "alsd_max_target_len": 20, "return_best_hypothesis": False},
            {"search_type": "tsd", "tsd_max_sym_exp_per_step": 3, "return_best_hypothesis": False},
            {"search_type": "maes", "maes_num_steps": 2, "maes_expansion_beta": 2, "return_best_hypothesis": False},
            {"search_type": "maes", "maes_num_steps": 3, "maes_expansion_beta": 1, "return_best_hypothesis": False},
        ],
    )
    def test_beam_decoding_SampledRNNTJoint(self, beam_config):
        token_list = [" ", "a", "b", "c"]
        vocab_size = len(token_list)
        beam_size = 1 if beam_config["search_type"] == "greedy" else 2

        encoder_output_size = 4
        decoder_output_size = 4
        joint_output_shape = 4

        prednet_cfg = {'pred_hidden': decoder_output_size, 'pred_rnn_layers': 1}
        jointnet_cfg = {
            'encoder_hidden': encoder_output_size,
            'pred_hidden': decoder_output_size,
            'joint_hidden': joint_output_shape,
            'activation': 'relu',
        }

        decoder = RNNTDecoder(prednet_cfg, vocab_size)
        joint_net = SampledRNNTJoint(jointnet_cfg, vocab_size, n_samples=2, vocabulary=token_list)

        beam = beam_decode.BeamRNNTInfer(decoder, joint_net, beam_size=beam_size, **beam_config,)

        # (B, D, T)
        enc_out = torch.randn(1, encoder_output_size, 30)
        enc_len = torch.tensor([30], dtype=torch.int32)

        with torch.no_grad():
            _ = beam(encoder_output=enc_out, encoded_lengths=enc_len)
