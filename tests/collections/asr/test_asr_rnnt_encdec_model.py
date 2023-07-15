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
import copy

import pytest
import torch
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models import EncDecRNNTModel
from nemo.collections.asr.modules import HATJoint, RNNTDecoder, RNNTJoint, SampledRNNTJoint, StatelessTransducerDecoder
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
def asr_model():
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
        }
    )

    model_instance = EncDecRNNTModel(cfg=modelConfig)
    return model_instance


class TestEncDecRNNTModel:
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_constructor(self, asr_model):
        asr_model.train()
        # TODO: make proper config and assert correct number of weights
        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        instance2 = EncDecRNNTModel.from_config_dict(confdict)
        assert isinstance(instance2, EncDecRNNTModel)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_forward(self, asr_model):
        asr_model = asr_model.eval()

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        asr_model.compute_eval_loss = False

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                logprobs_ins, _ = asr_model.forward(
                    input_signal=input_signal[i : i + 1], input_signal_length=length[i : i + 1]
                )
                logprobs_instance.append(logprobs_ins)
            logprobs_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            logprobs_batch, _ = asr_model.forward(input_signal=input_signal, input_signal_length=length)

        assert logprobs_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_vocab_change(self, asr_model):
        old_vocab = copy.deepcopy(asr_model.joint.vocabulary)
        nw1 = asr_model.num_weights
        asr_model.change_vocabulary(new_vocabulary=old_vocab)
        # No change
        assert nw1 == asr_model.num_weights
        new_vocab = copy.deepcopy(old_vocab)
        new_vocab.append('!')
        new_vocab.append('$')
        new_vocab.append('@')
        asr_model.change_vocabulary(new_vocabulary=new_vocab)
        # fully connected + bias
        # rnn embedding + joint + bias
        pred_embedding = 3 * (asr_model.decoder.pred_hidden)
        joint_joint = 3 * (asr_model.joint.joint_hidden + 1)
        assert asr_model.num_weights == (nw1 + (pred_embedding + joint_joint))

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_change_conv_asr_se_context_window(self, asr_model):
        old_cfg = copy.deepcopy(asr_model.cfg)
        asr_model.change_conv_asr_se_context_window(context_window=32)  # 32 * 0.01s context
        new_config = asr_model.cfg

        assert old_cfg.encoder.jasper[0].se_context_size == -1
        assert new_config.encoder.jasper[0].se_context_size == 32

        for name, m in asr_model.encoder.named_modules():
            if type(m).__class__.__name__ == 'SqueezeExcite':
                assert m.context_window == 32

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_change_conv_asr_se_context_window_no_config_update(self, asr_model):
        old_cfg = copy.deepcopy(asr_model.cfg)
        asr_model.change_conv_asr_se_context_window(context_window=32, update_config=False)  # 32 * 0.01s context
        new_config = asr_model.cfg

        assert old_cfg.encoder.jasper[0].se_context_size == -1
        assert new_config.encoder.jasper[0].se_context_size == -1  # no change

        for name, m in asr_model.encoder.named_modules():
            if type(m).__class__.__name__ == 'SqueezeExcite':
                assert m.context_window == 32

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_decoding_change(self, asr_model):
        assert isinstance(asr_model.decoding.decoding, greedy_decode.GreedyBatchedRNNTInfer)

        new_strategy = DictConfig({})
        new_strategy.strategy = 'greedy'
        new_strategy.greedy = DictConfig({'max_symbols': 10})
        asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(asr_model.decoding.decoding, greedy_decode.GreedyRNNTInfer)

        new_strategy = DictConfig({})
        new_strategy.strategy = 'beam'
        new_strategy.beam = DictConfig({'beam_size': 1})
        asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(asr_model.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert asr_model.decoding.decoding.search_type == "default"

        new_strategy = DictConfig({})
        new_strategy.strategy = 'beam'
        new_strategy.beam = DictConfig({'beam_size': 2})
        asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(asr_model.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert asr_model.decoding.decoding.search_type == "default"

        new_strategy = DictConfig({})
        new_strategy.strategy = 'tsd'
        new_strategy.beam = DictConfig({'beam_size': 2})
        asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(asr_model.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert asr_model.decoding.decoding.search_type == "tsd"

        new_strategy = DictConfig({})
        new_strategy.strategy = 'alsd'
        new_strategy.beam = DictConfig({'beam_size': 2})
        asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(asr_model.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert asr_model.decoding.decoding.search_type == "alsd"

    @pytest.mark.unit
    def test_GreedyRNNTInferConfig(self):
        # confidence_method_cfg is deprecated
        IGNORE_ARGS = ['decoder_model', 'joint_model', 'blank_index', 'confidence_method_cfg']

        result = assert_dataclass_signature_match(
            greedy_decode.GreedyRNNTInfer, greedy_decode.GreedyRNNTInferConfig, ignore_args=IGNORE_ARGS
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_GreedyBatchedRNNTInferConfig(self):
        # confidence_method_cfg is deprecated
        IGNORE_ARGS = ['decoder_model', 'joint_model', 'blank_index', 'confidence_method_cfg']

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
        for joint_type in [RNNTJoint, HATJoint]:
            joint_net = joint_type(jointnet_cfg, vocab_size, vocabulary=token_list)

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
        "greedy_class", [greedy_decode.GreedyMultiblankRNNTInfer, greedy_decode.GreedyBatchedMultiblankRNNTInfer],
    )
    def test_multiblank_rnnt_greedy_decoding(self, greedy_class):
        token_list = [" ", "a", "b", "c"]
        vocab_size = len(token_list)
        big_blank_durations = [2, 4]

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
        for joint_type in [RNNTJoint, HATJoint]:
            joint_net = joint_type(
                jointnet_cfg, vocab_size, vocabulary=token_list, num_extra_outputs=len(big_blank_durations)
            )

            greedy = greedy_class(
                decoder,
                joint_net,
                blank_index=len(token_list),
                big_blank_durations=big_blank_durations,
                max_symbols_per_step=5,
            )

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
        "greedy_class", [greedy_decode.GreedyMultiblankRNNTInfer, greedy_decode.GreedyBatchedMultiblankRNNTInfer],
    )
    def test_multiblank_rnnt_greedy_decoding(self, greedy_class):
        token_list = [" ", "a", "b", "c"]
        vocab_size = len(token_list)
        big_blank_durations = [2, 4]

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
        joint_net = RNNTJoint(
            jointnet_cfg, vocab_size, vocabulary=token_list, num_extra_outputs=len(big_blank_durations)
        )

        greedy = greedy_class(
            decoder,
            joint_net,
            blank_index=len(token_list),
            big_blank_durations=big_blank_durations,
            max_symbols_per_step=5,
        )

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
        "greedy_class", [greedy_decode.GreedyMultiblankRNNTInfer, greedy_decode.GreedyBatchedMultiblankRNNTInfer],
    )
    def test_multiblank_rnnt_greedy_decoding(self, greedy_class):
        token_list = [" ", "a", "b", "c"]
        vocab_size = len(token_list)
        big_blank_durations = [2, 4]

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
        for joint_type in [RNNTJoint, HATJoint]:
            joint_net = joint_type(
                jointnet_cfg, vocab_size, vocabulary=token_list, num_extra_outputs=len(big_blank_durations)
            )

            greedy = greedy_class(
                decoder,
                joint_net,
                blank_index=len(token_list),
                big_blank_durations=big_blank_durations,
                max_symbols_per_step=5,
            )

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
        for joint_type in [RNNTJoint, HATJoint]:
            joint_net = joint_type(jointnet_cfg, vocab_size, vocabulary=token_list)

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
        for joint_type in [RNNTJoint, HATJoint]:
            joint_net = joint_type(jointnet_cfg, vocab_size, vocabulary=token_list)

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
        for joint_type in [RNNTJoint, HATJoint]:
            joint_net = joint_type(jointnet_cfg, vocab_size, vocabulary=token_list)

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

        for joint_type in [RNNTJoint, HATJoint]:
            joint_net = joint_type(jointnet_cfg, vocab_size, vocabulary=token_list)

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

        for joint_type in [RNNTJoint, HATJoint]:
            joint_net = joint_type(jointnet_cfg, vocab_size, vocabulary=token_list)

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
        for joint_type in [RNNTJoint, HATJoint]:
            joint_net = joint_type(jointnet_cfg, vocab_size, vocabulary=token_list)

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
