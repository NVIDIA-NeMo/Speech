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
import os
import tempfile

import onnx
import pytest
import torch.cuda
from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo.collections.asr.models import (
    EncDecClassificationModel,
    EncDecCTCModel,
    EncDecRNNTModel,
    EncDecSpeakerLabelModel,
)
from nemo.collections.asr.parts.utils import asr_module_utils
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


class TestExportable:
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_EncDecCTCModel_export_to_onnx(self):
        model_config = DictConfig(
            {
                'preprocessor': DictConfig(self.preprocessor),
                'encoder': DictConfig(self.encoder_dict),
                'decoder': DictConfig(self.decoder_dict),
            }
        )
        model = EncDecCTCModel(cfg=model_config).cuda()
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'qn.onnx')
            model.export(
                output=filename, check_trace=True,
            )
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert onnx_model.graph.input[0].name == 'audio_signal'
            assert onnx_model.graph.output[0].name == 'logprobs'

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_EncDecClassificationModel_export_to_onnx(self, speech_classification_model):
        model = speech_classification_model.cuda()
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'edc.onnx')
            model.export(
                output=filename, check_trace=True,
            )
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert onnx_model.graph.input[0].name == 'audio_signal'
            assert onnx_model.graph.output[0].name == 'logits'

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_EncDecSpeakerLabelModel_export_to_onnx(self, speaker_label_model):
        model = speaker_label_model.cuda()
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'sl.onnx')
            model.export(output=filename)
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert onnx_model.graph.input[0].name == 'audio_signal'
            assert onnx_model.graph.output[0].name == 'logits'

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_EncDecCitrinetModel_export_to_onnx(self, citrinet_model):
        model = citrinet_model.cuda()
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'citri.onnx')
            model.export(output=filename)
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert onnx_model.graph.input[0].name == 'audio_signal'
            assert onnx_model.graph.input[1].name == 'length'
            assert onnx_model.graph.output[0].name == 'logprobs'

    @pytest.mark.pleasefixme
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_ConformerModel_export_to_onnx(self, conformer_model):
        model = conformer_model.cuda()
        with tempfile.TemporaryDirectory() as tmpdir, torch.cuda.amp.autocast():
            filename = os.path.join(tmpdir, 'conf.onnx')
            device = next(model.parameters()).device
            input_example = torch.randn(4, model.encoder._feat_in, 777, device=device)
            input_example_length = torch.full(size=(input_example.shape[0],), fill_value=777, device=device)
            model.export(
                output=filename, input_example=tuple([input_example, input_example_length]), check_trace=True,
            )

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_SqueezeformerModel_export_to_onnx(self, squeezeformer_model):
        model = squeezeformer_model.cuda()
        with tempfile.TemporaryDirectory() as tmpdir, torch.cuda.amp.autocast():
            filename = os.path.join(tmpdir, 'squeeze.ts')
            device = next(model.parameters()).device
            input_example = torch.randn(4, model.encoder._feat_in, 777, device=device)
            input_example_length = torch.full(size=(input_example.shape[0],), fill_value=777, device=device)
            model.export(output=filename, input_example=tuple([input_example, input_example_length]), check_trace=True)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_EncDecCitrinetModel_limited_SE_export_to_onnx(self, citrinet_model):
        model = citrinet_model.cuda()
        asr_module_utils.change_conv_asr_se_context_window(model, context_window=24, update_config=False)

        with tempfile.TemporaryDirectory() as tmpdir, torch.cuda.amp.autocast():
            filename = os.path.join(tmpdir, 'citri_se.onnx')
            model.export(
                output=filename, check_trace=True,
            )
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert onnx_model.graph.input[0].name == 'audio_signal'
            assert onnx_model.graph.input[1].name == 'length'
            assert onnx_model.graph.output[0].name == 'logprobs'

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_EncDecRNNTModel_export_to_onnx(self, citrinet_rnnt_model):
        model = citrinet_rnnt_model.cuda()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = 'citri_rnnt.onnx'
            filename = os.path.join(tmpdir, fn)
            files, descr = model.export(output=filename, verbose=False)

            encoder_filename = os.path.join(tmpdir, 'encoder-' + fn)
            assert files[0] == encoder_filename
            assert os.path.exists(encoder_filename)
            onnx_model = onnx.load(encoder_filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert len(onnx_model.graph.input) == 2
            assert len(onnx_model.graph.output) == 2
            assert onnx_model.graph.input[0].name == 'audio_signal'
            assert onnx_model.graph.input[1].name == 'length'
            assert onnx_model.graph.output[0].name == 'outputs'
            assert onnx_model.graph.output[1].name == 'encoded_lengths'

            decoder_joint_filename = os.path.join(tmpdir, 'decoder_joint-' + fn)
            assert files[1] == decoder_joint_filename
            assert os.path.exists(decoder_joint_filename)
            onnx_model = onnx.load(decoder_joint_filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed

            input_examples = model.decoder.input_example()
            assert type(input_examples[-1]) == tuple
            num_states = len(input_examples[-1])
            state_name = list(model.decoder.output_types.keys())[-1]

            # enc_logits + (all decoder inputs - state tuple) + flattened state list
            assert len(onnx_model.graph.input) == (1 + (len(input_examples) - 1) + num_states)
            assert onnx_model.graph.input[0].name == 'encoder_outputs'
            assert onnx_model.graph.input[1].name == 'targets'
            assert onnx_model.graph.input[2].name == 'target_length'

            if num_states > 0:
                for idx, ip in enumerate(onnx_model.graph.input[3:]):
                    assert ip.name == "input_" + state_name + '_' + str(idx + 1)

            assert len(onnx_model.graph.output) == (len(input_examples) - 1) + num_states
            assert onnx_model.graph.output[0].name == 'outputs'
            assert onnx_model.graph.output[1].name == 'prednet_lengths'

            if num_states > 0:
                for idx, op in enumerate(onnx_model.graph.output[2:]):
                    assert op.name == "output_" + state_name + '_' + str(idx + 1)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_EncDecRNNTModel_export_to_ts(self, citrinet_rnnt_model):
        model = citrinet_rnnt_model.cuda()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = 'citri_rnnt.ts'
            filename = os.path.join(tmpdir, fn)
            # Perform export + test with the input examples of the RNNT model.
            files, descr = model.export(output=filename, verbose=False, check_trace=True)

            encoder_filename = os.path.join(tmpdir, 'encoder-' + fn)
            assert files[0] == encoder_filename
            assert os.path.exists(encoder_filename)

            ts_encoder = torch.jit.load(encoder_filename)
            assert ts_encoder is not None

            arguments = ts_encoder.forward.schema.arguments[1:]  # First value is `self`
            assert arguments[0].name == 'audio_signal'
            assert arguments[1].name == 'length'

            decoder_joint_filename = os.path.join(tmpdir, 'decoder_joint-' + fn)
            assert files[1] == decoder_joint_filename
            assert os.path.exists(decoder_joint_filename)

            ts_decoder_joint = torch.jit.load(decoder_joint_filename)
            assert ts_decoder_joint is not None

            ts_decoder_joint_args = ts_decoder_joint.forward.schema.arguments[1:]  # First value is self

            input_examples = model.decoder.input_example()
            assert type(input_examples[-1]) == tuple
            num_states = len(input_examples[-1])
            state_name = list(model.decoder.output_types.keys())[-1]

            # enc_logits + (all decoder inputs - state tuple) + flattened state list
            assert len(ts_decoder_joint_args) == (1 + (len(input_examples) - 1) + num_states)
            assert ts_decoder_joint_args[0].name == 'encoder_outputs'
            assert ts_decoder_joint_args[1].name == 'targets'
            assert ts_decoder_joint_args[2].name == 'target_length'

            if num_states > 0:
                for idx, ip in enumerate(ts_decoder_joint_args[3:]):
                    assert ip.name == "input_" + state_name + '_' + str(idx + 1)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_EncDecCTCModel_adapted_export_to_onnx(self):
        model_config = DictConfig(
            {
                'preprocessor': DictConfig(self.preprocessor),
                'encoder': DictConfig(self.encoder_dict),
                'decoder': DictConfig(self.decoder_dict),
            }
        )

        # support adapter in encoder
        model_config.encoder.cls = model_config.encoder.cls + 'Adapter'  # ConvASREncoderAdapter

        # load model
        model = EncDecCTCModel(cfg=model_config)

        # add adapter
        adapter_cfg = OmegaConf.structured(
            LinearAdapterConfig(in_features=model_config.encoder.params.jasper[0].filters, dim=32)
        )
        model.add_adapter('temp', cfg=adapter_cfg)

        model = model.cuda()

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'qn.onnx')
            model.export(
                output=filename, check_trace=True,
            )
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert onnx_model.graph.input[0].name == 'audio_signal'
            assert onnx_model.graph.output[0].name == 'logprobs'

    def setup_method(self):
        self.preprocessor = {
            'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
            'params': dict({}),
        }

        self.encoder_dict = {
            'cls': 'nemo.collections.asr.modules.ConvASREncoder',
            'params': {
                'feat_in': 64,
                'activation': 'relu',
                'conv_mask': True,
                'jasper': [
                    {
                        'filters': 1024,
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

        self.decoder_dict = {
            'cls': 'nemo.collections.asr.modules.ConvASRDecoder',
            'params': {
                'feat_in': 1024,
                'num_classes': 28,
                'vocabulary': [
                    ' ',
                    'a',
                    'b',
                    'c',
                    'd',
                    'e',
                    'f',
                    'g',
                    'h',
                    'i',
                    'j',
                    'k',
                    'l',
                    'm',
                    'n',
                    'o',
                    'p',
                    'q',
                    'r',
                    's',
                    't',
                    'u',
                    'v',
                    'w',
                    'x',
                    'y',
                    'z',
                    "'",
                ],
            },
        }


@pytest.fixture()
def speech_classification_model():
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': 32,
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
        'cls': 'nemo.collections.asr.modules.ConvASRDecoderClassification',
        'params': {'feat_in': 32, 'num_classes': 30,},
    }

    modelConfig = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'labels': ListConfig(["dummy_cls_{}".format(i + 1) for i in range(30)]),
        }
    )
    model = EncDecClassificationModel(cfg=modelConfig)
    return model


@pytest.fixture()
def speaker_label_model():
    preprocessor = {
        '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
    }
    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
        'feat_in': 64,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': 512,
                'repeat': 1,
                'kernel': [1],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.0,
                'residual': False,
                'separable': False,
            }
        ],
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.SpeakerDecoder',
        'feat_in': 512,
        'num_classes': 2,
        'pool_mode': 'attention',
        'emb_sizes': [1024],
    }

    modelConfig = DictConfig(
        {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
    )
    speaker_model = EncDecSpeakerLabelModel(cfg=modelConfig)
    return speaker_model


@pytest.fixture()
def citrinet_model():
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': 80,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': 512,
                    'repeat': 1,
                    'kernel': [5],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                },
                {
                    'filters': 512,
                    'repeat': 5,
                    'kernel': [11],
                    'stride': [2],
                    'dilation': [1],
                    'dropout': 0.1,
                    'residual': True,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                    'stride_last': True,
                    'residual_mode': 'stride_add',
                },
                {
                    'filters': 512,
                    'repeat': 5,
                    'kernel': [13],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.1,
                    'residual': True,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                },
                {
                    'filters': 640,
                    'repeat': 1,
                    'kernel': [41],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': True,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                },
            ],
        },
    }

    decoder = {
        'cls': 'nemo.collections.asr.modules.ConvASRDecoder',
        'params': {'feat_in': 640, 'num_classes': 1024, 'vocabulary': list(chr(i % 28) for i in range(0, 1024))},
    }

    modelConfig = DictConfig(
        {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
    )
    citri_model = EncDecCTCModel(cfg=modelConfig)
    return citri_model


@pytest.fixture()
def citrinet_rnnt_model():
    labels = list(chr(i % 28) for i in range(0, 1024))
    model_defaults = {'enc_hidden': 640, 'pred_hidden': 256, 'joint_hidden': 320}

    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
        'feat_in': 80,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': 512,
                'repeat': 1,
                'kernel': [5],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.0,
                'residual': False,
                'separable': True,
                'se': True,
                'se_context_size': -1,
            },
            {
                'filters': 512,
                'repeat': 5,
                'kernel': [11],
                'stride': [2],
                'dilation': [1],
                'dropout': 0.1,
                'residual': True,
                'separable': True,
                'se': True,
                'se_context_size': -1,
                'stride_last': True,
                'residual_mode': 'stride_add',
            },
            {
                'filters': 512,
                'repeat': 5,
                'kernel': [13],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.1,
                'residual': True,
                'separable': True,
                'se': True,
                'se_context_size': -1,
            },
            {
                'filters': 640,
                'repeat': 1,
                'kernel': [41],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.0,
                'residual': True,
                'separable': True,
                'se': True,
                'se_context_size': -1,
            },
        ],
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
        'prednet': {'pred_hidden': 256, 'pred_rnn_layers': 1, 'dropout': 0.0},
    }

    joint = {
        '_target_': 'nemo.collections.asr.modules.RNNTJoint',
        'fuse_loss_wer': False,
        'jointnet': {'joint_hidden': 320, 'activation': 'relu', 'dropout': 0.0},
    }

    decoding = {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 5}}

    modelConfig = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'labels': labels,
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'joint': DictConfig(joint),
            'decoding': DictConfig(decoding),
        }
    )
    citri_model = EncDecRNNTModel(cfg=modelConfig)
    return citri_model


@pytest.fixture()
def conformer_model():
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
    encoder = {
        'cls': 'nemo.collections.asr.modules.ConformerEncoder',
        'params': {
            'feat_in': 80,
            'feat_out': -1,
            'n_layers': 2,
            'd_model': 256,
            'subsampling': 'striding',
            'subsampling_factor': 4,
            'subsampling_conv_channels': 512,
            'reduction': None,
            'reduction_position': None,
            'reduction_factor': 1,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rel_pos',
            'n_heads': 8,
            'att_context_size': [-1, -1],
            'xscaling': True,
            'untie_biases': True,
            'pos_emb_max_len': 500,
            'conv_kernel_size': 31,
            'dropout': 0.1,
            'dropout_pre_encoder': 0.1,
            'dropout_emb': 0.0,
            'dropout_att': 0.1,
        },
    }

    decoder = {
        'cls': 'nemo.collections.asr.modules.ConvASRDecoder',
        'params': {'feat_in': 256, 'num_classes': 1024, 'vocabulary': list(chr(i % 28) for i in range(0, 1024))},
    }

    modelConfig = DictConfig(
        {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
    )
    conformer_model = EncDecCTCModel(cfg=modelConfig)
    return conformer_model


@pytest.fixture()
def squeezeformer_model():
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
    encoder = {
        'cls': 'nemo.collections.asr.modules.SqueezeformerEncoder',
        'params': {
            'feat_in': 80,
            'feat_out': -1,
            'n_layers': 2,
            'adaptive_scale': True,
            'time_reduce_idx': 1,
            'time_recovery_idx': None,
            'd_model': 256,
            'subsampling': 'dw_striding',
            'subsampling_factor': 4,
            'subsampling_conv_channels': 512,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rel_pos',
            'n_heads': 8,
            'att_context_size': [-1, -1],
            'xscaling': True,
            'untie_biases': True,
            'pos_emb_max_len': 500,
            'conv_kernel_size': 31,
            'dropout': 0.1,
            'dropout_emb': 0.0,
            'dropout_att': 0.1,
        },
    }

    decoder = {
        'cls': 'nemo.collections.asr.modules.ConvASRDecoder',
        'params': {'feat_in': 256, 'num_classes': 1024, 'vocabulary': list(chr(i % 28) for i in range(0, 1024))},
    }

    modelConfig = DictConfig(
        {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
    )
    conformer_model = EncDecCTCModel(cfg=modelConfig)
    return conformer_model
