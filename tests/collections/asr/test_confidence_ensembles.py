# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import joblib
import pytest
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.metrics.wer import CTCDecodingConfig
from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel, EncDecRNNTModel
from nemo.collections.asr.models.confidence_ensemble import ConfidenceEnsembleModel
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig, ConfidenceMethodConfig


def get_model_config(model_class):
    preprocessor_config = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
    vocabulary = [' ', "'", 'a', 'b', 'c']  # does not matter, so keeping small
    encoder_config = {
        '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
        'feat_in': 64,
        'n_layers': 8,
        'd_model': 4,
    }
    if model_class is EncDecCTCModel:
        decoder_config = {
            '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
            'feat_in': None,
            'num_classes': len(vocabulary),
            'vocabulary': vocabulary,
        }
        model_config = DictConfig(
            {
                'compute_eval_loss': True,  # will be ignored by the model
                'preprocessor': DictConfig(preprocessor_config),
                'encoder': DictConfig(encoder_config),
                'decoder': DictConfig(decoder_config),
            }
        )
    else:
        decoder_config = {
            '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
            'prednet': {'pred_hidden': 4, 'pred_rnn_layers': 1},
        }
        joint_config = {
            '_target_': 'nemo.collections.asr.modules.RNNTJoint',
            'jointnet': {'joint_hidden': 4, 'activation': 'relu'},
        }
        decoding_config = {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 30}}
        loss_config = {'loss_name': 'default', 'warprnnt_numba_kwargs': {'fastemit_lambda': 0.001}}

        model_config = DictConfig(
            {
                'compute_eval_loss': True,
                'labels': ListConfig(vocabulary),
                'preprocessor': DictConfig(preprocessor_config),
                'model_defaults': DictConfig({'enc_hidden': 4, 'pred_hidden': 4}),
                'encoder': DictConfig(encoder_config),
                'decoder': DictConfig(decoder_config),
                'joint': DictConfig(joint_config),
                'decoding': DictConfig(decoding_config),
                'loss': DictConfig(loss_config),
                'optim': {'name': 'adamw'},
                'aux_ctc': {
                    'ctc_loss_weight': 0.3,
                    'use_cer': False,
                    'ctc_reduction': 'mean_batch',
                    'decoder': {
                        '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                        'feat_in': None,
                        'num_classes': len(vocabulary),
                        'vocabulary': vocabulary,
                    },
                    'decoding': DictConfig(CTCDecodingConfig),
                },
            }
        )
    model_config['target'] = f'{model_class.__module__}.{model_class.__name__}'

    return model_config


class TestConfidenceEnsembles:
    """Only basic tests that are very fast to run.

    There are much more extensive integration tests available in
    scripts/confidence_ensembles/test_confidence_ensembles.py
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_class0", [EncDecCTCModel, EncDecRNNTModel, EncDecHybridRNNTCTCModel],
    )
    @pytest.mark.parametrize(
        "model_class1", [EncDecCTCModel, EncDecRNNTModel, EncDecHybridRNNTCTCModel],
    )
    def test_model_creation_2models(self, tmp_path, model_class0, model_class1):
        """Basic test to check that ensemble of 2 models can be created."""
        model_config0 = get_model_config(model_class0)
        model_config1 = get_model_config(model_class1)

        # dummy pickle file for the model selection block
        joblib.dump({}, tmp_path / 'dummy.pkl')

        # default confidence
        confidence_config = ConfidenceConfig(
            # we keep frame confidences and apply aggregation manually to get full-utterance confidence
            preserve_frame_confidence=True,
            exclude_blank=True,
            aggregation="mean",
            method_cfg=ConfidenceMethodConfig(name="entropy", entropy_type="renyi", alpha=0.25, entropy_norm="lin",),
        )

        # just checking that no errors are raised when creating the model
        ConfidenceEnsembleModel(
            cfg=DictConfig(
                {
                    'model_selection_block': str(tmp_path / 'dummy.pkl'),
                    'confidence': confidence_config,
                    'temperature': 1.0,
                    'num_models': 2,
                    'model0': model_config0,
                    'model1': model_config1,
                }
            ),
            trainer=None,
        )

    def test_model_creation_5models(self, tmp_path):
        """Basic test to check that ensemble of 5 models can be created."""
        model_configs = [get_model_config(EncDecCTCModel) for _ in range(5)]

        # dummy pickle file for the model selection block
        joblib.dump({}, tmp_path / 'dummy.pkl')

        # default confidence
        confidence_config = ConfidenceConfig(
            # we keep frame confidences and apply aggregation manually to get full-utterance confidence
            preserve_frame_confidence=True,
            exclude_blank=True,
            aggregation="mean",
            method_cfg=ConfidenceMethodConfig(name="entropy", entropy_type="renyi", alpha=0.25, entropy_norm="lin",),
        )

        # just checking that no errors are raised when creating the model
        ConfidenceEnsembleModel(
            cfg=DictConfig(
                {
                    'model_selection_block': str(tmp_path / 'dummy.pkl'),
                    'confidence': confidence_config,
                    'temperature': 1.0,
                    'num_models': 2,
                    'model0': model_configs[0],
                    'model1': model_configs[1],
                    'model2': model_configs[2],
                    'model3': model_configs[3],
                    'model4': model_configs[4],
                }
            ),
            trainer=None,
        )
