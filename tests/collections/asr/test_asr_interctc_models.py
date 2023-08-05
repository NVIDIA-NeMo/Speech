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
from typing import Dict

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.metrics.wer import CTCDecodingConfig
from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel
from nemo.core.classes.mixins import AccessMixin


def jasper_encoder_config(num_layers=1) -> Dict:
    return {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
        'feat_in': 64,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': 4,
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
        ]
        * num_layers,
    }


def conformer_encoder_config() -> Dict:
    return {
        '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
        'feat_in': 64,
        'n_layers': 8,
        'd_model': 4,
    }


def squeezeformer_encoder_config() -> Dict:
    return {
        '_target_': 'nemo.collections.asr.modules.SqueezeformerEncoder',
        'feat_in': 64,
        'n_layers': 8,
        'd_model': 4,
    }


class TestInterCTCLoss:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_class", [EncDecCTCModel, EncDecHybridRNNTCTCModel],
    )
    @pytest.mark.parametrize(
        "encoder_config",
        [jasper_encoder_config(num_layers=8), conformer_encoder_config(), squeezeformer_encoder_config()],
    )
    @pytest.mark.parametrize(
        "apply_at_layers,loss_weights",
        [
            ([2, 4], [0.1, 0.3]),
            ([4], [0.3]),
            ([], []),
            # errors
            ([2, 4], [0.1]),
            ([2], [0.1, 0.3]),
            ([], [0.3]),
        ],
    )
    def test_forward(self, model_class, encoder_config, apply_at_layers, loss_weights):
        preprocessor_config = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
        vocabulary = [
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
        ]
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

            aux_ctc_config = {
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
            }
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
                    'aux_ctc': DictConfig(aux_ctc_config),
                }
            )
            # to avoid adding additional tests, we will always disable eval loss
            # when encoder is Squeezeformer - there is nothing specific to
            # particular encoder here, just picking a random one to test disabled
            # loss use-case.
            if encoder_config['_target_'] == 'nemo.collections.asr.modules.SqueezeformerEncoder':
                model_config['compute_eval_loss'] = False

        model_config.update(
            {
                'interctc': {'loss_weights': loss_weights, 'apply_at_layers': apply_at_layers},
                'optim': {'name': 'adamw'},
            }
        )

        class DummyDataset(torch.utils.data.Dataset):
            """Simply returns a single set of values."""

            def __init__(self, values):
                self.values = values

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return self.values

        # this sometimes results in all zeros in the output which breaks tests
        # so using this only for the ptl calls in the bottom, but using
        # processed signal directly initially to remove the chance of
        # this edge-case
        input_signal = torch.randn(size=(1, 512))
        input_length = torch.randint(low=161, high=500, size=[1])
        target = torch.randint(size=(1, input_length[0]), low=0, high=28)
        target_length = torch.tensor([input_length[0]])

        processed_signal = torch.randn(size=([1, 64, 12]))
        processed_length = torch.tensor([8])

        if len(apply_at_layers) != len(loss_weights):
            # has to throw an error here
            with pytest.raises(
                ValueError, match="Length of interctc.apply_at_layers has to match interctc.loss_weights"
            ):
                asr_model = model_class(cfg=model_config)
                asr_model.train()
                logprobs, _, _ = asr_model.forward(input_signal=input_signal, input_signal_length=input_length)
        else:
            asr_model = model_class(cfg=model_config)
            asr_model.train()
            AccessMixin.set_access_enabled(access_enabled=True)
            logprobs, *_ = asr_model.forward(
                processed_signal=processed_signal, processed_signal_length=processed_length
            )
            captured_tensors = asr_model.get_captured_interctc_tensors()
            AccessMixin.reset_registry(asr_model)
            assert len(captured_tensors) == len(apply_at_layers)
            for output in captured_tensors:
                # checking that values are not the same, if shape is the same
                assert output[0].shape != logprobs.shape or not torch.allclose(output[0], logprobs)
                # hybrid model returns output of encoder, so it's not expected to match
                if model_class is EncDecCTCModel:
                    assert output[0].shape == logprobs.shape

            ## Explicitly pass acclerator as cpu, since deafult val in PTL >= 2.0 is auto and it picks cuda
            ## which further causes an error in all reduce at: https://github.com/NVIDIA/NeMo/blob/v1.18.1/nemo/collections/asr/modules/conv_asr.py#L209
            ## and in https://github.com/NVIDIA/NeMo/blob/v1.18.1/nemo/collections/asr/modules/squeezeformer_encoder.py#L392 where device is CPU
            trainer = pl.Trainer(max_epochs=1, accelerator='cpu')
            trainer.fit(
                asr_model,
                train_dataloaders=torch.utils.data.DataLoader(
                    DummyDataset([input_signal, input_length, target, target_length]), collate_fn=lambda x: x[0],
                ),
                val_dataloaders=torch.utils.data.DataLoader(
                    DummyDataset([input_signal, input_length, target, target_length]), collate_fn=lambda x: x[0],
                ),
            )
            required_metrics = ['final_loss'] if len(loss_weights) > 0 else []
            required_metrics += [f'inter_ctc_loss_l{idx}' for idx in apply_at_layers]
            prefix = "val_"
            required_metrics += [f'{prefix}{metric}' for metric in required_metrics]
            required_metrics += [f'{prefix}wer'] + [f'{prefix}inter_wer_l{idx}' for idx in apply_at_layers]
            for metric in required_metrics:
                if 'loss' in metric and 'val_' in metric:
                    if model_config['compute_eval_loss']:
                        assert metric in trainer.logged_metrics
                    else:
                        assert metric not in trainer.logged_metrics
                else:
                    assert metric in trainer.logged_metrics

            trainer.test(
                asr_model,
                dataloaders=torch.utils.data.DataLoader(
                    DummyDataset([input_signal, input_length, target, target_length]), collate_fn=lambda x: x[0],
                ),
            )
            required_metrics = [f'inter_ctc_loss_l{idx}' for idx in apply_at_layers]
            prefix = 'test_'
            # note that "=" is on purpose here, not "+=", since we only log test metrics
            required_metrics = [f'{prefix}{metric}' for metric in required_metrics]
            required_metrics += [f'{prefix}wer'] + [f'{prefix}inter_wer_l{idx}' for idx in apply_at_layers]
            for metric in required_metrics:
                if 'loss' in metric:
                    if model_config['compute_eval_loss']:
                        assert metric in trainer.logged_metrics
                    else:
                        assert metric not in trainer.logged_metrics
                else:
                    assert metric in trainer.logged_metrics
