# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from typing import Dict

import lightning.pytorch as pl
import pytest
import torch
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "decoder_kind,upsampling_factor",
    [("transformer", 1), ("transformer", 2), ("conv_only", 1), ("conv_only", 2), ("conv", 1)],
)
def test_interctc_decoder_receives_intermediate_lengths_and_backpropagates(
    monkeypatch, decoder_kind, upsampling_factor
):
    from torch.nn.attention.flex_attention import create_mask

    from nemo.collections.asr.modules import transformer_encoder_utils

    # FlexAttention has no CPU backward; use equivalent masked math attention for this CPU regression.
    def cpu_attention(q, k, v, *, block_mask=None, score_mod=None):
        assert score_mod is None
        mask = None
        if block_mask is not None:
            mask = create_mask(block_mask.mask_mod, q.shape[0], q.shape[1], q.shape[2], k.shape[2], device=q.device)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    monkeypatch.setattr(transformer_encoder_utils, '_get_flex_attention', lambda _: cpu_attention)
    decoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
        'feat_in': 32,
        'num_classes': 3,
        'vocabulary': ['a', 'b', 'c'],
    }
    if decoder_kind != "conv":
        decoder.update(
            _target_='nemo.collections.speechlm2.parts.ctc_timestamp_utils.TransformerCTCDecoder',
            use_transformer=decoder_kind == "transformer",
            n_heads=2,
            n_layers=1,
            sync_max_audio_length=False,
            upsampling_factor=upsampling_factor,
        )
    model = EncDecCTCModel(
        cfg=DictConfig(
            {
                'preprocessor': {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'},
                'encoder': {
                    '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
                    'feat_in': 64,
                    'n_layers': 2,
                    'd_model': 32,
                    'n_heads': 2,
                },
                'decoder': decoder,
                'interctc': {'apply_at_layers': [0], 'loss_weights': [0.3]},
            }
        )
    ).train()
    calls = []
    forward = model.decoder.forward

    def capture(**kwargs):
        calls.append(kwargs)
        return forward(**kwargs)

    monkeypatch.setattr(model.decoder, "forward", capture)
    AccessMixin.set_access_enabled(access_enabled=True, guid=model.model_guid)
    try:
        log_probs, lengths, _ = model(
            processed_signal=torch.randn(2, 64, 64), processed_signal_length=torch.tensor([64, 48])
        )
        registry = {
            key: value
            for module in AccessMixin.get_module_registry(model.encoder).values()
            for key, value in module.items()
        }
        if upsampling_factor == 2:
            # Three target tokens require the four output frames, not just the two encoder frames.
            registry['interctc/layer_output_0'][0] = registry['interctc/layer_output_0'][0][:, :, :2]
            registry['interctc/layer_length_0'][0] = torch.full_like(registry['interctc/layer_length_0'][0], 2)
        intermediate_lengths = registry['interctc/layer_length_0'][0]
        intermediate = model.get_captured_interctc_tensors()
        assert len(intermediate) == 1
        assert intermediate[0][0].shape[1] == (4 if upsampling_factor == 2 else log_probs.shape[1])
        torch.testing.assert_close(intermediate[0][1], intermediate_lengths * upsampling_factor)
        if decoder_kind == "transformer" or upsampling_factor > 1:
            torch.testing.assert_close(calls[0]['encoded_lengths'] * upsampling_factor, lengths)
            assert calls[1]['encoded_lengths'] is intermediate_lengths
        else:
            assert all('encoded_lengths' not in call for call in calls)
        # Backpropagate the intermediate CTC loss alone, so the main loss cannot hide a broken gradient path.
        loss = model.loss(
            log_probs=intermediate[0][0],
            targets=torch.tensor([[0, 1, 0], [0, 1, 0]]) if upsampling_factor == 2 else torch.tensor([[0, 1], [1, 2]]),
            target_lengths=torch.tensor([3, 3]) if upsampling_factor == 2 else torch.tensor([2, 2]),
            input_lengths=intermediate[0][1],
        )
        assert torch.isfinite(loss)
        assert loss > 0
        loss.backward()
        gradients = [parameter.grad for parameter in model.decoder.parameters() if parameter.requires_grad]
        assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    finally:
        AccessMixin.reset_registry(model)
        AccessMixin.set_access_enabled(access_enabled=False, guid=model.model_guid)


class TestInterCTCLoss:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_class",
        [EncDecCTCModel, EncDecHybridRNNTCTCModel],
    )
    @pytest.mark.parametrize(
        "encoder_config",
        [jasper_encoder_config(num_layers=8), conformer_encoder_config()],
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
        input_length = torch.randint(low=321, high=500, size=[1])
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
            AccessMixin.set_access_enabled(access_enabled=True, guid=asr_model.model_guid)
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

            # Explicitly pass accelerator as cpu, since default val in PTL >= 2.0 is auto and it picks cuda
            # which further causes an error in all reduce at: https://github.com/NVIDIA-NeMo/Speech/blob/v1.18.1/nemo/collections/asr/modules/conv_asr.py#L209
            trainer = pl.Trainer(max_epochs=1, accelerator='cpu')
            trainer.fit(
                asr_model,
                train_dataloaders=torch.utils.data.DataLoader(
                    DummyDataset([input_signal, input_length, target, target_length]),
                    collate_fn=lambda x: x[0],
                ),
                val_dataloaders=torch.utils.data.DataLoader(
                    DummyDataset([input_signal, input_length, target, target_length]),
                    collate_fn=lambda x: x[0],
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
                    DummyDataset([input_signal, input_length, target, target_length]),
                    collate_fn=lambda x: x[0],
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
