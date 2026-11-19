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

from nemo.collections.asr.losses import ctc as ctc_loss_module
from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.modules import ConvASRDecoder
from nemo.collections.asr.parts.packed_sequence import PackedEncoderActivations, pack_encoder_output
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.core.classes.mixins import AccessMixin
from nemo.core.utils.optional_libs import TRITON_AVAILABLE


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
                    'subsampling_factor': 8,
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


PACKED_VOCABULARY = [' ', 'a', 'b', 'c', 'd', 'e']


def _packed_interctc_model(
    apply_at_layers=(0, 1), loss_weights=(0.1, 0.3), strategy='greedy_batch', decoder=None, decoding=None, **encoder
) -> EncDecCTCModel:
    """A CTC model over a small Conformer without dropout, with layer normalization in the convolution modules.

    Batch normalization takes its training statistics from the valid frames in the packed forward
    and from all frames in the padded one, so it would hide the parity that these tests assert in training mode.
    """
    encoder_config = {
        '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
        'feat_in': 16,
        'n_layers': 3,
        'd_model': 32,
        'n_heads': 2,
        'subsampling_factor': 4,
        'subsampling_conv_channels': 4,
        'conv_norm_type': 'layer_norm',
        'dropout': 0.0,
        'dropout_pre_encoder': 0.0,
        'dropout_emb': 0.0,
        'dropout_att': 0.0,
    }
    encoder_config.update(encoder)
    feat_out = encoder_config.get('feat_out', -1)
    decoder_config = {
        '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
        'feat_in': feat_out if feat_out > 0 else encoder_config['d_model'],
        'num_classes': len(PACKED_VOCABULARY),
        'vocabulary': PACKED_VOCABULARY,
    }
    if decoder is not None:
        decoder_config.update(decoder)
    torch.manual_seed(0)
    return EncDecCTCModel(
        cfg=DictConfig(
            {
                'preprocessor': {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'},
                'encoder': encoder_config,
                'decoder': decoder_config,
                'decoding': {'strategy': strategy, **(decoding or {})},
                'interctc': {'apply_at_layers': list(apply_at_layers), 'loss_weights': list(loss_weights)},
            }
        )
    )


def _interctc_step(
    model, features, lengths, targets, target_lengths, *, packed, compute_wer=True, compute_loss=True, log_prefix=""
):
    """The loss and the InterCTC metrics of one batch as a training step computes them, padded or packed."""
    AccessMixin.set_access_enabled(access_enabled=True, guid=model.model_guid)
    try:
        if packed:
            encoded = model.encoder.forward_sequence_packed(features, lengths)
            log_probs = model.decoder.forward_packed(encoded)
            loss = model.loss.forward_packed(log_probs=log_probs, targets=targets, target_lengths=target_lengths)
        else:
            encoded, encoded_lengths = model.encoder(audio_signal=features, length=lengths)
            log_probs = model.decoder(encoder_output=encoded)
            loss = model.loss(
                log_probs=log_probs, targets=targets, input_lengths=encoded_lengths, target_lengths=target_lengths
            )
        return model.add_interctc_losses(
            loss,
            targets,
            target_lengths,
            compute_wer=compute_wer,
            compute_loss=compute_loss,
            log_wer_num_denom=True,
            log_prefix=log_prefix,
        )
    finally:
        AccessMixin.reset_registry(model)
        AccessMixin.set_access_enabled(access_enabled=False, guid=model.model_guid)


def _padded_and_packed_models(device='cpu', dtype=torch.float64, **options):
    padded_model = _packed_interctc_model(**options).to(device=device, dtype=dtype).train()
    packed_model = _packed_interctc_model(**options).to(device=device, dtype=dtype).train()
    packed_model.load_state_dict(padded_model.state_dict())
    return padded_model, packed_model


def _interctc_batch(lengths, device='cpu', dtype=torch.float64):
    torch.manual_seed(1)
    lengths = torch.tensor(lengths, device=device)
    features = torch.randn(len(lengths), 16, int(lengths.max()), dtype=dtype, device=device)
    targets = torch.randint(0, len(PACKED_VOCABULARY), (len(lengths), 3), device=device)
    target_lengths = torch.tensor([3, 1, 2, 1, 0], device=device)[: len(lengths)]
    return features, lengths, targets, target_lengths


class TestSequencePackedInterCTC:
    """InterCTC over ``ConformerEncoder.forward_sequence_packed`` against the padded forward of the same batch."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "lengths",
        [
            (64, 9, 40, 3, 0),  # 3 input frames leave no encoder frame, as the empty utterance
            (4, 3, 0, 5, 1),  # one encoder frame or none per utterance
        ],
    )
    @pytest.mark.parametrize(
        "options",
        [
            {},
            {"apply_at_layers": (1,), "loss_weights": (0.3,), "strategy": "greedy"},
            {"feat_out": 24, "self_attention_model": "abs_pos"},
        ],
    )
    def test_losses_metrics_and_gradients_match_padded(self, lengths, options):
        padded_model, packed_model = _padded_and_packed_models(**options)
        batch = _interctc_batch(lengths)
        padded_loss, padded_metrics = _interctc_step(padded_model, *batch, packed=False)
        packed_loss, packed_metrics = _interctc_step(packed_model, *batch, packed=True)
        torch.testing.assert_close(packed_loss, padded_loss)
        layers = options.get("apply_at_layers", (0, 1))
        expected = {"final_loss"}
        for layer in layers:
            expected |= {f"inter_ctc_loss_l{layer}", f"inter_wer_l{layer}"}
            expected |= {f"inter_wer_num_l{layer}", f"inter_wer_denom_l{layer}"}
        assert set(packed_metrics) == set(padded_metrics) == expected
        for name, value in padded_metrics.items():
            torch.testing.assert_close(packed_metrics[name], value, msg=name)
        padded_loss.backward()
        packed_loss.backward()
        padded_parameters = dict(padded_model.named_parameters())
        for name, parameter in packed_model.named_parameters():
            reference = padded_parameters[name].grad
            assert (parameter.grad is None) == (reference is None), name
            if reference is not None:
                torch.testing.assert_close(parameter.grad, reference, msg=name)
        assert padded_parameters["decoder.decoder_layers.0.weight"].grad.abs().sum() > 0

    @pytest.mark.unit
    def test_captured_tensors_are_packed_log_probabilities(self):
        padded_model, packed_model = _padded_and_packed_models()
        features, lengths, _, _ = _interctc_batch((64, 9, 40))
        captures = []
        for model, packed in ((padded_model, False), (packed_model, True)):
            AccessMixin.set_access_enabled(access_enabled=True, guid=model.model_guid)
            try:
                if packed:
                    model.encoder.forward_sequence_packed(features, lengths)
                else:
                    model.encoder(audio_signal=features, length=lengths)
                captures.append(model.get_captured_interctc_tensors())
            finally:
                AccessMixin.reset_registry(model)
                AccessMixin.set_access_enabled(access_enabled=False, guid=model.model_guid)
        padded_captures, packed_captures = captures
        assert len(packed_captures) == len(padded_captures) == 2
        for (padded_log_probs, padded_lengths), (packed_log_probs, packed_lengths) in zip(
            padded_captures, packed_captures
        ):
            assert isinstance(packed_log_probs, PackedEncoderActivations)
            assert packed_lengths.tolist() == padded_lengths.tolist() == packed_log_probs.lengths.tolist()
            valid = torch.arange(padded_log_probs.shape[1])[None, :] < padded_lengths[:, None]
            torch.testing.assert_close(packed_log_probs.data, padded_log_probs[valid])

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "use_transformer,upsampling_factor,refusal",
        [
            (False, 1, None),
            (False, 2, "TransformerCTCDecoder has no sequence-packed form"),
            (True, 1, "TransformerCTCDecoder has no sequence-packed form"),
            (None, 1, "ConvASRDecoder has no forward_packed"),
        ],
    )
    def test_decoders_without_a_packed_form_are_refused(
        self, monkeypatch, use_transformer, upsampling_factor, refusal
    ):
        """The Transformer bridge and the upsampling of the Transformer CTC head have no packed form yet."""
        decoder = None
        if use_transformer is None:
            monkeypatch.delattr(ConvASRDecoder, "forward_packed")
        else:
            decoder = {
                '_target_': 'nemo.collections.speechlm2.parts.ctc_timestamp_utils.TransformerCTCDecoder',
                'use_transformer': use_transformer,
                'n_heads': 2,
                'n_layers': 1,
                'sync_max_audio_length': False,
                'upsampling_factor': upsampling_factor,
            }
        if refusal is None:
            # the Conv-only head without upsampling runs packed: losses and gradients as the padded step
            padded_model, packed_model = _padded_and_packed_models(decoder=decoder)
            batch = _interctc_batch((64, 9, 40))
            padded_loss, padded_metrics = _interctc_step(padded_model, *batch, packed=False)
            packed_loss, packed_metrics = _interctc_step(packed_model, *batch, packed=True)
            torch.testing.assert_close(packed_loss, padded_loss)
            for name, value in padded_metrics.items():
                torch.testing.assert_close(packed_metrics[name], value, msg=name)
            padded_loss.backward()
            packed_loss.backward()
            padded_parameters = dict(padded_model.named_parameters())
            for name, parameter in packed_model.named_parameters():
                torch.testing.assert_close(parameter.grad, padded_parameters[name].grad, msg=name)
            return
        model = _packed_interctc_model(decoder=decoder).double().train()
        features, lengths, _, _ = _interctc_batch((64, 9))
        AccessMixin.set_access_enabled(access_enabled=True, guid=model.model_guid)
        try:
            model.encoder.forward_sequence_packed(features, lengths)
            with pytest.raises(NotImplementedError, match=refusal):
                model.get_captured_interctc_tensors()
        finally:
            AccessMixin.reset_registry(model)
            AccessMixin.set_access_enabled(access_enabled=False, guid=model.model_guid)

    @pytest.mark.unit
    def test_loss_without_a_packed_form_is_refused(self, monkeypatch):
        model = _packed_interctc_model().double().train()
        features, lengths, targets, target_lengths = _interctc_batch((64, 9))
        AccessMixin.set_access_enabled(access_enabled=True, guid=model.model_guid)
        try:
            model.encoder.forward_sequence_packed(features, lengths)
            monkeypatch.delattr(type(model.loss), "forward_packed")
            with pytest.raises(NotImplementedError, match="CTCLoss has no forward_packed"):
                model.add_interctc_losses(torch.zeros((), dtype=torch.float64), targets, target_lengths, False)
        finally:
            AccessMixin.reset_registry(model)
            AccessMixin.set_access_enabled(access_enabled=False, guid=model.model_guid)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "strategy,decoding",
        [
            ("greedy", {"greedy": {"preserve_alignments": True}}),
            ("greedy_batch", {"confidence_cfg": {"preserve_frame_confidence": True}}),
            ("beam_batch", {"beam": {"beam_size": 2, "beam_alpha": None, "kenlm_path": None}}),
        ],
    )
    def test_wer_decoding_that_needs_log_probabilities_is_refused(self, strategy, decoding):
        """The packed WER is decoded from labels; decoding options that need the log probabilities raise."""
        padded_model, packed_model = _padded_and_packed_models(strategy=strategy, decoding=decoding)
        batch = _interctc_batch((64, 9, 40))
        _, padded_metrics = _interctc_step(padded_model, *batch, packed=False)
        assert "inter_wer_l0" in padded_metrics
        with pytest.raises(NotImplementedError, match="decoded from per-frame labels"):
            _interctc_step(packed_model, *batch, packed=True)
        # without the WER the same configuration runs
        packed_loss, packed_metrics = _interctc_step(packed_model, *batch, packed=True, compute_wer=False)
        assert torch.isfinite(packed_loss) and "inter_wer_l0" not in packed_metrics

    @pytest.mark.unit
    def test_metrics_without_loss_match_padded(self):
        """Validation without the loss: the WER metrics alone, with their prefix."""
        padded_model, packed_model = _padded_and_packed_models()
        batch = _interctc_batch((64, 9, 40, 3, 0))
        results = [
            _interctc_step(model.eval(), *batch, packed=packed, compute_loss=False, log_prefix="val_")
            for model, packed in ((padded_model, False), (packed_model, True))
        ]
        (padded_loss, padded_metrics), (packed_loss, packed_metrics) = results
        assert padded_loss is None and packed_loss is None
        expected = {f"val_inter_wer{kind}_l{layer}" for kind in ("", "_num", "_denom") for layer in (0, 1)}
        assert set(packed_metrics) == set(padded_metrics) == expected
        for name, value in padded_metrics.items():
            torch.testing.assert_close(packed_metrics[name], value, msg=name)

    @pytest.mark.unit
    @pytest.mark.parametrize("temperature", [1.0, 2.0])
    def test_decoder_forward_packed_matches_forward(self, temperature):
        decoder = _packed_interctc_model().decoder.double()
        decoder.temperature = temperature
        torch.manual_seed(2)
        lengths = torch.tensor([5, 0, 2])
        states = torch.randn(3, 5, 32, dtype=torch.float64, requires_grad=True)
        padded = decoder(encoder_output=states.transpose(1, 2))
        packed_states = states.detach().clone().requires_grad_()
        packed = decoder.forward_packed(pack_encoder_output(packed_states, lengths))
        valid = torch.arange(5)[None, :] < lengths[:, None]
        assert packed.lengths.tolist() == [5, 0, 2] and packed.data.shape == (7, len(PACKED_VOCABULARY) + 1)
        torch.testing.assert_close(packed.data, padded[valid])
        padded[valid].square().sum().backward()
        packed.data.square().sum().backward()
        torch.testing.assert_close(packed_states.grad[valid], states.grad[valid])

    @pytest.mark.unit
    def test_decoder_forward_packed_rejects_other_inputs(self):
        decoder = _packed_interctc_model().decoder.double()
        states = torch.randn(2, 5, 32, dtype=torch.float64)
        lengths = torch.tensor([5, 2])
        with pytest.raises(TypeError, match="PackedEncoderActivations"):
            decoder.forward_packed(states.transpose(1, 2))
        with pytest.raises(ValueError, match="feature width 32, got 7"):
            decoder.forward_packed(pack_encoder_output(states[..., :7], lengths))
        empty = decoder.forward_packed(pack_encoder_output(states[:0], lengths[:0]))
        assert empty.data.shape == (0, len(PACKED_VOCABULARY) + 1) and empty.lengths.tolist() == []
        decoder.add_adapter(name='adapter', cfg=LinearAdapterConfig(in_features=32, dim=4))
        with pytest.raises(NotImplementedError, match="Adapters"):
            decoder.forward_packed(pack_encoder_output(states, lengths))
        # a disabled adapter is not applied by forward either
        decoder.set_enabled_adapters(enabled=False)
        assert decoder.forward_packed(pack_encoder_output(states, lengths)).data.shape == (7, 7)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSequencePackedInterCTCCuda:
    """The FastConformer layout (``dw_striding`` by 8, ``rel_pos``) with InterCTC, as the trainings run it."""

    OPTIONS = {
        "subsampling": "dw_striding",
        "subsampling_factor": 8,
        "subsampling_conv_channels": 16,
        "conv_kernel_size": 9,
        # the padded forward would run the fused subsampling kernel, which differs from the plain stack by itself
        "use_triton": False,
    }
    LENGTHS = (600, 9, 300, 7, 0)

    @pytest.mark.unit
    def test_losses_and_gradients_match_padded(self):
        padded_model, packed_model = _padded_and_packed_models("cuda", torch.float32, **self.OPTIONS)
        batch = _interctc_batch(self.LENGTHS, "cuda", torch.float32)
        padded_loss, padded_metrics = _interctc_step(padded_model, *batch, packed=False)
        packed_loss, packed_metrics = _interctc_step(packed_model, *batch, packed=True)
        torch.testing.assert_close(packed_loss, padded_loss, rtol=1e-4, atol=1e-4)
        for name, value in padded_metrics.items():
            torch.testing.assert_close(packed_metrics[name], value, rtol=1e-4, atol=1e-4, msg=name)
        padded_loss.backward()
        packed_loss.backward()
        padded_parameters = dict(padded_model.named_parameters())
        for name, parameter in packed_model.named_parameters():
            reference = padded_parameters[name].grad
            # measured 5e-4 at most; the floor covers the key biases, whose gradient is zero in exact arithmetic
            assert (parameter.grad - reference).norm() <= 2e-3 * reference.norm() + 1e-5, name

    @pytest.mark.unit
    def test_bf16_autocast_losses_and_gradients_match_padded(self):
        padded_model, packed_model = _padded_and_packed_models("cuda", torch.float32, **self.OPTIONS)
        batch = _interctc_batch(self.LENGTHS, "cuda", torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            padded_loss, padded_metrics = _interctc_step(padded_model, *batch, packed=False, compute_wer=False)
            packed_loss, packed_metrics = _interctc_step(packed_model, *batch, packed=True, compute_wer=False)
        assert packed_loss.dtype == torch.float32
        torch.testing.assert_close(packed_loss, padded_loss, rtol=2e-2, atol=2e-2)
        for name, value in padded_metrics.items():
            torch.testing.assert_close(packed_metrics[name], value, rtol=2e-2, atol=2e-2, msg=name)
        padded_loss.backward()
        packed_loss.backward()
        # bf16 rounds differently in the padded and the packed kernels and small gradients carry that noise in full,
        # so the gradients are compared as one vector: measured 2.3e-3, the same as without InterCTC
        padded_parameters = dict(padded_model.named_parameters())
        difference = reference_norm = 0.0
        for name, parameter in packed_model.named_parameters():
            reference = padded_parameters[name].grad
            assert torch.isfinite(parameter.grad).all(), name
            difference += float((parameter.grad - reference).square().sum())
            reference_norm += float(reference.square().sum())
        assert difference**0.5 <= 1e-2 * reference_norm**0.5

    @pytest.mark.unit
    def test_padded_log_probabilities_are_never_formed(self, monkeypatch):
        """The loss of a packed capture runs the token-flat head and kernel: neither padded entry point is called."""
        if not TRITON_AVAILABLE:
            pytest.skip("the token-flat CTC kernel requires Triton")
        model = _packed_interctc_model(**self.OPTIONS).cuda().train()
        batch = _interctc_batch(self.LENGTHS, "cuda", torch.float32)

        def refuse(*args, **kwargs):
            raise AssertionError("a padded entry point was called")

        monkeypatch.setattr(model.decoder, "forward", refuse)
        monkeypatch.setattr(model.loss, "forward", refuse)
        monkeypatch.setattr(ctc_loss_module, "unpack_encoder_output", refuse)
        loss, metrics = _interctc_step(model, *batch, packed=True, compute_wer=False)
        assert torch.isfinite(loss) and set(metrics) == {"final_loss", "inter_ctc_loss_l0", "inter_ctc_loss_l1"}
        loss.backward()
