# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import math
from collections import Counter
from types import SimpleNamespace
from typing import List, Union
from unittest.mock import patch

import numpy as np
import onnx
import pytest
import torch
from examples.speaker_tasks.diarization.neural_diarizer.e2e_diarize_speech import (
    DiarizationConfig,
    get_tensor_path,
    load_diarization_model,
)
from omegaconf import DictConfig
from onnx.reference import ReferenceEvaluator

from nemo.collections.asr.losses.aux_diarization_loss import activity_loss, phantom_loss
from nemo.collections.asr.metrics.speaker_counting import speaker_count_metrics
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.models.sortformer_diar_models import _OversamplingDistributedSampler
from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
from nemo.collections.asr.parts.utils.sortformer_utils import (
    InferenceProfiler,
    configure_output_subsampling_factor,
    get_prediction_cache_metadata,
)


class RecordingSpecAugment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shapes = []

    def forward(self, input_spec, length):
        self.input_shapes.append(tuple(input_spec.shape))
        return input_spec


def _create_sortformer_model(
    high_resolution=False,
    output_subsampling_factor=None,
    include_transformer_encoder=True,
    frontend_encoder="conformer",
    causal_attn_rate=0.0,
    streaming_mode=False,
    max_causal_tail_len=0,
    causal_tail_prob=1.0,
    min_causal_tail_block_size=1,
    max_causal_tail_block_size=1,
    encoder_attn_mode="full",
    logits_loss=False,
    activity_weight=0.0,
    phantom_weight=0.0,
    phantom_target="both",
    phantom_threshold=0.25,
    phantom_temperature=0.5,
):
    if output_subsampling_factor is None:
        output_subsampling_factor = 1 if high_resolution else 8

    model = {
        'sample_rate': 16000,
        'pil_weight': 0.5,
        'ats_weight': 0.5,
        'activity_weight': activity_weight,
        'phantom_weight': phantom_weight,
        'phantom_target': phantom_target,
        'phantom_threshold': phantom_threshold,
        'phantom_temperature': phantom_temperature,
        'max_num_of_spks': 4,
        'high_resolution': high_resolution,
        'output_subsampling_factor': output_subsampling_factor,
        'async_streaming': False,
        'streaming_mode': streaming_mode,
        'max_causal_tail_len': max_causal_tail_len,
        'causal_tail_prob': causal_tail_prob,
        'min_causal_tail_block_size': min_causal_tail_block_size,
        'max_causal_tail_block_size': max_causal_tail_block_size,
    }
    transformer_frontend = frontend_encoder in {"transformer", "streaming_transformer"}
    model_defaults = {
        'fc_d_model': 128 if transformer_frontend else 32,
        'tf_d_model': 16,
    }
    preprocessor = {
        '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
        'normalize': 'per_feature',
        'window_size': 0.025,
        'sample_rate': 16000,
        'window_stride': 0.01,
        'window': 'hann',
        'features': 128 if transformer_frontend else 80,
        'n_fft': 512,
        'frame_splicing': 1,
        'dither': 0.00001,
    }

    sortformer_modules = {
        '_target_': 'nemo.collections.asr.modules.sortformer_modules.SortformerModules',
        'num_spks': model['max_num_of_spks'],
        'dropout_rate': 0.5,
        'fc_d_model': model_defaults['fc_d_model'],
        'tf_d_model': model_defaults['tf_d_model'],
        'causal_attn_rate': causal_attn_rate,
    }

    if transformer_frontend:
        # Keep the production Transformer architecture and options, but scale its depth and width for CPU unit tests.
        encoder = {
            '_target_': (
                'nemo.collections.asr.modules.StreamingTransformerEncoder'
                if frontend_encoder == "streaming_transformer"
                else 'nemo.collections.asr.modules.TransformerEncoder'
            ),
            'feat_in': preprocessor['features'],
            'feat_out': -1,
            'n_layers': 1,
            'd_model': model_defaults['fc_d_model'],
            'n_heads': 8,
            'subsampling': 'feature_stacking',
            'subsampling_factor': 8,
            'ff_expansion': 4.0,
            'self_attention_model': 'rope',
            'pos_emb_max_len': 5000,
            'xscaling': False,
            'qkv_bias': False,
            'qk_norm': False,
            'pre_block_norm': True,
            'causal_tail_len': 0,
            'causal_tail_block_size': 1,
            'drop_rate': 0.1,
            'dropout_pre_encoder': 0.1,
            'dropout_emb': 0.0,
            'sync_max_audio_length': True,
        }
        if encoder_attn_mode is not None:
            encoder['attn_mode'] = encoder_attn_mode
    else:
        encoder = {
            '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
            'feat_in': preprocessor['features'],
            'feat_out': -1,
            'n_layers': 1,
            'd_model': model_defaults['fc_d_model'],
            'subsampling': 'dw_striding',
            'subsampling_factor': 8,
            'subsampling_conv_channels': 256,
            'causal_downsampling': False,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rel_pos',
            'n_heads': 8,
            'att_context_size': [-1, -1],
            'att_context_style': 'regular',
            'xscaling': True,
            'untie_biases': True,
            'pos_emb_max_len': 5000,
            'conv_kernel_size': 9,
            'conv_norm_type': 'batch_norm',
            'conv_context_size': None,
            'dropout': 0.1,
            'dropout_pre_encoder': 0.1,
            'dropout_emb': 0.0,
            'dropout_att': 0.1,
            'stochastic_depth_drop_prob': 0.0,
            'stochastic_depth_mode': 'linear',
            'stochastic_depth_start_layer': 1,
        }

    transformer_encoder = {
        '_target_': 'nemo.collections.asr.modules.transformer.transformer_encoders.TransformerEncoder',
        'num_layers': 1,
        'hidden_size': model_defaults['tf_d_model'],
        'inner_size': 32,
        'num_attention_heads': 8,
        'attn_score_dropout': 0.5,
        'attn_layer_dropout': 0.5,
        'ffn_dropout': 0.5,
        'hidden_act': 'relu',
        'pre_ln': False,
        'pre_ln_final_layer_norm': True,
    }

    if logits_loss:
        loss = {
            '_target_': 'nemo.collections.asr.losses.bce_loss.BCEWithLogitsLoss',
            'reduction': 'mean',
        }
    else:
        loss = {
            '_target_': 'nemo.collections.asr.losses.bce_loss.BCELoss',
            'weight': None,
            'reduction': 'mean',
        }

    model_config = {
        'sample_rate': 16000,
        'pil_weight': 0.5,
        'ats_weight': 0.5,
        'activity_weight': activity_weight,
        'phantom_weight': phantom_weight,
        'phantom_target': phantom_target,
        'phantom_threshold': phantom_threshold,
        'phantom_temperature': phantom_temperature,
        'max_num_of_spks': 4,
        'high_resolution': high_resolution,
        'output_subsampling_factor': output_subsampling_factor,
        'streaming_mode': streaming_mode,
        'max_causal_tail_len': max_causal_tail_len,
        'causal_tail_prob': causal_tail_prob,
        'min_causal_tail_block_size': min_causal_tail_block_size,
        'max_causal_tail_block_size': max_causal_tail_block_size,
        'model_defaults': DictConfig(model_defaults),
        'encoder': DictConfig(encoder),
        'sortformer_modules': DictConfig(sortformer_modules),
        'preprocessor': DictConfig(preprocessor),
        'loss': DictConfig(loss),
        'optim': {
            'optimizer': 'Adam',
            'lr': 0.001,
            'betas': (0.9, 0.98),
        },
    }
    if include_transformer_encoder:
        model_config['transformer_encoder'] = DictConfig(transformer_encoder)
    modelConfig = DictConfig(model_config)
    model = SortformerEncLabelModel(cfg=modelConfig)
    return model


@pytest.fixture()
def sortformer_model():
    return _create_sortformer_model()


class TestSortformerEncLabelModelOffline:
    @pytest.mark.unit
    def test_constructor(self, sortformer_model):
        sortformer_model.streaming_mode = False
        sortformer_diar_model = sortformer_model.train()
        confdict = sortformer_diar_model.to_config_dict()
        instance2 = SortformerEncLabelModel.from_config_dict(confdict)
        assert isinstance(instance2, SortformerEncLabelModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "trainer_attached",
            "trainer_max_steps",
            "scheduler_max_steps",
            "scheduler_shape",
            "expected_max_steps",
            "expected_updates",
        ),
        [
            (True, 200, 100, "single", 200, [200]),
            (True, 200, 200, "tuple", 200, []),
            (False, None, 100, "single", 100, []),
            (True, None, 100, "single", 100, []),
            (True, 0, 100, "single", 100, []),
            (True, -1, 100, "single", 100, []),
        ],
        ids=["stale", "matching-and-no-max-steps", "no-trainer", "no-horizon", "zero-horizon", "negative-horizon"],
    )
    def test_on_train_start_repairs_scheduler_max_steps(
        self,
        trainer_attached,
        trainer_max_steps,
        scheduler_max_steps,
        scheduler_shape,
        expected_max_steps,
        expected_updates,
    ):
        class FakeScheduler:
            def __init__(self, max_steps):
                self._max_steps = max_steps
                self.updates = []

            @property
            def max_steps(self):
                return self._max_steps

            @max_steps.setter
            def max_steps(self, value):
                self._max_steps = value
                self.updates.append(value)

        model = _create_sortformer_model()
        model._trainer = SimpleNamespace(max_steps=trainer_max_steps) if trainer_attached else None
        scheduler = FakeScheduler(scheduler_max_steps)
        scheduler_without_max_steps = SimpleNamespace()
        schedulers = scheduler if scheduler_shape == "single" else (scheduler, scheduler_without_max_steps)

        with patch.object(model, "lr_schedulers", return_value=schedulers) as lr_schedulers:
            model.on_train_start()

        assert scheduler.max_steps == expected_max_steps
        assert scheduler.updates == expected_updates
        assert not hasattr(scheduler_without_max_steps, "max_steps")
        if trainer_attached and trainer_max_steps is not None and trainer_max_steps > 0:
            lr_schedulers.assert_called_once_with()
        else:
            lr_schedulers.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "activity_weight, audio_shape, audio_lengths",
        [
            (0.0, (2, 4000), (4000, 3200)),
            (0.25, (1, 3200), (2800,)),
        ],
    )
    def test_forward_returns_probabilities_by_default_and_aligned_logits_on_request(
        self, activity_weight, audio_shape, audio_lengths
    ):
        model = _create_sortformer_model(activity_weight=activity_weight).eval()
        audio = torch.randn(audio_shape)
        audio_lengths = torch.tensor(audio_lengths)
        with torch.no_grad():
            torch.manual_seed(0)
            default_preds = model(audio, audio_lengths)
            torch.manual_seed(0)
            preds, logits, activity_logits = model(audio, audio_lengths, return_logits=True)

        valid_mask = preds.sum(dim=-1) > 0
        assert isinstance(default_preds, torch.Tensor)
        assert preds.shape == logits.shape == default_preds.shape
        torch.testing.assert_close(default_preds, preds)
        torch.testing.assert_close(torch.sigmoid(logits[valid_mask]), preds[valid_mask])
        if activity_weight > 0.0:
            assert activity_logits.shape == (*preds.shape[:2], 3)
        else:
            assert activity_logits is None

    @pytest.mark.unit
    @pytest.mark.parametrize("streaming_mode", [False, True])
    def test_transformer_encoder_is_optional(self, streaming_mode):
        model = _create_sortformer_model(
            include_transformer_encoder=False,
            frontend_encoder="transformer",
        )
        model.streaming_mode = streaming_mode
        if streaming_mode:
            model.train()
        else:
            model.eval()
        audio = torch.randn(2, 8000)
        audio_lengths = torch.tensor([8000, 6400], dtype=torch.long)

        with torch.no_grad():
            preds = model(audio, audio_lengths)

        assert model.transformer_encoder is None
        assert preds.shape[0] == audio.shape[0]

    @pytest.mark.unit
    def test_transformer_encoder_rejects_causal_attention_rate(self) -> None:
        """Reject unsupported dynamic attention-context changes for the Transformer backbone."""
        with pytest.raises(
            ValueError,
            match="causal_attn_rate is only supported by encoders with configurable att_context_size",
        ):
            _create_sortformer_model(frontend_encoder="transformer", causal_attn_rate=0.5)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (2, 1),  # Example 1
            (1, 2),  # Example 2
        ],
    )
    def test_forward_infer(self, sortformer_model, batch_size, sample_len):
        sortformer_model.streaming_mode = False
        sortformer_diar_model = sortformer_model.eval()
        confdict = sortformer_diar_model.to_config_dict()
        sampling_rate = confdict['preprocessor']['sample_rate']
        input_signal = torch.randn(size=(batch_size, sample_len * sampling_rate))
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        with torch.no_grad():
            # batch size 1
            preds_list = []
            for i in range(input_signal.size(0)):
                preds = sortformer_diar_model.forward(input_signal[i : i + 1], input_signal_length[i : i + 1])
                preds_list.append(preds)
            preds_instance = torch.cat(preds_list, 0)

            # batch size 4
            preds_batch = sortformer_diar_model.forward(input_signal, input_signal_length)
        assert preds_instance.shape == preds_batch.shape

        diff = torch.mean(torch.abs(preds_instance - preds_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(preds_instance - preds_batch))
        assert diff <= 1e-6

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "audio_lengths, max_batch_dur",
        [
            ((4000, 3200, 1600), 0.45),
            ((3200, 2800, 2400, 1600), 0.30),
        ],
        ids=["two-sub-batches", "three-sub-batches"],
    )
    def test_oom_safe_feature_extraction_matches_unsplit_output(self, audio_lengths, max_batch_dur):
        model = _create_sortformer_model().eval()
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
        audio = torch.zeros(audio_lengths.shape[0], int(audio_lengths.max().item()))
        for sample_idx, sample_length in enumerate(audio_lengths.tolist()):
            sample_times = torch.arange(sample_length, dtype=torch.float32)
            audio[sample_idx, :sample_length] = torch.sin(sample_times * (sample_idx + 1) * 0.013)

        with torch.no_grad():
            model.max_batch_dur = 0
            expected_features, expected_lengths = model.process_signal(audio, audio_lengths)
            model.max_batch_dur = max_batch_dur
            actual_features, actual_lengths = model.process_signal(audio, audio_lengths)

        torch.testing.assert_close(actual_lengths, expected_lengths)
        assert actual_features.shape == expected_features.shape
        for sample_idx, feature_length in enumerate(actual_lengths.tolist()):
            torch.testing.assert_close(
                actual_features[sample_idx, :, :feature_length],
                expected_features[sample_idx, :, :feature_length],
                rtol=1e-5,
                atol=1e-5,
            )
            padding = actual_features[sample_idx, :, feature_length:]
            assert torch.all(padding == model.preprocessor.featurizer.pad_value)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "audio_width, audio_lengths, max_batch_dur, expected_input_shapes",
        [
            (6000, (4000, 3000, 2000, 1000), 0.45, ((1, 4000), (2, 3000), (1, 1000))),
            (24000, (20000, 4000, 3000), 0.50, ((1, 20000), (2, 4000))),
        ],
        ids=["duration-bounded", "singleton-over-limit"],
    )
    def test_oom_safe_feature_extraction_crops_and_bounds_sub_batches(
        self, audio_width, audio_lengths, max_batch_dur, expected_input_shapes
    ):
        model = _create_sortformer_model().eval()
        model.max_batch_dur = max_batch_dur
        audio = torch.randn(len(audio_lengths), audio_width)
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

        with patch.object(model.preprocessor, "forward", wraps=model.preprocessor.forward) as preprocessor_forward:
            with torch.no_grad():
                model.process_signal(audio, audio_lengths)

        calls = preprocessor_forward.call_args_list
        input_shapes = tuple(tuple(call.kwargs["input_signal"].shape) for call in calls)
        assert len(calls) > 1
        assert input_shapes == expected_input_shapes
        assert all(input_shape[-1] < audio_width for input_shape in input_shapes)
        for call in calls:
            input_signal = call.kwargs["input_signal"]
            input_lengths = call.kwargs["length"]
            assert input_signal.shape[-1] == input_lengths.max().item()
            padded_duration = input_signal.shape[0] * input_signal.shape[-1] / model.preprocessor._cfg.sample_rate
            assert padded_duration <= max_batch_dur or input_signal.shape[0] == 1

    @pytest.mark.unit
    def test_oom_safe_feature_extraction_uses_bfloat16_storage_during_bfloat16_inference(self):
        model = _create_sortformer_model().to(dtype=torch.bfloat16).eval()
        model.max_batch_dur = 0.30
        audio_lengths = torch.tensor([3200, 2400, 1600], dtype=torch.long)
        audio = torch.randn(audio_lengths.shape[0], int(audio_lengths.max().item()))

        with torch.no_grad():
            processed_features, processed_lengths = model.process_signal(audio, audio_lengths)

        expected_lengths = model.preprocessor.featurizer.get_seq_len(audio_lengths)
        assert processed_features.dtype == torch.bfloat16
        assert processed_lengths.dtype == torch.long
        torch.testing.assert_close(processed_lengths.cpu(), expected_lengths)


class TestSortformerEncLabelModelLossRepresentation:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "step_name, logits_loss, activity_weight, phantom_weight, expected_keyword, need_logits",
        [
            ("training", False, 0.0, 0.0, "probs", False),
            ("training", False, 0.3, 0.0, "probs", True),
            ("validation", False, 0.0, 0.0, "probs", False),
            ("validation", False, 0.0, 0.4, "probs", True),
            ("validation", True, 0.2, 0.4, "logits", True),
        ],
    )
    def test_training_and_validation_use_configured_loss_representation(
        self,
        step_name,
        logits_loss,
        activity_weight,
        phantom_weight,
        expected_keyword,
        need_logits,
    ):
        model = _create_sortformer_model(
            logits_loss=logits_loss,
            activity_weight=activity_weight,
            phantom_weight=phantom_weight,
        )
        logits = torch.randn(2, 5, model.sortformer_modules.n_spk, requires_grad=True)
        preds = torch.sigmoid(logits)
        activity_logits = torch.randn(2, 5, 3, requires_grad=True) if activity_weight > 0.0 else None
        outputs = (preds, logits, activity_logits) if need_logits else preds
        targets = torch.randint(0, 2, preds.shape).float()
        target_lens = torch.tensor([5, 3])
        batch = (torch.randn(2, 80), torch.tensor([80, 64]), targets, target_lens)
        model._optimizer = SimpleNamespace(param_groups=[{"lr": 1e-3}])
        model._trainer = SimpleNamespace(val_dataloaders=[])
        model.validation_step_outputs = []

        with (
            patch.object(model, "forward", return_value=outputs) as forward_mock,
            patch.object(model.loss, "forward", wraps=model.loss.forward) as loss_forward,
            patch.object(model, "log_dict") as log_dict_mock,
        ):
            if step_name == "training":
                metrics = model.training_step(batch, batch_idx=0)
                loss = metrics["loss"]
                logged_metrics = log_dict_mock.call_args.args[0]
            else:
                metrics = model.validation_step(batch, batch_idx=0)
                loss = metrics["val_loss"]
                logged_metrics = metrics

        assert torch.isfinite(loss)
        assert forward_mock.call_args.kwargs["return_logits"] is need_logits
        assert len(loss_forward.call_args_list) == 2
        unexpected_keyword = "probs" if expected_keyword == "logits" else "logits"
        for loss_call in loss_forward.call_args_list:
            assert expected_keyword in loss_call.kwargs
            assert unexpected_keyword not in loss_call.kwargs
        metric_prefix = "val_" if step_name == "validation" else ""
        assert f"{metric_prefix}activity_loss" in logged_metrics
        assert f"{metric_prefix}phantom_loss" in logged_metrics
        spk_count_prefix = "val" if step_name == "validation" else "train"
        for metric_name in ("spk_count_mae", "spk_count_acc"):
            metric = logged_metrics[f"{spk_count_prefix}_{metric_name}"]
            assert metric.ndim == 0
            assert torch.isfinite(metric)
        expected_loss = (
            model.ats_weight * logged_metrics[f"{metric_prefix}ats_loss"]
            + model.pil_weight * logged_metrics[f"{metric_prefix}pil_loss"]
            + activity_weight * logged_metrics[f"{metric_prefix}activity_loss"]
            + phantom_weight * logged_metrics[f"{metric_prefix}phantom_loss"]
        )
        torch.testing.assert_close(loss, expected_loss)

    @pytest.mark.unit
    @pytest.mark.parametrize("strict", [True])
    def test_logits_loss_strictly_loads_legacy_bce_state_dict(self, strict):
        legacy_model = _create_sortformer_model(logits_loss=False)
        logits_model = _create_sortformer_model(logits_loss=True)

        load_result = logits_model.load_state_dict(legacy_model.state_dict(), strict=strict)

        assert not load_result.missing_keys
        assert not load_result.unexpected_keys

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("preds", "targets", "target_lens", "expected_mae", "expected_acc"),
        [
            (
                (((0.9, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),),
                (((0.0, 0.0, 0.8, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),),
                (3,),
                0.0,
                1.0,
            ),
            (
                (((0.9, 0.8, 0.7, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),),
                (((0.0, 0.0, 0.0, 0.9), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),),
                (3,),
                2.0,
                0.0,
            ),
            (
                (((0.9, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),),
                (((0.0, 0.8, 0.7, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),),
                (3,),
                1.0,
                0.0,
            ),
            (
                (((0.0, 0.9, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.8, 0.0)),),
                (((0.0, 0.0, 0.0, 0.9), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),),
                (2,),
                0.0,
                1.0,
            ),
            (
                (
                    ((0.5, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
                    ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
                ),
                (
                    ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
                    ((0.0, 0.0, 0.5, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
                ),
                (3, 3),
                0.0,
                1.0,
            ),
        ],
        ids=["permutation", "over-count", "under-count", "padded-activity", "strict-threshold"],
    )
    def test_speaker_count_metrics(
        self,
        preds,
        targets,
        target_lens,
        expected_mae,
        expected_acc,
    ):
        spk_count_mae, spk_count_acc = speaker_count_metrics(
            torch.tensor(preds),
            torch.tensor(targets),
            torch.tensor(target_lens),
        )

        torch.testing.assert_close(spk_count_mae, torch.tensor(expected_mae))
        torch.testing.assert_close(spk_count_acc, torch.tensor(expected_acc))

    @pytest.mark.unit
    def test_multi_validation_epoch_end_averages_speaker_count_metrics(self):
        model = _create_sortformer_model()
        other_metrics = {
            "val_loss": torch.tensor(0.0),
            "val_ats_loss": torch.tensor(0.0),
            "val_pil_loss": torch.tensor(0.0),
            "val_activity_loss": torch.tensor(0.0),
            "val_phantom_loss": torch.tensor(0.0),
            "val_f1_acc": torch.tensor(0.0),
            "val_precision": torch.tensor(0.0),
            "val_recall": torch.tensor(0.0),
            "val_f1_acc_ats": torch.tensor(0.0),
        }
        outputs = [
            {
                **other_metrics,
                "val_spk_count_mae": torch.tensor(1.0),
                "val_spk_count_acc": torch.tensor(0.25),
            },
            {
                **other_metrics,
                "val_spk_count_mae": torch.tensor(3.0),
                "val_spk_count_acc": torch.tensor(0.75),
            },
        ]

        metrics = model.multi_validation_epoch_end(outputs)["log"]

        torch.testing.assert_close(metrics["val_spk_count_mae"], torch.tensor(2.0))
        torch.testing.assert_close(metrics["val_spk_count_acc"], torch.tensor(0.5))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "target_rows, target_len, expected_classes",
        [
            (
                (((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0)),),
                3,
                ((0, 1, 2, 2),),
            ),
            (
                (((0.5, 0.0, 0.0, 0.0), (0.51, 0.0, 0.0, 0.0), (0.7, 0.8, 0.9, 0.0), (0.0, 0.0, 0.0, 0.0)),),
                4,
                ((0, 1, 2, 0),),
            ),
            (
                (((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),),
                0,
                ((1, 0),),
            ),
        ],
    )
    def test_activity_loss_targets_padding_and_gradients(self, target_rows, target_len, expected_classes):
        targets = torch.tensor(target_rows)
        expected_classes = torch.tensor(expected_classes)
        target_lens = torch.tensor([target_len])
        logits_classes = expected_classes.clone()
        logits_classes[:, target_len:] = (logits_classes[:, target_len:] + 1) % 3
        activity_logits = torch.full((*expected_classes.shape, 3), -12.0)
        activity_logits.scatter_(-1, logits_classes.unsqueeze(-1), 12.0)
        activity_logits.requires_grad_()

        actual_classes = (targets > 0.5).sum(dim=-1).clamp(max=2).long()
        loss = activity_loss(activity_logits, targets, target_lens)
        loss.backward()

        assert torch.equal(actual_classes, expected_classes)
        assert loss.item() < 1e-6
        assert activity_logits.grad is not None
        if target_len > 0:
            assert activity_logits.grad[:, :target_len].abs().sum() > 0
        assert torch.count_nonzero(activity_logits.grad[:, target_len:]) == 0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "phantom_target, expected",
        [
            ("pil", (((1.0, 0.0), (0.0, 0.0)),)),
            ("ats", (((0.0, 1.0), (0.0, 0.0)),)),
            ("both", (((1.0, 1.0), (0.0, 0.0)),)),
        ],
    )
    def test_phantom_target_selection(self, phantom_target, expected):
        model = _create_sortformer_model(phantom_target=phantom_target)
        targets_pil = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        targets_ats = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])

        actual = model._get_phantom_targets(targets_pil, targets_ats)

        torch.testing.assert_close(actual, torch.tensor(expected))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "selected_channels, temperature",
        [
            ((0,), 0.5),
            ((0, 1), 1.5),
            ((), 0.75),
        ],
    )
    def test_phantom_loss_selection_normalization_and_gradients(self, selected_channels, temperature):
        num_spks = 4
        logits = torch.full((1, 4, num_spks), -2.0)
        phantom_targets = torch.zeros_like(logits)
        for speaker in range(num_spks - 1):
            if speaker in selected_channels:
                logits[0, 0, speaker] = 1.0 + speaker
                logits[0, 1, speaker] = 2.0 + speaker
            else:
                phantom_targets[0, 0, speaker] = 1.0
                logits[0, :3, speaker] = 4.0
        logits[0, 3] = 5.0
        logits.requires_grad_()

        loss = phantom_loss(
            logits=logits,
            phantom_targets=phantom_targets,
            target_lens=torch.tensor([3]),
            threshold=0.6,
            temperature=temperature,
        )
        expected = logits.new_zeros(())
        for speaker in selected_channels:
            frame_losses = torch.nn.functional.softplus(logits.detach()[0, :2, speaker])
            expected = expected + temperature * (
                torch.logsumexp(frame_losses / temperature, dim=0) - math.log(frame_losses.numel())
            )
        expected = expected / num_spks
        loss.backward()

        torch.testing.assert_close(loss, expected)
        assert torch.isfinite(loss)
        expected_gradient_mask = torch.zeros_like(logits, dtype=torch.bool)
        for speaker in selected_channels:
            expected_gradient_mask[0, :2, speaker] = True
        assert torch.equal(logits.grad != 0, expected_gradient_mask)

    @pytest.mark.unit
    @pytest.mark.parametrize(("logit", "is_selected"), [(0.0, False), (1.0, True)])
    def test_phantom_loss_uses_strict_probability_threshold(self, logit, is_selected):
        logits = torch.tensor([[[logit]]], requires_grad=True)

        loss = phantom_loss(
            logits=logits,
            phantom_targets=torch.zeros_like(logits),
            target_lens=torch.tensor([1]),
            threshold=0.5,
            temperature=0.5,
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert (logits.grad != 0).item() is is_selected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "field_name, invalid_value, exception_type, error_match",
        [
            ("activity_weight", -0.1, ValueError, "activity_weight must be a non-negative float"),
            ("phantom_weight", True, TypeError, "phantom_weight must be a non-negative float"),
            ("phantom_target", "union", ValueError, "phantom_target must be one of"),
            ("phantom_threshold", -0.1, ValueError, "phantom_threshold must be in"),
            ("phantom_threshold", 1.0, ValueError, "phantom_threshold must be in"),
            ("phantom_temperature", 0.0, ValueError, "phantom_temperature must be greater than zero"),
        ],
    )
    def test_auxiliary_loss_configuration_validation(self, field_name, invalid_value, exception_type, error_match):
        with pytest.raises(exception_type, match=error_match):
            _create_sortformer_model(**{field_name: invalid_value})


class TestSortformerEncLabelModelStreaming:
    @pytest.mark.unit
    @pytest.mark.parametrize("field_name", ["spkcache_len", "chunk_left_context"])
    def test_model_dependent_streaming_overrides_default_to_none(self, field_name):
        assert getattr(DiarizationConfig(), field_name) is None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "training",
            "causal_tail_prob",
            "min_block_size",
            "max_block_size",
            "random_values",
            "expected_runtime_values",
            "expected_randint_args",
        ),
        [
            (True, 1.0, 1, 1, (4,), (4, 1), [(1, 5)]),
            (True, 1.0, 2, 4, (2, 4), (2, 4), [(1, 5), (2, 4)]),
            (True, 0.0, 2, 4, (), (0, 1), []),
            (False, 1.0, 2, 4, (), (0, 1), []),
        ],
    )
    def test_causal_tail_sampling_and_reset(
        self,
        training,
        causal_tail_prob,
        min_block_size,
        max_block_size,
        random_values,
        expected_runtime_values,
        expected_randint_args,
    ):
        model = _create_sortformer_model(
            frontend_encoder="transformer",
            streaming_mode=True,
            max_causal_tail_len=5,
            causal_tail_prob=causal_tail_prob,
            min_causal_tail_block_size=min_block_size,
            max_causal_tail_block_size=max_block_size,
        )
        model.sortformer_modules.spkcache_len = 4
        model.sortformer_modules.fifo_len = 0
        model.sortformer_modules.chunk_len = 2
        model.sortformer_modules.spkcache_update_period = 2
        model.sortformer_modules.chunk_left_context = 0
        model.sortformer_modules.chunk_right_context = 0
        model.train(training)

        observed_runtime_values = []
        build_mask_mod = model.encoder._build_mask_mod

        def record_runtime_values(length):
            observed_runtime_values.append((model.encoder.causal_tail_len, model.encoder.causal_tail_block_size))
            return build_mask_mod(length)

        with (
            patch.object(model.encoder, "_build_mask_mod", side_effect=record_runtime_values),
            patch("nemo.collections.asr.models.sortformer_diar_models.random.random", return_value=0.0),
            patch(
                "nemo.collections.asr.models.sortformer_diar_models.random.randint",
                side_effect=random_values,
            ) as randint_mock,
            torch.no_grad(),
        ):
            model.forward_streaming(torch.randn(1, 128, 32), torch.tensor([32]))

        assert observed_runtime_values
        assert set(observed_runtime_values) == {expected_runtime_values}
        assert model.encoder.causal_tail_len == 0
        assert model.encoder.causal_tail_block_size == 1
        assert [mock_call.args for mock_call in randint_mock.call_args_list] == expected_randint_args

    @pytest.mark.unit
    def test_causal_tail_runtime_state_is_reset_when_streaming_step_raises(self):
        model = _create_sortformer_model(
            frontend_encoder="transformer",
            streaming_mode=True,
            max_causal_tail_len=5,
            min_causal_tail_block_size=2,
            max_causal_tail_block_size=4,
        ).train()
        streaming_chunk = (
            0,
            torch.randn(1, 8, 128),
            torch.tensor([8]),
            0,
            0,
        )

        def fail_streaming_step(**kwargs):
            assert model.encoder.causal_tail_len == 4
            assert model.encoder.causal_tail_block_size == 3
            raise RuntimeError("injected streaming failure")

        with (
            patch("nemo.collections.asr.models.sortformer_diar_models.random.random", return_value=0.0),
            patch("nemo.collections.asr.models.sortformer_diar_models.random.randint", side_effect=(4, 3)),
            patch.object(model.sortformer_modules, "streaming_feat_loader", return_value=[streaming_chunk]),
            patch.object(model, "forward_streaming_step", side_effect=fail_streaming_step),
            pytest.raises(RuntimeError, match="injected streaming failure"),
        ):
            model.forward_streaming(torch.randn(1, 128, 8), torch.tensor([8]))

        assert model.encoder.causal_tail_len == 0
        assert model.encoder.causal_tail_block_size == 1

    @pytest.mark.unit
    def test_causal_tail_accepts_default_full_attention_mode(self):
        model = _create_sortformer_model(
            frontend_encoder="transformer",
            streaming_mode=True,
            max_causal_tail_len=5,
            encoder_attn_mode=None,
        )

        assert model.encoder.attn_mode == "full"
        assert model.encoder.causal_tail_block_size == 1

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("settings", "exception_type", "error_match"),
        [
            ({"max_causal_tail_len": True}, TypeError, "max_causal_tail_len must be a non-negative integer"),
            ({"max_causal_tail_len": 1.5}, TypeError, "max_causal_tail_len must be a non-negative integer"),
            ({"max_causal_tail_len": -1}, ValueError, "max_causal_tail_len must be non-negative"),
            ({"causal_tail_prob": True}, TypeError, "causal_tail_prob must be a real number"),
            ({"causal_tail_prob": "0.5"}, TypeError, "causal_tail_prob must be a real number"),
            ({"causal_tail_prob": -0.1}, ValueError, "causal_tail_prob must be in"),
            ({"causal_tail_prob": 1.1}, ValueError, "causal_tail_prob must be in"),
            (
                {"min_causal_tail_block_size": True},
                TypeError,
                "min_causal_tail_block_size must be a positive integer",
            ),
            (
                {"max_causal_tail_block_size": 1.5},
                TypeError,
                "max_causal_tail_block_size must be a positive integer",
            ),
            (
                {"min_causal_tail_block_size": 0},
                ValueError,
                "min_causal_tail_block_size must be at least 1",
            ),
            (
                {"min_causal_tail_block_size": 3, "max_causal_tail_block_size": 2},
                ValueError,
                "max_causal_tail_block_size must be greater than or equal",
            ),
            (
                {"max_causal_tail_len": 5, "max_causal_tail_block_size": 6},
                ValueError,
                "max_causal_tail_block_size must be less than or equal to max_causal_tail_len",
            ),
        ],
    )
    def test_invalid_causal_tail_policy_is_rejected(self, settings, exception_type, error_match):
        with pytest.raises(exception_type, match=error_match):
            _create_sortformer_model(frontend_encoder="transformer", **settings)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("frontend_encoder", "streaming_mode", "error_match"),
        [
            ("transformer", False, "requires streaming_mode=True"),
            ("conformer", True, "requires an encoder that exposes causal_tail_len"),
            ("streaming_transformer", True, "requires TransformerEncoder with attn_mode='full'"),
        ],
    )
    def test_incompatible_causal_tail_configuration_is_rejected(self, frontend_encoder, streaming_mode, error_match):
        with pytest.raises(ValueError, match=error_match):
            _create_sortformer_model(
                frontend_encoder=frontend_encoder,
                streaming_mode=streaming_mode,
                max_causal_tail_len=4,
            )

    @pytest.mark.unit
    def test_disabled_causal_tail_does_not_modify_incompatible_encoder(self):
        model = _create_sortformer_model(frontend_encoder="conformer", streaming_mode=True).eval()

        assert not hasattr(model.encoder, "causal_tail_len")
        assert not hasattr(model.encoder, "causal_tail_block_size")
        with (
            patch.object(model.sortformer_modules, "streaming_feat_loader", return_value=[]),
            torch.no_grad(),
        ):
            model.forward_streaming(torch.randn(1, 80, 8), torch.tensor([8]))

        assert not hasattr(model.encoder, "causal_tail_len")
        assert not hasattr(model.encoder, "causal_tail_block_size")

    @pytest.mark.unit
    def test_constructor(self, sortformer_model):
        sortformer_model.streaming_mode = True
        sortformer_diar_model = sortformer_model.train()
        confdict = sortformer_diar_model.to_config_dict()
        instance2 = SortformerEncLabelModel.from_config_dict(confdict)
        assert isinstance(instance2, SortformerEncLabelModel)

    @pytest.mark.unit
    @pytest.mark.parametrize("async_streaming", [False, True])
    def test_high_resolution_streaming_returns_aligned_logits(self, async_streaming):
        model = _create_sortformer_model(high_resolution=True, activity_weight=0.25).eval()
        model.streaming_mode = True
        model.async_streaming = async_streaming
        audio = torch.randn(2, 4000)
        audio_lengths = torch.tensor([4000, 3200])

        with torch.no_grad():
            preds, logits, activity_logits = model(audio, audio_lengths, return_logits=True)

        valid_mask = preds.sum(dim=-1) > 0
        assert preds.shape == logits.shape
        assert activity_logits.shape == (*preds.shape[:2], 3)
        torch.testing.assert_close(torch.sigmoid(logits[valid_mask]), preds[valid_mask])

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "stacking_factor, feature_shape, input_lengths, expected_encoded_lengths",
        [(8, (2, 120, 80), (120, 91), (15, 12))],
    )
    def test_call_pre_encode_with_feature_stacking(
        self, sortformer_model, stacking_factor, feature_shape, input_lengths, expected_encoded_lengths
    ):
        sortformer_model.encoder.pre_encode = FeatureStacking(
            subsampling_factor=stacking_factor,
            feat_in=feature_shape[-1],
            feat_out=sortformer_model._cfg.model_defaults.fc_d_model,
        )
        features = torch.randn(feature_shape)
        lengths = torch.tensor(input_lengths)

        encoded, encoded_lengths = sortformer_model._call_pre_encode(features, lengths)

        assert encoded.shape == (
            feature_shape[0],
            max(expected_encoded_lengths),
            sortformer_model._cfg.model_defaults.fc_d_model,
        )
        assert torch.equal(encoded_lengths, torch.tensor(expected_encoded_lengths))

    @pytest.mark.unit
    @pytest.mark.parametrize("pre_encode_kind", ["feature_stacking", "conv_subsampling"])
    @pytest.mark.parametrize("diar_right_context", [0, 1, 3, 5, 13])
    @pytest.mark.parametrize("has_real_future_context", [True, False])
    def test_streaming_right_context_is_not_committed(
        self, pre_encode_kind, diar_right_context, has_real_future_context
    ):
        frontend_encoder = "transformer" if pre_encode_kind == "feature_stacking" else "conformer"
        model = _create_sortformer_model(frontend_encoder=frontend_encoder).eval()
        model.sortformer_modules.fifo_len = 100
        model.sortformer_modules.spkcache_update_period = 14
        streaming_state = model.sortformer_modules.init_streaming_state(batch_size=1)
        total_preds = torch.zeros(1, 0, model.sortformer_modules.n_spk)
        input_frames = (14 + diar_right_context) * model.encoder.subsampling_factor
        processed_signal = torch.randn(1, input_frames, model.encoder._feat_in)
        real_future_context = diar_right_context if has_real_future_context else 0
        processed_signal_length = torch.tensor([(14 + real_future_context) * model.encoder.subsampling_factor])
        right_offset = diar_right_context * model.encoder.subsampling_factor

        with torch.no_grad():
            for expected_length in (14, 28):
                streaming_state, total_preds = model.forward_streaming_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    streaming_state=streaming_state,
                    total_preds=total_preds,
                    right_offset=right_offset,
                )
                assert total_preds.shape[1] == expected_length
                assert streaming_state.fifo.shape[1] == expected_length

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "batch_size, spkcache_len, fifo_len, chunk_len, chunk_left_context, chunk_right_context, "
            "expected_chunk_frames, expected_pre_encode_frames"
        ),
        [(2, 5, 7, 3, 1, 2, 48, 6)],
    )
    def test_streaming_input_examples_match_model_dimensions(
        self,
        sortformer_model,
        batch_size,
        spkcache_len,
        fifo_len,
        chunk_len,
        chunk_left_context,
        chunk_right_context,
        expected_chunk_frames,
        expected_pre_encode_frames,
    ):
        sortformer_model.sortformer_modules.spkcache_len = spkcache_len
        sortformer_model.sortformer_modules.fifo_len = fifo_len
        sortformer_model.sortformer_modules.chunk_len = chunk_len
        sortformer_model.sortformer_modules.chunk_left_context = chunk_left_context
        sortformer_model.sortformer_modules.chunk_right_context = chunk_right_context

        chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths = (
            sortformer_model.streaming_input_examples(batch_size=batch_size)
        )

        chunk_frames = (
            chunk_left_context + chunk_len + chunk_right_context
        ) * sortformer_model.encoder.subsampling_factor
        assert chunk_frames == expected_chunk_frames
        assert chunk.shape == (batch_size, chunk_frames, sortformer_model.encoder._feat_in)
        assert chunk_lengths.tolist() == [chunk_frames] * batch_size
        assert spkcache.shape == (batch_size, spkcache_len, sortformer_model.sortformer_modules.fc_d_model)
        assert fifo.shape == (batch_size, fifo_len, sortformer_model.sortformer_modules.fc_d_model)
        assert torch.all(spkcache_lengths <= spkcache.shape[1])
        assert torch.all(fifo_lengths <= fifo.shape[1])
        with torch.no_grad():
            chunk_pre_encode_embs, chunk_pre_encode_lengths = sortformer_model._call_pre_encode(chunk, chunk_lengths)
        pre_encode_frames = chunk_left_context + chunk_len + chunk_right_context
        assert pre_encode_frames == expected_pre_encode_frames
        assert chunk_pre_encode_embs.shape[1] == pre_encode_frames
        assert chunk_pre_encode_lengths.tolist() == [pre_encode_frames] * batch_size

    @pytest.mark.unit
    @pytest.mark.parametrize("output_filename, mocked_export_result", [("model.onnx", "exported")])
    def test_streaming_export_accepts_explicit_input_example(
        self, sortformer_model, output_filename, mocked_export_result
    ):
        input_example = tuple(torch.empty(0) for _ in sortformer_model.input_names)
        with patch.object(sortformer_model, "export", return_value=mocked_export_result) as export_mock:
            result = sortformer_model.streaming_export(output_filename, input_example=input_example)

        assert result == mocked_export_result
        export_mock.assert_called_once_with(output_filename, input_example=input_example)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "output_filename, batch_size, mocked_export_result",
        [("model.onnx", 2, "exported")],
    )
    def test_streaming_export_uses_model_sized_defaults(
        self, sortformer_model, output_filename, batch_size, mocked_export_result
    ):
        input_example = tuple(torch.empty(0) for _ in sortformer_model.input_names)
        with (
            patch.object(sortformer_model, "streaming_input_examples", return_value=input_example) as examples_mock,
            patch.object(sortformer_model, "export", return_value=mocked_export_result) as export_mock,
        ):
            result = sortformer_model.streaming_export(output_filename, batch_size=batch_size)

        assert result == mocked_export_result
        examples_mock.assert_called_once_with(batch_size=batch_size)
        export_mock.assert_called_once_with(output_filename, input_example=input_example)

    @pytest.mark.unit
    @pytest.mark.parametrize("async_streaming", [False, True])
    def test_inference_profiler_reports_streaming_sections(self, sortformer_model, async_streaming):
        sortformer_model.streaming_mode = True
        sortformer_model.async_streaming = async_streaming
        sortformer_model.eval()
        profiler = InferenceProfiler(sortformer_model)
        profiler.install()
        profiler.install()

        with torch.no_grad():
            sortformer_model(torch.randn(2, 8000), torch.tensor([8000, 6000]))
        profiler.log_summary(audio_duration=1.0)

        expected_sections = {
            "streaming_step",
            "pre_encode",
            "state_concat",
            "frontend_encoder",
            "forward_infer",
            "prediction_mask",
            "state_update",
        }
        assert profiler.forward_calls == 1
        assert expected_sections <= profiler.section_times.keys()
        assert all(profiler.section_times[section] > 0 for section in expected_sections)
        assert all(profiler.section_calls[section] > 0 for section in expected_sections)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "batch_size, spkcache_len, fifo_len, chunk_len, chunk_left_context, chunk_right_context, "
            "processed_signal_lengths, expected_frontend_shape"
        ),
        [(2, 5, 7, 3, 1, 2, (48, 24), (2, 18, 32))],
    )
    def test_async_streaming_can_pad_encoder_input_to_max_length(
        self,
        sortformer_model,
        batch_size,
        spkcache_len,
        fifo_len,
        chunk_len,
        chunk_left_context,
        chunk_right_context,
        processed_signal_lengths,
        expected_frontend_shape,
    ):
        sortformer_model.streaming_mode = True
        sortformer_model.async_streaming = True
        sortformer_model.async_pad_to_max = True
        sortformer_model.sortformer_modules.spkcache_len = spkcache_len
        sortformer_model.sortformer_modules.fifo_len = fifo_len
        sortformer_model.sortformer_modules.chunk_len = chunk_len
        sortformer_model.sortformer_modules.chunk_left_context = chunk_left_context
        sortformer_model.sortformer_modules.chunk_right_context = chunk_right_context
        sortformer_model.eval()

        streaming_state = sortformer_model.sortformer_modules.init_streaming_state(
            batch_size=batch_size, async_streaming=True
        )
        frontend_inputs = []
        frontend_encoder = sortformer_model.frontend_encoder

        def capture_frontend_input(*args, **kwargs):
            frontend_inputs.append(kwargs["processed_signal"].shape)
            return frontend_encoder(*args, **kwargs)

        sortformer_model.frontend_encoder = capture_frontend_input
        with torch.no_grad():
            sortformer_model.forward_streaming_step(
                processed_signal=torch.randn(
                    batch_size, max(processed_signal_lengths), sortformer_model.encoder._feat_in
                ),
                processed_signal_length=torch.tensor(processed_signal_lengths),
                streaming_state=streaming_state,
                total_preds=torch.zeros(batch_size, 0, sortformer_model._cfg.max_num_of_spks),
            )

        assert frontend_inputs == [torch.Size(expected_frontend_shape)]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "chunk_len, audio_shape, audio_lengths",
        [(6, (2, 16000), (16000, 5000))],
    )
    def test_async_streaming_flushes_fifo_for_finalized_rows(
        self, sortformer_model, chunk_len, audio_shape, audio_lengths
    ):
        sortformer_model.streaming_mode = True
        sortformer_model.async_streaming = True
        sortformer_model.sortformer_modules.chunk_len = chunk_len
        sortformer_model.eval()
        streaming_update_async = sortformer_model.sortformer_modules.streaming_update_async
        updates = []

        def capture_streaming_update(**kwargs):
            state, chunk_preds = streaming_update_async(**kwargs)
            max_chunk_len = kwargs["chunk"].shape[1] - kwargs["lc"] - kwargs["rc"]
            finalized = (kwargs["chunk_lengths"] - kwargs["lc"]).clamp(min=0, max=max_chunk_len) == 0
            updates.append((finalized, state.fifo_lengths.clone()))
            return state, chunk_preds

        sortformer_model.sortformer_modules.streaming_update_async = capture_streaming_update
        with torch.no_grad():
            sortformer_model(torch.randn(audio_shape), torch.tensor(audio_lengths))

        assert updates
        finalized_masks = []
        for finalized, fifo_lengths in updates:
            finalized_masks.append(finalized)
            assert torch.count_nonzero(fifo_lengths[finalized]) == 0
        assert torch.stack(finalized_masks)[:, 1].any()

    @pytest.mark.unit
    @pytest.mark.parametrize("streaming_mode", [False, True])
    def test_spec_augment_is_applied_once_in_forward(self, sortformer_model, streaming_mode):
        sortformer_model.streaming_mode = streaming_mode
        sortformer_model.train()
        spec_augmentation = RecordingSpecAugment()
        sortformer_model.spec_augmentation = spec_augmentation
        audio = torch.randn(1, 8000)
        audio_length = torch.tensor([8000])

        sortformer_model(audio, audio_length)

        assert len(spec_augmentation.input_shapes) == 1
        assert spec_augmentation.input_shapes[0][:2] == (1, 80)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (2, 1),  # Example 1
            (1, 2),  # Example 2
        ],
    )
    def test_forward_infer(self, sortformer_model, batch_size, sample_len):
        sortformer_model.streaming_mode = True
        sortformer_diar_model = sortformer_model.eval()
        confdict = sortformer_diar_model.to_config_dict()
        sampling_rate = confdict['preprocessor']['sample_rate']
        input_signal = torch.randn(size=(batch_size, sample_len * sampling_rate))
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        with torch.no_grad():
            # batch size 1
            preds_list = []
            for i in range(input_signal.size(0)):
                preds = sortformer_diar_model.forward(input_signal[i : i + 1], input_signal_length[i : i + 1])
                preds_list.append(preds)
            preds_instance = torch.cat(preds_list, 0)

            # batch size 4
            preds_batch = sortformer_diar_model.forward(input_signal, input_signal_length)
        assert preds_instance.shape == preds_batch.shape

        diff = torch.mean(torch.abs(preds_instance - preds_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(preds_instance - preds_batch))
        assert diff <= 1e-6


class TestSortformerEncLabelModelHighResolution:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("batch_size, max_chunk_len, num_speakers, left_context, spkcache_lengths, fifo_lengths, " "chunk_lengths"),
        [(3, 3, 2, 1, (0, 2, 4), (1, 3, 0), (3, 1, 0))],
    )
    def test_async_high_resolution_chunk_extraction_is_vectorized_and_profiled(
        self,
        batch_size,
        max_chunk_len,
        num_speakers,
        left_context,
        spkcache_lengths,
        fifo_lengths,
        chunk_lengths,
    ):
        model = _create_sortformer_model(high_resolution=True).eval()
        upsample_factor = model.upsample_factor
        high_resolution_preds = torch.arange(
            batch_size * 10 * upsample_factor * num_speakers, dtype=torch.float32
        ).reshape(batch_size, 10 * upsample_factor, num_speakers)
        spkcache_lengths = torch.tensor(spkcache_lengths)
        fifo_lengths = torch.tensor(fifo_lengths)
        chunk_lengths = torch.tensor(chunk_lengths)
        expected = high_resolution_preds.new_zeros((batch_size, max_chunk_len * upsample_factor, num_speakers))
        for batch_idx in range(batch_size):
            start = (spkcache_lengths[batch_idx] + fifo_lengths[batch_idx] + left_context) * upsample_factor
            length = chunk_lengths[batch_idx] * upsample_factor
            expected[batch_idx, :length] = high_resolution_preds[batch_idx, start : start + length]
        profiler = InferenceProfiler(model)
        profiler.install()

        actual = model._extract_async_high_resolution_chunk_preds(
            high_resolution_preds=high_resolution_preds,
            spkcache_lengths=spkcache_lengths,
            fifo_lengths=fifo_lengths,
            chunk_lengths=chunk_lengths,
            max_chunk_len=max_chunk_len,
            lc_enc=left_context,
        )

        torch.testing.assert_close(actual, expected)
        assert actual.is_contiguous()
        assert profiler.section_calls["high_resolution_extract"] == 1
        assert profiler.section_times["high_resolution_extract"] > 0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "activity_weight, pred_frames, logit_frames, activity_shape, error_match",
        [
            (0.0, 3, None, None, "total_logits is required"),
            (0.0, 3, 2, None, "total_logits and total_preds must have the same shape"),
            (0.2, 3, 3, None, "total_activity_logits is required"),
            (0.2, 3, 3, (1, 2, 3), "total_activity_logits must align"),
            (0.2, 3, 3, (1, 3, 2), "total_activity_logits must align"),
        ],
    )
    def test_streaming_step_validates_cumulative_logits(
        self, activity_weight, pred_frames, logit_frames, activity_shape, error_match
    ):
        model = _create_sortformer_model(high_resolution=True, activity_weight=activity_weight).eval()
        num_speakers = model.sortformer_modules.n_spk
        total_preds = torch.zeros(1, pred_frames, num_speakers)
        total_logits = None if logit_frames is None else torch.zeros(1, logit_frames, num_speakers)
        total_activity_logits = None if activity_shape is None else torch.zeros(activity_shape)

        with pytest.raises(ValueError, match=error_match):
            model.forward_streaming_step(
                processed_signal=torch.zeros(1, 1, model.encoder._feat_in),
                processed_signal_length=torch.tensor([1]),
                streaming_state=model.sortformer_modules.init_streaming_state(batch_size=1),
                total_preds=total_preds,
                return_logits=True,
                total_logits=total_logits,
                total_activity_logits=total_activity_logits,
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "speaker_permutation, num_input_frames",
        [((2, 0, 3, 1), 120), ((0, 1, 2, 3), 96)],
    )
    def test_streaming_step_accumulates_permuted_and_sliced_logits(self, speaker_permutation, num_input_frames):
        model = _create_sortformer_model(high_resolution=True, activity_weight=0.2).eval()
        streaming_state = model.sortformer_modules.init_streaming_state(batch_size=1)
        streaming_state.spk_perm = torch.tensor([speaker_permutation])
        total_preds = torch.zeros(1, 0, model.sortformer_modules.n_spk)
        processed_signal = torch.randn(1, num_input_frames, model.encoder._feat_in)
        processed_signal_length = torch.tensor([num_input_frames])
        captured = {}
        forward_infer = model.forward_infer

        def capture_forward_infer(*args, **kwargs):
            outputs = forward_infer(*args, **kwargs)
            captured["logits"] = outputs[1]
            captured["activity_logits"] = outputs[2]
            return outputs

        with torch.no_grad(), patch.object(model, "forward_infer", side_effect=capture_forward_infer):
            streaming_state, total_preds, total_logits, total_activity_logits = model.forward_streaming_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                streaming_state=streaming_state,
                total_preds=total_preds,
                return_logits=True,
            )
            first_logits = total_logits.clone()

            inverse_permutation = torch.argsort(torch.tensor(speaker_permutation))
            expected_logits = captured["logits"][:, : total_logits.shape[1], inverse_permutation]
            expected_activity_logits = captured["activity_logits"][:, : total_activity_logits.shape[1]]
            torch.testing.assert_close(total_logits, expected_logits)
            torch.testing.assert_close(total_activity_logits, expected_activity_logits)

            _, total_preds, total_logits, total_activity_logits = model.forward_streaming_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                streaming_state=streaming_state,
                total_preds=total_preds,
                total_logits=total_logits,
                total_activity_logits=total_activity_logits,
                return_logits=True,
            )

        assert total_logits.shape == total_preds.shape
        assert total_activity_logits.shape == (*total_preds.shape[:2], 3)
        torch.testing.assert_close(total_logits[:, : first_logits.shape[1]], first_logits)
        torch.testing.assert_close(torch.sigmoid(total_logits), total_preds)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "embedding_batch_size, embedding_frame_count, embedding_lengths",
        [(2, 5, (5, 4))],
    )
    def test_non_strict_warm_start_from_legacy_state_dict(
        self, embedding_batch_size, embedding_frame_count, embedding_lengths
    ):
        low_resolution_model = _create_sortformer_model().eval()
        high_resolution_model = _create_sortformer_model(high_resolution=True).eval()

        load_result = high_resolution_model.load_state_dict(low_resolution_model.state_dict(), strict=False)
        emb_seq = torch.randn(
            embedding_batch_size,
            embedding_frame_count,
            low_resolution_model._cfg.model_defaults.tf_d_model,
        )
        emb_seq_length = torch.tensor(embedding_lengths)
        with torch.no_grad():
            low_resolution_preds = low_resolution_model.forward_infer(emb_seq, emb_seq_length)
            high_resolution_preds = high_resolution_model.forward_infer(emb_seq, emb_seq_length)

        assert set(load_result.missing_keys) == {
            "sortformer_modules.subpixel_upsample.weight",
            "sortformer_modules.subpixel_upsample.bias",
        }
        assert not load_result.unexpected_keys
        assert torch.allclose(
            high_resolution_preds,
            low_resolution_preds.repeat_interleave(high_resolution_model.upsample_factor, dim=1),
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "high_resolution, audio_shape, audio_lengths",
        [(True, (2, 16000), (16000, 12000))],
    )
    def test_constructor_and_exact_output_length(self, high_resolution, audio_shape, audio_lengths):
        model = _create_sortformer_model(high_resolution=high_resolution).eval()
        audio = torch.randn(audio_shape)
        lengths = torch.tensor(audio_lengths, dtype=torch.long)

        with torch.no_grad():
            _, feature_lengths = model.process_signal(audio, lengths)
            preds = model(audio, lengths)

        assert model.high_resolution
        assert model.upsample_factor == model.encoder.subsampling_factor
        assert model.output_subsampling_factor == 1
        assert preds.shape[1] == feature_lengths.max()
        assert torch.count_nonzero(preds[1, feature_lengths[1] :]) == 0

    @pytest.mark.unit
    @pytest.mark.parametrize("output_subsampling_factor", [1, 2, 3, 8, 16])
    def test_forward_returns_configured_output_resolution(self, output_subsampling_factor):
        model = _create_sortformer_model(
            high_resolution=True,
            output_subsampling_factor=output_subsampling_factor,
        ).eval()
        audio = torch.randn(2, 8000)
        lengths = torch.tensor([8000, 6400], dtype=torch.long)

        with torch.no_grad():
            _, feature_lengths = model.process_signal(audio, lengths)
            preds = model(audio, lengths)

        expected_max_length = math.ceil(feature_lengths.max().item() / output_subsampling_factor)
        second_length = math.ceil(feature_lengths[1].item() / output_subsampling_factor)
        assert preds.shape[1] == expected_max_length
        assert torch.count_nonzero(preds[1, second_length:]) == 0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "high_resolution, requested_output_factor, expected_output_factor, expected_upsample_factor",
        [(False, 3, 8, 1)],
    )
    def test_low_resolution_overrides_output_subsampling_factor(
        self,
        high_resolution,
        requested_output_factor,
        expected_output_factor,
        expected_upsample_factor,
    ):
        model = _create_sortformer_model(
            high_resolution=high_resolution,
            output_subsampling_factor=requested_output_factor,
        )

        assert model.output_subsampling_factor == model.encoder.subsampling_factor == expected_output_factor
        assert model.upsample_factor == expected_upsample_factor

    @pytest.mark.unit
    @pytest.mark.parametrize("output_subsampling_factor", [16, 24])
    def test_low_resolution_forward_can_downsample_further(self, output_subsampling_factor):
        model = _create_sortformer_model(
            high_resolution=False,
            output_subsampling_factor=output_subsampling_factor,
        ).eval()
        audio = torch.randn(2, 8000)
        lengths = torch.tensor([8000, 6400], dtype=torch.long)

        with torch.no_grad():
            _, feature_lengths = model.process_signal(audio, lengths)
            preds = model(audio, lengths)

        assert preds.shape[1] == math.ceil(feature_lengths.max().item() / output_subsampling_factor)
        assert model.sortformer_modules.subpixel_upsample is None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "high_resolution, requested_factor, expected_factor",
        [(True, 3, 3), (False, 3, 8), (False, 16, 16), (True, None, 1)],
    )
    def test_inference_output_subsampling_override(self, high_resolution, requested_factor, expected_factor):
        model = _create_sortformer_model(high_resolution=high_resolution)

        result = configure_output_subsampling_factor(model, requested_factor)

        assert result == expected_factor
        assert model.output_subsampling_factor == expected_factor
        assert model._cfg.output_subsampling_factor == expected_factor

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "high_resolution, invalid_factor, error_match",
        [(True, 0, "output_subsampling_factor must be a positive integer")],
    )
    def test_inference_output_subsampling_override_rejects_invalid_factor(
        self, high_resolution, invalid_factor, error_match
    ):
        model = _create_sortformer_model(high_resolution=high_resolution)

        with pytest.raises(ValueError, match=error_match):
            configure_output_subsampling_factor(model, invalid_factor)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_path, pretrained_name, loader_name, expected_kwargs",
        [
            (
                "/models/diarizer.ckpt",
                None,
                "load_from_checkpoint",
                {"checkpoint_path": "/models/diarizer.ckpt", "strict": False},
            ),
            (
                "/models/diarizer.nemo",
                None,
                "restore_from",
                {"restore_path": "/models/diarizer.nemo"},
            ),
            (
                None,
                "nvidia/diar_sortformer_4spk-v1",
                "from_pretrained",
                {"model_name": "nvidia/diar_sortformer_4spk-v1"},
            ),
        ],
        ids=["checkpoint", "nemo", "hugging-face"],
    )
    def test_load_diarization_model_selects_configured_source(
        self, model_path, pretrained_name, loader_name, expected_kwargs
    ):
        map_location = torch.device("cpu")
        expected_model = object()
        with patch.object(SortformerEncLabelModel, loader_name, return_value=expected_model) as loader:
            actual_model = load_diarization_model(
                model_path=model_path,
                pretrained_name=pretrained_name,
                map_location=map_location,
            )

        assert actual_model is expected_model
        loader.assert_called_once_with(**expected_kwargs, map_location=map_location)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_path, pretrained_name",
        [(None, None), ("/models/diarizer.nemo", "nvidia/diar_sortformer_4spk-v1")],
        ids=["missing", "ambiguous"],
    )
    def test_load_diarization_model_requires_exactly_one_source(self, model_path, pretrained_name):
        with pytest.raises(ValueError, match="Specify exactly one"):
            load_diarization_model(
                model_path=model_path,
                pretrained_name=pretrained_name,
                map_location=torch.device("cpu"),
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "model_filename, manifest_filename, cache_filename, output_subsampling_factor, expected_model_id, "
            "expected_tensor_filename"
        ),
        [("model.nemo", "sample.json", "custom/predictions.pt", 8, "model_sf8", "sample")],
    )
    def test_explicit_prediction_tensor_path_avoids_automatic_directory(
        self,
        tmp_path,
        model_filename,
        manifest_filename,
        cache_filename,
        output_subsampling_factor,
        expected_model_id,
        expected_tensor_filename,
    ):
        explicit_path = tmp_path / cache_filename
        cfg = SimpleNamespace(
            model_path=str(tmp_path / model_filename),
            dataset_manifest=str(tmp_path / manifest_filename),
            output_subsampling_factor=output_subsampling_factor,
            out_preds_tensors=str(explicit_path),
        )

        tensor_path, model_id, tensor_filename = get_tensor_path(cfg)

        assert tensor_path == str(explicit_path.absolute())
        assert model_id == expected_model_id
        assert tensor_filename == expected_tensor_filename
        assert not (tmp_path / "pred_tensors").exists()

    @pytest.mark.unit
    def test_pretrained_model_prediction_cache_identity(self, tmp_path):
        manifest_path = tmp_path / "sample.json"
        manifest_path.write_text("{}\n")
        cfg = SimpleNamespace(
            model_path=None,
            pretrained_name="nvidia/diar_sortformer_4spk-v1",
            dataset_manifest=str(manifest_path),
            output_subsampling_factor=8,
            precision="32",
            presort_manifest=True,
            async_streaming=False,
            async_pad_to_max=False,
            async_desync_updates=False,
            chunk_len=6,
            chunk_left_context=0,
            chunk_right_context=7,
            spkcache_len=0,
            spkcache_update_period=144,
            fifo_len=188,
        )
        model = _create_sortformer_model()

        metadata = get_prediction_cache_metadata(cfg, model, {"sample": {}})

        assert metadata["model_path"] == cfg.pretrained_name
        assert metadata["model_size"] is None
        assert metadata["model_mtime_ns"] is None

    @pytest.mark.unit
    def test_pretrained_model_prediction_tensor_path_uses_repository_name(self, tmp_path):
        cfg = SimpleNamespace(
            model_path=None,
            pretrained_name="nvidia/diar_sortformer_4spk-v1",
            dataset_manifest=str(tmp_path / "sample.json"),
            output_subsampling_factor=8,
            out_preds_tensors=str(tmp_path / "predictions.pt"),
        )

        _, model_id, tensor_filename = get_tensor_path(cfg)

        assert model_id == "diar_sortformer_4spk-v1_sf8"
        assert tensor_filename == "sample"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_filename, manifest_filename, output_subsampling_factor",
        [("model.nemo", "sample.json", 8)],
    )
    def test_prediction_tensor_cache_is_disabled_without_explicit_path(
        self, tmp_path, model_filename, manifest_filename, output_subsampling_factor
    ):
        cfg = SimpleNamespace(
            model_path=str(tmp_path / model_filename),
            dataset_manifest=str(tmp_path / manifest_filename),
            output_subsampling_factor=output_subsampling_factor,
            out_preds_tensors=None,
        )

        tensor_path, _, _ = get_tensor_path(cfg)

        assert tensor_path is None
        assert not (tmp_path / "pred_tensors").exists()

    @pytest.mark.unit
    @pytest.mark.parametrize("output_subsampling_factor", [0, -1, 1.5, True])
    def test_output_subsampling_factor_must_be_a_positive_integer(self, output_subsampling_factor):
        with pytest.raises(ValueError, match="output_subsampling_factor must be a positive integer"):
            _create_sortformer_model(
                high_resolution=True,
                output_subsampling_factor=output_subsampling_factor,
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "audio_shape, audio_lengths, learning_rate, target_length_trim",
        [((2, 8000), (8000, 6400), 1e-3, 5)],
    )
    def test_high_resolution_training_loss_is_finite_and_updates_upsampler(
        self, audio_shape, audio_lengths, learning_rate, target_length_trim
    ):
        model = _create_sortformer_model(high_resolution=True).train()
        model._optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        audio = torch.randn(audio_shape)
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
        preds = model(audio, audio_lengths)
        targets = (torch.rand_like(preds) > 0.5).to(preds.dtype)
        target_lens = torch.tensor([preds.shape[1], preds.shape[1] - target_length_trim])

        metrics = model._get_aux_train_evaluations(preds, targets, target_lens)
        metrics["loss"].backward()

        assert torch.isfinite(metrics["loss"])
        assert model.sortformer_modules.subpixel_upsample.weight.grad is not None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "autocast_dtype, audio_shape, audio_lengths, target_length_trim",
        [(torch.bfloat16, (2, 4000), (4000, 3200), 3)],
    )
    def test_high_resolution_bfloat16_mixed_loss_is_finite(
        self, autocast_dtype, audio_shape, audio_lengths, target_length_trim
    ):
        model = _create_sortformer_model(high_resolution=True).eval()
        audio = torch.randn(audio_shape)
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

        with torch.autocast(device_type="cpu", dtype=autocast_dtype):
            preds = model(audio, audio_lengths)
            targets = (torch.rand_like(preds) > 0.5).to(preds.dtype)
            target_lens = torch.tensor([preds.shape[1], preds.shape[1] - target_length_trim])
            metrics = model._get_aux_validation_evaluations(preds, targets, target_lens)

        assert torch.isfinite(metrics["val_loss"])

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "reference_speaker_count, prediction_stream_count, expected_precision, expected_recall",
        [(5, 4, 1.0, 0.8)],
    )
    def test_test_metrics_count_speakers_beyond_model_capacity_as_false_negatives(
        self,
        reference_speaker_count,
        prediction_stream_count,
        expected_precision,
        expected_recall,
    ):
        model = _create_sortformer_model().eval()
        targets = torch.eye(reference_speaker_count).unsqueeze(0)
        preds = targets[:, :, :prediction_stream_count]
        model.batch_f1_accs_list = []
        model.batch_precision_list = []
        model.batch_recall_list = []
        model.batch_f1_accs_ats_list = []

        model._get_aux_test_batch_evaluations(
            batch_idx=0,
            preds=preds,
            targets=targets,
            target_lens=torch.tensor([reference_speaker_count]),
        )

        assert model.batch_precision_list[0].item() == pytest.approx(expected_precision)
        assert model.batch_recall_list[0].item() == pytest.approx(expected_recall)

    @pytest.mark.unit
    @pytest.mark.parametrize("output_subsampling_factor", [1, 3, 16])
    def test_legacy_dataloader_uses_high_resolution_targets(self, output_subsampling_factor):
        model = _create_sortformer_model(
            high_resolution=True,
            output_subsampling_factor=output_subsampling_factor,
        )
        dataset = SimpleNamespace(collection=[], eesd_train_collate_fn=lambda batch: batch)
        config = DictConfig(
            {
                "manifest_filepath": "unused.json",
                "sample_rate": 16000,
                "session_len_sec": 1,
                "num_spks": 4,
                "soft_targets": False,
                "batch_size": 1,
                "num_workers": 0,
                "use_lhotse": False,
            }
        )

        with patch(
            "nemo.collections.asr.models.sortformer_diar_models.AudioToSpeechE2ESpkDiarDataset",
            return_value=dataset,
        ) as dataset_constructor:
            model._SortformerEncLabelModel__setup_dataloader_from_config(config)

        assert dataset_constructor.call_args.kwargs["subsampling_factor"] == output_subsampling_factor
        assert dataset_constructor.call_args.kwargs["soft_label_thres"] == 0.5

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "is_training, limit_train_batches, expect_oversampling",
        [(True, 3, True), (True, 1.0, False), (False, 3, False)],
        ids=["fixed-training-limit", "fractional-training-limit", "validation-loader"],
    )
    def test_legacy_dataloader_derives_training_epoch_size(
        self,
        is_training: bool,
        limit_train_batches: Union[int, float],
        expect_oversampling: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify fixed oversampling is limited to integer-sized training epochs."""
        monkeypatch.setenv("PL_GLOBAL_SEED", "1234")
        model = _create_sortformer_model()
        model._trainer = SimpleNamespace(global_rank=1, limit_train_batches=limit_train_batches)
        model.world_size = 2
        dataset = torch.utils.data.TensorDataset(torch.arange(5))
        dataset.collection = list(range(5))
        dataset.eesd_train_collate_fn = lambda batch: batch
        config = DictConfig(
            {
                "manifest_filepath": "unused.json",
                "sample_rate": 16000,
                "soft_label_thres": 0.5,
                "session_len_sec": 1,
                "num_spks": 4,
                "soft_targets": False,
                "batch_size": 2,
                "num_workers": 0,
                "use_lhotse": False,
                "shuffle": True,
            }
        )

        with patch(
            "nemo.collections.asr.models.sortformer_diar_models.AudioToSpeechE2ESpkDiarDataset",
            return_value=dataset,
        ):
            if is_training:
                model.setup_training_data(config)
                dataloader = model._train_dl
            else:
                dataloader = model._SortformerEncLabelModel__setup_dataloader_from_config(config)

        assert isinstance(dataloader.sampler, _OversamplingDistributedSampler) is expect_oversampling
        if expect_oversampling:
            assert dataloader.sampler.shuffle is True
            assert dataloader.sampler.seed == 1234
            assert dataloader.sampler._batch_size == config.batch_size
            assert dataloader.sampler.num_samples == config.batch_size * limit_train_batches
            assert dataloader.sampler.total_size == model.world_size * dataloader.sampler.num_samples
            assert len(dataloader.sampler) == config.batch_size * limit_train_batches
            assert len(dataloader) == limit_train_batches
            assert len(list(dataloader.sampler)) == config.batch_size * limit_train_batches
        else:
            expected_sampler_type = (
                torch.utils.data.RandomSampler if is_training else torch.utils.data.SequentialSampler
            )
            assert isinstance(dataloader.sampler, expected_sampler_type)

    @pytest.mark.unit
    @pytest.mark.parametrize("shuffle", [False, True])
    def test_oversampling_sampler_balances_non_divisible_dataset(self, shuffle: bool) -> None:
        """Verify global cycles do not amplify distributed padding duplicates."""
        dataset = torch.utils.data.TensorDataset(torch.arange(5))
        rank_outputs: List[List[int]] = []
        for rank in range(2):
            sampler = _OversamplingDistributedSampler(
                dataset,
                num_samples_per_rank=20,
                batch_size=2,
                num_replicas=2,
                rank=rank,
                shuffle=shuffle,
                seed=17,
            )
            sampler.set_epoch(3)
            rank_outputs.append(list(sampler))

        assert [len(indices) for indices in rank_outputs] == [20, 20]
        assert Counter(rank_outputs[0] + rank_outputs[1]) == Counter({index: 8 for index in range(5)})

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("dataset_size", "target", "expected"),
        [
            (3, 8, [0, 1, 2, 0, 1, 2, 0, 1]),
            (4, 6, [0, 1, 2, 3, 0, 1]),
        ],
    )
    def test_oversampling_sampler_sequential_cycles(self, dataset_size: int, target: int, expected: List[int]) -> None:
        """Verify disabling shuffle preserves sequential order across cycles."""
        sampler = _OversamplingDistributedSampler(
            torch.utils.data.TensorDataset(torch.arange(dataset_size)),
            num_samples_per_rank=target,
            batch_size=2,
            num_replicas=1,
            rank=0,
            shuffle=False,
        )

        assert list(sampler) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("seed", "epoch", "expect_same"),
        [(17, 3, True), (18, 3, False), (17, 4, False)],
        ids=["same-seed-and-epoch", "different-seed", "different-epoch"],
    )
    def test_oversampling_sampler_shuffle_seed_and_epoch(self, seed: int, epoch: int, expect_same: bool) -> None:
        """Verify shuffled sequences reproduce only with the same seed and epoch."""
        dataset = torch.utils.data.TensorDataset(torch.arange(11))
        baseline = _OversamplingDistributedSampler(
            dataset,
            num_samples_per_rank=33,
            batch_size=3,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=17,
        )
        baseline.set_epoch(3)
        candidate = _OversamplingDistributedSampler(
            dataset,
            num_samples_per_rank=33,
            batch_size=3,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=seed,
        )
        candidate.set_epoch(epoch)

        assert (list(candidate) == list(baseline)) is expect_same

    @pytest.mark.unit
    @pytest.mark.parametrize(("completed_batches", "batch_size"), [(2, 2), (1, 3)])
    def test_oversampling_sampler_rotates_past_completed_batches(
        self, completed_batches: int, batch_size: int
    ) -> None:
        """Verify a resumed epoch starts at its first unconsumed local index."""

        def trainer_state(completed: int) -> SimpleNamespace:
            """Build the minimal Lightning progress state consumed by the sampler."""
            current = SimpleNamespace(completed=completed)
            batch_progress = SimpleNamespace(current=current)
            epoch_loop = SimpleNamespace(batch_progress=batch_progress)
            return SimpleNamespace(current_epoch=5, fit_loop=SimpleNamespace(epoch_loop=epoch_loop))

        dataset = torch.utils.data.TensorDataset(torch.arange(7))
        full_sampler = _OversamplingDistributedSampler(
            dataset,
            num_samples_per_rank=18,
            batch_size=batch_size,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=23,
            trainer=trainer_state(0),
        )
        resumed_sampler = _OversamplingDistributedSampler(
            dataset,
            num_samples_per_rank=18,
            batch_size=batch_size,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=23,
            trainer=trainer_state(completed_batches),
        )

        full = list(full_sampler)
        resumed = list(resumed_sampler)
        offset = completed_batches * batch_size
        assert resumed[: 18 - offset] == full[offset:]
        assert resumed == full[offset:] + full[:offset]

    @pytest.mark.unit
    @pytest.mark.parametrize(("num_replicas", "rank"), [(1, 0), (2, 1)])
    def test_oversampling_sampler_rejects_empty_dataset(self, num_replicas: int, rank: int) -> None:
        """Verify empty datasets fail clearly for single-rank and distributed sampling."""
        sampler = _OversamplingDistributedSampler(
            torch.utils.data.TensorDataset(torch.empty(0)),
            num_samples_per_rank=4,
            batch_size=2,
            num_replicas=num_replicas,
            rank=rank,
        )

        with pytest.raises(ValueError, match="training dataset is empty"):
            list(sampler)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "subsegment_options",
        [
            {
                "subsegment_mode": True,
                "subsegment_single_chunk_min_len_sec": 30.0,
                "subsegment_two_chunk_min_len_sec": 12.0,
                "subsegment_two_chunks_rate": 0.5,
                "subsegment_nspk_bias": 2.0,
                "subsegment_start_guard_sec": 0.3,
                "subsegment_min_first_spk_sec": 0.6,
                "subsegment_splice_silence_sec": 0.2,
                "validate_manifest_paths": False,
            },
            {
                "subsegment_mode": False,
                "subsegment_single_chunk_min_len_sec": 20.0,
                "subsegment_two_chunk_min_len_sec": 8.0,
                "subsegment_two_chunks_rate": 0.0,
                "subsegment_nspk_bias": 1.0,
                "subsegment_start_guard_sec": 0.0,
                "subsegment_min_first_spk_sec": 0.4,
                "subsegment_splice_silence_sec": 0.1,
                "validate_manifest_paths": True,
            },
        ],
        ids=["enabled-custom-options", "disabled-custom-options"],
    )
    def test_legacy_dataloader_forwards_subsegment_options(self, subsegment_options):
        model = _create_sortformer_model()
        dataset = SimpleNamespace(collection=[], eesd_train_collate_fn=lambda batch: batch)
        config = DictConfig(
            {
                "manifest_filepath": "unused.json",
                "sample_rate": 16000,
                "soft_label_thres": 0.5,
                "session_len_sec": 90,
                "num_spks": 4,
                "soft_targets": False,
                "batch_size": 1,
                "num_workers": 0,
                "use_lhotse": False,
                **subsegment_options,
            }
        )

        with patch(
            "nemo.collections.asr.models.sortformer_diar_models.AudioToSpeechE2ESpkDiarDataset",
            return_value=dataset,
        ) as dataset_constructor:
            model._SortformerEncLabelModel__setup_dataloader_from_config(config)

        dataset_kwargs = dataset_constructor.call_args.kwargs
        for option, expected_value in subsegment_options.items():
            if isinstance(expected_value, bool):
                assert dataset_kwargs[option] is expected_value
            else:
                assert dataset_kwargs[option] == expected_value

    @pytest.mark.unit
    @pytest.mark.parametrize("output_subsampling_factor", [1, 3, 16])
    def test_lhotse_dataloader_uses_high_resolution_targets(self, output_subsampling_factor):
        model = _create_sortformer_model(
            high_resolution=True,
            output_subsampling_factor=output_subsampling_factor,
        )
        config = DictConfig({"use_lhotse": True})

        with (
            patch(
                "nemo.collections.asr.models.sortformer_diar_models.LhotseAudioToSpeechE2ESpkDiarDataset"
            ) as dataset_constructor,
            patch(
                "nemo.collections.asr.models.sortformer_diar_models.get_lhotse_dataloader_from_config",
                return_value=object(),
            ),
        ):
            model._SortformerEncLabelModel__setup_dataloader_from_config(config)

        assert dataset_constructor.call_args.kwargs["cfg"].subsampling_factor == output_subsampling_factor

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "embedding_batch_size, embedding_frame_count, embedding_lengths, expected_output_shape, "
            "masked_start_frame, expected_masked_nonzero"
        ),
        [(2, 3, (3, 2), (2, 24, 4), 16, 0)],
    )
    def test_forward_infer_repeats_encoder_mask_at_high_resolution(
        self,
        embedding_batch_size,
        embedding_frame_count,
        embedding_lengths,
        expected_output_shape,
        masked_start_frame,
        expected_masked_nonzero,
    ):
        model = _create_sortformer_model(high_resolution=True).eval()
        emb_seq = torch.randn(
            embedding_batch_size,
            embedding_frame_count,
            model._cfg.model_defaults.tf_d_model,
        )
        emb_seq_length = torch.tensor(embedding_lengths)

        with torch.no_grad():
            preds = model.forward_infer(emb_seq, emb_seq_length)

        assert preds.shape == expected_output_shape
        assert torch.count_nonzero(preds[1, masked_start_frame:]) == expected_masked_nonzero

    @pytest.mark.unit
    @pytest.mark.parametrize("async_streaming", [False, True])
    @pytest.mark.parametrize(
        "high_resolution, output_subsampling_factor",
        [(True, 1), (True, 4), (True, 16), (False, 16)],
    )
    def test_full_streaming_output_uses_configured_resolution(
        self, async_streaming, high_resolution, output_subsampling_factor
    ):
        model = _create_sortformer_model(
            high_resolution=high_resolution,
            output_subsampling_factor=output_subsampling_factor,
        ).eval()
        model.streaming_mode = True
        model.async_streaming = async_streaming
        audio = torch.randn(1, 8000)
        audio_lengths = torch.tensor([8000], dtype=torch.long)

        with torch.no_grad():
            _, feature_lengths = model.process_signal(audio, audio_lengths)
            preds = model(audio, audio_lengths)

        assert preds.shape[1] == math.ceil(feature_lengths.item() / output_subsampling_factor)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "high_resolution, output_subsampling_factor",
        [(True, 3), (False, 24)],
    )
    def test_streaming_rejects_output_downsampling_across_chunk_boundaries(
        self, high_resolution, output_subsampling_factor
    ):
        model = _create_sortformer_model(
            high_resolution=high_resolution, output_subsampling_factor=output_subsampling_factor
        )

        with pytest.raises(ValueError, match="native chunk prediction length"):
            model._check_streaming_parameters()

    @pytest.mark.unit
    @pytest.mark.parametrize("async_streaming", [False, True])
    @pytest.mark.parametrize("output_subsampling_factor", [1, 4])
    def test_streaming_emits_configured_resolution_and_updates_cache_with_coarse_predictions(
        self, async_streaming, output_subsampling_factor
    ):
        model = _create_sortformer_model(
            high_resolution=True, output_subsampling_factor=output_subsampling_factor
        ).eval()
        model.streaming_mode = True
        model.async_streaming = async_streaming
        processed_signal = torch.randn(1, 120, 80)
        processed_signal_length = torch.tensor([120])
        streaming_state = model.sortformer_modules.init_streaming_state(batch_size=1, async_streaming=async_streaming)
        total_preds = torch.zeros(1, 0, model._cfg.max_num_of_spks)
        captured = {}

        def capture_streaming_update(**kwargs):
            captured["preds"] = kwargs["preds"]
            return streaming_update(**kwargs)

        if async_streaming:
            streaming_update = model.sortformer_modules.streaming_update_async
            model.sortformer_modules.streaming_update_async = capture_streaming_update
        else:
            streaming_update = model.sortformer_modules.streaming_update
            model.sortformer_modules.streaming_update = capture_streaming_update
        with torch.no_grad():
            _, total_preds = model.forward_streaming_step(
                processed_signal,
                processed_signal_length,
                streaming_state,
                total_preds,
            )

        expected_ratio = model.upsample_factor // output_subsampling_factor
        assert total_preds.shape[1] == captured["preds"].shape[1] * expected_ratio

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "chunk_shape, chunk_lengths, batch_size, initial_spkcache_lengths, initial_fifo_lengths",
        [((1, 120, 80), (120,), 1, (0,), (0,))],
    )
    def test_streaming_export_keeps_coarse_prediction_resolution(
        self,
        chunk_shape,
        chunk_lengths,
        batch_size,
        initial_spkcache_lengths,
        initial_fifo_lengths,
    ):
        model = _create_sortformer_model(high_resolution=True).eval()
        chunk = torch.randn(chunk_shape)
        chunk_lengths = torch.tensor(chunk_lengths)
        spkcache = torch.zeros(batch_size, 0, model._cfg.model_defaults.fc_d_model)
        spkcache_lengths = torch.tensor(initial_spkcache_lengths, dtype=torch.long)
        fifo = torch.zeros(batch_size, 0, model._cfg.model_defaults.fc_d_model)
        fifo_lengths = torch.tensor(initial_fifo_lengths, dtype=torch.long)

        with torch.no_grad():
            preds, _, chunk_pre_encode_lengths = model.forward_for_export(
                chunk,
                chunk_lengths,
                spkcache,
                spkcache_lengths,
                fifo,
                fifo_lengths,
            )

        assert preds.shape[1] == chunk_pre_encode_lengths.max()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "batch_size, state_capacity, chunk_lengths, export_spkcache_lengths, export_fifo_lengths, "
            "runtime_state_length_cases"
        ),
        [
            (
                2,
                8,
                (24, 16),
                (2, 5),
                (3, 4),
                (
                    ((0, 0), (0, 0)),
                    ((4, 7), (6, 1)),
                    ((8, 8), (8, 8)),
                ),
            )
        ],
    )
    def test_streaming_onnx_export_handles_runtime_state_lengths(
        self,
        tmp_path,
        batch_size,
        state_capacity,
        chunk_lengths,
        export_spkcache_lengths,
        export_fifo_lengths,
        runtime_state_length_cases,
    ):
        model = _create_sortformer_model().eval()
        chunk = torch.randn(batch_size, max(chunk_lengths), model.encoder._feat_in)
        chunk_lengths = torch.tensor(chunk_lengths)
        spkcache = torch.randn(batch_size, state_capacity, model.sortformer_modules.fc_d_model)
        fifo = torch.randn(batch_size, state_capacity, model.sortformer_modules.fc_d_model)
        export_inputs = (
            chunk,
            chunk_lengths,
            spkcache,
            torch.tensor(export_spkcache_lengths),
            fifo,
            torch.tensor(export_fifo_lengths),
        )
        runtime_state_lengths = tuple(
            (torch.tensor(spkcache_lengths), torch.tensor(fifo_lengths))
            for spkcache_lengths, fifo_lengths in runtime_state_length_cases
        )
        with torch.no_grad():
            expected_outputs = [
                model.forward_for_export(chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths)
                for spkcache_lengths, fifo_lengths in runtime_state_lengths
            ]

        output_path = tmp_path / "streaming_sortformer.onnx"
        model.export(str(output_path), input_example=export_inputs, dynamic_axes={})
        evaluator = ReferenceEvaluator(onnx.load(output_path))

        for (spkcache_lengths, fifo_lengths), expected in zip(runtime_state_lengths, expected_outputs):
            inputs = (chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths)
            actual = evaluator.run(
                None, {name: value.detach().cpu().numpy() for name, value in zip(model.input_names, inputs)}
            )
            for actual_tensor, expected_tensor in zip(actual, expected):
                np.testing.assert_allclose(actual_tensor, expected_tensor.detach().cpu().numpy(), rtol=1e-4, atol=1e-4)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "high_resolution, output_subsampling_factor, expected_frame_count",
        [(False, 3, 8), (True, 1, 1), (True, 3, 3)],
    )
    def test_diarize_postprocessing_uses_native_output_step(
        self, high_resolution, output_subsampling_factor, expected_frame_count
    ):
        model = _create_sortformer_model(
            high_resolution=high_resolution,
            output_subsampling_factor=output_subsampling_factor,
        ).eval()
        model._diarize_audio_rttm_map = {"sample": {"offset": 0.0}}
        outputs = torch.zeros(1, 3, model._cfg.max_num_of_spks)
        diarize_config = SimpleNamespace(postprocessing_params=None, include_tensor_outputs=False)

        with (
            patch(
                "nemo.collections.asr.models.sortformer_diar_models.predlist_to_timestamps",
                return_value=[[[] for _ in range(model._cfg.max_num_of_spks)]],
            ) as postprocess,
            patch(
                "nemo.collections.asr.models.sortformer_diar_models.generate_diarization_output_lines",
                return_value=[],
            ),
        ):
            model._diarize_output_processing(outputs, ["sample"], diarize_config)

        postprocess.assert_called_once()
        call_kwargs = postprocess.call_args.kwargs
        assert len(call_kwargs["batch_preds_list"]) == 1
        assert torch.equal(call_kwargs["batch_preds_list"][0], outputs)
        assert call_kwargs["audio_rttm_map_dict"] == model._diarize_audio_rttm_map
        assert call_kwargs["cfg_vad_params"] is diarize_config.postprocessing_params
        assert call_kwargs["unit_10ms_frame_count"] == expected_frame_count
        assert call_kwargs["bypass_postprocessing"] is False
