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
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
import pytest
import torch
from examples.speaker_tasks.diarization.neural_diarizer.e2e_diarize_speech import DiarizationConfig, get_tensor_path
from omegaconf import DictConfig
from onnx.reference import ReferenceEvaluator

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
from nemo.collections.asr.parts.utils.sortformer_utils import InferenceProfiler, configure_output_subsampling_factor


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
):
    if output_subsampling_factor is None:
        output_subsampling_factor = 1 if high_resolution else 8

    model = {
        'sample_rate': 16000,
        'pil_weight': 0.5,
        'ats_weight': 0.5,
        'max_num_of_spks': 4,
        'high_resolution': high_resolution,
        'output_subsampling_factor': output_subsampling_factor,
        'async_streaming': False,
        'streaming_mode': False,
    }
    model_defaults = {
        'fc_d_model': 128 if frontend_encoder == "transformer" else 32,
        'tf_d_model': 16,
    }
    preprocessor = {
        '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
        'normalize': 'per_feature',
        'window_size': 0.025,
        'sample_rate': 16000,
        'window_stride': 0.01,
        'window': 'hann',
        'features': 128 if frontend_encoder == "transformer" else 80,
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
    }

    if frontend_encoder == "transformer":
        # Keep the production Transformer architecture and options, but scale its depth and width for CPU unit tests.
        encoder = {
            '_target_': 'nemo.collections.asr.modules.TransformerEncoder',
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
            'attn_mode': 'full',
            'drop_rate': 0.1,
            'dropout_pre_encoder': 0.1,
            'dropout_emb': 0.0,
            'sync_max_audio_length': True,
        }
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

    loss = {
        '_target_': 'nemo.collections.asr.losses.bce_loss.BCELoss',
        'weight': None,
        'reduction': 'mean',
    }

    model_config = {
        'sample_rate': 16000,
        'pil_weight': 0.5,
        'ats_weight': 0.5,
        'max_num_of_spks': 4,
        'high_resolution': high_resolution,
        'output_subsampling_factor': output_subsampling_factor,
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
    @pytest.mark.parametrize("streaming_mode", [False, True])
    def test_transformer_encoder_is_optional(self, streaming_mode):
        model = _create_sortformer_model(
            include_transformer_encoder=False,
            frontend_encoder="transformer",
        )
        model.streaming_mode = streaming_mode
        if streaming_mode:
            model.sortformer_modules.causal_attn_rate = 1.0
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


class TestSortformerEncLabelModelStreaming:
    @pytest.mark.unit
    @pytest.mark.parametrize("field_name", ["spkcache_len", "chunk_left_context"])
    def test_model_dependent_streaming_overrides_default_to_none(self, field_name):
        assert getattr(DiarizationConfig(), field_name) is None

    @pytest.mark.unit
    def test_constructor(self, sortformer_model):
        sortformer_model.streaming_mode = True
        sortformer_diar_model = sortformer_model.train()
        confdict = sortformer_diar_model.to_config_dict()
        instance2 = SortformerEncLabelModel.from_config_dict(confdict)
        assert isinstance(instance2, SortformerEncLabelModel)

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
                "soft_label_thres": 0.5,
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
