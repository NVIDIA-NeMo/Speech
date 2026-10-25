# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NeMo Speech OneLogger lifecycle tracing."""

import os
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from lightning.pytorch.callbacks import Callback as PTLCallback
from omegaconf import OmegaConf

import nemo.lightning.one_logger_callback as one_logger_module
from nemo.collections.asr.one_logger import ASRThroughputPolicy
from nemo.collections.speechlm2.one_logger import SALMThroughputPolicy
from nemo.core.classes.modelPT import ModelPT
from nemo.lightning.base_callback import BaseCallback
from nemo.lightning.callback_group import CallbackGroup, callback_context, with_model_init_callbacks
from nemo.lightning.one_logger_callback import (
    OneLoggerNeMoCallback,
    _get_throughput_interval,
    _should_enable_for_current_rank,
    get_one_logger_init_config,
)
from nemo.lightning.speech_throughput import SpeechThroughputPolicy
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO
from nemo.utils.callbacks.nemo_model_checkpoint import NeMoModelCheckpoint


@pytest.fixture(autouse=True)
def reset_one_logger_callback_singleton():
    previous_instance = OneLoggerNeMoCallback._instance
    OneLoggerNeMoCallback._instance = None
    yield
    OneLoggerNeMoCallback._instance = previous_instance


def _enabled_callback():
    provider = MagicMock()
    provider_class = MagicMock()
    provider_class.instance.return_value = provider
    provider.recorder.start.side_effect = lambda name, **kwargs: SimpleNamespace(name=name, attributes=kwargs)
    init_config = {
        "application_name": "nemo-speech",
        "session_tag_or_fn": "test",
        "enable_for_current_rank": True,
        "world_size_or_fn": 1,
    }
    with ExitStack() as stack:
        stack.enter_context(
            patch('nemo.lightning.one_logger_callback.get_one_logger_init_config', return_value=init_config)
        )
        stack.enter_context(patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider', provider_class))
        stack.enter_context(patch('nemo.lightning.one_logger_callback.OneLoggerConfig'))
        stack.enter_context(
            patch(
                'nemo.lightning.one_logger_callback.on_app_start',
                return_value=SimpleNamespace(name='application'),
            )
        )
        callback = OneLoggerNeMoCallback()
    return callback, provider


class TestOneLoggerNeMoCallback:
    def test_inherits_only_nemo_callback(self):
        callback, _ = _enabled_callback()

        assert isinstance(callback, BaseCallback)
        assert isinstance(callback, PTLCallback)
        assert type(callback).__bases__ == (BaseCallback,)

    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    def test_init_configures_application_without_training_schema(
        self, mock_get_config, mock_provider_class, mock_config_class, mock_app_start
    ):
        init_config = {
            "application_name": "nemo-speech",
            "session_tag_or_fn": "test",
            "enable_for_current_rank": True,
            "world_size_or_fn": 2,
        }
        mock_get_config.return_value = init_config
        provider = mock_provider_class.instance.return_value

        callback = OneLoggerNeMoCallback()

        mock_config_class.assert_called_once_with(**init_config)
        provider.with_base_config.assert_called_once_with(mock_config_class.return_value)
        provider.with_base_config.return_value.with_export_config.assert_called_once_with()
        provider.with_base_config.return_value.with_export_config.return_value.configure_provider.assert_called_once_with()
        provider.set_training_telemetry_config.assert_not_called()
        mock_app_start.assert_called_once_with()
        assert callback._provider is provider

    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    def test_disabled_rank_does_not_initialize_onelogger(
        self, mock_get_config, mock_provider_class, mock_config_class, mock_app_start
    ):
        mock_get_config.return_value = {
            "application_name": "nemo-speech",
            "session_tag_or_fn": "test",
            "enable_for_current_rank": False,
            "world_size_or_fn": 8,
        }

        callback = OneLoggerNeMoCallback()

        assert callback.enabled_for_current_rank is False
        assert callback._provider is None
        mock_provider_class.instance.assert_not_called()
        mock_config_class.assert_not_called()
        mock_app_start.assert_not_called()

    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    def test_provider_initialization_failure_disables_callback_without_raising(
        self, mock_get_config, mock_provider_class, mock_config_class, mock_app_start
    ):
        mock_get_config.return_value = {
            "application_name": "nemo-speech",
            "session_tag_or_fn": "test",
            "enable_for_current_rank": True,
            "world_size_or_fn": 1,
        }
        provider = mock_provider_class.instance.return_value
        provider.with_base_config.return_value.with_export_config.return_value.configure_provider.side_effect = (
            RuntimeError("exporter unavailable")
        )

        callback = OneLoggerNeMoCallback()

        assert callback.enabled_for_current_rank is False
        assert callback._provider is None
        assert callback._application_span is None
        mock_app_start.assert_not_called()

    def test_lifecycle_spans_are_paired_by_identity(self):
        callback, provider = _enabled_callback()
        lifecycle = (
            (callback.on_model_init_start, callback.on_model_init_end, "nemo_speech.model_initialization"),
            (
                callback.on_dataloader_init_start,
                callback.on_dataloader_init_end,
                "nemo_speech.data_loader_initialization",
            ),
            (callback.on_optimizer_init_start, callback.on_optimizer_init_end, "nemo_speech.optimizer_initialization"),
            (callback.on_load_checkpoint_start, callback.on_load_checkpoint_end, "nemo_speech.checkpoint_load"),
        )

        for start, end, name in lifecycle:
            start()
            span = callback._current_span(name)
            end()
            provider.recorder.stop.assert_any_call(span)

        assert callback._active_spans == []

    def test_nested_same_name_spans_are_paired_last_in_first_out(self):
        callback, provider = _enabled_callback()

        callback.on_model_init_start()
        outer = callback._current_span("nemo_speech.model_initialization")
        callback.on_model_init_start()
        inner = callback._current_span("nemo_speech.model_initialization")
        callback.on_model_init_end()
        callback.on_model_init_end()

        assert inner is not outer
        assert provider.recorder.stop.call_args_list == [call(inner), call(outer)]
        assert callback._active_spans == []

    def test_training_and_validation_report_only_loop_timing(self):
        callback, provider = _enabled_callback()
        trainer = module = object()

        callback.on_train_start(trainer, module)
        training_span = callback._current_span("nemo_speech.training")
        callback.on_validation_start(trainer, module)
        validation_span = callback._current_span("nemo_speech.validation")
        callback.on_validation_end(trainer, module)
        callback.on_train_end(trainer, module)

        assert provider.recorder.start.call_args_list == [
            call("nemo_speech.training", span_attributes=None),
            call("nemo_speech.validation", span_attributes=None),
        ]
        assert provider.recorder.stop.call_args_list == [call(validation_span), call(training_span)]

    @patch('nemo.lightning.one_logger_callback.Event.create')
    def test_dynamic_asr_throughput_is_reported_over_logging_window(self, mock_event_create):
        callback, provider = _enabled_callback()
        mock_event_create.side_effect = lambda name, attributes: (name, attributes.to_json())
        trainer = SimpleNamespace(log_every_n_steps=2, global_step=8, accumulate_grad_batches=2)
        model = SimpleNamespace(
            one_logger_throughput_policy=ASRThroughputPolicy,
            preprocessor=SimpleNamespace(_sample_rate=16000),
        )
        batches = [
            (
                torch.zeros(2, 32000),
                torch.tensor([16000, 32000]),
                torch.zeros(2, 8),
                torch.tensor([5, 7]),
            ),
            (torch.zeros(1, 8000), torch.tensor([8000]), torch.zeros(1, 4), torch.tensor([4])),
        ]

        with (
            patch('nemo.lightning.one_logger_callback._get_throughput_interval', return_value=2),
            patch('nemo.lightning.one_logger_callback.time.monotonic', side_effect=[10.0, 12.0]),
            patch('nemo.lightning.one_logger_callback.torch.cuda.Event') as cuda_event,
        ):
            callback.on_train_start(trainer, model)
            for batch_idx, batch in enumerate(batches):
                callback.on_train_batch_start(trainer, model, batch, batch_idx)
                if batch_idx == 1:
                    callback.on_before_optimizer_step(trainer, model, object())
                    trainer.global_step = 9
                callback.on_train_batch_end(trainer, model, None, batch, batch_idx)

        cuda_event.assert_not_called()

        training_span = callback._current_span("nemo_speech.training")
        event = provider.recorder.event.call_args_list[-1]
        assert event.args[0] is training_span
        assert event.args[1][0] == "nemo_speech.throughput"
        attributes = event.args[1][1]
        assert {key: attributes[key] for key in ("scope", "rank", "policy", "global_step", "window_batches")} == {
            "scope": "rank",
            "rank": 0,
            "policy": "asr",
            "global_step": 9,
            "window_batches": 2,
        }
        assert attributes["window_optimizer_steps"] == 1
        assert attributes["examples"] == 3
        assert attributes["mean_batch_size"] == pytest.approx(3.0)
        assert attributes["examples_per_second"] == pytest.approx(1.5)
        assert attributes["window_seconds"] == pytest.approx(2.0)
        assert attributes["input_audio_seconds"] == pytest.approx(3.5)
        assert attributes["input_audio_seconds_per_second"] == pytest.approx(1.75)
        assert attributes["input_audio_seconds_per_step"] == pytest.approx(3.5)
        assert attributes["target_text_tokens"] == pytest.approx(16.0)
        assert attributes["target_text_tokens_per_second"] == pytest.approx(8.0)
        assert attributes["target_text_tokens_per_step"] == pytest.approx(16.0)
        assert all("micro_batch" not in name for name in attributes)
        assert all("_per_example" not in name for name in attributes)

    @patch('nemo.lightning.one_logger_callback.Event.create')
    def test_salm_reports_exact_multimodal_tokens_per_second(self, mock_event_create):
        callback, provider = _enabled_callback()
        mock_event_create.side_effect = lambda name, attributes: (name, attributes.to_json())
        trainer = SimpleNamespace(log_every_n_steps=1, global_step=3)
        model = SimpleNamespace(
            device=torch.device('cpu'),
            one_logger_throughput_policy=SALMThroughputPolicy,
            sampling_rate=16000,
            _last_batch_num_tokens=torch.tensor(30),
            _last_batch_num_examples=2,
        )
        batch = {"audio_lens": torch.tensor([16000, 8000])}

        with (
            patch('nemo.lightning.one_logger_callback._get_throughput_interval', return_value=1),
            patch('nemo.lightning.one_logger_callback.time.monotonic', side_effect=[10.0, 12.0]),
        ):
            callback.on_train_start(trainer, model)
            callback.on_train_batch_start(trainer, model, batch, 0)
            callback.on_before_optimizer_step(trainer, model, object())
            trainer.global_step = 4
            callback.on_train_batch_end(trainer, model, None, batch, 0)

        attributes = provider.recorder.event.call_args.args[1][1]
        assert attributes["policy"] == "salm"
        assert attributes["input_audio_seconds_per_second"] == pytest.approx(0.75)
        assert attributes["multimodal_tokens"] == pytest.approx(30.0)
        assert attributes["multimodal_tokens_per_second"] == pytest.approx(15.0)
        assert attributes["multimodal_tokens_per_step"] == pytest.approx(30.0)
        assert attributes["mean_batch_size"] == pytest.approx(2.0)

    @patch('nemo.lightning.one_logger_callback.Event.create')
    def test_accumulation_window_waits_for_an_optimizer_boundary(self, mock_event_create):
        callback, provider = _enabled_callback()
        mock_event_create.side_effect = lambda name, attributes: (name, attributes.to_json())
        trainer = SimpleNamespace(log_every_n_steps=2, global_step=0, accumulate_grad_batches=3)
        model = SimpleNamespace(
            one_logger_throughput_policy=ASRThroughputPolicy,
            preprocessor=SimpleNamespace(_sample_rate=16000),
        )
        batch = (torch.zeros(1, 16000), torch.tensor([16000]), torch.zeros(1, 2), torch.tensor([2]))

        with (
            patch('nemo.lightning.one_logger_callback._get_throughput_interval', return_value=2),
            patch('nemo.lightning.one_logger_callback.time.monotonic', side_effect=[10.0, 13.0]),
        ):
            callback.on_train_start(trainer, model)
            for batch_idx in range(3):
                callback.on_train_batch_start(trainer, model, batch, batch_idx)
                if batch_idx == 2:
                    callback.on_before_optimizer_step(trainer, model, object())
                    trainer.global_step = 1
                callback.on_train_batch_end(trainer, model, None, batch, batch_idx)
                if batch_idx == 1:
                    provider.recorder.event.assert_not_called()

        attributes = provider.recorder.event.call_args.args[1][1]
        assert attributes["window_batches"] == 3
        assert attributes["window_optimizer_steps"] == 1
        assert attributes["examples"] == 3
        assert attributes["mean_batch_size"] == pytest.approx(3.0)
        assert attributes["target_text_tokens_per_step"] == pytest.approx(6.0)

    @patch('nemo.lightning.one_logger_callback.Event.create')
    def test_partial_accumulation_window_omits_per_step_means(self, mock_event_create):
        callback, provider = _enabled_callback()
        mock_event_create.side_effect = lambda name, attributes: (name, attributes.to_json())
        trainer = SimpleNamespace(log_every_n_steps=100, global_step=0, accumulate_grad_batches=2)
        model = SimpleNamespace(
            one_logger_throughput_policy=ASRThroughputPolicy,
            preprocessor=SimpleNamespace(_sample_rate=16000),
        )
        batch = (torch.zeros(1, 16000), torch.tensor([16000]), torch.zeros(1, 2), torch.tensor([2]))

        with patch('nemo.lightning.one_logger_callback.time.monotonic', side_effect=[10.0, 13.0]):
            callback.on_train_start(trainer, model)
            for batch_idx in range(3):
                callback.on_train_batch_start(trainer, model, batch, batch_idx)
                if batch_idx == 1:
                    callback.on_before_optimizer_step(trainer, model, object())
                    trainer.global_step = 1
                callback.on_train_batch_end(trainer, model, None, batch, batch_idx)
            callback.on_validation_start(trainer, model)

        attributes = provider.recorder.event.call_args.args[1][1]
        assert attributes["window_optimizer_steps"] == 1
        assert attributes["examples"] == 3
        assert attributes["examples_per_second"] == pytest.approx(1.0)
        assert "mean_batch_size" not in attributes
        assert "input_audio_seconds_per_step" not in attributes
        assert "target_text_tokens_per_step" not in attributes

    @patch('nemo.lightning.one_logger_callback.Event.create')
    def test_cuda_window_publishes_later_without_synchronizing_training(self, mock_event_create):
        callback, provider = _enabled_callback()
        mock_event_create.side_effect = lambda name, attributes: (name, attributes.to_json())
        trainer = SimpleNamespace(log_every_n_steps=1, global_step=2)
        model = SimpleNamespace(
            device=torch.device('cuda', 0),
            one_logger_throughput_policy=ASRThroughputPolicy,
            preprocessor=SimpleNamespace(_sample_rate=16000),
        )
        batch = (torch.zeros(1, 16000), torch.tensor([16000]), torch.zeros(1, 3), torch.tensor([3]))
        start_event = MagicMock()
        end_event = MagicMock()
        next_start_event = MagicMock()
        end_event.query.side_effect = [False, True]
        start_event.elapsed_time.return_value = 2000.0

        with (
            patch('nemo.lightning.one_logger_callback._get_throughput_interval', return_value=1),
            patch('nemo.lightning.one_logger_callback.time.monotonic', side_effect=[10.0, 12.0, 20.0]),
            patch(
                'nemo.lightning.one_logger_callback.torch.cuda.Event',
                side_effect=[start_event, end_event, next_start_event],
            ),
            patch('nemo.lightning.one_logger_callback.torch.cuda.device'),
        ):
            callback.on_train_start(trainer, model)
            callback.on_train_batch_start(trainer, model, batch, 0)
            callback.on_train_batch_end(trainer, model, None, batch, 0)
            provider.recorder.event.assert_not_called()
            callback.on_train_batch_start(trainer, model, batch, 1)

        end_event.synchronize.assert_not_called()
        assert end_event.query.call_count == 2
        attributes = provider.recorder.event.call_args.args[1][1]
        assert attributes["window_seconds"] == 2.0
        assert attributes["input_audio_seconds_per_second"] == 0.5
        assert attributes["target_text_tokens_per_second"] == 1.5

    def test_cuda_pending_window_stays_bounded_while_gpu_work_is_unresolved(self):
        callback, provider = _enabled_callback()
        trainer = SimpleNamespace(log_every_n_steps=1, global_step=1)
        model = SimpleNamespace(
            device=torch.device('cuda', 0),
            one_logger_throughput_policy=ASRThroughputPolicy,
            preprocessor=SimpleNamespace(_sample_rate=16000),
        )
        batch = (torch.zeros(1, 16000), torch.tensor([16000]), torch.zeros(1, 3), torch.tensor([3]))
        start_event = MagicMock()
        end_event = MagicMock()
        end_event.query.return_value = False

        with (
            patch('nemo.lightning.one_logger_callback._get_throughput_interval', return_value=1),
            patch('nemo.lightning.one_logger_callback.time.monotonic', side_effect=[10.0, 12.0]),
            patch(
                'nemo.lightning.one_logger_callback.torch.cuda.Event',
                side_effect=[start_event, end_event],
            ) as cuda_event,
            patch('nemo.lightning.one_logger_callback.torch.cuda.device'),
        ):
            callback.on_train_start(trainer, model)
            callback.on_train_batch_start(trainer, model, batch, 0)
            callback.on_train_batch_end(trainer, model, None, batch, 0)
            pending = callback._pending_throughput

            for batch_idx in range(1, 5):
                callback.on_train_batch_start(trainer, model, batch, batch_idx)
                callback.on_train_batch_end(trainer, model, None, batch, batch_idx)

        assert callback._pending_throughput is pending
        assert callback._throughput_batches == 0
        assert cuda_event.call_count == 2
        assert end_event.query.call_count == 5
        end_event.synchronize.assert_not_called()
        provider.recorder.event.assert_not_called()

    @patch('nemo.lightning.one_logger_callback.Event.create')
    def test_lifecycle_boundaries_close_partial_windows(self, mock_event_create):
        callback, provider = _enabled_callback()
        mock_event_create.side_effect = lambda name, attributes: (name, attributes.to_json())
        trainer = SimpleNamespace(log_every_n_steps=100, global_step=4)
        model = SimpleNamespace(
            one_logger_throughput_policy=ASRThroughputPolicy,
            preprocessor=SimpleNamespace(_sample_rate=16000),
        )
        batch = (torch.zeros(1, 16000), torch.tensor([16000]), torch.zeros(1, 2), torch.tensor([2]))

        with patch('nemo.lightning.one_logger_callback.time.monotonic', side_effect=[10.0, 12.0, 20.0, 22.0]):
            callback.on_train_start(trainer, model)
            callback.on_train_batch_start(trainer, model, batch, 0)
            callback.on_train_batch_end(trainer, model, None, batch, 0)
            callback.on_validation_start(trainer, model)
            callback.on_validation_end(trainer, model)
            callback.on_train_batch_start(trainer, model, batch, 1)
            callback.on_train_batch_end(trainer, model, None, batch, 1)
            callback.on_save_checkpoint_start(global_step=4)

        throughput_events = [
            event.args[1][1]
            for event in provider.recorder.event.call_args_list
            if event.args[1][0] == "nemo_speech.throughput"
        ]
        assert [event["window_seconds"] for event in throughput_events] == [2.0, 2.0]
        assert all(event["window_batches"] == 1 for event in throughput_events)

    def test_empty_measurements_do_not_emit_throughput(self):
        callback, provider = _enabled_callback()
        trainer = SimpleNamespace(log_every_n_steps=1, global_step=1)
        model = SimpleNamespace(one_logger_throughput_policy=SpeechThroughputPolicy)
        batch = {"unknown": torch.tensor([1])}

        callback.on_train_start(trainer, model)
        callback.on_train_batch_start(trainer, model, batch, 0)
        callback.on_train_batch_end(trainer, model, None, batch, 0)

        provider.recorder.event.assert_not_called()
        assert callback._throughput_batches == 0

    def test_malformed_custom_policy_never_interrupts_training(self):
        class MalformedPolicy:
            name = "malformed"

            def measure(self, model, batch):
                del model, batch
                return {"invalid": object()}

        callback, provider = _enabled_callback()
        trainer = SimpleNamespace(log_every_n_steps=1, global_step=1)
        model = SimpleNamespace(one_logger_throughput_policy=MalformedPolicy)
        batch = {"value": torch.tensor([1])}

        with (
            patch('nemo.lightning.one_logger_callback._get_throughput_interval', return_value=1),
            patch('nemo.lightning.one_logger_callback.time.monotonic', side_effect=[10.0, 11.0]),
        ):
            callback.on_train_start(trainer, model)
            callback.on_train_batch_start(trainer, model, batch, 0)
            callback.on_train_batch_end(trainer, model, None, batch, 0)

        provider.recorder.event.assert_not_called()
        assert callback._throughput_policy is None

    def test_validation_batches_do_not_emit_throughput(self):
        callback, provider = _enabled_callback()
        batch = {"audio_lens": torch.tensor([10])}

        callback.on_validation_batch_start(object(), object(), batch, 0)
        callback.on_validation_batch_end(object(), object(), None, batch, 0)

        provider.recorder.start.assert_not_called()
        provider.recorder.event.assert_not_called()

    @patch('nemo.lightning.one_logger_callback.Event.create')
    def test_checkpoint_outcomes_are_explicit_and_async_safe(self, mock_event_create):
        callback, provider = _enabled_callback()
        mock_event_create.side_effect = lambda name, attributes: (name, attributes.to_json())

        callback.on_save_checkpoint_start(3)
        success_span = callback._current_span("nemo_speech.checkpoint_save")
        callback.on_save_checkpoint_success(3)
        callback.on_save_checkpoint_end()

        callback.on_save_checkpoint_start(4, async_save=True)
        async_span = callback._current_span("nemo_speech.checkpoint_save")
        callback.on_save_checkpoint_end()
        callback.on_save_checkpoint_success(4)

        callback.on_save_checkpoint_start(5)
        failed_span = callback._current_span("nemo_speech.checkpoint_save")
        callback.on_save_checkpoint_failure(5)
        callback.on_save_checkpoint_end()

        assert [
            (args[0], kwargs["span_attributes"].to_json()) for args, kwargs in provider.recorder.start.call_args_list
        ] == [
            ("nemo_speech.checkpoint_save", {"global_step": 3, "asynchronous": False}),
            ("nemo_speech.checkpoint_save", {"global_step": 4, "asynchronous": True}),
            ("nemo_speech.checkpoint_save", {"global_step": 5, "asynchronous": False}),
        ]
        assert provider.recorder.event.call_args_list == [
            call(success_span, ("nemo_speech.checkpoint_save_success", {"global_step": 3})),
            call(callback._application_span, ("nemo_speech.checkpoint_save_success", {"global_step": 4})),
            call(failed_span, ("nemo_speech.checkpoint_save_failure", {"global_step": 5})),
        ]
        assert provider.recorder.stop.call_args_list == [call(success_span), call(async_span), call(failed_span)]

    @patch('nemo.lightning.one_logger_callback.on_app_end')
    def test_app_end_closes_unfinished_spans(self, mock_app_end):
        callback, provider = _enabled_callback()
        callback.on_train_start(object(), object())
        callback.on_validation_start(object(), object())
        validation_span = callback._current_span("nemo_speech.validation")
        training_span = callback._current_span("nemo_speech.training")

        callback.on_app_end()

        assert provider.recorder.stop.call_args_list == [call(validation_span), call(training_span)]
        mock_app_end.assert_called_once_with()


class TestOneLoggerConfiguration:
    def test_disabled_mode_does_not_import_onelogger(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("nemo.lightning.one_logger_callback.importlib.import_module") as import_module,
        ):
            group = CallbackGroup()

        import_module.assert_not_called()
        assert group.callbacks == []

    def test_missing_optional_dependency_is_a_noop(self):
        missing = ModuleNotFoundError("No module named nv_one_logger", name="nv_one_logger")
        with (
            patch.dict(os.environ, {"NEMO_ONE_LOGGER_ENABLED": "true"}, clear=True),
            patch.multiple(
                one_logger_module,
                OneLoggerConfig=None,
                OneLoggerErrorHandlingStrategy=None,
                Attributes=None,
                Event=None,
                on_app_end=None,
                on_app_start=None,
                TrainingTelemetryProvider=None,
            ),
            patch("nemo.lightning.one_logger_callback.importlib.import_module", side_effect=missing),
        ):
            group = CallbackGroup()

        assert group.callbacks == []

    def test_init_config_is_modality_independent(self):
        with patch.dict(
            os.environ,
            {
                "NEMO_ONE_LOGGER_ENABLED": "true",
                "SLURM_JOB_NAME": "speech-job",
                "WORLD_SIZE": "4",
                "RANK": "0",
            },
            clear=True,
        ):
            config = get_one_logger_init_config()

        assert config["application_name"] == "nemo-speech"
        assert config["session_tag_or_fn"] == "speech-job"
        assert config["world_size_or_fn"] == 4
        assert config["enable_for_current_rank"] is True
        assert "telemetry_config" not in config
        assert all("batch" not in key and "sequence" not in key and "token" not in key for key in config)

    @pytest.mark.parametrize("value", [None, "0", "false", "no", "off", "invalid"])
    def test_disabled_without_explicit_opt_in(self, value):
        environment = {"RANK": "0"}
        if value is not None:
            environment["NEMO_ONE_LOGGER_ENABLED"] = value
        with patch.dict(os.environ, environment, clear=True):
            assert not _should_enable_for_current_rank()

    def test_explicit_single_process_enable(self):
        with patch.dict(os.environ, {"NEMO_ONE_LOGGER_ENABLED": "true"}, clear=True):
            assert _should_enable_for_current_rank()

    def test_throughput_interval_is_low_frequency_by_default_and_configurable(self):
        trainer = SimpleNamespace(log_every_n_steps=10)
        with patch.dict(os.environ, {}, clear=True):
            assert _get_throughput_interval(trainer) == 100
        with patch.dict(os.environ, {"NEMO_ONE_LOGGER_THROUGHPUT_INTERVAL": "250"}, clear=True):
            assert _get_throughput_interval(trainer) == 250

    def test_distributed_rank_selection(self):
        with patch.dict(
            os.environ,
            {"NEMO_ONE_LOGGER_ENABLED": "true", "RANK": "1", "WORLD_SIZE": "4"},
            clear=True,
        ):
            assert not _should_enable_for_current_rank()
        with patch.dict(
            os.environ,
            {"NEMO_ONE_LOGGER_ENABLED": "true", "RANK": "0", "WORLD_SIZE": "4"},
            clear=True,
        ):
            assert _should_enable_for_current_rank()


class TestCallbackGroup:
    @pytest.mark.unit
    def test_attaches_callback_once_after_existing_callbacks(self):
        group = CallbackGroup.__new__(CallbackGroup)
        callback = BaseCallback()
        existing = PTLCallback()
        group._callbacks = [callback]
        trainer = SimpleNamespace(callbacks=[existing])

        group.attach_to_trainer(trainer)
        group.attach_to_trainer(trainer)

        assert trainer.callbacks == [existing, callback]

    @pytest.mark.unit
    def test_disabled_callback_is_not_attached(self):
        group = CallbackGroup.__new__(CallbackGroup)
        callback = BaseCallback()
        callback.enabled_for_current_rank = False
        group._callbacks = [callback]
        trainer = SimpleNamespace(callbacks=[])

        group.attach_to_trainer(trainer)

        assert trainer.callbacks == []

    @pytest.mark.unit
    def test_modelpt_subclass_init_emits_one_paired_span(self):
        class ParentModel(ModelPT):
            def __init__(self):
                super().__init__(cfg=OmegaConf.create({}))

            @classmethod
            def list_available_models(cls):
                return []

            def setup_training_data(self, train_data_config):
                pass

            def setup_validation_data(self, val_data_config):
                pass

        class ChildModel(ParentModel):
            def __init__(self):
                super().__init__()

        group = MagicMock()
        with patch('nemo.lightning.callback_group.CallbackGroup.get_instance', return_value=group):
            ChildModel()

        group.on_model_init_start.assert_called_once_with()
        group.on_model_init_end.assert_called_once_with()

    @pytest.mark.unit
    def test_eager_dataloader_failure_closes_dataloader_and_model_spans(self):
        class BrokenModel(ModelPT):
            @classmethod
            def list_available_models(cls):
                return []

            def setup_training_data(self, train_data_config):
                raise RuntimeError("dataloader failed")

            def setup_validation_data(self, val_data_config):
                pass

        group = MagicMock()
        with (
            patch('nemo.lightning.callback_group.CallbackGroup.get_instance', return_value=group),
            pytest.raises(RuntimeError, match="dataloader failed"),
        ):
            BrokenModel(cfg=OmegaConf.create({"train_ds": {}}))

        assert group.method_calls == [
            call.on_model_init_start(),
            call.on_dataloader_init_start(),
            call.on_dataloader_init_end(),
            call.on_model_init_end(),
        ]

    @pytest.mark.unit
    def test_model_class_decorator_emits_one_paired_span(self):
        @with_model_init_callbacks
        class Model:
            def __init__(self):
                pass

        group = MagicMock()
        with patch('nemo.lightning.callback_group.CallbackGroup.get_instance', return_value=group):
            Model()

        group.on_model_init_start.assert_called_once_with()
        group.on_model_init_end.assert_called_once_with()

    @pytest.mark.unit
    def test_callback_context_always_emits_end(self):
        group = MagicMock()
        with patch('nemo.lightning.callback_group.CallbackGroup.get_instance', return_value=group):
            with pytest.raises(RuntimeError, match="boom"):
                with callback_context('on_model_init_start', 'on_model_init_end'):
                    raise RuntimeError("boom")

        group.on_model_init_start.assert_called_once_with()
        group.on_model_init_end.assert_called_once_with()


class TestCheckpointIntegration:
    @pytest.mark.unit
    @patch('nemo.utils.callbacks.nemo_model_checkpoint.CallbackGroup.get_instance')
    def test_checkpoint_lifecycle_success(self, mock_get_group, tmp_path):
        group = mock_get_group.return_value
        callback = NeMoModelCheckpoint(dirpath=tmp_path, save_top_k=-1)
        callback.set_checkpoint_unfinished_marker = MagicMock()
        callback.remove_checkpoint_unfinished_marker = MagicMock()
        trainer = SimpleNamespace(
            global_step=7,
            callbacks=[],
            is_global_zero=False,
            loggers=[],
            save_checkpoint=MagicMock(),
        )

        callback._save_checkpoint(trainer, str(tmp_path / "model.ckpt"))

        group.on_save_checkpoint_start.assert_called_once_with(7, async_save=False)
        group.on_save_checkpoint_success.assert_called_once_with(7)
        group.on_save_checkpoint_end.assert_called_once_with()

    @pytest.mark.unit
    @patch('nemo.utils.callbacks.nemo_model_checkpoint.CallbackGroup.get_instance')
    def test_async_checkpoint_reports_success_only_after_finalization(self, mock_get_group, tmp_path):
        group = mock_get_group.return_value
        callback = NeMoModelCheckpoint(dirpath=tmp_path, save_top_k=-1, async_save=True)
        callback.set_checkpoint_unfinished_marker = MagicMock()
        callback.remove_checkpoint_unfinished_marker = MagicMock()
        trainer = SimpleNamespace(
            global_step=8,
            callbacks=[],
            is_global_zero=False,
            loggers=[],
            strategy=SimpleNamespace(checkpoint_io=MagicMock(spec=AsyncFinalizableCheckpointIO)),
            save_checkpoint=MagicMock(),
        )

        callback._save_checkpoint(trainer, str(tmp_path / "model.ckpt"))

        group.on_save_checkpoint_start.assert_called_once_with(8, async_save=True)
        group.on_save_checkpoint_success.assert_not_called()
        group.on_save_checkpoint_failure.assert_not_called()
        group.on_save_checkpoint_end.assert_called_once_with()

        finalize_fn = trainer.save_checkpoint.call_args.kwargs["storage_options"]["finalize_fn"]
        finalize_fn()

        group.on_save_checkpoint_success.assert_called_once_with(8)
        group.on_save_checkpoint_failure.assert_not_called()

    @pytest.mark.unit
    @patch('nemo.utils.callbacks.nemo_model_checkpoint.CallbackGroup.get_instance')
    def test_checkpoint_lifecycle_ends_on_failure(self, mock_get_group, tmp_path):
        group = mock_get_group.return_value
        callback = NeMoModelCheckpoint(dirpath=tmp_path, save_top_k=-1)
        callback.set_checkpoint_unfinished_marker = MagicMock()
        trainer = SimpleNamespace(
            global_step=9,
            callbacks=[],
            save_checkpoint=MagicMock(side_effect=RuntimeError("save failed")),
        )

        with pytest.raises(RuntimeError, match="save failed"):
            callback._save_checkpoint(trainer, str(tmp_path / "model.ckpt"))

        group.on_save_checkpoint_start.assert_called_once_with(9, async_save=False)
        group.on_save_checkpoint_success.assert_not_called()
        group.on_save_checkpoint_failure.assert_called_once_with(9)
        group.on_save_checkpoint_end.assert_called_once_with()


def test_export_all_symbols():
    from nemo.lightning.one_logger_callback import __all__

    assert __all__ == ["OneLoggerNeMoCallback"]
