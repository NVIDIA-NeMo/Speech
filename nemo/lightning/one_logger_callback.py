# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneLogger lifecycle tracing for NeMo Speech."""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from typing import Any

import torch

from nemo.lightning.base_callback import BaseCallback
from nemo.lightning.speech_throughput import ThroughputValue, select_throughput_policy
from nemo.utils import logging

__all__ = ["OneLoggerNeMoCallback"]

OneLoggerConfig = None
OneLoggerErrorHandlingStrategy = None
Attributes = None
Event = None
on_app_end = None
on_app_start = None
TrainingTelemetryProvider = None

_SPAN_MODEL_INIT = "nemo_speech.model_initialization"
_SPAN_DATALOADER_INIT = "nemo_speech.data_loader_initialization"
_SPAN_OPTIMIZER_INIT = "nemo_speech.optimizer_initialization"
_SPAN_CHECKPOINT_LOAD = "nemo_speech.checkpoint_load"
_SPAN_CHECKPOINT_SAVE = "nemo_speech.checkpoint_save"
_SPAN_TRAINING = "nemo_speech.training"
_SPAN_VALIDATION = "nemo_speech.validation"
_EVENT_THROUGHPUT = "nemo_speech.throughput"
_MIN_THROUGHPUT_REPORT_INTERVAL = 100


def _get_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _get_world_size() -> int:
    for name in ("WORLD_SIZE", "SLURM_NTASKS"):
        value = _get_env_int(name)
        if value is not None and value > 0:
            return value
    return 1


def _get_job_name() -> str:
    return os.environ.get("EXP_NAME") or os.environ.get("SLURM_JOB_NAME") or "nemo-run"


def _get_rank() -> int | None:
    """Resolve a global rank from torchrun, Slurm, or Lightning launcher metadata."""
    rank = _get_env_int("RANK")
    if rank is None:
        rank = _get_env_int("SLURM_PROCID")
    if rank is None:
        local_rank = _get_env_int("LOCAL_RANK")
        if local_rank is not None:
            node_rank = _get_env_int("NODE_RANK") or 0
            local_world_size = _get_env_int("LOCAL_WORLD_SIZE") or 1
            rank = node_rank * local_world_size + local_rank
    return rank


def _should_enable_for_current_rank() -> bool:
    """Enable only after explicit opt-in, and export from rank zero."""
    enabled = os.environ.get("NEMO_ONE_LOGGER_ENABLED")
    if enabled is None or enabled.lower() not in {"1", "true", "yes", "on"}:
        return False
    rank = _get_rank()
    return rank in (None, 0)


def get_one_logger_init_config() -> dict[str, Any]:
    """Build modality-independent OneLogger application configuration."""
    return {
        "application_name": "nemo-speech",
        "session_tag_or_fn": _get_job_name(),
        "enable_for_current_rank": _should_enable_for_current_rank(),
        "world_size_or_fn": _get_world_size(),
    }


class OneLoggerNeMoCallback(BaseCallback):
    """Trace NeMo Speech lifecycle and dynamic, modality-specific throughput.

    NeMo Speech batches may contain waveforms, frames, text, codec tokens, or a
    mixture of modalities, often with dynamic batching and gradient
    accumulation. This adapter measures the actual work in each batch and keeps
    different modalities as independent units instead of deriving them from a
    static batch size or sequence length.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        init_config = get_one_logger_init_config()
        self.enabled_for_current_rank = init_config["enable_for_current_rank"]
        self._provider = None
        self._application_span = None
        self._active_spans = []
        self._throughput_policy = None
        self._throughput_interval = 1
        self._throughput_sums = {}
        self._throughput_batches = 0
        self._throughput_examples = 0
        self._throughput_optimizer_steps = 0
        self._throughput_start_global_step = None
        self._throughput_last_global_step = None
        self._throughput_step_aligned = False
        self._optimizer_step_in_batch = False
        self._throughput_started_at = None
        self._throughput_cuda_start = None
        self._throughput_cuda_device = None
        self._pending_throughput = None
        self._initialized = True
        if not self.enabled_for_current_rank:
            return

        try:
            _load_one_logger()
            init_config["error_handling_strategy"] = (
                OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR
            )
            provider = TrainingTelemetryProvider.instance()
            provider.with_base_config(OneLoggerConfig(**init_config)).with_export_config().configure_provider()
            self._provider = provider
            self._application_span = on_app_start()
        # OneLogger setup must not make model construction or training fail.
        except Exception as error:  # noqa: BLE001
            logging.warning("Disabling OneLogger after initialization failed: %s", error)
            self.enabled_for_current_rank = False
            self._provider = None
            self._application_span = None

    def _start_span(self, name: str, attributes: Attributes | None = None) -> None:
        if self._provider is None:
            return
        span = self._provider.recorder.start(name, span_attributes=attributes)
        if span is not None:
            self._active_spans.append((name, span))

    def _current_span(self, name: str):
        return next((span for span_name, span in reversed(self._active_spans) if span_name == name), None)

    def _stop_span(self, name: str) -> None:
        if self._provider is None:
            return
        for index in range(len(self._active_spans) - 1, -1, -1):
            span_name, span = self._active_spans[index]
            if span_name == name:
                self._active_spans.pop(index)
                self._provider.recorder.stop(span)
                return

    def on_app_end(self) -> None:
        if self._provider is None:
            return
        self._report_throughput(block=True)
        while self._active_spans:
            _, span = self._active_spans.pop()
            self._provider.recorder.stop(span)
        on_app_end()

    def on_model_init_start(self) -> None:
        self._start_span(_SPAN_MODEL_INIT)

    def on_model_init_end(self) -> None:
        self._stop_span(_SPAN_MODEL_INIT)

    def on_dataloader_init_start(self) -> None:
        self._start_span(_SPAN_DATALOADER_INIT)

    def on_dataloader_init_end(self) -> None:
        self._stop_span(_SPAN_DATALOADER_INIT)

    def on_optimizer_init_start(self) -> None:
        self._start_span(_SPAN_OPTIMIZER_INIT)

    def on_optimizer_init_end(self) -> None:
        self._stop_span(_SPAN_OPTIMIZER_INIT)

    def on_load_checkpoint_start(self) -> None:
        self._start_span(_SPAN_CHECKPOINT_LOAD)

    def on_load_checkpoint_end(self) -> None:
        self._stop_span(_SPAN_CHECKPOINT_LOAD)

    def on_save_checkpoint_start(self, global_step: int, async_save: bool = False) -> None:
        self._report_throughput(global_step=global_step)
        self._start_span(
            _SPAN_CHECKPOINT_SAVE,
            Attributes({"global_step": global_step, "asynchronous": async_save}),
        )

    def on_save_checkpoint_success(self, global_step: int) -> None:
        self._checkpoint_event("nemo_speech.checkpoint_save_success", global_step)

    def on_save_checkpoint_failure(self, global_step: int) -> None:
        self._checkpoint_event("nemo_speech.checkpoint_save_failure", global_step)

    def _checkpoint_event(self, name: str, global_step: int) -> None:
        if self._provider is None:
            return
        span = self._current_span(_SPAN_CHECKPOINT_SAVE) or self._application_span
        if span is not None:
            self._provider.recorder.event(span, Event.create(name, Attributes({"global_step": global_step})))

    def on_save_checkpoint_end(self, global_step: int | None = None) -> None:
        del global_step
        self._publish_ready_throughput()
        self._stop_span(_SPAN_CHECKPOINT_SAVE)

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        """Start training telemetry and select the model throughput policy."""
        self._start_span(_SPAN_TRAINING)
        if self._provider is None:
            return
        try:
            self._throughput_policy = select_throughput_policy(pl_module)
        # Telemetry must never interrupt training, including downstream custom policies.
        except Exception as error:  # noqa: BLE001
            self._disable_throughput(error)
            return
        self._throughput_interval = _get_throughput_interval(trainer)
        self._pending_throughput = None
        self._reset_throughput_window()

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        self._report_throughput(trainer=trainer, block=True)
        self._stop_span(_SPAN_TRAINING)

    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
        """Poll completed telemetry and start a nonblocking timing window."""
        del batch, batch_idx
        self._optimizer_step_in_batch = False
        if self._throughput_policy is None:
            return
        self._publish_ready_throughput()
        if self._pending_throughput is not None:
            return
        if self._throughput_batches:
            return
        global_step = int(getattr(trainer, "global_step", 0))
        self._throughput_start_global_step = global_step
        self._throughput_last_global_step = global_step
        self._throughput_started_at = time.monotonic()
        device = getattr(pl_module, "device", None)
        if isinstance(device, torch.device) and device.type == "cuda":
            try:
                with torch.cuda.device(device):
                    self._throughput_cuda_device = device
                    self._throughput_cuda_start = torch.cuda.Event(enable_timing=True)
                    self._throughput_cuda_start.record()
            except RuntimeError as error:
                self._disable_throughput(error)

    def on_before_optimizer_step(self, trainer: Any, pl_module: Any, optimizer: Any) -> None:
        """Count optimizer-step boundaries so gradient accumulation is reflected."""

        del trainer, pl_module, optimizer
        if self._throughput_policy is not None and self._throughput_started_at is not None:
            self._throughput_optimizer_steps += 1
            self._optimizer_step_in_batch = True

    def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Collect deferred work measurements for the completed training batch."""
        del outputs, batch_idx
        if self._throughput_policy is None or self._pending_throughput is not None:
            return
        try:
            measurements = self._throughput_policy.measure(pl_module, batch)
            if not measurements:
                self._reset_throughput_window()
                return
            for name, value in measurements.items():
                if not isinstance(value, ThroughputValue):
                    value = ThroughputValue(value)
                self._throughput_sums.setdefault(name, []).append(value)

            num_examples = self._throughput_policy.num_examples(pl_module, batch)
            if num_examples is None:
                self._throughput_examples = None
            elif not isinstance(num_examples, int) or isinstance(num_examples, bool) or num_examples < 0:
                raise TypeError("ThroughputPolicy.num_examples must return a non-negative int or None")
            elif self._throughput_examples is not None:
                self._throughput_examples += num_examples

            current_global_step = int(getattr(trainer, "global_step", 0))
            global_step_advanced = (
                self._throughput_last_global_step is not None
                and current_global_step > self._throughput_last_global_step
            )
            self._throughput_last_global_step = current_global_step
            self._throughput_batches += 1
            accumulation = int(getattr(trainer, "accumulate_grad_batches", 1) or 1)
            optimizer_boundary = self._optimizer_step_in_batch or global_step_advanced or accumulation <= 1
            self._throughput_step_aligned = optimizer_boundary
            if self._throughput_batches >= self._throughput_interval and optimizer_boundary:
                self._report_throughput(trainer=trainer)
        # Telemetry must never interrupt training, including downstream custom policies.
        except Exception as error:  # noqa: BLE001
            self._disable_throughput(error)

    def on_validation_start(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        self._report_throughput(trainer=trainer)
        self._start_span(_SPAN_VALIDATION)

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        self._publish_ready_throughput()
        self._stop_span(_SPAN_VALIDATION)

    def _report_throughput(self, trainer: Any = None, global_step: int | None = None, block: bool = False) -> None:
        if self._provider is None:
            return
        try:
            if self._throughput_batches and self._throughput_started_at is not None:
                self._close_throughput_window(trainer, global_step)
            self._publish_ready_throughput(block=block)
        # OneLogger/exporter and custom-policy failures must not interrupt training.
        except Exception as error:  # noqa: BLE001
            self._disable_throughput(error)

    def _close_throughput_window(self, trainer: Any, global_step: int | None) -> None:
        host_ended_at = time.monotonic()
        end_global_step = int(global_step if global_step is not None else getattr(trainer, "global_step", 0))
        observed_steps = (
            max(end_global_step - self._throughput_start_global_step, 0)
            if self._throughput_start_global_step is not None
            else 0
        )
        attributes = {
            "scope": "rank",
            "rank": _get_rank() or 0,
            "policy": self._throughput_policy.name,
            "global_step": end_global_step,
            "window_batches": self._throughput_batches,
            "window_optimizer_steps": max(self._throughput_optimizer_steps, observed_steps),
        }
        if self._throughput_examples is not None:
            attributes["examples"] = self._throughput_examples
        totals = _deferred_totals(self._throughput_sums)
        if self._throughput_cuda_start is None:
            self._publish_throughput(
                attributes,
                totals,
                host_ended_at - self._throughput_started_at,
                step_aligned=self._throughput_step_aligned,
            )
        else:
            with torch.cuda.device(self._throughput_cuda_device):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
            self._pending_throughput = _PendingThroughputWindow(
                attributes,
                totals,
                self._throughput_cuda_start,
                end,
                self._throughput_step_aligned,
            )
        self._reset_throughput_window()

    def _publish_ready_throughput(self, block: bool = False) -> None:
        try:
            pending = self._pending_throughput
            if pending is None:
                return
            if block:
                pending.end.synchronize()
            elif not pending.end.query():
                return
            self._pending_throughput = None
            duration = pending.start.elapsed_time(pending.end) / 1000.0
            self._publish_throughput(
                pending.attributes,
                pending.totals,
                duration,
                step_aligned=pending.step_aligned,
            )
        # Polling, materialization, and exporter failures must not interrupt training.
        except Exception as error:  # noqa: BLE001
            self._disable_throughput(error)

    def _publish_throughput(
        self,
        attributes: dict[str, Any],
        totals: dict[str, Any],
        duration: float,
        step_aligned: bool,
    ) -> None:
        if duration <= 0:
            return
        attributes["window_seconds"] = duration
        optimizer_steps = attributes["window_optimizer_steps"]
        examples = attributes.get("examples")
        if examples is not None:
            attributes["examples_per_second"] = examples / duration
            if step_aligned and optimizer_steps > 0:
                attributes["mean_batch_size"] = examples / optimizer_steps
        for name, value in totals.items():
            value = value.detach().item() if torch.is_tensor(value) else float(value)
            attributes[name] = value
            attributes[f"{name}_per_second"] = value / duration
            if step_aligned and optimizer_steps > 0:
                attributes[f"{name}_per_step"] = value / optimizer_steps
        span = self._current_span(_SPAN_TRAINING) or self._application_span
        if span is not None:
            self._provider.recorder.event(span, Event.create(_EVENT_THROUGHPUT, Attributes(attributes)))

    def _disable_throughput(self, error: Exception) -> None:
        logging.warning("Disabling OneLogger throughput reporting after measurement failed: %s", error)
        self._throughput_policy = None
        self._pending_throughput = None
        self._reset_throughput_window()

    def _reset_throughput_window(self) -> None:
        self._throughput_sums = {}
        self._throughput_batches = 0
        self._throughput_examples = 0
        self._throughput_optimizer_steps = 0
        self._throughput_start_global_step = None
        self._throughput_last_global_step = None
        self._throughput_step_aligned = False
        self._optimizer_step_in_batch = False
        self._throughput_started_at = None
        self._throughput_cuda_start = None
        self._throughput_cuda_device = None


@dataclass
class _PendingThroughputWindow:
    attributes: dict[str, Any]
    totals: dict[str, Any]
    start: Any
    end: Any
    step_aligned: bool


def _load_one_logger() -> None:
    """Load optional OneLogger bindings only after explicit opt-in."""

    global Attributes, Event, OneLoggerConfig, OneLoggerErrorHandlingStrategy
    global TrainingTelemetryProvider, on_app_end, on_app_start

    bindings = (
        OneLoggerConfig,
        OneLoggerErrorHandlingStrategy,
        Attributes,
        Event,
        on_app_end,
        on_app_start,
        TrainingTelemetryProvider,
    )
    if all(binding is not None for binding in bindings):
        return

    config = importlib.import_module("nv_one_logger.api.config")
    attributes = importlib.import_module("nv_one_logger.core.attributes")
    event = importlib.import_module("nv_one_logger.core.event")
    callbacks = importlib.import_module("nv_one_logger.training_telemetry.api.callbacks")
    provider = importlib.import_module("nv_one_logger.training_telemetry.api.training_telemetry_provider")

    if OneLoggerConfig is None:
        OneLoggerConfig = config.OneLoggerConfig
    if OneLoggerErrorHandlingStrategy is None:
        OneLoggerErrorHandlingStrategy = config.OneLoggerErrorHandlingStrategy
    if Attributes is None:
        Attributes = attributes.Attributes
    if Event is None:
        Event = event.Event
    if on_app_end is None:
        on_app_end = callbacks.on_app_end
    if on_app_start is None:
        on_app_start = callbacks.on_app_start
    if TrainingTelemetryProvider is None:
        TrainingTelemetryProvider = provider.TrainingTelemetryProvider


def _get_throughput_interval(trainer: Any) -> int:
    configured = _get_env_int("NEMO_ONE_LOGGER_THROUGHPUT_INTERVAL")
    if configured is not None and configured > 0:
        return configured
    logging_interval = int(getattr(trainer, "log_every_n_steps", 1) or 1)
    return max(logging_interval, _MIN_THROUGHPUT_REPORT_INTERVAL)


def _deferred_totals(measurements: dict[str, list[ThroughputValue]]) -> dict[str, Any]:
    totals = {}
    for name, values in measurements.items():
        host_total = 0.0
        tensor_groups = {}
        for measurement in values:
            value = measurement.value
            if torch.is_tensor(value):
                key = (value.device, value.dtype, measurement.scale)
                tensor_groups.setdefault(key, []).append(value.detach().reshape(-1))
            elif isinstance(value, (tuple, list)):
                host_total += sum(value) * measurement.scale
            else:
                host_total += float(value) * measurement.scale
        tensor_totals = [torch.cat(tensors).sum() * scale for (_, _, scale), tensors in tensor_groups.items()]
        if tensor_totals:
            total = sum(tensor_totals[1:], tensor_totals[0]) + host_total
            totals[name] = total
        else:
            totals[name] = host_total
    return totals
