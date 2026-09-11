# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import logging
import os
import time
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from omegaconf import open_dict

if TYPE_CHECKING:
    from nemo.collections.asr.models import SortformerEncLabelModel


def configure_output_subsampling_factor(
    diar_model: "SortformerEncLabelModel",
    output_subsampling_factor: Optional[int],
) -> int:
    """
    Apply an inference-time output resolution override and return the effective factor.

    Args:
        diar_model (SortformerEncLabelModel): Model whose output resolution is configured.
        output_subsampling_factor (Optional[int]): Requested output factor in 10 ms feature frames. If ``None``,
            the model's current factor is retained.

    Returns:
        effective_output_subsampling_factor (int): Applied output subsampling factor.
    """
    if output_subsampling_factor is None:
        return diar_model.output_subsampling_factor
    if type(output_subsampling_factor) is not int or output_subsampling_factor < 1:
        raise ValueError(f"output_subsampling_factor must be a positive integer, got {output_subsampling_factor}")
    native_output_factor = 1 if diar_model.high_resolution else diar_model.encoder.subsampling_factor
    if output_subsampling_factor % native_output_factor != 0:
        logging.warning(
            f"output_subsampling_factor={output_subsampling_factor} must be an integer multiple of the model's "
            f"native subsampling factor ({native_output_factor}). Using {native_output_factor} instead."
        )
        output_subsampling_factor = native_output_factor

    diar_model.output_subsampling_factor = output_subsampling_factor
    with open_dict(diar_model._cfg):
        diar_model._cfg.output_subsampling_factor = output_subsampling_factor
    return output_subsampling_factor


class InferenceProfiler:
    """Measure inference wall time and streaming-step components without including evaluation."""

    _STREAMING_STEP_SECTIONS = (
        "pre_encode",
        "state_concat",
        "frontend_encoder",
        "forward_infer",
        "prediction_mask",
        "state_update",
        "high_resolution_extract",
        "downsample_preds",
    )

    def __init__(self, model: "SortformerEncLabelModel"):
        """
        Initialize inference profiling for a Sortformer model.

        Args:
            model (SortformerEncLabelModel): Model whose inference methods will be profiled.
        """
        self.model = model
        self.forward_time = 0.0
        self.preprocessor_time = 0.0
        self.forward_calls = 0
        self.preprocessor_calls = 0
        self.section_times: Dict[str, float] = {}
        self.section_calls: Dict[str, int] = {}
        self._cuda_events = {}
        self._installed = False

    def _synchronize(self):
        """Synchronize pending CUDA work before recording wall-clock time."""
        if self.model.device.type == 'cuda':
            torch.cuda.synchronize(self.model.device)

    def _flush_cuda_events(self):
        """Accumulate completed CUDA event timings and clear the pending events."""
        for section, events in self._cuda_events.items():
            elapsed = sum(start.elapsed_time(end) for start, end in events) / 1000
            self.section_times[section] = self.section_times.get(section, 0.0) + elapsed
        self._cuda_events.clear()

    def _section_wrapper(self, section, function):
        """
        Wrap a callable to record its invocation count and elapsed time.

        Args:
            section (str): Profiling section under which measurements are accumulated.
            function (Callable): Callable to profile.

        Returns:
            timed_function (Callable): Wrapped callable that records profiling measurements.
        """

        @wraps(function)
        def timed_function(*args, **kwargs):
            self.section_calls[section] = self.section_calls.get(section, 0) + 1
            if self.model.device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                stream = torch.cuda.current_stream(self.model.device)
                start.record(stream)
                try:
                    return function(*args, **kwargs)
                finally:
                    end.record(stream)
                    self._cuda_events.setdefault(section, []).append((start, end))

            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.section_times[section] = self.section_times.get(section, 0.0) + (time.perf_counter() - start)

        return timed_function

    def _install_section(self, instance, method_name, section):
        """
        Replace an instance method with a profiled wrapper.

        Args:
            instance (object): Object whose bound method is replaced.
            method_name (str): Name of the method to wrap.
            section (str): Profiling section under which measurements are accumulated.
        """
        original_method = getattr(instance, method_name)
        setattr(instance, method_name, self._section_wrapper(section, original_method))

    def install(self):
        """Install profiling wrappers by monkey-patching model methods; repeated calls are no-ops."""
        if self._installed:
            return
        self._installed = True

        original_process_signal = self.model.process_signal
        original_forward = self.model.forward
        sortformer_modules = self.model.sortformer_modules

        self._install_section(self.model, "_call_pre_encode", "pre_encode")
        self._install_section(sortformer_modules, "concat_and_pad", "state_concat")
        self._install_section(sortformer_modules, "concat_embs", "state_concat")
        if hasattr(self.model.frontend_encoder, "forward"):
            self._install_section(self.model.frontend_encoder, "forward", "frontend_encoder")
        else:
            self._install_section(self.model, "frontend_encoder", "frontend_encoder")
        self._install_section(self.model, "forward_infer", "forward_infer")
        self._install_section(sortformer_modules, "apply_mask_to_preds", "prediction_mask")
        self._install_section(sortformer_modules, "streaming_update_async", "state_update")
        self._install_section(sortformer_modules, "streaming_update", "state_update")
        self._install_section(
            self.model,
            "_extract_async_high_resolution_chunk_preds",
            "high_resolution_extract",
        )
        self._install_section(sortformer_modules, "downsample_preds", "downsample_preds")
        self._install_section(sortformer_modules, "_compress_spkcache", "cache_compress")
        self._install_section(self.model, "forward_streaming_step", "streaming_step")

        def timed_process_signal(*args, **kwargs):
            self._synchronize()
            start = time.perf_counter()
            try:
                return original_process_signal(*args, **kwargs)
            finally:
                self._synchronize()
                self.preprocessor_time += time.perf_counter() - start
                self.preprocessor_calls += 1

        def timed_forward(*args, **kwargs):
            self._synchronize()
            start = time.perf_counter()
            try:
                return original_forward(*args, **kwargs)
            finally:
                self._synchronize()
                self.forward_time += time.perf_counter() - start
                self.forward_calls += 1
                self._flush_cuda_events()

        self.model.process_signal = timed_process_signal
        self.model.forward = timed_forward

    def log_summary(self, audio_duration: float):
        """
        Log accumulated inference timing measurements.

        Args:
            audio_duration (float): Duration of processed audio in seconds.
        """
        self._synchronize()
        self._flush_cuda_events()
        if audio_duration <= 0 or self.forward_time <= 0:
            logging.warning(
                f"Cannot summarize inference profile with audio_duration={audio_duration} "
                f"and forward_time={self.forward_time}."
            )
            return

        main_inference_time = max(0.0, self.forward_time - self.preprocessor_time)
        preprocessor_percent = 100 * self.preprocessor_time / self.forward_time
        main_inference_percent = 100 * main_inference_time / self.forward_time
        logging.info(
            "Inference profile: "
            f"audio={audio_duration:.2f}s, model_forward={self.forward_time:.3f}s "
            f"(RTF={self.forward_time / audio_duration:.6f}, {audio_duration / self.forward_time:.2f}x realtime), "
            f"preprocessor={self.preprocessor_time:.3f}s ({preprocessor_percent:.2f}%, "
            f"RTF={self.preprocessor_time / audio_duration:.6f}), "
            f"main_inference={main_inference_time:.3f}s ({main_inference_percent:.2f}%, "
            f"RTF={main_inference_time / audio_duration:.6f}), "
            f"calls={self.forward_calls}"
        )

        streaming_step_time = self.section_times.get("streaming_step", 0.0)
        if streaming_step_time <= 0:
            return

        measured_step_time = sum(self.section_times.get(section, 0.0) for section in self._STREAMING_STEP_SECTIONS)
        other_step_time = max(0.0, streaming_step_time - measured_step_time)
        logging.info(
            f"Streaming step profile: total={streaming_step_time:.3f}s, "
            f"calls={self.section_calls.get('streaming_step', 0)}, "
            f"per_call={1000 * streaming_step_time / self.section_calls['streaming_step']:.3f}ms"
        )
        for section in self._STREAMING_STEP_SECTIONS:
            section_time = self.section_times.get(section, 0.0)
            if section_time <= 0:
                continue
            calls = self.section_calls.get(section, 0)
            logging.info(
                f"  {section}: total={section_time:.3f}s, "
                f"step={100 * section_time / streaming_step_time:.2f}%, "
                f"calls={calls}, per_call={1000 * section_time / calls:.3f}ms"
            )
        logging.info(f"  other: total={other_step_time:.3f}s, step={100 * other_step_time / streaming_step_time:.2f}%")

        cache_compress_time = self.section_times.get("cache_compress", 0.0)
        state_update_time = self.section_times.get("state_update", 0.0)
        if cache_compress_time > 0 and state_update_time > 0:
            calls = self.section_calls["cache_compress"]
            logging.info(
                f"  cache_compress (inside state_update): total={cache_compress_time:.3f}s, "
                f"state_update={100 * cache_compress_time / state_update_time:.2f}%, "
                f"calls={calls}, per_call={1000 * cache_compress_time / calls:.3f}ms"
            )


def get_prediction_cache_metadata(cfg, diar_model, infer_audio_rttm_dict) -> Dict:
    """
    Describe inputs and inference settings that affect cached prediction tensors.

    Args:
        cfg (DiarizationConfig): Inference configuration containing model, manifest, and streaming settings.
        diar_model (SortformerEncLabelModel): Sortformer model containing speaker and score-boost settings.
        infer_audio_rttm_dict (Dict): Recordings to process, keyed in inference order.

    Returns:
        metadata (Dict): Cache schema, input identities, and inference settings.
    """
    model_path = Path(cfg.model_path).expanduser().resolve()
    manifest_path = Path(cfg.dataset_manifest).expanduser().resolve()
    model_stat = model_path.stat()
    manifest_stat = manifest_path.stat()
    modules = diar_model.sortformer_modules
    return {
        "version": 1,
        "model_path": str(model_path),
        "model_size": model_stat.st_size,
        "model_mtime_ns": model_stat.st_mtime_ns,
        "manifest_path": str(manifest_path),
        "manifest_size": manifest_stat.st_size,
        "manifest_mtime_ns": manifest_stat.st_mtime_ns,
        "recording_ids": list(infer_audio_rttm_dict),
        "num_speakers": int(diar_model._cfg.max_num_of_spks),
        "output_subsampling_factor": int(cfg.output_subsampling_factor),
        "precision": str(cfg.precision),
        "presort_manifest": bool(cfg.presort_manifest),
        "streaming_mode": bool(diar_model.streaming_mode),
        "async_streaming": bool(cfg.async_streaming),
        "async_pad_to_max": bool(cfg.async_pad_to_max),
        "async_desync_updates": bool(cfg.async_desync_updates),
        "chunk_len": int(cfg.chunk_len),
        "chunk_left_context": int(cfg.chunk_left_context),
        "chunk_right_context": int(cfg.chunk_right_context),
        "spkcache_len": int(cfg.spkcache_len),
        "spkcache_update_period": int(cfg.spkcache_update_period),
        "fifo_len": int(cfg.fifo_len),
        "strong_boost_rate": float(modules.strong_boost_rate),
        "weak_boost_rate": float(modules.weak_boost_rate),
        "scores_boost_latest": float(modules.scores_boost_latest),
    }


def validate_prediction_tensors(predictions, metadata: Dict) -> List[torch.Tensor]:
    """
    Validate cached prediction count and tensor dimensions.

    Args:
        predictions (List[torch.Tensor]): Predictions with shape ``(1, frames, speakers)``.
        metadata (Dict): Cache metadata containing ``recording_ids`` and ``num_speakers``.

    Returns:
        predictions (List[torch.Tensor]): Validated predictions normalized to a list.
    """
    if not isinstance(predictions, (list, tuple)):
        raise ValueError(f"Prediction cache must contain a list of tensors, got {type(predictions).__name__}")
    if len(predictions) != len(metadata["recording_ids"]):
        raise ValueError(
            f"Prediction cache contains {len(predictions)} recordings, "
            f"but the manifest contains {len(metadata['recording_ids'])}"
        )
    num_speakers = metadata["num_speakers"]
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, torch.Tensor) or prediction.ndim != 3 or prediction.shape[0] != 1:
            raise ValueError(f"Prediction {index} must have shape (1, frames, speakers)")
        if prediction.shape[-1] != num_speakers:
            raise ValueError(
                f"Prediction {index} has {prediction.shape[-1]} speakers, but the model expects {num_speakers}"
            )
    return list(predictions)


def load_prediction_tensors(tensor_path: str, expected_metadata: Dict) -> List[torch.Tensor]:
    """
    Load prediction tensors and reject caches created with incompatible settings.

    Args:
        tensor_path (str): Path to a prediction cache.
        expected_metadata (Dict): Metadata required for cache compatibility.

    Returns:
        predictions (List[torch.Tensor]): Validated tensors with shape ``(1, frames, speakers)``.
    """
    payload = torch.load(tensor_path, weights_only=True)
    if isinstance(payload, (list, tuple)):
        logging.warning("Loading a legacy prediction cache without metadata validation.")
        return validate_prediction_tensors(payload, expected_metadata)
    if not isinstance(payload, dict) or "metadata" not in payload or "predictions" not in payload:
        raise ValueError("Prediction cache must contain 'metadata' and 'predictions'")

    cached_metadata = payload["metadata"]
    mismatched_keys = [
        key for key, expected_value in expected_metadata.items() if cached_metadata.get(key) != expected_value
    ]
    if mismatched_keys:
        mismatch_list = ", ".join(mismatched_keys)
        raise ValueError(
            f"Prediction cache metadata does not match the current inference settings: {mismatch_list}. "
            "Use overwrite_preds_tensors=True or choose a different out_preds_tensors path."
        )
    return validate_prediction_tensors(payload["predictions"], expected_metadata)


def save_prediction_tensors(tensor_path: str, predictions: List[torch.Tensor], metadata: Dict) -> None:
    """
    Atomically save prediction tensors and their cache-compatibility metadata.

    Args:
        tensor_path (str): Destination path for the prediction cache.
        predictions (List[torch.Tensor]): Predictions with shape ``(1, frames, speakers)``.
        metadata (Dict): Cache-compatibility metadata saved with the predictions.
    """
    path = Path(tensor_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as tmp:
        temporary_path = Path(tmp.name)
    try:
        torch.save({"metadata": metadata, "predictions": predictions}, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
