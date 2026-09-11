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

import copy
import logging
import math
import os
import time
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import torch
from omegaconf import open_dict

from nemo.collections.asr.parts.preprocessing.features import normalize_batch

if TYPE_CHECKING:
    from nemo.collections.asr.models import SortformerEncLabelModel


class SortformerStreamingSession:
    """Stateful raw-audio session for a fixed batch of streaming Sortformer inputs.

    Each row owns independent waveform buffering, progress, finalization, and speaker-cache state. Incoming waveform
    chunks may have different valid lengths, and the session uses Sortformer's asynchronous streaming update so idle,
    active, and finalized rows can coexist in one batch.

    Args:
        model: Streaming ``SortformerEncLabelModel`` in evaluation mode.
        batch_size: Fixed number of independent audio streams owned by the session.
        max_speakers: Number of enabled speaker channels for every stream. A scalar applies to the complete batch; a
            sequence or tensor supplies one value per row. By default, every model speaker channel is enabled.
    """

    def __init__(
        self,
        model: "SortformerEncLabelModel",
        batch_size: int = 1,
        max_speakers: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
    ):
        if not model.streaming_mode:
            raise ValueError("SortformerStreamingSession requires a model with streaming_mode=True")
        if model.training:
            raise ValueError("SortformerStreamingSession requires an evaluation model; call model.eval() first")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        self.model = model
        self.batch_size = batch_size
        self._max_speakers = max_speakers
        self.device = model.device
        self._normalization = model.preprocessor.featurizer.normalize
        self._preprocessor = copy.deepcopy(model.preprocessor).to(self.device).eval()
        self._preprocessor.featurizer.normalize = None
        self._preprocessor.featurizer.dither = 0.0
        self._preprocessor.featurizer.pad_to = 0

        self._hop_length = self._preprocessor.hop_length
        self._n_fft = self._preprocessor.featurizer.n_fft
        self._stft_margin_frames = math.ceil((self._n_fft // 2 + 1) / self._hop_length) + 1
        self._chunk_frames = model.sortformer_modules.chunk_len * model.encoder.subsampling_factor
        self._left_context_frames = model.sortformer_modules.chunk_left_context * model.encoder.subsampling_factor
        self._right_context_frames = model.sortformer_modules.chunk_right_context * model.encoder.subsampling_factor
        self.reset()

    @torch.inference_mode()
    def diarize_step(
        self,
        audio_chunks: torch.Tensor,
        audio_chunk_lengths: Optional[torch.Tensor] = None,
        is_final: Union[bool, torch.Tensor] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Consume one raw-audio chunk per stream and return newly committed speaker probabilities.

        Args:
            audio_chunks: Float waveforms with shape ``(batch_size, max_num_samples)``. A one-dimensional tensor is
                also accepted when ``batch_size=1``.
            audio_chunk_lengths: Valid sample count for each padded row, with shape ``(batch_size,)``. If omitted,
                every row uses the complete waveform width.
            is_final: Boolean finalization mask with shape ``(batch_size,)`` or one boolean applied to every row.
                Finalized rows can remain in later calls with a zero audio length while other rows continue.

        Returns:
            padded_probabilities: Newly committed probabilities with shape
                ``(batch_size, max_new_frames, num_speakers)``.
            probability_lengths: Valid output frames for each row, with shape ``(batch_size,)``.
        """
        audio_chunks, audio_chunk_lengths = self._validate_audio_chunks(audio_chunks, audio_chunk_lengths)
        final_mask = self._validate_final_mask(is_final)
        input_lengths = audio_chunk_lengths.tolist()
        final_flags = final_mask.tolist()

        for stream_index, chunk_length in enumerate(input_lengths):
            if self._finalized[stream_index] and chunk_length > 0:
                raise RuntimeError(
                    f"Cannot supply audio to finalized stream {stream_index}; call reset() before reusing the session"
                )
            if chunk_length > 0:
                chunk = audio_chunks[stream_index, :chunk_length]
                self._audio_buffers[stream_index] = torch.cat([self._audio_buffers[stream_index], chunk])
                self._received_samples[stream_index] += chunk_length

        emitted = [[] for _ in range(self.batch_size)]
        while True:
            ready_groups = self._get_ready_groups(final_flags)
            if not ready_groups:
                break

            for (left_offset, right_offset), requests in ready_groups.items():
                processed_signal, processed_signal_length = self._extract_feature_batch(requests)
                empty_preds = processed_signal.new_zeros((self.batch_size, 0, self.model.sortformer_modules.n_spk))
                self.streaming_state, chunk_preds = self.model.forward_streaming_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    streaming_state=self.streaming_state,
                    total_preds=empty_preds,
                    left_offset=left_offset,
                    right_offset=right_offset,
                    async_streaming=True,
                )

                for stream_index, _, _, central_end in requests:
                    committed_feature_frames = central_end - self._next_feature_frames[stream_index]
                    output_length = math.ceil(committed_feature_frames / self.model.output_subsampling_factor)
                    if output_length > chunk_preds.shape[1]:
                        raise RuntimeError(
                            "Streaming model returned fewer prediction frames than required: "
                            f"needed {output_length}, got {chunk_preds.shape[1]}"
                        )
                    emitted[stream_index].append(chunk_preds[stream_index, :output_length])
                    self._next_feature_frames[stream_index] = central_end

        self._compact_audio_buffers()
        for stream_index, is_final_stream in enumerate(final_flags):
            if is_final_stream:
                self._finalized[stream_index] = True

        return self._pad_emitted_outputs(emitted)

    def reset(self) -> None:
        """Clear every stream's buffered audio and initialize a fresh batched asynchronous model state."""
        self.streaming_state = self.model.sortformer_modules.init_streaming_state(
            batch_size=self.batch_size,
            async_streaming=True,
            device=self.device,
            max_speakers=self._max_speakers,
        )
        self._max_speakers = self.streaming_state.max_speakers
        self._audio_buffers = [torch.empty(0, dtype=torch.float32, device=self.device) for _ in range(self.batch_size)]
        self._audio_buffer_starts = [0] * self.batch_size
        self._received_samples = [0] * self.batch_size
        self._next_feature_frames = [0] * self.batch_size
        self._finalized = [False] * self.batch_size

    def _get_ready_groups(self, final_flags: List[bool]):
        ready_groups = {}
        for stream_index in range(self.batch_size):
            if self._finalized[stream_index]:
                continue
            available_frames = self._available_feature_frames(stream_index, is_final=final_flags[stream_index])
            next_frame = self._next_feature_frames[stream_index]
            if next_frame >= available_frames:
                continue

            central_end = min(next_frame + self._chunk_frames, available_frames)
            if not final_flags[stream_index] and central_end + self._right_context_frames > available_frames:
                continue

            feature_start = max(0, next_frame - self._left_context_frames)
            feature_end = min(central_end + self._right_context_frames, available_frames)
            offsets = (next_frame - feature_start, feature_end - central_end)
            ready_groups.setdefault(offsets, []).append((stream_index, feature_start, feature_end, central_end))
        return ready_groups

    def _available_feature_frames(self, stream_index: int, is_final: bool) -> int:
        received_samples = self._received_samples[stream_index]
        sample_count = torch.tensor(received_samples, device=self.device)
        offline_frames = int(self._preprocessor.featurizer.get_seq_len(sample_count).item())
        if is_final:
            return max(0, offline_frames)

        stable_samples = received_samples - self._n_fft // 2
        if stable_samples < 0:
            return 0
        stable_frames = stable_samples // self._hop_length + 1
        return max(0, min(offline_frames, stable_frames))

    def _compact_audio_buffers(self) -> None:
        for stream_index in range(self.batch_size):
            first_needed_frame = max(
                0,
                self._next_feature_frames[stream_index] - self._left_context_frames - self._stft_margin_frames,
            )
            first_needed_sample = first_needed_frame * self._hop_length
            drop_samples = first_needed_sample - self._audio_buffer_starts[stream_index]
            if drop_samples > 0:
                self._audio_buffers[stream_index] = self._audio_buffers[stream_index][drop_samples:].clone()
                self._audio_buffer_starts[stream_index] = first_needed_sample

    def _extract_feature_batch(self, requests):
        audio_segments = []
        audio_lengths = []
        local_feature_ranges = []
        for stream_index, feature_start, feature_end, _ in requests:
            segment_start_frame = max(0, feature_start - self._stft_margin_frames)
            segment_start_sample = segment_start_frame * self._hop_length
            segment_end_sample = min(
                self._received_samples[stream_index],
                (feature_end + self._stft_margin_frames) * self._hop_length,
            )
            buffer_start = segment_start_sample - self._audio_buffer_starts[stream_index]
            buffer_end = segment_end_sample - self._audio_buffer_starts[stream_index]
            audio_segment = self._audio_buffers[stream_index][buffer_start:buffer_end]
            audio_segments.append(audio_segment)
            audio_lengths.append(audio_segment.shape[0])
            local_start = feature_start - segment_start_frame
            local_feature_ranges.append((local_start, local_start + feature_end - feature_start))

        padded_audio = torch.nn.utils.rnn.pad_sequence(audio_segments, batch_first=True)
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long, device=self.device)
        features, feature_lengths = self._preprocessor(input_signal=padded_audio, length=audio_lengths)

        feature_windows = []
        window_lengths = []
        for request_index, (local_start, local_end) in enumerate(local_feature_ranges):
            if feature_lengths[request_index] < local_end:
                raise RuntimeError(
                    "Streaming preprocessor returned fewer feature frames than required: "
                    f"needed {local_end}, got {feature_lengths[request_index].item()}"
                )
            window = features[request_index, :, local_start:local_end].transpose(0, 1)
            feature_windows.append(window)
            window_lengths.append(window.shape[0])

        active_features = torch.nn.utils.rnn.pad_sequence(feature_windows, batch_first=True).transpose(1, 2)
        active_lengths = torch.tensor(window_lengths, dtype=torch.long, device=self.device)
        if self._normalization:
            active_features, _, _ = normalize_batch(active_features, active_lengths, self._normalization)
        active_features = active_features.transpose(1, 2)

        batch_features = active_features.new_zeros(
            (self.batch_size, active_features.shape[1], active_features.shape[2])
        )
        batch_lengths = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        for request_index, (stream_index, _, _, _) in enumerate(requests):
            feature_length = window_lengths[request_index]
            batch_features[stream_index, :feature_length] = active_features[request_index, :feature_length]
            batch_lengths[stream_index] = feature_length
        return batch_features, batch_lengths

    def _pad_emitted_outputs(self, emitted):
        num_speakers = self.model.sortformer_modules.n_spk
        dtype = next(self.model.parameters()).dtype
        row_outputs = [
            torch.cat(row, dim=0) if row else torch.zeros((0, num_speakers), dtype=dtype, device=self.device)
            for row in emitted
        ]
        output_lengths = torch.tensor([row.shape[0] for row in row_outputs], dtype=torch.long, device=self.device)
        max_output_length = max((row.shape[0] for row in row_outputs), default=0)
        padded_outputs = torch.zeros(
            (self.batch_size, max_output_length, num_speakers), dtype=dtype, device=self.device
        )
        for stream_index, row in enumerate(row_outputs):
            padded_outputs[stream_index, : row.shape[0]] = row
        return padded_outputs, output_lengths

    def _validate_audio_chunks(self, audio_chunks, audio_chunk_lengths):
        if not isinstance(audio_chunks, torch.Tensor):
            raise TypeError(f"audio_chunks must be a torch.Tensor, got {type(audio_chunks).__name__}")
        if audio_chunks.ndim == 1 and self.batch_size == 1:
            audio_chunks = audio_chunks.unsqueeze(0)
        if audio_chunks.ndim != 2 or audio_chunks.shape[0] != self.batch_size:
            raise ValueError(
                f"audio_chunks must have batch dimension {self.batch_size} and shape "
                f"({self.batch_size}, max_num_samples); got {tuple(audio_chunks.shape)}"
            )
        audio_chunks = audio_chunks.detach().to(device=self.device, dtype=torch.float32)

        if audio_chunk_lengths is None:
            audio_chunk_lengths = torch.full(
                (self.batch_size,), audio_chunks.shape[1], dtype=torch.long, device=self.device
            )
        elif not isinstance(audio_chunk_lengths, torch.Tensor):
            raise TypeError(
                f"audio_chunk_lengths must be a torch.Tensor or None, got {type(audio_chunk_lengths).__name__}"
            )
        elif audio_chunk_lengths.shape != (self.batch_size,):
            raise ValueError(
                f"audio_chunk_lengths must have shape ({self.batch_size},), got {tuple(audio_chunk_lengths.shape)}"
            )
        elif torch.is_floating_point(audio_chunk_lengths) or audio_chunk_lengths.dtype == torch.bool:
            raise TypeError("audio_chunk_lengths must contain integers")
        else:
            audio_chunk_lengths = audio_chunk_lengths.to(device=self.device, dtype=torch.long)

        if torch.any(audio_chunk_lengths < 0) or torch.any(audio_chunk_lengths > audio_chunks.shape[1]):
            raise ValueError(f"audio_chunk_lengths must be between 0 and {audio_chunks.shape[1]} samples")
        return audio_chunks, audio_chunk_lengths

    def _validate_final_mask(self, is_final):
        if isinstance(is_final, bool):
            return torch.full((self.batch_size,), is_final, dtype=torch.bool, device=self.device)
        if not isinstance(is_final, torch.Tensor):
            raise TypeError(f"is_final must be a boolean or torch.Tensor, got {type(is_final).__name__}")
        if is_final.dtype != torch.bool or is_final.shape != (self.batch_size,):
            raise ValueError(f"is_final must be boolean with shape ({self.batch_size},), got {tuple(is_final.shape)}")
        return is_final.to(device=self.device)


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
