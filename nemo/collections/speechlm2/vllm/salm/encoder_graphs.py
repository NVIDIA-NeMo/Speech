# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""CUDA graphs for the ParallelExpertEncoder perception forward in the vLLM plugin.

vLLM encodes the audio of the requests it admits in one engine step, typically a few dozen short clips. At that
batch size the perception forward is bound by kernel launches and Python: ~1.8k kernels per call, with the GPU idle
more than half the time. This module captures the forward (everything after the mel preprocessor) into CUDA
graphs over a grid of (batch, frames) buckets and replays them.

Audio up to ``max_windows`` online-inference windows replays one graph for the whole forward (the windowed path's
control flow depends only on shapes). Longer audio, of any length, runs the windowed path with each of its pieces
replayed from graphs: the ASR encoder over batches of equal-width windows, and every streaming diarization step. Its
last window is padded to a multiple of ``last_window_bucket`` frames, so a few window widths and diarization state
shapes cover every length. Rows are padded to the bucket with copies of the first row and frames with zeros, which
the encoder masks by length like the padding of any shorter row in a batch.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.common.parts.optional_cuda_graphs import SharedGraphBuffers, capture_cuda_graph, copy_padded_rows

logger = logging.getLogger(__name__)

# Config keys (checkpoint ``config.json`` or vLLM ``--hf-overrides``) that set the graph buckets, mapped to
# EncoderGraphRunner arguments.
ENCODER_GRAPH_CONFIG_KEYS = {
    "encoder_cuda_graph_batch_sizes": "batch_sizes",
    "encoder_cuda_graph_frame_bucket": "frame_bucket",
    "encoder_cuda_graph_max_windows": "max_windows",
    "encoder_cuda_graph_long_audio": "long_audio",
    "encoder_cuda_graph_last_window_bucket": "last_window_bucket",
}
# Capture compiles the encoder's attention once per shape class it meets (batch size 1, sequence length, default
# dtype of each branch). Every bucket is captured in one call, which can need more variants than dynamo's default
# recompile limit; past the limit, compiled functions run uncompiled code that cannot be captured.
_CAPTURE_RECOMPILE_LIMIT = 64
# Up to this many rows, one graph replay costs about the same as one of a smaller batch, so splitting the rows over
# several graphs only adds replays.
_MIN_SPLIT_ROWS = 8
# Graph cache limit for the long-audio window graphs; the start-up capture needs about a dozen per batch size.
_MAX_ONLINE_GRAPHS = 1024
# Diarization step graphs are small, so long audio pads their rows less than the ASR graphs': every count up to 16,
# then sizes at most 1.5x apart.
_STEP_BATCH_SIZES = (*range(1, 17), 24, 32, 48, 64, 96, 128, 192, 256)


class EncoderGraphRunner:
    """Replays the perception forward from CUDA graphs captured per (batch, frames) bucket.

    Args:
        perception: Perception module whose encoder is a ParallelExpertEncoder (``chunk_feat_len`` and
            ``online_inference()``).
        batch_sizes: Graph batch sizes; a call's rows are split over them, largest first.
        frame_bucket: Graph length granularity in feature frames. Buckets never cross a window boundary.
        max_windows: Longest audio graphed as one whole forward, in online-inference windows.
        long_audio: Graph longer audio window by window; without it, longer audio runs eagerly.
        last_window_bucket: Granularity of the last window's padding for long audio, in feature frames; coarser
            buckets need fewer graphs and pad more. At least the encoder's right context.
        compile_layers: ``torch.compile`` every transformer block of the encoder first. This fuses the
            memory-bound LayerNorm, RoPE and residual operations that dominate the encoder's GPU time at these
            batch sizes, and changes numerics at bf16 rounding level.

    Set ``ctc_timestamp_inputs`` before the first call to also capture the encoder's CTC timestamp inputs, so
    that calls can return them.
    """

    def __init__(
        self,
        perception: nn.Module,
        batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        frame_bucket: int = 256,
        max_windows: int = 2,
        long_audio: bool = True,
        last_window_bucket: int = 1000,
        compile_layers: bool = False,
    ):
        if not batch_sizes or any(int(b) < 1 for b in batch_sizes):
            raise ValueError(f"batch_sizes must be positive integers, got {batch_sizes!r}")
        if frame_bucket < 1 or max_windows < 1 or last_window_bucket < 1:
            raise ValueError(
                "frame_bucket, max_windows and last_window_bucket must be positive, "
                f"got {frame_bucket}, {max_windows}, {last_window_bucket}"
            )
        self.perception = perception
        encoder = perception.encoder
        self.window = int(encoder.chunk_feat_len)
        # Window graphs need a ParallelExpertEncoder's online-inference graph caches.
        self.long_audio = long_audio and hasattr(encoder, "capturing_online_graphs")
        # The window before the last one must keep its full right context.
        right_context = int(getattr(encoder, "right_ctx_feat_len", 0))
        self.last_window_bucket = min(max(last_window_bucket, right_context), self.window)
        self.max_frames = self.window * max_windows
        self.frame_bucket = frame_bucket
        self.batch_sizes = sorted({int(b) for b in batch_sizes})
        # Graphs over more frames hold fewer rows, so no graph needs more memory than the largest batch of
        # single windows.
        self.max_rows_x_frames = self.batch_sizes[-1] * self.window
        # Every long-audio call holds at most the largest batch size of rows.
        self.step_batch_sizes = [b for b in _STEP_BATCH_SIZES if b < self.batch_sizes[-1]] + self.batch_sizes[-1:]
        self.graphs: dict[tuple[int, int], tuple] = {}
        # Calls replay one graph at a time, so all graphs read their features from, and write their embeddings to,
        # shared buffers; the pool then holds one graph's intermediates rather than every graph's outputs.
        self.buffers = SharedGraphBuffers()
        self.pool = None
        self.enabled = True
        self.ctc_timestamp_inputs = False
        # Audio longer than the whole-forward graphs runs the online path, whose ASR windows and streaming
        # diarization steps replay their own graphs, from the same pool. Without long_audio it stays eager.
        encoder.online_cuda_graphs = self.long_audio
        encoder.online_graph_split = self._split
        encoder.online_graph_bucket = self._bucket
        encoder.online_graph_buffers = self.buffers
        encoder.online_max_graphs = _MAX_ONLINE_GRAPHS
        if compile_layers:
            for name, module in perception.encoder.named_modules():
                if isinstance(module, nn.ModuleList) and name.endswith("layers"):
                    for layer in module:
                        layer.compile(dynamic=True)

    def _sizes_for(self, frames: int) -> list[int]:
        return [b for b in self.batch_sizes if b * frames <= self.max_rows_x_frames] or self.batch_sizes[:1]

    def _bucket(self, rows: int) -> int | None:
        """Smallest diarization step batch size holding ``rows``, or ``None`` past the largest."""
        return next((b for b in self.step_batch_sizes if b >= rows), None)

    def _split(self, rows: int, frames: int) -> list[int]:
        """Cover ``rows`` with graph batch sizes: the largest that fits, padding only the last piece.

        Fewer than ``_MIN_SPLIT_ROWS`` remaining rows go into one graph rather than several smaller ones, since a
        replay costs about the same at any batch size up to that.
        """
        sizes = self._sizes_for(frames)
        pieces = []
        while rows > 0:
            fit = [b for b in sizes if b <= rows]
            if fit and (rows >= _MIN_SPLIT_ROWS or fit[-1] == rows):
                pieces.append(fit[-1])
            else:
                pieces.append(next((b for b in sizes if b >= rows), sizes[-1]))
            rows -= pieces[-1]
        return pieces

    def _frame_bucket(self, frames: int) -> int:
        # Never pad across a window boundary: the padded input then runs the same windows as eager inference.
        return min(
            math.ceil(frames / self.window) * self.window, math.ceil(frames / self.frame_bucket) * self.frame_bucket
        )

    def _long_frames(self, frames: int) -> int:
        """Padded length of long audio: whole windows, then the last window rounded up to ``last_window_bucket``."""
        full = (math.ceil(frames / self.window) - 1) * self.window
        last = math.ceil((frames - full) / self.last_window_bucket) * self.last_window_bucket
        return full + min(last, self.window)

    def _forward(self, x: torch.Tensor, x_len: torch.Tensor):
        kwargs = {"return_ctc_timestamp_inputs": True} if self.ctc_timestamp_inputs else {}
        with self.perception.encoder.online_inference():
            out = self.perception(processed_signal=x, processed_signal_length=x_len, **kwargs)
        return out if self.ctc_timestamp_inputs else (*out, None)

    def _capture_all(self, n_feat: int, dtype: torch.dtype) -> None:
        """Capture every bucket at the first call, largest first so smaller graphs reuse the pool's blocks."""
        lengths = {self._frame_bucket(t) for t in range(1, self.max_frames + 1)}  # every bucket a call can reach
        with self._without_window_graphs():
            for frames in sorted(lengths, reverse=True):
                for rows in reversed(self._sizes_for(frames)):
                    self._capture(rows, frames, n_feat, dtype)
        if self.long_audio:
            try:
                self._capture_long(n_feat, dtype)
            except Exception as error:  # noqa: BLE001 - keep the whole-forward graphs
                logger.warning("Long-audio encoder CUDA graph capture failed; long audio runs eagerly: %s", error)
                self._disable_long_audio()

    def _capture_long(self, n_feat: int, dtype: torch.dtype) -> None:
        """Capture the long-audio graphs by encoding silence at every step batch size, window count and last width.

        The diarization state shapes change until the speaker cache and FIFO have filled, then repeat with the cache
        update period, so window counts up to that point meet every step shape a longer input would. The ASR window
        graphs are then completed for every batch size of every window width.
        """
        device = next(self.perception.parameters()).device
        encoder = self.perception.encoder
        encoder.online_graph_pool = self.pool
        chunk = int(encoder.online_inference_length)
        filled = math.ceil((int(encoder.diar_spkcache_len) + int(encoder.diar_fifo_len)) / chunk)
        period = math.ceil(max(int(encoder.diar_spkcache_update_period), chunk) / chunk)
        first = self.max_frames // self.window + 1
        lasts = {
            min(t, self.window)
            for t in range(self.last_window_bucket, self.window + self.last_window_bucket, self.last_window_bucket)
        }
        with encoder.capturing_online_graphs():
            for rows in reversed(self.step_batch_sizes):
                for windows in range(first, max(first, filled + period + 2) + 1):
                    for frames in sorted({self._long_frames((windows - 1) * self.window + t) for t in lasts}):
                        x = torch.zeros(rows, n_feat, frames, device=device, dtype=dtype)
                        self._forward(x, torch.full((rows,), frames, device=device, dtype=torch.long))
            encoder.complete_online_asr_graphs(self._sizes_for)
        logger.info("Captured %d long-audio encoder CUDA graphs", self.long_graph_count())

    def _disable_long_audio(self) -> None:
        if self.long_audio:
            self.long_audio = False
            self.perception.encoder.reset_online_graphs()

    @contextlib.contextmanager
    def _without_window_graphs(self):
        """The whole-forward graphs replay without Python, so their warmup runs must not grow the window graphs."""
        if not self.long_audio:
            yield
            return
        encoder = self.perception.encoder
        encoder.online_cuda_graphs = False
        try:
            yield
        finally:
            encoder.online_cuda_graphs = True

    def long_graph_count(self) -> int:
        """Number of captured graphs that serve long audio window by window."""
        return self.perception.encoder.online_graph_count() if self.long_audio else 0

    def _capture(self, rows: int, frames: int, n_feat: int, dtype: torch.dtype) -> None:
        device = next(self.perception.parameters()).device
        x = self.buffers.view("features", (rows, n_feat, frames), dtype, device).zero_()
        x_len = torch.full((rows,), frames, device=device, dtype=torch.long)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()

        def run():
            emb, emb_len, states = self._forward(x, x_len)
            emb = self.buffers.store("embeddings", emb)
            if states is not None:
                fields = {name: self.buffers.store(name, getattr(states, name)) for name in _tensor_fields(states)}
                states = dataclasses.replace(states, **fields)
            return emb, emb_len, states

        graph, (emb, emb_len, states) = capture_cuda_graph(run, pool=self.pool)
        self.graphs[rows, frames] = (graph, x, x_len, emb, emb_len, states)
        logger.info("Captured encoder CUDA graph for %d rows x %d frames", rows, frames)

    @torch.no_grad()
    def __call__(
        self, audio_signal: torch.Tensor, audio_lengths: torch.Tensor, return_ctc_timestamp_inputs: bool = False
    ):
        """Encode a batch of waveforms.

        Returns per-row embeddings, and with ``return_ctc_timestamp_inputs`` also the batch's
        ``CTCTimestampInputs``, like the perception forward. Returns None (encode eagerly) when the audio is longer
        than the whole-forward graphs without ``long_audio``, capture failed, or the inputs are requested from graphs
        captured without them.
        """
        if not self.enabled or (return_ctc_timestamp_inputs and not self.ctc_timestamp_inputs):
            return None
        processed, processed_len = self.perception.maybe_preprocess_audio(audio_signal, audio_lengths, None, None)
        n_feat, frames = processed.shape[1], processed.shape[2]
        if frames > self.max_frames and not self.long_audio:
            return None
        if not self.graphs:
            try:
                with _compile_limit_for_capture():
                    self._capture_all(n_feat, processed.dtype)
                # The capture's eager runs leave cached blocks behind; hand them back to the serving engine.
                torch.cuda.empty_cache()
            except Exception as error:  # noqa: BLE001 - a failed capture must not break inference
                logger.warning("Encoder CUDA graph capture failed; encoding eagerly from now on: %s", error)
                self.graphs.clear()
                self.pool = None
                self.enabled = False
                self._disable_long_audio()
                torch.cuda.empty_cache()
                return None
        if frames > self.max_frames:
            return self._call_long(processed, processed_len, return_ctc_timestamp_inputs)
        bucket = self._frame_bucket(frames)
        out, pieces, start = [], [], 0
        for size in self._split(processed.shape[0], bucket):
            rows = processed[start : start + size]
            n = rows.shape[0]
            graph, x, x_len, emb, emb_len, states = self.graphs[size, bucket]
            x[:, :, frames:].zero_()
            copy_padded_rows(x[:, :, :frames], rows)
            copy_padded_rows(x_len, processed_len[start : start + n])
            start += n
            graph.replay()
            # One host sync per piece for the lengths; the clones leave the static output buffer free for reuse.
            out.extend(emb[i, :length].clone() for i, length in enumerate(emb_len[:n].tolist()))
            if return_ctc_timestamp_inputs:
                pieces.append(_rows(states, n))
        return (out, _cat_states(pieces)) if return_ctc_timestamp_inputs else out

    def _call_long(self, processed: torch.Tensor, processed_len: torch.Tensor, return_ctc_timestamp_inputs: bool):
        """Encode audio longer than the whole-forward graphs through the online path and its window graphs.

        All rows run together (up to the largest batch size per pass): the ASR windows of every row are split over the
        graph batch sizes, and only the diarization steps pad the rows to one.
        """
        frames, padded = processed.shape[2], self._long_frames(processed.shape[2])
        out, pieces = [], []
        for start in range(0, processed.shape[0], self.batch_sizes[-1]):
            x = F.pad(processed[start : start + self.batch_sizes[-1]], (0, padded - frames))
            emb, emb_len, states = self._forward(x, processed_len[start : start + self.batch_sizes[-1]])
            out.extend(emb[i, :length] for i, length in enumerate(emb_len.tolist()))
            if return_ctc_timestamp_inputs:
                pieces.append(states)
        return (out, _cat_states(pieces)) if return_ctc_timestamp_inputs else out


def _tensor_fields(states) -> list[str]:
    return [f.name for f in dataclasses.fields(states) if isinstance(getattr(states, f.name), torch.Tensor)]


def _cat_states(pieces: list):
    """Concatenate the rows of several ``CTCTimestampInputs``."""
    if len(pieces) == 1:
        return pieces[0]
    tensors = {name: torch.cat([getattr(p, name) for p in pieces]) for name in _tensor_fields(pieces[0])}
    return dataclasses.replace(pieces[0], **tensors)


def _rows(states, n: int):
    """Copies of the first ``n`` rows of every tensor in captured ``CTCTimestampInputs``."""
    return dataclasses.replace(states, **{name: getattr(states, name)[:n].clone() for name in _tensor_fields(states)})


def encoder_graph_options(config) -> dict:
    """Return the EncoderGraphRunner arguments set in a model config (see ``ENCODER_GRAPH_CONFIG_KEYS``).

    Args:
        config: HF model config, after ``--hf-overrides``.

    Returns:
        Keyword arguments for EncoderGraphRunner; unset keys are left out, so the runner's defaults apply.
    """
    options = {}
    for key, argument in ENCODER_GRAPH_CONFIG_KEYS.items():
        value = getattr(config, key, None)
        if value is not None:
            if argument == "batch_sizes":
                value = tuple(int(b) for b in value)
            elif argument == "long_audio":
                value = bool(value)
            else:
                value = int(value)
            options[argument] = value
    return options


def _compile_limit_for_capture():
    """Raise dynamo's per-function recompile limit to ``_CAPTURE_RECOMPILE_LIMIT`` while the graphs are captured."""
    config = torch._dynamo.config
    name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    return config.patch(**{name: max(getattr(config, name), _CAPTURE_RECOMPILE_LIMIT)})
