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
"""CUDA graphs for the online (windowed) inference path of :class:`ParallelExpertEncoder`.

Online inference encodes long audio window by window, carrying the Sortformer speaker state forward. Two pieces of it
are replayed from graphs, both owned by one :class:`OnlineGraphs` that the encoder holds as ``online_graphs``
(``None``, the default, runs eagerly):

* ASR windows (:class:`_ASRWindowGraphs`): the ASR encoder over a batch of equal-width windows, keyed by shape; the
  batch is split over graph batch sizes (``OnlineGraphs.split``), the last piece padded with copies of its first row.
* Streaming diarization steps (:class:`_StreamingStepGraphs`): one synchronous Sortformer step, keyed by the shapes of
  its state and window. The state shapes change until the speaker cache and FIFO fill, then repeat, so a few keys cover
  any length. Rows are padded to ``OnlineGraphs.bucket(rows)``.

A graph is captured at the second sight of its key, or at the first inside :meth:`OnlineGraphs.capturing` (a
warmup). Only synchronous streaming (``async_streaming=False``, no speaker permutation) can be captured, since the
asynchronous cache update reads a device scalar on the host; such steps run eagerly. Whole-forward graphs of short
audio are not here: the vLLM plugin captures them (``speechlm2/vllm/salm/encoder_graphs.py``) and drives this
object through :class:`OnlineGraphs`.
"""
from __future__ import annotations

import contextlib
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from nemo.collections.common.parts.optional_cuda_graphs import SharedGraphBuffers, capture_cuda_graph, copy_padded_rows

__all__ = ["OnlineGraphs"]

logger = logging.getLogger(__name__)


@dataclass
class OnlineGraphs:
    """The CUDA graphs of one encoder's online inference, with the settings they share.

    Args:
        split (Optional[Callable]): ``split(rows, frames) -> sizes`` covering a batch of ASR windows with graph batch
            sizes (the last piece padded). Without it, windows are graphed at their exact row count.
        bucket (Optional[Callable]): ``bucket(rows) -> size``, the graph batch size a streaming Sortformer step's rows
            are padded to, or ``None`` to run the step eagerly. Without it, steps are graphed at their exact row count.
        buffers (Optional[SharedGraphBuffers]): Static input and output buffers shared by graphs that never replay
            concurrently, or ``None`` to give every graph its own.
        max_graphs (int): The most graphs each cache keeps; new keys past it run eagerly.
        pool: CUDA graph memory pool handle to capture into, or ``None`` for a private pool per cache.
        enabled (bool): Replay graphs; when false, online inference runs eagerly and nothing is captured.
    """

    split: Optional[Callable[[int, int], list[int]]] = None
    bucket: Optional[Callable[[int], Optional[int]]] = None
    buffers: Optional[SharedGraphBuffers] = None
    max_graphs: int = 16
    pool: Any = None
    enabled: bool = True
    capture_now: bool = field(default=False, init=False)
    _asr_windows: Optional[_ASRWindowGraphs] = field(default=None, init=False, repr=False)
    _diar_steps: Optional[_StreamingStepGraphs] = field(default=None, init=False, repr=False)

    def asr_windows(self, encoder, rows: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode equal-width ASR windows with ``encoder``, replaying graphs where available."""
        if self._asr_windows is None:
            self._asr_windows = _ASRWindowGraphs(self.max_graphs, self.pool, self.buffers)
        return self._asr_windows(encoder, rows, lengths, self.split, capture_now=self.capture_now)

    def diar_step(self, step, state, chunk: torch.Tensor, chunk_length: torch.Tensor, extra_key: tuple):
        """Run one streaming Sortformer step, replaying a graph where available (see :class:`_StreamingStepGraphs`)."""
        if self._diar_steps is None:
            self._diar_steps = _StreamingStepGraphs(self.max_graphs, self.pool, self.buffers)
        return self._diar_steps(step, state, chunk, chunk_length, extra_key, self.bucket, capture_now=self.capture_now)

    @contextlib.contextmanager
    def capturing(self):
        """Capture every new graph key at its first sight (normally the second), e.g. during a warmup."""
        previous, self.capture_now = self.capture_now, True
        try:
            yield
        finally:
            self.capture_now = previous

    @contextlib.contextmanager
    def paused(self):
        """Run online inference eagerly inside the block, e.g. while another graph is being captured."""
        previous, self.enabled = self.enabled, False
        try:
            yield
        finally:
            self.enabled = previous

    def complete_asr_windows(self, encoder, batch_sizes: Callable[[int], list[int]]) -> None:
        """Capture the ASR window graphs at every batch size for every window shape captured so far.

        Args:
            encoder (nn.Module): ASR encoder taking ``audio_signal`` and ``length``.
            batch_sizes (Callable): ``batch_sizes(frames) -> sizes`` for windows of ``frames`` input frames.
        """
        if self._asr_windows is not None:
            self._asr_windows.capture_batch_sizes(encoder, batch_sizes)

    def count(self) -> int:
        """Number of captured graphs (ASR windows and streaming Sortformer steps)."""
        return sum(len(cache.graphs) for cache in (self._asr_windows, self._diar_steps) if cache is not None)

    def clear(self) -> None:
        """Drop every captured graph."""
        self._asr_windows = self._diar_steps = None


class _GraphCache:
    """CUDA graphs keyed by input shapes, captured lazily (inference only).

    A key gets a graph the second time it is seen (or at once with ``capture_now``), so shapes that occur only once
    stay eager. Past ``max_graphs`` graphs, new keys run eagerly. If a capture fails, everything runs eagerly from
    then on.

    Args:
        max_graphs (int): Maximum number of captured graphs.
        pool: CUDA graph memory pool handle to capture into, or ``None`` for a private pool.
        buffers (Optional[SharedGraphBuffers]): Buffers for the graphs' static inputs and outputs shared with other
            graphs that never replay concurrently, or ``None`` to give every graph its own.
    """

    def __init__(self, max_graphs: int = 16, pool=None, buffers: Optional[SharedGraphBuffers] = None):
        self.max_graphs = max_graphs
        self.enabled = True
        self.seen = set()
        self.graphs = {}
        self.pool = pool
        self.buffers = buffers if buffers is not None else _PrivateGraphBuffers()

    def _lookup(self, key, capture, capture_now: bool = False):
        """
        Return the graph entry for ``key``, capturing it with ``capture()`` when due, or ``None`` to run eagerly.

        Args:
            key: Hashable input signature.
            capture (Callable): Captures the graph and returns its entry.
            capture_now (bool): Capture at the first sight of ``key``.

        Returns:
            The cached entry, or ``None``.
        """
        if key in self.graphs:
            return self.graphs[key]
        if not self.enabled or len(self.graphs) >= self.max_graphs or (key not in self.seen and not capture_now):
            self.seen.add(key)
            return None
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        try:
            self.graphs[key] = capture()
        except Exception as error:  # noqa: BLE001 - a failed capture must not break inference
            logger.warning(f"CUDA graph capture of {type(self).__name__} failed; running eagerly: {error}")
            self.enabled = False
            self.graphs.clear()
            return None
        return self.graphs[key]


class _ASRWindowGraphs(_GraphCache):
    """CUDA graphs of the ASR encoder over batches of equal-width windows.

    ``split(rows, frames)`` covers a batch with graph batch sizes; each piece is padded to its size with copies of its
    first row, so a few graphs serve any window count. Outputs are cloned, so callers own them.
    """

    def __call__(self, encoder, rows: torch.Tensor, lengths: torch.Tensor, split=None, capture_now: bool = False):
        """
        Encode ``rows`` with ``encoder``, replaying a graph per piece where one is available.

        Args:
            encoder (nn.Module): ASR encoder taking ``audio_signal`` and ``length``.
            rows (torch.Tensor): Window mel features shaped ``(N, D, T)``.
            lengths (torch.Tensor): Valid window frame counts shaped ``(N,)``.
            split (Optional[Callable]): ``split(rows, frames) -> sizes``; ``None`` graphs the exact row count.
            capture_now (bool): Capture new keys at their first sight.

        Returns:
            torch.Tensor: Encoded windows shaped ``(N, D_out, T_out)``.
        """
        sizes = split(rows.shape[0], rows.shape[-1]) if split is not None else [rows.shape[0]]
        outputs, start = [], 0
        for size in sizes:
            piece, piece_lengths = rows[start : start + size], lengths[start : start + size]
            n = piece.shape[0]
            start += n
            key = (size, tuple(piece.shape[1:]), piece.dtype, lengths.dtype)
            entry = self._lookup(key, lambda: self._capture(encoder, size, piece, piece_lengths), capture_now)
            if entry is None:
                outputs.append(encoder(audio_signal=piece, length=piece_lengths)[0])
                continue
            graph, static_rows, static_lengths, encoded = entry
            copy_padded_rows(static_rows, piece)
            copy_padded_rows(static_lengths, piece_lengths)
            graph.replay()
            outputs.append(encoded[:n].clone())
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)

    def capture_batch_sizes(self, encoder, batch_sizes) -> None:
        """
        Capture every window shape seen so far at every graph batch size.

        Args:
            encoder (nn.Module): ASR encoder taking ``audio_signal`` and ``length``.
            batch_sizes (Callable): ``batch_sizes(frames) -> sizes`` for windows of ``frames`` input frames.
        """
        for (_, shape, rows_dtype, lengths_dtype), entry in list(self.graphs.items()):
            device = entry[1].device
            for size in batch_sizes(shape[-1]):
                rows = torch.zeros(size, *shape, device=device, dtype=rows_dtype)
                lengths = torch.full((size,), shape[-1], device=device, dtype=lengths_dtype)
                self(encoder, rows, lengths, capture_now=True)

    def _capture(self, encoder, size: int, piece: torch.Tensor, piece_lengths: torch.Tensor):
        """Capture ``encoder`` on static ``size``-row buffers shaped like ``piece``."""
        static_rows = self.buffers.view("asr_windows", (size, *piece.shape[1:]), piece.dtype, piece.device)
        static_rows.copy_(piece[:1].expand_as(static_rows))
        static_lengths = piece_lengths[:1].expand(size).clone()

        def run():
            encoded, _ = encoder(audio_signal=static_rows, length=static_lengths)
            return self.buffers.store("asr_encoded", encoded)

        graph, encoded = capture_cuda_graph(run, pool=self.pool)
        return graph, static_rows, static_lengths, encoded


class _StreamingStepGraphs(_GraphCache):
    """CUDA graphs of one synchronous streaming Sortformer step, keyed by every input shape (inference only).

    The state shapes change while the speaker cache fills, then stay fixed, so a few keys cover any number of steps.
    With a ``bucket`` function, the rows are padded to a graph batch size with copies of the first row, so a few
    batch sizes cover any number of rows. A replay copies the state and the window into the graph's input buffers and
    clones its outputs, so callers own every returned tensor.
    """

    _STATE_FIELDS = (
        "spkcache",
        "spkcache_lengths",
        "spkcache_preds",
        "fifo",
        "fifo_lengths",
        "fifo_preds",
        "mean_sil_emb",
        "n_sil_frames",
    )

    def __call__(
        self,
        step,
        state,
        chunk: torch.Tensor,
        chunk_length: torch.Tensor,
        extra_key: tuple,
        bucket=None,
        capture_now: bool = False,
    ):
        """
        Run ``step`` eagerly or replay its graph for this input signature.

        Args:
            step (Callable): Function mapping ``(state, chunk, chunk_length)`` to ``(state, predictions)``.
            state (Any): Sortformer streaming state.
            chunk (torch.Tensor): Window mel features, time-major, shaped ``(B, T, D)``.
            chunk_length (torch.Tensor): Valid window frame counts shaped ``(B,)``.
            extra_key (tuple): Hashable step options that are part of the graph key.
            bucket (Optional[Callable]): ``bucket(rows) -> size`` to pad the rows to, or ``None`` to run eagerly;
                without it, the exact row count is graphed.
            capture_now (bool): Capture a new key at its first sight.

        Returns:
            tuple[Any, torch.Tensor]: Updated streaming state and this window's speaker predictions.
        """
        rows = chunk.shape[0]
        size = bucket(rows) if bucket is not None else rows
        if size is None:
            return step(state, chunk, chunk_length)
        tensors = {name: getattr(state, name) for name in self._STATE_FIELDS if getattr(state, name, None) is not None}
        if size != rows:
            tensors = {name: _pad_rows(t, size) for name, t in tensors.items()}
            chunk, chunk_length = _pad_rows(chunk, size), _pad_rows(chunk_length, size)
        key = (
            extra_key,
            bool(state.spkcache_compressed),
            tuple((name, tuple(t.shape), t.dtype) for name, t in tensors.items()),
            tuple(chunk.shape),
            chunk.dtype,
            tuple(chunk_length.shape),
        )
        entry = self._lookup(key, lambda: self._capture(step, state, tensors, chunk, chunk_length), capture_now)
        if entry is None:
            return step(state, chunk[:rows], chunk_length[:rows])
        graph, static_tensors, static_chunk, static_length, out_state, out_preds = entry
        for name, t in tensors.items():
            static_tensors[name].copy_(t)
        static_chunk.copy_(chunk)
        static_length.copy_(chunk_length)
        graph.replay()
        new_state = copy.copy(out_state)
        for name, value in vars(out_state).items():
            if isinstance(value, torch.Tensor):
                setattr(new_state, name, (value[:rows] if value.dim() and value.shape[0] == size else value).clone())
        return new_state, out_preds[:rows].clone()

    def _capture(self, step, state, tensors, chunk, chunk_length):
        """
        Capture ``step`` on static copies of its inputs.

        Args:
            step (Callable): Function mapping ``(state, chunk, chunk_length)`` to ``(state, predictions)``.
            state (Any): Sortformer streaming state used as the template.
            tensors (dict[str, torch.Tensor]): State tensors to give static input buffers.
            chunk (torch.Tensor): Window mel features, time-major, shaped ``(B, T, D)``.
            chunk_length (torch.Tensor): Valid window frame counts shaped ``(B,)``.

        Returns:
            tuple: The graph, its static state tensors, chunk and length buffers, and its output state and predictions.
        """
        static_tensors = {name: t.clone() for name, t in tensors.items()}
        static_chunk = self.buffers.view("diar_chunk", tuple(chunk.shape), chunk.dtype, chunk.device).copy_(chunk)
        static_length = chunk_length.clone()

        def run():
            # The step reassigns state attributes, so each run gets its own shallow copy of the state.
            fresh = copy.copy(state)
            for name, t in static_tensors.items():
                setattr(fresh, name, t)
            out_state, preds = step(fresh, static_chunk, static_length)
            # Unchanged state tensors are the static inputs; the rest move to shared buffers.
            static = {id(t) for t in static_tensors.values()}
            for name, value in list(vars(out_state).items()):
                if isinstance(value, torch.Tensor) and id(value) not in static:
                    setattr(out_state, name, self.buffers.store(f"diar_state.{name}", value))
            return out_state, self.buffers.store("diar_preds", preds)

        graph, (out_state, out_preds) = capture_cuda_graph(run, pool=self.pool)
        return graph, static_tensors, static_chunk, static_length, out_state, out_preds


def _pad_rows(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Pad ``tensor`` to ``size`` rows with copies of its first row."""
    return torch.cat([tensor, tensor[:1].expand(size - tensor.shape[0], *tensor.shape[1:])])


class _PrivateGraphBuffers(SharedGraphBuffers):
    """SharedGraphBuffers that give every graph its own buffers."""

    def view(self, slot: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=device)
