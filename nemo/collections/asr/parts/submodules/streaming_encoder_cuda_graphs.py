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

"""CUDA graphs for the cache-aware streaming encoder step.

The cache-aware streaming encoder (see
:class:`~nemo.collections.asr.parts.mixins.streaming.StreamingEncoder`) launches on the order of a
thousand small kernels per streaming step, so low-latency streaming inference is host-bound: the
GPU is idle for most of the step while the host enqueues the launches. The steady-state step has
fully static shapes (the cache tensors are allocated at full size up front and only
``cache_last_channel_len`` values change between steps), so the whole step can be captured once into
a :class:`torch.cuda.CUDAGraph` and replayed with a single launch.

The RNNT/TDT label-looping decoders use the same idea. The encoder is simpler because it has no
data-dependent control flow, so plain CUDA graphs are enough (no conditional nodes, no
``cuda-python`` requirement). Outputs are written into stable buffers owned by the regular caching
allocator so that no consumer (for example the decoder's own CUDA graph) aliases the encoder graph's
private memory pool.

Non-uniform steps run eager: the first step(s) (different pre-encode cache handling), the final step
of an utterance (``keep_all_outputs=True``), and any call whose shapes/parameters do not match an
already-captured graph.
"""

from dataclasses import dataclass
from typing import Any

import torch

from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum

__all__ = ["CudaGraphsStreamingEncoderStep"]


def _cuda_autocast_enabled() -> bool:
    """Return whether CUDA autocast is active, tolerant of the PyTorch version.

    ``torch.is_autocast_enabled`` gained a device-type argument in newer PyTorch; on older
    versions it takes no argument and reports the CUDA autocast state. We try the device-specific
    call first and fall back to the no-argument form.
    """
    try:
        return torch.is_autocast_enabled("cuda")
    except TypeError:
        return torch.is_autocast_enabled()


@dataclass
class _CapturedStep:
    """A captured CUDA graph with its static input and stable output buffers."""

    graph: torch.cuda.CUDAGraph
    device: torch.device
    static_inputs: dict[str, torch.Tensor]
    stable_outputs: tuple[torch.Tensor, ...]

    def get_stable_outputs_copy(self) -> tuple[torch.Tensor, ...]:
        """Return outputs with storage that is not overwritten by the next graph replay."""
        return tuple(output.clone() for output in self.stable_outputs)


class CudaGraphsStreamingEncoderStep(WithOptionalCudaGraphs):
    """Optional CUDA-graph replay for ``StreamingEncoder.cache_aware_stream_step``.

    An instance of this class is attached to a streaming encoder by
    :meth:`~nemo.collections.asr.parts.mixins.streaming.StreamingEncoder.set_streaming_cuda_graphs`.
    Calls are dispatched through :meth:`stream_step`; steady-state calls (same shapes, same
    streaming parameters) are captured once and subsequently replayed with a single
    ``cudaGraphLaunch`` instead of ~10^3 individual kernel launches.

    Guarantees:
        * for the covered non-autocast configurations the graph path is expected to preserve eager
          execution semantics; the unit tests assert ``torch.equal`` (not ``allclose``) against
          eager (the replay is not an approximation with a numerical tolerance);
        * safe interop with the decoder's CUDA graphs (outputs live outside the graph pool);
        * automatic eager fallback for non-uniform steps and on any capture failure
          (unless a mode is forced via :meth:`force_cuda_graphs_mode`).

    Args:
        encoder: the streaming encoder (a ``StreamingEncoder`` + ``nn.Module``) to accelerate.
        warmup_steps: number of eager calls with an identical key before that key is captured;
            also serves as cuDNN/cuBLAS autotuning warmup. Must be >= 1.
        max_graphs: maximum number of distinct graphs kept alive (distinct shape/parameter
            keys); further keys run eager.
    """

    class CudaGraphsMode(PrettyStrEnum):
        FULL_GRAPH = "full_graph"  # capture the whole encoder step, fastest implementation
        NO_GRAPHS = "no_graphs"  # eager execution, for debugging/testing purposes

    def __init__(self, encoder, warmup_steps: int = 3, max_graphs: int = 8):
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")
        self.encoder = encoder
        self.warmup_steps = warmup_steps
        self.max_graphs = max_graphs
        self.cuda_graphs_mode: CudaGraphsStreamingEncoderStep.CudaGraphsMode | None = None
        self.cuda_graphs_allow_fallback: bool = True
        self.allow_cuda_graphs: bool = True
        self._graphs: dict[tuple, _CapturedStep] = {}
        self._key_counts: dict[tuple, int] = {}
        self.maybe_enable_cuda_graphs()

    # ------------------------------------------------------------------ WithOptionalCudaGraphs API
    def force_cuda_graphs_mode(self, mode: str | None):
        """Set the graphs mode explicitly (testing only); a forced mode disallows fallback."""
        self.cuda_graphs_mode = self.CudaGraphsMode(mode) if mode is not None else None
        self.cuda_graphs_allow_fallback = False
        self.reset_cuda_graphs_state()

    def maybe_enable_cuda_graphs(self) -> bool:
        """Enable CUDA graphs if allowed and CUDA is available; return True if state changed."""
        if self.cuda_graphs_mode is not None:
            return False
        if not self.allow_cuda_graphs or not torch.cuda.is_available():
            return False
        self.cuda_graphs_mode = self.CudaGraphsMode.FULL_GRAPH
        self.reset_cuda_graphs_state()
        return True

    def disable_cuda_graphs(self) -> bool:
        """Disable CUDA graphs (e.g. for a training epoch); return True if state changed."""
        if self.cuda_graphs_mode is None:
            return False
        self.cuda_graphs_mode = None
        self.reset_cuda_graphs_state()
        return True

    def reset_cuda_graphs_state(self):
        """Drop all captured graphs. Synchronizes the device first: destroying a graph's
        private memory pool while replays or other graphs are in flight is unsafe."""
        if self._graphs and torch.cuda.is_available():
            for device in {captured.device for captured in self._graphs.values()}:
                torch.cuda.synchronize(device)
        self._graphs.clear()
        self._key_counts.clear()

    # ------------------------------------------------------------------------------- dispatching
    def _make_key(self, signal: torch.Tensor, keep_all_outputs: bool, drop_extra_pre_encoded: int | None) -> tuple:
        """Key identifying a uniform step: input shape, dtype, device and the streaming
        parameters that alter the captured computation."""
        streaming_cfg = self.encoder.streaming_cfg
        return (
            tuple(signal.shape),
            str(signal.dtype),
            signal.device.index,
            bool(keep_all_outputs),
            -1 if drop_extra_pre_encoded is None else int(drop_extra_pre_encoded),
            tuple(self.encoder.att_context_size),
            streaming_cfg.last_channel_cache_size,
            streaming_cfg.valid_out_len,
        )

    def _can_use_graphs(self, signal, cache_last_channel, keep_all_outputs, bypass_pre_encode) -> bool:
        """Static-shape, inference-only, cached streaming steps qualify for capture/replay.

        CUDA autocast runs eager. A captured graph fixes the dtype/kernel/workspace choices at
        capture time, so replaying a non-autocast graph under autocast (or the reverse) can
        silently return wrong-precision outputs. Cache-aware streaming models run in float32 anyway.
        """
        return (
            self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH
            and not self.encoder.training
            and signal.is_cuda
            and cache_last_channel is not None
            and not keep_all_outputs
            and not bypass_pre_encode
            and not _cuda_autocast_enabled()
            and not torch.cuda.is_current_stream_capturing()
        )

    def stream_step(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor | None = None,
        cache_last_channel: torch.Tensor | None = None,
        cache_last_time: torch.Tensor | None = None,
        cache_last_channel_len: torch.Tensor | None = None,
        keep_all_outputs: bool = True,
        drop_extra_pre_encoded: int | None = None,
        bypass_pre_encode: bool = False,
    ) -> tuple[Any, ...]:
        """Drop-in replacement for ``StreamingEncoder.cache_aware_stream_step``."""
        eager_kwargs = dict(
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            bypass_pre_encode=bypass_pre_encode,
        )
        if processed_signal_length is None or not self._can_use_graphs(
            processed_signal, cache_last_channel, keep_all_outputs, bypass_pre_encode
        ):
            return self.encoder._cache_aware_stream_step_impl(processed_signal, **eager_kwargs)

        key = self._make_key(processed_signal, keep_all_outputs, drop_extra_pre_encoded)
        captured = self._graphs.get(key)
        if captured is not None:
            return self._replay(
                captured,
                processed_signal,
                processed_signal_length,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            )

        self._key_counts[key] = self._key_counts.get(key, 0) + 1
        if self._key_counts[key] > self.warmup_steps and len(self._graphs) < self.max_graphs:
            try:
                captured = self._capture(key, eager_kwargs, processed_signal)
            except Exception as e:  # noqa: BLE001 - any capture failure must not break inference
                if not self.cuda_graphs_allow_fallback:
                    raise RuntimeError(
                        "CUDA graph capture of the streaming encoder step failed and the mode is forced"
                    ) from e
                logging.warning(
                    f"CUDA graph capture of the streaming encoder step failed, falling back to eager "
                    f"execution. Streaming will be slower. Reason: {e}"
                )
                self.cuda_graphs_mode = self.CudaGraphsMode.NO_GRAPHS
                self.reset_cuda_graphs_state()
                return self.encoder._cache_aware_stream_step_impl(processed_signal, **eager_kwargs)
            return self._replay(
                captured,
                processed_signal,
                processed_signal_length,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            )

        return self.encoder._cache_aware_stream_step_impl(processed_signal, **eager_kwargs)

    # --------------------------------------------------------------------------- capture / replay
    def _capture(self, key: tuple, eager_kwargs: dict, signal: torch.Tensor) -> _CapturedStep:
        """Capture one steady-state step into a CUDA graph keyed by ``key``.

        Outputs are copied (inside the capture) into stable buffers allocated from the regular
        caching allocator, so downstream consumers never hold references into the graph's
        private memory pool. This is required for correct interop with the decoder's own CUDA
        graphs and removes the need to clone outputs on every replay.
        """
        device = signal.device
        with torch.inference_mode():
            static_inputs = {
                "signal": signal.clone(),
                "length": eager_kwargs["processed_signal_length"].clone(),
                "cache_last_channel": eager_kwargs["cache_last_channel"].clone(),
                "cache_last_time": eager_kwargs["cache_last_time"].clone(),
                "cache_last_channel_len": eager_kwargs["cache_last_channel_len"].clone(),
            }

            def run_step():
                return self.encoder._cache_aware_stream_step_impl(
                    static_inputs["signal"],
                    processed_signal_length=static_inputs["length"],
                    cache_last_channel=static_inputs["cache_last_channel"],
                    cache_last_time=static_inputs["cache_last_time"],
                    cache_last_channel_len=static_inputs["cache_last_channel_len"],
                    keep_all_outputs=eager_kwargs["keep_all_outputs"],
                    drop_extra_pre_encoded=eager_kwargs["drop_extra_pre_encoded"],
                    bypass_pre_encode=False,
                )

            # dry run: learns output shapes/dtypes and warms up kernel selection for this key
            dry_run_outputs = run_step()
            stable_outputs = tuple(torch.empty_like(t) for t in dry_run_outputs)
            del dry_run_outputs

            # quiesce all streams: capturing while other work is in flight is unsafe
            torch.cuda.synchronize(device)
            stream_for_graph = torch.cuda.Stream(device)
            stream_for_graph.wait_stream(torch.cuda.default_stream(device))
            graph = torch.cuda.CUDAGraph()
            with (
                torch.cuda.stream(stream_for_graph),
                torch.cuda.graph(graph, stream=stream_for_graph, capture_error_mode="thread_local"),
            ):
                outputs = run_step()
                for stable, out in zip(stable_outputs, outputs):
                    stable.copy_(out)  # recorded in the graph: every replay refreshes stable buffers

        captured = _CapturedStep(graph, device, static_inputs, stable_outputs)
        self._graphs[key] = captured
        logging.info(
            f"Captured CUDA graph for streaming encoder step: signal={tuple(signal.shape)}, "
            f"att_context_size={tuple(self.encoder.att_context_size)} "
            f"({len(self._graphs)}/{self.max_graphs} graphs)"
        )
        return captured

    def _replay(
        self,
        captured: _CapturedStep,
        signal: torch.Tensor,
        length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Copy inputs into the static buffers and replay the captured step."""
        # inference_mode: the static buffers are inference tensors (created during capture);
        # in-place updates to them are only allowed from inside inference mode
        with torch.inference_mode():
            static = captured.static_inputs
            static["signal"].copy_(signal)
            static["length"].copy_(length)
            static["cache_last_channel"].copy_(cache_last_channel)
            static["cache_last_time"].copy_(cache_last_time)
            static["cache_last_channel_len"].copy_(cache_last_channel_len)
            captured.graph.replay()
            return captured.get_stable_outputs_copy()
