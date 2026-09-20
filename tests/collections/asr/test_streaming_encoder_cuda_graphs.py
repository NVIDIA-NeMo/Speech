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

import pytest
import torch

from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.streaming_encoder_cuda_graphs import (
    CudaGraphsStreamingEncoderStep,
    _CapturedStep,
)
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs


def make_encoder(device: str) -> ConformerEncoder:
    """A small cache-aware streaming FastConformer-style encoder for fast tests."""
    torch.manual_seed(42)
    encoder = ConformerEncoder(
        feat_in=80,
        n_layers=2,
        d_model=64,
        n_heads=4,
        ff_expansion_factor=2,
        subsampling='dw_striding',
        subsampling_factor=8,
        subsampling_conv_channels=32,
        causal_downsampling=True,
        self_attention_model='rel_pos',
        att_context_size=[[8, 1], [8, 0]],
        att_context_style='chunked_limited',
        conv_kernel_size=5,
        conv_context_size='causal',
    )
    encoder = encoder.to(device).eval()
    encoder.setup_streaming_params()
    return encoder


def steady_chunks(encoder: ConformerEncoder, batch_size: int, num_steps: int, device: str):
    """Fixed-shape steady-state mel chunks (as produced by the streaming audio buffer, which
    prepends `pre_encode_cache_size` mel frames of context to every steady-state chunk)."""
    torch.manual_seed(123)
    chunk_len = encoder.streaming_cfg.chunk_size[1] + encoder.streaming_cfg.pre_encode_cache_size[1]
    return [torch.randn(batch_size, encoder._feat_in, chunk_len, device=device) for _ in range(num_steps)]


def run_stream(encoder: ConformerEncoder, chunks, batch_size: int, device: str, keep_all_last: bool = True):
    """Run the cache-aware streaming loop over the chunks; returns per-step output tuples."""
    cache_lc, cache_lt, cache_len = encoder.get_initial_cache_state(batch_size=batch_size)
    drop = encoder.streaming_cfg.drop_extra_pre_encoded
    chunk_len = torch.full((batch_size,), chunks[0].size(-1), dtype=torch.int64, device=device)
    outputs = []
    with torch.inference_mode():
        for step, chunk in enumerate(chunks):
            is_last = step == len(chunks) - 1
            out = encoder.cache_aware_stream_step(
                processed_signal=chunk,
                processed_signal_length=chunk_len,
                cache_last_channel=cache_lc,
                cache_last_time=cache_lt,
                cache_last_channel_len=cache_len,
                keep_all_outputs=keep_all_last and is_last,
                drop_extra_pre_encoded=drop,
            )
            outputs.append(tuple(t.clone() for t in out))
            _, _, cache_lc, cache_lt, cache_len = out
    return outputs


def assert_stream_outputs_equal(reference, candidate):
    assert len(reference) == len(candidate)
    for step, (ref, got) in enumerate(zip(reference, candidate)):
        for i, (r, g) in enumerate(zip(ref, got)):
            assert torch.equal(r, g), f"step {step}, output {i}: graphed != eager"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require a CUDA device")
class TestStreamingEncoderCudaGraphsGPU:
    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_graphed_stream_step_matches_eager(self, batch_size: int):
        """Graphed steady-state steps must be exactly equal to eager (torch.equal), including the
        cache-fill phase and the final keep_all_outputs=True step (eager fallback)."""
        device = "cuda"
        encoder = make_encoder(device)
        chunks = steady_chunks(encoder, batch_size, num_steps=8, device=device)

        reference = run_stream(encoder, chunks, batch_size, device)

        helper = encoder.set_streaming_cuda_graphs(enabled=True, warmup_steps=2)
        try:
            graphed = run_stream(encoder, chunks, batch_size, device)
            num_graphs = len(helper._graphs)
        finally:
            encoder.set_streaming_cuda_graphs(enabled=False)

        assert num_graphs >= 1, "no CUDA graph was captured"
        assert_stream_outputs_equal(reference, graphed)

    @pytest.mark.unit
    def test_input_and_output_cache_buffers_are_distinct(self):
        """The captured step must read caches from the static input buffers and write the next
        caches into separate stable output buffers. If they aliased, a replay would overwrite its
        own inputs. Also checks a multi-step replay keeps the cache flow consistent (no crash)."""
        device = "cuda"
        encoder = make_encoder(device)
        helper = encoder.set_streaming_cuda_graphs(enabled=True, warmup_steps=2)
        try:
            chunks = steady_chunks(encoder, 2, num_steps=8, device=device)
            run_stream(encoder, chunks, 2, device, keep_all_last=False)
            assert len(helper._graphs) >= 1
            captured = next(iter(helper._graphs.values()))
            in_clc = captured.static_inputs["cache_last_channel"].data_ptr()
            in_clt = captured.static_inputs["cache_last_time"].data_ptr()
            # output tuple: (encoded, encoded_len, cache_last_channel_next, cache_last_time_next, len)
            out_clc = captured.stable_outputs[2].data_ptr()
            out_clt = captured.stable_outputs[3].data_ptr()
            assert in_clc != out_clc, "input/output cache_last_channel buffers alias"
            assert in_clt != out_clt, "input/output cache_last_time buffers alias"
        finally:
            encoder.set_streaming_cuda_graphs(enabled=False)

    @pytest.mark.unit
    def test_replayed_outputs_are_not_overwritten(self):
        """Each replay must return outputs with storage independent from the graph buffers."""
        device = "cuda"
        encoder = make_encoder(device)
        helper = encoder.set_streaming_cuda_graphs(enabled=True, warmup_steps=1)
        try:
            chunks = steady_chunks(encoder, 2, num_steps=3, device=device)
            cache_lc, cache_lt, cache_len = encoder.get_initial_cache_state(batch_size=2)
            chunk_len = torch.full((2,), chunks[0].size(-1), dtype=torch.int64, device=device)
            drop = encoder.streaming_cfg.drop_extra_pre_encoded

            with torch.inference_mode():
                eager = encoder.cache_aware_stream_step(
                    chunks[0], chunk_len, cache_lc, cache_lt, cache_len, False, drop
                )
                first_replay = encoder.cache_aware_stream_step(
                    chunks[1], chunk_len, eager[2], eager[3], eager[4], False, drop
                )
                first_replay_snapshot = tuple(t.clone() for t in first_replay)
                second_replay = encoder.cache_aware_stream_step(
                    chunks[2], chunk_len, first_replay[2], first_replay[3], first_replay[4], False, drop
                )

            assert len(helper._graphs) == 1
            for output, snapshot, next_output in zip(first_replay, first_replay_snapshot, second_replay):
                assert torch.equal(output, snapshot)
                assert output.data_ptr() != next_output.data_ptr()
        finally:
            encoder.set_streaming_cuda_graphs(enabled=False)

    @pytest.mark.unit
    def test_att_context_switch_uses_new_graph(self):
        """Changing the att context must not reuse a stale graph (key includes the context)."""
        device = "cuda"
        batch_size = 2
        encoder = make_encoder(device)
        helper = encoder.set_streaming_cuda_graphs(enabled=True, warmup_steps=1)
        try:
            chunks = steady_chunks(encoder, batch_size, num_steps=5, device=device)
            run_stream(encoder, chunks, batch_size, device)
            graphs_first_mode = len(helper._graphs)

            encoder.set_default_att_context_size([8, 0])
            encoder.setup_streaming_params()
            chunks = steady_chunks(encoder, batch_size, num_steps=5, device=device)

            encoder_graphless = make_encoder(device)
            encoder_graphless.set_default_att_context_size([8, 0])
            encoder_graphless.setup_streaming_params()
            reference = run_stream(encoder_graphless, chunks, batch_size, device)

            graphed = run_stream(encoder, chunks, batch_size, device)
            assert len(helper._graphs) > graphs_first_mode, "no new graph captured for the new context"
            assert_stream_outputs_equal(reference, graphed)
        finally:
            encoder.set_streaming_cuda_graphs(enabled=False)

    @pytest.mark.unit
    def test_training_mode_runs_eager(self):
        """No capture and no replay may happen while the encoder is in training mode."""
        device = "cuda"
        encoder = make_encoder(device)
        helper = encoder.set_streaming_cuda_graphs(enabled=True, warmup_steps=1)
        try:
            encoder.train()
            chunks = steady_chunks(encoder, 2, num_steps=4, device=device)
            with torch.no_grad():
                run_stream(encoder, chunks, 2, device, keep_all_last=False)
            assert len(helper._graphs) == 0, "graph captured in training mode"
        finally:
            encoder.eval()
            encoder.set_streaming_cuda_graphs(enabled=False)

    @pytest.mark.unit
    def test_autocast_does_not_replay_non_autocast_graph(self):
        """Under an active autocast context the graph path must be skipped: a graph captured
        under one autocast state would replay it regardless of the caller's state, which could
        silently return outputs for the wrong precision (fp32 where the caller expects bf16)."""
        device = "cuda"
        encoder = make_encoder(device)
        helper = encoder.set_streaming_cuda_graphs(enabled=True, warmup_steps=1)
        try:
            chunks = steady_chunks(encoder, 2, num_steps=4, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                run_stream(encoder, chunks, 2, device, keep_all_last=False)
            assert len(helper._graphs) == 0, "graph captured/used under autocast"
        finally:
            encoder.set_streaming_cuda_graphs(enabled=False)

    @pytest.mark.unit
    def test_max_graphs_limit(self):
        """Once max_graphs distinct keys are captured, further keys stay eager (no unbounded
        capture)."""
        device = "cuda"
        batch_size = 2
        encoder = make_encoder(device)
        helper = encoder.set_streaming_cuda_graphs(enabled=True, warmup_steps=1, max_graphs=1)
        try:
            run_stream(encoder, steady_chunks(encoder, batch_size, 4, device), batch_size, device)
            assert len(helper._graphs) == 1

            encoder.set_default_att_context_size([8, 0])
            encoder.setup_streaming_params()
            run_stream(encoder, steady_chunks(encoder, batch_size, 4, device), batch_size, device)
            assert len(helper._graphs) == 1, "captured more graphs than max_graphs"
        finally:
            encoder.set_streaming_cuda_graphs(enabled=False)

    @pytest.mark.unit
    def test_forced_no_graphs_mode(self):
        """Forced no_graphs mode must never capture, and results must match eager."""
        device = "cuda"
        batch_size = 2
        encoder = make_encoder(device)
        chunks = steady_chunks(encoder, batch_size, num_steps=5, device=device)
        reference = run_stream(encoder, chunks, batch_size, device)

        helper = encoder.set_streaming_cuda_graphs(enabled=True, warmup_steps=1)
        try:
            helper.force_cuda_graphs_mode("no_graphs")
            outputs = run_stream(encoder, chunks, batch_size, device)
            assert len(helper._graphs) == 0
            assert_stream_outputs_equal(reference, outputs)
        finally:
            encoder.set_streaming_cuda_graphs(enabled=False)

    @pytest.mark.unit
    def test_disable_enable_cuda_graphs(self):
        """WithOptionalCudaGraphs API: disable drops graphs, enable allows re-capture."""
        device = "cuda"
        encoder = make_encoder(device)
        helper = encoder.set_streaming_cuda_graphs(enabled=True, warmup_steps=1)
        try:
            chunks = steady_chunks(encoder, 2, num_steps=4, device=device)
            run_stream(encoder, chunks, 2, device)
            assert len(helper._graphs) >= 1

            assert helper.disable_cuda_graphs() is True
            assert helper.cuda_graphs_mode is None
            assert len(helper._graphs) == 0
            assert helper.disable_cuda_graphs() is False  # already disabled

            run_stream(encoder, chunks, 2, device)  # eager path, must not capture
            assert len(helper._graphs) == 0

            assert helper.maybe_enable_cuda_graphs() is True
            run_stream(encoder, chunks, 2, device)
            assert len(helper._graphs) >= 1
        finally:
            encoder.set_streaming_cuda_graphs(enabled=False)

    @pytest.mark.unit
    def test_recursive_toggle_via_mixin(self):
        """The model-level hooks use WithOptionalCudaGraphs recursion over submodules."""
        device = "cuda"
        encoder = make_encoder(device)
        helper = encoder.set_streaming_cuda_graphs(enabled=True)
        try:
            WithOptionalCudaGraphs.disable_cuda_graphs_recursive(encoder, attribute_path="_stream_step_cuda_graphs")
            assert helper.cuda_graphs_mode is None
            WithOptionalCudaGraphs.enable_cuda_graphs_recursive(encoder, attribute_path="_stream_step_cuda_graphs")
            assert helper.cuda_graphs_mode is helper.CudaGraphsMode.FULL_GRAPH
        finally:
            encoder.set_streaming_cuda_graphs(enabled=False)


class TestStreamingEncoderCudaGraphsCPU:
    @pytest.mark.unit
    def test_maybe_enable_without_cuda(self, monkeypatch):
        """Without CUDA the helper stays disabled and dispatch falls back to eager."""
        encoder = make_encoder("cpu")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        helper = encoder.set_streaming_cuda_graphs(enabled=True)
        try:
            assert helper.cuda_graphs_mode is None
            assert helper.maybe_enable_cuda_graphs() is False

            chunks = steady_chunks(encoder, 2, num_steps=3, device="cpu")
            outputs = run_stream(encoder, chunks, 2, "cpu")
            assert len(helper._graphs) == 0
            assert len(outputs) == 3
        finally:
            encoder.set_streaming_cuda_graphs(enabled=False)

    @pytest.mark.unit
    def test_set_streaming_cuda_graphs_detach(self):
        """Disabling removes the helper and restores plain eager dispatch."""
        encoder = make_encoder("cpu")
        encoder.set_streaming_cuda_graphs(enabled=True)
        assert encoder._stream_step_cuda_graphs is not None
        encoder.set_streaming_cuda_graphs(enabled=False)
        assert encoder._stream_step_cuda_graphs is None

    @pytest.mark.unit
    def test_asr_model_hooks_toggle_encoder_graphs(self, monkeypatch):
        """`ASRModel.disable_cuda_graphs()` / `maybe_enable_cuda_graphs()` must also toggle the
        encoder streaming graphs, so the train/val Lightning hooks manage them like the decoder
        graphs. Uses the real ASRModel methods on a minimal container (no model download)."""

        class _Container:
            maybe_enable_cuda_graphs = ASRModel.maybe_enable_cuda_graphs
            disable_cuda_graphs = ASRModel.disable_cuda_graphs

            def __init__(self, encoder):
                self.encoder = encoder
                self.decoding = None  # no decoder in this minimal container

        # pretend CUDA is available so the helper can enter FULL_GRAPH mode on CPU
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        encoder = make_encoder("cpu")
        helper = encoder.set_streaming_cuda_graphs(enabled=True)
        container = _Container(encoder)
        try:
            assert helper.cuda_graphs_mode is helper.CudaGraphsMode.FULL_GRAPH
            assert container.disable_cuda_graphs() is True  # on_train_epoch_start path
            assert helper.cuda_graphs_mode is None
            assert container.maybe_enable_cuda_graphs() is True  # on_train_epoch_end path
            assert helper.cuda_graphs_mode is helper.CudaGraphsMode.FULL_GRAPH
        finally:
            encoder.set_streaming_cuda_graphs(enabled=False)

    @pytest.mark.unit
    def test_reset_synchronizes_every_captured_graph_device(self, monkeypatch):
        """Teardown must synchronize each device that owns a captured graph pool."""
        encoder = make_encoder("cpu")
        helper = CudaGraphsStreamingEncoderStep(encoder)
        device_zero = torch.device("cuda:0")
        device_one = torch.device("cuda:1")
        helper._graphs = {
            (0,): _CapturedStep(None, device_zero, {}, ()),
            (1,): _CapturedStep(None, device_one, {}, ()),
            (2,): _CapturedStep(None, device_zero, {}, ()),
        }
        synchronized_devices = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "synchronize", synchronized_devices.append)

        helper.reset_cuda_graphs_state()

        assert set(synchronized_devices) == {device_zero, device_one}
        assert helper._graphs == {}

    @pytest.mark.unit
    def test_warmup_steps_validation(self):
        encoder = make_encoder("cpu")
        with pytest.raises(ValueError):
            CudaGraphsStreamingEncoderStep(encoder, warmup_steps=0)
