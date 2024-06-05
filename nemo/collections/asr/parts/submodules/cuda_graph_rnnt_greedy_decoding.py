# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import numpy as np
import torch

try:
    from cuda import cudart

    HAVE_CUDA_PYTHON = True
except ImportError:
    HAVE_CUDA_PYTHON = False
from typing import List, Optional

from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.core.utils.cuda_python_utils import (
    check_cuda_python_cuda_graphs_conditional_nodes_supported,
    cu_call,
    run_nvrtc,
    with_conditional_node,
)

_CUDA_PROGRAM_NAME = b"while_loop_conditional.cu"


def create_outer_for_loop_kernel():
    """
    Creates a kernel that evaluates whether or not to enter the for loop body.
    Effectively substitutes for `for time_idx in range(trip_count)`
    such that that for loop can run on a GPU.
    """
    kernel_string = r"""\
    typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

    extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

    extern "C" __global__
    void for_loop_conditional(cudaGraphConditionalHandle handle, const long *time_idx, const long *trip_count)
    {
     cudaGraphSetConditional(handle, *time_idx < *trip_count);
    }
    """
    return run_nvrtc(kernel_string, b"for_loop_conditional", _CUDA_PROGRAM_NAME)


def create_inner_while_loop_kernel():
    """
    Evaluates whether or not to keep evaluating the inner while loop body.
    Continue until all elements of the batch output blank or the while loop
    has run max_symbols times.
    """
    kernel_string = r"""\
    typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

    extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

    extern "C" __global__
    void while_loop_conditional(cudaGraphConditionalHandle handle, const bool *not_blank, const long *symbols_added, const long *max_symbols)
    {
     cudaGraphSetConditional(handle, *not_blank && *symbols_added < *max_symbols);
    }
    """
    return run_nvrtc(kernel_string, b"while_loop_conditional", _CUDA_PROGRAM_NAME)


class RNNTGreedyDecodeCudaGraph:
    def __init__(self, max_symbols: int, caller):
        if HAVE_CUDA_PYTHON:
            check_cuda_python_cuda_graphs_conditional_nodes_supported()
        else:
            raise ValueError("Cannot instantiate RNNTGreedyDecodeCudaGraph without `pip install cuda-python`")

        assert max_symbols is not None

        self.max_symbols = max_symbols

        # These are cuda torch.Tensors which will be lazily allocated the first time _reinitialize() is called.
        # We don't do it here because we don't know which cuda device we are using yet.
        self.symbols_added_t = None
        self.max_symbols_t = None
        self.not_all_blank_t = None
        self.time_idx_t = None
        self.max_out_len_t = None

        self.encoder_output = None
        self.encoder_output_length = None
        self.f = None
        # We also lazily initialize a variable holding the current device
        self.device = None

        # Reasonable default maximum time. 375 frames * (80ms / frame) = 30 seconds
        # 80ms is the frame size of recent fastconformer models
        # This does not affect correctness.
        self.max_time = 375
        self.batch_size = 0

        self.scores_cpu = None
        self.labels_cpu = None
        self.graph = None

        self.first_call = True

        self.caller = caller

    def _reinitialize(self, max_time, batch_size, encoder_output, encoder_output_length):
        if self.first_call:
            # We need to call the original _greedy_decode_blank_as_pad
            # implementation at least once beforehand in order to make
            # sure that pytorch is "initialized". Pytorch may be
            # uninitialized if this code runs before any other pytorch
            # operation in this process. Pytorch often lazily
            # initializes things like a cudnnHandle_t via
            # cudnnCreate(), which can involve synchronizing with the
            # host. Such actions are not stream capturable to a graph.
            with torch.cuda.stream(torch.cuda.Stream(self.device)):
                self.caller._greedy_decode_blank_as_pad_loop_frames(
                    encoder_output, encoder_output_length, encoder_output.device
                )

            self.device = encoder_output.device

            self.symbols_added_t = torch.tensor(0, dtype=torch.int64, device=encoder_output.device)
            self.max_symbols_t = torch.tensor(self.max_symbols, dtype=torch.int64, device=encoder_output.device)
            self.not_all_blank_t = torch.tensor(True, dtype=torch.bool, device=encoder_output.device)

            self.time_idx_t = torch.tensor(0, dtype=torch.int64, device=encoder_output.device)
            self.max_out_len_t = torch.tensor(0, dtype=torch.int64, device=encoder_output.device)

            self.first_call = False

        self.max_time = max(self.max_time, max_time)
        self.batch_size = max(self.batch_size, batch_size)

        self.encoder_output = torch.zeros(
            (self.batch_size, self.max_time, encoder_output.shape[-1]),
            dtype=encoder_output.dtype,
            device=encoder_output.device,
        )
        self.encoder_output_length = torch.zeros(
            (self.batch_size,), dtype=encoder_output_length.dtype, device=encoder_output_length.device
        )

        self.zero_t = torch.tensor(0.0, dtype=encoder_output.dtype, device=encoder_output.device)
        self.blank_index_t = torch.tensor(self.caller._blank_index, dtype=torch.long, device=encoder_output.device)

        self.scores_cpu = torch.zeros(
            (self.batch_size, self.max_time, self.max_symbols),
            dtype=encoder_output.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.labels_cpu = torch.zeros(
            (self.batch_size, self.max_time, self.max_symbols), dtype=torch.int64, device="cpu", pin_memory=True
        )

        self.graph = None

        self.graph = torch.cuda.CUDAGraph()

        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.device)
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(self.graph, stream=stream_for_graph, capture_error_mode="thread_local"),
        ):
            # This is failing...
            self.f = torch.zeros(
                (self.batch_size, 1, self.encoder_output.shape[-1]),
                dtype=encoder_output.dtype,
                device=encoder_output.device,
            )
            hidden = self.caller.decoder.initialize_state(self.f)
            self.last_label = torch.full(
                [self.batch_size], fill_value=self.caller._SOS, dtype=torch.long, device=encoder_output.device
            )
            self.blank_mask = torch.full(
                [self.batch_size], fill_value=0, dtype=torch.bool, device=encoder_output.device
            )
            self.seq_idx_t = torch.zeros([1], dtype=torch.int64, device=encoder_output.device)

            self.scores = torch.zeros(
                (self.max_time * self.max_symbols, self.batch_size),
                dtype=encoder_output.dtype,
                device=encoder_output.device,
            )
            self.labels = torch.full(
                (self.max_time * self.max_symbols, self.batch_size),
                fill_value=self.caller._blank_index,
                dtype=torch.int64,
                device=encoder_output.device,
            )
            # Get max sequence length
            self.max_out_len_t = self.encoder_output_length.max()

            capture_status, _, graph, _, _ = cu_call(
                cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=self.device).cuda_stream)
            )
            assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

            (for_loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            for_loop_kernel = create_outer_for_loop_kernel()
            time_idx_ptr = np.array([self.time_idx_t.data_ptr()], dtype=np.uint64)
            max_out_len_ptr = np.array([self.max_out_len_t.data_ptr()], dtype=np.uint64)
            for_loop_args = np.array(
                [for_loop_conditional_handle.getPtr(), time_idx_ptr.ctypes.data, max_out_len_ptr.ctypes.data],
                dtype=np.uint64,
            )

            with with_conditional_node(for_loop_kernel, for_loop_args, for_loop_conditional_handle, self.device):
                torch.index_select(self.encoder_output, 1, self.time_idx_t.unsqueeze(0), out=self.f)

                self.not_all_blank_t.fill_(True)
                self.symbols_added_t.fill_(0)

                torch.ge(self.time_idx_t, self.encoder_output_length, out=self.blank_mask)

                while_loop_kernel = create_inner_while_loop_kernel()
                (while_loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
                not_blank_ptr = np.array([self.not_all_blank_t.data_ptr()], dtype=np.uint64)
                symbols_added_ptr = np.array([self.symbols_added_t.data_ptr()], dtype=np.uint64)
                max_symbols_ptr = np.array([self.max_symbols_t.data_ptr()], dtype=np.uint64)
                while_loop_args = np.array(
                    [
                        while_loop_conditional_handle.getPtr(),
                        not_blank_ptr.ctypes.data,
                        symbols_added_ptr.ctypes.data,
                        max_symbols_ptr.ctypes.data,
                    ],
                    dtype=np.uint64,
                )
                with with_conditional_node(
                    while_loop_kernel, while_loop_args, while_loop_conditional_handle, self.device
                ):
                    g, hidden_prime = self.caller._pred_step(
                        self.last_label.unsqueeze(1), hidden, batch_size=self.batch_size
                    )
                    logp = self.caller._joint_step(self.f, g, log_normalize=None)[:, 0, 0, :]

                    v, k = logp.max(1)
                    torch.where(self.blank_mask, self.zero_t, v, out=v)
                    torch.where(self.blank_mask, self.blank_index_t, k, out=k)
                    # Commented out code unnecessarily causes D2H copy, which is synchronous. See pytorch issue #105641
                    # self.scores[self.seq_idx_t, :] = v
                    # self.labels[self.seq_idx_t, :] = k
                    self.scores.index_copy_(0, self.seq_idx_t, v.unsqueeze(0))
                    self.labels.index_copy_(0, self.seq_idx_t, k.unsqueeze(0))

                    self.blank_mask.logical_or_(k == self.caller._blank_index)

                    not_blank_mask = ~self.blank_mask

                    self.caller.decoder.batch_replace_states_mask(
                        src_states=hidden_prime, dst_states=hidden, mask=not_blank_mask
                    )
                    torch.where(self.blank_mask, self.last_label, k, out=self.last_label)

                    torch.any(not_blank_mask, 0, out=self.not_all_blank_t)
                    self.symbols_added_t += 1
                    self.seq_idx_t += 1

                self.time_idx_t += 1
                self.seq_idx_t += self.max_symbols_t - self.symbols_added_t

            self.scores_cpu.copy_(
                self.scores.transpose(0, 1).contiguous().reshape((self.batch_size, self.max_time, self.max_symbols)),
                non_blocking=True,
            )
            self.labels_cpu.copy_(
                self.labels.transpose(0, 1).contiguous().reshape((self.batch_size, self.max_time, self.max_symbols)),
                non_blocking=True,
            )

            self.last_label.fill_(self.caller._SOS)
            self.time_idx_t.fill_(0)

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):
        if partial_hypotheses is not None:
            raise NotImplementedError(
                "`partial_hypotheses` support is not available "
                "with Frame-Looping algorithm with Cuda graphs (not implemented yet)"
            )

        if self.caller.preserve_alignments:
            raise NotImplementedError(
                "`preserve_alignments` support is not available"
                "with Frame-Looping algorithm with Cuda graphs (not implemented yet)"
            )

        if self.caller.preserve_frame_confidence:
            raise NotImplementedError(
                "`preserve_frame_confidence` support is not available"
                "with Frame-Looping algorithm with Cuda graphs (not implemented yet)"
            )

        batch_size = x.shape[0]
        # We could use out_len.max() here instead of x.shape[1], in
        # case for some reason the user passes in a larger buffer than
        # required, since we know that `out_len.max() <= x.shape[1]`.
        max_time = x.shape[1]

        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_gpu_dtype())

        if max_time > self.max_time or batch_size > self.batch_size or self.device != x.device:
            # In the first two cases, we need to recreate the cuda
            # graph to handle larger tensor sizes. In the third case,
            # we need to recreate the graph, as well as all tensors,
            # because the computation is now happening on a different
            # GPU. Therefore, in the third case, we unconditionally
            # set self.first_call to True to make sure that all
            # possibly blocking initializers are initialized properly
            # again on the new device.
            if self.device != x.device:
                self.first_call = True
            self._reinitialize(max_time, batch_size, x, out_len)

        self.encoder_output[: x.shape[0], : x.shape[1], ...].copy_(x)
        self.encoder_output_length[: out_len.shape[0]].copy_(out_len)
        self.graph.replay()
        torch.cuda.current_stream(device=self.device).synchronize()

        self.scores_cpu[self.labels_cpu == self.caller._blank_index] = 0.0
        total_scores = self.scores_cpu.sum(dtype=torch.float32, axis=(1, 2))

        tokens_per_timestep = (self.labels_cpu != self.caller._blank_index).sum(axis=-1)
        timesteps_packed = torch.repeat_interleave(
            torch.arange(self.max_time).repeat(self.batch_size), tokens_per_timestep.flatten()
        )
        timestep_segments = tokens_per_timestep.sum(axis=-1)

        valid_labels_mask = self.labels_cpu != self.caller._blank_index
        labels_segments = valid_labels_mask.sum(axis=(1, 2))
        labels_packed = self.labels_cpu[valid_labels_mask]

        hypotheses = [
            rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batch_size)
        ]

        timestep_start = 0
        labels_start = 0
        for i in range(batch_size):
            hypotheses[i].timestep = timesteps_packed[timestep_start : timestep_start + timestep_segments[i]].tolist()
            timestep_start += timestep_segments[i]
            hypotheses[i].score = float(total_scores[i])
            hypotheses[i].y_sequence = labels_packed[labels_start : labels_start + labels_segments[i]].tolist()
            labels_start += labels_segments[i]

        return hypotheses
