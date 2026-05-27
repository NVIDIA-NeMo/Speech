# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn as nn
from numba import cuda

from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging

MAX_THREAD_BUFFER = 512


@cuda.jit()
def spec_augment_kernel(
    x: torch.Tensor,
    x_len: torch.Tensor,
    freq_starts: torch.Tensor,
    freq_widths: torch.Tensor,
    time_starts: torch.Tensor,
    time_widths: torch.Tensor,
    mask_value: float,
):
    """
    Numba CUDA kernel to perform SpecAugment in-place on the GPU.
    Parallelize over freq and time axis, parallel threads over batch.
    Sequential over masks (adaptive in time).

    Args:
        x: Pytorch tensor of shape [B, F, T] with the acoustic features.
        x_len: Pytorch tensor of shape [B] with the lengths of the padded sequence.
        freq_starts: Pytorch tensor of shape [B, M_f] with the start indices of freq masks.
        freq_widths: Pytorch tensor of shape [B, M_f] with the width of freq masks.
        time_starts: Pytorch tensor of shape [B, M_t] with the start indices of time masks.
        time_widths: Pytorch tensor of shape [B, M_t] with the width of time masks.
        mask_value: Float value that will be used as mask value.
    """
    f = cuda.blockIdx.x  # indexes the Freq dim
    t = cuda.blockIdx.y  # indexes the Time dim
    tid = cuda.threadIdx.x  # index of the current mask
    threads_per_block = cuda.blockDim.x

    # Compute the number of masks over freq axis
    len_f = freq_starts.shape[1]
    # For all samples in the batch, apply the freq mask
    for bidx in range(0, x.shape[0], threads_per_block):
        # Resolve the index of the batch (case where more masks than MAX_THREAD_BUFFER)
        bm_idx = bidx + tid

        # Access mask only if valid sample id in batch
        if bm_idx < x.shape[0]:
            # For `len_f` number of freq masks that must be applied
            for fidx in range(0, len_f):
                # Access the start index and width of this freq mask
                f_start = freq_starts[bm_idx, fidx]
                f_width = freq_widths[bm_idx, fidx]

                # If block idx `f` >= start and < (start + width) of this freq mask
                if f >= f_start and f < (f_start + f_width):
                    x[bm_idx, f, t] = mask_value

    # Compute the number of masks over time axis
    len_t = time_starts.shape[1]
    # For all samples in the batch, apply the time mask
    for b_idx in range(0, x.shape[0], threads_per_block):
        # Resolve the index of the batch (case where more masks than MAX_THREAD_BUFFER)
        bm_idx = b_idx + tid

        # Access mask only if valid sample id in batch
        if bm_idx < x.shape[0]:
            # For `len_t` number of freq masks that must be applied
            for tidx in range(0, len_t):
                # Access the start index and width of this time mask
                t_start = time_starts[bm_idx, tidx]
                t_width = time_widths[bm_idx, tidx]

                # If block idx `t` >= start and < (start + width) of this time mask
                if t >= t_start and t < (t_start + t_width):
                    # Current block idx `t` < current seq length x_len[b]
                    # This ensure that we mask only upto the length of that sample
                    # Everything after that index is padded value so unnecessary to mask
                    if t < x_len[bm_idx]:
                        x[bm_idx, f, t] = mask_value


def spec_augment_launch_heuristics(x: torch.Tensor, length: torch.Tensor):
    """
    Heuristics to determins whether pytorch implementation or numba implementation is selected.
    Assumes numba cuda is supported.

    Args:
        x: Torch tensor of shape [B, F, T]
        length: Optional, Torch of tensor of shape [B] - containing lengths of the tensor.

    Returns:
        True if numba kernel should be selected, else False
    """
    if not x.is_cuda:
        return False

    if length is None:
        return False

    if x.shape[0] < 8:
        return False

    return True


def launch_spec_augment_kernel(
    x: torch.Tensor,
    x_len: torch.Tensor,
    freq_starts: torch.Tensor,
    freq_lengths: torch.Tensor,
    time_starts: torch.Tensor,
    time_lengths: torch.Tensor,
    freq_masks: int,
    time_masks: int,
    mask_value: float,
):
    """
    Helper method to launch the SpecAugment kernel

    Args:
        x: Pytorch tensor of shape [B, F, T] with the acoustic features.
        x_len: Pytorch tensor of shape [B] with the lengths of the padded sequence.
        freq_starts: Pytorch tensor of shape [B, M_f] with the start indices of freq masks.
        freq_widths: Pytorch tensor of shape [B, M_f] with the width of freq masks.
        time_starts: Pytorch tensor of shape [B, M_t] with the start indices of time masks.
        time_widths: Pytorch tensor of shape [B, M_t] with the width of time masks.
        freq_masks: Int value that determines the number of time masks.
        time_masks: Int value that determines the number of freq masks.
        mask_value: Float value that will be used as mask value.

    Returns:
        The spec augmented tensor 'x'
    """
    # Setup CUDA stream
    sh = x.shape
    stream = cuda.external_stream(torch.cuda.current_stream(x.device).cuda_stream)

    if time_masks > 0 or freq_masks > 0:
        # Parallelize over freq and time axis, parallel threads over batch
        # Sequential over masks (adaptive in time).
        blocks_per_grid = tuple([sh[1], sh[2]])
        # threads_per_block = min(MAX_THREAD_BUFFER, max(freq_masks, time_masks))
        threads_per_block = min(MAX_THREAD_BUFFER, x.shape[0])

        # Numba does not support fp16, force cast to fp32 temporarily at the expense of memory
        original_dtype = x.dtype
        cast_x = False
        if x.dtype == torch.float16:
            x = x.float()
            cast_x = True

        # Launch CUDA kernel
        spec_augment_kernel[blocks_per_grid, threads_per_block, stream, 0](
            x, x_len, freq_starts, freq_lengths, time_starts, time_lengths, mask_value
        )
        torch.cuda.synchronize()

        # Recast back to original dtype if earlier cast was performed
        if cast_x:
            x = x.to(dtype=original_dtype)

    return x


class SpecAugmentNumba(nn.Module, Typing):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).

    Utilizes a Numba CUDA kernel to perform inplace edit of the input without loops.
    Parallelize over freq and time axis, parallel threads over batch.
    Sequential over masks (adaptive in time).

    Args:
        freq_masks - how many frequency segments should be cut
        time_masks - how many time segments should be cut
        freq_width - maximum number of frequencies to be cut in one segment
        time_width - maximum number of time steps to be cut in one segment.
            Can be a positive integer or a float value in the range [0, 1].
            If positive integer value, defines maximum number of time steps
            to be cut in one segment.
            If a float value, defines maximum percentage of timesteps that
            are cut adaptively.
        rng: Ignored.
    """

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            "input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {"augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    def __init__(
        self,
        freq_masks=0,
        time_masks=0,
        freq_width=10,
        time_width=0.1,
        rng=None,
        mask_value=0.0,
    ):
        super().__init__()
        # Message to mention that numba specaugment kernel will be available
        # if input device is CUDA and lengths are provided
        logging.debug("Numba SpecAugment kernel is available")

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.time_width = time_width

        self.mask_value = mask_value

        # Unused
        self.rng = rng
        if self.rng is not None:
            logging.warning("`rng` was supplied to SpecAugmentNumba, but it is not used.")

        if isinstance(time_width, int):
            self.adaptive_temporal_width = False
        else:
            if time_width > 1.0 or time_width < 0.0:
                raise ValueError('If `time_width` is a float value, must be in range [0, 1]')

            self.adaptive_temporal_width = True

    @typecheck()
    @torch.no_grad()
    def forward(self, input_spec, length):
        sh = input_spec.shape
        bs = sh[0]

        # Construct the freq and time masks as well as start positions
        if self.freq_masks > 0:
            freq_starts = torch.randint(
                0, sh[1] - self.freq_width + 1, size=[bs, self.freq_masks], device=input_spec.device
            )
            freq_lengths = torch.randint(0, self.freq_width + 1, size=[bs, self.freq_masks], device=input_spec.device)
        else:
            freq_starts = torch.zeros([bs, 1], dtype=torch.int64, device=input_spec.device)
            freq_lengths = torch.zeros([bs, 1], dtype=torch.int64, device=input_spec.device)

        if self.time_masks > 0:
            if self.adaptive_temporal_width:
                time_width = (length * self.time_width).int().clamp(min=1)
            else:
                time_width = (
                    torch.tensor(self.time_width, dtype=torch.int32, device=input_spec.device)
                    .unsqueeze(0)
                    .repeat(sh[0])
                )

            time_starts = []
            time_lengths = []
            for idx in range(sh[0]):
                time_starts.append(
                    torch.randint(
                        0, max(1, length[idx] - time_width[idx]), size=[1, self.time_masks], device=input_spec.device
                    )
                )
                time_lengths.append(
                    torch.randint(0, time_width[idx] + 1, size=[1, self.time_masks], device=input_spec.device)
                )

            time_starts = torch.cat(time_starts, 0)
            time_lengths = torch.cat(time_lengths, 0)

        else:
            time_starts = torch.zeros([bs, 1], dtype=torch.int64, device=input_spec.device)
            time_lengths = torch.zeros([bs, 1], dtype=torch.int64, device=input_spec.device)

        x = launch_spec_augment_kernel(
            input_spec,
            length,
            freq_starts=freq_starts,
            freq_lengths=freq_lengths,
            time_starts=time_starts,
            time_lengths=time_lengths,
            freq_masks=self.freq_masks,
            time_masks=self.time_masks,
            mask_value=self.mask_value,
        )

        return x
