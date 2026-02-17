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

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.numba.spec_augment import spec_aug_numba
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__


def get_cfg(seed=0, dtype='float32', **kwargs):
    # fmt: off
    default = dict(b=2, f=80, t=20, device='cuda',
                  freq_masks=2, time_masks=2, freq_width=27, time_width=0.05, mask_value=0.0,
                  seed=seed, dtype=dtype)
    default.update(**kwargs)
    cfg = OmegaConf.create(default)
    # fmt: on
    return cfg


# fmt: off
def prepare_data(b, f, t, device='cuda', freq_masks=0, time_masks=0, freq_width=10, time_width=0.1,
                 seed=0, dtype='float32',
                 **kwargs):
    torch.manual_seed(seed)

    if dtype == 'float16':
        dtype = torch.float16
    else:
        dtype = torch.float

    x = torch.randn([b, f, t], dtype=dtype, device=device)
    x_len = torch.randint(t, size=[b], device=x.device)

    sh = x.shape
    bs = sh[0]

    if isinstance(time_width, int):
        adaptive_temporal_width = False
    else:
        if time_width > 1.0 or time_width < 0.0:
            raise ValueError('If `time_width` is a float value, must be in range [0, 1]')

        adaptive_temporal_width = True

    orginal_time_width = time_width

    # Construct the freq and time masks as well as start positions
    if freq_masks > 0:
        freq_starts = torch.randint(0, sh[1] - freq_width + 1, size=[bs, freq_masks], device=x.device)
        freq_lengths = torch.randint(0, freq_width + 1, size=[bs, freq_masks], device=x.device)
    else:
        freq_starts = torch.zeros([bs, 1], dtype=torch.int64, device=x.device)
        freq_lengths = torch.zeros([bs, 1], dtype=torch.int64, device=x.device)

    if time_masks > 0:
        if adaptive_temporal_width:
            time_width = (x_len * orginal_time_width).int().clamp(min=1)
        else:
            time_width = (
                    torch.tensor(orginal_time_width, dtype=torch.int32, device=x.device)
                    .unsqueeze(0)
                    .repeat(sh[0])
                )

        time_starts = []
        time_lengths = []
        for idx in range(sh[0]):
            time_starts.append(
                torch.randint(
                    0, max(1, x_len[idx] - time_width[idx]), size=[1, time_masks], device=x.device
                )
            )
            time_lengths.append(
                torch.randint(0, time_width[idx] + 1, size=[1, time_masks], device=x.device)
            )

        time_starts = torch.cat(time_lengths, 0)
        time_lengths = torch.cat(time_lengths, 0)
    else:
        time_starts = torch.zeros([bs, 1], dtype=torch.int64, device=x.device)
        time_lengths = torch.zeros([bs, 1], dtype=torch.int64, device=x.device)

    output = dict(
        x=x,
        x_len=x_len,
        freq_starts=freq_starts,
        freq_lengths=freq_lengths,
        time_starts=time_starts,
        time_lengths=time_lengths,
        sh=sh,
    )
    return output
# fmt: on


def launch_kernel(data, cfg):
    # Launch CUDA kernel
    # fmt: off
    data['x'] = spec_aug_numba.launch_spec_augment_kernel(
        x=data['x'], x_len=data['x_len'],
        freq_starts=data['freq_starts'], freq_lengths=data['freq_lengths'],
        time_starts=data['time_starts'], time_lengths=data['time_lengths'],
        freq_masks=cfg.freq_masks, time_masks=cfg.time_masks, mask_value=cfg.mask_value
    )
    # fmt: on


def freq_mask_check(x, x_len, f_start, f_len, mask_value, bidx):
    check_result = True
    for fidx in range(f_start, f_start + f_len):
        if not (x[bidx, fidx, :] == mask_value).all():
            check_result = False
            break
    assert check_result


def time_mask_check(x, x_len, t_start, t_len, mask_value, bidx):
    check_result = True
    for tidx in range(t_start, t_start + t_len):
        if tidx >= x_len[bidx]:
            # this sample has smaller length than the time index of mask, ignore
            continue

        if not (x[bidx, :, tidx] == mask_value).all():
            check_result = False
            break
    assert check_result


class TestSpecAugmentNumba:
    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.parametrize('dtype', ['float16', 'float32'])
    def test_spec_aug_kernel(self, dtype):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        cfg = get_cfg(seed=0, dtype=dtype)
        cfg.freq_masks = 2
        cfg.time_masks = 10

        data = prepare_data(**cfg)

        launch_kernel(data, cfg)
        x, x_len, sh = data['x'], data['x_len'], data['sh']

        # Assert freq masks are correct
        for bidx in range(sh[0]):
            for f_start, f_len in zip(data['freq_starts'][bidx], data['freq_lengths'][bidx]):
                freq_mask_check(x, x_len, f_start, f_len, mask_value=cfg.mask_value, bidx=bidx)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.parametrize('dtype', ['float16', 'float32'])
    def test_spec_aug_kernel_large_batch(self, dtype):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        # Change max threads per block temporarily
        original_buffer = spec_aug_numba.MAX_THREAD_BUFFER
        spec_aug_numba.MAX_THREAD_BUFFER = 4

        cfg = get_cfg(seed=0, dtype=dtype)
        cfg.freq_masks = 2
        cfg.time_masks = 10
        cfg.b = spec_aug_numba.MAX_THREAD_BUFFER + 1

        data = prepare_data(**cfg)

        launch_kernel(data, cfg)
        x, x_len, sh = data['x'], data['x_len'], data['sh']

        # Assert freq masks are correct
        for bidx in range(sh[0]):
            for f_start, f_len in zip(data['freq_starts'][bidx], data['freq_lengths'][bidx]):
                freq_mask_check(x, x_len, f_start, f_len, mask_value=cfg.mask_value, bidx=bidx)

        # Assert time masks are correct
        for bidx in range(sh[0]):
            for t_start, t_len in zip(data['time_starts'][bidx], data['time_lengths'][bidx]):
                time_mask_check(x, x_len, t_start, t_len, mask_value=cfg.mask_value, bidx=bidx)

        spec_aug_numba.MAX_THREAD_BUFFER = original_buffer

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_spec_aug_kernel_mask_value(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        cfg = get_cfg(seed=0)
        cfg.freq_masks = 2
        cfg.time_masks = 10
        cfg.mask_value = -1.0

        data = prepare_data(**cfg)

        launch_kernel(data, cfg)
        x, x_len, sh = data['x'], data['x_len'], data['sh']

        # Assert freq masks are correct
        for bidx in range(sh[0]):
            for f_start, f_len in zip(data['freq_starts'][bidx], data['freq_lengths'][bidx]):
                freq_mask_check(x, x_len, f_start, f_len, mask_value=cfg.mask_value, bidx=bidx)

        # Assert time masks are correct
        for bidx in range(sh[0]):
            for t_start, t_len in zip(data['time_starts'][bidx], data['time_lengths'][bidx]):
                time_mask_check(x, x_len, t_start, t_len, mask_value=cfg.mask_value, bidx=bidx)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_spec_aug_kernel_grad(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        cfg = get_cfg(seed=0)
        cfg.freq_masks = 2
        cfg.time_masks = 10

        data = prepare_data(**cfg)

        launch_kernel(data, cfg)

        result = data['x']  # inplace modification via kernel
        y = torch.ones_like(result, requires_grad=True)
        z = y + result
        z.mean().backward()

        assert y.grad is not None

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_spec_aug_kernel_no_freq_mask(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        cfg = get_cfg(seed=0)
        cfg.freq_masks = 0
        cfg.time_masks = 10

        data = prepare_data(**cfg)

        launch_kernel(data, cfg)
        x, x_len, sh = data['x'], data['x_len'], data['sh']

        # Assert time masks are correct
        for bidx in range(sh[0]):
            for t_start, t_len in zip(data['time_starts'][bidx], data['time_lengths'][bidx]):
                time_mask_check(x, x_len, t_start, t_len, mask_value=cfg.mask_value, bidx=bidx)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_spec_aug_kernel_no_time_mask(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        cfg = get_cfg(seed=0)
        cfg.freq_masks = 2
        cfg.time_masks = 0

        data = prepare_data(**cfg)

        launch_kernel(data, cfg)
        x, x_len, sh = data['x'], data['x_len'], data['sh']

        # Assert freq masks are correct
        for bidx in range(sh[0]):
            for f_start, f_len in zip(data['freq_starts'][bidx], data['freq_lengths'][bidx]):
                freq_mask_check(x, x_len, f_start, f_len, mask_value=cfg.mask_value, bidx=bidx)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_spec_aug_kernel_no_freq_time_mask(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        cfg = get_cfg(seed=0)
        cfg.freq_masks = 0
        cfg.time_masks = 0

        data = prepare_data(**cfg)

        x, x_len, sh = data['x'], data['x_len'], data['sh']
        x_copy = x.clone()
        launch_kernel(data, cfg)

        # Assert no data edits occurred
        assert (data['x'] - x_copy).abs().mean() <= 1e-9
