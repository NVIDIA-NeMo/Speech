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

import numpy as np
import pytest
import torch
from numba import cuda

from nemo.collections.asr.losses.rnnt_pytorch import MultiblankRNNTLossPytorch, TDTLossPytorch
from nemo.collections.asr.parts.numba.rnnt_loss import rnnt_numpy
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import certify_inputs
from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import gpu_rnnt_kernel, reduce
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__


DTYPES = [torch.float32]
if numba_utils.is_numba_cuda_fp16_supported():
    DTYPES.append(torch.float16)


def log_softmax(x, axis=-1):
    x = torch.from_numpy(x)  # zero-copy
    x = x.float()
    x = torch.log_softmax(x, dim=axis)
    x = x.numpy()
    return x


def log_softmax_grad(x, axis=-1):
    x = torch.tensor(x, requires_grad=True)  # alloc memory
    y = torch.log_softmax(x, dim=axis)
    y.sum().backward()
    return x.grad.numpy()


class TestRNNTCUDAKernels:
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Reductions can only be run when CUDA is available")
    @pytest.mark.unit
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_compute_alphas_kernel(self, dtype):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random = np.random.RandomState(0)
        original_shape = [1, 5, 11, 3]
        B, T, U, V = original_shape
        threshold = 1e-5 if dtype == torch.float32 else 3e-4

        # Numpy kernel
        x = random.randn(*original_shape)
        labels = np.array([[1, 1, 1, 2, 2, 2, 1, 2, 2, 1]])  # [1, 10]
        label_len = len(labels[0]) + 1
        blank_idx = 0

        x_np = log_softmax(x, axis=-1)
        ground_alphas, ground_log_likelihood = rnnt_numpy.forward_pass(
            x_np[0, :, :label_len, :], labels[0, : label_len - 1], blank_idx
        )

        # Pytorch kernel
        device = torch.device('cuda')
        if hasattr(cuda, 'external_stream'):
            stream = cuda.external_stream(torch.cuda.current_stream(device).cuda_stream)
        else:
            stream = cuda.default_stream()

        x_c = torch.tensor(x, device=device, dtype=dtype)
        labels_c = torch.tensor(labels, device=device, dtype=torch.int64)

        # Allocate workspace memory
        denom = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        alphas = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        llForward = torch.zeros(B, device=device, dtype=x_c.dtype)
        input_lengths = torch.tensor([T], dtype=torch.int64, device=device)
        label_lengths = torch.tensor([len(labels[0])], dtype=torch.int64, device=device)

        # certify input data
        certify_inputs(x_c, labels_c, input_lengths, label_lengths)

        # flatten activation tensor (for pointer based indexing)
        x_c = x_c.view([-1])

        # call kernel
        # log softmax reduction
        reduce.reduce_max(x_c, denom, rows=V, cols=B * T * U, minus=False, stream=stream)
        reduce.reduce_exp(x_c, denom, rows=V, cols=B * T * U, minus=True, stream=stream)

        # alpha kernel
        gpu_rnnt_kernel.compute_alphas_kernel[B, U, stream, 0](
            x_c, denom, alphas, llForward, input_lengths, label_lengths, labels_c, B, T, U, V, blank_idx,
        )

        # sync kernel
        stream.synchronize()

        # reshape alphas
        alphas = alphas.view([B, T, U])
        diff = ground_alphas - alphas[0].cpu().numpy()

        assert np.abs(diff).mean() <= threshold
        assert np.square(diff).mean() <= (threshold ** 2)

        ll_diff = ground_log_likelihood - llForward[0].cpu().numpy()

        assert np.abs(ll_diff).mean() <= threshold
        assert np.square(ll_diff).mean() <= (threshold ** 2)

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Reductions can only be run when CUDA is available")
    @pytest.mark.unit
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_compute_betas_kernel(self, dtype):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random = np.random.RandomState(0)
        original_shape = [1, 5, 11, 3]
        B, T, U, V = original_shape
        threshold = 1e-5 if dtype == torch.float32 else 3e-4

        # Numpy kernel
        x = random.randn(*original_shape)
        labels = np.array([[1, 1, 1, 2, 2, 2, 1, 2, 2, 1]])  # [1, 10]
        label_len = len(labels[0]) + 1
        blank_idx = 0

        x_np = log_softmax(x, axis=-1)
        ground_alphas, ground_log_likelihood = rnnt_numpy.backward_pass(
            x_np[0, :, :label_len, :], labels[0, : label_len - 1], blank_idx
        )

        # Pytorch kernel
        device = torch.device('cuda')
        if hasattr(cuda, 'external_stream'):
            stream = cuda.external_stream(torch.cuda.current_stream(device).cuda_stream)
        else:
            stream = cuda.default_stream()

        x_c = torch.tensor(x, device=device, dtype=dtype)
        labels_c = torch.tensor(labels, device=device, dtype=torch.int64)

        # Allocate workspace memory
        denom = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        betas = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        llBackward = torch.zeros(B, device=device, dtype=x_c.dtype)
        input_lengths = torch.tensor([T], dtype=torch.int64, device=device)
        label_lengths = torch.tensor([len(labels[0])], dtype=torch.int64, device=device)

        # certify input data
        certify_inputs(x_c, labels_c, input_lengths, label_lengths)

        # flatten activation tensor (for pointer based indexing)
        x_c = x_c.view([-1])

        # call kernel
        # log softmax reduction
        reduce.reduce_max(x_c, denom, rows=V, cols=B * T * U, minus=False, stream=stream)
        reduce.reduce_exp(x_c, denom, rows=V, cols=B * T * U, minus=True, stream=stream)

        # beta kernel
        gpu_rnnt_kernel.compute_betas_kernel[B, U, stream, 0](
            x_c, denom, betas, llBackward, input_lengths, label_lengths, labels_c, B, T, U, V, blank_idx,
        )

        # sync kernel
        stream.synchronize()

        # reshape alphas
        betas = betas.view([B, T, U])
        diff = ground_alphas - betas[0].cpu().numpy()

        assert np.abs(diff).mean() <= threshold
        assert np.square(diff).mean() <= (threshold ** 2)

        ll_diff = ground_log_likelihood - llBackward[0].cpu().numpy()

        assert np.abs(ll_diff).mean() <= threshold
        assert np.square(ll_diff).mean() <= (threshold ** 2)

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Reductions can only be run when CUDA is available")
    @pytest.mark.unit
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_compute_grads_kernel(self, dtype):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        fastemit_lambda = 0.0
        clamp = 0.0

        random = np.random.RandomState(0)
        original_shape = [1, 5, 11, 3]
        B, T, U, V = original_shape
        threshold = 1e-5 if dtype == torch.float32 else 3e-5

        # Numpy kernel
        x = random.randn(*original_shape)
        labels = torch.from_numpy(np.array([[1, 1, 1, 2, 2, 2, 1, 2, 2, 1]], dtype=np.int64))  # [1, 10]
        audio_len = torch.from_numpy(np.array([T], dtype=np.int64))
        label_len = torch.from_numpy(np.array([U - 1], dtype=np.int64))
        blank_idx = 0

        x_np = torch.from_numpy(x)
        x_np.requires_grad_(True)

        """
        Here we will directly utilize the numpy variant of the loss without explicitly calling
        the numpy functions for alpha, beta and grads. 
        
        This is because the grads returned by the rnnt_numpy.transduce_batch() are :
        d/dx (alpha + beta alignment)(log_softmax(x)). 
        But according to the chain rule, we'd still need to compute the gradient of log_softmax(x)
        and update the alignments by hand. Instead, we will rely on pytorch to compute the gradient 
        of the log_softmax(x) step and propagate it backwards. 
        """
        loss_func = rnnt_numpy.RNNTLoss(blank_idx, fastemit_lambda=fastemit_lambda, clamp=clamp)
        loss_val = loss_func(x_np, labels, audio_len, label_len)
        loss_val.sum().backward()
        true_grads = x_np.grad

        # Pytorch kernel
        device = torch.device('cuda')
        if hasattr(cuda, 'external_stream'):
            stream = cuda.external_stream(torch.cuda.current_stream(device).cuda_stream)
        else:
            stream = cuda.default_stream()

        x_c = torch.tensor(x, device=device, dtype=dtype)
        labels_c = labels.clone().to(device=device, dtype=torch.int64)

        # Allocate workspace memory
        denom = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        alphas = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        betas = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        llForward = torch.zeros(B, device=device, dtype=x_c.dtype)
        llBackward = torch.zeros(B, device=device, dtype=x_c.dtype)
        input_lengths = torch.tensor([T], dtype=torch.int64, device=device)
        label_lengths = torch.tensor([len(labels[0])], dtype=torch.int64, device=device)

        # certify input data
        certify_inputs(x_c, labels_c, input_lengths, label_lengths)

        # flatten activation tensor (for pointer based indexing)
        x_c = x_c.view([-1])
        grads = torch.zeros_like(x_c, requires_grad=False)

        # call kernel
        # log softmax reduction
        reduce.reduce_max(x_c, denom, rows=V, cols=B * T * U, minus=False, stream=stream)
        reduce.reduce_exp(x_c, denom, rows=V, cols=B * T * U, minus=True, stream=stream)

        # alpha kernel
        gpu_rnnt_kernel.compute_alphas_kernel[B, U, stream, 0](
            x_c, denom, alphas, llForward, input_lengths, label_lengths, labels_c, B, T, U, V, blank_idx,
        )

        # beta kernel
        gpu_rnnt_kernel.compute_betas_kernel[B, U, stream, 0](
            x_c, denom, betas, llBackward, input_lengths, label_lengths, labels_c, B, T, U, V, blank_idx,
        )

        # gamma kernel
        grad_blocks_per_grid = B * T * U
        grad_threads_per_block = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE
        gpu_rnnt_kernel.compute_grad_kernel[grad_blocks_per_grid, grad_threads_per_block, stream, 0](
            grads,
            x_c,
            denom,
            alphas,
            betas,
            llForward,
            input_lengths,
            label_lengths,
            labels_c,
            B,
            T,
            U,
            V,
            blank_idx,
            fastemit_lambda,
            clamp,
        )

        # sync kernel
        stream.synchronize()

        # reshape grads
        grads = grads.view([B, T, U, V])
        diff = true_grads - grads[0].cpu().numpy()

        assert np.abs(diff).mean() <= threshold
        assert np.square(diff).mean() <= (threshold ** 2) * 5.0

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Reductions can only be run when CUDA is available")
    @pytest.mark.unit
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_compute_grads_kernel_fastemit(self, dtype):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        fastemit_lambda = 0.001
        clamp = 0.0

        random = np.random.RandomState(0)
        original_shape = [1, 5, 11, 3]
        B, T, U, V = original_shape
        threshold = 1e-5 if dtype == torch.float32 else 3e-5

        # Numpy kernel
        x = random.randn(*original_shape)
        labels = torch.from_numpy(np.array([[1, 1, 1, 2, 2, 2, 1, 2, 2, 1]], dtype=np.int64))  # [1, 10]
        audio_len = torch.from_numpy(np.array([T], dtype=np.int64))
        label_len = torch.from_numpy(np.array([U - 1], dtype=np.int64))
        blank_idx = 0

        x_np = torch.from_numpy(x)
        x_np.requires_grad_(True)

        """
        Here we will directly utilize the numpy variant of the loss without explicitly calling
        the numpy functions for alpha, beta and grads. 

        This is because the grads returned by the rnnt_numpy.transduce_batch() are :
        d/dx (alpha + beta alignment)(log_softmax(x)). 
        But according to the chain rule, we'd still need to compute the gradient of log_softmax(x)
        and update the alignments by hand. Instead, we will rely on pytorch to compute the gradient 
        of the log_softmax(x) step and propagate it backwards. 
        """
        loss_func = rnnt_numpy.RNNTLoss(blank_idx, fastemit_lambda=fastemit_lambda, clamp=clamp)
        loss_val = loss_func(x_np, labels, audio_len, label_len)
        loss_val.sum().backward()
        true_grads = x_np.grad

        # Pytorch kernel
        device = torch.device('cuda')
        if hasattr(cuda, 'external_stream'):
            stream = cuda.external_stream(torch.cuda.current_stream(device).cuda_stream)
        else:
            stream = cuda.default_stream()

        x_c = torch.tensor(x, device=device, dtype=dtype)
        labels_c = labels.clone().to(device=device, dtype=torch.int64)

        # Allocate workspace memory
        denom = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        alphas = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        betas = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        llForward = torch.zeros(B, device=device, dtype=x_c.dtype)
        llBackward = torch.zeros(B, device=device, dtype=x_c.dtype)
        input_lengths = torch.tensor([T], dtype=torch.int64, device=device)
        label_lengths = torch.tensor([len(labels[0])], dtype=torch.int64, device=device)

        # certify input data
        certify_inputs(x_c, labels_c, input_lengths, label_lengths)

        # flatten activation tensor (for pointer based indexing)
        x_c = x_c.view([-1])
        grads = torch.zeros_like(x_c, requires_grad=False)

        # call kernel
        # log softmax reduction
        reduce.reduce_max(x_c, denom, rows=V, cols=B * T * U, minus=False, stream=stream)
        reduce.reduce_exp(x_c, denom, rows=V, cols=B * T * U, minus=True, stream=stream)

        # alpha kernel
        gpu_rnnt_kernel.compute_alphas_kernel[B, U, stream, 0](
            x_c, denom, alphas, llForward, input_lengths, label_lengths, labels_c, B, T, U, V, blank_idx,
        )

        # beta kernel
        gpu_rnnt_kernel.compute_betas_kernel[B, U, stream, 0](
            x_c, denom, betas, llBackward, input_lengths, label_lengths, labels_c, B, T, U, V, blank_idx,
        )

        # gamma kernel
        grad_blocks_per_grid = B * T * U
        grad_threads_per_block = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE
        gpu_rnnt_kernel.compute_grad_kernel[grad_blocks_per_grid, grad_threads_per_block, stream, 0](
            grads,
            x_c,
            denom,
            alphas,
            betas,
            llForward,
            input_lengths,
            label_lengths,
            labels_c,
            B,
            T,
            U,
            V,
            blank_idx,
            fastemit_lambda,
            clamp,
        )

        # sync kernel
        stream.synchronize()

        # reshape grads
        grads = grads.view([B, T, U, V])
        diff = true_grads - grads[0].cpu().numpy()

        assert np.abs(diff).mean() <= threshold
        assert np.square(diff).mean() <= (threshold ** 2) * 5

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Reductions can only be run when CUDA is available")
    @pytest.mark.unit
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_compute_grads_kernel_clamp(self, dtype):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        fastemit_lambda = 0.0
        clamp = 0.1

        random = np.random.RandomState(0)
        original_shape = [1, 5, 11, 3]
        B, T, U, V = original_shape
        threshold = 1e-5 if dtype == torch.float32 else 3e-5

        # Numpy kernel
        x = random.randn(*original_shape)
        labels = torch.from_numpy(np.array([[1, 1, 1, 2, 2, 2, 1, 2, 2, 1]], dtype=np.int64))  # [1, 10]
        audio_len = torch.from_numpy(np.array([T], dtype=np.int64))
        label_len = torch.from_numpy(np.array([U - 1], dtype=np.int64))
        blank_idx = 0

        x_np = torch.from_numpy(x)
        x_np.requires_grad_(True)

        """
        Here we will directly utilize the numpy variant of the loss without explicitly calling
        the numpy functions for alpha, beta and grads. 

        This is because the grads returned by the rnnt_numpy.transduce_batch() are :
        d/dx (alpha + beta alignment)(log_softmax(x)). 
        But according to the chain rule, we'd still need to compute the gradient of log_softmax(x)
        and update the alignments by hand. Instead, we will rely on pytorch to compute the gradient 
        of the log_softmax(x) step and propagate it backwards. 
        """
        loss_func = rnnt_numpy.RNNTLoss(blank_idx, fastemit_lambda=fastemit_lambda, clamp=clamp)
        loss_val = loss_func(x_np, labels, audio_len, label_len)
        loss_val.sum().backward()
        true_grads = x_np.grad

        # Pytorch kernel
        device = torch.device('cuda')
        if hasattr(cuda, 'external_stream'):
            stream = cuda.external_stream(torch.cuda.current_stream(device).cuda_stream)
        else:
            stream = cuda.default_stream()

        x_c = torch.tensor(x, device=device, dtype=dtype)
        labels_c = labels.clone().to(device=device, dtype=torch.int64)

        # Allocate workspace memory
        denom = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        alphas = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        betas = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        llForward = torch.zeros(B, device=device, dtype=x_c.dtype)
        llBackward = torch.zeros(B, device=device, dtype=x_c.dtype)
        input_lengths = torch.tensor([T], dtype=torch.int64, device=device)
        label_lengths = torch.tensor([len(labels[0])], dtype=torch.int64, device=device)

        # certify input data
        certify_inputs(x_c, labels_c, input_lengths, label_lengths)

        # flatten activation tensor (for pointer based indexing)
        x_c = x_c.view([-1])
        grads = torch.zeros_like(x_c, requires_grad=False)

        # call kernel
        # log softmax reduction
        reduce.reduce_max(x_c, denom, rows=V, cols=B * T * U, minus=False, stream=stream)
        reduce.reduce_exp(x_c, denom, rows=V, cols=B * T * U, minus=True, stream=stream)

        # alpha kernel
        gpu_rnnt_kernel.compute_alphas_kernel[B, U, stream, 0](
            x_c, denom, alphas, llForward, input_lengths, label_lengths, labels_c, B, T, U, V, blank_idx,
        )

        # beta kernel
        gpu_rnnt_kernel.compute_betas_kernel[B, U, stream, 0](
            x_c, denom, betas, llBackward, input_lengths, label_lengths, labels_c, B, T, U, V, blank_idx,
        )

        # gamma kernel
        grad_blocks_per_grid = B * T * U
        grad_threads_per_block = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE
        gpu_rnnt_kernel.compute_grad_kernel[grad_blocks_per_grid, grad_threads_per_block, stream, 0](
            grads,
            x_c,
            denom,
            alphas,
            betas,
            llForward,
            input_lengths,
            label_lengths,
            labels_c,
            B,
            T,
            U,
            V,
            blank_idx,
            fastemit_lambda,
            clamp,
        )

        # sync kernel
        stream.synchronize()

        # reshape grads
        grads = grads.view([B, T, U, V])
        diff = true_grads - grads[0].cpu().numpy()

        assert np.abs(diff).mean() <= threshold
        assert np.square(diff).mean() <= (threshold ** 2) * 5


class TestTDTCUDAKernels:
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Reductions can only be run when CUDA is available")
    @pytest.mark.unit
    def test_compute_alphas_kernel(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random = np.random.RandomState(0)
        original_shape = [1, 15, 11, 3]
        durations = [0, 1, 2]
        B, T, U, V = original_shape
        Vd = len(durations)

        duration_act_shape = [B, T, U, Vd]
        sigma = 0.05

        # for passing into the kernel function -- it expected unnormalized logits
        x = random.randn(*original_shape)
        # for passing into the pytorch function -- it expected normalized logits
        normalized_x = log_softmax(x, axis=-1) - 0.05

        xd = random.randn(*duration_act_shape)
        # duration logits are normalized before passing into the loss computation.
        xd = log_softmax(xd, axis=-1)

        labels = np.array([[1, 1, 1, 1, 0, 0, 1, 0, 0, 1]])  # [1, 10]
        blank_idx = V - 1

        pytorch_tdt_loss = TDTLossPytorch(blank_idx, durations, sigma=sigma)

        # Pytorch kernel
        device = torch.device('cuda')
        if hasattr(cuda, 'external_stream'):
            stream = cuda.external_stream(torch.cuda.current_stream(device).cuda_stream)
        else:
            stream = cuda.default_stream()

        x = torch.tensor(x, device=device, dtype=torch.float32)
        normalized_x = torch.tensor(normalized_x, device=device, dtype=torch.float32)
        xd = torch.tensor(xd, device=device, dtype=torch.float32)
        labels = torch.tensor(labels, device=device, dtype=torch.long)
        durations = torch.tensor(durations, device=device, dtype=torch.long)

        # Allocate workspace memory
        denom = torch.zeros(B * T * U, device=device, dtype=x.dtype)
        alphas = torch.zeros(B * T * U, device=device, dtype=x.dtype)
        llForward = torch.zeros(B, device=device, dtype=x.dtype)
        input_lengths = torch.tensor([T], dtype=torch.long, device=device)
        label_lengths = torch.tensor([U - 1], dtype=torch.long, device=device)

        ground_log_likelihood, ground_alphas = pytorch_tdt_loss.compute_forward_prob(
            normalized_x, xd, labels, input_lengths, label_lengths
        )

        # certify input data
        certify_inputs(x, labels, input_lengths, label_lengths)

        # flatten activation tensor (for pointer based indexing)
        x = x.view([-1])
        xd = xd.view([-1])

        # call kernel
        # log softmax reduction
        reduce.reduce_max(x, denom, rows=V, cols=B * T * U, minus=False, stream=stream)
        reduce.reduce_exp(x, denom, rows=V, cols=B * T * U, minus=True, stream=stream)

        # alpha kernel
        gpu_rnnt_kernel.compute_tdt_alphas_kernel[B, U, stream, 0](
            x,
            xd,
            denom,
            sigma,
            alphas,
            llForward,
            input_lengths,
            label_lengths,
            labels,
            B,
            T,
            U,
            V,
            blank_idx,
            durations,
            Vd,
        )

        # sync kernel
        stream.synchronize()

        # reshape alphas
        alphas = alphas.view([B, T, U])
        diff = torch.norm(ground_alphas - alphas)
        ll_diff = torch.norm(ground_log_likelihood - llForward)

        assert diff <= 1e-3
        assert ll_diff <= 1e-3


class TestMultiblankRNNTCUDAKernels:
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Reductions can only be run when CUDA is available")
    @pytest.mark.unit
    def test_compute_alphas_kernel(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random = np.random.RandomState(0)
        original_shape = [1, 15, 11, 6]
        big_blank_durations = [2, 3, 4]
        B, T, U, V = original_shape
        num_big_blanks = len(big_blank_durations)

        sigma = 0.05

        # for passing into the kernel function -- it expected unnormalized logits
        x = random.randn(*original_shape)
        # for passing into the pytorch function -- it expected normalized logits
        normalized_x = log_softmax(x, axis=-1) - sigma

        labels = np.array([[1, 1, 1, 1, 0, 0, 1, 0, 0, 1]])  # [1, 10]
        blank_idx = V - 1

        pytorch_multiblank_loss = MultiblankRNNTLossPytorch(blank_idx, big_blank_durations, sigma=sigma)

        # Pytorch kernel
        device = torch.device('cuda')
        if hasattr(cuda, 'external_stream'):
            stream = cuda.external_stream(torch.cuda.current_stream(device).cuda_stream)
        else:
            stream = cuda.default_stream()

        x = torch.tensor(x, device=device, dtype=torch.float32)
        normalized_x = torch.tensor(normalized_x, device=device, dtype=torch.float32)
        labels = torch.tensor(labels, device=device, dtype=torch.long)
        big_blank_durations = torch.tensor(big_blank_durations, device=device, dtype=torch.long)

        # Allocate workspace memory
        denom = torch.zeros(B * T * U, device=device, dtype=x.dtype)
        alphas = torch.zeros(B * T * U, device=device, dtype=x.dtype)
        llForward = torch.zeros(B, device=device, dtype=x.dtype)
        input_lengths = torch.tensor([T], dtype=torch.long, device=device)
        label_lengths = torch.tensor([U - 1], dtype=torch.long, device=device)

        ground_log_likelihood, ground_alphas = pytorch_multiblank_loss.compute_forward_prob(
            normalized_x, labels, input_lengths, label_lengths
        )

        # certify input data
        certify_inputs(x, labels, input_lengths, label_lengths)

        # flatten activation tensor (for pointer based indexing)
        x = x.view([-1])

        # call kernel
        # log softmax reduction
        reduce.reduce_max(x, denom, rows=V, cols=B * T * U, minus=False, stream=stream)
        reduce.reduce_exp(x, denom, rows=V, cols=B * T * U, minus=True, stream=stream)

        # alpha kernel
        gpu_rnnt_kernel.compute_multiblank_alphas_kernel[B, U, stream, 0](
            x,
            denom,
            sigma,
            alphas,
            llForward,
            input_lengths,
            label_lengths,
            labels,
            B,
            T,
            U,
            V,
            blank_idx,
            big_blank_durations,
            num_big_blanks,
        )

        # sync kernel
        stream.synchronize()

        # reshape alphas
        alphas = alphas.view([B, T, U])
        diff = torch.norm(ground_alphas - alphas)
        ll_diff = torch.norm(ground_log_likelihood - llForward)

        assert diff <= 1e-3
        assert ll_diff <= 1e-3
