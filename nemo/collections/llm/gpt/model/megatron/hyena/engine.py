# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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
import torch.nn.functional as F
from einops import rearrange


def adjust_filter_shape_for_broadcast(u, h):
    """
    Adjust filter shape for broadcasting compatibility with input tensor.
    """
    h = h.squeeze()  # Standardize to [D, L] from [1, D, L] and [D, 1, L]

    # Case: u: [B, D, L], k_f: [D, L]
    if len(u.shape) > len(h.shape):
        h = h.unsqueeze(0)

    # Case: u: [B, D1, D2, L], k_f: [B, D, L]
    if len(u.shape) > 3:
        h = h.unsqueeze(1)
    return h


def fftconv_func(*, u, k, D):
    """
    Compute fast Fourier transform convolution with bias addition.

    This function performs convolution using FFT for efficient computation of long sequences.
    The convolution is computed in the frequency domain and then transformed back to the time domain.
    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    k_f = adjust_filter_shape_for_broadcast(u, k_f)
    k = k.squeeze()

    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    return y + u * D.unsqueeze(-1)


def parallel_fir(
    *,
    u,  # B L D
    weight,
    bias,
    L,
    gated_bias,
    fir_length,
    compute_state,
):
    """Compute parallel finite impulse response filtering with optional state computation."""
    L = u.shape[1]
    u = rearrange(u, "b l d -> b d l")

    if fir_length >= 128:
        with torch.autocast("cuda"):
            z = fftconv_func(
                u=u.to(torch.float32),
                k=weight[:, :, :L].to(torch.float32),
                D=bias,
            ).to(dtype=u.dtype)
    else:
        z = F.conv1d(
            u.to(torch.float32),
            weight.to(torch.float32),
            bias=None,
            stride=1,
            padding=fir_length - 1,
            groups=u.shape[1],  # always set to D, regardless of filter grouping
        )[..., :L]

        z = z.to(u.dtype)

        if bias is not None:
            if gated_bias:
                z = z + bias[None, :, None] * u
            else:
                z = z + bias[None, :, None]

    fir_state = None
    if compute_state:
        fir_state = u[..., -fir_length + 1 :]
    return z, fir_state


def parallel_iir(*, z_pre, h, D, L, poles, t, hidden_size, compute_state):
    """Compute the output state of the short convolutional filter."""
    fft_size = 2 * L
    x1, x2, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)

    x1v = x1 * v

    H = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
    X_s = torch.fft.fft(x1v.to(dtype=torch.float32), n=fft_size)
    X = X_s[..., : H.shape[-1]]
    if len(z_pre.shape) > 3:
        H = H.unsqueeze(1)
    y = torch.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]
    y = y.to(dtype=x1v.dtype)
    y = (y + x1v * D.unsqueeze(-1)) * x2

    iir_state = None
    if compute_state:
        iir_state = prefill_via_modal_fft(
            x1v=x1v,
            X_s=X_s,
            L=L,
            t=t,
            poles=poles,
        )

    return y.permute(0, 2, 1), iir_state


def step_fir(*, u, fir_state, weight, bias=None, gated_bias=False, flip_filter=False):
    """Steps forward FIR filters in the architecture.
    FIR filters generally include truncated convolutions in Hyena with an explicit or
    hybrid time-domain parametrization:
    * Short FIR filters in Hyena featurizers
    * Short and medium FIR filters in Hyena operators
    Note:
        `fir_state` contains the last FIR filter length - 1 elements of `u`: `u_(L-2), u_{L-1), ...`
        We assume dimensions of `short_filter_weight` to be `[d, 1, short_filter_len]`.
    """
    weight = weight.squeeze()

    cache_size = fir_state.shape[-1]
    filter_length = weight.shape[-1]
    if flip_filter:
        weight = weight.flip(-1)
        weight = weight[..., -cache_size - 1 :].unsqueeze(0)
    else:
        weight = weight[..., : cache_size + 1].unsqueeze(0)

    input_dtype = u.dtype
    weight = weight.to(torch.float32)
    u = u.to(torch.float32)
    fir_state = fir_state.to(torch.float32)
    bias = bias.to(torch.float32) if bias is not None else None

    h0, h = weight[..., -1], weight[..., :-1]
    y = h0 * u + torch.sum(fir_state * h, dim=-1)

    if bias is not None:
        if gated_bias:
            y = y + bias * u
        else:
            y = y + bias

    # Update the state
    if cache_size < filter_length - 1:
        fir_state = torch.cat([fir_state, u[..., None]], dim=-1)
    else:
        fir_state = torch.roll(fir_state, -1, dims=2)
        fir_state[..., -1] = u

    return y.to(input_dtype), fir_state


def step_iir(*, x2, x1, v, D, residues, poles, iir_state):
    """Steps forward IIR filters in the architecture."""
    x1v = x1 * v
    poles = torch.exp(poles)  # poles arg contains log_poles
    poles = poles[..., 0][None]  # squeeze dummy seqlen dim and add dummy batch dim
    residues = residues[None]  # add dummy batch dim
    iir_state = poles * iir_state + x1v[..., None]

    res_state = torch.sum(residues * iir_state, dim=-1)
    y = x2 * (res_state + D * x1v)
    return y, iir_state


def prefill_via_modal_fft(*, x1v, L, poles, t, X_s):
    """
    Compute the IIR state via a single FFT
    """
    # When the model has a long convolution derived from a recurrence in modal form and prefill_style is "fft",
    # we split the filter into poles and residues and reuse FFT computation on the input.
    bs = x1v.shape[0]
    fft_size = 2 * L
    state_s = (poles.to(torch.float32) * t).exp()
    state_S = torch.fft.fft(state_s, n=fft_size).repeat(bs, 1, 1, 1)  # B, D, state_dim, 2 * L
    state = torch.fft.ifft(X_s[..., None, :] * state_S, n=fft_size)
    # Do not try to fix `UserWarning: Casting complex values to real discards
    # the imaginary part` by inserting state.real conversion anywhere before
    # float32 conversion. It will increase memory usage. Instead, let fp32
    # conversion efficiently drop the complex part for us.
    return state[..., L - 1].to(dtype=torch.float32)
