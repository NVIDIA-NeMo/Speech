# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from numba import jit


def phase_vocoder(D: np.ndarray, rate: float, phi_advance: np.ndarray, scale_buffer: np.ndarray):
    """
    Optimized implementation of phase vocoder from Librosa.
    Reference implementation:
        - https://librosa.github.io/librosa/generated/librosa.core.phase_vocoder.html
    Args:
        D: Complex spectograms of shape [d, t, complex=2].
        rate: Speed rate, must be float greater than 0.
        phi_advance: Precomputed phase advance buffer array of length [n_fft + 1]
        scale_buffer: Precomputed numpy buffer array of length [n_fft + 1]
    Returns:
        Complex64 ndarray of shape [d, t / rate, complex=2]
    """
    time_steps = np.arange(0, D.shape[1], rate, dtype=np.float64)

    # Create an empty output array
    d_stretch = np.zeros((D.shape[0], len(time_steps)), D.dtype, order='F')

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[:, 0])

    # Pad 0 columns to simplify boundary logic
    D = np.pad(D, [(0, 0), (0, 2)], mode='constant')

    d_stretch = _phase_vocoder_kernel(D, time_steps, phi_advance, d_stretch, phase_acc, scale_buffer)

    return d_stretch


@jit(nopython=True, nogil=True)
def _phase_vocoder_kernel(D, time_steps, phi_advance, d_stretch, phase_acc, scale_buffer):
    """
    Numba optimized kernel to compute the phase vocoder step.
    Args:
        D: Complex spectograms of shape [d, t, complex=2].
        rate: Speed rate, must be float greater than 0.
        time_steps: Numpy ndarray of linearly spaced time steps, shape = [t]
        phi_advance: Precomputed phase advance buffer array of length [n_fft + 1]
        d_stretch: Output complex matrix of shape [d, t / rate, complex=2]
        phase_acc: Phase accumulator initialized to first sample of shape [d, complex=2]
        scale_buffer: Precomputed numpy buffer array of length [n_fft + 1]
    Returns:
        Complex64 ndarray of shape [d, t / rate, complex=2]
    """
    two_pi = 2.0 * np.pi

    for (t, step) in enumerate(time_steps):
        columns = D[:, int(step) : int(step + 2)]
        columns_0 = columns[:, 0]
        columns_1 = columns[:, 1]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)
        mag = (1.0 - alpha) * np.abs(columns_0) + alpha * np.abs(columns_1)

        # Store to output array
        d_stretch[:, t] = mag * np.exp(1.0j * phase_acc)

        # Compute phase advance
        dphase = np.angle(columns_1) - np.angle(columns_0) - phi_advance

        # Wrap to -pi:pi range
        scale = dphase / two_pi
        np.round(scale, 0, scale_buffer)

        dphase = dphase - two_pi * scale_buffer

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch
