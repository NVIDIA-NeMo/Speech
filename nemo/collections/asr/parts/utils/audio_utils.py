# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import math
from typing import Iterable, Optional, Union

import librosa
import numpy as np
import numpy.typing as npt
import scipy
import soundfile as sf
import torch
from scipy.spatial.distance import pdist, squareform

from nemo.utils import logging

SOUND_VELOCITY = 343.0  # m/s
ChannelSelectorType = Union[int, Iterable[int], str]


def get_samples(audio_file: str, target_sr: int = 16000, dtype: str = 'float32'):
    """
    Read the samples from the given audio_file path. If not specified, the input audio file is automatically
    resampled to 16kHz.

    Args:
        audio_file (str):
            Path to the input audio file
        target_sr (int):
            Targeted sampling rate
    Returns:
        samples (numpy.ndarray):
            Time-series sample data from the given audio file
    """
    with sf.SoundFile(audio_file, 'r') as f:
        samples = f.read(dtype=dtype)
        if f.samplerate != target_sr:
            samples = librosa.core.resample(samples, orig_sr=f.samplerate, target_sr=target_sr)
        samples = samples.transpose()
    return samples


def select_channels(signal: npt.NDArray, channel_selector: Optional[ChannelSelectorType] = None) -> npt.NDArray:
    """
    Convert a multi-channel signal to a single-channel signal by averaging over channels or selecting a single channel,
    or pass-through multi-channel signal when channel_selector is `None`.
    
    Args:
        signal: numpy array with shape (..., num_channels)
        channel selector: string denoting the downmix mode, an integer denoting the channel to be selected, or an iterable
                          of integers denoting a subset of channels. Channel selector is using zero-based indexing.
                          If set to `None`, the original signal will be returned. Uses zero-based indexing.

    Returns:
        numpy array
    """
    if signal.ndim == 1:
        # For one-dimensional input, return the input signal.
        if channel_selector not in [None, 0, 'average']:
            raise ValueError(
                'Input signal is one-dimensional, channel selector (%s) cannot not be used.', str(channel_selector)
            )
        return signal

    num_channels = signal.shape[-1]
    num_samples = signal.size // num_channels  # handle multi-dimensional signals

    if num_channels >= num_samples:
        logging.warning(
            'Number of channels (%d) is greater or equal than number of samples (%d). Check for possible transposition.',
            num_channels,
            num_samples,
        )

    # Samples are arranged as (num_channels, ...)
    if channel_selector is None:
        # keep the original multi-channel signal
        pass
    elif channel_selector == 'average':
        # default behavior: downmix by averaging across channels
        signal = np.mean(signal, axis=-1)
    elif isinstance(channel_selector, int):
        # select a single channel
        if channel_selector >= num_channels:
            raise ValueError(f'Cannot select channel {channel_selector} from a signal with {num_channels} channels.')
        signal = signal[..., channel_selector]
    elif isinstance(channel_selector, Iterable):
        # select multiple channels
        if max(channel_selector) >= num_channels:
            raise ValueError(
                f'Cannot select channel subset {channel_selector} from a signal with {num_channels} channels.'
            )
        signal = signal[..., channel_selector]
        # squeeze the channel dimension if a single-channel is selected
        # this is done to have the same shape as when using integer indexing
        if len(channel_selector) == 1:
            signal = np.squeeze(signal, axis=-1)
    else:
        raise ValueError(f'Unexpected value for channel_selector ({channel_selector})')

    return signal


def sinc_unnormalized(x: float) -> float:
    """Unnormalized sinc.
    
    Args:
        x: input value
        
    Returns:
        Calculates sin(x)/x 
    """
    return np.sinc(x / np.pi)


def theoretical_coherence(
    mic_positions: npt.NDArray,
    sample_rate: float,
    field: str = 'spherical',
    fft_length: int = 512,
    sound_velocity: float = SOUND_VELOCITY,
) -> npt.NDArray:
    """Calculate a theoretical coherence matrix for given mic positions and field type.
    
    Args:
        mic_positions: 3D Cartesian coordinates of microphone positions, shape (num_mics, 3)
        field: string denoting the type of the soundfield
        sample_rate: sampling rate of the input signal in Hz
        fft_length: length of the fft in samples
        sound_velocity: speed of sound in m/s
    
    Returns:
        Calculated coherence with shape (num_subbands, num_mics, num_mics)
    """
    assert mic_positions.shape[1] == 3, "Expecting 3D microphone positions"
    num_mics = mic_positions.shape[0]

    if num_mics < 2:
        raise ValueError(f'Expecting at least 2 microphones, received {num_mics}')

    num_subbands = fft_length // 2 + 1
    angular_freq = 2 * np.pi * sample_rate * np.arange(0, num_subbands) / fft_length
    desired_coherence = np.zeros((num_subbands, num_mics, num_mics))

    mic_distance = squareform(pdist(mic_positions))

    for p in range(num_mics):
        desired_coherence[:, p, p] = 1.0
        for q in range(p + 1, num_mics):
            dist_pq = mic_distance[p, q]
            if field == 'spherical':
                desired_coherence[:, p, q] = sinc_unnormalized(angular_freq * dist_pq / sound_velocity)
            else:
                raise ValueError(f'Unknown noise field {field}.')
            # symmetry
            desired_coherence[:, q, p] = desired_coherence[:, p, q]

    return desired_coherence


def estimated_coherence(S: npt.NDArray, eps: float = 1e-16) -> npt.NDArray:
    """Estimate complex-valued coherence for the input STFT-domain signal.
    
    Args:
        S: STFT of the signal with shape (num_subbands, num_frames, num_channels)
        eps: small regularization constant
        
    Returns:
        Estimated coherence with shape (num_subbands, num_channels, num_channels)
    """
    if S.ndim != 3:
        raise RuntimeError('Expecting the input STFT to be a 3D array')

    num_subbands, num_frames, num_channels = S.shape

    if num_channels < 2:
        raise ValueError('Expecting at least 2 microphones')

    psd = np.mean(np.abs(S) ** 2, axis=1)
    estimated_coherence = np.zeros((num_subbands, num_channels, num_channels), dtype=complex)

    for p in range(num_channels):
        estimated_coherence[:, p, p] = 1.0
        for q in range(p + 1, num_channels):
            cross_psd = np.mean(S[:, :, p] * np.conjugate(S[:, :, q]), axis=1)
            estimated_coherence[:, p, q] = cross_psd / np.sqrt(psd[:, p] * psd[:, q] + eps)
            # symmetry
            estimated_coherence[:, q, p] = np.conjugate(estimated_coherence[:, p, q])

    return estimated_coherence


def generate_approximate_noise_field(
    mic_positions: npt.NDArray,
    noise_signal: npt.NDArray,
    sample_rate: float,
    field: str = 'spherical',
    fft_length: int = 512,
    method: str = 'cholesky',
    sound_velocity: float = SOUND_VELOCITY,
):
    """
    Args:
        mic_positions: 3D microphone positions, shape (num_mics, 3)
        noise_signal: signal used to generate the approximate noise field, shape (num_samples, num_mics).
                      Different channels need to be independent.
        sample_rate: sampling rate of the input signal
        field: string denoting the type of the soundfield
        fft_length: length of the fft in samples
        method: coherence decomposition method
        sound_velocity: speed of sound in m/s
        
    Returns:
        Signal with coherence approximately matching the desired coherence, shape (num_samples, num_channels)
        
    References:
        E.A.P. Habets, I. Cohen and S. Gannot, 'Generating nonstationary multisensor
        signals under a spatial coherence constraint', Journal of the Acoustical Society
        of America, Vol. 124, Issue 5, pp. 2911-2917, Nov. 2008.
    """
    assert fft_length % 2 == 0
    num_mics = mic_positions.shape[0]

    if num_mics < 2:
        raise ValueError('Expecting at least 2 microphones')

    desired_coherence = theoretical_coherence(
        mic_positions=mic_positions,
        field=field,
        sample_rate=sample_rate,
        fft_length=fft_length,
        sound_velocity=sound_velocity,
    )

    return transform_to_match_coherence(signal=noise_signal, desired_coherence=desired_coherence, method=method)


def transform_to_match_coherence(
    signal: npt.NDArray,
    desired_coherence: npt.NDArray,
    method: str = 'cholesky',
    ref_channel: int = 0,
    corrcoef_threshold: float = 0.2,
) -> npt.NDArray:
    """Transform the input multichannel signal to match the desired coherence.
    
    Note: It's assumed that channels are independent.
    
    Args:
        signal: independent noise signals with shape (num_samples, num_channels)
        desired_coherence: desired coherence with shape (num_subbands, num_channels, num_channels)
        method: decomposition method used to construct the transformation matrix
        ref_channel: reference channel for power normalization of the input signal
        corrcoef_threshold: used to detect input signals with high correlation between channels
        
    Returns:
        Signal with coherence approximately matching the desired coherence, shape (num_samples, num_channels)

    References:
        E.A.P. Habets, I. Cohen and S. Gannot, 'Generating nonstationary multisensor
        signals under a spatial coherence constraint', Journal of the Acoustical Society
        of America, Vol. 124, Issue 5, pp. 2911-2917, Nov. 2008.
    """
    num_channels = signal.shape[1]
    num_subbands = desired_coherence.shape[0]
    assert desired_coherence.shape[1] == num_channels
    assert desired_coherence.shape[2] == num_channels

    fft_length = 2 * (num_subbands - 1)

    # remove DC component
    signal = signal - np.mean(signal, axis=0)

    # channels needs to have equal power, so normalize with the ref mic
    signal_power = np.mean(np.abs(signal) ** 2, axis=0)
    signal = signal * np.sqrt(signal_power[ref_channel]) / np.sqrt(signal_power)

    # input channels should be uncorrelated
    # here, we just check for high correlation coefficients between channels to detect ill-constructed inputs
    corrcoef_matrix = np.corrcoef(signal.transpose())
    # mask the diagonal elements
    np.fill_diagonal(corrcoef_matrix, 0.0)
    if np.any(np.abs(corrcoef_matrix) > corrcoef_threshold):
        raise RuntimeError(
            f'Input channels are correlated above the threshold {corrcoef_threshold}. Max abs off-diagonal element of the coefficient matrix: {np.abs(corrcoef_matrix).max()}.'
        )

    # analysis transform
    S = librosa.stft(signal.transpose(), n_fft=fft_length)
    # (channel, subband, frame) -> (subband, frame, channel)
    S = S.transpose(1, 2, 0)

    # generate output signal for each subband
    X = np.zeros_like(S)

    # factorize the desired coherence (skip the DC component)
    if method == 'cholesky':
        L = np.linalg.cholesky(desired_coherence[1:])
        A = L.swapaxes(1, 2)
    elif method == 'evd':
        w, V = np.linalg.eig(desired_coherence[1:])
        # scale eigenvectors
        A = np.sqrt(w)[:, None, :] * V
        # prepare transform matrix
        A = A.swapaxes(1, 2)
    else:
        raise ValueError(f'Unknown method {method}')

    # transform vectors at each time step:
    #   x_t = A^T * s_t
    # or in matrix notation: X = S * A
    X[1:, ...] = np.matmul(S[1:, ...], A)

    # synthesis transform
    # transpose X from (subband, frame, channel) to (channel, subband, frame)
    x = librosa.istft(X.transpose(2, 0, 1), length=len(signal))
    # (channel, sample) -> (sample, channel)
    x = x.transpose()

    return x


def rms(x: np.ndarray) -> float:
    """Calculate RMS value for the input signal.

    Args:
        x: input signal

    Returns:
        RMS of the input signal.
    """
    return np.sqrt(np.mean(np.abs(x) ** 2))


def mag2db(mag: float, eps: Optional[float] = 1e-16) -> float:
    """Convert magnitude ratio from linear scale to dB.

    Args:
        mag: linear magnitude value
        eps: small regularization constant

    Returns:
        Value in dB.
    """
    return 20 * np.log10(mag + eps)


def db2mag(db: float) -> float:
    """Convert value in dB to linear magnitude ratio.
    
    Args:
        db: magnitude ratio in dB

    Returns:
        Magnitude ratio in linear scale.
    """
    return 10 ** (db / 20)


def pow2db(power: float, eps: Optional[float] = 1e-16) -> float:
    """Convert power ratio from linear scale to dB.

    Args:
        power: power ratio in linear scale
        eps: small regularization constant
    
    Returns:
        Power in dB.
    """
    return 10 * np.log10(power + eps)


def get_segment_start(signal: np.ndarray, segment: np.ndarray) -> int:
    """Get starting point of `segment` in `signal`.
    We assume that `segment` is a sub-segment of `signal`.
    For example, `signal` may be a 10 second audio signal,
    and `segment` could be the signal between 2 seconds and
    5 seconds. This function will then return the index of
    the sample where `segment` starts (at 2 seconds).

    Args:
        signal: numpy array with shape (num_samples,)
        segment: numpy array with shape (num_samples,)

    Returns:
        Index of the start of `segment` in `signal`.
    """
    if len(signal) <= len(segment):
        raise ValueError(
            f'segment must be shorter than signal: len(segment) = {len(segment)}, len(signal) = {len(signal)}'
        )
    cc = scipy.signal.correlate(signal, segment, mode='valid')
    return np.argmax(cc)


def calculate_sdr_numpy(
    estimate: np.ndarray,
    target: np.ndarray,
    scale_invariant: bool = False,
    convolution_invariant: bool = False,
    convolution_filter_length: Optional[int] = None,
    remove_mean: bool = True,
    sdr_max: Optional[float] = None,
    eps: float = 1e-10,
) -> float:
    """Calculate signal-to-distortion ratio.

        SDR = 10 * log10( ||t||_2^2 / (||e-t||_2^2 + alpha * ||t||^2)

    where
        alpha = 10^(-sdr_max/10)

    Optionally, apply scale-invariant scaling to target signal.

    Args:
        estimate: estimated signal
        target: target signal

    Returns:
        SDR in dB.
    """
    if scale_invariant and convolution_invariant:
        raise ValueError('Arguments scale_invariant and convolution_invariant cannot be used simultaneously.')

    if remove_mean:
        estimate = estimate - np.mean(estimate)
        target = target - np.mean(target)

    if scale_invariant or (convolution_invariant and convolution_filter_length == 1):
        target = scale_invariant_target_numpy(estimate=estimate, target=target, eps=eps)
    elif convolution_invariant:
        target = convolution_invariant_target_numpy(
            estimate=estimate, target=target, filter_length=convolution_filter_length, eps=eps
        )

    target_pow = np.mean(np.abs(target) ** 2)
    distortion_pow = np.mean(np.abs(estimate - target) ** 2)

    if sdr_max is not None:
        distortion_pow = distortion_pow + 10 ** (-sdr_max / 10) * target_pow

    sdr = 10 * np.log10(target_pow / (distortion_pow + eps) + eps)
    return sdr


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angle in radians to [-pi, pi]

    Args:
        x: angle in radians

    Returns:
        Angle in radians wrapped to [-pi, pi]
    """
    pi = torch.tensor(math.pi, device=x.device)
    return torch.remainder(x + pi, 2 * pi) - pi


def convmtx_numpy(x: np.ndarray, filter_length: int, delay: int = 0, n_steps: Optional[int] = None) -> np.ndarray:
    """Construct a causal convolutional matrix from x delayed by `delay` samples.

    Args:
        x: input signal, shape (N,)
        filter_length: length of the filter in samples
        delay: delay the signal by a number of samples
        n_steps: total number of time steps (rows) for the output matrix

    Returns:
        Convolutional matrix, shape (n_steps, filter_length)
    """
    if x.ndim != 1:
        raise ValueError(f'Expecting one-dimensional signal. Received signal with shape {x.shape}')

    if n_steps is None:
        # Keep the same length as the input signal
        n_steps = len(x)

    # pad as necessary
    x_pad = np.hstack([np.zeros(delay), x])
    if (pad_len := n_steps - len(x_pad)) > 0:
        x_pad = np.hstack([x_pad, np.zeros(pad_len)])
    else:
        x_pad = x_pad[:n_steps]

    return scipy.linalg.toeplitz(x_pad, np.hstack([x_pad[0], np.zeros(filter_length - 1)]))


def convmtx_mc_numpy(x: np.ndarray, filter_length: int, delay: int = 0, n_steps: Optional[int] = None) -> np.ndarray:
    """Construct a causal multi-channel convolutional matrix from `x` delayed by `delay` samples.

    Args:
        x: input signal, shape (N, M)
        filter_length: length of the filter in samples
        delay: delay the signal by a number of samples
        n_steps: total number of time steps (rows) for the output matrix

    Returns:
        Multi-channel convolutional matrix, shape (n_steps, M * filter_length)
    """
    if x.ndim != 2:
        raise ValueError(f'Expecting two-dimensional signal. Received signal with shape {x.shape}')

    mc_mtx = []

    for m in range(x.shape[1]):
        mc_mtx.append(convmtx_numpy(x[:, m], filter_length=filter_length, delay=delay, n_steps=n_steps))

    return np.hstack(mc_mtx)


def scale_invariant_target_numpy(estimate: np.ndarray, target: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Calculate convolution-invariant target for a given estimated signal.
    
    Calculate scaled target obtained by solving

        min_scale || scale * target - estimate ||^2

    Args:
        estimate: one-dimensional estimated signal, shape (T,)
        target: one-dimensional target signal, shape (T,)
        eps: regularization constans

    Returns:
        Scaled target signal, shape (T,)
    """
    assert target.ndim == estimate.ndim == 1, f'Only one-dimensional inputs supported'

    estimate_dot_target = np.mean(estimate * target)
    target_pow = np.mean(np.abs(target) ** 2)
    scale = estimate_dot_target / (target_pow + eps)
    return scale * target


def convolution_invariant_target_numpy(
    estimate: np.ndarray, target: np.ndarray, filter_length, diag_reg: float = 1e-8, eps: float = 1e-10
) -> np.ndarray:
    """Calculate convolution-invariant target for a given estimated signal.
    
    Calculate target filtered with a linear f obtained by solving

        min_filter || conv(filter, target) - estimate ||^2

    Args:
        estimate: one-dimensional estimated signal
        target: one-dimensional target signal
        filter_length: length of the (convolutive) filter
        diag_reg: multiplicative factor for relative diagonal loading
        eps: absolute diagonal loading
    """
    assert target.ndim == estimate.ndim == 1, f'Only one-dimensional inputs supported'

    n_fft = 2 ** math.ceil(math.log2(len(target) + len(estimate) - 1))

    T = np.fft.rfft(target, n=n_fft)
    E = np.fft.rfft(estimate, n=n_fft)

    # target autocorrelation
    tt_corr = np.fft.irfft(np.abs(T) ** 2, n=n_fft)
    # target-estimate crosscorrelation
    te_corr = np.fft.irfft(T.conj() * E, n=n_fft)

    # Use only filter_length
    tt_corr = tt_corr[:filter_length]
    te_corr = te_corr[:filter_length]

    if diag_reg is not None:
        tt_corr[0] += diag_reg * tt_corr[0] + eps

    # Construct the Toeplitz system matrix
    TT = scipy.linalg.toeplitz(tt_corr)

    # Solve the linear system for the optimal filter
    filt = np.linalg.solve(TT, te_corr)

    # Calculate filtered target
    T_filt = T * np.fft.rfft(filt, n=n_fft)
    target_filt = np.fft.irfft(T_filt, n=n_fft)

    return target_filt[: len(target)]


def toeplitz(x: torch.Tensor) -> torch.Tensor:
    """Create Toeplitz matrix for one-dimensional signals along the last dimension.

    Args:
        x: tensor with shape (..., T)

    Returns:
        Tensor with shape (..., T, T)
    """
    length = x.size(-1)
    x = torch.cat([x[..., 1:].flip(dims=(-1,)), x], dim=-1)
    return x.unfold(-1, length, 1).flip(dims=(-1,))
