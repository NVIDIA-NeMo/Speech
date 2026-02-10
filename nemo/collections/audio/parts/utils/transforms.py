# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
#
# NOTE: The code below originates from torchaudio repository, version 2.9.
#       It can be found under: https://github.com/pytorch/audio/tree/release/2.9
#       The modifications applied are mostly cosmetic.
#       The inclusion of this code in NeMo allows us to avoid
#       a dependency with a problematic build process.
#       This code is licensed under the BSD 2-Clause License,
#       included verbatim from the torchaudio repository below:
#
# BSD 2-Clause License
#
# Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import math
import warnings
from typing import Callable, Optional, Union

import torch
from torch import Tensor

__all__ = ["Spectrogram", "MelSpectrogram", "MFCC", "Resample"]


class Spectrogram(torch.nn.Module):
    r"""Create a spectrogram from a audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float or None, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for magnitude, 2 for power, etc.
            If None, then the complex spectrum is returned instead. (Default: ``2``)
        normalized (bool or str, optional): Whether to normalize by magnitude after stft. If input is str, choices are
            ``"window"`` and ``"frame_length"``, if specific normalization type is desirable. ``True`` maps to
            ``"window"``. (Default: ``False``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy (Default: ``True``)
        return_complex (bool, optional):
            Deprecated and not used.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = torchaudio.transforms.Spectrogram(n_fft=800)
        >>> spectrogram = transform(waveform)

    """

    __constants__ = ["n_fft", "win_length", "hop_length", "pad", "power", "normalized"]

    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: Optional[float] = 2.0,
        normalized: Union[bool, str] = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        return_complex: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequencies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        if return_complex is not None:
            warnings.warn(
                "`return_complex` argument is now deprecated and is not effective."
                "`torchaudio.transforms.Spectrogram(power=None)` always returns a tensor with "
                "complex dtype. Please remove the argument in the function call."
            )

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Dimension (..., freq, time), where freq is
            ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        return spectrogram(
            waveform,
            self.pad,
            self.window,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.power,
            self.normalized,
            self.center,
            self.pad_mode,
            self.onesided,
        )


class MelSpectrogram(torch.nn.Module):
    r"""Create MelSpectrogram for a raw audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    This is a composition of :py:func:`torchaudio.transforms.Spectrogram`
    and :py:func:`torchaudio.transforms.MelScale`.

    Sources
        * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
        * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
        * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``None``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for magnitude, 2 for power, etc. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided: Deprecated and unused.
        norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.MelSpectrogram(sample_rate)
        >>> mel_specgram = transform(waveform)  # (channel, n_mels, time)

    See also:
        :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
        generate the filter banks.
    """

    __constants__ = ["sample_rate", "n_fft", "win_length", "hop_length", "pad", "n_mels", "f_min"]

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super(MelSpectrogram, self).__init__()

        if onesided is not None:
            warnings.warn(
                "Argument 'onesided' has been deprecated and has no influence on the behavior of this module."
            )

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.spectrogram = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=self.power,
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
        )
        self.mel_scale = MelScale(
            self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1, norm, mel_scale
        )

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        specgram = self.spectrogram(waveform)
        mel_specgram = self.mel_scale(specgram)
        return mel_specgram


class MFCC(torch.nn.Module):
    r"""Create the Mel-frequency cepstrum coefficients from an audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    By default, this calculates the MFCC on the DB-scaled Mel spectrogram.
    This is not the textbook implementation, but is implemented here to
    give consistency with librosa.

    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_mfcc (int, optional): Number of mfc coefficients to retain. (Default: ``40``)
        dct_type (int, optional): type of DCT (discrete cosine transform) to use. (Default: ``2``)
        norm (str, optional): norm to use. (Default: ``"ortho"``)
        log_mels (bool, optional): whether to use log-mel spectrograms instead of db-scaled. (Default: ``False``)
        melkwargs (dict or None, optional): arguments for MelSpectrogram. (Default: ``None``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.MFCC(
        >>>     sample_rate=sample_rate,
        >>>     n_mfcc=13,
        >>>     melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        >>> )
        >>> mfcc = transform(waveform)

    See also:
        :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
        generate the filter banks.
    """

    __constants__ = ["sample_rate", "n_mfcc", "dct_type", "top_db", "log_mels"]

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        dct_type: int = 2,
        norm: str = "ortho",
        log_mels: bool = False,
        melkwargs: Optional[dict] = None,
    ) -> None:
        super(MFCC, self).__init__()
        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError("DCT type not supported: {}".format(dct_type))
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.0
        self.amplitude_to_DB = AmplitudeToDB("power", self.top_db)

        melkwargs = melkwargs or {}
        self.MelSpectrogram = MelSpectrogram(sample_rate=self.sample_rate, **melkwargs)

        if self.n_mfcc > self.MelSpectrogram.n_mels:
            raise ValueError("Cannot select more MFCC coefficients than # mel bins")
        dct_mat = create_dct(self.n_mfcc, self.MelSpectrogram.n_mels, self.norm)
        self.register_buffer("dct_mat", dct_mat)
        self.log_mels = log_mels

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: specgram_mel_db of size (..., ``n_mfcc``, time).
        """
        mel_specgram = self.MelSpectrogram(waveform)
        if self.log_mels:
            log_offset = 1e-6
            mel_specgram = torch.log(mel_specgram + log_offset)
        else:
            mel_specgram = self.amplitude_to_DB(mel_specgram)

        # (..., time, n_mels) dot (n_mels, n_mfcc) -> (..., n_nfcc, time)
        mfcc = torch.matmul(mel_specgram.transpose(-1, -2), self.dct_mat).transpose(-1, -2)
        return mfcc


class Resample(torch.nn.Module):
    r"""Resample a signal from one frequency to another. A resampling method can be given.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Note:
        If resampling on waveforms of higher precision than float32, there may be a small loss of precision
        because the kernel is cached once as float32. If high precision resampling is important for your application,
        the functional form will retain higher precision, but run slower because it does not cache the kernel.
        Alternatively, you could rewrite a transform that caches a higher precision kernel.

    Args:
        orig_freq (int, optional): The original frequency of the signal. (Default: ``16000``)
        new_freq (int, optional): The desired frequency. (Default: ``16000``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``sinc_interp_hann``, ``sinc_interp_kaiser``] (Default: ``"sinc_interp_hann"``)
        lowpass_filter_width (int, optional): Controls the sharpness of the filter, more == sharper
            but less efficient. (Default: ``6``)
        rolloff (float, optional): The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies. (Default: ``0.99``)
        beta (float or None, optional): The shape parameter used for kaiser window.
        dtype (torch.device, optional):
            Determnines the precision that resampling kernel is pre-computed and cached. If not provided,
            kernel is computed with ``torch.float64`` then cached as ``torch.float32``.
            If you need higher precision, provide ``torch.float64``, and the pre-computed kernel is computed and
            cached as ``torch.float64``. If you use resample with lower precision, then instead of providing this
            providing this argument, please use ``Resample.to(dtype)``, so that the kernel generation is still
            carried out on ``torch.float64``.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.Resample(sample_rate, sample_rate/10)
        >>> waveform = transform(waveform)
    """

    def __init__(
        self,
        orig_freq: int = 16000,
        new_freq: int = 16000,
        resampling_method: str = "sinc_interp_hann",
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        beta: Optional[float] = None,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.gcd = math.gcd(int(self.orig_freq), int(self.new_freq))
        self.resampling_method = resampling_method
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.beta = beta

        if self.orig_freq != self.new_freq:
            kernel, self.width = _get_sinc_resample_kernel(
                self.orig_freq,
                self.new_freq,
                self.gcd,
                self.lowpass_filter_width,
                self.rolloff,
                self.resampling_method,
                beta,
                dtype=dtype,
            )
            self.register_buffer("kernel", kernel)

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Output signal of dimension (..., time).
        """
        if self.orig_freq == self.new_freq:
            return waveform
        return _apply_sinc_resample_kernel(waveform, self.orig_freq, self.new_freq, self.gcd, self.kernel, self.width)


class MelScale(torch.nn.Module):
    r"""Turn a normal STFT into a mel frequency STFT with triangular filter banks.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`. (Default: ``201``)
        norm (str or None, optional): If ``"slaney"``, divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> spectrogram_transform = transforms.Spectrogram(n_fft=1024)
        >>> spectrogram = spectrogram_transform(waveform)
        >>> melscale_transform = transforms.MelScale(sample_rate=sample_rate, n_stft=1024 // 2 + 1)
        >>> melscale_spectrogram = melscale_transform(spectrogram)

    See also:
        :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
        generate the filter banks.
    """

    __constants__ = ["n_mels", "sample_rate", "f_min", "f_max"]

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.mel_scale = mel_scale

        if f_min > self.f_max:
            raise ValueError("Require f_min: {} <= f_max: {}".format(f_min, self.f_max))

        fb = melscale_fbanks(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm, self.mel_scale)
        self.register_buffer("fb", fb)

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """

        # (..., time, freq) dot (freq, n_mels) -> (..., n_mels, time)
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)

        return mel_specgram


class AmplitudeToDB(torch.nn.Module):
    r"""Turn a tensor from the power/amplitude scale to the decibel scale.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    This output depends on the maximum value in the input tensor, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        stype (str, optional): scale of input tensor (``"power"`` or ``"magnitude"``). The
            power being the elementwise square of the magnitude. (Default: ``"power"``)
        top_db (float or None, optional): minimum negative cut-off in decibels.  A reasonable
            number is 80. (Default: ``None``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.AmplitudeToDB(stype="amplitude", top_db=80)
        >>> waveform_db = transform(waveform)
    """

    __constants__ = ["multiplier", "amin", "ref_value", "db_multiplier"]

    def __init__(self, stype: str = "power", top_db: Optional[float] = None) -> None:
        super(AmplitudeToDB, self).__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError("top_db must be positive value")
        self.top_db = top_db
        self.multiplier = 10.0 if stype == "power" else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x: Tensor) -> Tensor:
        r"""Numerically stable implementation from Librosa.

        https://librosa.org/doc/latest/generated/librosa.amplitude_to_db.html

        Args:
            x (Tensor): Input tensor before being converted to decibel scale.

        Returns:
            Tensor: Output tensor in decibel scale.
        """
        return amplitude_to_DB(x, self.multiplier, self.amin, self.db_multiplier, self.top_db)


def resample(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
) -> Tensor:
    r"""Resamples the waveform at the new frequency using bandlimited interpolation. :cite:`RESAMPLE`.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Note:
        ``transforms.Resample`` precomputes and reuses the resampling kernel, so using it will result in
        more efficient computation if resampling multiple waveforms with the same resampling parameters.

    Args:
        waveform (Tensor): The input signal of dimension `(..., time)`
        orig_freq (int): The original frequency of the signal
        new_freq (int): The desired frequency
        lowpass_filter_width (int, optional): Controls the sharpness of the filter, more == sharper
            but less efficient. (Default: ``6``)
        rolloff (float, optional): The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies. (Default: ``0.99``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``"sinc_interp_hann"``, ``"sinc_interp_kaiser"``] (Default: ``"sinc_interp_hann"``)
        beta (float or None, optional): The shape parameter used for kaiser window.

    Returns:
        Tensor: The waveform at the new frequency of dimension `(..., time).`
    """

    if orig_freq <= 0.0 or new_freq <= 0.0:
        raise ValueError("Original frequency and desired frequecy should be positive")

    if orig_freq == new_freq:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))

    kernel, width = _get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
        waveform.device,
        waveform.dtype,
    )
    resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)
    return resampled


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
    device: torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
):
    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise Exception(
            "Frequencies must be of integer type to ensure quality resampling computation. "
            "To work around this, manually convert both frequencies to integer values "
            "that maintain their resampling rate ratio before passing them into the function. "
            "Example: To downsample a 44100 hz waveform by a factor of 8, use "
            "`orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`. "
            "For more information, please refer to https://github.com/pytorch/audio/issues/1487."
        )

    if resampling_method not in ["sinc_interp_hann", "sinc_interp_kaiser"]:
        raise ValueError("Invalid resampling method: {}".format(resampling_method))

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)
    # This will perform antialiasing filtering by removing the highest frequencies.
    # At first I thought I only needed this when downsampling, but when upsampling
    # you will get edge artifacts without this, as the edge is equivalent to zero padding,
    # which will add high freq artifacts.
    base_freq *= rolloff

    # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
    # using the sinc interpolation formula:
    #   x(t) = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - t))
    # We can then sample the function x(t) with a different sample rate:
    #    y[j] = x(j / new_freq)
    # or,
    #    y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))

    # We see here that y[j] is the convolution of x[i] with a specific filter, for which
    # we take an FIR approximation, stopping when we see at least `lowpass_filter_width` zeros crossing.
    # But y[j+1] is going to have a different set of weights and so on, until y[j + new_freq].
    # Indeed:
    # y[j + new_freq] = sum_i x[i] sinc(pi * orig_freq * ((i / orig_freq - (j + new_freq) / new_freq))
    #                 = sum_i x[i] sinc(pi * orig_freq * ((i - orig_freq) / orig_freq - j / new_freq))
    #                 = sum_i x[i + orig_freq] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))
    # so y[j+new_freq] uses the same filter as y[j], but on a shifted version of x by `orig_freq`.
    # This will explain the F.conv1d after, with a stride of orig_freq.
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    # If orig_freq is still big after GCD reduction, most filters will be very unbalanced, i.e.,
    # they will have a lot of almost zero values to the left or to the right...
    # There is probably a way to evaluate those filters more efficiently, but this is kept for
    # future work.
    idx_dtype = dtype if dtype is not None else torch.float64

    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq

    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)

    # we do not use built in torch windows here as we need to evaluate the window
    # at specific positions, not over a regular grid.
    if resampling_method == "sinc_interp_hann":
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        # sinc_interp_kaiser
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = torch.tensor(float(beta))
        window = torch.i0(beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)

    t *= math.pi

    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale

    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)

    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: Tensor,
    width: int,
):
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
    resampled = resampled[..., :target_length]

    # unpack batch
    resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
    return resampled


def spectrogram(
    waveform: Tensor,
    pad: int,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: Optional[float],
    normalized: Union[bool, str],
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    return_complex: Optional[bool] = None,
) -> Tensor:
    r"""Create a spectrogram or a batch of spectrograms from a raw audio signal.
    The spectrogram can be either magnitude-only or complex.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): Tensor of audio of dimension `(..., time)`
        pad (int): Two sided padding of signal
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size
        power (float or None): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for magnitude, 2 for power, etc.
            If None, then the complex spectrum is returned instead.
        normalized (bool or str): Whether to normalize by magnitude after stft. If input is str, choices are
            ``"window"`` and ``"frame_length"``, if specific normalization type is desirable. ``True`` maps to
            ``"window"``. When normalized on ``"window"``, waveform is normalized upon the window's L2 energy. If
            normalized on ``"frame_length"``, waveform is normalized by dividing by
            :math:`(\text{frame\_length})^{0.5}`.
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            Default: ``True``
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. Default: ``"reflect"``
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy. Default: ``True``
        return_complex (bool, optional):
            Deprecated and not used.

    Returns:
        Tensor: Dimension `(..., freq, time)`, freq is
        ``n_fft // 2 + 1`` and ``n_fft`` is the number of
        Fourier bins, and time is the number of window hops (n_frame).
    """
    if return_complex is not None:
        warnings.warn(
            "`return_complex` argument is now deprecated and is not effective."
            "`torchaudio.functional.spectrogram(power=None)` always returns a tensor with "
            "complex dtype. Please remove the argument in the function call."
        )

    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    frame_length_norm, window_norm = _get_spec_norms(normalized)

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=frame_length_norm,
        onesided=onesided,
        return_complex=True,
    )

    # unpack batch
    spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

    if window_norm:
        spec_f /= window.pow(2.0).sum().sqrt()
    if power is not None:
        if power == 1.0:
            return spec_f.abs()
        return spec_f.abs().pow(power)
    return spec_f


def _get_spec_norms(normalized: Union[str, bool]):
    frame_length_norm, window_norm = False, False
    if torch.jit.isinstance(normalized, str):
        if normalized not in ["frame_length", "window"]:
            raise ValueError("Invalid normalized parameter: {}".format(normalized))
        if normalized == "frame_length":
            frame_length_norm = True
        elif normalized == "window":
            window_norm = True
    elif torch.jit.isinstance(normalized, bool):
        if normalized:
            window_norm = True
    else:
        raise TypeError("Input type not supported")
    return frame_length_norm, window_norm


def amplitude_to_DB(
    x: Tensor, multiplier: float, amin: float, db_multiplier: float, top_db: Optional[float] = None
) -> Tensor:
    r"""Turn a spectrogram from the power/amplitude scale to the decibel scale.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    The output of each tensor in a batch depends on the maximum value of that tensor,
    and so may return different values for an audio clip split into snippets vs. a full clip.

    Args:

        x (Tensor): Input spectrogram(s) before being converted to decibel scale.
            The expected shapes are ``(freq, time)``, ``(channel, freq, time)`` or
            ``(..., batch, channel, freq, time)``.

            .. note::

               When ``top_db`` is specified, cut-off values are computed for each audio
               in the batch. Therefore if the input shape is 4D (or larger), different
               cut-off values are used for audio data in the batch.
               If the input shape is 2D or 3D, a single cutoff value is used.

        multiplier (float): Use 10. for power and 20. for amplitude
        amin (float): Number to clamp ``x``
        db_multiplier (float): Log10(max(reference value and amin))
        top_db (float or None, optional): Minimum negative cut-off in decibels. A reasonable number
            is 80. (Default: ``None``)

    Returns:
        Tensor: Output tensor in decibel scale
    """
    x_db = multiplier * torch.log10(torch.clamp(x, min=amin))
    x_db -= multiplier * db_multiplier

    if top_db is not None:
        # Expand batch
        shape = x_db.size()
        packed_channels = shape[-3] if x_db.dim() > 2 else 1
        x_db = x_db.reshape(-1, packed_channels, shape[-2], shape[-1])

        x_db = torch.max(x_db, (x_db.amax(dim=(-3, -2, -1)) - top_db).view(-1, 1, 1, 1))

        # Repack batch
        x_db = x_db.reshape(shape)

    return x_db


def create_dct(n_mfcc: int, n_mels: int, norm: Optional[str]) -> Tensor:
    r"""Create a DCT transformation matrix with shape (``n_mels``, ``n_mfcc``),
    normalized depending on norm.

    .. devices:: CPU

    .. properties:: TorchScript

    Args:
        n_mfcc (int): Number of mfc coefficients to retain
        n_mels (int): Number of mel filterbanks
        norm (str or None): Norm to use (either "ortho" or None)

    Returns:
        Tensor: The transformation matrix, to be right-multiplied to
        row-wise data of size (``n_mels``, ``n_mfcc``).
    """

    if norm is not None and norm != "ortho":
        raise ValueError('norm must be either "ortho" or None')

    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(float(n_mels))
    k = torch.arange(float(n_mfcc)).unsqueeze(1)
    dct = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)  # size (n_mfcc, n_mels)

    if norm is None:
        dct *= 2.0
    else:
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.t()


def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> Tensor:
    r"""Create a frequency bin conversion matrix.

    .. devices:: CPU

    .. properties:: TorchScript

    Note:
        For the sake of the numerical compatibility with librosa, not all the coefficients
        in the resulting filter bank has magnitude of 1.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/mel_fbanks.png
           :alt: Visualization of generated filter bank

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A @ melscale_fbanks(A.size(-1), ...)``.

    """

    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb


def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    r"""Convert Hz to Mels.

    Args:
        freqs (float): Frequencies in Hz
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        mels (float): Frequency in Mels
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep

    return mels


def _mel_to_hz(mels: Tensor, mel_scale: str = "htk") -> Tensor:
    """Convert mel bin numbers to frequencies.

    Args:
        mels (Tensor): Mel frequencies
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        freqs (Tensor): Mels converted in Hz
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs


def _create_triangular_filterbank(
    all_freqs: Tensor,
    f_pts: Tensor,
) -> Tensor:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb
