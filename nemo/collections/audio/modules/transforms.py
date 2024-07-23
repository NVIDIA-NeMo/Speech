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
from typing import Dict, Optional, Tuple

import torch
from einops import rearrange

from nemo.collections.asr.parts.preprocessing.features import make_seq_mask_like
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging

try:
    import torchaudio
    import torchaudio.functional
    import torchaudio.transforms

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


class AudioToSpectrogram(NeuralModule):
    """Transform a batch of input multi-channel signals into a batch of
    STFT-based spectrograms.

    Args:
        fft_length: length of FFT
        hop_length: length of hops/shifts of the sliding window
        power: exponent for magnitude spectrogram. Default `None` will
               return a complex-valued spectrogram
        magnitude_power: Transform magnitude of the spectrogram as x^magnitude_power.
        scale: Positive scaling of the spectrogram.
    """

    def __init__(self, fft_length: int, hop_length: int, magnitude_power: float = 1.0, scale: float = 1.0):
        super().__init__()

        # For now, assume FFT length is divisible by two
        if fft_length % 2 != 0:
            raise ValueError(f'fft_length = {fft_length} must be divisible by 2')

        self.fft_length = fft_length
        self.hop_length = hop_length
        self.pad_mode = 'constant'
        window = torch.hann_window(self.win_length)
        self.register_buffer('window', window)

        self.num_subbands = fft_length // 2 + 1

        if magnitude_power <= 0:
            raise ValueError(f'Magnitude power needs to be positive: current value {magnitude_power}')
        self.magnitude_power = magnitude_power

        if scale <= 0:
            raise ValueError(f'Scale needs to be positive: current value {scale}')
        self.scale = scale

        logging.debug('Initialized %s with:', self.__class__.__name__)
        logging.debug('\tfft_length:      %s', fft_length)
        logging.debug('\thop_length:      %s', hop_length)
        logging.debug('\tmagnitude_power: %s', magnitude_power)
        logging.debug('\tscale:           %s', scale)

    @property
    def win_length(self) -> int:
        return self.fft_length

    def stft(self, x: torch.Tensor):
        """Apply STFT as in torchaudio.transforms.Spectrogram(power=None)

        Args:
            x_spec: Input time-domain signal, shape (..., T)

        Returns:
            Time-domain signal ``x_spec = STFT(x)``, shape (..., F, N).
        """
        # pack batch
        B, C, T = x.size()
        x = rearrange(x, 'B C T -> (B C) T')

        x_spec = torch.stft(
            input=x,
            n_fft=self.fft_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode=self.pad_mode,
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # unpack batch
        x_spec = rearrange(x_spec, '(B C) F N -> B C F N', B=B, C=C)

        return x_spec

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType()),
        }

    @typecheck()
    def forward(
        self, input: torch.Tensor, input_length: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a batch of C-channel input signals
        into a batch of complex-valued spectrograms.

        Args:
            input: Time-domain input signal with C channels, shape (B, C, T)
            input_length: Length of valid entries along the time dimension, shape (B,)

        Returns:
            Output spectrogram with F subbands and N time frames, shape (B, C, F, N)
            and output length with shape (B,).
        """
        B, T = input.size(0), input.size(-1)
        input = input.view(B, -1, T)

        # STFT output (B, C, F, N)
        with torch.cuda.amp.autocast(enabled=False):
            output = self.stft(input.float())

            if self.magnitude_power != 1:
                # apply power on the magnitude
                output = torch.pow(output.abs(), self.magnitude_power) * torch.exp(1j * output.angle())

            if self.scale != 1:
                # apply scaling of the coefficients
                output = self.scale * output

        if input_length is not None:
            # Mask padded frames
            output_length = self.get_output_length(input_length=input_length)

            length_mask: torch.Tensor = make_seq_mask_like(
                lengths=output_length, like=output, time_dim=-1, valid_ones=False
            )
            output = output.masked_fill(length_mask, 0.0)
        else:
            # Assume all frames are valid for all examples in the batch
            output_length = output.size(-1) * torch.ones(B, device=output.device).long()

        return output, output_length

    def get_output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """Get length of valid frames for the output.

        Args:
            input_length: number of valid samples, shape (B,)

        Returns:
            Number of valid frames, shape (B,)
        """
        # centered STFT results in (T // hop_length + 1) frames for T samples (cf. torch.stft)
        output_length = input_length.div(self.hop_length, rounding_mode='floor').add(1).long()
        return output_length


class AudioToSpectrogramTA(NeuralModule):
    """Transform a batch of input multi-channel signals into a batch of
    STFT-based spectrograms. Using torchaudio.

    Args:
        fft_length: length of FFT
        hop_length: length of hops/shifts of the sliding window
        power: exponent for magnitude spectrogram. Default `None` will
               return a complex-valued spectrogram
        magnitude_power: Transform magnitude of the spectrogram as x^magnitude_power.
        scale: Positive scaling of the spectrogram.
    """

    def __init__(self, fft_length: int, hop_length: int, magnitude_power: float = 1.0, scale: float = 1.0):
        if not HAVE_TORCHAUDIO:
            logging.error('Could not import torchaudio. Some features might not work.')

            raise ModuleNotFoundError(
                f"torchaudio is not installed but is necessary to instantiate a {self.__class__.__name__}"
            )

        super().__init__()

        # For now, assume FFT length is divisible by two
        if fft_length % 2 != 0:
            raise ValueError(f'fft_length = {fft_length} must be divisible by 2')

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=fft_length, hop_length=hop_length, power=None, pad_mode='constant'
        )

        # number of subbands
        self.num_subbands = fft_length // 2 + 1

        if magnitude_power <= 0:
            raise ValueError(f'Magnitude power needs to be positive: current value {magnitude_power}')
        self.magnitude_power = magnitude_power

        if scale <= 0:
            raise ValueError(f'Scale needs to be positive: current value {scale}')
        self.scale = scale

        logging.debug('Initialized %s with:', self.__class__.__name__)
        logging.debug('\tfft_length:      %s', fft_length)
        logging.debug('\thop_length:      %s', hop_length)
        logging.debug('\tmagnitude_power: %s', magnitude_power)
        logging.debug('\tscale:           %s', scale)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType()),
        }

    @typecheck()
    def forward(
        self, input: torch.Tensor, input_length: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a batch of C-channel input signals
        into a batch of complex-valued spectrograms.

        Args:
            input: Time-domain input signal with C channels, shape (B, C, T)
            input_length: Length of valid entries along the time dimension, shape (B,)

        Returns:
            Output spectrogram with F subbands and N time frames, shape (B, C, F, N)
            and output length with shape (B,).
        """
        B, T = input.size(0), input.size(-1)
        input = input.view(B, -1, T)

        # STFT output (B, C, F, N)
        with torch.cuda.amp.autocast(enabled=False):
            output = self.stft(input.float())

            if self.magnitude_power != 1:
                # apply power on the magnitude
                output = torch.pow(output.abs(), self.magnitude_power) * torch.exp(1j * output.angle())

            if self.scale != 1:
                # apply scaling of the coefficients
                output = self.scale * output

        if input_length is not None:
            # Mask padded frames
            output_length = self.get_output_length(input_length=input_length)

            length_mask: torch.Tensor = make_seq_mask_like(
                lengths=output_length, like=output, time_dim=-1, valid_ones=False
            )
            output = output.masked_fill(length_mask, 0.0)
        else:
            # Assume all frames are valid for all examples in the batch
            output_length = output.size(-1) * torch.ones(B, device=output.device).long()

        return output, output_length

    def get_output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """Get length of valid frames for the output.

        Args:
            input_length: number of valid samples, shape (B,)

        Returns:
            Number of valid frames, shape (B,)
        """
        output_length = input_length.div(self.stft.hop_length, rounding_mode='floor').add(1).long()
        return output_length


class SpectrogramToAudio(NeuralModule):
    """Transform a batch of input multi-channel spectrograms into a batch of
    time-domain multi-channel signals.

    Args:
        fft_length: length of FFT
        hop_length: length of hops/shifts of the sliding window
        magnitude_power: Transform magnitude of the spectrogram as x^(1/magnitude_power).
        scale: Spectrogram will be scaled with 1/scale before the inverse transform.
    """

    def __init__(self, fft_length: int, hop_length: int, magnitude_power: float = 1.0, scale: float = 1.0):
        super().__init__()

        # For now, assume FFT length is divisible by two
        if fft_length % 2 != 0:
            raise ValueError(f'fft_length = {fft_length} must be divisible by 2')

        self.fft_length = fft_length
        self.hop_length = hop_length
        window = torch.hann_window(self.win_length)
        self.register_buffer('window', window)

        self.num_subbands = fft_length // 2 + 1

        if magnitude_power <= 0:
            raise ValueError(f'Magnitude power needs to be positive: current value {magnitude_power}')
        self.magnitude_power = magnitude_power

        if scale <= 0:
            raise ValueError(f'Scale needs to be positive: current value {scale}')
        self.scale = scale

        logging.debug('Initialized %s with:', self.__class__.__name__)
        logging.debug('\tfft_length:      %s', fft_length)
        logging.debug('\thop_length:      %s', hop_length)
        logging.debug('\tmagnitude_power: %s', magnitude_power)
        logging.debug('\tscale:           %s', scale)

    @property
    def win_length(self) -> int:
        return self.fft_length

    def istft(self, x_spec: torch.Tensor):
        """Apply iSTFT as in torchaudio.transforms.InverseSpectrogram

        Args:
            x_spec: Input complex-valued spectrogram, shape (..., F, N)

        Returns:
            Time-domain signal ``x = iSTFT(x_spec)``, shape (..., T).
        """
        if not x_spec.is_complex():
            raise ValueError("Expected `x_spec` to be complex dtype.")

        # pack batch
        B, C, F, N = x_spec.size()
        x_spec = rearrange(x_spec, 'B C F N -> (B C) F N')

        x = torch.istft(
            input=x_spec,
            n_fft=self.fft_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            length=None,
            return_complex=False,
        )

        # unpack batch
        x = rearrange(x, '(B C) T -> B C T', B=B, C=C)

        return x

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "output_length": NeuralType(('B',), LengthsType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor, input_length: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert input complex-valued spectrogram to a time-domain
        signal. Multi-channel IO is supported.

        Args:
            input: Input spectrogram for C channels, shape (B, C, F, N)
            input_length: Length of valid entries along the time dimension, shape (B,)

        Returns:
            Time-domain signal with T time-domain samples and C channels, (B, C, T)
            and output length with shape (B,).
        """
        B, F, N = input.size(0), input.size(-2), input.size(-1)
        assert F == self.num_subbands, f'Number of subbands F={F} not matching self.num_subbands={self.num_subbands}'
        input = input.view(B, -1, F, N)

        # iSTFT output (B, C, T)
        with torch.cuda.amp.autocast(enabled=False):
            output = input.cfloat()

            if self.scale != 1:
                # apply 1/scale on the coefficients
                output = output / self.scale

            if self.magnitude_power != 1:
                # apply 1/power on the magnitude
                output = torch.pow(output.abs(), 1 / self.magnitude_power) * torch.exp(1j * output.angle())
            output = self.istft(output)

        if input_length is not None:
            # Mask padded samples
            output_length = self.get_output_length(input_length=input_length)

            length_mask: torch.Tensor = make_seq_mask_like(
                lengths=output_length, like=output, time_dim=-1, valid_ones=False
            )
            output = output.masked_fill(length_mask, 0.0)
        else:
            # Assume all frames are valid for all examples in the batch
            output_length = output.size(-1) * torch.ones(B, device=output.device).long()

        return output, output_length

    def get_output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """Get length of valid samples for the output.

        Args:
            input_length: number of valid frames, shape (B,)

        Returns:
            Number of valid samples, shape (B,)
        """
        # centered STFT results in ((N-1) * hop_length) time samples for N frames (cf. torch.istft)
        output_length = input_length.sub(1).mul(self.hop_length).long()
        return output_length


class SpectrogramToAudioTA(NeuralModule):
    """Transform a batch of input multi-channel spectrograms into a batch of
    time-domain multi-channel signals. Using torchaudio.

    Args:
        fft_length: length of FFT
        hop_length: length of hops/shifts of the sliding window
        magnitude_power: Transform magnitude of the spectrogram as x^(1/magnitude_power).
        scale: Spectrogram will be scaled with 1/scale before the inverse transform.
    """

    def __init__(self, fft_length: int, hop_length: int, magnitude_power: float = 1.0, scale: float = 1.0):
        if not HAVE_TORCHAUDIO:
            logging.error('Could not import torchaudio. Some features might not work.')

            raise ModuleNotFoundError(
                f"torchaudio is not installed but is necessary to instantiate a {self.__class__.__name__}"
            )

        super().__init__()

        # For now, assume FFT length is divisible by two
        if fft_length % 2 != 0:
            raise ValueError(f'fft_length = {fft_length} must be divisible by 2')

        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=fft_length, hop_length=hop_length, pad_mode='constant'
        )

        self.num_subbands = fft_length // 2 + 1

        if magnitude_power <= 0:
            raise ValueError(f'Magnitude power needs to be positive: current value {magnitude_power}')
        self.magnitude_power = magnitude_power

        if scale <= 0:
            raise ValueError(f'Scale needs to be positive: current value {scale}')
        self.scale = scale

        logging.debug('Initialized %s with:', self.__class__.__name__)
        logging.debug('\tfft_length:      %s', fft_length)
        logging.debug('\thop_length:      %s', hop_length)
        logging.debug('\tmagnitude_power: %s', magnitude_power)
        logging.debug('\tscale:           %s', scale)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "output_length": NeuralType(('B',), LengthsType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor, input_length: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert input complex-valued spectrogram to a time-domain
        signal. Multi-channel IO is supported.

        Args:
            input: Input spectrogram for C channels, shape (B, C, F, N)
            input_length: Length of valid entries along the time dimension, shape (B,)

        Returns:
            Time-domain signal with T time-domain samples and C channels, (B, C, T)
            and output length with shape (B,).
        """
        B, F, N = input.size(0), input.size(-2), input.size(-1)
        assert F == self.num_subbands, f'Number of subbands F={F} not matching self.num_subbands={self.num_subbands}'
        input = input.view(B, -1, F, N)

        # iSTFT output (B, C, T)
        with torch.cuda.amp.autocast(enabled=False):
            output = input.cfloat()

            if self.scale != 1:
                # apply 1/scale on the coefficients
                output = output / self.scale

            if self.magnitude_power != 1:
                # apply 1/power on the magnitude
                output = torch.pow(output.abs(), 1 / self.magnitude_power) * torch.exp(1j * output.angle())
            output = self.istft(output)

        if input_length is not None:
            # Mask padded samples
            output_length = self.get_output_length(input_length=input_length)

            length_mask: torch.Tensor = make_seq_mask_like(
                lengths=output_length, like=output, time_dim=-1, valid_ones=False
            )
            output = output.masked_fill(length_mask, 0.0)
        else:
            # Assume all frames are valid for all examples in the batch
            output_length = output.size(-1) * torch.ones(B, device=output.device).long()

        return output, output_length

    def get_output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """Get length of valid samples for the output.

        Args:
            input_length: number of valid frames, shape (B,)

        Returns:
            Number of valid samples, shape (B,)
        """
        output_length = input_length.sub(1).mul(self.istft.hop_length).long()
        return output_length
