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

    def __init__(
        self, fft_length: int, hop_length: int, magnitude_power: float = 1.0, scale: float = 1.0, center: bool = True
    ):
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
        self.center = center
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
            center=self.center,
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
        with torch.amp.autocast(input.device.type, enabled=False):
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


class SpectrogramToAudio(NeuralModule):
    """Transform a batch of input multi-channel spectrograms into a batch of
    time-domain multi-channel signals.

    Args:
        fft_length: length of FFT
        hop_length: length of hops/shifts of the sliding window
        magnitude_power: Transform magnitude of the spectrogram as x^(1/magnitude_power).
        scale: Spectrogram will be scaled with 1/scale before the inverse transform.

    Streaming usage (center=False):

    ```python
    # analysis should use the same window and center=False
    # Prefer hamming for center=False (see note below)
    window = torch.hamming_window(fft_length)
    spec2audio = SpectrogramToAudio(fft_length=fft_length, hop_length=hop_length, center=False)
    spec2audio.window = window
    spec2audio.use_streaming = True
    spec2audio.reset_streaming()

    parts = []
    for t in range(0, N, K):
        frames = spec[..., t : t + K]   # (B, C, F, K), complex
        out, _ = spec2audio(input=frames)
        parts.append(out)
    tail = spec2audio.stream_finalize()
    x_stream = torch.cat(parts + [tail], dim=-1)
    ```

    Notes: ``window`` must match analysis; call ``reset_streaming()`` before a new stream;
    ``stream_finalize()`` flushes the tail (empty if ``hop_length == win_length``).
    With ``center=False``, certain windows (e.g., Hann) may error in
    some PyTorch versions; Hamming works reliably. See
    `PyTorch issue #91309 <https://github.com/pytorch/pytorch/issues/91309>`_.
    """

    def __init__(
        self, fft_length: int, hop_length: int, magnitude_power: float = 1.0, scale: float = 1.0, center: bool = True
    ):
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
        self.center = center
        if scale <= 0:
            raise ValueError(f'Scale needs to be positive: current value {scale}')
        self.scale = scale

        logging.debug('Initialized %s with:', self.__class__.__name__)
        logging.debug('\tfft_length:      %s', fft_length)
        logging.debug('\thop_length:      %s', hop_length)
        logging.debug('\tmagnitude_power: %s', magnitude_power)
        logging.debug('\tscale:           %s', scale)

        # --- Streaming state (initialized lazily) ---
        # Time-domain overlap-add buffers (initialized lazily)
        self._ola_accum: Optional[torch.Tensor] = None
        self._ola_weight: Optional[torch.Tensor] = None
        # Kept for backward compatibility; not used in OLA implementation
        self.use_streaming: bool = False

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
        # pack batch
        B, C, F, N = x_spec.size()
        x_spec = rearrange(x_spec, 'B C F N -> (B C) F N')

        x = torch.istft(
            input=x_spec,
            n_fft=self.fft_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
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

        Offline mode (default): processes the entire input spectrogram at once.
        Streaming mode: expects one or more frames (N>=1) and returns hop_length * N samples per call.

        Args:
            input: Input spectrogram for C channels, shape (B, C, F, N)
            input_length: Length of valid entries along the time dimension, shape (B,)

        Returns:
            - Offline: (B, C, T_total), lengths (B,)
            - Streaming (N=1): (B, C, hop_length), lengths (B,) filled with hop_length
        """
        B, F, N = input.size(0), input.size(-2), input.size(-1)
        assert F == self.num_subbands, f'Number of subbands F={F} not matching self.num_subbands={self.num_subbands}'
        input = input.view(B, -1, F, N)

        if not input.is_complex():
            raise ValueError("Expected `input` to be complex dtype.")

        # iSTFT output (B, C, T)
        with torch.amp.autocast(input.device.type, enabled=False):
            output = input.cfloat()

            if self.scale != 1:
                # apply 1/scale on the coefficients
                output = output / self.scale

            if self.magnitude_power != 1:
                # apply 1/power on the magnitude
                output = torch.pow(output.abs(), 1 / self.magnitude_power) * torch.exp(1j * output.angle())

            # --- Streaming mode ---
            if self.use_streaming:
                # Streaming expects a single frame at a time to avoid internal iteration.
                out_stream = self.stream_update(output)  # (B, C, <= hop_length)
                out_len = torch.full((B,), out_stream.size(-1), dtype=torch.long, device=out_stream.device)
                return out_stream, out_len

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

    @property
    def _stream_initialized(self) -> bool:
        """Return True if streaming buffers are initialized."""
        return (self._ola_accum is not None) and (self._ola_weight is not None)

    @property
    def _eps(self) -> float:
        """Machine epsilon for the active streaming dtype."""
        dtype = self._ola_weight.dtype if self._ola_weight is not None else self.window.dtype
        return torch.finfo(dtype).eps

    # ------------------------------------------------------------------
    # Streaming iSTFT API (frame-by-frame with overlap-add buffering)
    # ------------------------------------------------------------------
    def _init_stream_buffers(self, shape_like: torch.Tensor) -> None:
        """Initialize streaming buffers based on an input tensor."""
        if self._stream_initialized:
            return
        if shape_like.dim() != 4:
            raise ValueError("Expected input of shape (B, C, F, N_frames) for streaming.")
        B, C = shape_like.size(0), shape_like.size(1)
        device = shape_like.device
        # Real-valued buffers for accumulated time-domain samples and weights
        dtype = torch.float32 if shape_like.dtype == torch.complex64 else torch.float64
        self._ola_accum = torch.zeros(B, C, self.win_length, device=device, dtype=dtype)
        self._ola_weight = torch.zeros(B, C, self.win_length, device=device, dtype=dtype)

    def reset_streaming(self) -> None:
        """Reset the internal streaming buffers.

        Re-initialization happens lazily on the next call to `stream_update`.
        """
        self._ola_accum = None
        self._ola_weight = None

    def _shift_left_inplace(self, buffer: torch.Tensor) -> None:
        """Shift buffer left by hop length and zero-fill the tail in-place."""
        hop = self.hop_length
        buffer[..., :-hop] = buffer[..., hop:].clone()
        buffer[..., -hop:] = 0.0

    @torch.no_grad()
    def stream_update(self, input: torch.Tensor) -> torch.Tensor:
        """Consume one or more spectrogram frames (N>=1) and return hop_length * N samples via OLA.

        Steps per frame:
        - inverse FFT
        - apply synthesis window
        - overlap-add into accumulation buffer
        - emit first hop_length samples normalized by window-sum-square
        - shift buffers left by hop_length
        """
        if not input.is_complex():
            raise ValueError("Expected `input` to be complex dtype for streaming.")

        if self.center:
            raise ValueError("Streaming iSTFT requires center=False.")

        # Lazily initialize buffers
        self._init_stream_buffers(input)

        B, C, F, num_frames = input.size()
        assert F == self.num_subbands, f"Number of subbands F={F} not matching self.num_subbands={self.num_subbands}"

        # Vectorized inverse FFT over frequency bins (dim=-2), yields (B, C, T, N)
        frames_time = torch.fft.irfft(input, n=self.fft_length, dim=-2)

        # Prepare window and ensure buffers are on correct device/dtype
        hop = self.hop_length
        emitted_parts = []
        # Window shaped for broadcasting over frames
        win = self.window.to(frames_time.device, dtype=frames_time.dtype).view(1, 1, self.win_length, 1)
        win_sq = win[..., 0].squeeze(-1).pow(2)  # (1, 1, T)
        frames_time_windowed = frames_time * win  # (B, C, T, N)

        # Ensure buffers on correct device/dtype
        self._ola_accum = self._ola_accum.to(frames_time_windowed.device, dtype=frames_time_windowed.dtype)
        self._ola_weight = self._ola_weight.to(frames_time_windowed.device, dtype=frames_time_windowed.dtype)

        # Iterate over frames for OLA
        for t in range(num_frames):
            frame_t = frames_time_windowed[..., t]  # (B, C, T)

            # Overlap-add accumulation and window-sum-square weights
            self._ola_accum.add_(frame_t)
            self._ola_weight.add_(win_sq)

            # Emit first hop_length samples with normalization
            denom = torch.clamp(self._ola_weight[..., :hop], min=self._eps)
            emitted = self._ola_accum[..., :hop] / denom
            emitted_parts.append(emitted)

            # Shift buffers left by hop_length
            self._shift_left_inplace(self._ola_accum)
            self._shift_left_inplace(self._ola_weight)

        return torch.cat(emitted_parts, dim=-1)

    @torch.no_grad()
    def stream_finalize(self) -> torch.Tensor:
        """Flush the remaining buffered samples (final tail for center=False).

        After processing the last frame, the streaming loop has emitted N*hop
        samples. The remaining tail corresponds to the last (win_length - hop)
        samples, which we return after proper window-sum-square normalization.
        """
        if not self._stream_initialized:
            return torch.tensor((), device=self.window.device)

        tail_len = self.win_length - self.hop_length
        if tail_len <= 0:
            return torch.tensor((), device=self.window.device)

        denom_tail = torch.clamp(self._ola_weight[..., :tail_len], min=self._eps)
        tail = self._ola_accum[..., :tail_len] / denom_tail
        return tail
