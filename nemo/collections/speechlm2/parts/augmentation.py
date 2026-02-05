# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Audio augmentation utilities for speech models."""

import glob
import os
import random
import subprocess
import tempfile
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, fftconvolve, lfilter

try:
    import pyloudnorm as pyln

    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False


class AudioAugmenter:
    """Audio augmentation with noise, impulse responses, and codec simulation."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._noise_files_cache = {}
        self._roomir_files_cache = {}
        self._micir_files_cache = {}
        self._lowpass_filter_cache = {}

    def add_noise_to_batch(
        self,
        batch_audio: torch.Tensor,
        noise_folder: str,
        snr_db: float = 20.0,
        noise_prob_scale_user: float = 0.3,
        noise_prob_scale_user_min_snr: float = -15.0,
        noise_prob_scale_user_max_snr: float = 24.0,
        snr_measure_dur: float = 0.0,
        noise_resample: bool = True,
        noise_prob_low_pass: float = 0.1,
    ) -> torch.Tensor:
        """Add noise to batch audio with specified SNR."""
        batch_size = batch_audio.shape[0]
        audio_length = batch_audio.shape[1]

        if noise_folder not in self._noise_files_cache:
            noise_files = [f for f in glob.glob(os.path.join(noise_folder, "*.wav"))]
            if not noise_files:
                raise ValueError(f"No noise files found in {noise_folder}")
            self._noise_files_cache[noise_folder] = noise_files
        else:
            noise_files = self._noise_files_cache[noise_folder]

        for i in range(batch_size):

            def get_scale_factor(signal, noise, snr_db):
                if snr_measure_dur > 0:
                    signal = signal[: int(snr_measure_dur * self.sample_rate)]
                    noise = noise[: int(snr_measure_dur * self.sample_rate)]
                signal_power = torch.mean(signal**2) + 1e-8
                noise_power = torch.mean(noise**2) + 1e-8

                target_noise_power = signal_power / (10 ** (snr_db / 10))
                scaling_factor = torch.sqrt(target_noise_power / noise_power)
                return scaling_factor

            if random.random() < noise_prob_scale_user:
                scaling_factor = get_scale_factor(
                    batch_audio[i],
                    batch_audio[i],
                    random.randint(int(noise_prob_scale_user_min_snr), int(noise_prob_scale_user_max_snr)),
                )
                batch_audio[i] = batch_audio[i] * scaling_factor

            def get_noise(noise_files):
                noise_path = random.choice(noise_files)
                noise, sr = sf.read(noise_path, dtype='float32')

                if noise_resample and sr != self.sample_rate:
                    noise = librosa.resample(noise, orig_sr=sr, target_sr=self.sample_rate)

                if len(noise.shape) > 1:
                    noise = np.mean(noise, axis=1)

                noise_tensor = torch.tensor(noise, dtype=batch_audio.dtype, device=batch_audio.device)
                scaling_factor = get_scale_factor(batch_audio[i], noise_tensor, snr_db)
                noise_tensor = noise_tensor * scaling_factor
                return noise_tensor

            noise = get_noise(noise_files)
            noise2 = get_noise(noise_files)
            noise3 = get_noise(noise_files)
            noise = torch.cat([noise, noise2, noise3], axis=0)

            if noise.size(0) < audio_length:
                repeat_times = (audio_length // noise.size(0)) + 1
                noise = noise.repeat(repeat_times)[:audio_length]
            else:
                start_idx = torch.randint(0, noise.size(0) - audio_length + 1, (1,)).item()
                noise = noise[start_idx : start_idx + audio_length]

            if random.random() < noise_prob_low_pass:
                cutoff = 1000.0
                noise = self._apply_lowpass_filter(noise, cutoff)

            batch_audio[i] = batch_audio[i] + noise

        return batch_audio

    def _apply_lowpass_filter(self, audio: torch.Tensor, cutoff: float, order: int = 5) -> torch.Tensor:
        """Apply a low-pass Butterworth filter to audio."""
        cache_key = (cutoff, self.sample_rate, order)
        if cache_key not in self._lowpass_filter_cache:
            nyquist = 0.5 * self.sample_rate
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            self._lowpass_filter_cache[cache_key] = (b, a)

        b, a = self._lowpass_filter_cache[cache_key]
        y_cpu = lfilter(b, a, audio.cpu().numpy())
        y_gpu = torch.tensor(y_cpu, dtype=torch.float32, device=audio.device)
        return y_gpu

    def add_room_ir_to_batch(
        self,
        batch_audio: torch.Tensor,
        audio_lens: Optional[torch.Tensor],
        roomir_folder: str,
        use_loudness_norm: bool = True,
    ) -> torch.Tensor:
        """Apply room impulse response to batch audio."""
        batch_size = batch_audio.shape[0]

        if roomir_folder not in self._roomir_files_cache:
            roomir_files = [f for f in glob.glob(os.path.join(roomir_folder, "*.wav"))]
            if not roomir_files:
                raise ValueError(f"No room IR files found in {roomir_folder}")
            self._roomir_files_cache[roomir_folder] = roomir_files
        else:
            roomir_files = self._roomir_files_cache[roomir_folder]

        for i in range(batch_size):
            audio_length = audio_lens[i].item() if audio_lens is not None else batch_audio.shape[1]

            ir_path = random.choice(roomir_files)
            ir, sr = sf.read(ir_path, dtype='float32')

            if sr != self.sample_rate:
                ir = librosa.resample(ir, orig_sr=sr, target_sr=self.sample_rate)

            if len(ir.shape) > 1:
                ir = np.mean(ir, axis=1)

            audio_cpu = batch_audio[i, :audio_length].cpu().numpy()

            if use_loudness_norm and PYLOUDNORM_AVAILABLE:
                meter = pyln.Meter(self.sample_rate)
                try:
                    speech_loudness = meter.integrated_loudness(audio_cpu)
                    # Check for invalid loudness values (-inf for silent audio)
                    if speech_loudness == float('-inf') or not np.isfinite(speech_loudness):
                        use_loudness_norm = False
                except Exception:
                    use_loudness_norm = False

            convolved = fftconvolve(audio_cpu, ir, mode="full")[:audio_length]

            # Calculate RMS before convolution for gain compensation
            input_rms = np.sqrt(np.mean(audio_cpu**2)) + 1e-8
            convolved_rms = np.sqrt(np.mean(convolved**2)) + 1e-8

            if use_loudness_norm and PYLOUDNORM_AVAILABLE:
                try:
                    convolved_loudness = meter.integrated_loudness(convolved)
                    # Check for invalid loudness values
                    if (
                        convolved_loudness != float('-inf')
                        and np.isfinite(convolved_loudness)
                        and np.isfinite(speech_loudness)
                    ):
                        convolved = pyln.normalize.loudness(convolved, convolved_loudness, speech_loudness)
                        # Validate output doesn't contain NaN or inf
                        if not np.isfinite(convolved).all():
                            convolved = fftconvolve(audio_cpu, ir, mode="full")[:audio_length]
                    else:
                        # Fallback: Use RMS-based gain compensation to restore signal level
                        gain_compensation = input_rms / convolved_rms
                        gain_compensation = min(gain_compensation, 10.0)  # Max 20dB boost
                        convolved = convolved * gain_compensation
                except Exception:
                    # Still apply gain compensation as fallback
                    gain_compensation = input_rms / convolved_rms
                    gain_compensation = min(gain_compensation, 10.0)
                    convolved = convolved * gain_compensation
            else:
                # If loudness normalization is disabled, apply simple RMS-based gain compensation
                gain_compensation = input_rms / convolved_rms
                gain_compensation = min(gain_compensation, 10.0)
                convolved = convolved * gain_compensation

            # Clip to prevent extreme values
            convolved = np.clip(convolved, -1.0, 1.0)

            batch_audio[i, :audio_length] = torch.tensor(convolved, dtype=batch_audio.dtype, device=batch_audio.device)

        return batch_audio

    def add_mic_ir_to_batch(
        self,
        batch_audio: torch.Tensor,
        audio_lens: Optional[torch.Tensor],
        micir_folder: str,
        use_loudness_norm: bool = True,
    ) -> torch.Tensor:
        """Apply microphone impulse response to batch audio."""
        batch_size = batch_audio.shape[0]

        if micir_folder not in self._micir_files_cache:
            micir_files = [f for f in glob.glob(os.path.join(micir_folder, "*.wav"))]
            if not micir_files:
                raise ValueError(f"No mic IR files found in {micir_folder}")
            self._micir_files_cache[micir_folder] = micir_files
        else:
            micir_files = self._micir_files_cache[micir_folder]

        for i in range(batch_size):
            audio_length = audio_lens[i].item() if audio_lens is not None else batch_audio.shape[1]

            ir_path = random.choice(micir_files)
            ir, sr = sf.read(ir_path, dtype='float32')

            if sr != self.sample_rate:
                ir = librosa.resample(ir, orig_sr=sr, target_sr=self.sample_rate)

            if len(ir.shape) > 1:
                ir = np.mean(ir, axis=1)

            audio_cpu = batch_audio[i, :audio_length].cpu().numpy()

            if use_loudness_norm and PYLOUDNORM_AVAILABLE:
                meter = pyln.Meter(self.sample_rate)
                try:
                    speech_loudness = meter.integrated_loudness(audio_cpu)
                    # Check for invalid loudness values (-inf for silent audio)
                    if speech_loudness == float('-inf') or not np.isfinite(speech_loudness):
                        use_loudness_norm = False
                except Exception:
                    use_loudness_norm = False

            convolved = fftconvolve(audio_cpu, ir, mode="full")[:audio_length]

            # Calculate RMS before convolution for gain compensation
            input_rms = np.sqrt(np.mean(audio_cpu**2)) + 1e-8
            convolved_rms = np.sqrt(np.mean(convolved**2)) + 1e-8

            if use_loudness_norm and PYLOUDNORM_AVAILABLE:
                try:
                    convolved_loudness = meter.integrated_loudness(convolved)
                    # Check for invalid loudness values
                    if (
                        convolved_loudness != float('-inf')
                        and np.isfinite(convolved_loudness)
                        and np.isfinite(speech_loudness)
                    ):
                        convolved = pyln.normalize.loudness(convolved, convolved_loudness, speech_loudness)
                        # Validate output doesn't contain NaN or inf
                        if not np.isfinite(convolved).all():
                            convolved = fftconvolve(audio_cpu, ir, mode="full")[:audio_length]
                    else:
                        # Fallback: Use RMS-based gain compensation
                        gain_compensation = input_rms / convolved_rms
                        gain_compensation = min(gain_compensation, 10.0)
                        convolved = convolved * gain_compensation
                except Exception:
                    # Still apply gain compensation as fallback
                    gain_compensation = input_rms / convolved_rms
                    gain_compensation = min(gain_compensation, 10.0)
                    convolved = convolved * gain_compensation
            else:
                # If loudness normalization is disabled, apply simple RMS-based gain compensation
                gain_compensation = input_rms / convolved_rms
                gain_compensation = min(gain_compensation, 10.0)
                convolved = convolved * gain_compensation

            # Clip to prevent extreme values
            convolved = np.clip(convolved, -1.0, 1.0)

            batch_audio[i, :audio_length] = torch.tensor(convolved, dtype=batch_audio.dtype, device=batch_audio.device)

        return batch_audio

    def add_codec_to_batch(
        self,
        batch_audio: torch.Tensor,
        audio_lens: Optional[torch.Tensor],
        codec_settings: dict,
    ) -> torch.Tensor:
        """Apply codec degradation to batch audio using FFmpeg."""
        batch_size = batch_audio.shape[0]

        for i in range(batch_size):
            audio_length = audio_lens[i].item() if audio_lens is not None else batch_audio.shape[1]

            codec_name, codec_args = random.choice(list(codec_settings.items()))

            audio_cpu = batch_audio[i, :audio_length].cpu().numpy()

            try:
                degraded = self._apply_ffmpeg_codec(audio_cpu, codec_args)

                # Validate output doesn't contain NaN or inf
                if np.isfinite(degraded).all():
                    degraded = np.clip(degraded, -1.0, 1.0)
                    batch_audio[i, :audio_length] = torch.tensor(
                        degraded, dtype=batch_audio.dtype, device=batch_audio.device
                    )
            except Exception:
                # Codec failed, skip augmentation for this sample
                pass

        return batch_audio

    def _apply_ffmpeg_codec(self, audio: np.ndarray, codec_args: list) -> np.ndarray:
        """Apply audio compression/decompression using FFmpeg."""
        target_len = len(audio)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_wav = os.path.join(tmpdir, "input.wav")
            fmt = codec_args[codec_args.index("-f") + 1] if "-f" in codec_args else "wav"
            mid_file = os.path.join(tmpdir, f"compressed.{fmt}")
            out_wav = os.path.join(tmpdir, "output.wav")

            sf.write(in_wav, audio, samplerate=self.sample_rate, subtype='PCM_16')

            subprocess.run(
                ["ffmpeg", "-y", "-i", in_wav] + codec_args + [mid_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

            subprocess.run(
                ["ffmpeg", "-y", "-i", mid_file, "-ar", str(self.sample_rate), out_wav],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

            decoded, _ = sf.read(out_wav, dtype='float32')

            if len(decoded) > target_len:
                decoded = decoded[:target_len]
            elif len(decoded) < target_len:
                pad = np.zeros(target_len - len(decoded), dtype=decoded.dtype)
                decoded = np.concatenate([decoded, pad])

            return decoded


DEFAULT_CODEC_SETTINGS = {
    "high_libopus_8k_5k": ["-ar", "8000", "-c:a", "libopus", "-application", "voip", "-b:a", "5.5k", "-f", "ogg"],
    "high_g726_8k_16k": ["-ar", "8000", "-c:a", "adpcm_g726", "-b:a", "16k", "-f", "wav"],
    "med_libopus_8k_9k": ["-ar", "8000", "-c:a", "libopus", "-application", "voip", "-b:a", "9.5k", "-f", "ogg"],
    "med_libopus_16k_12k": ["-ar", "16000", "-c:a", "libopus", "-application", "voip", "-b:a", "12k", "-f", "ogg"],
    "med_libvorbis_16k_32k": ["-ar", "16000", "-c:a", "libvorbis", "-b:a", "32k", "-f", "ogg"],
    "med_mp3_16k_32k": ["-ar", "16000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "32k", "-f", "mp3"],
    "low_mulaw_8k": ["-ar", "8000", "-c:a", "pcm_mulaw", "-f", "wav"],
    "low_alaw_8k": ["-ar", "8000", "-c:a", "pcm_alaw", "-f", "wav"],
    "low_g722_16k": ["-ar", "16000", "-c:a", "g722", "-f", "wav"],
    "low_g726_8k_32k": ["-ar", "8000", "-c:a", "adpcm_g726", "-b:a", "32k", "-f", "wav"],
    "low_libopus_16k_32k": ["-ar", "16000", "-c:a", "libopus", "-application", "audio", "-b:a", "32k", "-f", "ogg"],
    "low_libopus_24k_32k": ["-ar", "24000", "-c:a", "libopus", "-application", "audio", "-b:a", "32k", "-f", "ogg"],
    "low_libopus_24k_48k": ["-ar", "24000", "-c:a", "libopus", "-application", "audio", "-b:a", "48k", "-f", "ogg"],
    "low_libvorbis_24k_64k": ["-ar", "24000", "-c:a", "libvorbis", "-b:a", "64k", "-f", "ogg"],
    "low_mp3_24k_64k": ["-ar", "24000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"],
}
