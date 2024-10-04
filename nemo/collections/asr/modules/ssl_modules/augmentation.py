# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import torch

from nemo.collections.asr.data.ssl_dataset import AudioNoiseBatch


class SpeakerNoiseAugmentation(object):
    def __init__(
        self,
        prob: float = 0.0,
        noise_ratio: float = 0.0,
        min_r_speech: float = -5.0,
        max_r_speech: float = 5.0,
        min_r_noise: float = -5.0,
        max_r_noise: float = 20.0,
        min_mix_rate: float = 0.0,
        max_mix_rate: float = 1.0,
    ):
        super().__init__()
        self.prob = prob
        self.noise_ratio = noise_ratio
        self.min_r_speech = min_r_speech
        self.max_r_speech = max_r_speech
        self.min_r_noise = min_r_noise
        self.max_r_noise = max_r_noise
        self.min_mix_rate = min_mix_rate
        self.max_mix_rate = max_mix_rate

        if not (0 <= self.prob <= 1):
            raise ValueError(f"prob must be in [0, 1], got: {self.prob}")
        if not (0 <= self.noise_ratio <= 1):
            raise ValueError(f"noise_ratio must be in [0, 1], got: {self.noise_ratio}")
        if not (self.min_r_speech <= self.max_r_speech):
            raise ValueError(
                f"min_r_speech must be no greater than max_r_speech, got: min={self.min_r_speech} and max={self.max_r_speech}"
            )
        if not (self.min_r_noise <= self.max_r_noise):
            raise ValueError(
                f"min_r_noise must be no greater than max_r_noise, got: min={self.min_r_noise} and max={self.max_r_noise}"
            )
        if not (0 <= self.min_mix_rate <= self.max_mix_rate <= 1):
            raise ValueError(
                f"min_mix_rate must be no greater than max_mix_rate, and both must be in [0, 1], got: {self.min_mix_rate} and {self.max_mix_rate}"
            )

    def repeat_noise(self, noise: torch.Tensor, noise_len: int, max_audio_len: int) -> torch.Tensor:
        noise = noise[:noise_len]
        if noise_len < max_audio_len:
            noise = noise.repeat(max_audio_len // noise_len + 1)
        noise = noise[:max_audio_len]
        return noise

    def pad_or_trim_noise(self, noise: torch.Tensor, max_audio_len: int, value=0) -> torch.Tensor:
        noise_len = noise.size(0)
        if noise_len < max_audio_len:
            pad = (0, max_audio_len - noise_len)
            noise = torch.nn.functional.pad(noise, pad, value=value)
        else:
            noise = noise[:max_audio_len]
        return noise

    def __call__(self, batch: AudioNoiseBatch) -> AudioNoiseBatch:
        audio_signal = batch.audio
        audio_lengths = batch.audio_len
        batch_size = audio_signal.size(0)
        max_audio_len = audio_signal.size(1)

        noise = batch.noise
        noise_len = batch.noise_len
        noisy_audio = batch.noisy_audio
        noisy_audio_len = batch.noisy_audio_len
        for i in range(batch_size):
            if np.random.rand() > self.prob:
                continue

            # randomly select the length of mixing segment
            if 0 <= self.min_mix_rate < self.max_mix_rate <= 1:
                mix_len = np.random.randint(
                    int(audio_lengths[i] * self.min_mix_rate), int(audio_lengths[i] * self.max_mix_rate)
                )
            else:
                mix_len = max(1, int(audio_lengths[i] * self.min_mix_rate))

            # randomly select position to start the mixing
            mix_start_idx = np.random.randint(audio_lengths[i] - mix_len)

            # randomly select the energy ratio between speech and noise
            if np.random.rand() < self.noise_ratio or batch_size == 1:
                energy_ratio = np.random.uniform(self.min_r_noise, self.max_r_noise)
            else:
                energy_ratio = np.random.uniform(self.min_r_speech, self.max_r_speech)
                j = np.random.choice([x for x in range(batch_size) if x != i])
                noise[i] = audio_signal[j].clone()
                noise_len[i] = audio_lengths[j]

            # repeat noise to match the length of audio mix length if necessary
            if noise_len[i] <= mix_len:
                # repeat noise to match the length of audio mix length
                noise_start_idx = 0
                noise[i] = self.pad_or_trim_noise(self.repeat_noise(noise[i], noise_len[i], mix_len), max_audio_len)
                noise_len[i] = mix_len
            else:
                # randomly select a segment of noise
                noise_start_idx = np.random.randint(noise_len[i] - mix_len)

            # calculate the scale factor for noise
            audio_energy = torch.sum(audio_signal[i, : audio_lengths[i]] ** 2) / audio_lengths[i]
            noise_energy = torch.sum(noise[i, : noise_len[i]] ** 2) / noise_len[i] if noise_len[i] > 0 else 0
            mix_scale = math.sqrt(audio_energy / (10 ** (energy_ratio / 10) * noise_energy)) if noise_energy > 0 else 0

            # get the residual signal to be added to original audio
            noise_clip = noise[i, noise_start_idx : noise_start_idx + mix_len]
            noise_signal = torch.zeros_like(audio_signal[i])
            noise_signal[mix_start_idx : mix_start_idx + mix_len] = mix_scale * noise_clip

            # add noise to audio
            noisy_audio[i] = audio_signal[i] + noise_signal
            noisy_audio_len[i] = audio_lengths[i]
            noise[i] = noise_signal
            noise_len[i] = audio_lengths[i]

        return AudioNoiseBatch(
            sample_id=batch.sample_id,
            audio=batch.audio,
            audio_len=batch.audio_len,
            noise=noise,
            noise_len=noise_len,
            noisy_audio=noisy_audio,
            noisy_audio_len=noisy_audio_len,
        )


class MultiSpeakerNoiseAugmentation(SpeakerNoiseAugmentation):
    def __init__(
        self,
        prob: float = 0.0,
        noise_ratio: float = 0.0,
        min_r_speech: float = -5.0,
        max_r_speech: float = 5.0,
        min_r_noise: float = -5.0,
        max_r_noise: float = 20.0,
        min_mix_rate: float = 0.0,
        max_mix_rate: float = 1.0,
        min_num_segments: int = 1,
        max_num_segments: int = 5,
        min_num_speakers: int = 1,
        max_num_speakers: int = 4,
    ):
        super().__init__(
            prob=prob,
            noise_ratio=noise_ratio,
            min_r_speech=min_r_speech,
            max_r_speech=max_r_speech,
            min_r_noise=min_r_noise,
            max_r_noise=max_r_noise,
            min_mix_rate=min_mix_rate,
            max_mix_rate=max_mix_rate,
        )
        self.min_num_segments = min_num_segments
        self.max_num_segments = max_num_segments
        self.min_num_speakers = min_num_speakers
        self.max_num_speakers = max_num_speakers

    def __call__(self, batch: AudioNoiseBatch) -> AudioNoiseBatch:
        audio_signal = batch.audio
        audio_lengths = batch.audio_len
        batch_size = audio_signal.size(0)

        noise = batch.noise
        noise_len = batch.noise_len
        noisy_audio = batch.noisy_audio
        noisy_audio_len = batch.noisy_audio_len
        for i in range(batch_size):
            if np.random.rand() > self.prob:
                continue

            # randomly select the length of mixing segment
            if 0 <= self.min_mix_rate < self.max_mix_rate <= 1:
                mix_rate = np.random.uniform(self.min_mix_rate, self.max_mix_rate)
            else:
                mix_rate = self.min_mix_rate
            mix_len = max(1, int(audio_lengths[i] * mix_rate))

            # randomly select the number of segments
            num_segments = np.random.randint(self.min_num_segments, self.max_num_segments + 1)
            num_speakers = np.random.randint(self.min_num_speakers, self.max_num_speakers + 1)
            num_speakers = min(num_speakers, batch_size)

            # randomly chunk mix_len into num_segments
            segment_lens = np.random.multinomial(mix_len, [1 / num_segments] * num_segments)

            # randomly select the energy ratio between speech and noise
            if np.random.rand() < self.noise_ratio or batch_size == 1:
                mode = "noise"
                energy_ratio = np.random.uniform(self.min_r_noise, self.max_r_noise)
            else:
                mode = "speech"
                energy_ratio = np.random.uniform(self.min_r_speech, self.max_r_speech)

            noise_segments = self.get_noise_segments(i, batch, segment_lens, num_speakers, mode)
            noise_signal = torch.zeros_like(audio_signal[i])
            min_start_idx = 0
            max_start_idx = audio_lengths[i] - mix_len
            for j in range(num_segments):
                start_idx = min_start_idx
                if min_start_idx < max_start_idx:
                    start_idx = np.random.randint(min_start_idx, max_start_idx)
                noise_signal[start_idx : start_idx + segment_lens[j]] = noise_segments[j]
                min_start_idx = start_idx + segment_lens[j]
                max_start_idx += segment_lens[j]

            # calculate the scale factor for noise
            audio_energy = torch.sum(audio_signal[i, : audio_lengths[i]] ** 2) / audio_lengths[i]
            noise_energy = torch.sum(noise_signal[: audio_lengths[i]] ** 2) / audio_lengths[i]
            mix_scale = math.sqrt(audio_energy / (10 ** (energy_ratio / 10) * noise_energy)) if noise_energy > 0 else 0

            # get the residual signal to be added to original audio
            noise_signal = mix_scale * noise_signal

            # add noise to audio
            noisy_audio[i] = audio_signal[i] + noise_signal
            noisy_audio_len[i] = audio_lengths[i]
            noise[i] = noise_signal
            noise_len[i] = audio_lengths[i]

        return AudioNoiseBatch(
            sample_id=batch.sample_id,
            audio=batch.audio,
            audio_len=batch.audio_len,
            noise=noise,
            noise_len=noise_len,
            noisy_audio=noisy_audio,
            noisy_audio_len=noisy_audio_len,
        )

    def get_noise_segments(self, batch_idx, batch, segment_lens, num_speakers, mode):
        audio_signal = batch.audio
        audio_lengths = batch.audio_len
        noise = batch.noise
        noise_len = batch.noise_len
        batch_size = noise.size(0)
        max_audio_len = audio_signal.size(1)
        noise_segments = []
        if mode == "noise":
            noise_padded = self.pad_or_trim_noise(
                self.repeat_noise(noise[batch_idx], noise_len[batch_idx], max_audio_len), max_audio_len
            )
            start_idx = 0
            for segment_len in segment_lens:
                noise_segments.append(noise_padded[start_idx : start_idx + segment_len])
                start_idx += segment_len
            return noise_segments

        if mode != "speech":
            raise ValueError(f"mode must be either 'noise' or 'speech', got: {mode}")

        speaker_candidates = [x for x in range(batch_size) if x != batch_idx]
        speaker_candidates = np.random.choice(speaker_candidates, min(num_speakers, batch_size - 1), replace=False)
        sid = 0
        for seg_len in segment_lens:
            bid = speaker_candidates[sid]
            if seg_len > audio_lengths[bid]:
                audio_segment = self.pad_or_trim_noise(
                    self.repeat_noise(audio_signal[bid], audio_lengths[bid], seg_len), seg_len
                )
            else:
                start_idx = np.random.randint(audio_lengths[bid] - seg_len) if audio_lengths[bid] > seg_len else 0
                audio_segment = audio_signal[bid][start_idx : start_idx + seg_len].clone()
            noise_segments.append(audio_segment)
            sid += 1
            if sid >= len(speaker_candidates):
                sid = np.random.randint(len(speaker_candidates))

        return noise_segments
