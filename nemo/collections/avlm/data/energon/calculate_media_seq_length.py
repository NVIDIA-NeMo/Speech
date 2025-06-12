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

import math
import torch


def calculate_encoded_audio_seq_length(
    model_type: str,
    audio_length: int,
    fixed_max_audio_length: int,
    sample_rate: int,
    window_stride: int,
    encoder_down_sampling: int,
    num_mel_bins: int,
    patch_size: int,
    time_stride: int,
    frequency_stride: int,
    max_spectrogram_length: int,
):
    """
    Calculate the sequence length of the encoded audio based on:
      - length of the audio signal
      - audio encoder's configuration (e.g. model_type, window_stride, encoder_down_sampling, etc.)
    """

    if model_type in ["whisper"]:
        assert window_stride is not None, "window_stride must be provided for whisper model"
        assert sample_rate is not None, "sample_rate must be provided for whisper model"
        assert encoder_down_sampling is not None, "encoder_down_sampling must be provided for whisper model"

        def calculate_num_mel_frames(audio_length, sample_rate, window_stride, window_length=None):
            """
            Calculate the number of mel frames from an audio signal.

            Parameters:
            - audio_length (int): Total number of audio samples.
            - sample_rate (int or float): Sampling rate of the audio (samples per second).
            - window_stride (float): The time (in seconds) between successive frames.
            - window_length (float, optional): Window length in seconds. If provided, this function
            uses the standard formula: floor((N - window_length_in_samples) / hop_length) + 1.
            Otherwise, it uses the simplified calculation based on the window stride only.

            Returns:
            - int: The number of mel frames.
            """

            # when a model uses a fixed max audio length, we use the fixed max audio length instead of the audio_length
            # e.g. for whisper model, the audio_length is padded to 30 seconds
            if fixed_max_audio_length is not None:
                audio_length = fixed_max_audio_length

            hop_length_samples = int(window_stride * sample_rate)

            if window_length is None:
                num_frames = math.ceil((audio_length + 1) / hop_length_samples)
            else:
                window_length_samples = int(window_length * sample_rate)
                num_frames = math.floor((audio_length - window_length_samples) / hop_length_samples) + 1

            return num_frames

        num_mel_frames = calculate_num_mel_frames(audio_length, sample_rate, window_stride)
        encoder_seq_length = math.ceil(num_mel_frames / encoder_down_sampling)

    elif model_type == "wavlm":  # WavLM model
        # For WavLM, use the exact convolutional calculation logic
        # WavLM uses a series of convolutional layers with different kernels and strides
        conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        conv_stride = [5, 2, 2, 2, 2, 2, 2]

        # Function to calculate output length of 1D convolution
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # Start with the original input length
        input_length = audio_length

        # Apply each convolutional layer
        for kernel_size, stride in zip(conv_kernel, conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)

        # The result is the encoder sequence length
        encoder_seq_length = input_length

    elif model_type == "ast":  # audio spectrogram transformer
        assert patch_size is not None, "patch_size must be provided for ast model"
        assert frequency_stride is not None, "frequency_stride must be provided for ast model"
        assert max_spectrogram_length is not None, "max_spectrogram_length must be provided for ast model"

        # AST uses a fixed-size spectrogram and divides it into patches
        # The exact formula is based on how CNN output dimensions are calculated
        # See: https://cs231n.github.io/convolutional-networks/#conv
        frequency_out_dimension = (num_mel_bins - patch_size) // frequency_stride + 1
        time_out_dimension = (max_spectrogram_length - patch_size) // time_stride + 1

        # Number of patches is the product of these dimensions
        num_patches = frequency_out_dimension * time_out_dimension

        # Add 2 for the cls_token and distillation_token
        encoder_seq_length = num_patches + 2

        print(
            f"AST patches: freq_dim={frequency_out_dimension}, time_dim={time_out_dimension}, "
            f"patches={num_patches}, total={encoder_seq_length}"
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return max(1, int(encoder_seq_length))


def calculate_encoded_image_seq_length(
    num_one_image_tiles: int,
    model_type: str = None,
    img_width: int = None,
    img_height: int = None,
    patch_size: int = None,
    projection_downsample_factor: int = None,
):
    """
    Calculate the sequence length of the encoded image based on:
      - shape of the image
      - vision encoder's configuration (e.g. model_type, patch_size, etc.)
    """

    if model_type == "vit":
        img_seq_length = (img_width // patch_size) * (img_height // patch_size)
        encoder_seq_length = num_one_image_tiles * img_seq_length
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if projection_downsample_factor is not None:
        encoder_seq_length = encoder_seq_length // (projection_downsample_factor**2)

    return encoder_seq_length
