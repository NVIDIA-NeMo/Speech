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

from collections import defaultdict


import torch
from megatron.energon import SimilarityInterleavedSample
from transformers import LlavaNextConfig as HFLlavaNextConfig

from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.sample_encoder import SimilarityInterleavedEncoder
from nemo.collections.vlm.llava_next.data.sample import LlavaNextTextSample
from nemo.collections.vlm.llava_next.model.utils import get_number_of_features
from nemo.utils import logging


class LlavaNextSimilarityInterleavedSampleEncoder(SimilarityInterleavedEncoder):
    """LlavaNextSimilarityInterleavedSampleEncoder"""

    def __init__(self, tokenizer, image_processor, multimodal_sample_config=MultiModalSampleConfig()):
        """
        Initialize the LlavaNextSimilarityInterleavedSampleEncoder, inherited from SimilarityInterleavedEncoder for
        multimodal samples
        focused on similarity interleaved data to support LLaVANeXT

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        multimodal_sample_config (MultiModalSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to MultiModalSampleConfig().
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.hf_llava_next_config = HFLlavaNextConfig()

    def process_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self.image_processor.preprocess(image, return_tensors='pt', do_rescale=False)['pixel_values'][0]
        assert isinstance(image, torch.Tensor)
        return image

    def encode(self, input_sample: SimilarityInterleavedSample, output_sample: LlavaNextTextSample):
        images = input_sample.images
        texts = input_sample.texts
        matched_text_indices = input_sample.matched_text_indices
        # Sort images according to matched_text_indices
        sorted_images_orig = [img for _, img in sorted(zip(matched_text_indices, images), key=lambda x: x[0])]
        sorted_images = [self.process_image(chunk) for chunk in sorted_images_orig]
        # sorted_indices = sorted(matched_text_indices)
        # Group images based on indices

        grouped_indices = defaultdict(int)
        for idx in matched_text_indices:
            grouped_indices[idx] += 1

        interleaved_list = []
        sorted_images_orig_i = 0

        resized_height, resized_width = (
            self.hf_llava_next_config.vision_config.image_size,
            self.hf_llava_next_config.vision_config.image_size,
        )
        # Traverse through texts and interleave images properly
        for text_idx, text in enumerate(texts):
            # If images should be placed before the text
            if not self.image_following_text and text_idx in grouped_indices:
                for _ in range(grouped_indices[text_idx]):
                    _, orig_height, orig_width = sorted_images_orig[sorted_images_orig_i].shape
                    num_image_tokens = get_number_of_features(
                        orig_height,
                        orig_width,
                        resized_height,
                        resized_width,
                        self.hf_llava_next_config.image_grid_pinpoints,
                        self.hf_llava_next_config.vision_config.patch_size,
                    )
                    interleaved_list.extend([self.image_token.token_id] * num_image_tokens)
                    sorted_images_orig_i += 1

            # Add the text
            interleaved_list.append(text)

            # If images should be placed after the text
            if self.image_following_text and text_idx in grouped_indices:
                for _ in range(grouped_indices[text_idx]):
                    _, orig_height, orig_width = sorted_images_orig[sorted_images_orig_i].shape
                    num_image_tokens = get_number_of_features(
                        orig_height,
                        orig_width,
                        resized_height,
                        resized_width,
                        self.hf_llava_next_config.image_grid_pinpoints,
                        self.hf_llava_next_config.vision_config.patch_size,
                    )
                    interleaved_list.extend([self.image_token.token_id] * num_image_tokens)
                    sorted_images_orig_i += 1

        # if last index is image token,pad with ignore placeholder
        if interleaved_list[-1] == self.image_token.token_id:
            interleaved_list.append(self.ignore_place_holder)

        # Merge consecutve text entries with a space between them
        final_sequence = []
        for item in interleaved_list:
            if final_sequence and isinstance(final_sequence[-1], str) and isinstance(item, str):
                final_sequence[-1] += " " + item
            else:
                final_sequence.append(item)
        tokenized_chunks = []
        for chunk in final_sequence:
            if chunk in [self.ignore_place_holder, self.image_token.token_id]:
                tokenized_chunks.append(chunk)
            else:
                tokenized_chunks.extend(self.tokenizer(chunk, add_special_tokens=False).input_ids)
        tokens = torch.tensor(tokenized_chunks, dtype=torch.long)
        logging.debug(
            f"Multimodal dataloader encode similarity interleaved sample tokenized chunks {tokenized_chunks}"
        )
        image_tensor = torch.concatenate(sorted_images, dim=0)

        labels = self.compute_labels(tokens)
        tokens = tokens[:-1]
        loss_mask = self.compute_loss_mask(labels)

        image_heights = [img.shape[1] for img in input_sample.images]
        image_widths = [img.shape[2] for img in input_sample.images]
        image_sizes = torch.tensor([image_heights, image_widths], dtype=torch.long).T

        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.hf_llava_next_config.image_grid_pinpoints,
                patch_size=self.hf_llava_next_config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]

        output_sample.__key__ = input_sample.__key__
        output_sample.images = image_tensor
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        output_sample.num_media_tiles = image_num_patches
        output_sample.attention_mask = torch.ones(len(tokens), dtype=torch.long)
        output_sample.image_sizes = image_sizes
        return output_sample


import numpy as np

# Borrowed from HF LLaVA Next impelemntation
# "https://github.com/huggingface/transformers/blob/"
# "53fad641cfdb5105e2470bcf3ef17ea8e25cc300/src/transformers/models/llava_next/modeling_llava_next.py#L77"


def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    """
    Calculate the number of patches after the preprocessing for images of any resolution.

    Args:
        image_size (`torch.LongTensor` or `np.ndarray` or `Tuple[int, int]`):
            The size of the input image in the format (height, width). ?
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        int: the number of patches
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.
    This is done by calculating the effective and wasted resolution for each possible resolution.
    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.
    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].
    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit
