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
from typing import List, Tuple

import torch

# 'These functions implementation is adapted from
# https://github.com/huggingface/transformers/blob/
# 53fad641cfdb5105e2470bcf3ef17ea8e25cc300/src/transformers/models/llava_next/modeling_llava_next.py'


def get_image_sequence_length(img_h, img_w, patch_dim, add_class_token, class_token_len):
    """Get image sequence length given image size, patch size, and class token."""
    num_patches_per_dim_h = img_h // patch_dim
    num_patches_per_dim_w = img_w // patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    return num_patches + (class_token_len if add_class_token else 0)


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.
    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).
    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    import numpy as np

    if not isinstance(original_size, (list, tuple)):
        if not isinstance(original_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(original_size)} not valid ",
                "should be either list, tuple, np.ndarray or tensor",
            )
        original_size = original_size.tolist()
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


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


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.
    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.
    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    import numpy as np

    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(image_size)} not valid, "
                "should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


# These functions implementation is adapted from
# https://github.com/huggingface/transformers/blob/
# 53fad641cfdb5105e2470bcf3ef17ea8e25cc300/src/transformers/models/llava_next/modeling_llava_next.py#L655'


def pack_image_features(image_features, image_sizes, vision_feature_select_strategy, image_newline=None):
    """
    Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.
    Args:
        image_features (`List[torch.Tensor]` of length num_images,
        each of shape `(num_patches, image_length, embed_dim)`)
            List of image feature tensor, each contains all the visual feature of all patches.
        image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
            Actual image size of each images (H, W).
        vision_feature_select_strategy (`str`)
            The feature selection strategy used to select the vision feature from the vision backbone.
        image_newline (`torch.Tensor` of shape `(embed_dim)`)
            New line embedding vector.
    Returns:
        image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
        feature_lens (`List[int]`)
            token length of each image in image_features
    """
    from transformers import LlavaNextConfig

    config = LlavaNextConfig()
    new_image_features = []
    feature_lens = []

    for image_idx, image_feature in enumerate(image_features):
        if image_feature.shape[0] > 1:
            base_image_feature = image_feature[0]
            image_feature = image_feature[1:]
            height = width = config.vision_config.image_size // config.vision_config.patch_size

            if vision_feature_select_strategy == "default":
                expected_num_patches = height * width
            elif vision_feature_select_strategy == "full":
                expected_num_patches = height * width + 1
            if expected_num_patches != base_image_feature.shape[0]:
                raise ValueError("The number of patches is not consistent with the image size.")

            num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                image_sizes[image_idx],
                config.image_grid_pinpoints,
                config.vision_config.image_size,
            )
            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
            image_feature = unpad_image(image_feature, image_sizes[image_idx])
            if image_newline is not None:
                image_feature = torch.cat(
                    (
                        image_feature,
                        image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                    ),
                    dim=-1,
                )
            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
        else:
            image_feature = image_feature[0]
            if image_newline is not None:
                image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
        new_image_features.append(image_feature)
        feature_lens.append(image_feature.size(0))
    image_features = torch.cat(new_image_features, dim=0)
    feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
    return image_features, feature_lens


def get_number_of_features(
    orig_height: int,
    orig_width: int,
    height: int,
    width: int,
    image_grid_pinpoints: List[Tuple[int, int]],
    patch_size: int,
) -> int:
    """
    Calculate the number of image features after the preprocessing for images of any resolution.
    This is used to calculate the number of image tokens.
    """

    height_best_resolution, width_best_resolution = select_best_resolution(
        [orig_height, orig_width], image_grid_pinpoints
    )
    scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

    patches_height = height // patch_size
    patches_width = width // patch_size
    unpadded_features, newline_features = get_unpadded_features(
        orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
    )
    # The base patch covers the entire image (+1 for the CLS)
    # We do not add any CLS token as we assume the vision strategy is "default"
    # TODO(abhi, yash): Check if we need other vision strategies
    # base_features = patches_height * patches_width + self.num_additional_image_tokens

    base_features = patches_height * patches_width

    num_image_tokens = unpadded_features + newline_features + base_features
    return num_image_tokens


def get_unpadded_features(height, width, patches_height, patches_width, scale_height, scale_width):
    """
    Get number of features for a given image with height/width. LLaVA-NeXT is different from LLaVA
    because it divided each image into patches depending on its resolution. Therefore we need to calculate how many
    patches an image is divided into and get the number of features from that.
    """
    current_height = patches_height * scale_height
    current_width = patches_width * scale_width

    original_aspect_ratio = width / height
    current_aspect_ratio = current_width / current_height
    if original_aspect_ratio > current_aspect_ratio:
        new_height = (height * current_width) // width
        padding = (current_height - new_height) // 2
        current_height -= padding * 2
    else:
        new_width = (width * current_height) // height
        padding = (current_width - new_width) // 2
        current_width -= padding * 2

    unpadded_features = current_height * current_width
    newline_features = current_height
    return (unpadded_features, newline_features)
