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
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/autoaugment.py

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

_MAX_LEVEL = 10  # Maximum integer strength of an augmentation, if applicable.


class ImageNetPolicy:
    """Definition of an ImageNetPolicy.

    Implements a fixed AutoAugment data augmentation policy targeted at
    ImageNet training by randomly applying at runtime one of the 25 pre-defined
    data augmentation sub-policies provided in Reference [1].

    Usage example as a Pytorch Transform:
    >>> transform=transforms.Compose([transforms.Resize(256),
    >>>                               ImageNetPolicy(),
    >>>                               transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        """Initialize an ImageNetPolicy.

        Args:
            fillcolor (tuple): RGB color components of the color to be used for
            filling when needed (default: (128, 128, 128), which
            corresponds to gray).
        """
        # Instantiate a list of sub-policies.
        # Each entry of the list is a SubPolicy which consists of
        # two augmentation operations,
        # each of those parametrized as operation, probability, magnitude.
        # Those two operations are applied sequentially on the image upon call.
        self.policies = [
            SubPolicy("posterize", 0.4, 8, "rotate", 0.6, 9, fillcolor),
            SubPolicy("solarize", 0.6, 5, "autocontrast", 0.6, 5, fillcolor),
            SubPolicy("equalize", 0.8, 8, "equalize", 0.6, 3, fillcolor),
            SubPolicy("posterize", 0.6, 7, "posterize", 0.6, 6, fillcolor),
            SubPolicy("equalize", 0.4, 7, "solarize", 0.2, 4, fillcolor),
            SubPolicy("equalize", 0.4, 4, "rotate", 0.8, 8, fillcolor),
            SubPolicy("solarize", 0.6, 3, "equalize", 0.6, 7, fillcolor),
            SubPolicy("posterize", 0.8, 5, "equalize", 1.0, 2, fillcolor),
            SubPolicy("rotate", 0.2, 3, "solarize", 0.6, 8, fillcolor),
            SubPolicy("equalize", 0.6, 8, "posterize", 0.4, 6, fillcolor),
            SubPolicy("rotate", 0.8, 8, "color", 0.4, 0, fillcolor),
            SubPolicy("rotate", 0.4, 9, "equalize", 0.6, 2, fillcolor),
            SubPolicy("equalize", 0.0, 7, "equalize", 0.8, 8, fillcolor),
            SubPolicy("invert", 0.6, 4, "equalize", 1.0, 8, fillcolor),
            SubPolicy("color", 0.6, 4, "contrast", 1.0, 8, fillcolor),
            SubPolicy("rotate", 0.8, 8, "color", 1.0, 2, fillcolor),
            SubPolicy("color", 0.8, 8, "solarize", 0.8, 7, fillcolor),
            SubPolicy("sharpness", 0.4, 7, "invert", 0.6, 8, fillcolor),
            SubPolicy("shearX", 0.6, 5, "equalize", 1.0, 9, fillcolor),
            SubPolicy("color", 0.4, 0, "equalize", 0.6, 3, fillcolor),
            SubPolicy("equalize", 0.4, 7, "solarize", 0.2, 4, fillcolor),
            SubPolicy("solarize", 0.6, 5, "autocontrast", 0.6, 5, fillcolor),
            SubPolicy("invert", 0.6, 4, "equalize", 1.0, 8, fillcolor),
            SubPolicy("color", 0.6, 4, "contrast", 1.0, 8, fillcolor),
            SubPolicy("equalize", 0.8, 8, "equalize", 0.6, 3, fillcolor),
        ]

    def __call__(self, img):
        """Define call method for ImageNetPolicy class."""
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        """Define repr method for ImageNetPolicy class."""
        return "ImageNetPolicy"


class SubPolicy:
    """Definition of a SubPolicy.

    A SubPolicy consists of two augmentation operations,
    each of those parametrized as operation, probability, magnitude.
    The two operations are applied sequentially on the image upon call.
    """

    def __init__(
        self, operation1, probability1, magnitude_idx1, operation2, probability2, magnitude_idx2, fillcolor,
    ):
        """Initialize a SubPolicy.

        Args:
            operation1 (str): Key specifying the first augmentation operation.
            There are fourteen key values altogether (see supported_ops below
            listing supported operations). probability1 (float): Probability
            within [0., 1.] of applying the first augmentation operation.
            magnitude_idx1 (int): Integer specifiying the strength of the first
            operation as an index further used to derive the magnitude from a
            range of possible values.
            operation2 (str): Key specifying the second augmentation operation.
            probability2 (float): Probability within [0., 1.] of applying the
            second augmentation operation.
            magnitude_idx2 (int): Integer specifiying the strength of the
            second operation as an index further used to derive the magnitude
            from a range of possible values.
            fillcolor (tuple): RGB color components of the color to be used for
            filling.
        Returns:
        """
        # List of supported operations for operation1 and operation2.
        supported_ops = [
            "shearX",
            "shearY",
            "translateX",
            "translateY",
            "rotate",
            "color",
            "posterize",
            "solarize",
            "contrast",
            "sharpness",
            "brightness",
            "autocontrast",
            "equalize",
            "invert",
        ]
        assert (operation1 in supported_ops) and (
            operation2 in supported_ops
        ), "SubPolicy:one of oper1 or oper2 refers to an unsupported operation."

        assert (
            0.0 <= probability1 <= 1.0 and 0.0 <= probability2 <= 1.0
        ), "SubPolicy: prob1 and prob2 should be within [0., 1.]."

        assert (
            isinstance(magnitude_idx1, int) and 0 <= magnitude_idx1 <= 10
        ), "SubPolicy: idx1 should be specified as an integer within [0, 10]."

        assert (
            isinstance(magnitude_idx2, int) and 0 <= magnitude_idx2 <= 10
        ), "SubPolicy: idx2 should be specified as an integer within [0, 10]."

        # Define a dictionary where each key refers to a specific type of
        # augmentation and the corresponding value is a range of ten possible
        # magnitude values for that augmentation.
        num_levels = _MAX_LEVEL + 1
        ranges = {
            "shearX": np.linspace(0, 0.3, num_levels),
            "shearY": np.linspace(0, 0.3, num_levels),
            "translateX": np.linspace(0, 150 / 331, num_levels),
            "translateY": np.linspace(0, 150 / 331, num_levels),
            "rotate": np.linspace(0, 30, num_levels),
            "color": np.linspace(0.0, 0.9, num_levels),
            "posterize": np.round(np.linspace(8, 4, num_levels), 0).astype(np.int64),
            "solarize": np.linspace(256, 0, num_levels),  # range [0, 256]
            "contrast": np.linspace(0.0, 0.9, num_levels),
            "sharpness": np.linspace(0.0, 0.9, num_levels),
            "brightness": np.linspace(0.0, 0.9, num_levels),
            "autocontrast": [0] * num_levels,  # This augmentation doesn't use magnitude parameter.
            "equalize": [0] * num_levels,  # This augmentation doesn't use magnitude parameter.
            "invert": [0] * num_levels,  # This augmentation doesn't use magnitude parameter.
        }

        def rotate_with_fill(img, magnitude):
            """Define rotation transformation with fill.

            The input image is first rotated, then it is blended together with
            a gray mask of the same size. Note that fillcolor as defined
            elsewhere in this module doesn't apply here.

            Args:
                magnitude (float): rotation angle in degrees.
            Returns:
                rotated_filled (PIL Image): rotated image with gray filling for
                disoccluded areas unveiled by the rotation.
            """
            rotated = img.convert("RGBA").rotate(magnitude)
            rotated_filled = Image.composite(rotated, Image.new("RGBA", rotated.size, (128,) * 4), rotated)
            return rotated_filled.convert(img.mode)

        # Define a dictionary of augmentation functions where each key refers
        # to a specific type of augmentation and the corresponding value defines
        # the augmentation itself using a lambda function.
        # pylint: disable=unnecessary-lambda
        func_dict = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0,),
                fillcolor=fillcolor,
            ),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1]),),
                fillcolor=fillcolor,
            ),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
        }

        # Store probability, function and magnitude of the first augmentation
        # for the sub-policy.
        self.probability1 = probability1
        self.operation1 = func_dict[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]

        # Store probability, function and magnitude of the second augmentation
        # for the sub-policy.
        self.probability2 = probability2
        self.operation2 = func_dict[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        """Define call method for SubPolicy class."""
        # Randomly apply operation 1.
        if random.random() < self.probability1:
            img = self.operation1(img, self.magnitude1)

        # Randomly apply operation 2.
        if random.random() < self.probability2:
            img = self.operation2(img, self.magnitude2)

        return img
