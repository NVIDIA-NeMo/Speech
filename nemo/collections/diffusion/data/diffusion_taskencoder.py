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

import random
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.energon import DefaultTaskEncoder, Sample, SkipSample
from megatron.energon.task_encoder.base import stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys

from nemo.lightning.io.mixin import IOMixin
from nemo.utils.sequence_packing_utils import first_fit_decreasing


@dataclass
class DiffusionSample(Sample):
    """
    Data class representing a sample for diffusion tasks.

    Attributes:
        video (torch.Tensor): Video latents (C T H W).
        t5_text_embeddings (torch.Tensor): Text embeddings (S D).
        t5_text_mask (torch.Tensor): Mask for text embeddings.
        loss_mask (torch.Tensor): Mask indicating valid positions for loss computation.
        image_size (Optional[torch.Tensor]): Tensor containing image dimensions.
        fps (Optional[torch.Tensor]): Frame rate of the video.
        num_frames (Optional[torch.Tensor]): Number of frames in the video.
        padding_mask (Optional[torch.Tensor]): Mask indicating padding positions.
        seq_len_q (Optional[torch.Tensor]): Sequence length for query embeddings.
        seq_len_kv (Optional[torch.Tensor]): Sequence length for key/value embeddings.
        pos_ids (Optional[torch.Tensor]): Positional IDs.
        latent_shape (Optional[torch.Tensor]): Shape of the latent tensor.
    """

    video: torch.Tensor  # video latents (C T H W)
    t5_text_embeddings: torch.Tensor  # (S D)
    t5_text_mask: torch.Tensor  # 1
    loss_mask: torch.Tensor
    image_size: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None
    num_frames: Optional[torch.Tensor] = None
    padding_mask: Optional[torch.Tensor] = None
    seq_len_q: Optional[torch.Tensor] = None
    seq_len_kv: Optional[torch.Tensor] = None
    pos_ids: Optional[torch.Tensor] = None
    latent_shape: Optional[torch.Tensor] = None

    def to_dict(self) -> dict:
        """Converts the sample to a dictionary."""
        return dict(
            video=self.video,
            t5_text_embeddings=self.t5_text_embeddings,
            t5_text_mask=self.t5_text_mask,
            loss_mask=self.loss_mask,
            image_size=self.image_size,
            fps=self.fps,
            num_frames=self.num_frames,
            padding_mask=self.padding_mask,
            seq_len_q=self.seq_len_q,
            seq_len_kv=self.seq_len_kv,
            pos_ids=self.pos_ids,
            latent_shape=self.latent_shape,
        )

    def __add__(self, other: Any) -> int:
        """Adds the sequence length of this sample with another sample or integer."""
        if isinstance(other, DiffusionSample):
            # Combine the values of the two instances
            return self.seq_len_q.item() + other.seq_len_q.item()
        elif isinstance(other, int):
            # Add an integer to the value
            return self.seq_len_q.item() + other
        raise NotImplementedError

    def __radd__(self, other: Any) -> int:
        """Handles reverse addition for summing with integers."""
        # This is called if sum or other operations start with a non-DiffusionSample object.
        # e.g., sum([DiffusionSample(1), DiffusionSample(2)]) -> the 0 + DiffusionSample(1) calls __radd__.
        if isinstance(other, int):
            return self.seq_len_q.item() + other
        raise NotImplementedError

    def __lt__(self, other: Any) -> bool:
        """Compares this sample's sequence length with another sample or integer."""
        if isinstance(other, DiffusionSample):
            return self.seq_len_q.item() < other.seq_len_q.item()
        elif isinstance(other, int):
            return self.seq_len_q.item() < other
        raise NotImplementedError


def cook(sample: dict) -> dict:
    """
    Processes a raw sample dictionary from energon dataset and returns a new dictionary with specific keys.

    Args:
        sample (dict): The input dictionary containing the raw sample data.

    Returns:
        dict: A new dictionary containing the processed sample data with the following keys:
            - All keys from the result of `basic_sample_keys(sample)`
            - 'json': The contains meta data like resolution, aspect ratio, fps, etc.
            - 'pth': contains video latent tensor
            - 'pickle': contains text embeddings
    """
    return dict(
        **basic_sample_keys(sample),
        json=sample['.json'],
        pth=sample['.pth'],
        pickle=sample['.pickle'],
    )


class BasicDiffusionTaskEncoder(DefaultTaskEncoder, IOMixin):
    """
    BasicDiffusionTaskEncoder is a class that encodes image/video samples for diffusion tasks.
    Attributes:
        cookers (list): A list of Cooker objects used for processing.
        max_frames (int, optional): The maximum number of frames to consider from the video. Defaults to None.
        text_embedding_padding_size (int): The padding size for text embeddings. Defaults to 512.
    Methods:
        __init__(*args, max_frames=None, text_embedding_padding_size=512, **kwargs):
            Initializes the BasicDiffusionTaskEncoder with optional maximum frames and text embedding padding size.
        encode_sample(sample: dict) -> dict:
            Encodes a given sample dictionary containing video and text data.
            Args:
                sample (dict): A dictionary containing 'pth' for video latent and 'json' for additional info.
            Returns:
                dict: A dictionary containing encoded video, text embeddings, text mask, and loss mask.
            Raises:
                SkipSample: If the video latent contains NaNs, Infs, or is not divisible by the tensor parallel size.
    """

    cookers = [
        Cooker(cook),
    ]

    def __init__(
        self,
        *args,
        max_frames: int = None,
        text_embedding_padding_size: int = 512,
        seq_length: int = None,
        max_seq_length: int = None,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        aesthetic_score: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_frames = max_frames
        self.text_embedding_padding_size = text_embedding_padding_size
        self.seq_length = seq_length
        self.max_seq_length = max_seq_length
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.aesthetic_score = aesthetic_score

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: dict) -> dict:
        """
        Encodes video / text sample.
        """
        video_latent = sample['pth']

        if torch.isnan(video_latent).any() or torch.isinf(video_latent).any():
            raise SkipSample()
        if torch.max(torch.abs(video_latent)) > 1e3:
            raise SkipSample()

        info = sample['json']
        if info['aesthetic_score'] < self.aesthetic_score:
            raise SkipSample()

        C, T, H, W = video_latent.shape
        seq_len = (
            video_latent.shape[-1]
            * video_latent.shape[-2]
            * video_latent.shape[-3]
            // self.patch_spatial**2
            // self.patch_temporal
        )
        is_image = T == 1

        if self.seq_length is not None and seq_len > self.seq_length:
            raise SkipSample()
        if self.max_seq_length is not None and seq_len > self.max_seq_length:
            raise SkipSample()

        if self.max_frames is not None:
            video_latent = video_latent[:, : self.max_frames, :, :]

        video_latent = rearrange(
            video_latent,
            'C (T pt) (H ph) (W pw) -> (T H W) (ph pw pt C)',
            ph=self.patch_spatial,
            pw=self.patch_spatial,
            pt=self.patch_temporal,
        )

        if is_image:
            t5_text_embeddings = torch.from_numpy(sample['pickle']).to(torch.bfloat16)
        else:
            t5_text_embeddings = torch.from_numpy(sample['pickle'][0]).to(torch.bfloat16)
        t5_text_embeddings_seq_length = t5_text_embeddings.shape[0]

        if t5_text_embeddings_seq_length > self.text_embedding_padding_size:
            t5_text_embeddings = t5_text_embeddings[: self.text_embedding_padding_size]
        else:
            t5_text_embeddings = F.pad(
                t5_text_embeddings,
                (
                    0,
                    0,
                    0,
                    self.text_embedding_padding_size - t5_text_embeddings_seq_length,
                ),
            )
        t5_text_mask = torch.ones(t5_text_embeddings_seq_length, dtype=torch.bfloat16)

        if is_image:
            h, w = info['image_height'], info['image_width']
            fps = torch.tensor([30] * 1, dtype=torch.bfloat16)
            num_frames = torch.tensor([1] * 1, dtype=torch.bfloat16)
        else:
            h, w = info['height'], info['width']
            fps = torch.tensor([info['framerate']] * 1, dtype=torch.bfloat16)
            num_frames = torch.tensor([info['num_frames']] * 1, dtype=torch.bfloat16)
        image_size = torch.tensor([[h, w, h, w]] * 1, dtype=torch.bfloat16)

        pos_ids = rearrange(
            pos_id_3d.get_pos_id_3d(t=T // self.patch_temporal, h=H // self.patch_spatial, w=W // self.patch_spatial),
            'T H W d -> (T H W) d',
        )

        if self.seq_length is not None and self.max_seq_length is None:
            pos_ids = F.pad(pos_ids, (0, 0, 0, self.seq_length - seq_len))
            loss_mask = torch.zeros(self.seq_length, dtype=torch.bfloat16)
            loss_mask[:seq_len] = 1
            video_latent = F.pad(video_latent, (0, 0, 0, self.seq_length - seq_len))
        else:
            loss_mask = torch.ones(seq_len, dtype=torch.bfloat16)

        return DiffusionSample(
            __key__=sample['__key__'],
            __restore_key__=sample['__restore_key__'],
            __subflavor__=None,
            __subflavors__=sample['__subflavors__'],
            video=video_latent,
            t5_text_embeddings=t5_text_embeddings,
            t5_text_mask=t5_text_mask,
            image_size=image_size,
            fps=fps,
            num_frames=num_frames,
            loss_mask=loss_mask,
            seq_len_q=torch.tensor(seq_len, dtype=torch.int32),
            seq_len_kv=torch.tensor(self.text_embedding_padding_size, dtype=torch.int32),
            pos_ids=pos_ids,
            latent_shape=torch.tensor([C, T, H, W], dtype=torch.int32),
        )

    def select_samples_to_pack(self, samples: List[DiffusionSample]) -> List[List[DiffusionSample]]:
        """
        Selects sequences to pack for mixed image-video training.
        """
        results = first_fit_decreasing(samples, self.max_seq_length)
        random.shuffle(results)
        return results

    @stateless
    def pack_selected_samples(self, samples: List[DiffusionSample]) -> DiffusionSample:
        """Construct a new Diffusion sample by concatenating the sequences."""

        def stack(attr):
            return torch.stack([getattr(sample, attr) for sample in samples], dim=0)

        def cat(attr):
            return torch.cat([getattr(sample, attr) for sample in samples], dim=0)

        video = concat_pad([i.video for i in samples], self.max_seq_length)
        loss_mask = concat_pad([i.loss_mask for i in samples], self.max_seq_length)
        pos_ids = concat_pad([i.pos_ids for i in samples], self.max_seq_length)

        return DiffusionSample(
            __key__=",".join([s.__key__ for s in samples]),
            __restore_key__=(),  # Will be set by energon based on `samples`
            __subflavor__=None,
            __subflavors__=samples[0].__subflavors__,
            video=video,
            t5_text_embeddings=cat('t5_text_embeddings'),
            t5_text_mask=cat('t5_text_mask'),
            # image_size=stack('image_size'),
            # fps=stack('fps'),
            # num_frames=stack('num_frames'),
            loss_mask=loss_mask,
            seq_len_q=stack('seq_len_q'),
            seq_len_kv=stack('seq_len_kv'),
            pos_ids=pos_ids,
            latent_shape=stack('latent_shape'),
        )

    @stateless
    def batch(self, samples: List[DiffusionSample]) -> dict:
        """Return dictionary with data for batch."""
        if self.max_seq_length is None:
            # no packing
            return super().batch(samples).to_dict()

        # packing
        sample = samples[0]
        return dict(
            video=sample.video.unsqueeze_(0),
            t5_text_embeddings=sample.t5_text_embeddings.unsqueeze_(0),
            t5_text_mask=sample.t5_text_mask.unsqueeze_(0),
            loss_mask=sample.loss_mask.unsqueeze_(0),
            # image_size=sample.image_size,
            # fps=sample.fps,
            # num_frames=sample.num_frames,
            # padding_mask=sample.padding_mask.unsqueeze_(0),
            seq_len_q=sample.seq_len_q,
            seq_len_kv=sample.seq_len_kv,
            pos_ids=sample.pos_ids.unsqueeze_(0),
            latent_shape=sample.latent_shape,
        )


class PosID3D:
    """
    Generates 3D positional IDs for video data.

    Attributes:
        max_t (int): Maximum number of time frames.
        max_h (int): Maximum height dimension.
        max_w (int): Maximum width dimension.
    """

    def __init__(self, *, max_t=32, max_h=128, max_w=128):
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.generate_pos_id()

    def generate_pos_id(self):
        """Generates a grid of positional IDs based on max_t, max_h, and max_w."""
        self.grid = torch.stack(
            torch.meshgrid(
                torch.arange(self.max_t, device='cpu'),
                torch.arange(self.max_h, device='cpu'),
                torch.arange(self.max_w, device='cpu'),
            ),
            dim=-1,
        )

    def get_pos_id_3d(self, *, t, h, w):
        """Retrieves positional IDs for specified dimensions."""
        if t > self.max_t or h > self.max_h or w > self.max_w:
            self.max_t = max(self.max_t, t)
            self.max_h = max(self.max_h, h)
            self.max_w = max(self.max_w, w)
            self.generate_pos_id()
        return self.grid[:t, :h, :w]


def pad_divisible(x, padding_value=0):
    """
    Pads the input tensor to make its size divisible by a specified value.

    Args:
        x (torch.Tensor): Input tensor.
        padding_value (int): The value to make the tensor size divisible by.

    Returns:
        torch.Tensor: Padded tensor.
    """
    if padding_value == 0:
        return x
    # Get the size of the first dimension
    n = x.size(0)

    # Compute the padding needed to make the first dimension divisible by 16
    padding_needed = (padding_value - n % padding_value) % padding_value

    if padding_needed <= 0:
        return x

    # Create a new shape with the padded first dimension
    new_shape = list(x.shape)
    new_shape[0] += padding_needed

    # Create a new tensor filled with zeros
    x_padded = torch.zeros(new_shape, dtype=x.dtype, device=x.device)

    # Assign the original tensor to the beginning of the new tensor
    x_padded[:n] = x
    return x_padded


def concat_pad(tensor_list, max_seq_length):
    """
    Efficiently concatenates a list of tensors along the first dimension and pads with zeros
    to reach max_seq_length.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to concatenate and pad.
        max_seq_length (int): The desired size of the first dimension of the output tensor.

    Returns:
        torch.Tensor: A tensor of shape [max_seq_length, ...], where ... represents the remaining dimensions.
    """
    import torch

    # Get common properties from the first tensor
    other_shape = tensor_list[0].shape[1:]
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device

    # Initialize the result tensor with zeros
    result = torch.zeros((max_seq_length, *other_shape), dtype=dtype, device=device)

    current_index = 0
    for tensor in tensor_list:
        length = tensor.shape[0]
        # Directly assign the tensor to the result tensor without checks
        result[current_index : current_index + length] = tensor
        current_index += length

    return result


pos_id_3d = PosID3D()


def cook_raw_images(sample: dict) -> dict:
    """
    Processes a raw sample dictionary from energon dataset and returns a new dictionary with specific keys.

    Args:
        sample (dict): The input dictionary containing the raw sample data.

    Returns:
        dict: A new dictionary containing the processed sample data with the following keys:
            - All keys from the result of `basic_sample_keys(sample)`
            - 'jpg': original images
            - 'png': contains control images
            - 'txt': contains raw text
    """
    return dict(
        **basic_sample_keys(sample),
        images=sample['jpg'],
        hint=sample['png'],
        txt=sample['txt'],
    )


class RawImageDiffusionTaskEncoder(DefaultTaskEncoder, IOMixin):
    '''
    Dummy task encoder takes raw image input on CrudeDataset.
    '''

    cookers = [
        # Cooker(cook),
        Cooker(cook_raw_images),
    ]
