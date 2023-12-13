# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import torch
import torch.distributed.nn
import torch.nn as nn
from torch import distributed as dist
from torch.nn import functional as F

from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def gather_features(
    image_features, text_features, local_loss=False, gather_with_grad=False,
):
    """
    Gathers image and text features across multiple data parallel processes.

    This function is designed to work in a distributed environment where multiple
    processes are handling different portions of data. It gathers the image and text
    features from all processes to form a complete set of features across the entire dataset.
    This is crucial for calculating loss in models like CLIP, especially when the model is
    trained in a data parallel fashion.

    Parameters:
    image_features (Tensor): A tensor containing the image features.
    text_features (Tensor): A tensor containing the text features.
    local_loss (bool, optional): A flag to determine whether to use local loss calculation.
                                 Defaults to False.
    gather_with_grad (bool, optional): A flag to enable gathering with gradient computation.
                                       This is not currently working in the latest PyTorch version.
                                       Defaults to False.

    Returns:
    Tuple[Tensor, Tensor]: A tuple containing the gathered image features and text features
                           across all processes.
    """
    data_parallel_world_size = parallel_state.get_data_parallel_world_size()
    data_parallel_rank = parallel_state.get_data_parallel_rank()
    data_parallel_group = parallel_state.get_data_parallel_group()

    if gather_with_grad:
        # TODO (yuya): this is not working in current version of pytorch
        # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py#L48
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)

    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(data_parallel_world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(data_parallel_world_size)]
        dist.all_gather(gathered_image_features, image_features, group=data_parallel_group)
        dist.all_gather(gathered_text_features, text_features, group=data_parallel_group)
        # TODO (yuya): check what's this
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            # https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
            gathered_image_features[data_parallel_rank] = image_features
            gathered_text_features[data_parallel_rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    """
    A custom loss module for CLIP (Contrastive Language–Image Pretraining) training.

    This module is specifically designed for calculating the loss in CLIP model training,
    supporting features like local loss calculation, gradient gathering, and label caching
    for efficiency in a distributed training setup.

    Parameters:
    local_loss (bool, optional): If True, calculates loss locally on each data parallel process.
                                 Defaults to False.
    gather_with_grad (bool, optional): If True, gathers gradients during loss calculation.
                                       Currently not functional in the latest PyTorch version.
                                       Defaults to False.
    cache_labels (bool, optional): If True, caches labels for reuse in subsequent iterations,
                                   improving performance. Defaults to False.

    Attributes:
    world_size (int): The size of the data parallel group (number of processes).
    rank (int): The rank of the current process within the data parallel group.

    Methods:
    forward(output_tensor): Computes the loss given the model's output tensor. This involves
                            gathering features across processes, computing logits, and
                            calculating the final cross-entropy loss.
    """

    def __init__(
        self, local_loss=False, gather_with_grad=False, cache_labels=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        self.world_size = parallel_state.get_data_parallel_world_size()
        self.rank = parallel_state.get_data_parallel_rank()

    def forward(self, output_tensor):
        image_features, text_features, logit_scale = output_tensor
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features, self.local_loss, self.gather_with_grad
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        # TODO (yuya): this is not necessary; not necessary if global!
        reduced_loss = average_losses_across_data_parallel_group([total_loss])
        return total_loss, {"loss": reduced_loss}
