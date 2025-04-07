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

# Copyright (C) 2024 Apple Inc. All Rights Reserved.

# IMPORTANT:  This Apple software is supplied to you by Apple
# Inc. ("Apple") in consideration of your agreement to the following
# terms, and your use, installation, modification or redistribution of
# this Apple software constitutes acceptance of these terms.  If you do
# not agree with these terms, please do not use, install, modify or
# redistribute this Apple software.

# In consideration of your agreement to abide by the following terms, and
# subject to these terms, Apple grants you a personal, non-exclusive
# license, under Apple's copyrights in this original Apple software (the
# "Apple Software"), to use, reproduce, modify and redistribute the Apple
# Software, with or without modifications, in source and/or binary forms;
# provided that if you redistribute the Apple Software in its entirety and
# without modifications, you must retain this notice and the following
# text and disclaimers in all such redistributions of the Apple Software.
# Neither the name, trademarks, service marks or logos of Apple Inc. may
# be used to endorse or promote products derived from the Apple Software
# without specific prior written permission from Apple.  Except as
# expressly stated in this notice, no other rights or licenses, express or
# implied, are granted by Apple herein, including but not limited to any
# patent rights that may be infringed by your derivative works or by other
# works in which the Apple Software may be incorporated.

# The Apple Software is provided by Apple on an "AS IS" basis.  APPLE
# MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION
# THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND
# OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS.

# IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION,
# MODIFICATION AND/OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED
# AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE),
# STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# -------------------------------------------------------------------------------
# SOFTWARE DISTRIBUTED WITH CUT CROSS ENTROPY:

# The Cut Cross Entropy software includes a number of subcomponents with separate
# copyright notices and license terms - please see the file ACKNOWLEDGEMENTS.md.
# -------------------------------------------------------------------------------

import torch
from nemo.utils.import_utils import safe_import_from


linear_cross_entropy, HAVE_LINEAR_LOSS_CE = safe_import_from(
    "cut_cross_entropy",
    "linear_cross_entropy",
)
if HAVE_LINEAR_LOSS_CE:
    # Get the module and patch triton version check
    import cut_cross_entropy.tl_utils as tl_utils

    # Create replacement functions
    def new_is_triton_greater_or_equal(version_str):
        """Check if pytorch-triton version is greater than or equal to the specified version.

        Args:
            version_str: Version string to check

        Returns:
            bool: True if pytorch-triton version >= specified version
        """
        import pkg_resources

        try:
            pytorch_triton_version = pkg_resources.get_distribution('pytorch-triton').version
            current = pkg_resources.parse_version(pytorch_triton_version)
            required = pkg_resources.parse_version(version_str)
            print(f"Current pytorch-triton version: {pytorch_triton_version}, Required triton version: {version_str}")
            return current >= required
        except pkg_resources.DistributionNotFound:
            print("pytorch-triton not found")
            return False

    def new_is_triton_greater_or_equal_3_2_0():
        """Check if pytorch-triton version is greater than or equal to 3.1.0

        Returns:
            bool: True if pytorch-triton version >= 3.1.0
        """
        return new_is_triton_greater_or_equal('3.1.0')

    # Apply the monkey patches
    tl_utils.is_triton_greater_or_equal = new_is_triton_greater_or_equal
    tl_utils.is_triton_greater_or_equal_3_2_0 = new_is_triton_greater_or_equal_3_2_0


def fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    lm_weight: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    logit_softcapping: float = 0,
    accuracy_threshold: str = "auto",
):
    """
    Compute fused linear cross entropy loss that matches PyTorch's cross_entropy behavior.

    Args:
        hidden_states: Input hidden states
        lm_weight: Weight matrix for linear transformation
        labels: Target labels
        num_items_in_batch: Number of valid tokens (where labels != ignore_index)
        ignore_index: Value to ignore in labels (default: -100)
        reduction: Reduction method ('mean' or 'sum')
        logit_softcapping: Value for softcapping logits (0 means no capping)
        accuracy_threshold: Threshold for accuracy computation
    """
    # First compute loss with sum reduction to handle normalization ourselves
    if logit_softcapping == 0:
        logit_softcapping = None

    # Compute loss with shift=False to match PyTorch behavior
    # Set filter_eps=None to avoid any token filtering
    loss = linear_cross_entropy(
        hidden_states,
        lm_weight,
        targets=labels,
        ignore_index=ignore_index,
        softcap=logit_softcapping,
        reduction="sum",  # Use sum reduction to handle normalization ourselves
        shift=False,  # Match PyTorch behavior
        filter_eps=None,  # No token filtering
    )

    # Match PyTorch's cross_entropy behavior:
    # For mean reduction, divide by number of valid tokens
    if reduction == "mean":
        if num_items_in_batch is None:
            num_items_in_batch = torch.sum(labels != ignore_index).item()
        loss = loss / num_items_in_batch
    return loss
