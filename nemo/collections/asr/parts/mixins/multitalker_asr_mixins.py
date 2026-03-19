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

from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import ListConfig

from nemo.utils import logging

__all__ = ['SpeakerKernelMixin']


def get_spk_kernel_class(spk_kernel_type, input_size, d_model, dropout=0.5):
    if spk_kernel_type == 'ff':
        return nn.Sequential(
            nn.Linear(input_size, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, input_size)
        )
    else:
        raise ValueError(f"Invalid speaker kernel type: {spk_kernel_type}")
    # TODO: conv2d and mha speaker kernel classes


class SpeakerKernelMixin(ABC):
    """
    Mixin class for models that need speaker kernel functionality.

    This mixin provides:
    - Speaker kernel initialization
    - Hook attachment for applying speaker kernels at specific encoder layers
    - Support for both active and background speaker kernels

    Models using this mixin should have the following config parameters:
    - spk_kernel_type: Type of speaker kernel ('mask', 'concat', 'sinusoidal')
    - spk_kernel_layers: List of layer indices where to apply speaker kernels
    - add_bg_spk_kernel: Whether to add background speaker kernels
    """

    def _init_speaker_kernel_config(self, cfg):
        """
        Initialize speaker kernel configuration from model config.

        Args:
            cfg: Model configuration containing speaker kernel parameters
        """
        # Speaker kernel config
        self.spk_kernel_type = cfg.get('spk_kernel_type', None)
        self.spk_kernel_layers = cfg.get('spk_kernel_layers', [0])
        self.add_bg_spk_kernel = cfg.get('add_bg_spk_kernel', True)

        # Initialize speaker target containers
        self.spk_targets = None
        if self.add_bg_spk_kernel:
            self.bg_spk_targets = None

        # Initialize speaker kernels
        self._init_spk_kernel()

    def _init_spk_kernel(self):
        """Initialize speaker kernel modules and register them to encoder layers."""
        if not isinstance(self.spk_kernel_layers, ListConfig):
            if self.spk_kernel_type is not None:
                raise ValueError(f"spk_kernel_layers must be a list, got {type(self.spk_kernel_layers)}")
            return

        # Initialize speaker kernels for each specified layer
        hidden_size = self.cfg.model_defaults.enc_hidden
        self.spk_kernels = torch.nn.ModuleDict()
        if self.add_bg_spk_kernel:
            self.bg_spk_kernels = torch.nn.ModuleDict()

        # Create kernel for each layer index
        for layer_idx in self.spk_kernel_layers:
            self.spk_kernels[str(layer_idx)] = get_spk_kernel_class(
                spk_kernel_type=self.spk_kernel_type,
                input_size=hidden_size,
                d_model=self.cfg.encoder.d_model,
                dropout=0.5,
            )
            if self.add_bg_spk_kernel:
                self.bg_spk_kernels[str(layer_idx)] = get_spk_kernel_class(
                    spk_kernel_type=self.spk_kernel_type,
                    input_size=hidden_size,
                    d_model=self.cfg.encoder.d_model,
                    dropout=0.5,
                )

        if self.spk_kernels:
            logging.info(f"Initialized speaker kernels for layers: {list(self.spk_kernels.keys())}")
            self._attach_spk_kernel_hooks()
        else:
            logging.info("No speaker kernels initialized")

    def _attach_spk_kernel_hooks(self):
        """
        Attach speaker kernel hooks to encoder layers.
        Speaker kernels will inject the speaker information into the encoder layers.
        """
        # Only attach hooks if not already attached
        if hasattr(self, 'encoder_hooks'):
            return

        self.encoder_hooks = []
        for layer_idx, kernel in self.spk_kernels.items():
            idx = int(layer_idx)

            if idx == 0:
                hook = self.encoder.layers[idx].register_forward_pre_hook(
                    self._get_spk_kernel_hook_pre_layer(layer_idx), with_kwargs=True
                )

            if idx > 0:
                # Attach a post-hook after each layer from 0 to 16.
                # Since idx > 0, we attach to layer idx-1.
                hook = self.encoder.layers[idx - 1].register_forward_hook(
                    self._get_spk_kernel_hook_post_layer(layer_idx)
                )
            self.encoder_hooks.append(hook)

    def _get_spk_kernel_hook_pre_layer(self, layer_idx: str):
        """
        Returns a hook function for applying speaker kernel transformation.

        Args:
            layer_idx (str): Index of the layer to apply the kernel

        Returns:
            callable: Hook function that applies speaker kernel
        """

        def hook_fn(module, args, kwargs):
            # Pre-hooks with with_kwargs=True must return a (new_args, new_kwargs) tuple.
            # The input tensor is passed as a keyword argument, so we find it in 'kwargs'.

            if 'x' in kwargs:
                x = kwargs['x']
                x_spk = self.spk_kernels[layer_idx](self.mask_with_speaker_targets(x, self.spk_targets))
                # residual connection
                x = x + x_spk
                if self.add_bg_spk_kernel:
                    x_bg_spk = self.bg_spk_kernels[layer_idx](
                        self.mask_with_speaker_targets(x, self.bg_spk_targets, default_value=0.0)
                    )
                    x = x + x_bg_spk
                kwargs['x'] = x
            elif args:
                # Fallback in case the call signature ever changes
                x, *rest = args
                x_spk = self.spk_kernels[layer_idx](self.mask_with_speaker_targets(x, self.spk_targets))
                # residual connection
                x = x + x_spk
                if self.add_bg_spk_kernel:
                    x_bg_spk = self.bg_spk_kernels[layer_idx](
                        self.mask_with_speaker_targets(x, self.bg_spk_targets, default_value=0.0)
                    )
                    x = x + x_bg_spk
                args = (x, *rest)

            return args, kwargs

        return hook_fn

    def _get_spk_kernel_hook_post_layer(self, layer_idx: str):
        """
        Returns a hook function for applying speaker kernel transformation.

        Args:
            layer_idx (str): Index of the layer to apply the kernel

        Returns:
            callable: Hook function that applies speaker kernel
        """

        def hook_fn(module, input, output):
            if self.spk_targets is None:
                return output

            if isinstance(output, tuple):
                x, *cache = output
            else:
                x = output

            x_spk = self.spk_kernels[layer_idx](self.mask_with_speaker_targets(x, self.spk_targets))
            # residual connection
            x = x + x_spk

            if self.add_bg_spk_kernel:
                x_bg_spk = self.bg_spk_kernels[layer_idx](
                    self.mask_with_speaker_targets(x, self.bg_spk_targets, default_value=0.0)
                )
                x = x + x_bg_spk

            if isinstance(output, tuple):
                return (x, *cache)
            return x

        return hook_fn

    def _cleanup_speaker_kernel_hooks(self):
        """
        Clean up speaker kernel hooks to prevent memory leaks.
        Can be called during model cleanup or when switching between modes.
        """
        if hasattr(self, 'encoder_hooks'):
            for hook in self.encoder_hooks:
                try:
                    hook.remove()
                except Exception as e:
                    logging.warning(f"Failed to remove speaker kernel hook: {e}")
            delattr(self, 'encoder_hooks')
            logging.info("Speaker kernel hooks cleaned up")

    def set_speaker_targets(
        self, spk_targets: Optional[torch.Tensor] = None, bg_spk_targets: Optional[torch.Tensor] = None
    ):
        """
        Set speaker targets for the model.

        Args:
            spk_targets: Main speaker targets tensor
            bg_spk_targets: Background speaker targets tensor
        """
        self.spk_targets = spk_targets
        if self.add_bg_spk_kernel:
            self.bg_spk_targets = bg_spk_targets

    def clear_speaker_targets(self):
        """Clear speaker targets."""
        self.spk_targets = None
        if self.add_bg_spk_kernel:
            self.bg_spk_targets = None

    def solve_length_mismatch(self, x: torch.Tensor, mask: torch.Tensor, default_value: float = 1.0):
        """
        Solve length mismatch between x and mask.
        """
        if mask is None:
            mask = torch.ones_like(x[:, :, 0]) * default_value
            logging.warning(
                f"Mask is None, triggering single speaker mode and assigning all ones with shape: {mask.shape}"
            )

        if mask.shape[1] < x.shape[1]:
            # pad zero to the left
            mask = torch.nn.functional.pad(mask, (x.shape[1] - mask.shape[1], 0), mode='constant', value=default_value)

        if mask.shape[1] > x.shape[1]:
            mask = mask[:, -x.shape[1] :]

        return mask

    def mask_with_speaker_targets(self, x: torch.Tensor, spk_targets: torch.Tensor, default_value: float = 1.0):
        """
        Mask the input with speaker targets.
        """
        mask = self.solve_length_mismatch(x, spk_targets, default_value)
        x_spk = x * mask.unsqueeze(2)
        return x_spk

    def concat_with_speaker_targets(self, x: torch.Tensor, spk_targets: torch.Tensor):
        """
        Concatenate the input with speaker targets.
        """
        mask = self.solve_length_mismatch(x, spk_targets)
        x_spk = x * mask.unsqueeze(2)
        return x_spk
