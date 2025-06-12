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


import os
import re
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch.distributed
import torch.nn.functional as F
from megatron.core.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector as MCoreMultimodalProjector
from megatron.core.tensor_parallel.layers import ColumnParallelLinear

try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )
except ImportError:
    from nemo.utils import logging

    # These Defaults are needed to make sure the code compiles
    TEColumnParallelLinear = None
    TENorm = None
    TERowParallelLinear = None
    logging.warning(
        "Failed to import Transformer Engine dependencies. "
        "`from megatron.core.transformer.custom_layers.transformer_engine import *`"
        "If using NeMo Run, this is expected. Otherwise, please verify the Transformer Engine installation."
    )

from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import CLIPVisionConfig, CLIPVisionModel

from nemo.lightning import io


def set_input_tensor(self, tensor):
    """Sets input tensor func place holder"""
    pass


def get_image_sequence_length(img_h, img_w, patch_dim, add_class_token, class_token_len):
    """Get image sequence length given image size, patch size, and class token."""
    num_patches_per_dim_h = img_h // patch_dim
    num_patches_per_dim_w = img_w // patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    return num_patches + (class_token_len if add_class_token else 0)


class DownSampleBlock(torch.nn.Module):
    """Downsample block following the ViLA-VLM paper."""

    # pylint: disable=line-too-long
    # Implement from https://github.com/NVlabs/VILA/blob/3522eef015e48d73cf83fc2b949cd464dab1ba3c/llava/model/multimodal_projector/base_projector.py#L48
    # small adjusmtnet with x.transpose(0, 1)

    def forward(self, x):
        """Downsample the input tensor."""
        x = x.transpose(0, 1)
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x):
        """Flatten the input tensor and make it square."""
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.contiguous()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


@dataclass
class MultimodalProjectorConfig(TransformerConfig, io.IOMixin):
    """
    For MLP, fc1 in shape of input_size, ffn_hidden_size, fc2 in shape of ffn_hidden_size, hidden_size
    """

    projector_type: str = "mlp2x_gelu"
    layer_spec: Optional[MLPSubmodules] = None
    input_size: Optional[int] = 1024
    hidden_size: int = 1024
    ffn_hidden_size: int = 1024
    activation_func: Callable = F.gelu
    bias: bool = True
    bias_activation_fusion: bool = True
    num_layers: int = 1  # placeholder, NOT used!
    num_attention_heads: int = 8  # placeholder, NOT used!

    def configure_model(self) -> "MCoreMultimodalProjector":
        # pylint: disable=C0115,C0116
        if self.projector_type.startswith("mcore") and self.layer_spec is None:
            self.add_bias_linear = self.bias
            if self.projector_type == "mcore_mlp":
                self.projector_type = "mlp"  # strip "mcore_" for mcore init
                self.layer_spec = ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TEColumnParallelLinear,
                        linear_fc2=TERowParallelLinear,
                    ),
                )
                self.layer_spec = self.layer_spec.submodules
            elif self.projector_type == "mcore_affine":
                self.projector_type = "affine"  # strip "mcore_" for mcore init
                self.layer_spec = MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=None)
            else:
                raise NotImplementedError(f"Not supported projector type `{self.projector_type}`")

            return MCoreMultimodalProjector(
                self,
                self.layer_spec,
                projector_type=self.projector_type,
                input_size=self.input_size,
            )

        # if using vila's downsample + mlp projector
        if self.projector_type == "vila_downsample_mlp":
            model = torch.nn.Sequential(
                DownSampleBlock(),
                torch.nn.LayerNorm(self.input_size * 4, dtype=self.params_dtype),
                torch.nn.Linear(self.input_size * 4, self.hidden_size, bias=True, dtype=self.params_dtype),
                torch.nn.GELU(),
                torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True, dtype=self.params_dtype),
            )
            from types import MethodType

            model.set_input_tensor = MethodType(set_input_tensor, model)
            return model

        # e.g. "mlp2x_gelu"
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [torch.nn.Linear(self.input_size, self.ffn_hidden_size, bias=True, dtype=self.params_dtype)]
            for _ in range(1, mlp_depth):
                modules.append(torch.nn.GELU())
                modules.append(
                    torch.nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=True, dtype=self.params_dtype)
                )
            model = torch.nn.Sequential(*modules)
            from types import MethodType

            model.set_input_tensor = MethodType(set_input_tensor, model)
        else:
            raise NotImplementedError(f"Not supported projector type `{self.projector_type}`")

        return model


@dataclass
class HFCLIPVisionConfig(CLIPVisionConfig, io.IOMixin):
    """
    https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/clip/configuration_clip.py#L261
    """

    hidden_size: int = 1024
    add_class_token: bool = False
    class_token_len: int = 1
    num_image_embeddings_per_tile: Optional[int] = None
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None

    def __post_init__(self, *args, **kwargs) -> None:
        # pylint: disable=C0115,C0116
        CLIPVisionConfig.__init__(self, *args, **kwargs, hidden_size=self.hidden_size)
        if self.pretrained_model_name_or_path is not None:
            config = CLIPVisionConfig.from_pretrained(self.pretrained_model_name_or_path)
            for key, value in config.to_dict().items():
                setattr(self, key, value)
        self.num_image_embeddings_per_tile = get_image_sequence_length(
            img_h=self.image_size,
            img_w=self.image_size,
            patch_dim=self.patch_size,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
        )

    def configure_model(self) -> "CLIPVisionModel":
        # pylint: disable=C0115,C0116
        # Monkey patch the method to the vision encoder
        CLIPVisionModel.set_input_tensor = set_input_tensor

        if self.pretrained_model_name_or_path is None:
            model = CLIPVisionModel(self)
        else:
            model = CLIPVisionModel.from_pretrained(self.pretrained_model_name_or_path)

        # add attribute "tensor_parallel_grad_reduce" to the model for TP grad all-reduce
        model.tensor_parallel_grad_reduce = True

        return model


@dataclass
class CLIPViTConfig(TransformerConfig, io.IOMixin):
    """MCore CLIP ViT Config"""

    ln_pre_impl: Union[ModuleSpec, type] = TENorm
    ln_post_impl: Union[ModuleSpec, type] = TENorm
    add_class_token: bool = True
    class_token_len: int = 1
    patch_dim: int = 14
    img_h: int = 336
    img_w: int = 336
    vision_model_type: str = "clip"  # ["clip", "siglip", "internvit"]
    num_image_embeddings_per_tile: Optional[int] = None
    transformer_layer_spec: ModuleSpec = None

    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!

    def __post_init__(self):
        # pylint: disable=C0115,C0116
        if self.vision_model_type == "siglip":
            self.add_class_token = False
            self.class_token_len = 0
        self.num_image_embeddings_per_tile = get_image_sequence_length(
            img_h=self.img_h,
            img_w=self.img_w,
            patch_dim=self.patch_dim,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
        )

    def configure_model(self) -> "BaseCLIPViTModel":
        # pylint: disable=C0115,C0116
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            from nemo.collections.vlm.layer_specs import get_layer_spec_te

            transformer_layer_spec = get_layer_spec_te(is_vit=True)
        return BaseCLIPViTModel(
            self,
            transformer_layer_spec,
            ln_pre_impl=self.ln_pre_impl,
            ln_post_impl=self.ln_post_impl,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            img_h=self.img_h,
            img_w=self.img_w,
            model_subtype=self.vision_model_type,
        )


class BaseCLIPViTModel(MCoreCLIPViTModel):
    """CLIP ViT vision model."""

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, num_unused_layers: int = 0
    ) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        if num_unused_layers > 0:
            unused_layers = self.decoder.layers[-num_unused_layers:]
            self.decoder.layers = self.decoder.layers[:-num_unused_layers]
            x = super().forward(x, attention_mask)
            self.decoder.layers.extend(unused_layers)
            return x

        return super().forward(x, attention_mask)
