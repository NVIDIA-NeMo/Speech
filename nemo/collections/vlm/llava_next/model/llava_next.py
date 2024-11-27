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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
import torch.distributed
from megatron.core.inference_params import InferenceParams
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import LlavaNextForConditionalGeneration

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import Llama2Config7B, Llama2Config13B, LlamaConfig
from nemo.collections.vlm.llava_next.model.base import LlavaNextConfig, MCoreLlavaNextModel
from nemo.collections.vlm.neva.model.base import HFCLIPVisionConfig, MultimodalProjectorConfig, NevaModel
from nemo.collections.vlm.neva.model.llava import HFLlavaImporter
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule


@dataclass
class LlavaNextConfig7B(LlavaNextConfig):
    """
    Configuration class for the 7B parameter variant of the LLaVA 16 model.

    Inherits all attributes and methods from Llava15Config7B without modification.
    """

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Llama2Config7B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: HFCLIPVisionConfig(pretrained_model_name_or_path="openai/clip-vit-large-patch14-336")
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(input_size=1024, hidden_size=4096, ffn_hidden_size=4096)
    )


@dataclass
class LlavaNextConfig13B(LlavaNextConfig):
    """
    Configuration class for the 13B parameter variant of the LLaVA 16 model.

    Inherits all attributes and methods from Llava15Config13B without modification.
    """

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Llama2Config13B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: HFCLIPVisionConfig(pretrained_model_name_or_path="openai/clip-vit-large-patch14-336")
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(input_size=1024, hidden_size=5120, ffn_hidden_size=5120)
    )


class LlavaNextModel(NevaModel):
    """
    The LLaVA Next model class, extending NevaModel.

    Attributes:
        config (LlavaNextConfig): Configuration object for the model.
        optim (Optional[OptimizerModule]): Optimizer module. Defaults to a Megatron optimizer.
        tokenizer (Optional[TokenizerSpec]): Tokenizer specification for processing text inputs.
        model_transform (Optional[Callable[[torch.nn.Module], torch.nn.Module]]):
            Optional transformation applied to the model after initialization.
    """

    def __init__(
        self,
        config: LlavaNextConfig,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[torch.nn.Module], torch.nn.Module]] = None,
    ):
        """
        Initializes the LlavaNextModel.

        Args:
            config (LlavaNextConfig): Configuration object for the model.
            optim (Optional[OptimizerModule]): optimizer module. Defaults to Megatron optimizer.
            tokenizer (Optional[TokenizerSpec]): Optional tokenizer specification for processing text inputs.
            model_transform (Optional[Callable[[torch.nn.Module], torch.nn.Module]]):
                Optional transformation function applied to the model after initialization.
        """
        super().__init__(
            config=config,
            optim=optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True)),
            tokenizer=tokenizer,
            model_transform=model_transform,
        )

    def configure_model(self) -> MCoreLlavaNextModel:
        """
        Configures the underlying model instance if it has not been initialized.

        Returns:
            MCoreLlavaNextModel: The configured model instance.
        """
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        image_sizes: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        media: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: InferenceParams = None,
        num_media_tiles: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the LLaVA Next model.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch, text_seq_len].
            position_ids (torch.Tensor): Position IDs of shape [batch, text_seq_len].
            image_sizes (torch.Tensor): Raw image sizes before tiling, of shape [batch, 2].
            loss_mask (Optional[torch.Tensor]): Text loss mask of shape [batch, text_seq_len].
            attention_mask (Optional[torch.Tensor]): Attention mask shape [batch, text_seq_len].
            media (Optional[torch.Tensor]): Input media tensor.
            labels (Optional[torch.Tensor]): Target labels of shape [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters.
            num_media_tiles (Optional[List[int]]): Number of tiles per image. Default assumes 1 tile per image.

        Returns:
            torch.Tensor: The model output. Shape depends on whether labels are provided.
            - If `labels` is provided: Loss tensor of shape [batch, seq_len].
            - If `labels` is not provided: Logits tensor of shape [batch, seq_len, vocab_size].
        """
        output_tensor = self.module(
            media=media,
            input_ids=input_ids,
            position_ids=position_ids,
            image_sizes=image_sizes,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            labels=labels,
            inference_params=inference_params,
            num_media_tiles=num_media_tiles,
        )

        return output_tensor


@io.model_importer(LlavaNextModel, "hf")
class HFLlavaNextImporter(
    HFLlavaImporter,
    io.ModelConnector["LlavaNextForConditionalGeneration", LlavaNextModel],
):
    """
    Importer class for converting HuggingFace LLaVA Next checkpoint to NeMo format.

    Inherits:
        HFLlavaImporter: Base class for HuggingFace LLaVA model importers.
        io.ModelConnector: Connector interface to handle setup, save, and load using the Lightning framework.

    Methods:
        init: Initializes a new LlavaNextModel instance.
        apply: Converts the HuggingFace model to NeMo format and saves it.
        config: Generates and returns the LlavaNextConfig for the model.
    """

    def init(self) -> LlavaNextModel:
        """
        Initializes the LlavaNextModel.

        Returns:
            LlavaNextModel: An instance of the LLaVA Next model initialized with the configuration.
        """
        return LlavaNextModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """
        Converts the HuggingFace LLaVA Next model to NeMo format and saves it to the specified path.

        Args:
            output_path (Path): The path where the converted NeMo model will be saved.

        Returns:
            Path: The output path where the NeMo model was saved.
        """

        source = LlavaNextForConditionalGeneration.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target, image_newline=True)
        print(f"Converted Llava next model to Nemo, saving to {output_path}")

        self.nemo_save(output_path, trainer)

        print(f"Converted Llava next model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    @property
    def config(self) -> LlavaNextConfig:
        """
        Generates the configuration for the LLaVA Next model based on the HuggingFace model.

        Returns:
            LlavaNextConfig: A configuration object for the LLaVA Next model.
        """
        from transformers import LlavaConfig as HFLlavaConfig

        source = HFLlavaConfig.from_pretrained(str(self))
        text_conifg = source.text_config

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        language_transformer_config = LlamaConfig(
            num_layers=text_conifg.num_hidden_layers,
            hidden_size=text_conifg.hidden_size,
            ffn_hidden_size=text_conifg.intermediate_size,
            num_attention_heads=text_conifg.num_attention_heads,
            init_method_std=text_conifg.initializer_range,
            layernorm_epsilon=text_conifg.rms_norm_eps,
            num_query_groups=text_conifg.num_key_value_heads,
            rotary_base=text_conifg.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(text_conifg.vocab_size),
            share_embeddings_and_output_weights=False,
        )
        vision_transformer_config = HFCLIPVisionConfig(
            pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"
        )
        vision_projection_config = MultimodalProjectorConfig(input_size=1024, hidden_size=4096, ffn_hidden_size=4096)

        output = LlavaNextConfig(
            language_transformer_config=language_transformer_config,
            vision_transformer_config=vision_transformer_config,
            vision_projection_config=vision_projection_config,
            vision_feature_layer=source.vision_feature_layer,
        )

        return output


__all__ = [
    "LlavaNextModel",
]
