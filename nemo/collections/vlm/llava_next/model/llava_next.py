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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
import torch.distributed
from megatron.core.inference_params import InferenceParams
from megatron.core.optimizer import OptimizerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import CLIPVisionConfig
from transformers import LlavaConfig as HFLlavaConfig
from transformers import LlavaNextForConditionalGeneration

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import Llama2Config7B, Llama2Config13B, LlamaConfig
from nemo.collections.vlm.llava_next.model.base import LlavaNextConfig, MCoreLlavaNextModel
from nemo.collections.vlm.neva.model.base import NevaModel
from nemo.collections.vlm.neva.model.llava import HFLlavaImporter
from nemo.collections.vlm.vision.base import HFCLIPVisionConfig, MultimodalProjectorConfig
from nemo.lightning import io, teardown
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import logging


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
        packed_seq_params: Optional[PackedSeqParams] = None,
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
            packed_seq_params=packed_seq_params,
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


@io.model_exporter(LlavaNextModel, "hf")
class HFLlavaNextExporter(io.ModelConnector[LlavaNextModel, "LlavaNextForConditionalGeneration"]):
    """
    Exporter class for converting NeMo LLaVA Next model to HuggingFace format.

    Inherits:
        io.ModelConnector: Connector interface to handle setup, save, and load using the Lightning framework.

    Methods:
        init: Initializes a new HuggingFace LLaVA Next model instance.
        apply: Converts the NeMo model to HuggingFace format and saves it.
        convert_state: Maps and transforms the state dictionary from NeMo to HuggingFace format.
        config: Generates and returns the HuggingFace LLaVA config for the model.
    """

    def init(self) -> "LlavaNextForConditionalGeneration":
        """
        Initializes a HuggingFace LlavaNextForConditionalGeneration model.

        Args:
            dtype: The data type to use for the model (default: torch.bfloat16)

        Returns:
            LlavaNextForConditionalGeneration: A HuggingFace LLaVA Next model initialized with the configuration.
        """
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return LlavaNextForConditionalGeneration(self.config)

    def apply(self, output_path: Path) -> Path:
        """
        Converts the NeMo LLaVA Next model to HuggingFace format and saves it to the specified path.

        Args:
            output_path (Path): The path where the converted HuggingFace model will be saved.

        Returns:
            Path: The output path where the HuggingFace model was saved.
        """
        source, _ = self.nemo_load(str(self))
        target = self.init()
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        try:
            self.tokenizer.tokenizer.save_pretrained(output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        return output_path

    def convert_state(self, source, target):
        # pylint: disable=C0115,C0116,line-too-long
        """
        Maps and transforms the state dictionary from NeMo to HuggingFace format.

        Args:
            source: The source NeMo model.
            target: The target HuggingFace model.

        Returns:
            The target HuggingFace model with the converted state.
        """
        # Define the state mapping from NeMo to HuggingFace
        mapping = {
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
        }

        # Map vision projection components
        if "vision_projection.encoder.linear_fc1.weight" in source.module.state_dict().keys():
            mapping.update(
                {
                    "vision_projection.encoder.linear_fc1.weight": "model.multi_modal_projector.linear_1.weight",
                    "vision_projection.encoder.linear_fc1.bias": "model.multi_modal_projector.linear_1.bias",
                    "vision_projection.encoder.linear_fc2.weight": "model.multi_modal_projector.linear_2.weight",
                    "vision_projection.encoder.linear_fc2.bias": "model.multi_modal_projector.linear_2.bias",
                }
            )
        elif "vision_projection.0.weight" in source.module.state_dict().keys():
            mapping.update(
                {
                    "vision_projection.0.weight": "model.multi_modal_projector.linear_1.weight",
                    "vision_projection.0.bias": "model.multi_modal_projector.linear_1.bias",
                    "vision_projection.2.weight": "model.multi_modal_projector.linear_2.weight",
                    "vision_projection.2.bias": "model.multi_modal_projector.linear_2.bias",
                }
            )

        # Check for image_newline and add it to mapping if it exists
        if "image_newline" in source.module.state_dict().keys():
            mapping.update({"image_newline": "model.image_newline"})

        # Map vision model components
        if "vision_model.vision_model.embeddings.class_embedding" in source.module.state_dict().keys():
            mapping.update(
                {
                    "vision_model.vision_model.**": "model.vision_tower.vision_model.**",
                }
            )
        elif "vision_model.class_token" in source.module.state_dict().keys():
            mapping.update(
                {
                    "vision_model.conv1.weight": "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
                    "vision_model.position_embeddings.weight": "model.vision_tower.vision_model.embeddings.position_embedding.weight",
                    "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.vision_tower.vision_model.encoder.layers.*.layer_norm1.weight",
                    "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "model.vision_tower.vision_model.encoder.layers.*.layer_norm1.bias",
                    "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.vision_tower.vision_model.encoder.layers.*.layer_norm2.weight",
                    "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "model.vision_tower.vision_model.encoder.layers.*.layer_norm2.bias",
                    "vision_model.decoder.layers.*.self_attention.linear_proj.weight": "model.vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.weight",
                    "vision_model.decoder.layers.*.self_attention.linear_proj.bias": "model.vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.bias",
                    "vision_model.decoder.layers.*.mlp.linear_fc1.weight": "model.vision_tower.vision_model.encoder.layers.*.mlp.fc1.weight",
                    "vision_model.decoder.layers.*.mlp.linear_fc1.bias": "model.vision_tower.vision_model.encoder.layers.*.mlp.fc1.bias",
                    "vision_model.decoder.layers.*.mlp.linear_fc2.weight": "model.vision_tower.vision_model.encoder.layers.*.mlp.fc2.weight",
                    "vision_model.decoder.layers.*.mlp.linear_fc2.bias": "model.vision_tower.vision_model.encoder.layers.*.mlp.fc2.bias",
                    "vision_model.ln_pre.weight": "model.vision_tower.vision_model.pre_layrnorm.weight",
                    "vision_model.ln_pre.bias": "model.vision_tower.vision_model.pre_layrnorm.bias",
                }
            )

        # Add transformations for specialized tensor manipulations
        transforms = [
            _export_language_qkv,
            _export_vision_qkv,
            _export_vision_qkv_bias,
            _export_language_linear_fc1,
            _export_embedding,
        ]

        # If word embeddings are not shared, add the head export transform
        if not source.config.language_transformer_config.share_embeddings_and_output_weights:
            transforms.append(_export_language_head)

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self) -> "TokenizerSpec":
        """
        Gets the tokenizer from the loaded model context.

        Returns:
            The tokenizer specification.
        """
        return io.load_context(str(self), subpath="model").tokenizer

    @property
    def config(self) -> "HFLlavaConfig":
        """
        Generates the configuration for the HuggingFace LLaVA Next model based on the NeMo model.

        Returns:
            HFLlavaConfig: A configuration object for the HuggingFace LLaVA Next model.
        """
        from transformers import LlamaConfig as HFLlamaConfig

        source = io.load_context(str(self), subpath="model.config")
        language_config = source.language_transformer_config
        vit_path = getattr(
            source.vision_transformer_config, "pretrained_model_name_or_path", "openai/clip-vit-large-patch14-336"
        )
        vision_config = CLIPVisionConfig.from_pretrained(vit_path)

        # Create text config for HuggingFace model
        text_config = HFLlamaConfig(
            num_hidden_layers=language_config.num_layers,
            hidden_size=language_config.hidden_size,
            intermediate_size=language_config.ffn_hidden_size,
            num_attention_heads=language_config.num_attention_heads,
            max_position_embeddings=language_config.seq_length,
            initializer_range=language_config.init_method_std,
            rms_norm_eps=language_config.layernorm_epsilon,
            num_key_value_heads=language_config.num_query_groups,
            rope_theta=language_config.rotary_base,
            vocab_size=self.tokenizer.vocab_size,
            tie_word_embeddings=language_config.share_embeddings_and_output_weights,
        )
        # Create the LlavaConfig for HuggingFace
        return HFLlavaConfig(
            text_config=text_config,
            vision_config=vision_config,
            vision_feature_layer=source.vision_feature_layer,
            # Add any additional LlavaNext-specific configurations
            model_type="llava_next",
        )


# Define transformation functions needed for the exporter


@io.state_transform(
    source_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "model.language_model.layers.*.self_attn.q_proj.weight",
        "model.language_model.layers.*.self_attn.k_proj.weight",
        "model.language_model.layers.*.self_attn.v_proj.weight",
    ),
)
def _export_language_qkv(ctx: io.TransformCTX, linear_qkv):
    """Transforms NeMo's fused QKV weights to separate Q, K, V weights for HuggingFace format."""
    megatron_config = ctx.source.config.language_transformer_config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, hidden_size])
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()

    return q_proj, k_proj, v_proj


@io.state_transform(
    source_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "model.vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.weight",
        "model.vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.weight",
        "model.vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.weight",
    ),
)
def _export_vision_qkv(ctx: io.TransformCTX, linear_qkv):
    """Transforms NeMo's fused vision QKV weights to separate Q, K, V weights for HuggingFace format."""
    megatron_config = ctx.source.config.vision_transformer_config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, hidden_size])
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()

    return q_proj, k_proj, v_proj


@io.state_transform(
    source_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
    target_key=(
        "model.vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.bias",
        "model.vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.bias",
        "model.vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.bias",
    ),
)
def _export_vision_qkv_bias(ctx: io.TransformCTX, linear_qkv_bias):
    """Transforms NeMo's fused vision QKV biases to separate Q, K, V biases for HuggingFace format."""
    megatron_config = ctx.source.config.vision_transformer_config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv_bias = linear_qkv_bias.reshape([qkv_total_dim, head_size])
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj_bias = linear_qkv_bias[q_slice].reshape(-1).cpu()
    k_proj_bias = linear_qkv_bias[k_slice].reshape(-1).cpu()
    v_proj_bias = linear_qkv_bias[v_slice].reshape(-1).cpu()

    return q_proj_bias, k_proj_bias, v_proj_bias


@io.state_transform(
    source_key="vision_model.class_token",
    target_key="vision_tower.vision_model.embeddings.class_embedding",
)
def _export_cls_token(ctx: io.TransformCTX, class_token):
    """Transforms the class token from NeMo to HuggingFace format."""
    return class_token.reshape(-1)


@io.state_transform(
    source_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
    target_key=(
        "model.language_model.layers.*.mlp.gate_proj.weight",
        "model.language_model.layers.*.mlp.up_proj.weight",
    ),
)
def _export_language_linear_fc1(ctx: io.TransformCTX, linear_fc1):
    """Splits NeMo's fused MLP linear_fc1 weight into gate_proj and up_proj for HuggingFace format."""
    gate_proj, up_proj = torch.chunk(linear_fc1, 2, dim=0)
    return gate_proj, up_proj


@io.state_transform(
    source_key="language_model.embedding.word_embeddings.weight",
    target_key="model.language_model.embed_tokens.weight",
)
def _export_embedding(ctx: io.TransformCTX, embedding):
    """Transforms the word embeddings from NeMo to HuggingFace format."""
    hf_config = ctx.target.config.text_config
    # Prune any padding to match the HuggingFace vocab size
    return embedding[: hf_config.vocab_size, :]


@io.state_transform(
    source_key="language_model.output_layer.weight",
    target_key="lm_head.weight",
)
def _export_language_head(ctx: io.TransformCTX, output_weight):
    """Transforms the output layer from NeMo to HuggingFace format."""
    hf_config = ctx.target.config.text_config
    # Prune any padding to match the HuggingFace vocab size
    return output_weight[: hf_config.vocab_size, :]


__all__ = [
    "LlavaNextModel",
]
