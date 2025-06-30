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
from typing import TYPE_CHECKING, Dict, Tuple, Union

import torch
import transformers
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import AutoConfig as HFAutoConfig
from transformers import AutoModelForImageTextToText
from transformers import Qwen2_5_VLConfig as HFQwen25VLConfig
from transformers import Qwen2VLConfig as HFQwen2VLConfig
from transformers import Qwen2VLForConditionalGeneration
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig as HFQwen25VLVisionConfig
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig as HFQwen2VLVisionConfig

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import (
    Qwen2Config,
    Qwen2Config1P5B,
    Qwen2Config7B,
    Qwen2Config72B,
    Qwen25Config3B,
    Qwen25Config7B,
    Qwen25Config32B,
    Qwen25Config72B,
)
from nemo.collections.vlm.neva.model.llava import export_qkv, export_qkv_bias
from nemo.collections.vlm.qwen2vl.model.base import (
    Qwen2VLConfig,
    Qwen2VLModel,
    Qwen2VLVisionConfig,
    Qwen25VLVisionConfig,
)
from nemo.collections.vlm.vision import MultimodalProjectorConfig
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_distributed_model_weights
from nemo.lightning import io, teardown
from nemo.lightning.io.state import _ModelState
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer


# Note: these Qwen2VL configs are copied from the corresponding HF model. You may need to modify the parameter for
# your own needs
@dataclass
class Qwen2VLConfig2B(Qwen2VLConfig):
    """Qwen2VL Config 2B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(
        default_factory=lambda: Qwen2Config1P5B(share_embeddings_and_output_weights=True)
    )
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen2VLVisionConfig(num_layers=32, num_attention_heads=16)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(input_size=5120, hidden_size=1536, ffn_hidden_size=5120)
    )


@dataclass
class Qwen2VLConfig7B(Qwen2VLConfig):
    """Qwen2VL Config 7B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen2Config7B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen2VLVisionConfig(num_layers=32, num_attention_heads=16)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(input_size=5120, hidden_size=3584, ffn_hidden_size=5120)
    )


@dataclass
class Qwen2VLConfig72B(Qwen2VLConfig):
    """Qwen2VL Config 72B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen2Config72B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen2VLVisionConfig(num_layers=32, num_attention_heads=16)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(input_size=5120, hidden_size=8192, ffn_hidden_size=5120)
    )


@dataclass
class Qwen25VLConfig3B(Qwen2VLConfig):
    """Qwen2.5VL Config 3B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen25Config3B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen25VLVisionConfig(num_layers=32, num_attention_heads=16)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(
            projector_type="mcore_mlp", input_size=5120, hidden_size=2048, ffn_hidden_size=5120
        )
    )


@dataclass
class Qwen25VLConfig7B(Qwen2VLConfig):
    """Qwen2.5VL Config 7B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen25Config7B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen25VLVisionConfig(num_layers=32, num_attention_heads=16)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(
            projector_type="mcore_mlp", input_size=5120, hidden_size=3584, ffn_hidden_size=5120
        )
    )


@dataclass
class Qwen25VLConfig32B(Qwen2VLConfig):
    """Qwen2.5VL Config 32B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen25Config32B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen25VLVisionConfig(num_layers=32, num_attention_heads=16, ffn_hidden_size=3456)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(
            projector_type="mcore_mlp", input_size=5120, hidden_size=5120, ffn_hidden_size=5120
        )
    )


@dataclass
class Qwen25VLConfig72B(Qwen2VLConfig):
    """Qwen2.5VL Config 72B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen25Config72B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen25VLVisionConfig(num_layers=32, num_attention_heads=16, ffn_hidden_size=3456)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(
            projector_type="mcore_mlp", input_size=5120, hidden_size=8192, ffn_hidden_size=5120
        )
    )


@io.model_importer(Qwen2VLModel, "hf")
class HFQwen2VLImporter(io.ModelConnector["Qwen2VLForConditionalGeneration", Qwen2VLModel]):
    """Qwen2VL Model HF Importer"""

    def init(self) -> Qwen2VLModel:
        # pylint: disable=C0115,C0116
        return Qwen2VLModel(self.config, model_version="qwen2-vl", tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        # pylint: disable=C0115,C0116
        source = AutoModelForImageTextToText.from_pretrained(str(self), trust_remote_code=True)
        hf_config = HFAutoConfig.from_pretrained(str(self), trust_remote_code=True)
        self.is_v2_5 = hf_config.model_type == "qwen2_5_vl"

        target = self.init()
        trainer = self.nemo_setup(target)
        source = source.to(dtype_from_hf(hf_config))
        target = target.to(dtype_from_hf(hf_config))
        self.convert_state(source, target)
        print(f"Converted Qwen2VL model to Nemo, saving to {output_path}")
        self.nemo_save(output_path, trainer)
        print(f"Converted Qwen2VL model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        # pylint: disable=C0115,C0116,C0301
        mapping = {
            "visual.patch_embed.proj.weight": "vision_model.conv1.weight",
            "visual.blocks.*.norm1.weight": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "visual.blocks.*.norm1.bias": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
            "visual.blocks.*.norm2.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "visual.blocks.*.norm2.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            "visual.blocks.*.attn.proj.weight": "vision_model.decoder.layers.*.self_attention.linear_proj.weight",
            "visual.blocks.*.attn.proj.bias": "vision_model.decoder.layers.*.self_attention.linear_proj.bias",
            "model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.norm.weight": "language_model.decoder.final_layernorm.weight",
            # "lm_head.weight": "language_model.output_layer.weight",
        }
        if not target.config.language_transformer_config.share_embeddings_and_output_weights:
            mapping.update({"lm_head.weight": "language_model.output_layer.weight"})

        if self.is_v2_5:
            mapping.update(
                {
                    "visual.blocks.*.mlp.down_proj.weight": "vision_model.decoder.layers.*.mlp.linear_fc2.weight",
                    "visual.blocks.*.mlp.down_proj.bias": "vision_model.decoder.layers.*.mlp.linear_fc2.bias",
                    "visual.merger.ln_q.weight": "vision_model.decoder.final_layernorm.weight",
                }
            )
        else:
            mapping.update(
                {
                    "visual.blocks.*.mlp.fc1.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.weight",
                    "visual.blocks.*.mlp.fc1.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.bias",
                    "visual.blocks.*.mlp.fc2.weight": "vision_model.decoder.layers.*.mlp.linear_fc2.weight",
                    "visual.blocks.*.mlp.fc2.bias": "vision_model.decoder.layers.*.mlp.linear_fc2.bias",
                    "visual.merger.ln_q.weight": "vision_model.decoder.final_layernorm.weight",
                    "visual.merger.ln_q.bias": "vision_model.decoder.final_layernorm.bias",
                }
            )
        if "vision_projection.encoder.linear_fc1.weight" in target.module.state_dict().keys():
            mapping.update(
                {
                    "visual.merger.mlp.0.weight": "vision_projection.encoder.linear_fc1.weight",
                    "visual.merger.mlp.0.bias": "vision_projection.encoder.linear_fc1.bias",
                    "visual.merger.mlp.2.weight": "vision_projection.encoder.linear_fc2.weight",
                    "visual.merger.mlp.2.bias": "vision_projection.encoder.linear_fc2.bias",
                }
            )
        elif "vision_projection.0.weight" in target.module.state_dict().keys():
            mapping.update(
                {
                    "visual.merger.mlp.0.weight": "vision_projection.0.weight",
                    "visual.merger.mlp.0.bias": "vision_projection.0.bias",
                    "visual.merger.mlp.2.weight": "vision_projection.2.weight",
                    "visual.merger.mlp.2.bias": "vision_projection.2.bias",
                }
            )
        else:
            raise KeyError("Unable to map vision projection keys.")

        transforms = [
            _import_language_qkv,
            _import_language_qkv_bias,
            _import_vision_qkv,
            _import_vision_qkv_bias,
            _import_linear_fc1,
        ]
        if self.is_v2_5:
            transforms += [
                _import_vision_linear_fc1_weight,
                _import_vision_linear_fc1_bias,
            ]
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self) -> "AutoTokenizer":
        # pylint: disable=C0115,C0116
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(str(self))

    @property
    def config(self) -> Qwen2VLConfig:
        # pylint: disable=C0115,C0116
        from packaging.version import Version

        if Version(transformers.__version__) > Version('4.51.3'):
            # Todo: need to fix with newest version of transformers
            raise ValueError(
                f"Current version of transformers is {transformers.__version__},"
                f"Please lower the version to be <= 4.51.3"
            )

        hf_config = HFAutoConfig.from_pretrained(str(self), trust_remote_code=True)
        is_v2_5 = hf_config.model_type == "qwen2_5_vl"

        def make_vocab_size_divisible_by(vocab_size):
            # pylint: disable=C0115,C0116
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        text_config = hf_config
        language_transformer_config = Qwen2Config(
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            num_query_groups=text_config.num_key_value_heads,
            rotary_base=text_config.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(text_config.vocab_size),
            share_embeddings_and_output_weights=text_config.tie_word_embeddings,
            vocab_size=text_config.vocab_size,
            fp16=(dtype_from_hf(text_config) == torch.float16),
            bf16=(dtype_from_hf(text_config) == torch.bfloat16),
            params_dtype=dtype_from_hf(text_config),
        )

        # Use MCore instead of Pytorch
        vision_config = hf_config.vision_config
        if is_v2_5:
            vision_transformer_config = Qwen25VLVisionConfig(
                ffn_hidden_size=vision_config.intermediate_size,
                fp16=(dtype_from_hf(hf_config) == torch.float16),
                bf16=(dtype_from_hf(hf_config) == torch.bfloat16),
                params_dtype=dtype_from_hf(hf_config),
            )
            merge_hidden_size = vision_config.hidden_size * (vision_config.spatial_merge_size**2)
            vision_projection_config = MultimodalProjectorConfig(
                input_size=merge_hidden_size,
                hidden_size=vision_config.out_hidden_size,
                ffn_hidden_size=merge_hidden_size,
                projector_type="mcore_mlp",
                fp16=(dtype_from_hf(hf_config) == torch.float16),
                bf16=(dtype_from_hf(hf_config) == torch.bfloat16),
                params_dtype=dtype_from_hf(hf_config),
            )
        else:
            vision_transformer_config = Qwen2VLVisionConfig(
                fp16=(dtype_from_hf(hf_config) == torch.float16),
                bf16=(dtype_from_hf(hf_config) == torch.bfloat16),
                params_dtype=dtype_from_hf(hf_config),
            )
            merge_hidden_size = vision_config.embed_dim * (vision_config.spatial_merge_size**2)
            vision_projection_config = MultimodalProjectorConfig(
                input_size=merge_hidden_size,
                hidden_size=vision_config.hidden_size,
                ffn_hidden_size=merge_hidden_size,
                projector_type="mcore_mlp",
                fp16=(dtype_from_hf(hf_config) == torch.float16),
                bf16=(dtype_from_hf(hf_config) == torch.bfloat16),
                params_dtype=dtype_from_hf(hf_config),
            )

        output = Qwen2VLConfig(
            language_transformer_config=language_transformer_config,
            vision_transformer_config=vision_transformer_config,
            vision_projection_config=vision_projection_config,
            vision_feature_layer=-1,
            fp16=(dtype_from_hf(hf_config) == torch.float16),
            bf16=(dtype_from_hf(hf_config) == torch.bfloat16),
            params_dtype=dtype_from_hf(hf_config),
        )

        return output


@io.model_exporter(Qwen2VLModel, "hf")
class HFQwen2VLExporter(io.ModelConnector[Qwen2VLModel, "Qwen2VLForConditionalGeneration"]):
    """
    Exporter class for converting NeMo Qwen2VL model to HuggingFace format.

    Inherits:
        io.ModelConnector: Connector interface to handle setup, save, and load using the Lightning framework.

    Methods:
        init: Initializes a new HuggingFace Qwen2VL model instance.
        apply: Converts the NeMo model to HuggingFace format and saves it.
        convert_state: Maps and transforms the state dictionary from NeMo to HuggingFace format.
        config: Generates and returns the HuggingFace Qwen2VL config for the model.
    """

    def init(self, dtype=torch.bfloat16) -> "Qwen2VLForConditionalGeneration":
        """
        Initializes a HuggingFace Qwen2VLForConditionalGeneration model.

        Args:
            dtype: The data type to use for the model (default: torch.bfloat16)

        Returns:
            Qwen2VLForConditionalGeneration: A HuggingFace Qwen2VL model initialized with the configuration.
        """
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForImageTextToText.from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        """
        Converts the NeMo Qwen2VL model to HuggingFace format and saves it to the specified path.

        Args:
            output_path (Path): The path where the converted HuggingFace model will be saved.

        Returns:
            Path: The output path where the HuggingFace model was saved.
        """
        logging.info("Loading Qwen2VL NeMo checkpoint. This may take a while...")
        source, source_config = self.ckpt_load(self)
        logging.info("Qwen2VL NeMo checkpoint loaded.")
        logging.info("Initializing the HF model..")
        target = self.init()
        logging.info("Start Converting the model..")
        target = self.convert_state(source, target, source_config)
        target = target.cpu()
        target.save_pretrained(output_path)

        try:
            self.tokenizer.tokenizer.save_pretrained(output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        print(f"Converted Qwen2VL model saved to {output_path}")

        return output_path

    def convert_state(self, source, target, source_config):
        # pylint: disable=C0115,C0116,line-too-long
        """
        Maps and transforms the state dictionary from NeMo to HuggingFace format.

        Args:
            source: The source NeMo model.
            target: The target HuggingFace model.

        Returns:
            The target HuggingFace model with the converted state.
        """

        mapping = {
            "vision_model.conv1.weight": "visual.patch_embed.proj.weight",
            "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "visual.blocks.*.norm1.weight",
            "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "visual.blocks.*.norm1.bias",
            "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "visual.blocks.*.norm2.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "visual.blocks.*.norm2.bias",
            "vision_model.decoder.layers.*.self_attention.linear_proj.weight": "visual.blocks.*.attn.proj.weight",
            "vision_model.decoder.layers.*.self_attention.linear_proj.bias": "visual.blocks.*.attn.proj.bias",
            "language_model.embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.final_layernorm.weight": "model.norm.weight",
            # "language_model.output_layer.weight": "lm_head.weight",
        }
        if source_config.language_transformer_config.share_embeddings_and_output_weights:
            mapping.update({"language_model.embedding.word_embeddings.weight": "lm_head.weight"})
        else:
            mapping.update({"language_model.output_layer.weight": "lm_head.weight"})

        if self.is_v2_5:
            mapping.update(
                {
                    "vision_model.decoder.layers.*.mlp.linear_fc2.weight": "visual.blocks.*.mlp.down_proj.weight",
                    "vision_model.decoder.layers.*.mlp.linear_fc2.bias": "visual.blocks.*.mlp.down_proj.bias",
                    "vision_model.decoder.final_layernorm.weight": "visual.merger.ln_q.weight",
                }
            )

        else:
            mapping.update(
                {
                    "vision_model.decoder.layers.*.mlp.linear_fc1.weight": "visual.blocks.*.mlp.fc1.weight",
                    "vision_model.decoder.layers.*.mlp.linear_fc1.bias": "visual.blocks.*.mlp.fc1.bias",
                    "vision_model.decoder.layers.*.mlp.linear_fc2.weight": "visual.blocks.*.mlp.fc2.weight",
                    "vision_model.decoder.layers.*.mlp.linear_fc2.bias": "visual.blocks.*.mlp.fc2.bias",
                    "vision_model.decoder.final_layernorm.weight": "visual.merger.ln_q.weight",
                    "vision_model.decoder.final_layernorm.bias": "visual.merger.ln_q.bias",
                }
            )
        if "vision_projection.encoder.linear_fc1.weight" in source.state_dict().keys():
            mapping.update(
                {
                    "vision_projection.encoder.linear_fc1.weight": "visual.merger.mlp.0.weight",
                    "vision_projection.encoder.linear_fc1.bias": "visual.merger.mlp.0.bias",
                    "vision_projection.encoder.linear_fc2.weight": "visual.merger.mlp.2.weight",
                    "vision_projection.encoder.linear_fc2.bias": "visual.merger.mlp.2.bias",
                }
            )
        elif "vision_projection.0.weight" in source.state_dict().keys():
            mapping.update(
                {
                    "vision_projection.0.weight": "visual.merger.mlp.0.weight",
                    "vision_projection.0.bias": "visual.merger.mlp.0.bias",
                    "vision_projection.2.weight": "visual.merger.mlp.2.weight",
                    "vision_projection.2.bias": "visual.merger.mlp.2.bias",
                }
            )
        else:
            raise KeyError("Unable to map vision projection keys.")

        transforms = [
            _export_language_qkv,
            _export_language_qkv_bias,
            _export_vision_qkv,
            _export_vision_qkv_bias,
            _export_linear_fc1,
        ]
        if self.is_v2_5:
            transforms += [
                _export_vision_linear_fc1_weight,
                _export_vision_linear_fc1_bias,
            ]

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

    def ckpt_load(self, path: Path) -> Tuple[Dict, Dict]:
        """
        This function loads the state dict directly from a distributed checkpoint, and modify the state dict
        so that it is consistent with the key names you would get from loading the checkpoint into a model.
        This is a more memory-efficient method to obtain a state dict without initializing the nemo model.

        Args:
            path (Path): The path from which the model will be loaded.

        Returns
        -------
            Tuple[Dict, Dict]: The loaded state dict and the yaml config dict.
        """
        config = io.load_context(str(self), subpath="model.config")
        dist_ckpt_folder = path / "weights"
        state_dict = {}

        langauge_layers = config.language_transformer_config.num_layers
        vision_layers = config.vision_transformer_config.num_layers
        distributed_model_weights = load_distributed_model_weights(dist_ckpt_folder, True).items()
        for k, v in distributed_model_weights:
            if "_extra_state" in k:
                continue
            new_k = k.replace("module.", "")
            if "layers" in new_k and (v.size(0) == langauge_layers or v.size(0) == vision_layers):
                # Only split layers
                for i in range(v.size(0)):
                    state_dict[new_k.replace("layers", f"layers.{str(i)}")] = v[i]
            state_dict[new_k] = v

        source = _ModelState(state_dict)
        return source, config

    @property
    def config(self) -> "HFQwen2VLConfig":
        """
        Generates the configuration for the HuggingFace Qwen2VL model based on the NeMo model.

        Returns:
            HFQwen2VLConfig: A configuration object for the HuggingFace Qwen2VL model.
        """
        from packaging.version import Version

        if Version(transformers.__version__) > Version('4.51.3'):
            # Todo: need to fix with newest version of transformers
            raise ValueError(
                f"Current version of transformers is {transformers.__version__},"
                f"Please lower the version to be <= 4.51.3"
            )
        source = io.load_context(str(self), subpath="model.config")

        language_config = source.language_transformer_config
        vision_model_config = source.vision_transformer_config
        vision_projection_config = source.vision_projection_config

        self.is_v2_5 = hasattr(vision_model_config, "fullatt_block_indexes") and (
            vision_model_config.fullatt_block_indexes != None
        )

        if self.is_v2_5:
            vision_config = HFQwen25VLVisionConfig(
                depth=vision_model_config.num_layers,
                embed_dim=vision_model_config.embed_dim,
                hidden_size=vision_model_config.hidden_size,
                out_hidden_size=language_config.hidden_size,
                hidden_act="silu",
                mlp_ratio=int(vision_projection_config.ffn_hidden_size // vision_model_config.hidden_size),
                num_heads=vision_model_config.num_attention_heads,
                in_channels=3,
                patch_size=vision_model_config.patch_dim,
                spatial_merge_size=vision_model_config.spatial_merge_size,
                spatial_patch_size=vision_model_config.spatial_patch_size,
                temporal_patch_size=vision_model_config.temporal_patch_size,
                initializer_range=vision_model_config.init_method_std,
                fullatt_block_indexes=[7, 15, 23, 31],
                tokens_per_second=2,
                model_type="qwen2_5_vl",
                torch_dtype="bfloat16",
            ).to_dict()

            # Create the LlavaConfig for HuggingFace
            hf_config = HFQwen25VLConfig(
                vision_config=vision_config,
                num_hidden_layers=language_config.num_layers,
                hidden_size=language_config.hidden_size,
                intermediate_size=language_config.ffn_hidden_size,
                num_attention_heads=language_config.num_attention_heads,
                max_window_layers=70,
                max_position_embeddings=language_config.seq_length,
                initializer_range=language_config.init_method_std,
                rms_norm_eps=language_config.layernorm_epsilon,
                num_key_value_heads=language_config.num_query_groups,
                rope_theta=language_config.rotary_base,
                vocab_size=language_config.vocab_size,
                rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
                tie_word_embeddings=language_config.share_embeddings_and_output_weights,
                torch_dtype="bfloat16",
                # vocab_size=self.tokenizer.vocab_size,
                bos_token_id=151643,
                eos_token_id=151645,
                vision_start_token_id=151652,
                vision_end_token_id=151653,
                vision_token_id=151654,
                image_token_id=151655,
                video_token_id=51656,
            )
            return hf_config
        else:
            vision_config = HFQwen2VLVisionConfig(
                depth=vision_model_config.num_layers,
                embed_dim=vision_model_config.embed_dim,
                hidden_size=vision_projection_config.hidden_size,
                hidden_act="quick_gelu",
                mlp_ratio=int(vision_projection_config.ffn_hidden_size // vision_model_config.hidden_size),
                num_heads=vision_model_config.num_attention_heads,
                in_channels=3,
                patch_size=vision_model_config.patch_dim,
                spatial_merge_size=vision_model_config.spatial_merge_size,
                spatial_patch_size=vision_model_config.spatial_patch_size,
                temporal_patch_size=vision_model_config.temporal_patch_size,
                initializer_range=vision_model_config.init_method_std,
                model_type="qwen2_vl",
                torch_dtype="bfloat16",
            ).to_dict()

            # Create the Qwen2VLConfig for HuggingFace
            # if transformers > 4.51.3, use Qwen2VLTextConfig as text_config
            # https://github.com/huggingface/transformers/pull/37268
            return HFQwen2VLConfig(
                num_hidden_layers=language_config.num_layers,
                hidden_size=language_config.hidden_size,
                intermediate_size=language_config.ffn_hidden_size,
                num_attention_heads=language_config.num_attention_heads,
                initializer_range=language_config.init_method_std,
                rms_norm_eps=language_config.layernorm_epsilon,
                num_key_value_heads=language_config.num_query_groups,
                rope_theta=language_config.rotary_base,
                tie_word_embeddings=language_config.share_embeddings_and_output_weights,
                vocab_size=language_config.vocab_size,
                vision_config=vision_config,
                torch_dtype="bfloat16",
            )


def import_qkv(q, k, v, head_num, num_query_groups, heads_per_group, hidden_size, head_size):
    # pylint: disable=C0115,C0116
    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights_l = []
    for i in range(num_query_groups):
        qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_weights_l.append(k[i : i + 1, :, :])
        qkv_weights_l.append(v[i : i + 1, :, :])
    qkv_weights = torch.cat(qkv_weights_l)
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key=("visual.blocks.*.attn.qkv.weight",),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_vision_qkv(ctx: io.TransformCTX, hf_qkv_weights):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.vision_transformer_config

    slice = int(hf_qkv_weights.shape[0] / 3)
    assert slice == megatron_config.hidden_size
    q = hf_qkv_weights[:slice, :]
    k = hf_qkv_weights[slice : slice * 2, :]
    v = hf_qkv_weights[slice * 2 :, :]

    return import_qkv(
        q,
        k,
        v,
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=megatron_config.hidden_size,
        head_size=megatron_config.kv_channels,
    )


@io.state_transform(
    source_key=("visual.blocks.*.attn.qkv.bias",),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_vision_qkv_bias(ctx: io.TransformCTX, hf_qkv_bias):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.vision_transformer_config

    slice = int(hf_qkv_bias.shape[0] / 3)
    assert slice == megatron_config.hidden_size

    q_bias = hf_qkv_bias[:slice]
    k_bias = hf_qkv_bias[slice : slice * 2]
    v_bias = hf_qkv_bias[slice * 2 :]

    return import_qkv(
        q_bias.unsqueeze(-1),
        k_bias.unsqueeze(-1),
        v_bias.unsqueeze(-1),
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=1,
        head_size=megatron_config.kv_channels,
    ).squeeze(-1)


@io.state_transform(
    source_key=(
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_language_qkv(ctx: io.TransformCTX, q, k, v):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.language_transformer_config
    return import_qkv(
        q,
        k,
        v,
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=megatron_config.hidden_size,
        head_size=megatron_config.kv_channels,
    )


@io.state_transform(
    source_key=(
        "model.layers.*.self_attn.q_proj.bias",
        "model.layers.*.self_attn.k_proj.bias",
        "model.layers.*.self_attn.v_proj.bias",
    ),
    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_language_qkv_bias(ctx: io.TransformCTX, q_bias, k_bias, v_bias):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.language_transformer_config
    return import_qkv(
        q_bias.unsqueeze(-1),
        k_bias.unsqueeze(-1),
        v_bias.unsqueeze(-1),
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=1,
        head_size=megatron_config.kv_channels,
    ).squeeze(-1)


@io.state_transform(
    source_key=("vision_model.embeddings.class_embedding",),
    target_key="vision_model.class_token",
)
def _import_cls_token(ctx: io.TransformCTX, cls_token):
    # pylint: disable=C0115,C0116
    return cls_token.reshape(1, 1, -1)


@io.state_transform(
    source_key=(
        "model.layers.*.mlp.gate_proj.weight",
        "model.layers.*.mlp.up_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(down, gate):
    # pylint: disable=C0115,C0116
    return torch.cat((down, gate), axis=0)


@io.state_transform(
    source_key=("visual.blocks.*.mlp.gate_proj.weight", "visual.blocks.*.mlp.up_proj.weight"),
    target_key="vision_model.decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_vision_linear_fc1_weight(down, gate):
    # pylint: disable=C0115,C0116
    return torch.cat((down, gate), axis=0)


@io.state_transform(
    source_key=("visual.blocks.*.mlp.gate_proj.bias", "visual.blocks.*.mlp.up_proj.bias"),
    target_key="vision_model.decoder.layers.*.mlp.linear_fc1.bias",
)
def _import_vision_linear_fc1_bias(down, gate):
    # pylint: disable=C0115,C0116
    return torch.cat((down, gate), axis=0)


def export_qkv(linear_qkv, head_num, num_query_groups, heads_per_group, hidden_size, head_size):
    # pylint: disable=C0115,C0116
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, -1])
    hidden_size = linear_qkv.size(-1)
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


def export_qkv_bias(qkv_bias: torch.Tensor, head_num, num_query_groups, heads_per_group, head_size):
    """
    Split interleave-concatenated qkv bias to separate q, k, v bias

    Example: export layer linear_qkv bias to HF {q|k|v}_proj bias
    """
    qkv_total_dim = head_num + 2 * num_query_groups

    qkv_bias = qkv_bias.reshape([qkv_total_dim, head_size])
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_bias = qkv_bias[q_slice].reshape(-1).cpu()
    k_bias = qkv_bias[k_slice].reshape(-1).cpu()
    v_bias = qkv_bias[v_slice].reshape(-1).cpu()

    return q_bias, k_bias, v_bias


@io.state_transform(
    source_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
    target_key="visual.blocks.*.attn.qkv.weight",
)
def _export_vision_qkv(ctx: io.TransformCTX, qkv):
    # pylint: disable=C0115,C0116
    hf_config = ctx.target.config.vision_config
    hidden_size = hf_config.embed_dim if hf_config.model_type == "qwen2_vl" else hf_config.hidden_size
    return torch.cat(
        export_qkv(
            qkv,
            head_num=hf_config.num_heads,
            num_query_groups=hf_config.num_heads,
            heads_per_group=hf_config.num_heads // hf_config.num_heads,
            hidden_size=hidden_size,
            head_size=hidden_size // hf_config.num_heads,
        ),
        axis=0,
    )


@io.state_transform(
    source_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
    target_key="visual.blocks.*.attn.qkv.bias",
)
def _export_vision_qkv_bias(ctx: io.TransformCTX, qkv_bias):
    # pylint: disable=C0115,C0116
    hf_config = ctx.target.config.vision_config
    hidden_size = hf_config.embed_dim if hf_config.model_type == "qwen2_vl" else hf_config.hidden_size
    return torch.cat(
        export_qkv_bias(
            qkv_bias,
            head_num=hf_config.num_heads,
            num_query_groups=hf_config.num_heads,
            heads_per_group=hf_config.num_heads // hf_config.num_heads,
            head_size=hidden_size // hf_config.num_heads,
        ),
        axis=0,
    )


@io.state_transform(
    source_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ),
)
def _export_language_qkv(ctx: io.TransformCTX, qkv):
    # pylint: disable=C0115,C0116
    hf_config = ctx.target.config
    return export_qkv(
        qkv,
        head_num=hf_config.num_attention_heads,
        num_query_groups=hf_config.num_key_value_heads,
        heads_per_group=hf_config.num_attention_heads // hf_config.num_key_value_heads,
        hidden_size=hf_config.hidden_size,
        head_size=hf_config.hidden_size // hf_config.num_attention_heads,
    )


@io.state_transform(
    source_key="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
    target_key=(
        "model.layers.*.self_attn.q_proj.bias",
        "model.layers.*.self_attn.k_proj.bias",
        "model.layers.*.self_attn.v_proj.bias",
    ),
)
def _export_language_qkv_bias(ctx: io.TransformCTX, qkv_bias):
    # pylint: disable=C0115,C0116
    hf_config = ctx.target.config
    return export_qkv_bias(
        qkv_bias,
        head_num=hf_config.num_attention_heads,
        num_query_groups=hf_config.num_key_value_heads,
        heads_per_group=hf_config.num_attention_heads // hf_config.num_key_value_heads,
        head_size=hf_config.hidden_size // hf_config.num_attention_heads,
    )


@io.state_transform(
    source_key="vision_model.class_token",
    target_key="vision_model.embeddings.class_embedding",
)
def _export_cls_token(ctx: io.TransformCTX, cls_token):
    # pylint: disable=C0115,C0116
    return cls_token.squeeze()


@io.state_transform(
    source_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
    target_key=(
        "model.layers.*.mlp.gate_proj.weight",
        "model.layers.*.mlp.up_proj.weight",
    ),
)
def _export_linear_fc1(linear_fc1):
    # pylint: disable=C0115,C0116
    gate_proj, up_proj = torch.chunk(linear_fc1, 2, dim=0)
    return gate_proj, up_proj


@io.state_transform(
    source_key="vision_model.decoder.layers.*.mlp.linear_fc1.weight",
    target_key=(
        "visual.blocks.*.mlp.gate_proj.weight",
        "visual.blocks.*.mlp.up_proj.weight",
    ),
)
def _export_vision_linear_fc1_weight(vision_fc1_weight):
    # pylint: disable=C0115,C0116
    gate_proj, up_proj = torch.chunk(vision_fc1_weight, 2, dim=0)
    return gate_proj, up_proj


@io.state_transform(
    source_key="vision_model.decoder.layers.*.mlp.linear_fc1.bias",
    target_key=(
        "visual.blocks.*.mlp.gate_proj.bias",
        "visual.blocks.*.mlp.up_proj.bias",
    ),
)
def _export_vision_linear_fc1_bias(vision_fc1_bias):
    # pylint: disable=C0115,C0116
    gate_proj, up_proj = torch.chunk(vision_fc1_bias, 2, dim=0)
    return gate_proj, up_proj
