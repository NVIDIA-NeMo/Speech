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

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import torch
from torch import nn

from nemo.collections.common.tokenizers import AutoTokenizer, TokenizerSpec
from nemo.collections.llm.bert.model.base import BertConfig, BertModel
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils import logging

if TYPE_CHECKING:
    from transformers import BertConfig as HFBertConfig
    from transformers import BertModel


@dataclass
class MegatronBertConfig(BertConfig):
    """Configs for training megatron-style Bert Model."""

    bert_type: str = 'megatron'
    add_pooler: bool = False
    init_method_std: float = 0.02
    hidden_dropout: float = 0.1
    normalization: float = 'LayerNorm'
    layernorm_epsilon: float = 1e-5
    apply_query_key_layer_scaling: bool = False
    position_embedding_type: str = "learned_absolute"
    bert_binary_head: bool = False


@dataclass
class MegatronBertLargeConfig(MegatronBertConfig):
    """Configs for Bert-Large in megatron style."""

    num_layers: int = 24
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 16


@dataclass
class MegatronBertBaseConfig(MegatronBertConfig):
    """Configs for Bert-Base in megatron style."""

    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 12


@dataclass
class HuggingFaceBertConfig(BertConfig):
    """Configs for models in https://huggingface.co/google-bert"""

    bert_type: str = 'huggingface'
    add_pooler: bool = True
    init_method_std: float = 0.02
    hidden_dropout: float = 0.1
    normalization: float = 'LayerNorm'
    layernorm_epsilon: float = 1e-5
    apply_query_key_layer_scaling: bool = False
    position_embedding_type: str = "learned_absolute"


@dataclass
class HuggingFaceBertBaseConfig(HuggingFaceBertConfig):
    """Configs for model in https://huggingface.co/google-bert/bert-base-uncased"""

    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 12


@dataclass
class HuggingFaceBertLargeConfig(HuggingFaceBertConfig):
    """Configs for model in https://huggingface.co/google-bert/bert-large-uncased"""

    num_layers: int = 24
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 16


class HuggingFaceBertModel(BertModel):
    """Google Bert Model."""

    def __init__(
        self,
        config: Annotated[Optional[BertConfig], Config[BertConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or BertConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)


@io.model_importer(HuggingFaceBertModel, "hf")
class HuggingFaceBertImporter(io.ModelConnector["BertForMaskedLM", BertModel]):
    """Importer Connector for converting HF Google Bert Model to NeMo"""

    def __init__(self, *args, **kwargs):
        if sys.version_info > (3, 11):
            # In Python versions <= 3.11, *Path classes don’t have a __init__ method,
            # and do all their initialization in __new__/ helper methods.
            # Only need to call super().__init__ if version > 3.11
            super().__init__(*args)
        self.type = kwargs.get('type', 'model')

    def init(self) -> HuggingFaceBertModel:
        return HuggingFaceBertModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import BertForMaskedLM, BertForNextSentencePrediction, BertForPreTraining, BertModel

        source = BertForPreTraining.from_pretrained(str(self), torch_dtype='auto')
        # Depending on add_lm_head, bert_binary_head, we initialize the bert model differently:
        if self.type == 'model':
            source = BertModel.from_pretrained(str(self), torch_dtype='auto')
        elif self.type == 'pretraining':
            source = BertForPreTraining.from_pretrained(str(self), torch_dtype='auto')
        elif self.type == 'masked':
            source = BertForMaskedLM.from_pretrained(str(self), torch_dtype='auto')
        elif self.type == 'classification':
            source = BertForNextSentencePrediction.from_pretrained(str(self), torch_dtype='auto')

        logging.info(
            f"Initializing Bert Model with pooler={self.config.add_pooler} "
            f"lm_head={self.config.add_lm_head}  binary_head={self.config.bert_binary_head}"
        )
        target = self.init()
        trainer = self.nemo_setup(target)

        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        logging.info(f"Converted Bert model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """Converting HF state dict to NeMo state dict."""
        mapping = {
            "embeddings.position_embeddings.weight": "embedding.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight": "embedding.tokentype_embeddings.weight",
            "embeddings.LayerNorm.weight": "encoder.initial_layernorm.weight",
            "embeddings.LayerNorm.bias": "encoder.initial_layernorm.bias",
            "encoder.layer.*.attention.output.dense.weight": "encoder.layers.*.self_attention.linear_proj.weight",
            "encoder.layer.*.attention.output.dense.bias": "encoder.layers.*.self_attention.linear_proj.bias",
            "encoder.layer.*.attention.output.LayerNorm.weight": "encoder.layers.*.post_att_layernorm.weight",
            "encoder.layer.*.attention.output.LayerNorm.bias": "encoder.layers.*.post_att_layernorm.bias",
            "encoder.layer.*.intermediate.dense.weight": "encoder.layers.*.mlp.linear_fc1.weight",
            "encoder.layer.*.intermediate.dense.bias": "encoder.layers.*.mlp.linear_fc1.bias",
            "encoder.layer.*.output.dense.weight": "encoder.layers.*.mlp.linear_fc2.weight",
            "encoder.layer.*.output.dense.bias": "encoder.layers.*.mlp.linear_fc2.bias",
            "encoder.layer.*.output.LayerNorm.weight": "encoder.layers.*.post_mlp_layernorm.weight",
            "encoder.layer.*.output.LayerNorm.bias": "encoder.layers.*.post_mlp_layernorm.bias",
        }
        if self.config.add_pooler:
            mapping.update(
                {
                    "pooler.dense.weight": "pooler.dense.weight",
                    "pooler.dense.bias": "pooler.dense.bias",
                }
            )

        # When instantiated HF's BertModel, or BertModelForPretraining, BertForMaskedLM, BertForNextSentencePrediction
        # The prefix for state dict is slightly different, therefore we need different transforms to take care of.
        if self.type == 'model':
            transforms = [_import_qkv_2, _import_qkv_bias_2, _import_embedding_2]
        else:
            transforms = [_import_qkv, _import_qkv_bias, _import_embedding]

        if self.type == 'pretraining' or self.type == 'masked':
            # For models with output layers, we need to convert the bias weights
            transforms.append(_import_output_bias)

        if self.type != 'model':
            # adding the 'bert.' prefix so that the state dict matches.
            mapping = {f'bert.{k}': v for k, v in mapping.items()}

        if self.config.add_lm_head:
            mapping.update(
                {
                    "cls.predictions.transform.dense.weight": "lm_head.dense.weight",
                    "cls.predictions.transform.dense.bias": "lm_head.dense.bias",
                    "cls.predictions.transform.LayerNorm.weight": "lm_head.layer_norm.weight",
                    "cls.predictions.transform.LayerNorm.bias": "lm_head.layer_norm.bias",
                }
            )
        if self.config.bert_binary_head:
            mapping.update(
                {
                    "cls.seq_relationship.weight": "binary_head.weight",
                    "cls.seq_relationship.bias": "binary_head.bias",
                }
            )
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        """Retrieve Tokenizer from HF"""
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> BertConfig:
        """Generate NeMo Config based on HF config"""
        from transformers import BertConfig as HFBertConfig

        source = HFBertConfig.from_pretrained(str(self))

        output = HuggingFaceBertConfig(
            bert_type='huggingface',
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.layer_norm_eps,
            seq_length=source.max_position_embeddings,
            add_lm_head=self.type == 'pretraining' or self.type == 'masked',
            bert_binary_head=self.type == 'pretraining' or self.type == 'classification',
            add_pooler=self.type != 'masked',
            share_embeddings_and_output_weights=True,
            num_tokentypes=2,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )
        return output


@io.model_exporter(HuggingFaceBertModel, "hf")
class HuggingFaceBertExporter(io.ModelConnector[BertModel, "BertModel"]):
    """Exporter Connector for converting NeMo Bert Model to HF"""

    def init(self, dtype=torch.bfloat16) -> "BertModel":
        from transformers import BertModel
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return BertModel._from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        source, _ = self.nemo_load(str(self))
        target = self.init(source.dtype)
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    @property
    def tokenizer(self):
        """Retrieve Tokenizer from HF"""
        return io.load_context(str(self)).model.tokenizer.tokenizer

    def convert_state(self, source, target):
        """Convert NeMo state dict to HF style"""
        mapping = {
            "embedding.position_embeddings.weight": "embeddings.position_embeddings.weight",
            "embedding.tokentype_embeddings.weight": "embeddings.token_type_embeddings.weight",
            "encoder.initial_layernorm.weight": "embeddings.LayerNorm.weight",
            "encoder.initial_layernorm.bias": "embeddings.LayerNorm.bias",
            "encoder.layers.*.self_attention.linear_proj.weight": "encoder.layer.*.attention.output.dense.weight",
            "encoder.layers.*.self_attention.linear_proj.bias": "encoder.layer.*.attention.output.dense.bias",
            "encoder.layers.*.post_att_layernorm.weight": "encoder.layer.*.attention.output.LayerNorm.weight",
            "encoder.layers.*.post_att_layernorm.bias": "encoder.layer.*.attention.output.LayerNorm.bias",
            "encoder.layers.*.mlp.linear_fc1.weight": "encoder.layer.*.intermediate.dense.weight",
            "encoder.layers.*.mlp.linear_fc1.bias": "encoder.layer.*.intermediate.dense.bias",
            "encoder.layers.*.mlp.linear_fc2.weight": "encoder.layer.*.output.dense.weight",
            "encoder.layers.*.mlp.linear_fc2.bias": "encoder.layer.*.output.dense.bias",
            "encoder.layers.*.post_mlp_layernorm.weight": "encoder.layer.*.output.LayerNorm.weight",
            "encoder.layers.*.post_mlp_layernorm.bias": "encoder.layer.*.output.LayerNorm.bias",
        }

        if source.config.add_pooler:
            mapping.update(
                {
                    "pooler.dense.weight": "pooler.dense.weight",
                    "pooler.dense.bias": "pooler.dense.bias",
                }
            )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[_export_qkv, _export_qkv_bias, _export_embedding],
        )

    @property
    def config(self) -> "HFBertConfig":
        """Generate HF Config based on NeMo config"""
        source: BertConfig = io.load_context(str(self), subpath="model.config")

        from transformers import BertConfig as HFBertConfig

        return HFBertConfig(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            layer_norm_eps=source.layernorm_epsilon,
            vocab_size=self.tokenizer.vocab_size,
        )


@io.state_transform(
    source_key=(
        "bert.encoder.layer.*.attention.self.query.weight",
        "bert.encoder.layer.*.attention.self.key.weight",
        "bert.encoder.layer.*.attention.self.value.weight",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv(ctx: io.TransformCTX, q, k, v):
    # Bert Model does not support GQA
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    hidden_size = megatron_config.hidden_size
    head_size = getattr(
        megatron_config, 'kv_channels', megatron_config.hidden_size // megatron_config.num_attention_heads
    )

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_q_tensor_shape)
    v = v.view(*new_q_tensor_shape)

    qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:]).to(dtype=q.dtype)
    for i in range(head_num):
        qkv_weights = torch.cat((qkv_weights, q[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
    qkv_weights = qkv_weights.reshape([head_size * (3 * head_num), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key=(
        "bert.encoder.layer.*.attention.self.query.bias",
        "bert.encoder.layer.*.attention.self.key.bias",
        "bert.encoder.layer.*.attention.self.value.bias",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_qkv_bias(ctx: io.TransformCTX, qb, kb, vb):
    # Bert Model does not support GQA
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    head_size = getattr(
        megatron_config, 'kv_channels', megatron_config.hidden_size // megatron_config.num_attention_heads
    )

    new_q_tensor_shape_bias = (head_num, head_size)

    bias_q = qb.view(*new_q_tensor_shape_bias)
    bias_k = kb.view(*new_q_tensor_shape_bias)
    bias_v = vb.view(*new_q_tensor_shape_bias)

    qkv_biases = torch.empty((0, head_size)).to(dtype=qb.dtype)
    for i in range(head_num):
        qkv_biases = torch.cat((qkv_biases, bias_q[i : i + 1]))
        qkv_biases = torch.cat((qkv_biases, bias_k[i : i + 1]))
        qkv_biases = torch.cat((qkv_biases, bias_v[i : i + 1]))
    qkv_biases = qkv_biases.reshape([head_size * (3 * head_num)])

    return qkv_biases


@io.state_transform(
    source_key=("bert.embeddings.word_embeddings.weight",),
    target_key="embedding.word_embeddings.weight",
)
def _import_embedding(ctx: io.TransformCTX, embedding):
    divisible = ctx.target.config.make_vocab_size_divisible_by
    emb_size = embedding.size(0)
    padded_emb_size = int(math.ceil(emb_size / divisible) * divisible)
    if padded_emb_size > emb_size:
        zeros_to_add = torch.zeros(
            padded_emb_size - emb_size,
            embedding.size(1),
            dtype=embedding.dtype,
            device=embedding.device,
        ).to(dtype=embedding.dtype)
        # Concatenate the two tensors along rows
        padded_embedding = torch.cat((embedding, zeros_to_add), dim=0)
        return padded_embedding
    return embedding


@io.state_transform(
    source_key=("cls.predictions.decoder.bias",),
    target_key="output_layer.bias",
)
def _import_output_bias(ctx: io.TransformCTX, bias):
    divisible = ctx.target.config.make_vocab_size_divisible_by
    bias_size = bias.size(0)
    padded_bias_size = int(math.ceil(bias_size / divisible) * divisible)
    if padded_bias_size > bias_size:
        zeros_to_add = torch.zeros(
            padded_bias_size - bias_size,
            dtype=bias.dtype,
            device=bias.device,
        )
        # Concatenate the two tensors along rows
        padded_embedding = torch.cat((bias, zeros_to_add), dim=0)
        return padded_embedding
    return bias


@io.state_transform(
    source_key=(
        "encoder.layer.*.attention.self.query.weight",
        "encoder.layer.*.attention.self.key.weight",
        "encoder.layer.*.attention.self.value.weight",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv_2(ctx: io.TransformCTX, q, k, v):
    # Bert Model does not support GQA
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    hidden_size = megatron_config.hidden_size
    head_size = getattr(
        megatron_config, 'kv_channels', megatron_config.hidden_size // megatron_config.num_attention_heads
    )

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_q_tensor_shape)
    v = v.view(*new_q_tensor_shape)

    qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:]).to(dtype=q.dtype)
    for i in range(head_num):
        qkv_weights = torch.cat((qkv_weights, q[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
    qkv_weights = qkv_weights.reshape([head_size * (3 * head_num), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key=(
        "encoder.layer.*.attention.self.query.bias",
        "encoder.layer.*.attention.self.key.bias",
        "encoder.layer.*.attention.self.value.bias",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_qkv_bias_2(ctx: io.TransformCTX, qb, kb, vb):
    # Bert Model does not support GQA
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    head_size = getattr(
        megatron_config, 'kv_channels', megatron_config.hidden_size // megatron_config.num_attention_heads
    )

    new_q_tensor_shape_bias = (head_num, head_size)

    bias_q = qb.view(*new_q_tensor_shape_bias)
    bias_k = kb.view(*new_q_tensor_shape_bias)
    bias_v = vb.view(*new_q_tensor_shape_bias)

    qkv_biases = torch.empty((0, head_size)).to(dtype=qb.dtype)
    for i in range(head_num):
        qkv_biases = torch.cat((qkv_biases, bias_q[i : i + 1]))
        qkv_biases = torch.cat((qkv_biases, bias_k[i : i + 1]))
        qkv_biases = torch.cat((qkv_biases, bias_v[i : i + 1]))
    qkv_biases = qkv_biases.reshape([head_size * (3 * head_num)])

    return qkv_biases


@io.state_transform(
    source_key=("embeddings.word_embeddings.weight",),
    target_key="embedding.word_embeddings.weight",
)
def _import_embedding_2(ctx: io.TransformCTX, embedding):
    divisible = ctx.target.config.make_vocab_size_divisible_by
    emb_size = embedding.size(0)
    padded_emb_size = int(math.ceil(emb_size / divisible) * divisible)
    if padded_emb_size > emb_size:
        zeros_to_add = torch.zeros(
            padded_emb_size - emb_size,
            embedding.size(1),
            dtype=embedding.dtype,
            device=embedding.device,
        ).to(dtype=embedding.dtype)
        # Concatenate the two tensors along rows
        padded_embedding = torch.cat((embedding, zeros_to_add), dim=0)
        return padded_embedding
    return embedding


@io.state_transform(
    source_key="encoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "encoder.layer.*.attention.self.query.weight",
        "encoder.layer.*.attention.self.key.weight",
        "encoder.layer.*.attention.self.value.weight",
    ),
)
def _export_qkv(ctx: io.TransformCTX, linear_qkv):
    # Bert Model does not support GQA
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = head_num  # BERT Does not use GQA
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size

    head_size = getattr(
        megatron_config, 'kv_channels', megatron_config.hidden_size // megatron_config.num_attention_heads
    )
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
    source_key="encoder.layers.*.self_attention.linear_qkv.bias",
    target_key=(
        "encoder.layer.*.attention.self.query.bias",
        "encoder.layer.*.attention.self.key.bias",
        "encoder.layer.*.attention.self.value.bias",
    ),
)
def _export_qkv_bias(ctx: io.TransformCTX, qkv_bias):
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = head_num  # BERT does not use GQA
    heads_per_group = head_num // num_query_groups
    head_size = getattr(
        megatron_config, 'kv_channels', megatron_config.hidden_size // megatron_config.num_attention_heads
    )
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
    source_key="embedding.word_embeddings.weight",
    target_key="embeddings.word_embeddings.weight",
)
def _export_embedding(ctx: io.TransformCTX, embedding):
    megatron_config = ctx.target.config
    # prune padding.
    return embedding[: megatron_config.vocab_size, :]
