from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn.functional as F

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.lightning import io, teardown
from nemo.lightning.pytorch.optim import OptimizerModule

if TYPE_CHECKING:
    from transformers import MistralConfig, MistralForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer


@dataclass
class MixtralConfig8x7B(GPTConfig):
    """
    Config for Mixtral-8x7B model
    Official announcement: https://mistral.ai/news/mixtral-of-experts/
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    apply_query_key_layer_scaling: bool = False  # TODO: Should this be True?

    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 14336
    max_position_embeddings: int = 4096  # 32768
    seq_length: int = 4096  # 32768
    # MoE
    num_moe_experts: int = 8
    moe_router_topk: int = 1

    init_method_std: float = 0.02
    layernorm_epsilon: float = 1e-5
    # rotary
    rotary_percent: float = 0.5
    rotary_base: float = 10000


class MixtralModel(GPTModel):
    def __init__(
        self,
        config: Optional[MixtralConfig8x7B] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
    ):
        super().__init__(config or MixtralConfig8x7B(), optim=optim, tokenizer=tokenizer)


@io.model_importer(MixtralModel, ext="hf")
class HFMixtralImporter(io.ModelConnector["MixtralForCausalLM", MixtralModel]):
    def init(self) -> MixtralModel:
        return MixtralModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import MixtralForCausalLM

        source = MixtralForCausalLM.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.pre_mlp_layernorm.weight",
            # MoE
            "model.layers.*.block_sparse_moe.experts.*.w2.weight": "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
            "model.layers.*.block_sparse_moe.gate.weight": "decoder.layers.*.mlp.router.weight",
            # lm-head
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv, _import_moe_w1_w3])

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(str(self))

    @property
    def config(self) -> MixtralConfig8x7B:
        from transformers import MixtralConfig as HfMixtralConfig

        config = HfMixtralConfig.from_pretrained(str(self))
        return MixtralConfig8x7B(
            activation_func=F.silu,
            # network
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,  # TODO
            seq_length=config.max_position_embeddings,
            # RoPE
            position_embedding_type='rope',
            rotary_base=config.rope_theta,
            # Transformer config
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            num_moe_experts=config.num_local_experts,
            moe_router_topk=config.num_experts_per_tok,
            # norm
            normalization='RMSNorm',
            layernorm_epsilon=config.rms_norm_eps,
            # Init
            init_method_std=config.initializer_range,
            gated_linear_unit=True,
            # Vocab
            make_vocab_size_divisible_by=128,
        )


@io.state_transform(
    source_key=(
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_num = megatron_config.num_attention_heads
    head_size = hidden_size // head_num

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
    source_key=(
        "model.layers.*.block_sparse_moe.experts.*.w1.weight",
        "model.layers.*.block_sparse_moe.experts.*.w3.weight",
    ),
    target_key="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
)
def _import_moe_w1_w3(gate_proj, up_proj):
    return torch.cat((gate_proj, up_proj), axis=0)
