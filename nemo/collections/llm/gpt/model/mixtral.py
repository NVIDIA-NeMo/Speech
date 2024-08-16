from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.lightning import io, teardown
from nemo.lightning.pytorch.optim import OptimizerModule

if TYPE_CHECKING:
    from transformers import MixtralForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class MixtralConfig(GPTConfig):
    """
    Base config for Mixtral models.
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    apply_query_key_layer_scaling: bool = False

    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 14336
    max_position_embeddings: int = 4096
    seq_length: int = 4096
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False

    # MoE
    num_moe_experts: int = 8
    moe_router_topk: int = 1
    moe_router_pre_softmax: bool = True

    init_method_std: float = 0.02
    layernorm_epsilon: float = 1e-5
    # rotary
    rotary_percent: float = 1.0
    rotary_base: float = 1000000.0
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16


@dataclass
class MixtralConfig8x3B(MixtralConfig):
    """
    NeMo's Mixtral-8x3B model variant
    https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/main/launcher_scripts/conf/training/mixtral/mixtral_8x3b.yaml
    """

    num_layers: int = 32
    hidden_size: int = 2560
    num_attention_heads: int = 32
    ffn_hidden_size: int = 8960
    max_position_embeddings: int = 4096
    seq_length: int = 4096


@dataclass
class MixtralConfig8x7B(MixtralConfig):
    """
    Config for Mixtral-8x7B model
    Official announcement: https://mistral.ai/news/mixtral-of-experts/
    """

    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    max_position_embeddings: int = 4096
    seq_length: int = 4096


@dataclass
class MixtralConfig8x22B(MixtralConfig):
    """
    Config for Mixtral-8x7B model
    Official announcement: https://mistral.ai/news/mixtral-8x22b/
    """

    num_layers: int = 56
    hidden_size: int = 6144
    num_attention_heads: int = 48
    ffn_hidden_size: int = 16384
    max_position_embeddings: int = 4096
    seq_length: int = 4096
    # MoE
    num_moe_experts: int = 8
    moe_router_topk: int = 2


class MixtralModel(GPTModel):
    def __init__(
        self,
        config: Optional[Union[MixtralConfig8x7B, MixtralConfig8x22B]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config or MixtralConfig8x7B(), optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )


@io.model_importer(MixtralModel, ext="hf")
class HFMixtralImporter(io.ModelConnector["MixtralForCausalLM", MixtralModel]):
    def init(self) -> MixtralModel:
        return MixtralModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import MixtralForCausalLM

        source = MixtralForCausalLM.from_pretrained(str(self), torch_dtype='auto', use_safetensors=True)
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
    def config(self) -> MixtralConfig8x7B | MixtralConfig8x22B:
        from transformers import MixtralConfig as HfMixtralConfig

        config = HfMixtralConfig.from_pretrained(str(self))
        config_cls = MixtralConfig8x7B
        if '8x22b' in str(self).lower():
            config_cls = MixtralConfig8x22B
        return config_cls(
            bf16=getattr(config, "torch_dtype", None) == torch.bfloat16,
            activation_func=F.silu,
            # network
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            kv_channels=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
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
            moe_router_pre_softmax=True,
            # norm
            normalization='RMSNorm',
            layernorm_epsilon=config.rms_norm_eps,
            # Init
            init_method_std=config.initializer_range,
            gated_linear_unit=True,
            # Vocab
            make_vocab_size_divisible_by=128,
            # CPU init
            use_cpu_initialization=True,
            perform_initialization=False,
            params_dtype=getattr(config, "torch_dtype", torch.bfloat16),
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
    head_size = megatron_config.kv_channels

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


@io.model_exporter(MixtralModel, "hf")
class HFMixtralExporter(io.ModelConnector[MixtralModel, "MixtralForCausalLM"]):
    def init(self) -> "MixtralForCausalLM":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        # TODO: Make it work with lazy init
        # with torch.device("meta"):
        #     target = self.init()
        target = self.init()
        source, _ = self.nemo_load(str(self))
        target = self.convert_state(source, target)

        # TODO: Make sure we don't need to do this
        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # MoE
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": "model.layers.*.block_sparse_moe.experts.*.w2.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.block_sparse_moe.gate.weight",
            # lm-head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_export_qkv, _export_moe_w1_w3])

    @property
    def tokenizer(self):
        return io.load_ckpt(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "MixtralConfig":
        # Either MixtralConfig8x7B or MixtralConfig8x22B
        source: MixtralConfig8x7B = io.load_ckpt(str(self)).model.config

        from transformers import MixtralConfig as HfMixtralConfig

        return HfMixtralConfig(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            max_position_embeddings=source.max_position_embeddings,
            seq_length=source.max_position_embeddings,
            # RoPe
            rope_theta=source.rotary_base,
            # transformer config
            num_attention_heads=source.num_attention_heads,
            num_key_value_heads=source.num_query_groups,
            num_local_experts=config.num_moe_experts,
            num_experts_per_tok=config.moe_router_topk,
            # norm
            rms_norm_eps=source.layernorm_epsilon,
            # init
            initializer_range=source.init_method_std,
            # vocab
            vocab_size=self.tokenizer.vocab_size,
            head_dim=source.kv_channels,
        )


@io.state_transform(
    source_key="decoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ),
)
def _export_qkv(ctx: io.TransformCTX, linear_qkv):
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_num = megatron_config.num_attention_heads
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
    source_key="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
    target_key=(
        "model.layers.*.block_sparse_moe.experts.*.w1.weight",
        "model.layers.*.block_sparse_moe.experts.*.w3.weight",
    ),
)
def _export_moe_w1_w3(linear_fc1):
    gate_proj, up_proj = torch.chunk(linear_fc1, 2, dim=0)

    return gate_proj, up_proj
