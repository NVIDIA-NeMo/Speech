import torch
import torch.nn.functional as F

from nemo.collections.llm.gpt.model.mixtral import (
    MixtralConfig,
    MixtralConfig8x3B,
    MixtralConfig8x7B,
    MixtralConfig8x22B,
)


def test_mixtral_config():
    config = MixtralConfig()
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.gated_linear_unit is True
    assert config.num_layers == 32
    assert config.hidden_size == 4096
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 14336
    assert config.max_position_embeddings == 4096
    assert config.seq_length == 4096
    assert config.attention_dropout == 0.0
    assert config.hidden_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False
    assert config.num_moe_experts == 8
    assert config.moe_aux_loss_coeff == 0.01
    assert config.moe_router_topk == 2
    assert config.moe_router_pre_softmax is True
    assert config.moe_token_dispatcher_type == "alltoall"
    assert config.moe_router_load_balancing_type == "aux_loss"
    assert config.init_method_std == 0.02
    assert config.layernorm_epsilon == 1e-5
    assert config.rotary_percent == 1.0
    assert config.rotary_base == 1000000.0
    assert config.bf16 is True
    assert config.params_dtype == torch.bfloat16


def test_mixtral_config_8x3b():
    config = MixtralConfig8x3B()
    assert config.num_layers == 32
    assert config.hidden_size == 2560
    assert config.num_attention_heads == 32
    assert config.ffn_hidden_size == 8960
    assert config.max_position_embeddings == 4096
    assert config.seq_length == 4096


def test_mixtral_config_8x7b():
    config = MixtralConfig8x7B()
    assert config.num_layers == 32
    assert config.hidden_size == 4096
    assert config.ffn_hidden_size == 14336
    assert config.max_position_embeddings == 4096
    assert config.seq_length == 4096


def test_mixtral_config_8x22b():
    config = MixtralConfig8x22B()
    assert config.num_layers == 56
    assert config.hidden_size == 6144
    assert config.num_attention_heads == 48
    assert config.ffn_hidden_size == 16384
    assert config.max_position_embeddings == 4096
    assert config.seq_length == 4096
