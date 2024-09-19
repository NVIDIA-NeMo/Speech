import torch.nn.functional as F

from nemo.collections.llm.gpt.model.starcoder2 import (
    Starcoder2Config,
    Starcoder2Config3B,
    Starcoder2Config7B,
    Starcoder2Config15B,
)


def test_starcoder2_config():
    config = Starcoder2Config(num_layers=30, hidden_size=3072, num_attention_heads=24)
    assert config.normalization == "LayerNorm"
    assert config.activation_func == F.gelu
    assert config.add_bias_linear is True
    assert config.seq_length == 16384
    assert config.position_embedding_type == "rope"
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.init_method_std == 0.01
    assert config.share_embeddings_and_output_weights is False
    assert config.kv_channels == 3072 // 24
    assert config.num_query_groups == 24
    assert config.attention_softmax_in_fp32 is True
    assert config.bias_activation_fusion is True
    assert config.bias_dropout_fusion is True
    assert config.layernorm_epsilon == 1e-5
    assert config.rotary_percent == 1.0
    assert config.window_size is None


def test_starcoder2_config_3b():
    config = Starcoder2Config3B()
    assert config.num_layers == 30
    assert config.hidden_size == 3072
    assert config.ffn_hidden_size == 12288
    assert config.num_attention_heads == 24
    assert config.num_query_groups == 2
    assert config.init_method_std == 0.018042
    assert config.rotary_base == 999999.4420358813


def test_starcoder2_config_7b():
    config = Starcoder2Config7B()
    assert config.num_layers == 32
    assert config.hidden_size == 4608
    assert config.ffn_hidden_size == 18432
    assert config.num_attention_heads == 36
    assert config.num_query_groups == 4
    assert config.init_method_std == 0.018042
    assert config.rotary_base == 1_000_000


def test_starcoder2_config_15b():
    config = Starcoder2Config15B()
    assert config.num_layers == 40
    assert config.hidden_size == 6144
    assert config.ffn_hidden_size == 24576
    assert config.num_attention_heads == 48
    assert config.num_query_groups == 4
    assert config.init_method_std == 0.01275
    assert config.rotary_base == 100_000
