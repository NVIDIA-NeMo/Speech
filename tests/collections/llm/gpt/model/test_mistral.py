import torch.nn.functional as F

from nemo.collections.llm.gpt.model.mistral import MistralConfig7B, MistralNeMo2407Config12B, MistralNeMo2407Config123B


def test_mistral_config7b():
    config = MistralConfig7B()
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
    assert config.seq_length == 32768
    assert config.attention_dropout == 0.0
    assert config.hidden_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False
    assert config.init_method_std == 0.02
    assert config.layernorm_epsilon == 1e-5
    assert config.window_size == [4096, 0]


def test_mistral_nemo_config_12b():
    config = MistralNeMo2407Config12B()
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.gated_linear_unit is True
    assert config.num_layers == 40
    assert config.hidden_size == 5120
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 14336
    assert config.seq_length == 4096
    assert config.attention_dropout == 0.0
    assert config.hidden_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False
    assert config.init_method_std == 0.02
    assert config.layernorm_epsilon == 1e-5
    assert config.window_size is None
    assert config.rotary_percent == 1.0
    assert config.rotary_base == 1000000.0
    assert config.kv_channels == 128


def test_mistral_nemo_config_123b():
    config = MistralNeMo2407Config123B()
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.gated_linear_unit is True
    assert config.num_layers == 88
    assert config.hidden_size == 12288
    assert config.num_attention_heads == 96
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 28672
    assert config.seq_length == 4096
    assert config.attention_dropout == 0.0
    assert config.hidden_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False
    assert config.init_method_std == 0.02
    assert config.layernorm_epsilon == 1e-5
    assert config.window_size is None
    assert config.rotary_percent == 1.0
    assert config.rotary_base == 1000000.0
    assert config.kv_channels == 128
